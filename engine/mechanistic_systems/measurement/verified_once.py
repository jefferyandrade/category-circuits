"""Verify-once integrity receipts and single-pass verified reads.

Two mechanisms, both preserving every existing fail-closed check:

1. **Verify once, bind by receipt, reuse.**  A shared immutable file (the
   canonical dense table's ``values`` payload, the shared dense fit bundle)
   would otherwise be re-hashed by every category process.  After one full
   SHA-256 pass on a node, :func:`sha256_file_once` records a node-local
   receipt that binds the digest to the file's filesystem identity: real
   path, size, ``mtime_ns``, ``ctime_ns``, inode, device, host name and
   kernel boot id.  A later process on the same node boot that finds the
   identity unchanged reuses the digest; any identity change, a malformed or
   foreign receipt, or a disabled switch is a full hash, exactly as without
   the receipt.  This amortizes the integrity check (it is not dropped):
   content changes made through the filesystem change ``mtime``/``ctime``;
   what the amortized form does not re-detect is silent media corruption
   that leaves metadata untouched.

   The receipts live in a user-private directory (``IRG_VERIFY_ONCE_DIR``,
   default ``/tmp/<user>/irg_verify_once``).  The mechanism is on when
   ``IRG_VERIFY_ONCE`` is truthy, or when it is unset inside a Slurm job;
   ``IRG_VERIFY_ONCE=0`` always hashes in full.

2. **Hash while reading.**  :func:`read_verified_npz` reads an uncompressed
   ``.npz`` exactly once with positional reads on worker threads, directly
   into the member arrays, while the calling thread feeds the file bytes to
   SHA-256 strictly in file order.  The digest is therefore the SHA-256 of
   the whole file, byte for byte what :func:`sha256_file` returns, and the
   arrays equal what ``np.load`` returns; the previous path read the file
   twice (hash, then load).  ZIP CRC-32 checks are subsumed by the caller's
   whole-file SHA-256 comparison, which happens before any array is used.
   The archive layout (central directory, local headers, npy headers) is
   first parsed from a pre-read, then parsed AGAIN from the hashed bytes, and
   the two parses must be identical, so the offsets, dtypes and shapes the
   arrays were read with are those of the bytes the digest covers.
"""

from __future__ import annotations

import getpass
import hashlib
import io
import json
import os
import socket
import struct
import threading
import time
import zipfile
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from .run_products import effective_settings

VERIFY_ONCE_ENV = "IRG_VERIFY_ONCE"
VERIFY_ONCE_DIR_ENV = "IRG_VERIFY_ONCE_DIR"
VERIFY_ONCE_RECEIPT_SCHEMA = "irg_verify_once_file_digest_v1"
VERIFIED_BY_FULL_HASH = "sha256_full_pass"
VERIFIED_BY_RECEIPT = "node_verify_once_receipt"
READ_WORKERS_ENV = "IRG_VERIFIED_READ_WORKERS"
_DEFAULT_READ_WORKERS = 4
_READ_CHUNK_BYTES = 64 * 2**20
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


# ---------------------------------------------------------------------------
# Verify-once receipts
# ---------------------------------------------------------------------------


def verify_once_enabled(env: Mapping[str, str] | None = None) -> bool:
    source = os.environ if env is None else env
    raw = str(source.get(VERIFY_ONCE_ENV, "")).strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    return bool(str(source.get("SLURM_JOB_ID", "")).strip())


def _boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return ""


def verify_once_root(env: Mapping[str, str] | None = None) -> Path | None:
    """The user-private receipt directory, or ``None`` when unusable."""

    source = os.environ if env is None else env
    if not verify_once_enabled(source):
        return None
    override = str(source.get(VERIFY_ONCE_DIR_ENV, "")).strip()
    if override:
        root = Path(override)
    else:
        user = str(source.get("USER", "")).strip()
        if not user:
            try:
                user = getpass.getuser()
            except Exception:
                return None
        root = Path("/tmp") / user / "irg_verify_once"
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        stat = root.stat()
    except OSError:
        return None
    if stat.st_uid != os.getuid() or stat.st_mode & 0o022:
        return None
    return root


def file_identity(path: str | Path) -> dict[str, Any]:
    resolved = os.path.realpath(str(path))
    stat = os.stat(resolved)
    return {
        "path": resolved,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
        "inode": int(stat.st_ino),
        "device": int(stat.st_dev),
    }


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _receipt_path(root: Path, identity: Mapping[str, Any]) -> Path:
    key = hashlib.sha256(str(identity["path"]).encode("utf-8")).hexdigest()
    return root / f"{key}.json"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def lookup_verified_sha256(
    path: str | Path,
    *,
    root: Path,
    identity: Mapping[str, Any] | None = None,
) -> tuple[str | None, str]:
    """Return ``(sha256, "hit")`` or ``(None, reason)``; never raises."""

    try:
        current = dict(identity) if identity is not None else file_identity(path)
        receipt_path = _receipt_path(root, current)
        stat = receipt_path.stat()
        if stat.st_uid != os.getuid() or stat.st_mode & 0o022:
            return None, "receipt_not_private"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "no_receipt"
    except (OSError, ValueError, TypeError):
        return None, "receipt_unreadable"
    if not isinstance(receipt, dict):
        return None, "receipt_not_an_object"
    claimed = receipt.pop("content_sha256", None)
    if claimed != _canonical_sha256(receipt):
        return None, "receipt_content_hash_mismatch"
    if receipt.get("schema") != VERIFY_ONCE_RECEIPT_SCHEMA:
        return None, "receipt_schema_mismatch"
    if receipt.get("hostname") != socket.gethostname() or receipt.get(
        "boot_id"
    ) != _boot_id():
        return None, "receipt_from_another_node_or_boot"
    for name, value in current.items():
        if receipt.get(name) != value:
            return None, f"identity_changed:{name}"
    digest = receipt.get("sha256")
    if not _is_sha256(digest):
        return None, "receipt_digest_malformed"
    return str(digest), "hit"


def record_verified_sha256(
    path: str | Path,
    sha256: str,
    *,
    root: Path,
    identity_before: Mapping[str, Any],
) -> bool:
    """Record a digest computed by a full pass; skipped if the file moved."""

    try:
        if not _is_sha256(sha256):
            return False
        identity_after = file_identity(path)
        if dict(identity_after) != dict(identity_before):
            return False  # changed while it was hashed: never bind it
        receipt = {
            "schema": VERIFY_ONCE_RECEIPT_SCHEMA,
            "effective_settings": effective_settings(),
            **identity_after,
            "sha256": str(sha256),
            "hostname": socket.gethostname(),
            "boot_id": _boot_id(),
            "recorded_at_unix": time.time(),
        }
        receipt["content_sha256"] = _canonical_sha256(receipt)
        destination = _receipt_path(root, identity_after)
        partial = destination.with_name(
            f".{destination.name}.{os.getpid()}.{threading.get_ident()}.partial"
        )
        descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(descriptor, json.dumps(receipt, sort_keys=True).encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(partial, destination)
        return True
    except OSError:
        return False


def sha256_file_parallel(
    path: str | Path,
    *,
    workers: int | None = None,
    chunk_bytes: int = _READ_CHUNK_BYTES,
) -> str:
    """SHA-256 of a file; reads overlap on threads, hashing stays in order."""

    return _hash_file_in_order(Path(path), workers=workers, chunk_bytes=chunk_bytes)


def sha256_file_once(
    path: str | Path,
    *,
    env: Mapping[str, str] | None = None,
    hasher: Callable[[Path], str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """``(sha256, disclosure)`` of ``path``, verified once per node boot.

    ``hasher`` (default :func:`sha256_file_parallel`) runs on a miss.  The
    disclosure names how the digest was obtained so receipts can state it.
    """

    source = Path(path)
    root = verify_once_root(env)
    disclosure: dict[str, Any] = {
        "schema": "irg_verify_once_disclosure_v1",
        "verified_by": VERIFIED_BY_FULL_HASH,
        "receipt_root": None if root is None else str(root),
        "lookup": "disabled" if root is None else None,
        "recorded": False,
    }
    identity = file_identity(source)
    # Linux filesystems can timestamp multiple writes in the same clock tick.
    # Let a just-written inode leave that tick before hashing; otherwise a
    # same-size rewrite during a short hash can retain both mtime and ctime.
    # Sealed inputs are older than this interval and take no delay.
    tick_ns = 1_000_000_000 // int(os.sysconf("SC_CLK_TCK"))
    age_ns = time.time_ns() - identity["ctime_ns"]
    if root is not None and 0 <= age_ns < tick_ns:
        time.sleep((tick_ns - age_ns) / 1_000_000_000)
        identity = file_identity(source)
    if root is not None:
        digest, reason = lookup_verified_sha256(source, root=root, identity=identity)
        disclosure["lookup"] = reason
        if digest is not None:
            disclosure["verified_by"] = VERIFIED_BY_RECEIPT
            return digest, disclosure
    started = time.perf_counter()
    digest = (hasher or sha256_file_parallel)(source)
    disclosure["hash_seconds"] = float(time.perf_counter() - started)
    if root is not None:
        disclosure["recorded"] = record_verified_sha256(
            source, digest, root=root, identity_before=identity
        )
    return digest, disclosure


# ---------------------------------------------------------------------------
# Ordered-hash positional reader
# ---------------------------------------------------------------------------


def _read_workers(workers: int | None) -> int:
    if workers is None:
        raw = str(os.environ.get(READ_WORKERS_ENV, "")).strip()
        try:
            workers = int(raw) if raw else _DEFAULT_READ_WORKERS
        except ValueError:
            workers = _DEFAULT_READ_WORKERS
    return max(1, min(int(workers), 32))


def _pread_exact(descriptor: int, view: memoryview, offset: int) -> int:
    total = 0
    length = len(view)
    while total < length:
        if hasattr(os, "preadv"):
            count = os.preadv(descriptor, [view[total:]], offset + total)
        else:  # pragma: no cover - platforms without preadv
            data = os.pread(descriptor, length - total, offset + total)
            count = len(data)
            view[total : total + count] = data
        if count == 0:
            break
        total += count
    return total


def _read_segments_in_order(
    descriptor: int,
    segments: list[tuple[int, memoryview]],
    *,
    digest: Any,
    workers: int,
) -> None:
    """Fill each ``(offset, view)`` and hash the views in list order.

    Segments must tile the byte range they cover in increasing offset order.
    A short read raises: the caller bound the file size before planning.
    """

    if workers <= 1 or len(segments) <= 1:
        for offset, view in segments:
            got = _pread_exact(descriptor, view, offset)
            if got != len(view):
                raise OSError("short read while hashing a verified payload")
            digest.update(view)
        return
    window = max(2, 2 * workers)
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="verified-read"
    ) as executor:
        pending: list[Future[int]] = []
        submitted = 0
        try:
            for index in range(len(segments)):
                while submitted < len(segments) and submitted < index + window:
                    offset, view = segments[submitted]
                    pending.append(
                        executor.submit(_pread_exact, descriptor, view, offset)
                    )
                    submitted += 1
                got = pending[index].result()
                view = segments[index][1]
                if got != len(view):
                    raise OSError("short read while hashing a verified payload")
                digest.update(view)
                pending[index] = None  # type: ignore[call-overload]
        finally:
            for future in pending:
                if future is not None:
                    future.cancel()


def _split(offset: int, view: memoryview, chunk: int) -> list[tuple[int, memoryview]]:
    return [
        (offset + start, view[start : start + chunk])
        for start in range(0, len(view), chunk)
    ]


def _hash_file_in_order(
    path: Path,
    *,
    workers: int | None,
    chunk_bytes: int,
) -> str:
    digest = hashlib.sha256()
    worker_count = _read_workers(workers)
    chunk = max(4096, int(chunk_bytes))
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        size = int(os.fstat(descriptor).st_size)
        if worker_count <= 1:
            buffer = memoryview(bytearray(min(chunk, max(size, 1))))
            offset = 0
            while True:
                got = _pread_exact(descriptor, buffer, offset)
                if not got:
                    break
                digest.update(buffer[:got])
                offset += got
            return digest.hexdigest()
        pool = [memoryview(bytearray(chunk)) for _ in range(2 * worker_count)]
        offset = 0
        while offset < size:
            batch: list[tuple[int, memoryview]] = []
            for slot in pool:
                if offset >= size:
                    break
                length = min(chunk, size - offset)
                batch.append((offset, slot[:length]))
                offset += length
            _read_segments_in_order(
                descriptor, batch, digest=digest, workers=worker_count
            )
        # Read to EOF exactly as the sequential hasher does (normally empty).
        tail = memoryview(bytearray(1 << 20))
        while True:
            got = _pread_exact(descriptor, tail, offset)
            if not got:
                break
            digest.update(tail[:got])
            offset += got
    finally:
        os.close(descriptor)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Single-pass verified NPZ
# ---------------------------------------------------------------------------


class UnsupportedVerifiedNpz(ValueError):
    """The archive is not a plain uncompressed NPZ; use the fallback path."""


@dataclass
class VerifiedNpzPayload:
    """Arrays of one NPZ plus the SHA-256 of the exact file bytes read."""

    sha256: str
    arrays: dict[str, np.ndarray]
    identity: dict[str, Any]
    method: str
    read_seconds: float = 0.0
    _closed: bool = field(default=False, repr=False)

    @property
    def files(self) -> list[str]:
        return list(self.arrays)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]

    def __contains__(self, name: object) -> bool:
        return name in self.arrays

    def __enter__(self) -> "VerifiedNpzPayload":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._closed = True


_LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
_LOCAL_HEADER_SIGNATURE = 0x04034B50


def _npy_header(prefix: bytes) -> tuple[np.dtype, bool, tuple[int, ...], int]:
    stream = io.BytesIO(prefix)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise UnsupportedVerifiedNpz(f"npy version {version!r}")
    if dtype.hasobject:
        raise UnsupportedVerifiedNpz("object arrays are not admissible")
    return dtype, bool(fortran), tuple(int(value) for value in shape), stream.tell()


_NpzMember = tuple[str, int, int, int, np.dtype, bool, tuple[int, ...], int]


def _npz_layout(
    read_at: Callable[[int, int], bytes], handle: Any, size: int
) -> list[_NpzMember]:
    """The archive's member layout, parsed through ``read_at`` and ``handle``.

    Each member is ``(name, header_offset, data_offset, member_size, dtype,
    fortran, shape, npy_header_bytes)``; dtypes compare by value (byte order
    and structured fields included).
    """

    try:
        archive = zipfile.ZipFile(handle)
    except zipfile.BadZipFile as error:
        raise UnsupportedVerifiedNpz(f"not a zip archive: {error}") from error
    with archive:
        members = sorted(archive.infolist(), key=lambda info: info.header_offset)
    layout: list[_NpzMember] = []
    names: set[str] = set()
    cursor = 0
    for info in members:
        if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 0x1:
            raise UnsupportedVerifiedNpz("compressed or encrypted member")
        if not info.filename.endswith(".npy"):
            raise UnsupportedVerifiedNpz(f"non-npy member {info.filename!r}")
        name = info.filename[: -len(".npy")]
        if name in names:
            raise UnsupportedVerifiedNpz("duplicate member")
        names.add(name)
        header = read_at(info.header_offset, _LOCAL_HEADER.size)
        if len(header) != _LOCAL_HEADER.size:
            raise UnsupportedVerifiedNpz("truncated local header")
        fields = _LOCAL_HEADER.unpack(header)
        if fields[0] != _LOCAL_HEADER_SIGNATURE:
            raise UnsupportedVerifiedNpz("bad local header signature")
        data_offset = info.header_offset + _LOCAL_HEADER.size + fields[9] + fields[10]
        member_size = int(info.file_size)
        prefix = read_at(data_offset, min(member_size, 1 << 16))
        dtype, fortran, shape, npy_header_bytes = _npy_header(prefix)
        count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        nbytes = count * int(dtype.itemsize)
        if npy_header_bytes + nbytes != member_size:
            raise UnsupportedVerifiedNpz("member size differs from its npy header")
        if data_offset < cursor or data_offset + member_size > size:
            raise UnsupportedVerifiedNpz("members overlap or exceed the file")
        cursor = data_offset + member_size
        layout.append(
            (
                name,
                int(info.header_offset),
                int(data_offset),
                member_size,
                dtype,
                bool(fortran),
                tuple(shape),
                int(npy_header_bytes),
            )
        )
    return layout


def _plan_npz(descriptor: int, handle: Any, size: int) -> tuple[
    list[tuple[int, memoryview]], dict[str, np.ndarray], list[_NpzMember]
]:
    layout = _npz_layout(
        lambda offset, length: os.pread(descriptor, length, offset), handle, size
    )
    segments: list[tuple[int, memoryview]] = []
    arrays: dict[str, np.ndarray] = {}
    cursor = 0
    for name, _header_offset, data_offset, member_size, dtype, fortran, shape, npy_bytes in layout:
        nbytes = member_size - npy_bytes
        storage = np.empty(nbytes, dtype=np.uint8)
        if data_offset + npy_bytes > cursor:
            gap = memoryview(bytearray(data_offset + npy_bytes - cursor))
            segments.append((cursor, gap))
        segments.extend(
            _split(data_offset + npy_bytes, memoryview(storage), _READ_CHUNK_BYTES)
        )
        cursor = data_offset + member_size
        typed = storage.view(dtype) if nbytes else np.empty(0, dtype=dtype)
        if fortran:
            array = typed.reshape(shape[::-1]).transpose()
        else:
            array = typed.reshape(shape)
        arrays[name] = array
    if cursor < size:
        segments.extend(_split(cursor, memoryview(bytearray(size - cursor)), _READ_CHUNK_BYTES))
    return segments, arrays, layout


class _HashedBytes(io.RawIOBase):
    """A read-only file view of the hashed segments (which tile the file)."""

    def __init__(self, segments: list[tuple[int, memoryview]], size: int) -> None:
        super().__init__()
        self._segments = sorted(segments, key=lambda item: item[0])
        self._starts = [offset for offset, _view in self._segments]
        self._size = int(size)
        self._position = 0

    def read_at(self, offset: int, length: int) -> bytes:
        import bisect

        out = bytearray()
        position = int(offset)
        end = min(self._size, position + max(0, int(length)))
        while position < end:
            index = bisect.bisect_right(self._starts, position) - 1
            if index < 0:
                raise OSError("hashed view does not cover the requested bytes")
            start, view = self._segments[index]
            inner = position - start
            if inner >= len(view):
                raise OSError("hashed view does not cover the requested bytes")
            take = min(len(view) - inner, end - position)
            out += view[inner : inner + take]
            position += take
        return bytes(out)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._position = int(offset)
        elif whence == io.SEEK_CUR:
            self._position += int(offset)
        elif whence == io.SEEK_END:
            self._position = self._size + int(offset)
        else:  # pragma: no cover - io contract
            raise ValueError("invalid whence")
        return self._position

    def readinto(self, buffer: Any) -> int:
        data = self.read_at(self._position, len(buffer))
        buffer[: len(data)] = data
        self._position += len(data)
        return len(data)


def read_verified_npz(
    path: str | Path,
    *,
    workers: int | None = None,
) -> VerifiedNpzPayload:
    """Read an NPZ once: member arrays plus the whole-file SHA-256.

    Plain uncompressed archives are read with positional reads directly into
    their arrays.  Any other archive is read once into memory, hashed, and
    parsed by ``np.load`` from that same buffer.  In both forms the digest is
    over exactly the bytes the arrays came from.
    """

    source = Path(path)
    started = time.perf_counter()
    identity = file_identity(source)
    digest = hashlib.sha256()
    worker_count = _read_workers(workers)
    descriptor = os.open(str(source), os.O_RDONLY)
    try:
        size = int(os.fstat(descriptor).st_size)
        if size != identity["size_bytes"]:
            raise OSError("verified payload changed while it was opened")
        method = "single_pass_stored_npz"
        layout: list[_NpzMember] | None = None
        try:
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                segments, arrays, layout = _plan_npz(descriptor, handle, size)
        except UnsupportedVerifiedNpz:
            method = "single_pass_buffered_np_load"
            buffer = memoryview(bytearray(size))
            segments = _split(0, buffer, _READ_CHUNK_BYTES)
            arrays = {}
        _read_segments_in_order(
            descriptor, segments, digest=digest, workers=worker_count
        )
        tail = os.pread(descriptor, 1, size)
        if tail:
            raise OSError("verified payload grew while it was read")
    finally:
        os.close(descriptor)
    if file_identity(source) != identity:
        raise OSError("verified payload changed while it was read")
    if layout is not None:
        # The offsets, dtypes and shapes above came from a pre-read.  Parse the
        # layout again from the bytes the digest covers; any difference means
        # the arrays were not read with the hashed headers.
        hashed = _HashedBytes(segments, size)
        try:
            hashed_layout = _npz_layout(hashed.read_at, hashed, size)
        except UnsupportedVerifiedNpz as error:
            raise OSError(
                "verified payload headers changed between parse and hash"
            ) from error
        if hashed_layout != layout:
            raise OSError("verified payload headers changed between parse and hash")
    if method == "single_pass_buffered_np_load":
        with np.load(io.BytesIO(bytes(buffer)), allow_pickle=False) as loaded:
            arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    return VerifiedNpzPayload(
        sha256=digest.hexdigest(),
        arrays=arrays,
        identity=identity,
        method=method,
        read_seconds=float(time.perf_counter() - started),
    )


class VerifiedNpzPrefetch:
    """Start :func:`read_verified_npz` on a helper thread; join on demand.

    The caller compares ``result().sha256`` with its receipt before any array
    is used, and checks that the file identity it later observes (under its
    own lock) equals ``result().identity``; otherwise it reads again.
    """

    def __init__(self, path: str | Path, *, workers: int | None = None) -> None:
        self.path = Path(path)
        self._result: VerifiedNpzPayload | None = None
        self._error: BaseException | None = None
        self._consumed = False
        self._thread = threading.Thread(
            target=self._run,
            args=(workers,),
            name="verified-npz-prefetch",
            daemon=True,
        )
        self._thread.start()

    def _run(self, workers: int | None) -> None:
        try:
            if self.path.is_file():
                self._result = read_verified_npz(self.path, workers=workers)
                if self._consumed:  # abandoned while reading: drop it
                    self._result = None
        except BaseException as error:  # pragma: no cover - surfaced on join
            self._error = error

    def take(self) -> VerifiedNpzPayload | None:
        """Join and hand over the payload once (``None`` if absent/failed)."""

        self._thread.join()
        if self._consumed:
            return None
        self._consumed = True
        result, self._result = self._result, None
        return None if self._error is not None else result

    def discard(self) -> None:
        self._thread.join()
        self._consumed = True
        self._result = None

    def abandon(self) -> None:
        """Release the payload without waiting for the read to finish."""

        self._consumed = True
        self._result = None


__all__ = [
    "READ_WORKERS_ENV",
    "UnsupportedVerifiedNpz",
    "VERIFIED_BY_FULL_HASH",
    "VERIFIED_BY_RECEIPT",
    "VERIFY_ONCE_DIR_ENV",
    "VERIFY_ONCE_ENV",
    "VerifiedNpzPayload",
    "VerifiedNpzPrefetch",
    "file_identity",
    "lookup_verified_sha256",
    "read_verified_npz",
    "record_verified_sha256",
    "sha256_file_once",
    "sha256_file_parallel",
    "verify_once_enabled",
    "verify_once_root",
]
