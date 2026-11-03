"""IRG node-local uint8-bank staging.

This module implements the default-on ``IRG_NODE_LOCAL_BANK_STAGING``
mechanism with an explicit ``0``/``false``/``no``/``off`` kill switch.
It is intentionally distinct from the pre-existing ``CIRCUIT_NVME_STAGE_DIR``
staging path in :mod:`mechanistic_systems.measurement.contexts`.
"""

from __future__ import annotations

import fcntl
import getpass
import hashlib
import json
import os
import socket
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

DEFAULT_LOCK_TIMEOUT_SECONDS = 1800.0
_DEFAULT_MIN_FREE_GIB = 32.0
_COPY_BUFFER_BYTES = 16 * 2**20
_SCHEMA_NAME = "irg_node_local_bank_staging_v2"
_SOURCE_IDENTITY_SCHEMA = "certified_uint8_bank_payload_sha256_v1"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


class BankStagingRequiredError(RuntimeError):
    """Raised only when explicitly required node-local staging cannot complete."""

    def __init__(self, reason: str, receipt: Mapping[str, Any]) -> None:
        self.reason = str(reason)
        self.receipt = dict(receipt)
        super().__init__(f"required node-local bank staging failed: {self.reason}")


def _receipt() -> dict[str, Any]:
    return {
        "schema_name": _SCHEMA_NAME,
        "staging_mode": "disabled",
        "required_mode": False,
        "staged_path": None,
        "replica_key": None,
        "source_identity_hash": None,
        "replica_reused": False,
        "replica_rebuild_reason": None,
        "completion_receipt_path": None,
        "bank_bytes": None,
        "copy_seconds": 0.0,
        "copy_hashes": None,
        "probe_verification": None,
        "reuse_verification": None,
        "lock_wait_seconds": 0.0,
        "capacity_lock_wait_seconds": 0.0,
        "staging_root_free_bytes_before": None,
        "staging_required_bytes": None,
        "staging_root": None,
        "fallback_reason": None,
    }


def _fallback(receipt: dict[str, Any], reason: str) -> None:
    receipt["staging_mode"] = "fallback"
    receipt["staged_path"] = None
    receipt["fallback_reason"] = str(reason)


def _choose_staging_root(env: Mapping[str, str]) -> tuple[Path | None, str | None]:
    override = str(env.get("IRG_BANK_STAGING_DIR", "")).strip()
    if override:
        root = Path(override)
        try:
            root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return None, f"staging_root_creation_failed:{type(exc).__name__}:{exc}"
        return root, None

    user = str(env.get("USER", "")).strip()
    if not user:
        try:
            user = getpass.getuser()
        except Exception:
            user = "unknown"
    user = user or "unknown"
    tmpdir = str(env.get("TMPDIR", "")).strip()
    candidates = ([Path(tmpdir)] if tmpdir else []) + [
        Path("/scratch") / user,
        Path("/tmp") / user,
    ]
    for candidate in candidates:
        try:
            if candidate.is_dir() and os.access(candidate, os.W_OK):
                return candidate, None
        except Exception:
            continue
    return None, "no_writable_staging_root"


def _copy_hash_and_sync_sequential(source: Path, destination: Path) -> tuple[str, int]:
    """The v5 copier: one thread, 16 MiB reads, inline SHA-256, fsync."""

    digest = hashlib.sha256()
    copied_bytes = 0
    with source.open("rb") as source_handle, destination.open("wb") as target_handle:
        while True:
            payload = source_handle.read(_COPY_BUFFER_BYTES)
            if not payload:
                break
            target_handle.write(payload)
            digest.update(payload)
            copied_bytes += len(payload)
        target_handle.flush()
        os.fsync(target_handle.fileno())
    return digest.hexdigest(), copied_bytes


# Parallel staging copy.  On the main thread the cold-node bank copy (190 GB)
# runs at ~564 MB/s (5.3-6.6 min, 14-16% of a K=1 run).  Reads and writes run
# on worker threads with positional I/O into a fixed pool of large buffers,
# while the calling thread hashes the chunks strictly in file order (hashlib
# releases the GIL), so the digest and the staged bytes are exactly those of
# the sequential copier, as are the completion receipt, the identity checks
# and the reuse verification.
COPY_WORKERS_ENV = "IRG_BANK_STAGING_COPY_WORKERS"
COPY_CHUNK_MIB_ENV = "IRG_BANK_STAGING_COPY_CHUNK_MIB"
_DEFAULT_COPY_WORKERS = 4
_DEFAULT_COPY_CHUNK_BYTES = 64 * 2**20
_MIN_COPY_CHUNK_BYTES = 4096
_MAX_COPY_CHUNK_BYTES = 256 * 2**20


def _copy_settings(
    workers: int | None,
    chunk_bytes: int | None,
    env: Mapping[str, str] | None = None,
) -> tuple[int, int]:
    source = os.environ if env is None else env
    if workers is None:
        raw = str(source.get(COPY_WORKERS_ENV, "")).strip()
        try:
            workers = int(raw) if raw else _DEFAULT_COPY_WORKERS
        except ValueError:
            workers = _DEFAULT_COPY_WORKERS
    if chunk_bytes is None:
        raw = str(source.get(COPY_CHUNK_MIB_ENV, "")).strip()
        try:
            chunk_bytes = int(float(raw) * 2**20) if raw else _DEFAULT_COPY_CHUNK_BYTES
        except ValueError:
            chunk_bytes = _DEFAULT_COPY_CHUNK_BYTES
    workers = max(1, min(int(workers), 32))
    chunk_bytes = max(_MIN_COPY_CHUNK_BYTES, min(int(chunk_bytes), _MAX_COPY_CHUNK_BYTES))
    return workers, chunk_bytes


def _pread_into(descriptor: int, view: memoryview, offset: int) -> int:
    """Fill ``view`` from ``offset``; fewer bytes only at end of file."""

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


def _pwrite_all(descriptor: int, view: memoryview, offset: int) -> None:
    written = 0
    length = len(view)
    while written < length:
        count = os.pwrite(descriptor, view[written:], offset + written)
        if count <= 0:  # pragma: no cover - defensive
            raise OSError("short positional write while staging the bank")
        written += count


def _copy_hash_and_sync(
    source: Path,
    destination: Path,
    *,
    workers: int | None = None,
    chunk_bytes: int | None = None,
) -> tuple[str, int]:
    """Copy ``source`` to ``destination``, return (sha256, bytes), fsync.

    Byte-for-byte the v5 contract: the file is read to EOF, the digest is the
    SHA-256 of exactly the bytes written, in order, and the destination is
    fsynced before return.  ``workers`` positional-read/write threads share a
    pool of ``workers + 2`` buffers of ``chunk_bytes``; the calling thread
    consumes chunks strictly in offset order.
    """

    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    workers, chunk = _copy_settings(workers, chunk_bytes)
    if workers <= 1:
        return _copy_hash_and_sync_sequential(source, destination)
    digest = hashlib.sha256()
    source_fd = os.open(str(source), os.O_RDONLY)
    try:
        target_fd = os.open(
            str(destination), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o666
        )
        try:
            size = int(os.fstat(source_fd).st_size)
            chunk_count = (size + chunk - 1) // chunk
            buffers = [bytearray(chunk) for _ in range(min(chunk_count, workers + 2))]
            free = list(range(len(buffers)))

            def transfer(slot: int, index: int) -> tuple[int, int, int]:
                offset = index * chunk
                want = min(chunk, size - offset)
                view = memoryview(buffers[slot])[:want]
                got = _pread_into(source_fd, view, offset)
                if got:
                    _pwrite_all(target_fd, view[:got], offset)
                return slot, index, got

            copied = 0
            short = False
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="bank-staging-copy"
            ) as executor:
                pending: deque[Any] = deque()
                submitted = 0
                try:
                    for index in range(chunk_count):
                        while submitted < chunk_count and free:
                            pending.append(
                                executor.submit(transfer, free.pop(), submitted)
                            )
                            submitted += 1
                        slot, observed_index, got = pending.popleft().result()
                        if observed_index != index:  # pragma: no cover - invariant
                            raise RuntimeError("staging copy consumed chunks out of order")
                        digest.update(memoryview(buffers[slot])[:got])
                        copied += got
                        free.append(slot)
                        if got < min(chunk, size - index * chunk):
                            short = True  # EOF before the stat size: stop here
                            break
                finally:
                    for future in pending:
                        future.cancel()
            # Everything past ``copied`` is either absent or was written by a
            # chunk after a short one; the sequential copier would not have
            # written it, so truncate before appending any tail.
            if short:
                os.ftruncate(target_fd, copied)
            # Read to EOF exactly as the sequential copier does (a file that
            # grew after the stat is copied in full; normally this is empty).
            tail = bytearray(_COPY_BUFFER_BYTES)
            while True:
                got = _pread_into(source_fd, memoryview(tail), copied)
                if not got:
                    break
                _pwrite_all(target_fd, memoryview(tail)[:got], copied)
                digest.update(memoryview(tail)[:got])
                copied += got
            os.fsync(target_fd)
        finally:
            os.close(target_fd)
    finally:
        os.close(source_fd)
    return digest.hexdigest(), copied


def _copy_and_sync(source: Path, destination: Path) -> None:
    """Compatibility copier for uncertified callers on the legacy fallback path."""

    _copy_hash_and_sync(source, destination)


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    encoded = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")
    with partial.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    _fsync_directory(path.parent)


def _completion_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["receipt_hash"] = _canonical_hash(result)
    return result


def _load_completion_receipt(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    receipt_hash = str(payload.pop("receipt_hash", ""))
    if receipt_hash != _canonical_hash(payload):
        raise ValueError("completion_receipt_hash_mismatch")
    if payload.get("schema_name") != _SCHEMA_NAME or payload.get("complete") is not True:
        raise ValueError("completion_receipt_schema_or_state_mismatch")
    return {**payload, "receipt_hash": receipt_hash}


def _normalize_source_identity(
    identity: Mapping[str, Any],
    *,
    inputs_size: int,
    labels_size: int,
) -> dict[str, Any]:
    result = {
        "schema_name": _SOURCE_IDENTITY_SCHEMA,
        "inputs_sha256": str(identity.get("inputs_sha256", "")),
        "labels_sha256": str(identity.get("labels_sha256", "")),
        "inputs_size_bytes": int(identity.get("inputs_size_bytes", -1)),
        "labels_size_bytes": int(identity.get("labels_size_bytes", -1)),
    }
    for label in ("inputs_sha256", "labels_sha256"):
        value = result[label]
        if (
            len(value) != 64
            or value.lower() != value
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{label}_is_not_canonical_sha256")
    if (
        result["inputs_size_bytes"] != inputs_size
        or result["labels_size_bytes"] != labels_size
    ):
        raise ValueError("certified_source_sizes_differ")
    return result


def _acquire_flock(handle: Any, *, timeout: float) -> float:
    wait_started = time.monotonic()
    contended = False
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return float(time.monotonic() - wait_started) if contended else 0.0
        except BlockingIOError:
            contended = True
            if time.monotonic() - wait_started >= timeout:
                raise TimeoutError("lock_timeout")
            time.sleep(min(0.1, max(0.01, timeout / 20.0)))


def _file_stat_payload(path: Path, *, sha256: str) -> dict[str, Any]:
    stat = path.stat()
    return {
        "sha256": str(sha256),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
    }


def _verify_reusable_replica(
    *,
    target_dir: Path,
    staged_inputs: Path,
    staged_labels: Path,
    staged_receipt: Path,
    replica_key: str,
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not (
        target_dir.is_dir()
        and not target_dir.is_symlink()
        and staged_inputs.is_file()
        and staged_labels.is_file()
        and staged_receipt.is_file()
    ):
        raise ValueError("replica_files_incomplete")
    completion = _load_completion_receipt(staged_receipt)
    if (
        completion.get("replica_key") != replica_key
        or completion.get("source_identity") != dict(source_identity)
    ):
        raise ValueError("replica_identity_mismatch")
    expected_files = completion.get("files")
    if not isinstance(expected_files, Mapping):
        raise ValueError("replica_file_receipts_missing")
    for name, path in (("inputs.npy", staged_inputs), ("labels.npy", staged_labels)):
        expected = expected_files.get(name)
        if not isinstance(expected, Mapping):
            raise ValueError(f"replica_{name}_receipt_missing")
        stat = path.stat()
        observed = {
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "device": int(stat.st_dev),
            "inode": int(stat.st_ino),
        }
        if any(int(expected.get(key, -1)) != value for key, value in observed.items()):
            raise ValueError(f"replica_{name}_metadata_mismatch")
        identity_label = "inputs_sha256" if name == "inputs.npy" else "labels_sha256"
        if str(expected.get("sha256", "")) != str(source_identity[identity_label]):
            raise ValueError(f"replica_{name}_content_identity_mismatch")
    return completion


def _quarantine(target_dir: Path) -> Path:
    timestamp = int(time.time())
    quarantine = target_dir.with_name(f"{target_dir.name}.corrupt-{timestamp}")
    suffix = 1
    while quarantine.exists():
        quarantine = target_dir.with_name(f"{target_dir.name}.corrupt-{timestamp}-{suffix}")
        suffix += 1
    target_dir.rename(quarantine)
    return quarantine


def _cleanup_partials(paths: list[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def _resolve_staged_bank_compatibility(
    inputs_path: Path,
    labels_path: Path,
    *,
    verify_replica: Callable[[Path, Path], None] | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Resolve an optional flock-coordinated node-local bank replica.

    Every staging error is converted to a receipt-bearing fallback to the
    original paths. The function therefore never raises to its caller.
    """

    original_inputs = Path(inputs_path)
    original_labels = Path(labels_path)
    resolved_env = os.environ if env is None else env
    receipt = _receipt()
    partials_created: list[Path] = []
    lock_handle: Any = None
    lock_acquired = False
    copy_started: float | None = None
    target_dir: Path | None = None
    target_created_in_call = False

    try:
        if (
            str(resolved_env.get("IRG_NODE_LOCAL_BANK_STAGING", ""))
            .strip()
            .lower()
            in {"0", "false", "no", "off"}
        ):
            return original_inputs, original_labels, receipt

        staging_root, root_error = _choose_staging_root(resolved_env)
        if staging_root is None:
            _fallback(receipt, root_error or "no_writable_staging_root")
            return original_inputs, original_labels, receipt
        receipt["staging_root"] = str(staging_root)

        inputs_stat = original_inputs.stat()
        labels_stat = original_labels.stat()
        inputs_size = int(inputs_stat.st_size)
        labels_size = int(labels_stat.st_size)
        bank_bytes = inputs_size + labels_size
        receipt["bank_bytes"] = bank_bytes

        statvfs = os.statvfs(staging_root)
        free_bytes = int(statvfs.f_bavail) * int(statvfs.f_frsize)
        receipt["staging_root_free_bytes_before"] = free_bytes
        min_free_gib = float(
            resolved_env.get("IRG_BANK_STAGING_MIN_FREE_GIB", _DEFAULT_MIN_FREE_GIB)
        )
        if min_free_gib < 0:
            raise ValueError("IRG_BANK_STAGING_MIN_FREE_GIB must be nonnegative")
        required_bytes = bank_bytes + int(min_free_gib * 2**30)
        if free_bytes < required_bytes:
            _fallback(receipt, "insufficient_free_space")
            return original_inputs, original_labels, receipt

        bank_dir_name = original_inputs.resolve().parent.name
        if not bank_dir_name:
            raise ValueError("inputs path has no bank directory basename")
        layout_root = staging_root / "irg_uint8_banks"
        target_dir = layout_root / bank_dir_name
        staged_inputs = target_dir / "inputs.npy"
        staged_labels = target_dir / "labels.npy"
        staged_receipt = target_dir / "STAGED.json"
        lock_path = layout_root / f"{bank_dir_name}.lock"
        layout_root.mkdir(parents=True, exist_ok=True)

        timeout = float(
            resolved_env.get(
                "IRG_BANK_STAGING_LOCK_TIMEOUT_SECONDS",
                DEFAULT_LOCK_TIMEOUT_SECONDS,
            )
        )
        if timeout < 0:
            raise ValueError("IRG_BANK_STAGING_LOCK_TIMEOUT_SECONDS must be nonnegative")
        lock_handle = lock_path.open("a+b")
        wait_started = time.monotonic()
        contended = False
        while True:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_acquired = True
                receipt["lock_wait_seconds"] = (
                    float(time.monotonic() - wait_started) if contended else 0.0
                )
                break
            except BlockingIOError:
                contended = True
                if time.monotonic() - wait_started >= timeout:
                    receipt["lock_wait_seconds"] = float(time.monotonic() - wait_started)
                    _fallback(receipt, "lock_timeout")
                    return original_inputs, original_labels, receipt
                time.sleep(min(0.1, max(0.01, timeout / 20.0)))

        replica_files_present = (
            staged_receipt.is_file() and staged_inputs.is_file() and staged_labels.is_file()
        )
        replica_sizes_match = False
        if replica_files_present:
            replica_sizes_match = (
                int(staged_inputs.stat().st_size) == inputs_size
                and int(staged_labels.stat().st_size) == labels_size
            )
        if replica_files_present and replica_sizes_match:
            if verify_replica is not None:
                receipt["probe_verification"] = {"passed": False, "ran": True}
                try:
                    verify_replica(staged_inputs, staged_labels)
                except Exception as exc:
                    _quarantine(target_dir)
                    _fallback(
                        receipt,
                        "replica_probe_verification_failed:" f"{type(exc).__name__}:{exc}",
                    )
                    return original_inputs, original_labels, receipt
                receipt["probe_verification"] = {"passed": True, "ran": True}
            else:
                receipt["probe_verification"] = {"passed": False, "ran": False}
            receipt["staging_mode"] = "staged"
            receipt["staged_path"] = str(target_dir)
            return staged_inputs, staged_labels, receipt

        if target_dir.exists():
            _quarantine(target_dir)
        target_dir.mkdir(parents=True, exist_ok=False)
        target_created_in_call = True
        inputs_partial = target_dir / "inputs.npy.partial"
        labels_partial = target_dir / "labels.npy.partial"
        partials_created.extend((inputs_partial, labels_partial))
        copy_started = time.perf_counter()
        _copy_and_sync(original_inputs, inputs_partial)
        os.replace(inputs_partial, staged_inputs)
        _copy_and_sync(original_labels, labels_partial)
        os.replace(labels_partial, staged_labels)
        copy_seconds = float(time.perf_counter() - copy_started)
        receipt["copy_seconds"] = copy_seconds

        if (
            int(staged_inputs.stat().st_size) != inputs_size
            or int(staged_labels.stat().st_size) != labels_size
        ):
            _quarantine(target_dir)
            _fallback(receipt, "replica_size_verification_failed")
            return original_inputs, original_labels, receipt

        if verify_replica is not None:
            receipt["probe_verification"] = {"passed": False, "ran": True}
            try:
                verify_replica(staged_inputs, staged_labels)
            except Exception as exc:
                _quarantine(target_dir)
                _fallback(
                    receipt,
                    "replica_probe_verification_failed:" f"{type(exc).__name__}:{exc}",
                )
                return original_inputs, original_labels, receipt
            receipt["probe_verification"] = {"passed": True, "ran": True}
        else:
            receipt["probe_verification"] = {"passed": False, "ran": False}

        staged_payload = {
            "schema_name": _SCHEMA_NAME,
            "source_inputs_path": str(original_inputs.resolve()),
            "source_inputs_size_bytes": inputs_size,
            "source_inputs_mtime": float(inputs_stat.st_mtime),
            "source_labels_path": str(original_labels.resolve()),
            "source_labels_size_bytes": labels_size,
            "source_labels_mtime": float(labels_stat.st_mtime),
            "copied_bytes": bank_bytes,
            "copy_seconds": copy_seconds,
            "hostname": socket.gethostname(),
            "staged_at_unix": time.time(),
        }
        staged_json_partial = target_dir / "STAGED.json.partial"
        partials_created.append(staged_json_partial)
        _write_json_atomic(staged_receipt, staged_payload)
        receipt["staging_mode"] = "staged"
        receipt["staged_path"] = str(target_dir)
        return staged_inputs, staged_labels, receipt
    except Exception as exc:
        if copy_started is not None and receipt["copy_seconds"] == 0.0:
            receipt["copy_seconds"] = float(time.perf_counter() - copy_started)
        _cleanup_partials(partials_created)
        partials_created.clear()
        if target_created_in_call and target_dir is not None and target_dir.exists():
            try:
                _quarantine(target_dir)
            except Exception:
                pass
        _fallback(receipt, f"staging_exception:{type(exc).__name__}:{exc}")
        return original_inputs, original_labels, receipt
    finally:
        _cleanup_partials(partials_created)
        if lock_handle is not None:
            if lock_acquired:
                try:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
            try:
                lock_handle.close()
            except Exception:
                pass


def resolve_staged_bank(
    inputs_path: Path,
    labels_path: Path,
    *,
    verify_replica: Callable[[Path, Path], None] | None = None,
    source_identity: Mapping[str, Any] | None = None,
    required: bool | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Resolve a certified, content-keyed node-local bank replica.

    Certified callers provide the already-authoritative source SHA-256 and
    size identity, avoiding a source-sized pre-copy hash pass. The copy loop
    hashes bytes inline, and the completion receipt is written durably last.
    Calls without a certified identity retain the historical best-effort
    behavior. Only explicit ``required=True`` (or
    ``IRG_NODE_LOCAL_BANK_STAGING_REQUIRED=1``) converts fallback into a
    fail-closed exception.
    """

    original_inputs = Path(inputs_path)
    original_labels = Path(labels_path)
    resolved_env = os.environ if env is None else env
    receipt = _receipt()
    partials_created: list[Path] = []
    lock_handle: Any = None
    capacity_lock_handle: Any = None
    lock_acquired = False
    capacity_lock_acquired = False
    copy_started: float | None = None
    target_dir: Path | None = None
    target_created_in_call = False

    if required is not None and type(required) is not bool:
        raise TypeError("required must be a bool or None")
    required_mode = bool(required) if required is not None else False
    if required is None:
        raw_required = str(
            resolved_env.get("IRG_NODE_LOCAL_BANK_STAGING_REQUIRED", "")
        ).strip().lower()
        if raw_required in _TRUE_VALUES:
            required_mode = True
        elif raw_required and raw_required not in _FALSE_VALUES:
            _fallback(receipt, "invalid_required_mode")
            return original_inputs, original_labels, receipt
    receipt["required_mode"] = required_mode

    def fail(reason: str) -> tuple[Path, Path, dict[str, Any]]:
        _fallback(receipt, reason)
        if required_mode:
            raise BankStagingRequiredError(reason, receipt)
        return original_inputs, original_labels, receipt

    try:
        staging_switch = str(
            resolved_env.get("IRG_NODE_LOCAL_BANK_STAGING", "")
        ).strip().lower()
        if staging_switch in _FALSE_VALUES:
            if required_mode:
                return fail("staging_disabled_by_environment")
            return original_inputs, original_labels, receipt

        if source_identity is None:
            if required_mode:
                return fail("certified_source_identity_unavailable")
            return _resolve_staged_bank_compatibility(
                original_inputs,
                original_labels,
                verify_replica=verify_replica,
                env=resolved_env,
            )

        staging_root, root_error = _choose_staging_root(resolved_env)
        if staging_root is None:
            return fail(root_error or "no_writable_staging_root")
        receipt["staging_root"] = str(staging_root)

        inputs_size = int(original_inputs.stat().st_size)
        labels_size = int(original_labels.stat().st_size)
        bank_bytes = inputs_size + labels_size
        receipt["bank_bytes"] = bank_bytes
        normalized_identity = _normalize_source_identity(
            source_identity,
            inputs_size=inputs_size,
            labels_size=labels_size,
        )
        source_identity_hash = _canonical_hash(normalized_identity)
        replica_key = source_identity_hash
        receipt["source_identity_hash"] = source_identity_hash
        receipt["replica_key"] = replica_key

        min_free_gib = float(
            resolved_env.get("IRG_BANK_STAGING_MIN_FREE_GIB", _DEFAULT_MIN_FREE_GIB)
        )
        if min_free_gib < 0:
            raise ValueError("IRG_BANK_STAGING_MIN_FREE_GIB must be nonnegative")
        required_bytes = bank_bytes + int(min_free_gib * 2**30)
        receipt["staging_required_bytes"] = required_bytes

        timeout = float(
            resolved_env.get(
                "IRG_BANK_STAGING_LOCK_TIMEOUT_SECONDS",
                DEFAULT_LOCK_TIMEOUT_SECONDS,
            )
        )
        if timeout < 0:
            raise ValueError("IRG_BANK_STAGING_LOCK_TIMEOUT_SECONDS must be nonnegative")

        layout_root = staging_root / "irg_uint8_banks"
        layout_root.mkdir(parents=True, exist_ok=True)
        target_dir = layout_root / replica_key
        staged_inputs = target_dir / "inputs.npy"
        staged_labels = target_dir / "labels.npy"
        staged_receipt = target_dir / "STAGED.json"
        receipt["completion_receipt_path"] = str(staged_receipt)

        lock_handle = (layout_root / f"{replica_key}.lock").open("a+b")
        try:
            receipt["lock_wait_seconds"] = _acquire_flock(
                lock_handle,
                timeout=timeout,
            )
            lock_acquired = True
        except TimeoutError:
            return fail("lock_timeout")

        if target_dir.exists():
            try:
                completion = _verify_reusable_replica(
                    target_dir=target_dir,
                    staged_inputs=staged_inputs,
                    staged_labels=staged_labels,
                    staged_receipt=staged_receipt,
                    replica_key=replica_key,
                    source_identity=normalized_identity,
                )
                if verify_replica is not None:
                    receipt["probe_verification"] = {"passed": False, "ran": True}
                    verify_replica(staged_inputs, staged_labels)
                    receipt["probe_verification"] = {"passed": True, "ran": True}
                else:
                    receipt["probe_verification"] = {"passed": False, "ran": False}
                receipt["reuse_verification"] = {
                    "passed": True,
                    "completion_receipt_hash": completion["receipt_hash"],
                    "metadata_bound": True,
                }
                receipt["copy_hashes"] = {
                    name: str(payload["sha256"])
                    for name, payload in completion["files"].items()
                }
                receipt["staging_mode"] = "staged"
                receipt["staged_path"] = str(target_dir)
                receipt["replica_reused"] = True
                return staged_inputs, staged_labels, receipt
            except Exception as exc:
                receipt["replica_rebuild_reason"] = f"{type(exc).__name__}:{exc}"
                receipt["reuse_verification"] = {
                    "passed": False,
                    "reason": receipt["replica_rebuild_reason"],
                }
                _quarantine(target_dir)

        capacity_lock_handle = (layout_root / ".capacity.lock").open("a+b")
        try:
            receipt["capacity_lock_wait_seconds"] = _acquire_flock(
                capacity_lock_handle,
                timeout=timeout,
            )
            capacity_lock_acquired = True
        except TimeoutError:
            return fail("capacity_lock_timeout")

        statvfs = os.statvfs(staging_root)
        free_bytes = int(statvfs.f_bavail) * int(statvfs.f_frsize)
        receipt["staging_root_free_bytes_before"] = free_bytes
        if free_bytes < required_bytes:
            return fail("insufficient_free_space")

        target_dir.mkdir(parents=True, exist_ok=False)
        target_created_in_call = True
        _fsync_directory(layout_root)
        inputs_partial = target_dir / "inputs.npy.partial"
        labels_partial = target_dir / "labels.npy.partial"
        partials_created.extend((inputs_partial, labels_partial))
        copy_started = time.perf_counter()
        inputs_hash, copied_inputs_bytes = _copy_hash_and_sync(
            original_inputs,
            inputs_partial,
        )
        labels_hash, copied_labels_bytes = _copy_hash_and_sync(
            original_labels,
            labels_partial,
        )
        copy_seconds = float(time.perf_counter() - copy_started)
        receipt["copy_seconds"] = copy_seconds
        receipt["copy_hashes"] = {
            "inputs.npy": inputs_hash,
            "labels.npy": labels_hash,
        }
        if (
            copied_inputs_bytes != inputs_size
            or copied_labels_bytes != labels_size
            or inputs_hash != normalized_identity["inputs_sha256"]
            or labels_hash != normalized_identity["labels_sha256"]
        ):
            _quarantine(target_dir)
            return fail("inline_copy_identity_verification_failed")

        os.replace(inputs_partial, staged_inputs)
        os.replace(labels_partial, staged_labels)
        _fsync_directory(target_dir)

        if verify_replica is not None:
            receipt["probe_verification"] = {"passed": False, "ran": True}
            try:
                verify_replica(staged_inputs, staged_labels)
            except Exception as exc:
                _quarantine(target_dir)
                return fail(
                    "replica_probe_verification_failed:"
                    f"{type(exc).__name__}:{exc}"
                )
            receipt["probe_verification"] = {"passed": True, "ran": True}
        else:
            receipt["probe_verification"] = {"passed": False, "ran": False}

        completion = _completion_payload(
            {
                "schema_name": _SCHEMA_NAME,
                "complete": True,
                "replica_key": replica_key,
                "source_identity_hash": source_identity_hash,
                "source_identity": normalized_identity,
                "source_paths": {
                    "inputs": str(original_inputs.resolve()),
                    "labels": str(original_labels.resolve()),
                },
                "files": {
                    "inputs.npy": _file_stat_payload(
                        staged_inputs,
                        sha256=inputs_hash,
                    ),
                    "labels.npy": _file_stat_payload(
                        staged_labels,
                        sha256=labels_hash,
                    ),
                },
                "copied_bytes": copied_inputs_bytes + copied_labels_bytes,
                "copy_seconds": copy_seconds,
                "hostname": socket.gethostname(),
                "staged_at_unix": time.time(),
            }
        )
        partials_created.append(target_dir / "STAGED.json.partial")
        _write_json_atomic(staged_receipt, completion)
        receipt["reuse_verification"] = {
            "passed": True,
            "completion_receipt_hash": completion["receipt_hash"],
            "metadata_bound": True,
        }
        receipt["staging_mode"] = "staged"
        receipt["staged_path"] = str(target_dir)
        return staged_inputs, staged_labels, receipt
    except BankStagingRequiredError:
        raise
    except Exception as exc:
        if copy_started is not None and receipt["copy_seconds"] == 0.0:
            receipt["copy_seconds"] = float(time.perf_counter() - copy_started)
        _cleanup_partials(partials_created)
        partials_created.clear()
        if target_created_in_call and target_dir is not None and target_dir.exists():
            try:
                _quarantine(target_dir)
            except Exception:
                pass
        return fail(f"staging_exception:{type(exc).__name__}:{exc}")
    finally:
        _cleanup_partials(partials_created)
        for handle, acquired in (
            (capacity_lock_handle, capacity_lock_acquired),
            (lock_handle, lock_acquired),
        ):
            if handle is None:
                continue
            if acquired:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
            try:
                handle.close()
            except Exception:
                pass
