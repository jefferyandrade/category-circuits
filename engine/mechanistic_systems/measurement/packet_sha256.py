"""Native SHA-256 packet compression for the existing sampling identities."""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tempfile
import threading

import numpy as np


_LIBRARY = None
_LOAD_LOCK = threading.Lock()


def prewarm_packet_sha256():
    """Build and load once per worker, before submitting its branch planner."""

    global _LIBRARY
    with _LOAD_LOCK:
        if _LIBRARY is not None and _LIBRARY[0] == os.getpid():
            return _LIBRARY[1]
        source = Path(__file__).with_suffix(".c")
        build = Path(os.environ.get("IRG_I2_NATIVE_CACHE_DIR", "") or
                     str(Path(tempfile.gettempdir()) / "irg-i2-native"))
        build.mkdir(parents=True, exist_ok=True)
        identity = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        suffix = ".dylib" if sys.platform == "darwin" else ".so"
        target = build / f"packet_sha256-{sys.platform}-{platform.machine()}-{identity}{suffix}"
        with (build / "build.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not target.exists():
                temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
                compiler = shlex.split(os.environ.get("CC", "cc"))
                flags = ["-O3", "-std=c99", "-Wall", "-Wextra", "-Werror", "-fPIC",
                         "-dynamiclib" if sys.platform == "darwin" else "-shared"]
                try:
                    subprocess.run([*compiler, *flags, str(source), "-o", str(temporary)],
                                   check=True, capture_output=True, text=True)
                    temporary.replace(target)
                except subprocess.CalledProcessError as error:
                    raise RuntimeError(f"packet SHA-256 compilation failed: {error.stderr.strip()}") from error
                finally:
                    temporary.unlink(missing_ok=True)
        library = ctypes.CDLL(str(target))
        function = library.irg_sha256_packet
        function.argtypes = (
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        )
        function.restype = ctypes.c_int
        _LIBRARY = (os.getpid(), library)
        return library


def digest_packet(messages):
    """Digest a contiguous packet in one call outside the interpreter lock.

    The two timings come from the native call's monotonic and thread CPU
    clocks. Subtracting native CPU from wrapper CPU gives a conservative
    upper bound on wrapper CPU while holding the interpreter lock.
    """

    messages = tuple(messages)
    library = prewarm_packet_sha256()
    lengths = np.fromiter(map(len, messages), dtype=np.uint64, count=len(messages))
    offsets = np.empty(len(messages), dtype=np.uint64)
    if len(messages):
        offsets[0] = 0
        np.cumsum(lengths[:-1], out=offsets[1:])
    payload = np.frombuffer(b"".join(messages), dtype=np.uint8)
    output = np.empty((len(messages), 32), dtype=np.uint8)
    wall_seconds, thread_cpu_seconds = ctypes.c_double(), ctypes.c_double()
    code = library.irg_sha256_packet(
        payload.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), len(payload),
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), len(messages),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.byref(wall_seconds), ctypes.byref(thread_cpu_seconds),
    )
    if code:
        reason = "packet byte ranges are invalid" if code == 1 else "native packet clocks failed"
        raise RuntimeError(reason)
    raw = output.tobytes()
    return tuple(raw[index:index + 32] for index in range(0, len(raw), 32)), wall_seconds.value, thread_cpu_seconds.value
