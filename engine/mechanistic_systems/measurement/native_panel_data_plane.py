"""Streaming input data plane for native scoring and calibrated-BCE execution.

The statistical controllers own logical IID draws, Neyman targets, and stopping.
This module changes only how an already-frozen ordered set of *physical* context
IDs is gathered and delivered to the native Torch evaluator.

The optimized path reads the certified uint8 bank in bounded chunks, overlaps
ordered host gathering with GPU work, normalizes on the device, and retains a
byte-bounded physical-context LRU.  The exploratory image-folder fallback is
kept explicit and receipted; it cannot satisfy the optimized runtime gate.
"""

from __future__ import annotations

import os
import mmap
import platform
import queue
import resource
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from contracts.hashing import stable_digest
from contracts.runtime_receipts import runtime_receipt_hash
from .run_products import effective_settings
from mechanistic_systems.measurement.contexts import (
    PreprocessedUint8BankContextSource,
)
from mechanistic_systems.measurement.torch_adapter import MeasurementInputBatch
from mechanistic_systems.measurement.gpu_runtime import (
    configure_process_device_memory,
    cuda_reusable_memory_bytes,
    gpu_process_count,
    process_device_memory_budget_bytes,
)
from mechanistic_systems.measurement.device_image_gather import (
    DEVICE_GATHER_ENV,
    DeviceGatherError,
    DeviceImageGather,
    gather_budget_bytes,
    memlock_limit,
    register_readonly_bank,
)
from packages.loader_runtime.throughput import (
    PipelineCloseTimeout,
    PooledBackgroundIterator,
)

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


NATIVE_PANEL_DATA_PLANE_SCHEMA = "mechanistic_native_panel_data_plane_v2"

# Delivered-bytes integrity probe over pinned ring slots: one POSITION-
# WEIGHTED int64 sum of a strided byte sample, computed at staging,
# re-checked at consumption, and (for full-slot asynchronous copies)
# recomputed on the delivered device tensor. Prime stride so structured
# image data cannot alias the sample; position weights so multiset-equal
# rewrites (e.g. a permutation of the same class labels under a class-major
# terminal panel) cannot collide the sum exactly.
_DELIVERY_CHECKSUM_STRIDE = 8191
_DELIVERY_CHECKSUM_CONTRACT = "position_weighted_strided_int64_sum_stride8191_v2"
NATIVE_PANEL_RECEIPT_SCHEMA = "mechanistic_native_panel_receipt_v1"

# A scientific/semantic role is evidence identity; a physical role is the
# execution-policy and aggregate-telemetry bucket.  Keeping this translation
# in one closed registry prevents development workflows from silently inventing
# an execution policy while also avoiding the former requirement that every
# semantic alias be duplicated in the batch-size and counter implementations.
_PHYSICAL_ROLE_BY_SEMANTIC_ROLE = {
    "scoring": "scoring",
    "calibration": "calibration",
    "category_calibration": "calibration",
    "calibration_oracle_fused": "calibration_oracle_fused",
    "v0_paired_oracle": "v0_paired_oracle",
    # The matched-v0 full-train ceiling keeps K and the two disjoint O strata
    # as distinct semantic evidence roles while sharing the one registered
    # v0 paired-oracle execution policy.  These aliases are deliberately
    # closed here: an unregistered runner role must fail before streaming.
    "v0_train_oracle_calibration_k": "v0_paired_oracle",
    "v0_train_oracle_o_target": "v0_paired_oracle",
    "v0_train_oracle_o_complement": "v0_paired_oracle",
    "v1_candidate_oracle": "v1_candidate_oracle",
    "v1_parent_aware_oracle": "v1_candidate_oracle",
    "v1_correction_parent_shift": "v1_candidate_oracle",
    "v1_final_mask_calibration_fit": "calibration",
    "static_u1_c2_optimization_evaluation": "v1_candidate_oracle",
    "static_exact_gate_fresh_evaluation": "v1_candidate_oracle",
    "static_score_components_optimization_evaluation": "v1_candidate_oracle",
    "retrospective_magnitude_sanity_evaluation": "v1_candidate_oracle",
    "static_exact_gate_production_plane_development_evaluation": "terminal",
    "static_point_order_full_validation_development_evaluation": "terminal",
    "terminal": "terminal",
}
_SEMANTIC_ROLE_NAMES = tuple(_PHYSICAL_ROLE_BY_SEMANTIC_ROLE)
_ROLE_NAMES = tuple(dict.fromkeys(_PHYSICAL_ROLE_BY_SEMANTIC_ROLE.values()))


def _physical_native_panel_role(role: str) -> str:
    semantic_role = str(role)
    try:
        return _PHYSICAL_ROLE_BY_SEMANTIC_ROLE[semantic_role]
    except KeyError as error:
        raise ValueError(
            f"unknown native panel semantic role {semantic_role!r}"
        ) from error

# IRG_PERSISTENT_PANEL_PRODUCER: one producer pool whose lifetime is
# the data plane (the arm), not the look, plus a cross-look ready-buffer the
# provider fills the moment a look's chunk list is determined.  Timing only:
# WHICH contexts are drawn, panel composition, chunk boundaries, batch
# grouping, and delivery order are computed by the unchanged serve path; the
# lookahead merely moves the wall-clock instant at which a gather executes.
# The master gate is default ON; with its explicit kill switch set the
# serve path is structurally the per-look pipeline and the
# runtime receipt is unchanged.
_PERSISTENT_PANEL_PRODUCER_ENV = "IRG_PERSISTENT_PANEL_PRODUCER"
_PANEL_LOOKAHEAD_LOOKS_ENV = "IRG_PANEL_LOOKAHEAD_LOOKS"
_DEEP_IMAGE_LOOKAHEAD_ENV = "IRG_DEEP_IMAGE_LOOKAHEAD"
# Resident pre-gathered chunk budget.  At a fixed workers+depth (=16) bound
# posts were measured budget_exhausted with ~6.5 enqueued chunks
# per posted look and occasional larger fused unions; the derived default
# triples the pipeline's own in-flight bound so the measured posting
# volume never exhausts it, while staying a HARD bound (never unbounded).
_PANEL_LOOKAHEAD_CHUNK_BUDGET_ENV = "IRG_PANEL_LOOKAHEAD_CHUNK_BUDGET"
_LOOKAHEAD_MISMATCH_SAMPLE_CAP = 8
# The fp32 panel transient is written into a bounded per-shape
# ring of persistent buffers instead of fresh allocations — SAME ops, SAME
# kernels, SAME order; only allocation behavior changes, so bitwise holds
# by construction.  Default OFF.
_PANEL_NORMALIZE_BUFFER_REUSE_ENV = "IRG_PANEL_NORMALIZE_BUFFER_REUSE"
_IMAGE_GATHER_ENV = "IRG_IMAGE_GATHER"
_BANK_PAGE_ADVICE_ENV = "IRG_BANK_PAGE_ADVICE"
# One setting selects the worker's pinned byte pool and its storage bound.
# "auto" derives the working set; an explicit zero disables the byte pool.
_PINNED_POOL_MAX_BYTES_ENV = "IRG_PINNED_POOL_MAX_BYTES"
_PINNED_POOL_DEFAULT_BYTES = "auto"
# Normalization folded into conv1 weights/bias.  Machinery and acceptance
# protocol live here; see ``compute_uint8_folded_stem``.
_UINT8_FOLDED_STEM_ENV = "IRG_UINT8_FOLDED_STEM"


def compute_uint8_folded_stem(
    conv_weight: Any,
    conv_bias: Any,
    *,
    decode_scale: Any,
    decode_bias: Any,
    detach: bool = True,
) -> dict[str, Any]:
    """Fold uint8 decode normalization into first-conv weights.

    In the v0 plane's own decode parameterization
    ``x_norm = u8 * scale_c + bias_c``:

        conv(x_norm, W, b) == conv(u8_fp32, W * scale_c, b + sum_ckk(W * bias_c))

    with zero-padding in NORMALIZED space realized by padding the RAW
    input with ``pad_value_c = -bias_c / scale_c`` (the raw code of
    normalized zero; 255*mean in the legacy mean/std form) and running
    the folded conv with padding=0.

    Acceptance is numerical equivalence, not bitwise equality.
    The real-arithmetic identity has different fp32 rounding from
    normalize-then-conv. Tests state their tolerances; a composed GPU run
    must report any decision disagreement and its distance to tau.
    ``detach=False`` retains gradients when folding effective gated weights.

    Returns {"weight", "bias", "pad_value"} — pad_value shaped
    ``[1, C, 1, 1]`` for the legacy edge-fill pattern.
    """

    framework = _require_torch()
    weight = (conv_weight.detach() if detach else conv_weight).to(dtype=framework.float32)
    # Decode parameters are scientific preprocessing constants and are
    # normally materialized on CPU.  Production conv owners live on CUDA;
    # perform the one-time fold on the owner's device so the transaction
    # cannot fail (or accidentally copy the weight back to host) merely
    # because the certified constants are host-resident.
    scale = decode_scale.detach().to(
        device=weight.device, dtype=framework.float32
    ).reshape(-1)
    bias_c = decode_bias.detach().to(
        device=weight.device, dtype=framework.float32
    ).reshape(-1)
    if detach and (not bool(framework.isfinite(scale).all()) or bool((scale == 0).any())
                   or not bool(framework.isfinite(bias_c).all())):
        raise ValueError("folded stem requires finite decode constants and nonzero scales")
    if int(weight.shape[1]) != int(scale.numel()) or int(
        scale.numel()
    ) != int(bias_c.numel()):
        raise ValueError("folded stem channel axes disagree")
    folded_weight = weight * scale.view(1, -1, 1, 1)
    bias_shift = (weight.sum(dim=(2, 3)) * bias_c.view(1, -1)).sum(dim=1)
    base_bias = (
        (conv_bias.detach() if detach else conv_bias).to(
            device=weight.device, dtype=framework.float32
        )
        if conv_bias is not None
        else framework.zeros(
            (int(weight.shape[0]),),
            dtype=framework.float32,
            device=weight.device,
        )
    )
    return {
        "weight": folded_weight,
        "bias": base_bias + bias_shift,
        "pad_value": (-bias_c / scale).view(1, -1, 1, 1),
    }


# Speculative next-increment posting (opt-in; inert without the
# master gate and a positive lookahead depth; the provider consults
# ``speculative_extension_enabled`` before deriving any speculative ids).
_SPECULATIVE_EXTENSION_PREFETCH_ENV = "IRG_SPECULATIVE_EXTENSION_PREFETCH"
# Target-class stratum rows pinned on device once per arm;
# independent of the producer master gate (it changes where bytes come
# from, never pooling).  CPU devices degenerate to the host path.
_ON_TARGET_DEVICE_PIN_ENV = "IRG_ON_TARGET_DEVICE_PIN"
_FALSE_ENVIRONMENT_TOKENS = frozenset({"0", "false", "no", "off"})
_LOOKAHEAD_DEGRADATION_REASON_CAP = 64
_PIN_BUILD_CHUNK_ROWS = 512


@dataclass
class _PinnedHostPool:
    max_bytes: int
    # Whether this budget came from ``auto`` (derived working set).  Only an
    # auto pool may later grow, and only for another auto request.
    auto: bool = False
    rings: dict = field(default_factory=dict)
    cursors: dict = field(default_factory=dict)
    ids: set = field(default_factory=set)
    storage: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)
    lock: Any = field(default_factory=threading.Lock)
    warm_lock: Any = field(default_factory=threading.Lock)
    warmed_capacity: int = 0
    reserved_bytes: int = 0
    allocations: int = 0
    reuses: int = 0
    evictions: int = 0
    pretouched_bytes: int = 0
    unpooled_allocations: int = 0
    pageable_allocations: int = 0
    event_syncs: int = 0
    grown_slots: int = 0
    overflow_allocations: int = 0


_WORKER_PINNED_POOLS: dict[str, _PinnedHostPool] = {}
_WORKER_PINNED_POOLS_LOCK = threading.Lock()


def _worker_pinned_pool(device: Any, max_bytes: int, *, auto: bool = False) -> _PinnedHostPool:
    """One pinned byte budget per worker/device, retained across its planes.

    With ``auto`` each plane derives its own working set; planes of one
    worker then share the pool at the largest budget any of them needs (it
    never shrinks under live leases). An explicit budget cannot change: a
    pool created with an explicit budget refuses any different request,
    auto or not, and an auto pool refuses a different explicit request.
    An accepted explicit request (even an equal one) fixes the budget.
    """
    key = str(device)
    with _WORKER_PINNED_POOLS_LOCK:
        pool = _WORKER_PINNED_POOLS.get(key)
        if pool is None:
            pool = _PinnedHostPool(max_bytes, auto=bool(auto))
            _WORKER_PINNED_POOLS[key] = pool
        elif pool.max_bytes != max_bytes:
            if not (auto and pool.auto):
                raise ValueError("worker pinned pool byte budget changed; start a new worker for the new setting")
            with pool.lock:
                pool.max_bytes = max(int(pool.max_bytes), int(max_bytes))
        if not auto:
            # An accepted explicit request fixes the budget, even when it
            # equals an auto pool's current one; later auto planes cannot grow it.
            pool.auto = False
        return pool


def image_gather_worker_count(
    env: Mapping[str, str] | None = None, *, requested_workers: int | None = None,
) -> int:
    """Respect the launcher's count within affinity, task and cgroup limits."""

    from .cpu_resources import _available_cpu_count

    values = os.environ if env is None else env
    available = _available_cpu_count()
    task_cpus = str(values.get("SLURM_CPUS_PER_TASK", "")).strip()
    if task_cpus:
        try:
            available = min(available, max(1, int(task_cpus)))
        except ValueError:
            pass
    for name in ("CIRCUIT_PRODUCER_WORKERS", "IRG_PANEL_PRODUCER_WORKERS"):
        raw = str(values.get(name, "")).strip()
        if raw:
            requested_workers = int(raw)
            break
    requested = 8 if requested_workers is None else int(requested_workers)
    if requested < 1:
        raise ValueError("image gather worker count must be positive")
    return max(1, min(available, requested))

# Delivered row identity detectors share IRG_READOUT_LABEL_INTEGRITY with
# the provider's label detector. Device row digests sample the assembled
# uint8 image rows on the copy stream. Staged image and label checksum
# stamps travel with the payload and settle after delivery at the panel's
# existing synchronization. No host payload re-read or host retry runs.
# Readout slab row identity detection lives in native_irg_providers.
#
# Row digest (algorithm v2).  The row's bytes in
# logical (C, H, W) order are read as little-endian SIGNED int32 words s_i
# (per-byte words when the byte count is not a multiple of 4) and reduced
# by two position-weighted lanes, both in exact int64 arithmetic:
#
#   lane A = sum_i (2 i + 1) * s_i                (distinct odd weights)
#   lane B = sum_i w_B(i)   * s_i                 (w_B odd, < 2**15, from a
#                                                  32-bit multiply-xorshift
#                                                  mix of i)
#   digest = ((B mod 2**31) << 32) | (A mod 2**32)     (int64, >= 0)
#
# Every weight is odd, hence invertible modulo 2**32: a change of ANY single
# word (so of any single bit) changes lane A and therefore the digest --
# guaranteed, not probable (v1's weights 1..n mod 2**64 annihilated high
# bits of even-weighted words; 1.56 % of single-bit flips were invisible).
# No product or partial sum can overflow int64: below 2**16 words the exact
# sums are bounded by 2**31 * n**2 and 2**46 * n; at or beyond it every
# product is masked to 32 bits before the sum (explicit masking), which
# yields the same residues.  Integer arithmetic
# makes the host (numpy) and device (torch) values exactly equal and
# independent of reduction order.  Not a cryptographic hash: it certifies
# against misdelivery, not adversaries.
DELIVERY_INTEGRITY_ENVIRONMENT = "IRG_READOUT_LABEL_INTEGRITY"
ROW_DIGEST_ALGORITHM = "uint8_row_le_i32_words_dual_lane_odd_weight_exact_int64_v2"
PAYLOAD_STAMP_CONTRACT = "staged_checksum_carried_on_payload_verified_at_device_batch_v1"
# CLASS3: delivered device tensors that are not ring-backed are
# stream-recorded on the consumer stream (``MeasurementInputBatch``), and
# pinned-path labels are backed by the normalize ring slot.  Not gated by
# the integrity kill switch (a correctness fix, value-neutral).
STREAM_LIFETIME_CONTRACT = "pinned_labels_ring_backed_and_side_stream_tensors_record_stream_v1"
_ROW_DIGEST_MISMATCH_SAMPLE_CAP = 16
# Where along the delivery path a row digest disagreed with the authority.
ROW_DIGEST_STAGE_GATHER = "gather_vs_bank"
ROW_DIGEST_STAGE_COPY = "device_after_copy_vs_gather"
ROW_DIGEST_STAGE_USE = "device_at_first_use_vs_gather"
ROW_DIGEST_STAGE_AFTER_USE = "device_after_use_vs_gather"


def delivery_integrity_enabled() -> bool:
    """IRG_READOUT_LABEL_INTEGRITY armed unless explicitly disarmed."""

    return (
        str(os.environ.get(DELIVERY_INTEGRITY_ENVIRONMENT, "")).strip().lower()
        not in _FALSE_ENVIRONMENT_TOKENS
    )


class DeliveryIntegrityError(RuntimeError):
    """A delivered chunk failed a row or label identity probe.

    ``details`` records the affected rows and delivery stage.
    """

    fault_class = "delivery"

    def __init__(self, message: str, details: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.details = dict(details)


class RowDigestIntegrityError(DeliveryIntegrityError):
    fault_class = "image_row_digest"


class DeliveryChecksumIntegrityError(DeliveryIntegrityError):
    """CLASS3: the pre-existing strided H2D checksum failed.

    Carries the first checksum message verbatim (so existing log greps and
    tests still match) plus BOTH settle reports: every checksum failure and
    any row-digest failure of the same panel.  Fatal, as it always was.
    """

    fault_class = "delivery_checksum"


# The fused digest, the bank row digest table, the guard
# modes.  That module imports this one lazily (inside functions only).
from mechanistic_systems.measurement import bank_row_digest_table as _lane_g  # noqa: E402

# -- dual-lane word digest core (shared by I1, I3 and the canonical dense
#    reference rows).  ``words`` are int64 matrices [rows, n] holding the
#    row's words as SIGNED int32 values (or unsigned bytes); ``extra``
#    optional non-negative int64 [rows, k] words (< 2**32) appended at
#    positions n .. n+k-1 (the I3 label fold).
_DIGEST_MASK32 = 0xFFFFFFFF
_DIGEST_MASK31 = 0x7FFFFFFF
# Below this many words both lanes are summed exactly (|lane A| <
# 2**31 * n**2 < 2**63, |lane B| < 2**46 * n); at or above it every product
# is masked to 32 bits first (sum < n * 2**32).  Same residues either way.
_DIGEST_EXACT_WORDS = 1 << 16
_DIGEST_MAX_WORDS = 1 << 30
_DIGEST_HOST_BLOCK_ROWS = 32
_DIGEST_DEVICE_BLOCK_BYTES = 64 * 1024 * 1024
_DIGEST_WEIGHT_CACHE: dict[tuple[Any, ...], Any] = {}


def _digest_weight_values(index: Any) -> tuple[Any, Any]:
    """Lane weights for word positions ``index`` (numpy or torch int64).

    Identical integer expressions in both frameworks; every intermediate
    stays below 2**63 (32-bit operand times a multiplier below 2**31).
    """

    lane_a = index * 2 + 1
    mixed = ((index + 0x2545F491) * 0x5BD1E995) & _DIGEST_MASK32
    mixed = mixed ^ (mixed >> 15)
    mixed = (mixed * 0x27D4EB2D) & _DIGEST_MASK32
    mixed = mixed ^ (mixed >> 13)
    lane_b = ((mixed & 0x3FFF) << 1) | 1
    return lane_a, lane_b


def _digest_weights_host(start: int, count: int) -> np.ndarray:
    key = (int(start), int(count), "host")
    cached = _DIGEST_WEIGHT_CACHE.get(key)
    if cached is None:
        index = np.arange(int(start), int(start) + int(count), dtype=np.int64)
        lane_a, lane_b = _digest_weight_values(index)
        cached = np.ascontiguousarray(np.stack((lane_a, lane_b), axis=1))
        if len(_DIGEST_WEIGHT_CACHE) < 64:
            _DIGEST_WEIGHT_CACHE[key] = cached
    return cached


def _digest_weights_device(start: int, count: int, device: Any) -> tuple[Any, Any]:
    """Build integer weights on the device and cache them per CUDA stream."""

    framework = _require_torch()
    resolved = framework.device(device)
    if resolved.type == "cuda":
        stream = framework.cuda.current_stream(resolved)
        key = (int(start), int(count), str(stream.device), stream.cuda_stream)
    else:
        key = (int(start), int(count), str(resolved))
    cached = _DIGEST_WEIGHT_CACHE.get(key)
    if cached is None:
        index = framework.arange(
            int(start), int(start) + int(count), device=resolved,
            dtype=framework.int64,
        )
        cached = _digest_weight_values(index)
        if len(_DIGEST_WEIGHT_CACHE) < 64:
            _DIGEST_WEIGHT_CACHE[key] = cached
    return cached


def _digest_finish(lane_a: Any, lane_b: Any) -> Any:
    return ((lane_b & _DIGEST_MASK31) << 32) | (lane_a & _DIGEST_MASK32)


def digest_words_host(words: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
    """Dual-lane digest of an int64 word matrix on the host (numpy)."""

    rows, count = int(words.shape[0]), int(words.shape[1])
    if count >= _DIGEST_MAX_WORDS:
        raise ValueError("row digest is defined for rows below 2**30 words")
    weights = _digest_weights_host(0, count)
    lanes = np.empty((rows, 2), dtype=np.int64)
    if count < _DIGEST_EXACT_WORDS:
        # int64 matmul is exact integer arithmetic (numpy's integer loops,
        # no BLAS); the bounds above exclude overflow of every partial sum.
        # Row blocks keep the widened words small.
        for start in range(0, rows, _DIGEST_HOST_BLOCK_ROWS):
            stop = min(rows, start + _DIGEST_HOST_BLOCK_ROWS)
            np.matmul(
                np.asarray(words[start:stop], dtype=np.int64), weights, out=lanes[start:stop]
            )
    else:
        for start in range(0, rows, _DIGEST_HOST_BLOCK_ROWS):
            stop = min(rows, start + _DIGEST_HOST_BLOCK_ROWS)
            block = np.asarray(words[start:stop], dtype=np.int64)
            for lane in (0, 1):
                product = block * weights[:, lane]
                product &= _DIGEST_MASK32
                lanes[start:stop, lane] = product.sum(axis=1, dtype=np.int64)
    # Reduce to residues before folding extra words: every later term is
    # masked, so nothing below can overflow either.
    lanes &= _DIGEST_MASK32
    if extra is not None and int(extra.shape[1]):
        extra_weights = _digest_weights_host(count, int(extra.shape[1]))
        extra64 = np.asarray(extra, dtype=np.int64)
        for lane in (0, 1):
            product = extra64 * extra_weights[:, lane]
            product &= _DIGEST_MASK32
            lanes[:, lane] += product.sum(axis=1, dtype=np.int64)
    return _digest_finish(lanes[:, 0], lanes[:, 1]).astype(np.int64, copy=False)


def digest_words_device(words: Any, extra: Any | None = None) -> Any:
    """The same digest as :func:`digest_words_host`, in torch on any device.

    ``words`` may be int32 (the zero-copy view of the row bytes) or int64;
    multiplying by the int64 weights promotes exactly.  On CUDA the int32
    view is multiplied directly (the kernel widens in registers, so no
    widened copy of the words is materialized); on CPU, where mixed-dtype
    elementwise loops are not vectorized, the words are widened first.
    Both orders compute the same exact integers.
    """

    framework = _require_torch()
    if words.dtype != framework.int64 and words.device.type != "cuda":
        words = words.to(framework.int64)
    count = int(words.shape[1])
    if count >= _DIGEST_MAX_WORDS:
        raise ValueError("row digest is defined for rows below 2**30 words")
    lane_a_weights, lane_b_weights = _digest_weights_device(0, count, words.device)
    masked = count >= _DIGEST_EXACT_WORDS
    product = framework.mul(words, lane_a_weights)
    if masked:
        product.bitwise_and_(_DIGEST_MASK32)
    lane_a = product.sum(dim=1).bitwise_and_(_DIGEST_MASK32)
    framework.mul(words, lane_b_weights, out=product)
    if masked:
        product.bitwise_and_(_DIGEST_MASK32)
    lane_b = product.sum(dim=1).bitwise_and_(_DIGEST_MASK32)
    del product
    if extra is not None and int(extra.shape[1]):
        extra_a, extra_b = _digest_weights_device(count, int(extra.shape[1]), words.device)
        extra64 = extra.to(dtype=framework.int64)
        lane_a = lane_a + (extra64 * extra_a).bitwise_and_(_DIGEST_MASK32).sum(dim=1)
        lane_b = lane_b + (extra64 * extra_b).bitwise_and_(_DIGEST_MASK32).sum(dim=1)
    return _digest_finish(lane_a, lane_b)


def _byte_rows_to_words_host(flat: np.ndarray) -> np.ndarray:
    if flat.shape[1] % 4 == 0:
        return flat.view("<i4")
    return flat


def byte_row_digests_host(flat: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
    """Digest of uint8 byte rows ``[rows, bytes]`` (C-contiguous) on the host."""

    return digest_words_host(_byte_rows_to_words_host(flat), extra)


def byte_row_digests_device(flat: Any, extra: Any | None = None) -> Any:
    """Digest of uint8 byte rows ``[rows, bytes]`` in torch, in row blocks.

    ``flat`` must be contiguous.  Blocks bound the int64 temporaries to
    about ``_DIGEST_DEVICE_BLOCK_BYTES`` per call (the widened words and one
    product buffer), so the digest never reserves chunk-sized memory.
    """

    framework = _require_torch()
    rows = int(flat.shape[0])
    width = int(flat.shape[1]) if flat.ndim == 2 else 0
    if rows == 0:
        return framework.empty((0,), dtype=framework.int64, device=flat.device)
    word_count = width // 4 if width % 4 == 0 else width
    block_rows = max(1, _DIGEST_DEVICE_BLOCK_BYTES // max(1, 16 * word_count))
    outputs = []
    for start in range(0, rows, block_rows):
        stop = min(rows, start + block_rows)
        block = flat[start:stop]
        if width % 4 == 0:
            words = block.view(framework.int32)
        else:
            words = block.to(framework.int64)
        outputs.append(
            digest_words_device(words, None if extra is None else extra[start:stop])
        )
        del words
    return outputs[0] if len(outputs) == 1 else framework.cat(outputs)


def row_digests_host(images: Any) -> np.ndarray | None:
    """Per-row digest of a uint8 image batch on the host (numpy int64).

    Accepts a numpy array or a CPU torch tensor of shape [rows, ...].
    Returns None for non-uint8 inputs (the exploratory float fallback
    carries no bank bytes to certify).
    """

    if images is None:
        return None
    if not isinstance(images, np.ndarray):
        if not hasattr(images, "numpy"):
            return None
        framework = _require_torch()
        if images.dtype != framework.uint8:
            return None
        images = images.detach().contiguous().numpy()
    if images.dtype != np.uint8 or images.ndim < 1:
        return None
    rows = int(images.shape[0])
    if rows == 0:
        return np.empty((0,), dtype=np.int64)
    flat = np.ascontiguousarray(images).reshape(rows, -1)
    return byte_row_digests_host(flat)


def row_digests_device(images: Any) -> Any:
    """Digest the delivered uint8 rows on their producing stream."""
    return _lane_g.digest_rows_fused_device(images)


class _StagedChunkPayload(tuple):
    """The serve path's 4-tuple payload with integrity stamps attached.

    Unpacks exactly as ``(context_ids, images, labels, loaded_bytes)`` for
    every existing consumer; the extra attributes carry the per-row image
    digests (I1) and the payload-scoped staged checksums (I2).  Produced
    only while the guard is armed; disarmed, ``_gather_bank_chunk`` returns
    the plain tuple it always returned.
    """

    row_digests: np.ndarray | None
    image_checksum: int | None
    label_checksum: int | None

    def __new__(
        cls,
        context_ids: tuple[str, ...],
        images: Any,
        labels: Any,
        loaded_bytes: int,
        *,
        row_digests: np.ndarray | None = None,
        image_checksum: int | None = None,
        label_checksum: int | None = None,
    ) -> "_StagedChunkPayload":
        payload = super().__new__(cls, (context_ids, images, labels, int(loaded_bytes)))
        payload.row_digests = row_digests
        payload.image_checksum = None if image_checksum is None else int(image_checksum)
        payload.label_checksum = None if label_checksum is None else int(label_checksum)
        return payload


def _payload_row_digests(payload: Any) -> np.ndarray | None:
    return getattr(payload, "row_digests", None)


def _row_digest_defect(value: Any, rows: int) -> str | None:
    """Why ``value`` cannot certify ``rows`` row digests; None when it can.

    A digest array must hold exactly one int64 per row.  Comparing a
    shorter array with the reference broadcasts (an empty array against one
    row gives an empty result whose ``all()`` is true), so shape and dtype
    are checked before any comparison.  Reads metadata only (no sync).
    """

    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None or dtype is None:
        return f"not an array ({type(value).__name__})"
    count = 1
    for extent in tuple(shape):
        count *= int(extent)
    if count != int(rows):
        return f"{count} digest(s) for {int(rows)} row(s)"
    if str(dtype) not in ("int64", "torch.int64"):
        return f"dtype {dtype}, expected int64"
    return None


def _row_digest_pair_defects(pair: Sequence[Any]) -> dict[str, str]:
    """Malformed parts of one queued digest pair, keyed by stage name.

    ``reference`` is the authority the stages compare with; ``after_copy``
    is the copy-stream digest; ``first_use``/``after_use`` are checked only
    when recorded (a missing probe is handled by the missing-use rule).
    """

    rows = len(pair[1])
    defects: dict[str, str] = {}
    for key, value in (
        ("reference", pair[2]),
        ("after_copy", pair[3]),
        ("first_use", pair[5].get("first_use")),
        ("after_use", pair[5].get("after_use")),
    ):
        if value is None and key in ("first_use", "after_use"):
            continue
        defect = _row_digest_defect(value, rows)
        if defect is not None:
            defects[key] = defect
    # With a gather digest present, the bank table's digests are the
    # separate authority for the gather stage; they need the same check.
    state = pair[5]
    if (state.get("authority") == _lane_g.AUTHORITY_TABLE
            and state.get("gather_digest_present")):
        defect = _row_digest_defect(state.get("table_expected"), rows)
        if defect is not None:
            defects["table_expected"] = defect
    return defects


def _only_missing_use_probes(error: BaseException) -> bool:
    """True when a settlement error reports only missing use probes."""

    if not isinstance(error, RowDigestIntegrityError):
        return False
    details = getattr(error, "details", None) or {}
    chunks = tuple(details.get("chunks", ()) or ())
    return (bool(chunks) and not details.get("checksum_failures")
            and all(item.get("reason") == "required_probe_missing" for item in chunks))


def _payload_stamps(payload: Any) -> tuple[int | None, int | None] | None:
    image_checksum = getattr(payload, "image_checksum", None)
    label_checksum = getattr(payload, "label_checksum", None)
    if image_checksum is None and label_checksum is None:
        return None
    return (image_checksum, label_checksum)

# One fixed-width gather scheduler serves live panel demand and lookahead.
# Lower numeric values run first; an executing gather is never preempted.
_GATHER_PRIORITY_DEMAND = 0
_GATHER_PRIORITY_EXACT = 1
_GATHER_PRIORITY_SPECULATIVE = 2
_GATHER_SCHEDULING_IDENTITY = "demand0_exact1_speculative2_reserved_demand_capacity_v2"


class _PriorityGatherExecutor:
    """Minimal fixed-width priority executor for physical row gathering.

    ``PooledBackgroundIterator`` submits live serve work through the normal
    ``submit`` signature, which maps to priority 0.  Lookahead uses explicit
    exact/speculative priorities.  FIFO is retained *within* a priority, and
    results remain ordered by ``PooledBackgroundIterator`` rather than by
    worker completion.  Thus this changes only which independent gather gets
    a worker first; it cannot change panel composition or delivery order.
    """

    def __init__(self, *, max_workers: int, thread_name_prefix: str) -> None:
        self._max_workers = max(1, int(max_workers))
        self._thread_name_prefix = str(thread_name_prefix)
        self._queue: queue.PriorityQueue[tuple[int, int, Any]] = queue.PriorityQueue()
        self._lock = threading.Lock()
        self._sequence = 0
        self._shutdown = False
        self._submitted = [0, 0, 0]
        self._started = [0, 0, 0]
        self._completed = [0, 0, 0]
        self._cancelled_before_start = 0
        self._queue_high_water = 0
        # Priority cannot preempt a gather already inside NumPy/mmap. Preserve
        # one worker of capacity from speculative saturation so newly arriving
        # live demand or exact work always has a runnable lane. Single-worker
        # configurations stay fully serial.
        self._speculative_concurrency_limit = self._max_workers - 1 if self._max_workers > 1 else 1
        self._speculative_slots = threading.BoundedSemaphore(self._speculative_concurrency_limit)
        self._speculative_active = 0
        self._speculative_active_high_water = 0
        self._speculative_slot_deferrals = 0
        self._schedule_changed = threading.Event()
        self._threads = tuple(
            threading.Thread(
                target=self._worker,
                name=f"{self._thread_name_prefix}_{index}",
                daemon=True,
            )
            for index in range(self._max_workers)
        )
        for thread in self._threads:
            thread.start()

    def submit(
        self,
        fn: Any,
        /,
        *args: Any,
        priority: int = _GATHER_PRIORITY_DEMAND,
        **kwargs: Any,
    ) -> Future:
        resolved_priority = int(priority)
        if resolved_priority not in {
            _GATHER_PRIORITY_DEMAND,
            _GATHER_PRIORITY_EXACT,
            _GATHER_PRIORITY_SPECULATIVE,
        }:
            raise ValueError("unknown gather scheduling priority")
        future: Future = Future()
        with self._lock:
            if self._shutdown:
                raise RuntimeError("cannot schedule on a closed executor")
            sequence = self._sequence
            self._sequence += 1
            self._submitted[resolved_priority] += 1
            self._queue.put(
                (
                    resolved_priority,
                    sequence,
                    (future, fn, args, kwargs),
                )
            )
            self._queue_high_water = max(self._queue_high_water, self._queue.qsize())
            self._schedule_changed.set()
        return future

    def _worker(self) -> None:
        while True:
            priority, _sequence, task = self._queue.get()
            speculative_slot = False
            speculative_counted = False
            try:
                if task is None:
                    return
                future, fn, args, kwargs = task
                if future.cancelled():
                    with self._lock:
                        self._cancelled_before_start += 1
                    continue
                if int(priority) == _GATHER_PRIORITY_SPECULATIVE:
                    speculative_slot = self._speculative_slots.acquire(blocking=False)
                    if not speculative_slot:
                        # Put the unchanged FIFO identity back. The reserved
                        # worker then sleeps until a new higher-priority submit
                        # or a speculative completion changes eligibility.
                        self._queue.put((priority, _sequence, task))
                        with self._lock:
                            self._speculative_slot_deferrals += 1
                        self._schedule_changed.wait(timeout=0.01)
                        self._schedule_changed.clear()
                        continue
                if not future.set_running_or_notify_cancel():
                    with self._lock:
                        self._cancelled_before_start += 1
                    continue
                with self._lock:
                    self._started[int(priority)] += 1
                    if speculative_slot:
                        self._speculative_active += 1
                        speculative_counted = True
                        self._speculative_active_high_water = max(
                            self._speculative_active_high_water,
                            self._speculative_active,
                        )
                try:
                    result = fn(*args, **kwargs)
                except BaseException as error:
                    future.set_exception(error)
                else:
                    future.set_result(result)
                finally:
                    with self._lock:
                        self._completed[int(priority)] += 1
            finally:
                if speculative_slot:
                    if speculative_counted:
                        with self._lock:
                            self._speculative_active -= 1
                    self._speculative_slots.release()
                    self._schedule_changed.set()
                self._queue.task_done()

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        with self._lock:
            if self._shutdown:
                threads = self._threads
            else:
                self._shutdown = True
                if cancel_futures:
                    retained: list[tuple[int, int, Any]] = []
                    while True:
                        try:
                            queued = self._queue.get_nowait()
                        except queue.Empty:
                            break
                        self._queue.task_done()
                        if queued[2] is None:
                            retained.append(queued)
                        else:
                            if queued[2][0].cancel():
                                self._cancelled_before_start += 1
                    for queued in retained:
                        self._queue.put(queued)
                for _ in self._threads:
                    sequence = self._sequence
                    self._sequence += 1
                    self._queue.put((99, sequence, None))
                self._schedule_changed.set()
                threads = self._threads
        if wait:
            for thread in threads:
                thread.join()

    def receipt(self) -> dict[str, Any]:
        with self._lock:
            return {
                "effective_settings": effective_settings(),
                "identity": _GATHER_SCHEDULING_IDENTITY,
                "workers": int(self._max_workers),
                "submitted_by_priority": {
                    "demand": int(self._submitted[_GATHER_PRIORITY_DEMAND]),
                    "exact": int(self._submitted[_GATHER_PRIORITY_EXACT]),
                    "speculative": int(self._submitted[_GATHER_PRIORITY_SPECULATIVE]),
                },
                "started_by_priority": {
                    "demand": int(self._started[_GATHER_PRIORITY_DEMAND]),
                    "exact": int(self._started[_GATHER_PRIORITY_EXACT]),
                    "speculative": int(self._started[_GATHER_PRIORITY_SPECULATIVE]),
                },
                "completed_by_priority": {
                    "demand": int(self._completed[_GATHER_PRIORITY_DEMAND]),
                    "exact": int(self._completed[_GATHER_PRIORITY_EXACT]),
                    "speculative": int(self._completed[_GATHER_PRIORITY_SPECULATIVE]),
                },
                "cancelled_before_start": int(self._cancelled_before_start),
                "queue_high_water": int(self._queue_high_water),
                "speculative_concurrency_limit": int(self._speculative_concurrency_limit),
                "speculative_reserved_capacity": int(
                    self._max_workers - self._speculative_concurrency_limit
                ),
                "speculative_active": int(self._speculative_active),
                "speculative_active_high_water": int(self._speculative_active_high_water),
                "speculative_slot_deferrals": int(self._speculative_slot_deferrals),
            }


@dataclass
class _LookaheadEntry:
    """One pre-gathered chunk awaiting its serve-path consumer.

    ``taken`` and ``stale`` are mutually exclusive terminal states, always
    assigned under the pool lock so a stale flush can never race the serve
    path into double-releasing a pinned lease or its budget permit.
    ``speculative`` marks rung-1 posts whose consumption is uncertain; a
    non-speculative re-post of the same chunk re-parents (confirms) the
    entry instead of gathering the bytes twice.
    """

    key: tuple[str, ...]
    post_id: int
    future: Any = None
    taken: bool = False
    stale: bool = False
    speculative: bool = False
    # ``origin_speculative`` is immutable provenance; ``speculative`` is
    # the live scheduling class and becomes false when exact demand confirms
    # any row in the chunk.  ``demand_confirmed`` then outranks ordinary
    # exact supply in the row locator multimap.
    origin_speculative: bool = False
    # Optional provider-owned prediction lineage.  A generation is opaque to
    # the plane; safety comes from matching BOTH this identity and the
    # physical role on a later exact post that explicitly declares the
    # prediction completely resolved.  Legacy speculative posts leave both
    # fields unset and retain consumption-keyed lifetime.
    prediction_generation_id: str | None = None
    prediction_role: str | None = None
    demand_confirmed: bool = False
    parallel_shards: int = 1
    reserved_bytes: int = 0
    # Row-level salvage: chunk-relative row indices already delivered
    # to re-chunked serve chunks.  The payload tensors are immutable, so a
    # later exact take of a partially-salvaged entry is still correct; the
    # entry retires (lease + budget returned) once every row is consumed.
    consumed_rows: set[int] | None = None
    # Rows proved outside a generation's complete realized exact look.  Their
    # locators are removed immediately, but their bytes remain physically
    # resident until every still-live confirmed row has settled.  Keeping this
    # distinct from ``consumed_rows`` preserves delivery telemetry semantics.
    generation_retired_rows: set[int] | None = None
    claimed_rows: set[int] | None = None
    serving_claims: int = 0
    # A zero-copy resident slice transfers a VIEW of this entry's pinned
    # payload instead of assembling a second host tensor.  The entry cannot
    # retire while such a view is queued for H2D: its permit and backing
    # payload remain owned until the copy stream has recorded the source
    # slot's ready event.
    delivery_holds: int = 0
    # Most recently issued H2D event for a borrowed slice.  All slice copies
    # use the plane's one ordered copy stream, so the newest event dominates
    # earlier ones.  It is published to the pinned-ring owner only when the
    # entry finally retires; publishing it after a partial slice would make
    # still-resident unconsumed rows incorrectly reusable.
    delivery_event: Any = None
    retire_pending: bool = False
    registered: bool = False
    permit_release_scheduled: bool = False
    resident_bytes: int = 0


@dataclass(frozen=True)
class _RowLocator:
    """Metadata-only locator for one immutable bank row.

    The locator points into an entry's shared gathered tensor.  It never
    copies an image, so shifted chunk boundaries change only index metadata.
    """

    entry: _LookaheadEntry
    key_row: int


@dataclass(frozen=True)
class _PinSplitPayload:
    """A gathered chunk whose target-stratum rows live on device (rung 3).

    ``host_images``/``host_labels`` hold ONLY the non-pinned rows, in the
    chunk-relative order given by ``host_positions``; ``pin_positions`` and
    ``pin_slots`` address the remaining rows inside the device pin bank.
    Device batch assembly interleaves both sets back into the EXACT
    original chunk order (the bitwise scrutiny point).
    """

    context_ids: tuple[str, ...]
    host_images: Any
    host_labels: Any
    host_positions: tuple[int, ...]
    pin_positions: tuple[int, ...]
    pin_slots: tuple[int, ...]
    loaded_bytes: int
    # CLASS3: integrity stamps of the HOST part, in compact host order
    # (None when the guard is disarmed or the rows were assembled by a
    # path that carries no authority).  Pinned rows are certified against
    # the pin bank's own digests (``_pin_row_digests_host``).
    host_row_digests: Any = None
    host_image_checksum: int | None = None
    host_label_checksum: int | None = None


@dataclass
class _ResidentPinnedSlicePayload:
    """One ordered view into a completed resident pinned payload.

    ``host_images`` and ``host_labels`` alias ``owner_*`` storage.  The
    source entry carries one ``delivery_holds`` reference until device
    staging records (or synchronously aborts) the H2D.  Only the settlement
    method may detach the entry and return its residency permit.
    """

    context_ids: tuple[str, ...]
    host_images: Any
    host_labels: Any
    loaded_bytes: int
    entry: _LookaheadEntry
    pool: Any
    source_payload: Any
    owner_images: Any | None
    owner_labels: Any | None
    image_checksum: int | None
    label_checksum: int | None
    pin_split: _PinSplitPayload | None = None
    settled: bool = False
    # The owner's per-row image digests restricted to the served
    # interval (chunk order), or None when the owner carried none.
    row_digests: Any = None


class _PersistentPanelProducerPool:
    """Producer executors and lookahead state owned by one data plane.

    The gather executor replaces the per-look ``PooledBackgroundIterator``
    pools (spin-up amortized to once per arm); the single-worker stage
    executor replaces the per-look device-staging pool.  Lookahead
    residency is bounded by ``lookahead_budget_limit`` pre-gathered chunks
    — the serve pipeline's own in-flight bound — so combined pinned-ring
    pressure stays within the ring's existing lease/growth envelope
    (``staged_leased_until_h2d_event_complete_v2``); past the budget a post
    stops enqueuing (receipted) and the remaining chunks stay on the
    on-demand path.  Every counter is receipted in the data-plane runtime
    block.
    """

    def __init__(
        self, *, workers: int, depth: int, lookahead_looks: int,
        resident_chunk_cap: int | None = None,
        resident_byte_limit: int | None = None,
    ) -> None:
        self.workers = max(1, int(workers))
        self.depth = max(1, int(depth))
        self.lookahead_looks = max(0, int(lookahead_looks))
        self.gather_executor = _PriorityGatherExecutor(
            max_workers=self.workers,
            thread_name_prefix="panel-producer-gather",
        )
        self.stage_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="panel-producer-stage",
        )
        import os as _os

        budget_raw = _os.environ.get(_PANEL_LOOKAHEAD_CHUNK_BUDGET_ENV, "").strip()
        if budget_raw:
            self.lookahead_budget_limit = max(1, int(budget_raw))
            self.lookahead_budget_source = "env"
        else:
            self.lookahead_budget_limit = (self.workers + self.depth) * 3
            self.lookahead_budget_source = "derived_workers_plus_depth_x3"
        if resident_chunk_cap is not None:
            capped = min(
                self.lookahead_budget_limit, max(1, int(resident_chunk_cap))
            )
            if capped < self.lookahead_budget_limit:
                self.lookahead_budget_source += "_bounded_by_pinned_bytes"
            self.lookahead_budget_limit = capped
        self._budget = threading.Semaphore(self.lookahead_budget_limit)
        self.resident_byte_limit = resident_byte_limit
        self.reserved_bytes_current = 0
        self.reserved_bytes_high_water = 0
        self.budget_permits_acquired = 0
        self.budget_permits_released = 0
        self.budget_permits_current = 0
        self.budget_permits_high_water = 0
        self.reuse_count = 0
        self.enqueued = 0
        self.ready_hits = 0
        self.on_demand_misses = 0
        self.exposed_wait_seconds = 0.0
        self.posted_look_count = 0
        self.stale_flushed_chunks = 0
        self.budget_exhausted_posts = 0
        self.deduplicated_chunks = 0
        # Rung 1 accounting: speculative chunks enqueued / served / retired
        # unconsumed / confirmed by a later exact (non-speculative) post.
        self.speculative_posted = 0
        self.speculative_consumed = 0
        self.speculative_flushed = 0
        self.speculative_confirmed = 0
        # Generation-bound speculative reconciliation.  These counters are
        # intentionally separate from ``speculative_flushed``: the latter is
        # the whole-entry discard counter, while reconciliation
        # may retire only some locators and retain one shared backing payload.
        self.prediction_generations_reconciled = 0
        self.prediction_generation_entries_reconciled = 0
        self.prediction_generation_entries_retired = 0
        self.prediction_generation_rows_retired = 0
        self.prediction_generation_partial_entries = 0
        self.prediction_generation_deferred_by_claims_or_holds = 0
        self.prediction_generation_unmatched_noops = 0
        self.prediction_generation_mismatched_entries_untouched = 0
        self.prediction_generation_role_mismatch_refusals = 0
        self.prediction_generation_invalid_identity_refusals = 0
        self.prediction_generation_invalid_resolution_mode_refusals = 0
        # Row-level salvage + mismatch diagnostic accounting.
        self.salvaged_hits = 0
        self.salvaged_rows = 0
        self.speculative_salvaged = 0
        # Partial coverage + bounded on-demand top-up.
        self.salvaged_partial_hits = 0
        self.topup_rows = 0
        # Coverage-gated top-up — a mostly-missing chunk declines
        # salvage (the plain sharded on-demand gather is cheaper than a
        # sequential top-up of most of the chunk; measured).
        self.salvage_declined_low_coverage = 0
        # Consumption-keyed lifetime: entries live until SERVED and
        # are evicted only when a NEW post needs their budget slot (LRU by
        # post order).  A post-count TTL discards 90-95% of gathered bytes
        # unconsumed (measured) — data starvation from the buffer's own
        # aging heuristic, not pressure (the entry-count semaphore alone
        # bounds live chunks; actual host bytes are measured separately in
        # the receipt).
        self.demand_evicted_chunks = 0
        self.speculative_evicted_for_demand = 0
        self.speculative_budget_refusals_preserving_exact = 0
        # Demand-priority and row-addressed accounting.
        self.demand_promotions = 0
        self.demand_promoted_rows = 0
        self.queued_speculative_promoted = 0
        self.queued_lookahead_cancelled_for_demand = 0
        self.demand_ready_rows = 0
        self.demand_waited_rows = 0
        self.row_index_requests = 0
        self.row_index_hits = 0
        self.row_index_misses = 0
        self.row_index_declined_rows = 0
        self.row_index_assembly_failed_rows = 0
        self.pin_row_locator_refusals = 0
        self.pin_row_locator_hits = 0
        self.pin_row_locator_topup_rows = 0
        # CUDA row salvage used to assemble every shifted chunk first in a
        # fresh pageable tensor and then copy that entire tensor into a
        # pinned delivery slot.  Direct-to-pinned assembly removes that
        # redundant host copy while retaining the same lease/checksum path.
        self.direct_pinned_assembly_chunks = 0
        self.direct_pinned_assembly_rows = 0
        self.pageable_to_pinned_copy_elided_bytes = 0
        # A requested chunk that is an ordered contiguous slice of one
        # resident pinned entry can travel as a tensor view.  This removes
        # even the direct-pinned index_select/index_copy assembly pass.
        self.resident_pinned_slice_chunks = 0
        self.resident_pinned_slice_rows = 0
        self.resident_pinned_slice_host_copy_elided_bytes = 0
        self.resident_pinned_slice_pin_split_chunks = 0
        self.resident_pinned_slice_host_rows = 0
        self.resident_pinned_slice_pin_rows = 0
        self.resident_entries_current = 0
        self.resident_entries_high_water = 0
        self.resident_row_locators_current = 0
        self.resident_row_locators_high_water = 0
        self.resident_bytes_current = 0
        self.resident_bytes_high_water = 0
        # Miss classification (the mismatch diagnostic): no posted
        # rows overlapped the served chunk at all / every served row was
        # resident but salvage was unavailable (still gathering, or
        # a pin-split payload) / only part of the served rows were posted.
        self.miss_never_posted = 0
        self.miss_covered_salvage_unavailable = 0
        self.miss_partial_overlap = 0
        self.mismatch_samples: list[dict[str, Any]] = []
        self.degradation_reasons: list[str] = []
        self.closed = False
        self.lock = threading.Lock()
        self.entries: dict[tuple[str, ...], list[_LookaheadEntry]] = {}
        self.row_index: dict[str, list[_RowLocator]] = {}
        # Consumption-keyed lifetime: post-ordered live-entry map (LRU
        # eviction order).  Entries leave on take, salvage-exhaustion,
        # demand eviction, or shutdown — never on a later post's arrival.
        self.live_order: dict[int, _LookaheadEntry] = {}
        self.post_counter = 0

    def record_degradation(self, reason: str) -> None:
        with self.lock:
            if len(self.degradation_reasons) < (_LOOKAHEAD_DEGRADATION_REASON_CAP):
                self.degradation_reasons.append(str(reason))

    def try_reserve_budget(self) -> bool:
        acquired = self._budget.acquire(blocking=False)
        if acquired:
            with self.lock:
                self.budget_permits_acquired += 1
                self.budget_permits_current += 1
                self.budget_permits_high_water = max(
                    self.budget_permits_high_water,
                    self.budget_permits_current,
                )
        return bool(acquired)

    def release_budget(self) -> None:
        with self.lock:
            if self.budget_permits_current <= 0:
                raise RuntimeError("lookahead budget permit double release")
            self.budget_permits_released += 1
            self.budget_permits_current -= 1
        self._budget.release()


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("native panel data plane requires PyTorch")
    return torch


@dataclass(frozen=True)
class NativePanelExecutionSpec:
    """Runtime-only physical execution policy for one process."""

    scoring_batch_rows: int = 32
    calibration_batch_rows: int = 512
    calibration_oracle_fused_batch_rows: int = 512
    v0_paired_oracle_batch_rows: int = 256
    v1_candidate_oracle_batch_rows: int = 512
    terminal_batch_rows: int = 512
    prefetch_depth: int = 2
    producer_workers: int = 1
    device_cache_bytes: int = 2 << 30
    require_optimized_bank: bool = False
    unit_packing_degree: int = 1
    query_tile_size: int = 1
    working_set_prefetch: bool = False

    def __post_init__(self) -> None:
        for name in (
            "scoring_batch_rows",
            "calibration_batch_rows",
            "calibration_oracle_fused_batch_rows",
            "v0_paired_oracle_batch_rows",
            "v1_candidate_oracle_batch_rows",
            "terminal_batch_rows",
            "prefetch_depth",
            "producer_workers",
            "unit_packing_degree",
            "query_tile_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.device_cache_bytes) < 0:
            raise ValueError("device_cache_bytes cannot be negative")

    def batch_rows(self, role: str) -> int:
        physical_role = _physical_native_panel_role(role)
        mapping = {
            "scoring": self.scoring_batch_rows,
            "calibration": self.calibration_batch_rows,
            "calibration_oracle_fused": (self.calibration_oracle_fused_batch_rows),
            "v0_paired_oracle": self.v0_paired_oracle_batch_rows,
            "v1_candidate_oracle": self.v1_candidate_oracle_batch_rows,
            "terminal": self.terminal_batch_rows,
        }
        return int(mapping[physical_role])

    def to_runtime_dict(self) -> dict[str, Any]:
        return {
            "schema_name": "mechanistic_native_panel_execution_spec_v2",
            "effective_settings": effective_settings(),
            "scoring_batch_rows": int(self.scoring_batch_rows),
            "calibration_batch_rows": int(self.calibration_batch_rows),
            "calibration_oracle_fused_batch_rows": int(self.calibration_oracle_fused_batch_rows),
            "v0_paired_oracle_batch_rows": int(self.v0_paired_oracle_batch_rows),
            "v1_candidate_oracle_batch_rows": int(self.v1_candidate_oracle_batch_rows),
            "terminal_batch_rows": int(self.terminal_batch_rows),
            "prefetch_depth": int(self.prefetch_depth),
            "producer_workers": int(self.producer_workers),
            "device_cache_bytes": int(self.device_cache_bytes),
            "require_optimized_bank": bool(self.require_optimized_bank),
            "unit_packing_degree": int(self.unit_packing_degree),
            "query_tile_size": int(self.query_tile_size),
            "working_set_prefetch": bool(self.working_set_prefetch),
            "scientific_identity": "excluded",
        }


@dataclass
class _RoleCounters:
    calls: int = 0
    requested_physical_contexts: int = 0
    delivered_physical_contexts: int = 0
    input_cache_hits: int = 0
    input_cache_misses: int = 0
    bank_read_bytes: int = 0
    host_to_device_bytes: int = 0
    producer_wait_seconds: float = 0.0
    panel_wall_seconds: float = 0.0
    first_batch_latency_seconds: float = 0.0
    batch_count: int = 0
    device_cache_lookup_seconds: float = 0.0
    device_cache_insert_seconds: float = 0.0
    h2d_enqueue_seconds: float = 0.0
    normalization_enqueue_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            key: (float(value) if isinstance(value, float) else int(value))
            for key, value in self.__dict__.items()
        }


@dataclass(frozen=True)
class NativePanelReceipt:
    role: str
    requested_context_count: int
    delivered_context_count: int
    batch_rows: int
    batch_count: int
    delivered_batch_sizes: tuple[int, ...]
    cache_hits: int
    cache_misses: int
    bank_read_bytes: int
    host_to_device_bytes: int
    producer_wait_seconds: float
    host_gather_activity_seconds: float
    device_stage_activity_seconds: float
    device_copy_normalize_seconds: float
    device_cache_lookup_seconds: float
    device_cache_insert_seconds: float
    h2d_enqueue_seconds: float
    normalization_enqueue_seconds: float
    consumer_compute_controller_seconds: float
    phase_accounting_fraction: float
    first_batch_latency_seconds: float
    elapsed_seconds: float
    panel_hash: str
    optimized_bank: bool
    source_id: str
    schema_name: str = NATIVE_PANEL_RECEIPT_SCHEMA
    measured_consumer_activity_seconds: float = 0.0
    unattributed_consumer_seconds: float = 0.0
    bank_gather_bytes_per_second: float | None = None
    host_to_device_bytes_per_second: float | None = None
    # Set when the delivered row or label identity detector fails.
    integrity_failed: bool = False
    device_memory_breakdown: Mapping[str, Any] | None = None
    gpu_process_count: int | None = None
    device_gather: Mapping[str, Any] | None = None
    settings: Mapping[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = self._to_dict_base()
        if self.settings is not None:
            payload["effective_settings"] = dict(self.settings)
        payload["integrity_failed"] = bool(self.integrity_failed)
        if self.device_memory_breakdown is not None:
            payload["device_memory_breakdown"] = dict(self.device_memory_breakdown)
        if self.gpu_process_count is not None:
            payload["gpu_process_count"] = int(self.gpu_process_count)
        if self.device_gather is not None:
            payload["device_gather"] = dict(self.device_gather)
        return payload

    def _to_dict_base(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "effective_settings": effective_settings(),
            "role": self.role,
            "requested_context_count": int(self.requested_context_count),
            "delivered_context_count": int(self.delivered_context_count),
            "batch_rows": int(self.batch_rows),
            "batch_count": int(self.batch_count),
            "delivered_batch_sizes": tuple(int(value) for value in self.delivered_batch_sizes),
            "tail_policy": "fixed_occupancy_batches_then_single_tail_v2",
            "cache_hits": int(self.cache_hits),
            "cache_misses": int(self.cache_misses),
            "bank_read_bytes": int(self.bank_read_bytes),
            "host_to_device_bytes": int(self.host_to_device_bytes),
            "producer_wait_seconds": float(self.producer_wait_seconds),
            "host_gather_activity_seconds": float(self.host_gather_activity_seconds),
            "device_stage_activity_seconds": float(self.device_stage_activity_seconds),
            "device_copy_normalize_seconds": float(self.device_copy_normalize_seconds),
            "device_cache_lookup_seconds": float(self.device_cache_lookup_seconds),
            "device_cache_insert_seconds": float(self.device_cache_insert_seconds),
            "h2d_enqueue_seconds": float(self.h2d_enqueue_seconds),
            "normalization_enqueue_seconds": float(self.normalization_enqueue_seconds),
            "consumer_compute_controller_seconds": float(self.consumer_compute_controller_seconds),
            "phase_accounting_fraction": float(self.phase_accounting_fraction),
            "measured_consumer_activity_seconds": float(self.measured_consumer_activity_seconds),
            "unattributed_consumer_seconds": float(self.unattributed_consumer_seconds),
            "bank_gather_bytes_per_second": (
                None
                if self.bank_gather_bytes_per_second is None
                else float(self.bank_gather_bytes_per_second)
            ),
            "host_to_device_bytes_per_second": (
                None
                if self.host_to_device_bytes_per_second is None
                else float(self.host_to_device_bytes_per_second)
            ),
            "uncovered_pipeline_wait_fraction": (
                0.0
                if self.elapsed_seconds <= 0.0
                else float(self.producer_wait_seconds) / float(self.elapsed_seconds)
            ),
            "first_batch_latency_seconds": float(self.first_batch_latency_seconds),
            "elapsed_seconds": float(self.elapsed_seconds),
            "panel_hash": self.panel_hash,
            "optimized_bank": bool(self.optimized_bank),
            "source_id": self.source_id,
        }


@dataclass
class _DeviceGatherRun:
    """One choice across optimization and evaluation planes of one run."""

    planes: tuple[Any, ...]
    failure: str | None = None
    delivery_identity_error: DeliveryIntegrityError | None = None

    def record_identity_error(self, error: DeliveryIntegrityError) -> None:
        self.delivery_identity_error = error
        for plane in self.planes:
            if plane._device_gather is not None:
                plane._device_gather.delivery_identity_error = error
                plane._device_gather.failure_details = dict(getattr(error, "details", {}) or {})

    def record_failure(self, reason: str, details: Mapping[str, Any] | None = None) -> None:
        self.failure = reason
        for plane in self.planes:
            if plane._device_gather is not None:
                plane._device_gather.failure = reason
                if details is not None:
                    plane._device_gather.failure_details = dict(details)

    def fail(self, reason: str) -> None:
        self.record_failure(reason)
        raise DeviceGatherError(f"device delivery failed: {reason}")


def bind_device_gather_run(planes: Sequence[Any]) -> None:
    """Freeze one delivery choice before any member plane serves a row."""

    if not any(getattr(plane, "_device_gather_requested", False) for plane in planes):
        return
    members = tuple(dict.fromkeys(planes))
    if not members or not any(plane._device_gather_requested for plane in members):
        return
    if any(plane._receipts or (plane._device_gather is not None and plane._device_gather.delivered_rows)
           for plane in members):
        raise DeviceGatherError("run_binding_requires_unconsumed_planes")
    run = _DeviceGatherRun(members)
    for plane in members:
        plane._device_gather_run = run
    unavailable = tuple("device_gather_disabled" for plane in members if plane._device_gather is None)
    if unavailable:
        run.fail("run_member_preflight_failed:" + ";".join(unavailable))


def preflight_resident_image_sets(plans: Sequence[tuple[Any, Mapping[str, Any]]]) -> dict[str, int]:
    """Refuse worker populations that exceed their process share before a walk.

    Each population is a context-id sequence or a row count. Counts are useful
    for the full validation bank, whose row order is already fixed. Repeated
    references to a worker bank/name are charged once.
    """
    by_device: dict[str, dict[str, Any]] = {}
    seen: dict[tuple[int, str, str], int] = {}
    planned: dict[tuple[int, str, str], frozenset[int]] = {}
    for plane, populations in plans:
        gather = plane._device_gather
        if gather is None or not gather.worker_residency:
            continue
        device = str(gather.device)
        entry = by_device.setdefault(device, {"gather": gather, "required": 0})
        for name, rows in populations.items():
            if isinstance(rows, int):
                count = int(rows)
                if count < 1 or count > int(gather.bank.shape[0]):
                    raise ValueError("resident image population row count is outside the bank")
            else:
                ordinals = set(int(plane._records[str(value)].ordinal) for value in rows)
                count = len(ordinals)
                if count < 1:
                    raise ValueError("resident image population is empty")
            key = (id(gather.registration), device, str(name))
            planned_rows = frozenset(range(count) if isinstance(rows, int) else ordinals)
            if key in planned and not planned_rows <= planned[key]:
                raise DeviceGatherError("resident_image_population_changed")
            planned.setdefault(key, planned_rows)
            resident = gather.registration.resident.get((device, str(name)))
            if resident is not None:
                if any(row not in resident.slots for row in planned_rows):
                    raise DeviceGatherError("resident_image_population_changed")
            required = 0 if resident is not None else count * (gather.row_bytes + 16)
            previous = seen.get(key, 0)
            seen[key] = max(previous, required)
            entry["required"] += max(0, required - previous)
    result = {}
    for device, entry in by_device.items():
        gather, required = entry["gather"], entry["required"]
        gather._admit_resident_bytes(required)
        result[device] = required
    return result


class NativePanelDataPlane:
    """Ordered, streaming physical-context delivery for native execution."""

    source_id = "native-streaming-panel-data-plane-v1"

    def __init__(
        self,
        source: Any,
        *,
        spec: NativePanelExecutionSpec,
        device: Any,
        model_state_hash: str = "",
    ) -> None:
        framework = _require_torch()
        self.source = source
        self.optimized_bank = isinstance(source, PreprocessedUint8BankContextSource)
        self.spec = spec
        self.device = framework.device(device)
        self._process_device_memory = configure_process_device_memory(self.device)
        self.model_state_hash = str(model_state_hash)
        self._image_gather_enabled = os.environ.get(_IMAGE_GATHER_ENV, "0") == "1"
        self._bank_page_advice_enabled = os.environ.get(_BANK_PAGE_ADVICE_ENV, "0") == "1"
        self._bank_inputs = getattr(source, "_inputs", None)
        gather_default = "1" if self.optimized_bank and self.device.type == "cuda" else "0"
        self._device_gather_requested = os.environ.get(DEVICE_GATHER_ENV, gather_default) == "1"
        self._device_gather: DeviceImageGather | None = None
        self._device_gather_run: _DeviceGatherRun | None = None
        self._populated_bank_mapping: mmap.mmap | None = None
        self._bank_page_advice: dict[str, str] = {}
        if self.optimized_bank and self._bank_page_advice_enabled:
            if bool(getattr(source, "page_warm_performed", False)):
                self._bank_page_advice["mapping"] = "already_warmed_by_CIRCUIT_NVME_PAGE_WARM"
            else:
                self._prepare_bank_mapping()
        if spec.require_optimized_bank and not self.optimized_bank:
            raise ValueError(
                "performance-gated native execution requires a certified preprocessed uint8 bank"
            )
        self.axis = getattr(source, "axis", None)
        if self.axis is None:
            raise ValueError("native panel source must expose its ContextAxis")
        self._records = {str(record.context_id): record for record in self.axis.records}
        self._preprocessing_hash = self.axis.preprocessing.content_hash
        self._input_identity_hash = stable_digest(
            {
                "schema_name": "native_panel_input_identity_v1",
                "axis_hash": self.axis.content_hash,
                "preprocessing_hash": self._preprocessing_hash,
                "model_state_hash": self.model_state_hash,
            }
        )
        self._decode_scale, self._decode_bias = (
            self.axis.preprocessing.uint8_online_decode_parameters(
                num_channels=(
                    int(source._inputs.shape[1])  # noqa: SLF001 - typed bank adapter
                    if self.optimized_bank
                    else 3
                )
            )
        )
        self._cache: OrderedDict[tuple[str, str], int] = OrderedDict()
        self._cache_bytes = 0
        self._device_cache_pages: Any | None = None
        self._device_cache_free_slots: list[int] = []
        self._device_cache_row_bytes = 0
        self._workspace_rings: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        self._workspace_cursors: dict[tuple[Any, ...], int] = {}
        self._cache_lock = threading.RLock()
        # Preserve semantic summaries for scientific audit compatibility and
        # independently aggregate the physical execution buckets.  These must
        # not alias: a semantic role whose physical role has the same spelling
        # contributes once to each intentionally distinct summary.
        self._role_counters = {
            role: _RoleCounters() for role in _SEMANTIC_ROLE_NAMES
        }
        self._physical_role_counters = {
            role: _RoleCounters() for role in _ROLE_NAMES
        }
        self._receipts: list[NativePanelReceipt] = []
        self._receipt_lock = threading.RLock()
        self._copy_stream = (
            framework.cuda.Stream(device=self.device, priority=0)
            if self.device.type == "cuda" and framework.cuda.is_available()
            else None
        )
        self._decode_scale_device: Any | None = None
        self._decode_bias_device: Any | None = None
        self._delivery_local = threading.local()
        # Bank rows are already gathered by a persistent outer producer pool.
        # A new ThreadPoolExecutor for every physical chunk strands large
        # numpy advanced-indexing buffers in short-lived glibc thread
        # arenas, producing an observed ~300 MiB RSS ratchet per 2,000-row
        # panel.  Reuse one process-local shard pool so those arenas and their
        # buffers are reused across the whole arm.  Only worker lifetime
        # differs; row shards, concatenation order, and delivered bytes are
        # identical.
        self._bank_shard_executor = ThreadPoolExecutor(
            max_workers=max(1, int(self.spec.producer_workers)),
            thread_name_prefix="panel-bank-shard",
        )
        self._bank_shard_executor_uses = 0
        self._direct_gather_chunks = 0
        self._direct_gather_bytes = 0
        pool_raw = str(os.environ.get(_PINNED_POOL_MAX_BYTES_ENV, _PINNED_POOL_DEFAULT_BYTES)).strip()
        pool_bytes = self._pinned_working_set_bytes() if pool_raw == "auto" else max(0, int(pool_raw or "0"))
        pool_device = self._worker_pool_device(pool_bytes)
        self._bind_pinned_pool(
            _worker_pinned_pool(pool_device, pool_bytes, auto=pool_raw == "auto")
            if pool_device is not None else _PinnedHostPool(pool_bytes)
        )
        self._delivery_checksum_pairs: list[tuple[int, Any, str]] = []
        self._staging_integrity_errors: list[BaseException] = []
        self._delivery_checksums_verified = 0
        self._delivery_checksum_mismatches = 0
        self._integrity_scope_depth = 0
        self._integrity_scope_drained = True
        self._integrity_scope_copy_timings: list[Any] = []
        self._pending_copy_timings: list[Any] = []
        self._image_feed_copy_timings: list[Any] = []
        self._prefetch_executor: Any | None = None
        self._prefetch_lock = threading.Lock()
        self._prefetch_rows_submitted = 0
        self._prefetch_panels = 0
        # IRG_TERMINAL_ADAPTIVE_BATCH: optional per-role adaptive batch
        # policy (``mechanistic_systems.perf_gates.BatchPolicy``).  ``None``
        # (the default, and always the case when the env gate is unset) means
        # every role resolves to the exact ``NativePanelExecutionSpec``
        # constants — the OFF state is structurally today's behavior.
        self._batch_policy: Any | None = None
        self._batch_policy_applied_rows: dict[str, int] = {}
        self._batch_policy_source_role: dict[str, str] = {}
        # IRG_PERSISTENT_PANEL_PRODUCER: with the master gate disarmed
        # every field below stays inert (``_persistent_producer_pool``
        # returns None before touching any of them) and the serve path is
        # structurally the per-look pipeline.  Default ON;
        # explicit falsy tokens are the kill switch.
        self._persistent_producer_enabled = (
            os.environ.get(_PERSISTENT_PANEL_PRODUCER_ENV, "").strip().lower()
            not in _FALSE_ENVIRONMENT_TOKENS
        )
        lookahead_raw = os.environ.get(_PANEL_LOOKAHEAD_LOOKS_ENV, "").strip()
        self._deep_image_lookahead = os.environ.get(_DEEP_IMAGE_LOOKAHEAD_ENV, "0") == "1"
        if not self._persistent_producer_enabled:
            self._lookahead_looks = 0
        elif not lookahead_raw:
            self._lookahead_looks = 1
        else:
            self._lookahead_looks = max(0, int(lookahead_raw))
        self._persistent_pool: _PersistentPanelProducerPool | None = None
        self._persistent_pool_lock = threading.Lock()
        self._persistent_pool_closed = False
        self._persistent_producer_degradations: list[str] = []
        # Speculative extension prefetch: a provider-side hook
        # that rides the ready-buffer; it remains inert whenever the
        # persistent-producer master gate is disarmed. OPT-IN, because
        # almost all decisions finish before this next increment is consumed.
        self._speculative_extension_env = (
            os.environ.get(_SPECULATIVE_EXTENSION_PREFETCH_ENV, "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        # On-target device pin (default ON): built lazily on
        # install. The receipt names its enabled state and accumulated work.
        # CPU devices retain host serving.
        self._on_target_pin_env = (
            os.environ.get(_ON_TARGET_DEVICE_PIN_ENV, "").strip().lower()
            not in _FALSE_ENVIRONMENT_TOKENS
        )
        # IRG_ROW_SALVAGE: OPT-IN (default OFF), unlike the seven
        # harmonized default-ON gates: no canary has shown the salvage
        # path firing live on GPU.
        self._row_salvage_env = os.environ.get("IRG_ROW_SALVAGE", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        # IRG_PANEL_NORMALIZE_BUFFER_REUSE: bounded per-shape
        # normalize buffer rings.  OFF: every field inert; allocation
        # behavior is exactly that of fresh allocations.
        self._normalize_buffer_reuse = os.environ.get(_PANEL_NORMALIZE_BUFFER_REUSE_ENV, "") == "1"
        self._normalize_rings: dict[tuple, list[dict[str, Any]]] = {}
        self._normalize_ring_cursors: dict[tuple, int] = {}
        self._normalize_ring_lock = threading.Lock()
        self._normalize_ring_hits = 0
        self._normalize_ring_buffers = 0
        self._normalize_ring_bytes = 0
        # The evaluator binds the existing folded math and all
        # compatible delivery consumers in one transaction before serving.
        # Default OFF: at AlexNet conv1 size the
        # folded output differs from normalize-then-conv by up to 3e-5
        # (about 4e-6 of the output range); IRG_UINT8_FOLDED_STEM=1 opts in
        # and becomes the default only after the pilot's GPU A/B shows the
        # decisions agree.  Raw delivery is flipped ONLY by the atomic bind
        # transaction (torch_evaluator.bind_uint8_folded_stem), never by the
        # env alone.
        self._uint8_folded_stem_env = str(
            os.environ.get(_UINT8_FOLDED_STEM_ENV, "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._raw_panel_delivery = False
        self._uint8_folded_stem_binding: dict[str, Any] | None = None
        self._pin_lock = threading.Lock()
        self._pin_target_class_id: int | None = None
        self._pin_build_attempted = False
        self._pin_built = False
        self._pin_images: Any | None = None
        self._pin_labels: Any | None = None
        self._pin_slot_by_context: dict[str, int] = {}
        self._pin_bytes = 0
        self._pin_hit_rows = 0
        self._pin_gathered_rows = 0
        self._pin_degradations: list[str] = []
        # CLASS3 delivery-integrity guard (IRG_READOUT_LABEL_INTEGRITY,
        # default armed; disabled detectors still report their idle state).
        self._delivery_integrity_armed = delivery_integrity_enabled()
        self._pin_row_digests_host: np.ndarray | None = None
        # (chunk_index, context_ids, expected, observed_after_copy, label,
        #  use_state{"first_use", "after_use"}) — settled at panel end.
        self._row_digest_pairs: list[tuple[int, tuple[str, ...], np.ndarray, Any, str, dict]] = []
        self._row_digest_chunks_checked = 0
        self._row_digest_rows_checked = 0
        self._row_digest_mismatches = 0
        self._row_digest_mismatches_by_stage: dict[str, int] = {
            ROW_DIGEST_STAGE_GATHER: 0,
            ROW_DIGEST_STAGE_COPY: 0,
            ROW_DIGEST_STAGE_USE: 0,
            ROW_DIGEST_STAGE_AFTER_USE: 0,
        }
        self._row_digest_first_use_probes = 0
        self._row_digest_after_use_probes = 0
        self._row_digest_gather_witness_rows = 0
        # CLASS3: bank-label checks run by label consumers
        # outside ``_readouts`` on panels this plane served. The serving
        # plane owns these counters separately from provider readout work.
        self._consumer_label_checks: dict[str, dict[str, int]] = {}
        self._row_digest_rows_uncovered = 0
        self._row_digest_mismatch_samples: list[dict[str, Any]] = []
        self._payload_stamps_issued = 0
        if self._device_gather_requested:
            self._device_gather = self._initialize_device_gather()
        if (self._pinned_pool.max_bytes and self.optimized_bank
                and bool(getattr(source, "pin_memory", False))
                and self._device_gather is None
                and self.device.type == "cuda" and framework.cuda.is_available()):
            self._warm_pinned_pool()

    def plan_resident_populations(self, populations: Mapping[str, Any]) -> int:
        """Check this bank's planned residency; use the group helper for two banks."""
        return sum(preflight_resident_image_sets(((self, populations),)).values())

    def _initialize_device_gather(self) -> DeviceImageGather:
        """Register the worker's image feed, without implementation comparisons."""

        framework = _require_torch()
        if not self.optimized_bank:
            raise DeviceGatherError("requires_optimized_bank")
        if self.device.type != "cuda":
            raise DeviceGatherError("requires_cuda_device")
        if not framework.cuda.is_available():
            raise DeviceGatherError("cuda_unavailable")
        bank = self.source._inputs
        labels = np.asarray(self.source._labels)
        if labels.shape != (int(bank.shape[0]),) or labels.dtype.kind not in "iu":
            raise DeviceGatherError("bank_label_axis_check_failed")
        if any(int(record.ordinal) < 0 or int(record.ordinal) >= len(bank)
               for record in self._records.values()):
            raise DeviceGatherError("row_ordinal_out_of_range")
        with framework.cuda.device(self.device):
            if self.device.index is None:
                self.device = framework.device("cuda", framework.cuda.current_device())
            free, total = framework.cuda.mem_get_info(self.device)
            chunk_rows = min(len(bank), max(self.spec.batch_rows(role) for role in _ROLE_NAMES))
            budget = gather_budget_bytes(
                int(free),
                process_device_memory_budget_bytes(total),
                framework.cuda.memory_allocated(self.device),
                reusable_bytes=cuda_reusable_memory_bytes(
                    self.device, framework=framework, free_bytes=int(free),
                ),
                minimum_bytes=int(labels.size) * 8 + 2 * chunk_rows * (int(bank[0].nbytes) + 16),
            )
            registration = register_readonly_bank(bank, self.device, framework)
            gather = DeviceImageGather(
                registration, labels, framework, self.device, self._copy_stream,
                budget, chunk_rows, process_memory_budget_bytes=process_device_memory_budget_bytes(total),
            )
            return gather

    def _device_gather_receipt(self) -> dict[str, Any] | None:
        if not self._device_gather_requested:
            return None
        members = (self,) if self._device_gather_run is None else self._device_gather_run.planes
        active = tuple(plane._device_gather for plane in members if plane._device_gather is not None)
        memory = {
            "byte_budget_scope": "data_plane",
            "run_max_bytes": sum(gather.budget for gather in active),
            "run_current_reserved_bytes": sum(
                gather.reserved_bytes for gather in active if not gather.closed
            ),
        }
        if self._device_gather is not None:
            return {
                "requested": True, "selection_scope": "whole_run",
                "run_plane_count": 1 if self._device_gather_run is None else len(self._device_gather_run.planes),
                **memory,
                **self._device_gather.receipt(),
                "effective_settings": self._effective_delivery_settings(),
            }
        raise DeviceGatherError("requested device delivery is unavailable")

    def _device_gather_failure(self, error: Exception) -> None:
        if isinstance(error, (DeliveryIntegrityError, _lane_g.BankRowDigestTableError)):
            if self._device_gather_run is not None:
                self._device_gather_run.record_identity_error(error)
            elif self._device_gather is not None:
                self._device_gather.delivery_identity_error = error
                self._device_gather.failure_details = dict(getattr(error, "details", {}) or {})
            raise error
        reason = f"{type(error).__name__}:{error}"
        if self._device_gather is not None and self._device_gather.worker_residency:
            self._device_gather.failure = reason
            raise error
        if self._device_gather_run is not None:
            self._device_gather_run.record_failure(reason, getattr(error, "details", None))
        elif self._device_gather is not None:
            self._device_gather.failure = reason
            self._device_gather.failure_details = getattr(error, "details", None)
        raise DeviceGatherError(f"device delivery failed: {reason}") from error

    def _device_gather_error(self, reason: str) -> DeviceGatherError:
        return DeviceGatherError(f"device delivery failed: {reason}")

    def install_device_row_order(
        self, key: str, context_ids: Sequence[str], *, start: int = 0,
    ) -> None:
        """Post a fixed physical order; counts and draw registration stay with the provider."""

        self.install_device_row_reader(
            key, len(context_ids), lambda a, b: context_ids[a:b], start=start,
        )

    def install_device_row_reader(
        self, key: str, length: int, read: Any, *, start: int = 0, prefetch: bool = True,
    ) -> None:
        gather = self._device_gather
        if gather is None:
            return
        if gather.worker_residency:
            # The walk planner posts the actual next look through request().
            # A stream's fixed order is not its adaptive look's row list.
            return
        try:
            def ordinals(a: int, b: int) -> tuple[int, ...]:
                ids = read(a, b)
                return tuple(int(self._records[str(value)].ordinal) for value in ids)
            gather.planner.install_reader(key, length, ordinals, start=start)
            if prefetch:
                gather.prefetch()
        except Exception as error:
            self._device_gather_failure(error)

    def prefetch_device_rows(self) -> None:
        if self._device_gather is not None and not self._device_gather.worker_residency:
            try:
                self._device_gather.prefetch()
            except Exception as error:
                self._device_gather_failure(error)

    def request(self, rows: Sequence[str | int] | Any, *, ready_event: Any = None) -> Any:
        """Queue the next look's physical rows; return its delivery event.

        The walk planner supplies its original draws ahead. Strings are
        physical context IDs; integers are source-bank row ordinals. No draw
        generation, outcome prediction or image work runs on this host call.
        """

        gather = self._device_gather
        if gather is None:
            raise DeviceGatherError("image_request_requires_device_feed")
        if _require_torch().is_tensor(rows):
            return gather.request(rows, ready_event=ready_event)
        ordinals = tuple(
            int(self._records[row].ordinal) if isinstance(row, str) else int(row)
            for row in rows
        )
        return gather.request(ordinals, ready_event=ready_event)

    def _device_gather_batch(self, context_ids: tuple[str, ...]) -> tuple[Any, ...]:
        gather = self._device_gather
        if gather is None:
            raise DeviceGatherError("requested device delivery is unavailable")
        if gather.delivery_identity_error is not None:
            raise gather.delivery_identity_error
        if gather.failure is not None:
            raise self._device_gather_error(gather.failure)
        framework = _require_torch()
        ordinals = tuple(int(self._records[value].ordinal) for value in context_ids)
        started = time.perf_counter()
        rows_before = gather.gathered_rows
        checksum_cursor = len(self._delivery_checksum_pairs)
        probe = None
        try:
            with framework.cuda.stream(self._copy_stream):
                copy_start = copy_end = None
                if not self._device_look_delivery_active():
                    copy_start = framework.cuda.Event(enable_timing=True)
                    copy_end = framework.cuda.Event(enable_timing=True)
                    copy_start.record(self._copy_stream)
                resident_name = getattr(self._delivery_local, "resident_name", None)
                if gather.worker_residency and resident_name is not None:
                    images, labels, missing = gather.gather_resident(resident_name, ordinals)
                else:
                    images, labels, missing = gather.gather(ordinals)
                image_rows, label_rows = int(images.shape[0]), int(labels.shape[0])
                if image_rows != len(context_ids) or label_rows != len(context_ids):
                    raise DeviceGatherError(
                        f"device_gather_row_count_mismatch:expected={len(context_ids)};"
                        f"images={image_rows};labels={label_rows}"
                    )
                if not gather.worker_residency:
                    probe = self._queue_row_digest_check(context_ids, None, images, "device_gather")
                    if probe is None and not self._device_look_delivery_active():
                        raise DeviceGatherError("device_gather_requires_use_probes")
                binding = self._uint8_folded_stem_binding
                bound_raw = (self._raw_panel_delivery and binding is not None
                             and binding.get("raw_input_dtype") == "uint8"
                             and getattr(self._delivery_local, "role", None) != "scoring")
                inputs = images if bound_raw else framework.empty_like(
                    images, dtype=framework.float32, memory_format=framework.channels_last,
                )
                normalization_started = time.perf_counter()
                inputs = self._delivery_inputs(probe, images, inputs)
                normalize_seconds = time.perf_counter() - normalization_started
                # Labels are small. Keep their independent source stamp as on
                # the host path; images use every delivered byte on device.
                if not gather.worker_residency:
                    expected_labels = framework.as_tensor(np.array(self.source._labels[list(ordinals)], copy=True))
                    self._queue_delivery_check(
                        expected_labels, labels, "device_gather_labels",
                        expected_checksum=self._host_delivery_checksum(expected_labels),
                    )
                if copy_end is not None:
                    copy_end.record(self._copy_stream)
                ready = framework.cuda.Event()
                ready.record(self._copy_stream)
                if not gather.worker_residency or gather.planner.orders:
                    gather.prefetch()
        except Exception:
            if probe is not None:
                probe.discard("device_gather_stage_not_delivered")
            with self._pinned_ring_lock:
                del self._delivery_checksum_pairs[checksum_cursor:]
            raise
        batch = MeasurementInputBatch(
            context_ids, inputs, labels, input_identity_hash=self._input_identity_hash,
            ready_event=ready, delivery_probe=probe, stream_guarded_tensors=(images,),
        )
        loaded = (gather.gathered_rows - rows_before) * gather.row_bytes
        return (
            batch, len(context_ids) - missing, missing, loaded, loaded, 0.0,
            time.perf_counter() - started, copy_start, copy_end,
            {"device_cache_lookup_seconds": 0.0, "device_cache_insert_seconds": 0.0,
             "h2d_enqueue_seconds": 0.0, "normalization_enqueue_seconds": normalize_seconds},
        )

    def _worker_pool_device(self, pool_bytes: int) -> Any | None:
        """The CUDA device whose worker pool this plane shares, else None."""
        framework = _require_torch()
        if not (pool_bytes and self.device.type == "cuda" and framework.cuda.is_available()):
            return None
        if self.device.index is None:
            return framework.device("cuda", framework.cuda.current_device())
        return self.device

    def _bind_pinned_pool(self, pool: _PinnedHostPool) -> None:
        self._pinned_pool = pool
        self._pinned_host_rings = pool.rings
        self._pinned_ring_cursors = pool.cursors
        self._pinned_slot_ids = pool.ids
        self._pinned_slot_storage = pool.storage
        self._pinned_slot_state = pool.state
        self._pinned_ring_lock = pool.lock

    @property
    def _pinned_pool_max_bytes(self) -> int:
        return self._pinned_pool.max_bytes

    @property
    def _pinned_pool_reserved_bytes(self) -> int:
        return self._pinned_pool.reserved_bytes

    @property
    def _pinned_ring_event_syncs(self) -> int:
        return self._pinned_pool.event_syncs

    @property
    def _pinned_ring_grown_slots(self) -> int:
        return self._pinned_pool.grown_slots

    @property
    def _pinned_ring_overflow_allocations(self) -> int:
        return self._pinned_pool.overflow_allocations

    def _pinned_size_class(self, shape: Sequence[int], dtype: Any) -> int:
        elements = int(np.prod(shape))
        size = elements * _require_torch().empty((), dtype=dtype).element_size()
        # The two production image chunks deserve exact capacities: rounding
        # 38.5/77.1 MB to 64/128 MiB wastes most of the look-ahead headroom.
        if len(shape) == 4 and int(shape[0]) in (256, 512) and dtype == torch.uint8:
            return size
        return 1 << (size - 1).bit_length()

    def _pinned_working_set_bytes(self) -> int:
        pipeline = int(self.spec.producer_workers) + int(self.spec.prefetch_depth)
        permits = max(1, int(os.environ.get(
            _PANEL_LOOKAHEAD_CHUNK_BUDGET_ENV, pipeline * 3
        )))
        rows = max(self.spec.batch_rows(role) for role in _ROLE_NAMES)
        shape = (rows, *self._bank_inputs.shape[1:]) if self.optimized_bank else (rows, 3, 224, 224)
        chunk_bytes = self._pinned_size_class(shape, torch.uint8)
        label_bytes = self._pinned_size_class((rows,), torch.int64)
        # Existing resident permits + pipeline + branch headroom. This is
        # setup storage, shared by all roles in this plane, not a new limit
        # on the existing chunk semaphore.
        required = (permits + pipeline + 2 + int(self.spec.producer_workers)) * (chunk_bytes + label_bytes)
        return max(6 * 1024**3, required)

    def _warm_pinned_pool(self) -> None:
        """Allocate and touch reusable image/label storage before producers run."""
        with self._pinned_pool.warm_lock:
            rows = max(self.spec.batch_rows(role) for role in _ROLE_NAMES)
            image_shape = (rows, *self._bank_inputs.shape[1:])
            capacity = self._pinned_size_class(image_shape, torch.uint8)
            if self._pinned_pool.warmed_capacity >= capacity:
                return
            per_pair = (self._pinned_size_class(image_shape, torch.uint8)
                        + self._pinned_size_class((rows,), torch.int64))
            count = self._pinned_pool.max_bytes // per_pair
            leases = []
            try:
                for _ in range(count):
                    for shape, dtype in ((image_shape, torch.uint8), ((rows,), torch.int64)):
                        slot = self._pinned_host_slot(shape=shape, dtype=dtype, allow_transient=False)
                        if slot is None:
                            return
                        leases.append(slot)
            finally:
                for slot in leases:
                    self._release_lookahead_payload(((), slot, None, 0))

            self._pinned_pool.warmed_capacity = capacity

    def _pinned_host_slot(
        self, reference: Any | None = None, *, shape: Sequence[int] | None = None,
        dtype: Any | None = None, memory_format: Any = None,
        allow_transient: bool = True,
    ) -> Any:
        """Lease an exact-ring slot (OFF) or a byte-bounded pooled view (ON).

        A queued slot stays leased until its copy event completes. Idle larger
        slots can serve smaller requests; idle smaller classes are reclaimed
        when needed. Newly allocated storage is touched before its first gather.
        """
        framework = _require_torch()
        pinned = self.device.type == "cuda" and framework.cuda.is_available()
        layout = "channels_last" if memory_format is framework.channels_last else "contiguous"
        if reference is None:
            if shape is None or dtype is None:
                raise ValueError("pinned host allocation requires a reference or shape/dtype")
            slot_shape, slot_dtype = tuple(int(v) for v in shape), dtype
        else:
            if shape is not None or dtype is not None:
                raise ValueError("pinned host allocation cannot mix a reference with shape/dtype")
            slot_shape, slot_dtype = tuple(int(v) for v in reference.shape), reference.dtype
        if not slot_shape or any(v <= 0 for v in slot_shape):
            raise ValueError("pinned host allocation shape must be positive")
        needed_bytes = int(np.prod(slot_shape)) * framework.empty((), dtype=slot_dtype).element_size()
        pooled = self._pinned_pool.max_bytes > 0
        capacity = self._pinned_size_class(slot_shape, slot_dtype) if pooled else needed_bytes
        key = capacity if pooled else (slot_shape, slot_dtype, layout)
        depth = max(1, int(self.spec.prefetch_depth)) + max(1, int(self.spec.producer_workers)) + 2
        if layout == "channels_last" and len(slot_shape) == 4:
            n, c, h, w = slot_shape
            required_strides = (h * w * c, 1, w * c, c)
        else:
            strides = [1] * len(slot_shape)
            for index in range(len(slot_shape) - 2, -1, -1):
                strides[index] = strides[index + 1] * slot_shape[index + 1]
            required_strides = tuple(strides)

        def view(storage):
            return storage.view(slot_dtype).as_strided(slot_shape, required_strides)

        def idle(slot):
            state = self._pinned_slot_state[id(slot)]
            event = state.get("event")
            return not state.get("leased") or (event is not None and bool(event.query()))

        def evict(ring_key, index):
            slot = self._pinned_host_rings[ring_key].pop(index)
            storage = self._pinned_slot_storage.pop(id(slot))
            self._pinned_slot_ids.remove(id(slot))
            self._pinned_slot_state.pop(id(slot))
            self._pinned_pool.reserved_bytes -= int(storage.numel())
            self._pinned_pool.evictions += 1
            if not self._pinned_host_rings[ring_key]:
                del self._pinned_host_rings[ring_key]
                self._pinned_ring_cursors.pop(ring_key, None)

        blocking_event = None
        with self._pinned_ring_lock:
            chosen, waiting = None, None
            keys = sorted(self._pinned_host_rings) if pooled else (key,)
            for ring_key in keys:
                if pooled and ring_key < needed_bytes:
                    continue
                ring = self._pinned_host_rings.get(ring_key, [])
                if not ring:
                    continue
                start = self._pinned_ring_cursors.get(ring_key, 0) % len(ring)
                for offset in range(len(ring)):
                    index = (start + offset) % len(ring)
                    candidate = ring[index]
                    if idle(candidate):
                        chosen = (ring_key, index, candidate)
                        break
                    event = self._pinned_slot_state[id(candidate)].get("event")
                    if waiting is None and event is not None:
                        waiting = (ring_key, index, candidate, event)
                if chosen is not None:
                    break
            if pooled and chosen is None and self._pinned_pool.reserved_bytes + capacity > self._pinned_pool.max_bytes:
                for ring_key in tuple(self._pinned_host_rings):
                    if ring_key >= needed_bytes:
                        continue
                    ring = self._pinned_host_rings[ring_key]
                    for index in range(len(ring) - 1, -1, -1):
                        if idle(ring[index]):
                            evict(ring_key, index)
                        if self._pinned_pool.reserved_bytes + capacity <= self._pinned_pool.max_bytes:
                            break
                    if self._pinned_pool.reserved_bytes + capacity <= self._pinned_pool.max_bytes:
                        break
            ring = self._pinned_host_rings.get(key, [])
            can_grow = (self._pinned_pool.reserved_bytes + capacity <= self._pinned_pool.max_bytes
                        if pooled else len(ring) < depth + 4)
            if chosen is None and can_grow:
                storage = framework.empty(capacity, dtype=framework.uint8, pin_memory=pinned)
                storage.zero_()
                self._pinned_pool.pretouched_bytes += capacity
                slot = view(storage)
                if not ring:
                    self._pinned_host_rings[key] = ring
                if len(ring) >= depth:
                    self._pinned_pool.grown_slots += 1
                ring.append(slot)
                self._pinned_slot_ids.add(id(slot))
                self._pinned_slot_storage[id(slot)] = storage
                self._pinned_slot_state[id(slot)] = {"leased": True, "event": None, "staged_checksum": None}
                self._pinned_pool.reserved_bytes += capacity
                self._pinned_pool.allocations += 1
                return slot
            if chosen is not None:
                ring_key, index, slot = chosen
            elif waiting is not None:
                ring_key, index, slot, blocking_event = waiting
            else:
                if not allow_transient:
                    return None
                self._pinned_pool.overflow_allocations += 1
                # OFF retains the original pinned transient; ON never exceeds
                # its live pinned-byte bound and records any pageable fallback.
                kwargs = {"dtype": slot_dtype, "pin_memory": pinned and not pooled}
                if memory_format is not None:
                    kwargs["memory_format"] = memory_format
                slot = framework.empty(slot_shape, **kwargs)
                slot.zero_()
                self._pinned_pool.pretouched_bytes += needed_bytes
                if pinned and not pooled:
                    self._pinned_pool.unpooled_allocations += 1
                else:
                    self._pinned_pool.pageable_allocations += 1
                return slot
            ring = self._pinned_host_rings[ring_key]
            self._pinned_ring_cursors[ring_key] = (index + 1) % len(ring)
            if (tuple(slot.shape) != slot_shape or slot.dtype != slot_dtype
                    or tuple(slot.stride()) != required_strides):
                storage = self._pinned_slot_storage.pop(id(slot))
                self._pinned_slot_ids.remove(id(slot))
                self._pinned_slot_state.pop(id(slot))
                slot = view(storage)
                ring[index] = slot
                self._pinned_slot_ids.add(id(slot))
                self._pinned_slot_storage[id(slot)] = storage
            self._pinned_slot_state[id(slot)] = {"leased": True, "event": None, "staged_checksum": None}
            self._pinned_pool.reuses += 1
        if blocking_event is not None:
            blocking_event.synchronize()
            with self._pinned_ring_lock:
                self._pinned_pool.event_syncs += 1
        return slot

    def _delivery_checksum_flat(self, tensor: Any) -> Any:
        """Position-weighted strided int64 sum in the staged NHWC byte order.

        The same expression runs on the pinned host slot (at staging and at
        consumption) and on the device tensor the H2D produced, so any
        mutation of the slot between staging and the asynchronous copy is
        observable as a sum mismatch. Integer arithmetic keeps the value
        exactly comparable across host and device; position weights make
        the sum sensitive to reorderings and multiset-equal rewrites, not
        just totals (a plain sum over class labels is exactly invariant
        under permutation — the class-major terminal panel's natural
        collision). Honest coverage bounds: the stride (8191, prime)
        samples ~0.01% of bytes, so a cross-chunk overwrite is caught
        near-certainly per event (weighted-sum collision odds are far below
        1e-4) and practically certainly across a run, while a mutation
        confined to bytes strictly between sample points is structurally
        invisible; the device-cache hit path carries no checksum coverage
        (pre-existing surface). The receipt names this contract explicitly.
        """

        framework = _require_torch()
        if (
            self._image_gather_enabled
            and tensor.dim() == 4
            and tensor.is_contiguous()
            and not tensor.is_contiguous(memory_format=framework.channels_last)
        ):
            # The old staged slot is NHWC.  Direct gather keeps the final
            # host and uint8 device slots in CHW storage for fast `take`;
            # address the SAME NHWC sample positions without materializing
            # a transposed copy.  I2's weights, stride and values stay fixed.
            rows, channels, height, width = map(int, tensor.shape)
            count = rows * channels * height * width
            stride = _DELIVERY_CHECKSUM_STRIDE if count > _DELIVERY_CHECKSUM_STRIDE else 1
            position = framework.arange(0, count, stride, device=tensor.device)
            row = position // (height * width * channels)
            pixel = (position // channels) % (height * width)
            channel = position % channels
            sample = tensor[row, channel, pixel // width, pixel % width].to(
                dtype=framework.int64
            )
            weights = framework.arange(
                1, int(sample.numel()) + 1,
                dtype=framework.int64, device=sample.device,
            )
            return (sample * weights).sum()
        if tensor.dim() == 4 and tensor.is_contiguous(memory_format=framework.channels_last):
            flat = tensor.permute(0, 2, 3, 1).reshape(-1)
        elif tensor.is_contiguous():
            flat = tensor.reshape(-1)
        else:  # pragma: no cover - staging always produces contiguous slots
            flat = tensor.contiguous().reshape(-1)
        stride = _DELIVERY_CHECKSUM_STRIDE if int(flat.numel()) > _DELIVERY_CHECKSUM_STRIDE else 1
        sample = flat[::stride].to(dtype=framework.int64)
        weights = framework.arange(
            1,
            int(sample.numel()) + 1,
            dtype=framework.int64,
            device=sample.device,
        )
        return (sample * weights).sum()

    def _host_delivery_checksum(self, tensor: Any) -> int:
        return int(self._delivery_checksum_flat(tensor).item())

    def _device_look_delivery_active(self) -> bool:
        """Carry the consumer's look mode to its staging thread."""
        from mechanistic_systems.inference.device_bce import DEVICE_LOOK_ACTIVE

        panel = getattr(getattr(self, "_delivery_local", None), "panel", None)
        return bool(self.device.type == "cuda" and (DEVICE_LOOK_ACTIVE.get() or
                    (panel is not None and not panel.get("closed", False)
                     and panel.get("device_look", False))))

    def _record_staged_checksum(self, tensor: Any) -> int | None:
        """Stamp the just-staged slot's byte checksum onto its lease.

        Returns the stamp (CLASS3 I2: the caller also carries it on the
        payload) or None for a tensor that is not a pooled pinned slot.
        """

        if id(tensor) not in self._pinned_slot_ids:
            return None
        value = self._host_delivery_checksum(tensor)
        # A re-viewed slot gets a new tensor identity. The old view still
        # carries its lease's stamp even when only the lease check is active.
        tensor._native_panel_staged_checksum = int(value)
        with self._pinned_ring_lock:
            state = self._pinned_slot_state.get(id(tensor))
            if state is not None:
                state["staged_checksum"] = value
        return int(value)


    def _queue_delivery_check(
        self,
        host_tensor: Any,
        device_tensor: Any,
        label: str,
        *,
        expected_checksum: int | None = None,
    ) -> None:
        """Enqueue a device-side checksum of a full-slot H2D for later audit.

        Must be called inside the copy-stream context right after the copy
        is enqueued. The sum observes the delivered bytes; its disagreement
        flag settles after the decision or at the group tail.
        """

        if self._device_look_delivery_active() and "labels" not in str(label):
            return
        expected = expected_checksum
        if expected is None:
            with self._pinned_ring_lock:
                state = self._pinned_slot_state.get(id(host_tensor))
                expected = None if state is None else state.get("staged_checksum")
        if expected is None:
            return
        device_sum = self._delivery_checksum_flat(device_tensor)
        with self._pinned_ring_lock:
            self._delivery_checksum_pairs.append((int(expected), device_sum, str(label)))

    def _queue_row_digest_check(
        self,
        context_ids: tuple[str, ...],
        expected: Any,
        uint8_batch: Any,
        label: str,
    ) -> Any | None:
        """CLASS3 I1: digest the assembled device rows for a lazy audit.

        Called inside the copy-stream context after the uint8 batch is
        fully assembled (H2D rows, pin rows and cache hits alike), so the
        kernels read exactly the delivered bytes.  Nothing synchronizes
        here; the pair settles in ``_verify_delivery_checksums`` at panel
        end.  ``expected`` is the host authority (numpy int64, chunk order)
        or None when the path carried none (counted as uncovered rows).

        Returns the use probe to install on the delivered batch: a callable
        the consumer runs on ITS stream twice -- from ``wait_until_ready``
        (stage ``first_use``, right before the forward) and from
        ``mark_consumed`` (stage ``after_use``, after the forward and the
        label clone were enqueued and before the completion event releases
        the slot).  The copy-stream digest certifies what the H2D
        delivered; first-use and after-use equal to the authority certify
        that nothing rewrote the slot between the H2D audit and the end of
        the forward's reads, apart from an exact
        restore in between (ABA), which no digest can see.

        The delivered rows retain the device-side identity checks:

        * AUTHORITY.  FULL requires a bank row digest table bound
          (``bank_row_digest_table.table_authority_for``), the authority is
          the table's entries for the rows' bank ordinals: the same v2
          function over the certified bank bytes.  If the gather still
          computes its host digest, the device stages compare with that
          row identity and the table covers every delivered row
          at settle (the gather stage at full coverage, replacing the
          sampled bank re-read), so every delivered row is certified
          against the table.  If the gather computes none,
          the device stages compare with the table directly and a gather
          fault is reported at the after-copy stage.  Sampled mode may use
          the explicitly receipted gather-time fallback.
        * MODE (G4, ``IRG_READOUT_GUARD_MODE``): ``full`` (default) probes
          every chunk; ``sampled`` keeps the after-copy digest on every row
          and runs the use probes on a seeded 1-in-16 sample of chunks;
          ``disarmed`` does no row-digest work (rows receipted as
          disarmed, never as uncovered).
        * G3.  The returned probe exposes ``wants_first_use()`` and
          ``record_fused_first_use(digest, kind)``: a normalize that read the
          bytes itself (``normalize_with_first_use_digest``) records the
          first-use digest, and the consumer-stream first-use call is then a
          no-op.  The after-use probe is unchanged.

        """

        if self._device_look_delivery_active():
            guard = _lane_g.row_digest_guard(self)
            with guard.lock:
                guard.rows_disarmed += len(context_ids)
            return None
        if not self._delivery_integrity_armed:
            return None
        guard = _lane_g.row_digest_guard(self)
        rows = len(context_ids)
        if guard.mode == _lane_g.GUARD_MODE_DISARMED:
            with guard.lock:
                guard.rows_disarmed += rows
            return None
        witness = None if expected is None else np.asarray(expected, dtype=np.int64).reshape(-1)
        if witness is not None and int(witness.shape[0]) != rows:
            witness = None
        table = _lane_g.table_authority_for(self)
        table_expected = None
        if table is not None:
            # Ordinals derived here from the ids, independently of the gather.
            table_expected = _lane_g.authority_digests_for(self, table, context_ids)
            authority_kind = _lane_g.AUTHORITY_TABLE
            # The device stages compare with the gather's digest when the
            # gather computed one (stage for stage); that
            # digest is itself certified against the table on every row at
            # settle.  Without one, the table is the device reference.
            reference = witness if witness is not None else table_expected
        elif witness is not None:
            authority_kind = _lane_g.AUTHORITY_GATHER
            reference = witness
        else:
            with self._pinned_ring_lock:
                self._row_digest_rows_uncovered += rows
            return None
        observed = row_digests_device(uint8_batch)
        with guard.lock:
            chunk_ordinal = guard.chunks_queued
            guard.chunks_queued += 1
            guard.rows_queued += rows
            use_probes = guard.mode == _lane_g.GUARD_MODE_FULL or (
                guard.mode == _lane_g.GUARD_MODE_SAMPLED
                and _lane_g.sampled_use_probe_selected(chunk_ordinal)
            )
            if use_probes:
                guard.use_probe_chunks_sampled += 1
            else:
                guard.use_probe_chunks_skipped += 1
            if authority_kind == _lane_g.AUTHORITY_TABLE:
                guard.rows_table_authority += rows
            else:
                guard.rows_gather_authority += rows
        v6_state: dict[str, Any] = {
            "first_use": None,
            "after_use": None,
            "authority": authority_kind,
            "table_expected": table_expected,
            "gather_digest_present": witness is not None,
            "use_probes": bool(use_probes),
            "discarded": False,
            "consumer_used": False,
            "lifecycle": "queued",
            "lifecycle_lock": threading.Lock(),
            # The panel that staged this chunk; it records whether that
            # panel was drained, so a deferred scope settles each chunk
            # with its own panel's flag.
            "panel": getattr(self._delivery_local, "panel", None),
        }
        with self._pinned_ring_lock:
            chunk_index = len(self._row_digest_pairs)
            self._row_digest_pairs.append(
                (chunk_index, tuple(context_ids), reference, observed, str(label), v6_state)
            )
            v6_state["lifecycle"] = "ready"
        def discard(reason: str) -> None:
            if not reason or not str(reason).strip():
                raise ValueError("guard discard requires a reason")
            with v6_state["lifecycle_lock"]:
                if v6_state["discarded"]:
                    raise RuntimeError("guard batch already discarded")
                if v6_state["lifecycle"] not in ("queued", "ready") or v6_state["consumer_used"]:
                    raise RuntimeError("used or settled guard batch cannot be discarded")
                v6_state["discarded"] = True
                v6_state["discard_reason"] = reason
                with guard.lock:
                    source = v6_state.get("first_use_source")
                    if source == "fused":
                        guard.first_use_fused -= 1
                    elif source is not None:
                        guard.first_use_emulated -= 1
                    guard.rows_discarded += rows
                    guard.chunks_discarded += 1
                    guard.discards.append({
                        "chunk_index": chunk_index, "label": str(label),
                        "rows": rows, "reason": reason,
                    })
                    if authority_kind == _lane_g.AUTHORITY_TABLE:
                        guard.rows_table_authority -= rows
                    else:
                        guard.rows_gather_authority -= rows

        def advance_use(stage: str) -> None:
            if v6_state["discarded"]:
                raise RuntimeError("discarded guard batch cannot be used")
            if v6_state["lifecycle"] == "settled":
                raise RuntimeError("settled guard batch cannot be used")
            if stage == "after_use" and v6_state["lifecycle"] not in ("first-use", "after-use"):
                raise RuntimeError("after-use requires consumer first-use")
            if stage != "after_use" and v6_state["lifecycle"] == "after-use":
                raise RuntimeError("first-use cannot follow after-use")
            v6_state["consumer_used"] = True
            if stage == "after_use":
                v6_state["lifecycle"] = "after-use"
            elif v6_state["lifecycle"] == "ready":
                v6_state["lifecycle"] = "first-use"

        if not use_probes:
            # A sampled or inspection batch still needs an explicit release
            # when its consumer replaces it before delivery.
            def release_only(_device: Any, stage: str = "first_use") -> None:
                if stage not in ("first_use", "after_use"):
                    raise ValueError(f"unknown guard stage: {stage!r}")
                with v6_state["lifecycle_lock"]:
                    advance_use(stage)
            release_only.discard = discard  # type: ignore[attr-defined]
            return release_only

        def v6_probe(_device: Any, stage: str = "first_use") -> None:
            if stage not in ("first_use", "after_use"):
                raise ValueError(f"unknown guard stage: {stage!r}")
            key = "after_use" if stage == "after_use" else "first_use"
            with v6_state["lifecycle_lock"]:
                advance_use(key)
                if v6_state[key] is not None:
                    return
                v6_state[key] = row_digests_device(uint8_batch)
                with self._pinned_ring_lock:
                    if key == "after_use":
                        self._row_digest_after_use_probes += 1
                    else:
                        self._row_digest_first_use_probes += 1

        def wants_first_use() -> bool:
            with v6_state["lifecycle_lock"]:
                if v6_state["discarded"] or v6_state["lifecycle"] == "settled":
                    raise RuntimeError("inactive guard batch cannot be used")
                return v6_state["first_use"] is None

        def record_fused_first_use(digest: Any, kind: str = "fused") -> None:
            with v6_state["lifecycle_lock"]:
                if v6_state["discarded"] or v6_state["lifecycle"] == "settled":
                    raise RuntimeError("inactive guard batch cannot be used")
                if v6_state["first_use"] is not None:
                    return
                # This witness may be produced before the batch reaches a
                # consumer.  Only the consumer callback advances lifecycle.
                v6_state["first_use"] = digest
                v6_state["first_use_source"] = str(kind)
                with self._pinned_ring_lock:
                    self._row_digest_first_use_probes += 1
                with guard.lock:
                    if kind == "fused":
                        guard.first_use_fused += 1
                        guard.first_use_fused_attempts += 1
                    else:
                        guard.first_use_emulated += 1
                        guard.first_use_emulated_attempts += 1

        v6_probe.wants_first_use = wants_first_use  # type: ignore[attr-defined]
        v6_probe.record_fused_first_use = record_fused_first_use  # type: ignore[attr-defined]
        v6_probe.discard = discard  # type: ignore[attr-defined]
        return v6_probe

    @contextmanager
    def delivery_integrity_scope(self):
        """Queue delivery flags for every physical pass of one readout.

        Device values and event times settle after the decision or at export.
        """
        if (self._device_gather is not None and self._device_gather.worker_residency or not self._delivery_integrity_armed):
            yield
            return
        outer = self._integrity_scope_depth == 0
        if outer:
            self._integrity_scope_drained = True
            receipt_start = self.receipt_cursor()
        self._integrity_scope_depth += 1
        scope_error = None
        try:
            yield
        except BaseException as error:
            scope_error = error
            # A suspended panel has not yet recorded its own drained flag.
            # Its unreached probes must not replace the execution error.
            self._integrity_scope_drained = False
            raise
        finally:
            self._integrity_scope_depth -= 1
            if outer:
                try:
                    self._verify_delivery_checksums(
                        drained=self._integrity_scope_drained,
                        receipt_start=receipt_start, receipt_stop=self.receipt_cursor(),
                        on_error=scope_error is not None,
                    )
                    if scope_error is not None:
                        from mechanistic_systems.measurement.device_deferred import DEVICE_FAULTS

                        DEVICE_FAULTS.settle(block=True)
                except BaseException as integrity_error:
                    if scope_error is not None:
                        # The failed attempt is off the dispatch path. Settle
                        # recorded values before deciding which fault ended it.
                        from mechanistic_systems.measurement.device_deferred import DEVICE_FAULTS

                        try:
                            DEVICE_FAULTS.settle(block=True)
                        except BaseException as value_error:
                            integrity_error = value_error
                    if self._device_gather is not None:
                        reason = f"{type(integrity_error).__name__}:{integrity_error}"
                        if isinstance(integrity_error, DeliveryIntegrityError):
                            if self._device_gather_run is not None:
                                self._device_gather_run.record_identity_error(integrity_error)
                            else:
                                self._device_gather.delivery_identity_error = integrity_error
                                self._device_gather.failure_details = dict(integrity_error.details)
                        elif self._device_gather_run is not None:
                            self._device_gather_run.record_failure(reason)
                        else:
                            self._device_gather.failure = reason
                    with self._receipt_lock:
                        for index in range(receipt_start, len(self._receipts)):
                            self._receipts[index] = replace(
                                self._receipts[index], integrity_failed=True,
                                device_gather=self._device_gather_receipt(),
                            )
                    if scope_error is None:
                        raise
                    if _only_missing_use_probes(integrity_error):
                        scope_error.add_note(f"deferred delivery check also failed: {integrity_error!r}")
                    else:
                        integrity_error.add_note(f"readout attempt also failed: {scope_error!r}")
                        raise integrity_error
                finally:
                    # Copy times are read once their events complete
                    # (resolve_copy_timings), never by waiting here.
                    timings = self._integrity_scope_copy_timings
                    self._integrity_scope_copy_timings = []
                    if timings:
                        with self._receipt_lock:
                            self._pending_copy_timings.extend(timings)
                    if gpu_process_count() > 1:
                        self.release_buffer_rings(include_pinned=True, block=False)

    def copy_timing_events(self, start: int, stop: int) -> tuple[Any, ...]:
        """Completion events for this span's still-pending copy timings."""

        with self._receipt_lock:
            return tuple(
                events[-1][1] for index, events in self._pending_copy_timings
                if int(start) <= index < int(stop)
            )

    def resolve_copy_timings(self, *, block: bool = False, stop: int | None = None) -> None:
        """Fill the copy times of scoped panel receipts whose events completed."""

        with self._receipt_lock:
            while self._pending_copy_timings:
                index, events = self._pending_copy_timings[0]
                if stop is not None and index >= stop:
                    return
                last_event = events[-1][1]
                if not block and not last_event.query():
                    return
                if block:
                    last_event.synchronize()
                self._pending_copy_timings.pop(0)
                seconds = sum(float(start.elapsed_time(end)) / 1000.0 for start, end in events)
                receipt = self._receipts[index]
                self._receipts[index] = replace(
                    receipt, device_copy_normalize_seconds=seconds,
                    host_to_device_bytes_per_second=(receipt.host_to_device_bytes / seconds if seconds > 0 else None),
                )

    def _device_delivery_disagreement(self, pairs: Sequence[Any], digest_pairs: Sequence[Any]) -> Any:
        """One device flag: some queued checksum or digest disagrees (None: nothing queued).

        The expected values go up through pinned memory without blocking, so
        nothing here waits for the device.  Malformed digests are refused on
        the host before this is called.
        """
        framework = _require_torch()
        if self._copy_stream is not None:
            framework.cuda.current_stream(self.device).wait_stream(self._copy_stream)
        flags = []

        def upload(values, device):
            host = framework.as_tensor(np.asarray(values, dtype=np.int64))
            if getattr(device, "type", None) == "cuda":
                host = host.pin_memory()
            return host.to(device=device, non_blocking=True)

        if pairs:
            observed = framework.stack(tuple(p[1].reshape(()).to(framework.int64) for p in pairs))
            expected = upload([int(p[0]) for p in pairs], observed.device)
            flags.append(observed.ne(expected).any())
        if digest_pairs:
            device = digest_pairs[0][3].device
            references = upload(np.concatenate(tuple(pair[2] for pair in digest_pairs)), device)
            table_references = upload(np.concatenate(tuple(
                pair[5].get("table_expected")
                if pair[5].get("table_expected") is not None else pair[2]
                for pair in digest_pairs
            )), device)
            flags.append(references.ne(table_references).any())
            cursor = 0
            for pair in digest_pairs:
                reference = references[cursor : cursor + len(pair[2])]
                cursor += len(pair[2])
                for observed in (pair[3], pair[5]["first_use"], pair[5]["after_use"]):
                    if observed is not None:
                        flags.append(observed.reshape(-1).ne(reference).any())
        if not flags:
            return None
        return framework.stack(flags).any()

    def _verify_delivery_checksums(
        self, *, drained: bool = True, receipt_start: int | None = None,
        receipt_stop: int | None = None, on_error: bool = False,
    ) -> None:
        """Queue a device disagreement flag and clear this panel's probes.

        Host-known shape and lifecycle faults raise immediately. The flag
        settles after a decision or at the group tail. Only a failed flag
        reads the detailed checksum and row-digest reports on the host.
        """

        if self._device_look_delivery_active() and not on_error:
            from mechanistic_systems.inference.device_bce import defer_device_check

            with self._pinned_ring_lock:
                pairs = tuple(pair for pair in self._delivery_checksum_pairs
                              if "labels" in str(pair[2]))
                self._delivery_checksum_pairs.clear()
                self._row_digest_pairs.clear()
                self._staging_integrity_errors = []
            if pairs:
                framework = _require_torch()
                if self._copy_stream is not None:
                    framework.cuda.current_stream(self.device).wait_stream(self._copy_stream)
                observed = framework.stack(tuple(pair[1].reshape(()) for pair in pairs))
                expected = framework.tensor([int(pair[0]) for pair in pairs],
                    dtype=observed.dtype, device=observed.device)
                defer_device_check(observed.ne(expected).any(), lambda: RuntimeError(
                    "delivered labels differ from the labels staged for this look"))
            return
        armed = self._delivery_integrity_armed
        with self._pinned_ring_lock:
            pairs = list(self._delivery_checksum_pairs)
            self._delivery_checksum_pairs.clear()
            digest_pairs = list(self._row_digest_pairs)
            self._row_digest_pairs.clear()
            staging_errors = list(getattr(self, '_staging_integrity_errors', ()))
            self._staging_integrity_errors = []
        if any(_row_digest_pair_defects(pair) for pair in digest_pairs):
            self._settle_delivery(
                pairs, digest_pairs, staging_errors, drained=drained,
                verified_on_device=False,
            )
            return
        # The host-known part settles now (structural refusals, missing
        # use probes, staging errors, counters) with no device read; the value
        # comparison is one device flag settled off the critical path, whose
        # detailed host report is built only if it fails.
        flag = self._device_delivery_disagreement(pairs, digest_pairs)
        if flag is not None:
            from mechanistic_systems.measurement.device_deferred import DEVICE_FAULTS

            def failure() -> BaseException:
                try:
                    self._settle_delivery(
                        pairs, digest_pairs, (), drained=drained, verified_on_device=False,
                        record=False,
                    )
                except BaseException as error:  # the detailed refusal
                    reason = f"{type(error).__name__}:{error}"
                    if isinstance(error, (DeliveryIntegrityError, _lane_g.BankRowDigestTableError)):
                        if self._device_gather_run is not None:
                            self._device_gather_run.record_identity_error(error)
                        elif self._device_gather is not None:
                            self._device_gather.delivery_identity_error = error
                            self._device_gather.failure_details = dict(getattr(error, "details", {}) or {})
                    elif getattr(self, "_device_gather_run", None) is not None:
                        self._device_gather_run.record_failure(reason, getattr(error, "details", None))
                    elif getattr(self, "_device_gather", None) is not None:
                        self._device_gather.failure = reason
                        self._device_gather.failure_details = getattr(error, "details", None)
                    if receipt_start is not None:
                        with self._receipt_lock:
                            stop = len(self._receipts) if receipt_stop is None else min(len(self._receipts), receipt_stop)
                            for index in range(receipt_start, stop):
                                self._receipts[index] = replace(
                                    self._receipts[index], integrity_failed=True,
                                    device_gather=self._device_gather_receipt(),
                                )
                    return error
                return RowDigestIntegrityError(
                    "delivered rows disagreed with their checksums or digests on the device",
                    {"fault_class": RowDigestIntegrityError.fault_class, "chunks": ()},
                )

            DEVICE_FAULTS.note(flag, failure, fold_into_look=not on_error)
        self._settle_delivery(
            pairs, digest_pairs, staging_errors, drained=drained, verified_on_device=True,
        )

    def _settle_delivery(
        self,
        pairs: list[Any],
        digest_pairs: list[Any],
        staging_errors: Sequence[BaseException],
        *,
        drained: bool,
        verified_on_device: bool,
        record: bool = True,
    ) -> None:
        """Settle taken checksum and digest pairs on the host; raise on any refusal.

        ``record=False`` only builds the refusal (a second, reporting pass
        over pairs already counted).
        """

        armed = self._delivery_integrity_armed
        checksum_failures: list[dict[str, Any]] = []
        checksum_values = (
            [int(pair[0]) for pair in pairs] if verified_on_device else
            (_require_torch().stack(tuple(p[1].reshape(()) for p in pairs)).to(device="cpu").tolist() if pairs else [])
        )
        for (expected, _device_sum, label), observed in zip(pairs, checksum_values):
            if observed != int(expected):
                with self._pinned_ring_lock:
                    self._delivery_checksum_mismatches += 1
                    if not record:
                        self._delivery_checksums_verified -= 1
                message = (
                    f"delivered labels differ from the bank labels for their context ids "
                    f"({label}: bank={expected} delivered={observed})"
                    if str(label).startswith("device_gather_") else
                    "delivered device bytes diverged from staged host bytes "
                    f"({label}: staged={expected} delivered={observed}); the "
                    "asynchronous H2D read a mutated pinned slot"
                )
                if not armed:
                    raise RuntimeError(message)
                checksum_failures.append(
                    {
                        "label": str(label),
                        "staged": int(expected),
                        "delivered": int(observed),
                        "message": message,
                    }
                )
                continue
            if record:
                with self._pinned_ring_lock:
                    self._delivery_checksums_verified += 1
        digest_failures, digest_summary = (
            self._settle_row_digest_pairs(
                digest_pairs, drained=drained, device_values_verified=verified_on_device,
                record=record,
            )
            if digest_pairs
            else ([], {"panel_chunks": 0, "panel_rows": 0})
        )
        if checksum_failures:
            first_checksum = checksum_failures[0]
            raise DeliveryChecksumIntegrityError(
                first_checksum["message"]
                + (
                    f" [{len(checksum_failures)} checksum failure(s); "
                    f"{len(digest_failures)} row-digest record(s) in the same panel]"
                ),
                {
                    "fault_class": DeliveryChecksumIntegrityError.fault_class,
                    "drained": bool(drained),
                    "checksum_failures": tuple(checksum_failures),
                    "row_digest": {
                        "algorithm": ROW_DIGEST_ALGORITHM,
                        **digest_summary,
                        "stages": tuple(sorted({failure["stage"] for failure in digest_failures})),
                        "chunks": tuple(digest_failures),
                    },
                },
            )
        if staging_errors and not any(
            failure.get("reason") != "required_probe_missing" for failure in digest_failures
        ):
            raise staging_errors[0]
        if digest_failures:
            first = digest_failures[0]
            raise RowDigestIntegrityError(
                "delivered image rows diverged from the bank rows gathered for "
                f"their context ids ({len(digest_failures)} mismatch record(s) over "
                f"{digest_summary['panel_chunks']} chunks; first: stage={first['stage']} chunk "
                f"{first['chunk_index']} rows {first['positions']} contexts "
                f"{first['context_ids']})",
                {
                    "fault_class": RowDigestIntegrityError.fault_class,
                    "algorithm": ROW_DIGEST_ALGORITHM,
                    "drained": bool(drained),
                    **digest_summary,
                    "stages": tuple(sorted({failure["stage"] for failure in digest_failures})),
                    "chunks": tuple(digest_failures),
                    "checksum_failures": (),
                },
            )

    def _settle_row_digest_pairs(
        self, digest_pairs: list[tuple[int, tuple[str, ...], np.ndarray, Any, str, dict]],
        *,
        drained: bool = True,
        device_values_verified: bool = False,
        record: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """CLASS3 I1 settle of one panel's digest pairs; returns (failures, summary).

        The device stages settle exactly as in v5.2.  For a
        chunk whose authority is the bank row digest table, the gather stage
        compares the gather's digest (when the gather computed one) with the
        table on EVERY row instead of re-reading a sample of rows from the
        bank.  Chunks with the gather-time authority (no table; ``legacy``)
        settle exactly as in v5.2.
        """

        framework = _require_torch()
        # Shape/dtype of every digest, before any comparison or substitution
        # of expected values: a malformed digest is refused below and left
        # out of the concatenations, so it cannot shift other chunks' rows.
        defects = [_row_digest_pair_defects(pair) for pair in digest_pairs]
        if any(defects):
            device_values_verified = False
        if self._copy_stream is not None and not device_values_verified:
            # Panel end already synchronized the copy events; this is the
            # explicit statement that the copy-stream digest kernels have
            # completed.  The use digests were enqueued on the consumer's
            # stream, which the D2H copies below are ordered after.
            self._copy_stream.synchronize()
        # One D2H for the copy-stream digests and one per use stage for the
        # whole panel.
        copy_included = tuple("after_copy" not in defect for defect in defects)
        copy_tensors = tuple(
            pair[3].reshape(-1)
            for pair, included in zip(digest_pairs, copy_included)
            if included
        )
        observed_all = None if device_values_verified else (
            framework.cat(copy_tensors).to(device="cpu").numpy()
            if copy_tensors
            else np.empty((0,), dtype=np.int64)
        )
        stage_arrays: dict[str, np.ndarray] = {}
        stage_included: dict[str, tuple[bool, ...]] = {}
        for key in ("first_use", "after_use"):
            stage_included[key] = tuple(
                pair[5][key] is not None and key not in defect
                for pair, defect in zip(digest_pairs, defects)
            )
            tensors = tuple(
                pair[5][key].reshape(-1)
                for pair, included in zip(digest_pairs, stage_included[key])
                if included
            )
            stage_arrays[key] = (
                None if device_values_verified else
                framework.cat(tensors).to(device="cpu").numpy()
                if tensors
                else np.empty((0,), dtype=np.int64)
            )
        cursor = 0
        stage_cursors = {"first_use": 0, "after_use": 0}
        failures: list[dict[str, Any]] = []
        delivered_chunks = 0
        checked_rows = 0
        gather_witness_rows = 0
        witnessed_rows = 0
        stage_rows = {"gather_vs_bank": 0, "after_copy": 0, "first_use": 0, "after_use": 0}
        uncovered_rows = 0

        def report(stage: str, chunk_index: int, label: str, context_ids: tuple[str, ...],
                   positions: np.ndarray, expected: np.ndarray, observed: np.ndarray) -> None:
            sample = positions[:_ROW_DIGEST_MISMATCH_SAMPLE_CAP]
            failures.append(
                {
                    "stage": stage,
                    "chunk_index": int(chunk_index),
                    "label": str(label),
                    "rows": int(len(context_ids)),
                    "mismatch_count": int(positions.size),
                    "positions": tuple(int(value) for value in sample),
                    "context_ids": tuple(str(context_ids[int(value)]) for value in sample),
                    "expected": tuple(int(expected[int(value)]) for value in sample),
                    "observed": tuple(int(observed[int(value)]) for value in sample),
                    "chunk_head": tuple(str(value) for value in context_ids[:2]),
                    "chunk_tail": tuple(str(value) for value in context_ids[-2:]),
                }
            )

        def malformed(stage: str, chunk_index: int, label: str, context_ids: tuple[str, ...],
                      part: str, detail: str) -> None:
            rows = len(context_ids)
            sample = tuple(range(min(rows, _ROW_DIGEST_MISMATCH_SAMPLE_CAP)))
            failures.append({
                "stage": stage, "chunk_index": int(chunk_index),
                "label": str(label), "rows": rows, "mismatch_count": rows,
                "positions": sample,
                "context_ids": tuple(str(context_ids[i]) for i in sample),
                "expected": (), "observed": (), "reason": "malformed_digest",
                "digest": part, "detail": str(detail),
            })

        for position, (chunk_index, context_ids, expected, _observed, label, use_state) in enumerate(digest_pairs):
            rows = len(context_ids)
            defect = defects[position]
            consumer_used = False
            if "lifecycle_lock" in use_state:
                with use_state["lifecycle_lock"]:
                    consumer_used = bool(use_state["consumer_used"])
                    discarded_before_use = bool(use_state["discarded"] and not consumer_used)
                    use_state["lifecycle"] = "settled"
            else:
                discarded_before_use = bool(use_state.get("discarded"))
            delivered = int(not discarded_before_use)
            delivered_chunks += delivered
            # Cursors advance only over the digests that were concatenated.
            copy_rows = (
                expected if device_values_verified else observed_all[cursor : cursor + rows]
            ) if copy_included[position] else None
            if copy_included[position]:
                cursor += rows
            used_rows: dict[str, np.ndarray | None] = {}
            for key in ("first_use", "after_use"):
                if stage_included[key][position]:
                    used_rows[key] = (expected if device_values_verified else
                                      stage_arrays[key][stage_cursors[key] : stage_cursors[key] + rows])
                    stage_cursors[key] += rows
                else:
                    used_rows[key] = None
            full = _lane_g.row_digest_guard(self).mode == _lane_g.GUARD_MODE_FULL
            strict_authority = _lane_g.row_digest_guard(self).mode in (
                _lane_g.GUARD_MODE_FULL, _lane_g.GUARD_MODE_INSPECTION
            )
            if strict_authority and use_state.get("authority") != _lane_g.AUTHORITY_TABLE:
                raise _lane_g.BankRowDigestTableError("guard mode lost certified table authority")
            if "reference" in defect:
                # Nothing on this chunk can be compared with a malformed
                # authority; refuse the whole chunk.
                malformed(ROW_DIGEST_STAGE_COPY, chunk_index, label, context_ids,
                          "reference", defect["reference"])
                uncovered_rows += rows
                continue
            # Rows that some required digest could not cover.
            missing_required_use = bool(defect)
            missing_interrupted_use = False
            if copy_rows is None:
                malformed(ROW_DIGEST_STAGE_COPY, chunk_index, label, context_ids,
                          "after_copy", defect["after_copy"])
                flagged = np.arange(rows, dtype=np.int64)
            else:
                checked_rows += rows * delivered
                stage_rows["after_copy"] += rows * delivered
                flagged = np.empty(0, dtype=np.int64) if device_values_verified else np.flatnonzero(copy_rows != expected)
                if flagged.size:
                    report(ROW_DIGEST_STAGE_COPY, chunk_index, label, context_ids, flagged, expected, copy_rows)
            panel = use_state.get("panel")
            panel_drained = None if panel is None else panel.get("drained")
            chunk_drained = drained if panel_drained is None else bool(panel_drained)
            for key, stage in (
                ("first_use", ROW_DIGEST_STAGE_USE),
                ("after_use", ROW_DIGEST_STAGE_AFTER_USE),
            ):
                if discarded_before_use:
                    continue
                if key in defect:
                    # A recorded probe that cannot hold one digest per row
                    # is refused in every mode, like any recorded mismatch.
                    malformed(stage, chunk_index, label, context_ids, key, defect[key])
                    flagged = np.union1d(flagged, np.arange(rows, dtype=np.int64))
                    continue
                if use_state[key] is None:
                    # An interrupted panel can have staged chunks that the
                    # consumer never reached.  Missing use probes on those
                    # chunks are not evidence of changed image bytes.  Keep
                    # comparing every probe that was actually recorded.  A
                    # chunk the consumer already used is not exempt: its
                    # probes are required even when its panel stopped early.
                    if full and (chunk_drained or consumer_used):
                        missing_required_use = True
                        sample = tuple(range(min(rows, _ROW_DIGEST_MISMATCH_SAMPLE_CAP)))
                        failures.append({
                            "stage": stage, "chunk_index": int(chunk_index),
                            "label": str(label), "rows": rows, "mismatch_count": rows,
                            "positions": sample,
                            "context_ids": tuple(str(context_ids[i]) for i in sample),
                            "expected": (), "observed": (), "reason": "required_probe_missing",
                        })
                    elif full:
                        missing_interrupted_use = True
                    continue
                stage_rows[key] += rows
                used = used_rows[key]
                if device_values_verified:
                    continue
                positions = np.flatnonzero(used != expected)
                # A later stage reports only rows unchanged at earlier stages.
                fresh = np.setdiff1d(positions, flagged, assume_unique=True)
                if fresh.size:
                    report(stage, chunk_index, label, context_ids, fresh, expected, used)
                    flagged = np.union1d(flagged, fresh)
            if missing_required_use or missing_interrupted_use:
                uncovered_rows += rows
            if use_state.get("authority") == _lane_g.AUTHORITY_TABLE:
                if "table_expected" in defect:
                    # A malformed table digest cannot certify the gather.
                    malformed(ROW_DIGEST_STAGE_GATHER, chunk_index, label, context_ids,
                              "table_expected", defect["table_expected"])
                    continue
                stage_rows["gather_vs_bank"] += rows * delivered
                # The gather stage at full coverage -- the
                # gather's digest of every row it returned against the bank
                # row digest table (no bank re-read: the table is the bank).
                if use_state.get("gather_digest_present"):
                    table_expected = use_state["table_expected"]
                    gather_witness_rows += rows * delivered
                    witnessed_rows += rows * delivered
                    gathered_wrong = (np.empty(0, dtype=np.int64) if device_values_verified
                                      else np.flatnonzero(expected != table_expected))
                    if gathered_wrong.size:
                        report(
                            ROW_DIGEST_STAGE_GATHER,
                            chunk_index,
                            label,
                            context_ids,
                            gathered_wrong,
                            table_expected,  # the bank's digest is the authority here
                            expected,  # ... and the gather-time digest the observation
                        )
                continue
        if record and witnessed_rows:
            guard = _lane_g.row_digest_guard(self)
            with guard.lock:
                guard.rows_witness_compared += witnessed_rows
        if record:
            guard = _lane_g.row_digest_guard(self)
            with guard.lock:
                for stage, count in stage_rows.items():
                    guard.stage_rows[stage] += count
        with self._pinned_ring_lock:
            if record:
                self._row_digest_chunks_checked += delivered_chunks
                self._row_digest_rows_checked += checked_rows
                self._row_digest_gather_witness_rows += gather_witness_rows
                self._row_digest_rows_uncovered += uncovered_rows
            if failures:
                self._row_digest_mismatches += len(failures)
                for failure in failures:
                    self._row_digest_mismatches_by_stage[failure["stage"]] = (
                        self._row_digest_mismatches_by_stage.get(failure["stage"], 0) + 1
                    )
                    if len(self._row_digest_mismatch_samples) < _ROW_DIGEST_MISMATCH_SAMPLE_CAP:
                        self._row_digest_mismatch_samples.append(failure)
        return failures, {"panel_chunks": len(digest_pairs), "panel_rows": int(checked_rows)}

    def _prefetch_working_set(self, context_ids: tuple[str, ...]) -> None:
        """Fault the declared rows' pages ahead of the serial gathers.

        One strided byte per 4096-byte page is read per row in bank-ordinal
        order on daemon threads, so a cold shared-filesystem or freshly
        staged bank starts faulting its declared working set at queue depth
        greater than one before the first gather arrives. Runtime-only: no
        logical draw, order, or value changes.
        """

        inputs = getattr(self.source, "_inputs", None)
        if inputs is None or not hasattr(inputs, "shape"):
            return
        ordinals = np.fromiter(
            (int(self._records[value].ordinal) for value in context_ids),
            dtype=np.int64,
            count=len(context_ids),
        )
        ordinals = np.sort(ordinals)
        row_elements = int(np.prod(inputs.shape[1:], dtype=np.int64))
        item_bytes = int(inputs.dtype.itemsize)
        page_stride = max(1, 4096 // max(1, item_bytes))
        flat = inputs.reshape(inputs.shape[0], row_elements)
        with self._prefetch_lock:
            if self._prefetch_executor is None:
                from concurrent.futures import ThreadPoolExecutor

                self._prefetch_executor = ThreadPoolExecutor(
                    max_workers=max(1, int(self.spec.producer_workers)),
                    thread_name_prefix="bank-ws-prefetch",
                )
            executor = self._prefetch_executor
            self._prefetch_rows_submitted += int(ordinals.size)
            self._prefetch_panels += 1
        span = max(1, int(ordinals.size // max(1, int(self.spec.producer_workers))))
        for start in range(0, int(ordinals.size), span):
            block = ordinals[start : start + span]

            def touch(rows: np.ndarray = block) -> None:
                for ordinal in rows:
                    # Strided page-granular read; the tiny materialized
                    # array is discarded, only the faults matter.
                    _ = flat[int(ordinal), ::page_stride].sum()

            executor.submit(touch)

    @property
    def performance_gate_eligible(self) -> bool:
        return bool(self.optimized_bank)

    def _validate_contexts(self, context_ids: Sequence[str]) -> tuple[str, ...]:
        requested = tuple(str(value) for value in context_ids)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("physical panel data plane requires unique nonempty context IDs")
        missing = tuple(value for value in requested if value not in self._records)
        if missing:
            raise ValueError("panel contains unknown physical contexts: " + ",".join(missing[:8]))
        return requested

    def physical_execution_order(
        self,
        context_ids: Sequence[str],
    ) -> tuple[str, ...]:
        """Return an outcome-independent, bank-coalesced physical execution order.

        Logical IID order is owned by the caller and restored from the
        projected-readout slab. Physical source evaluation is therefore free
        to traverse immutable bank ordinals monotonically, avoiding a fresh
        random mmap gather in every fixed-size chunk.
        """

        requested = self._validate_contexts(context_ids)
        return tuple(
            sorted(
                requested,
                key=lambda value: (
                    int(self._records[value].ordinal),
                    value,
                ),
            )
        )

    def record_consumer_label_check(self, consumer: str, *, rows: int, outcome: str) -> None:
        """CLASS3: count one consumer-side bank-label check.

        ``outcome`` is ``passed`` when a check is queued, ``fault`` when its
        device flag fails, or ``uncovered`` when no row authority exists.
        Armed only; disarmed it is a no-op.
        """

        if not self._delivery_integrity_armed:
            return
        with self._pinned_ring_lock:
            counts = self._consumer_label_checks.setdefault(
                str(consumer),
                {"checks": 0, "rows_checked": 0, "uncovered": 0, "failures": 0},
            )
            if outcome == "uncovered":
                counts["uncovered"] += 1
                return
            if outcome == "fault":
                counts["failures"] += 1
                return
            counts["checks"] += 1
            counts["rows_checked"] += int(rows)

    def authoritative_labels(self, context_ids: Sequence[str]) -> np.ndarray | None:
        """Certified bank labels for ``context_ids`` (host int64), or None.

        The label a served batch carries for a context
        is, by construction, ``source._labels[records[context].ordinal]``
        (``_gather_bank_rows``).  This method re-derives that value directly
        from the bank/axis, bypassing the delivery pipeline (gather, pinned
        staging, H2D, device workspace, consumer clone), so a consumer can
        certify that the rows it received are the rows it asked for.  Pure
        and outcome-independent; returns None when no label authority exists (test
        doubles without labels) or a context is unknown to this plane, in
        which case callers skip the check and COUNT the pass as
        uncovered.  Only lookup failures are absorbed; any other
        exception propagates instead of silently disarming the check.
        """

        requested = tuple(str(value) for value in context_ids)
        labels = getattr(self.source, "_labels", None)
        try:
            if labels is not None:
                return np.fromiter(
                    (
                        int(labels[int(self._records[value].ordinal)])
                        for value in requested
                    ),
                    dtype=np.int64,
                    count=len(requested),
                )
            return np.fromiter(
                (int(self._records[value].label) for value in requested),
                dtype=np.int64,
                count=len(requested),
            )
        except (KeyError, IndexError, AttributeError, TypeError, ValueError):
            return None

    @staticmethod
    def _chunks(context_ids: tuple[str, ...], batch_rows: int) -> tuple[tuple[str, ...], ...]:
        """Emit stable full physical shapes followed by at most one tail."""

        count = len(context_ids)
        capacity = int(batch_rows)
        if count <= 0 or capacity <= 0:
            raise ValueError("balanced chunking requires positive rows")
        return tuple(context_ids[start : start + capacity] for start in range(0, count, capacity))

    def _prepare_bank_mapping(self) -> None:
        """Ask the kernel to populate the certified npy mapping when possible.

        Advice is a performance hint.  A failed or unsupported hint never
        changes the bytes read from the bank.  MAP_POPULATE requires a new
        mapping; its ndarray view retains the npy payload offset and shape.
        """

        inputs = self._bank_inputs
        mapping = getattr(inputs, "_mmap", None)
        if mapping is None:
            self._bank_page_advice["mapping"] = "not_memmap"
            return
        for name in ("MADV_WILLNEED", "MADV_HUGEPAGE"):
            advice = getattr(mmap, name, None)
            if advice is None or not hasattr(mapping, "madvise"):
                self._bank_page_advice[name] = "unavailable"
                continue
            try:
                mapping.madvise(advice)
            except (OSError, ValueError) as error:
                self._bank_page_advice[name] = f"rejected:{type(error).__name__}"
            else:
                self._bank_page_advice[name] = "accepted"
        populate = getattr(mmap, "MAP_POPULATE", None)
        private = getattr(mmap, "MAP_PRIVATE", None)
        if not bool(inputs.flags.c_contiguous):
            self._bank_page_advice["MAP_POPULATE"] = "unsupported_layout"
            return
        if populate is None or private is None or not getattr(inputs, "filename", None):
            self._bank_page_advice["MAP_POPULATE"] = "unavailable"
            return
        try:
            with open(inputs.filename, "rb") as bank_file:
                populated = mmap.mmap(
                    bank_file.fileno(), 0,
                    flags=private | populate,
                    prot=mmap.PROT_READ,
                )
            view = np.ndarray(
                inputs.shape, dtype=inputs.dtype, buffer=populated,
                offset=int(inputs.offset), order="C",
            )
        except (OSError, ValueError, TypeError) as error:
            self._bank_page_advice["MAP_POPULATE"] = f"rejected:{type(error).__name__}"
            return
        self._populated_bank_mapping = populated
        self._bank_inputs = view
        self._bank_page_advice["MAP_POPULATE"] = "accepted"

    def _gather_direct_pinned_chunk(
        self, context_ids: tuple[str, ...], *, parallel_shards: int
    ) -> Any | None:
        """Take bank rows directly into leased delivery tensors.

        NumPy's `take(out=...)` writes the final pinned storage; no pageable
        image array, concatenate, reorder copy, or staging copy is made.
        Shards own disjoint row ranges, so their writes cannot overlap.
        """

        framework = _require_torch()
        bank = self._bank_inputs
        labels_bank = self.source._labels  # noqa: SLF001 - bank adapter
        if bank is None or np.dtype(labels_bank.dtype) != np.dtype(np.int64):
            return None
        images, labels, pin = self._empty_row_assembly_host(
            row_count=len(context_ids), reference_images=bank,
            reference_labels=labels_bank,
            memory_format=framework.contiguous_format,
        )
        try:
            gathered_images, gathered_labels, row_digests = (
                self._gather_bank_rows_digested(
                    context_ids, parallel_shards=parallel_shards,
                    out_images=images, out_labels=labels,
                )
            )
            if pin:
                with self._pinned_ring_lock:
                    for old, delivered in ((images, gathered_images), (labels, gathered_labels)):
                        if old is not delivered:
                            self._pinned_slot_state[id(old)].update(
                                leased=False, event=None, staged_checksum=None
                            )
            images, labels = gathered_images, gathered_labels
            if pin:
                images, labels, image_stamp, label_stamp = self._stage_gathered_host_stamped(
                    images, labels
                )
            elif self._delivery_integrity_armed:
                image_stamp = self._host_delivery_checksum(images)
                label_stamp = self._host_delivery_checksum(labels)
                with self._pinned_ring_lock:
                    self._payload_stamps_issued += 2
            else:
                image_stamp = label_stamp = None
            self._direct_gather_chunks += 1
            self._direct_gather_bytes += int(images.numel())
            if not self._delivery_integrity_armed:
                return context_ids, images, labels, int(images.numel())
            return _StagedChunkPayload(
                context_ids, images, labels, int(images.numel()),
                row_digests=row_digests,
                image_checksum=image_stamp, label_checksum=label_stamp,
            )
        except BaseException:
            if pin:
                with self._pinned_ring_lock:
                    for slot in (images, labels):
                        state = self._pinned_slot_state.get(id(slot))
                        if state is not None:
                            state.update(leased=False, event=None, staged_checksum=None)
            raise

    def _gather_bank_chunk(
        self,
        context_ids: tuple[str, ...],
        *,
        parallel_shards: int = 1,
        _pin_exempt: bool = False,
    ) -> Any:
        # Rows resident in the device pin bank are excluded
        # from the host gather; the device batch re-interleaves them at
        # their exact chunk positions.  ``_pin_exempt`` marks the pin
        # build's own gathers (and tests of the raw host path).  With the
        # pin unarmed/unbuilt (or on CPU) this block is inert and the
        # return type is exactly the historical 4-tuple.
        if not _pin_exempt and self._pin_serve_active():
            host_ids, host_positions, pin_positions, pin_slots = self._pin_split_for_chunk(
                context_ids
            )
            with self._pin_lock:
                self._pin_gathered_rows += len(context_ids)
                self._pin_hit_rows += len(pin_positions)
            if pin_positions:
                host_images = None
                host_labels = None
                loaded_bytes = 0
                host_payload = None
                if host_ids:
                    host_payload = self._gather_bank_chunk(
                        host_ids,
                        parallel_shards=parallel_shards,
                        _pin_exempt=True,
                    )
                    _ids, host_images, host_labels, loaded_bytes = host_payload
                host_stamps = _payload_stamps(host_payload)
                return _PinSplitPayload(
                    context_ids=tuple(context_ids),
                    host_images=host_images,
                    host_labels=host_labels,
                    host_positions=host_positions,
                    pin_positions=pin_positions,
                    pin_slots=pin_slots,
                    loaded_bytes=int(loaded_bytes),
                    host_row_digests=_payload_row_digests(host_payload),
                    host_image_checksum=(None if host_stamps is None else host_stamps[0]),
                    host_label_checksum=(None if host_stamps is None else host_stamps[1]),
                )
        if (
            self._image_gather_enabled and context_ids
            and (self.device.type == "cuda" or not _pin_exempt)
        ):
            direct = self._gather_direct_pinned_chunk(
                context_ids, parallel_shards=parallel_shards
            )
            if direct is not None:
                return direct
        # Advanced mmap indexing already owns its result; an additional
        # ``.copy()`` doubled host memory traffic.  For a one-physical-batch
        # look, split the monotonically ordered read over the same bounded
        # producer width used by the panel pipeline.  This preserves the
        # exact physical and logical orders while allowing NVMe queue depth
        # greater than one before the first GPU batch.  (The gather
        # body lives in ``_gather_bank_rows`` so the salvage top-up shares
        # it — one implementation, never a mirror.)
        source_images, source_labels, row_digests = self._gather_bank_rows_digested(
            context_ids, parallel_shards=parallel_shards
        )
        images, labels, image_checksum, label_checksum = self._stage_gathered_host_stamped(
            source_images, source_labels
        )
        if not self._delivery_integrity_armed:
            return context_ids, images, labels, int(images.numel())
        # CLASS3: the authority (per-row digests of the gathered bank bytes)
        # and the staged stamps travel WITH the payload through every
        # queue, lookahead entry and ring on the way to ``_device_batch``.
        return _StagedChunkPayload(
            context_ids,
            images,
            labels,
            int(images.numel()),
            row_digests=row_digests,
            image_checksum=image_checksum,
            label_checksum=label_checksum,
        )

    def _gather_bank_rows(
        self,
        context_ids: tuple[str, ...],
        *,
        parallel_shards: int = 1,
    ) -> tuple[Any, Any]:
        """Gather UNSTAGED source rows in the requested order.

        The mmap gather of ``_gather_bank_chunk`` without the pinned-slot
        staging tail: the salvage top-up needs raw row bytes to place at
        served positions, never a leased delivery tensor.  Byte-identical
        per row to what ``_gather_bank_chunk`` would deliver for the same
        ids (the staging step is a value-preserving copy).
        """

        images, labels, _digests = self._gather_bank_rows_digested(
            context_ids, parallel_shards=parallel_shards, digests=False
        )
        return images, labels

    def _gather_bank_rows_digested(
        self,
        context_ids: tuple[str, ...],
        *,
        parallel_shards: int = 1,
        digests: bool = True,
        out_images: Any | None = None,
        out_labels: Any | None = None,
    ) -> tuple[Any, Any, np.ndarray | None]:
        """``_gather_bank_rows`` plus the CLASS3 I1 per-row digests.

        The digests are computed from the gathered numpy rows, in the
        requested order, BEFORE any staging copy: they are the authority
        every later stage of the delivery path is certified against.
        ``None`` when disarmed or ``digests=False``.
        """

        digests = bool(digests and not self._device_look_delivery_active())
        framework = _require_torch()
        ordinals = np.fromiter(
            (int(self._records[value].ordinal) for value in context_ids),
            dtype=np.int64,
            count=len(context_ids),
        )
        if out_images is not None:
            if out_labels is None:
                raise ValueError("direct gather requires both output buffers")
            if ordinals.size and (
                int(ordinals.min()) < 0
                or int(ordinals.max()) >= min(
                    int(self._bank_inputs.shape[0]), int(self.source._labels.shape[0])
                )
            ):
                raise IndexError("direct gather bank ordinal out of range")
            image_out = out_images.numpy()
            label_out = out_labels.numpy()
            count = min(max(1, int(parallel_shards)), len(context_ids))
            bounds = tuple(
                (len(context_ids) * index // count,
                 len(context_ids) * (index + 1) // count)
                for index in range(count)
            )

            def write_span(bounds_pair: tuple[int, int]) -> None:
                start, stop = bounds_pair
                np.take(self._bank_inputs, ordinals[start:stop], axis=0,
                        out=image_out[start:stop], mode="clip")
                np.take(self.source._labels, ordinals[start:stop], axis=0,
                        out=label_out[start:stop], mode="clip")

            if count == 1:
                write_span(bounds[0])
            else:
                self._bank_shard_executor_uses += 1
                tuple(self._bank_shard_executor.map(write_span, bounds))
            row_digests = None
            if digests and self._delivery_integrity_armed:
                row_digests = _lane_g.gather_authority_digests(self, ordinals)
                if row_digests is None and _lane_g.gather_host_digest_required(self):
                    row_digests = row_digests_host(image_out)
            return out_images, out_labels, row_digests
        already_ordered = bool(ordinals.size <= 1 or np.all(ordinals[:-1] <= ordinals[1:]))
        if already_ordered:
            sorted_ordinals = ordinals
            inverse = None
        else:
            order = np.argsort(ordinals, kind="stable")
            inverse = np.empty_like(order)
            inverse[order] = np.arange(order.size)
            sorted_ordinals = ordinals[order]
        shard_count = min(
            max(1, int(parallel_shards)),
            max(1, int(sorted_ordinals.size)),
        )

        def gather_ordinals(index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return (
                np.asarray(
                    self.source._inputs[index]  # noqa: SLF001 - bank adapter
                ),
                np.asarray(
                    self.source._labels[index],  # noqa: SLF001 - bank adapter
                    dtype=np.int64,
                ),
            )

        if shard_count == 1:
            images_array, labels_array = gather_ordinals(sorted_ordinals)
        else:
            ordinal_shards = tuple(
                shard for shard in np.array_split(sorted_ordinals, shard_count) if shard.size
            )
            self._bank_shard_executor_uses += 1
            gathered = tuple(
                self._bank_shard_executor.map(gather_ordinals, ordinal_shards)
            )
            images_array = np.concatenate(
                tuple(item[0] for item in gathered),
                axis=0,
            )
            labels_array = np.concatenate(
                tuple(item[1] for item in gathered),
                axis=0,
            )
        if inverse is not None:
            images_array = images_array[inverse]
            labels_array = labels_array[inverse]
        row_digests = None
        if digests and self._delivery_integrity_armed:
            # With a bound bank row digest table the
            # per-row digests are the table's entries for the ordinals read
            # (a lookup, not a host digest); the device stages certify the
            # delivered bytes against them and the settle re-derives the
            # ordinals from the ids.  FULL refuses a missing table; only an
            # explicitly weaker mode can use the gather-host fallback.
            row_digests = _lane_g.gather_authority_digests(self, ordinals)
            if row_digests is None and _lane_g.gather_host_digest_required(self):
                row_digests = row_digests_host(images_array)
        return (
            framework.from_numpy(images_array),
            framework.from_numpy(labels_array),
            row_digests,
        )

    def _stage_gathered_host(self, source_images: Any, source_labels: Any) -> tuple[Any, Any]:
        """Stage gathered host rows into their delivery tensors.

        Pure extraction of ``_gather_bank_chunk``'s staging tail (the
        the row-salvage assembler reuses it so leases and checksums have
        exactly one implementation).  CUDA+pin_memory: reusable pinned
        ring slots under the explicit lease lifecycle; otherwise plain
        contiguous channels_last tensors.
        """

        framework = _require_torch()
        if bool(getattr(self.source, "pin_memory", False)) and self.device.type == "cuda":
            # Row-addressed salvage can assemble directly into a pair of
            # leased pinned-ring slots.  Reusing those exact tensors here
            # avoids a second full pageable->pinned host copy while keeping
            # the normal checksum and H2D-event lifecycle unchanged.
            with self._pinned_ring_lock:
                image_state = self._pinned_slot_state.get(id(source_images))
                label_state = self._pinned_slot_state.get(id(source_labels))
                direct_pinned = (
                    id(source_images) in self._pinned_slot_ids
                    and id(source_labels) in self._pinned_slot_ids
                    and image_state is not None
                    and label_state is not None
                    and bool(image_state.get("leased"))
                    and bool(label_state.get("leased"))
                )
            if direct_pinned:
                self._record_staged_checksum(source_images)
                self._record_staged_checksum(source_labels)
                return source_images, source_labels
            leased: list[Any] = []
            try:
                # Reusable pinned ring slots replace a fresh cudaHostAlloc
                # per chunk; reuse is governed by the explicit lease
                # lifecycle, never by ring distance.
                images = self._pinned_host_slot(
                    source_images,
                    memory_format=framework.channels_last,
                )
                leased.append(images)
                images.copy_(source_images)
                labels = self._pinned_host_slot(source_labels)
                leased.append(labels)
                labels.copy_(source_labels)
                self._record_staged_checksum(images)
                self._record_staged_checksum(labels)
            except RuntimeError:
                # Release any lease taken before the failure: this chunk
                # falls back to unpooled tensors, so an abandoned leased
                # slot would otherwise leak until overflow accounting
                # noticed it.
                with self._pinned_ring_lock:
                    for slot in leased:
                        state = self._pinned_slot_state.get(id(slot))
                        if state is not None:
                            state["leased"] = False
                            state["event"] = None
                            state["staged_checksum"] = None
                images = source_images.contiguous(memory_format=framework.channels_last)
                labels = source_labels
        else:
            images = source_images.contiguous(memory_format=framework.channels_last)
            labels = source_labels
        return images, labels

    def _stage_gathered_host_stamped(
        self, source_images: Any, source_labels: Any
    ) -> tuple[Any, Any, int | None, int | None]:
        """``_stage_gathered_host`` plus the staged checksums (CLASS3 I2).

        Staging itself is untouched (one implementation, patched by the
        salvage tests under its own name); this wrapper reads the stamp
        the staging tail recorded on a pooled slot's lease, or computes
        the checksum of an unpooled delivery tensor (CPU, pageable
        fallback). The device delivery detector compares delivered data
        with the payload's stamp. Disarmed: the stamps are
        None and nothing extra runs.
        """

        images, labels = self._stage_gathered_host(source_images, source_labels)
        if not self._delivery_integrity_armed:
            return images, labels, None, None
        stamps: list[int] = []
        for tensor in (images, labels):
            with self._pinned_ring_lock:
                state = self._pinned_slot_state.get(id(tensor))
                lease_stamp = None if state is None else state.get("staged_checksum")
            stamps.append(
                int(lease_stamp)
                if lease_stamp is not None
                else int(self._host_delivery_checksum(tensor))
            )
        with self._pinned_ring_lock:
            self._payload_stamps_issued += 2
        return images, labels, stamps[0], stamps[1]

    def _empty_row_assembly_host(
        self,
        *,
        row_count: int,
        reference_images: Any,
        reference_labels: Any,
        memory_format: Any = None,
    ) -> tuple[Any, Any, bool]:
        """Allocate the final row-salvage host buffers exactly once.

        On CUDA with pinned delivery, the returned tensors are already
        leased pinned-ring slots and can be filled in served-row order.
        Shape and dtype are passed directly to the slot allocator, so no
        full-sized pageable proxy is allocated first. Other devices, pinned
        saturation, and allocation failure retain the ordinary pageable
        allocation. The boolean tells the caller whether a failed assembly
        must explicitly return leases.
        """

        framework = _require_torch()
        rows = int(row_count)
        if rows <= 0:
            raise ValueError("row assembly requires a positive row count")
        image_shape = (
            rows,
            *tuple(int(value) for value in reference_images.shape[1:]),
        )
        image_memory_format = memory_format if memory_format is not None else (
            framework.channels_last if len(image_shape) == 4 else framework.contiguous_format
        )
        def tensor_dtype(reference: Any) -> Any:
            if isinstance(reference.dtype, framework.dtype):
                return reference.dtype
            return framework.from_numpy(np.empty(0, dtype=reference.dtype)).dtype
        image_dtype = tensor_dtype(reference_images)
        label_dtype = tensor_dtype(reference_labels)
        if bool(getattr(self.source, "pin_memory", False)) and self.device.type == "cuda":
            images = None
            labels = None
            try:
                images = self._pinned_host_slot(
                    shape=image_shape,
                    dtype=image_dtype,
                    memory_format=image_memory_format,
                    allow_transient=False,
                )
                if images is not None:
                    labels = self._pinned_host_slot(
                        shape=(rows,),
                        dtype=label_dtype,
                        allow_transient=False,
                    )
            except BaseException as error:
                # Always return a first lease if the paired allocation did
                # not complete. cudaHostAlloc RuntimeError is a performance
                # degradation, not a serving failure; staging retries/falls
                # back exactly as it did before direct assembly existed.
                if images is not None or labels is not None:
                    self._release_lookahead_payload(((), images, labels, 0))
                images = None
                labels = None
                if not isinstance(error, RuntimeError):
                    raise
            if images is not None and labels is not None:
                with self._pinned_ring_lock:
                    image_state = self._pinned_slot_state.get(id(images))
                    label_state = self._pinned_slot_state.get(id(labels))
                    direct_pinned = (
                        id(images) in self._pinned_slot_ids
                        and id(labels) in self._pinned_slot_ids
                        and image_state is not None
                        and label_state is not None
                        and bool(image_state.get("leased"))
                        and bool(label_state.get("leased"))
                    )
                if direct_pinned:
                    return images, labels, True
            if images is not None or labels is not None:
                self._release_lookahead_payload(((), images, labels, 0))
        if self.device.type == "cuda":
            self._pinned_pool.pageable_allocations += 2
        return (
            framework.empty(
                image_shape,
                dtype=image_dtype,
                memory_format=image_memory_format,
            ),
            framework.empty((rows,), dtype=label_dtype),
            False,
        )

    def _gather_fallback_chunk(
        self, context_ids: tuple[str, ...]
    ) -> tuple[tuple[str, ...], Any, Any, int]:
        framework = _require_torch()
        batches = tuple(self.source.panel_batches(context_ids))
        if not batches:
            raise RuntimeError("fallback context source emitted no batches")
        ids = tuple(context_id for batch in batches for context_id in batch.context_ids)
        if ids != context_ids:
            raise ValueError("fallback context source changed panel ordering")
        images = framework.cat(tuple(framework.as_tensor(batch.inputs) for batch in batches), dim=0)
        labels = framework.cat(
            tuple(framework.as_tensor(batch.labels).reshape(-1) for batch in batches),
            dim=0,
        )
        return ids, images, labels, int(images.numel() * images.element_size())

    # ------------------------------------------------------------------ #
    # Persistent panel producer + cross-look lookahead
    # ------------------------------------------------------------------ #

    @property
    def lookahead_enabled(self) -> bool:
        """True when posted looks can reach a live ready-buffer."""

        return (
            self._device_gather is not None or (
            self._persistent_producer_enabled
            and not self._persistent_pool_closed
            and self._lookahead_looks > 0
            )
        )

    @property
    def speculative_extension_enabled(self) -> bool:
        """True when rung-1 speculative posts can reach the ready-buffer."""

        return (self._speculative_extension_env and self.lookahead_enabled
                and not (self._device_gather is not None and self._device_gather.worker_residency))

    @property
    def on_target_pin_enabled(self) -> bool:
        """True when the rung-3 pin may be installed on this plane."""

        return self._on_target_pin_env and self._device_gather is None

    def _persistent_producer_pool(
        self,
    ) -> _PersistentPanelProducerPool | None:
        if self._device_gather is not None or not self._persistent_producer_enabled or self._persistent_pool_closed:
            return None
        pool = self._persistent_pool
        if pool is not None:
            return None if pool.closed else pool
        with self._persistent_pool_lock:
            if self._persistent_pool_closed:
                return None
            if self._persistent_pool is None:
                try:
                    self._persistent_pool = _PersistentPanelProducerPool(
                        workers=int(self.spec.producer_workers),
                        depth=int(self.spec.prefetch_depth),
                        lookahead_looks=int(self._lookahead_looks),
                        resident_byte_limit=(
                            self._pinned_pool.max_bytes or None
                        ),
                    )
                except Exception as error:  # fail-soft: per-look pools
                    self._persistent_pool_closed = True
                    self._persistent_producer_degradations.append(
                        f"pool_construction_failed:{type(error).__name__}:{error}"
                    )
                    return None
            return self._persistent_pool

    def close_persistent_producer(self, *, wait: bool = True) -> None:
        """Shut the persistent producer down; serving degrades per-look.

        Idempotent.  Counters survive for the runtime receipt; queued
        stale gathers are cancelled and their pinned leases returned.
        """

        with self._persistent_pool_lock:
            pool = self._persistent_pool
            self._persistent_pool_closed = True
        if pool is not None and not pool.closed:
            self._evict_all_lookahead_entries(pool)
            pool.closed = True
            pool.gather_executor.shutdown(wait=wait, cancel_futures=True)
            pool.stage_executor.shutdown(wait=wait, cancel_futures=True)
        if self._device_gather is not None:
            if wait and self._device_gather.worker_residency:
                self.settle_image_feed()
            self._device_gather.close(wait=wait)
        if wait and gpu_process_count() > 1:
            self.release_buffer_rings(include_pinned=True)

    def settle_image_feed(self) -> None:
        """Bulk-read delivery identity and timing after the GPU's work."""

        gather = self._device_gather
        if gather is None or not gather.worker_residency:
            return
        try:
            gather.settle_identity()
        except DeviceGatherError as error:
            members = (self,) if self._device_gather_run is None else self._device_gather_run.planes
            if self._device_gather_run is not None:
                self._device_gather_run.record_failure(gather.failure or str(error))
            for plane in members:
                with plane._receipt_lock:
                    for index, receipt in enumerate(plane._receipts):
                        plane._receipts[index] = replace(
                            receipt, integrity_failed=True,
                            device_gather=plane._device_gather_receipt(),
                        )
            raise
        timings, self._image_feed_copy_timings = self._image_feed_copy_timings, []
        with self._receipt_lock:
            for index, events in timings:
                seconds = sum(float(start.elapsed_time(end)) / 1000.0 for start, end in events)
                receipt = self._receipts[index]
                self._receipts[index] = replace(
                    receipt, device_copy_normalize_seconds=seconds,
                    host_to_device_bytes_per_second=(
                        receipt.host_to_device_bytes / seconds if seconds > 0 else None
                    ),
                )

    def release_buffer_rings(
        self, *, include_pinned: bool = False, block: bool = True,
    ) -> None:
        """Release idle per-shape buffers after the pass's workers settle.

        ``block=False`` releases only completed buffers during dispatch.
        Blocking release settles consumers at shutdown.
        Lookahead may still hold host leases; those slots remain until consumed
        or retired. A completed copy event also makes a host slot releasable.
        """

        if self._copy_stream is not None:
            if not block and not self._copy_stream.query():
                return
            if block:
                self._copy_stream.synchronize()
        if not block and any(
            slot.get("busy") and (
                (slot.get("completion") is None and self.device.type == "cuda")
                or (slot.get("completion") is not None and not slot["completion"].query())
            )
            for rings in (self._workspace_rings, self._normalize_rings)
            for ring in rings.values() for slot in ring
        ):
            return
        for rings in (self._workspace_rings, self._normalize_rings):
            for ring in rings.values():
                for slot in ring:
                    event = slot.get("completion")
                    if block and slot.get("busy") and event is not None:
                        event.synchronize()
            rings.clear()
        self._workspace_cursors.clear()
        self._normalize_ring_cursors.clear()
        self._normalize_ring_buffers = 0
        self._normalize_ring_bytes = 0
        if include_pinned:
            with self._pinned_ring_lock:
                for key, ring in tuple(self._pinned_host_rings.items()):
                    retained = []
                    for slot in ring:
                        state = self._pinned_slot_state.get(id(slot), {})
                        event = state.get("event")
                        if state.get("leased") and event is None:
                            retained.append(slot)
                            continue
                        if event is not None:
                            if not block and not event.query():
                                retained.append(slot)
                                continue
                            if block:
                                event.synchronize()
                        self._pinned_slot_ids.discard(id(slot))
                        self._pinned_slot_state.pop(id(slot), None)
                    if retained:
                        self._pinned_host_rings[key] = retained
                        self._pinned_ring_cursors[key] = 0
                    else:
                        del self._pinned_host_rings[key]
                        self._pinned_ring_cursors.pop(key, None)

    @staticmethod
    def _lookahead_entry_class(entry: _LookaheadEntry) -> str:
        if entry.demand_confirmed:
            return "exact_confirmed"
        if entry.speculative:
            return "speculative"
        return "exact"

    @staticmethod
    def _lookahead_future_state(entry: _LookaheadEntry) -> str:
        future = entry.future
        if future is None or future.cancelled():
            return "cancelled"
        if future.done():
            return "done"
        if future.running():
            return "running"
        return "queued"

    @classmethod
    def _lookahead_entry_rank(cls, entry: _LookaheadEntry) -> tuple[int, int, int]:
        """Deterministic multimap precedence required by demand serving."""

        class_rank = {
            "exact_confirmed": 0,
            "exact": 1,
            "speculative": 2,
        }[cls._lookahead_entry_class(entry)]
        state_rank = {
            "done": 0,
            "running": 1,
            "queued": 2,
            "cancelled": 3,
        }[cls._lookahead_future_state(entry)]
        # Newer demand is the deterministic final tie-break.
        return class_rank, state_rank, -int(entry.post_id)

    @classmethod
    def _lookahead_resident_state_receipt(
        cls, pool: _PersistentPanelProducerPool
    ) -> dict[str, Any]:
        classes = ("exact_confirmed", "exact", "speculative")
        states = ("done", "running", "queued", "cancelled")
        matrix = {entry_class: {state: 0 for state in states} for entry_class in classes}
        with pool.lock:
            for entry in pool.live_order.values():
                if not entry.registered:
                    continue
                matrix[cls._lookahead_entry_class(entry)][cls._lookahead_future_state(entry)] += 1
            return {
                "entry_counts": matrix,
                "permits": {
                    "acquired": int(pool.budget_permits_acquired),
                    "released": int(pool.budget_permits_released),
                    "current": int(pool.budget_permits_current),
                    "high_water": int(pool.budget_permits_high_water),
                    "limit": int(pool.lookahead_budget_limit),
                    "exact_once_balance": bool(
                        pool.budget_permits_current
                        == pool.budget_permits_acquired - pool.budget_permits_released
                    ),
                },
                **(
                    {
                        "pinned_budget_bytes": {
                            "limit": int(pool.resident_byte_limit),
                            "reserved_current": int(pool.reserved_bytes_current),
                            "reserved_high_water": int(pool.reserved_bytes_high_water),
                        }
                    }
                    if pool.resident_byte_limit is not None else {}
                ),
            }

    @staticmethod
    def _remove_row_locator_locked(
        pool: _PersistentPanelProducerPool,
        entry: _LookaheadEntry,
        key_row: int,
    ) -> bool:
        context_id = entry.key[int(key_row)]
        locators = pool.row_index.get(context_id)
        if not locators:
            return False
        for index, locator in enumerate(locators):
            if locator.entry is entry and locator.key_row == int(key_row):
                locators.pop(index)
                if not locators:
                    pool.row_index.pop(context_id, None)
                pool.resident_row_locators_current -= 1
                return True
        return False

    @classmethod
    def _register_lookahead_entry_locked(
        cls,
        pool: _PersistentPanelProducerPool,
        entry: _LookaheadEntry,
    ) -> None:
        pool.entries.setdefault(entry.key, []).append(entry)
        pool.live_order[id(entry)] = entry
        entry.registered = True
        pool.resident_entries_current += 1
        pool.resident_entries_high_water = max(
            pool.resident_entries_high_water,
            pool.resident_entries_current,
        )
        for key_row, context_id in enumerate(entry.key):
            pool.row_index.setdefault(context_id, []).append(
                _RowLocator(entry=entry, key_row=key_row)
            )
            pool.resident_row_locators_current += 1
        pool.resident_row_locators_high_water = max(
            pool.resident_row_locators_high_water,
            pool.resident_row_locators_current,
        )

    @classmethod
    def _detach_lookahead_entry_locked(
        cls,
        pool: _PersistentPanelProducerPool,
        entry: _LookaheadEntry,
        *,
        taken: bool = False,
        stale: bool = False,
    ) -> bool:
        """Detach one entry and transfer its single permit to the caller."""

        if entry.registered:
            pool.reserved_bytes_current -= int(entry.reserved_bytes)
            entry.reserved_bytes = 0
            bucket = pool.entries.get(entry.key)
            if bucket and entry in bucket:
                bucket.remove(entry)
                if not bucket:
                    pool.entries.pop(entry.key, None)
            pool.live_order.pop(id(entry), None)
            for key_row in range(len(entry.key)):
                cls._remove_row_locator_locked(pool, entry, key_row)
            pool.resident_entries_current -= 1
            pool.resident_bytes_current -= int(entry.resident_bytes)
            entry.resident_bytes = 0
            entry.registered = False
        entry.taken = bool(entry.taken or taken)
        entry.stale = bool(entry.stale or stale)
        if entry.permit_release_scheduled:
            return False
        entry.permit_release_scheduled = True
        return True

    @classmethod
    def _best_bucket_entry_locked(
        cls, bucket: Sequence[_LookaheadEntry] | None
    ) -> _LookaheadEntry | None:
        eligible = [
            entry
            for entry in (bucket or ())
            if not entry.taken
            and not entry.stale
            and not entry.retire_pending
            and entry.serving_claims == 0
            and entry.delivery_holds == 0
            and not entry.consumed_rows
            and not entry.claimed_rows
            and cls._lookahead_future_state(entry) != "cancelled"
        ]
        return min(eligible, key=cls._lookahead_entry_rank, default=None)

    @classmethod
    def _remove_consumed_row_locators_locked(
        cls,
        pool: _PersistentPanelProducerPool,
        entry: _LookaheadEntry,
        key_rows: Sequence[int],
    ) -> None:
        for key_row in key_rows:
            cls._remove_row_locator_locked(pool, entry, int(key_row))

    @staticmethod
    def _lookahead_entry_settled_rows(entry: _LookaheadEntry) -> set[int]:
        """Rows no longer live for entry-lifetime accounting."""

        return set(entry.consumed_rows or ()).union(entry.generation_retired_rows or ())

    @classmethod
    def _lookahead_entry_fully_settled(cls, entry: _LookaheadEntry) -> bool:
        return len(cls._lookahead_entry_settled_rows(entry)) == len(entry.key)

    def _lookahead_payload_residency(self, payload: Any) -> tuple[int, tuple[int, ...]]:
        """Return actual host bytes and device-resident row positions.

        Pin-split payloads may contain rows whose bytes live only in the
        on-target pin bank.  Those rows are still row-addressable: the
        locator points to the immutable payload and the salvage assembler
        later materializes the row as a pin slot in a fresh
        ``_PinSplitPayload``.  The returned positions are therefore
        diagnostic only; they are not removed from the row index.
        """

        if isinstance(payload, _PinSplitPayload):
            tensors = (payload.host_images, payload.host_labels)
            host_positions = (
                set(int(value) for value in payload.host_positions)
                if payload.host_images is not None and payload.host_labels is not None
                else set()
            )
            device_rows = tuple(
                row for row in range(len(payload.context_ids)) if row not in host_positions
            )
        else:
            _ids, images, labels, _loaded_bytes = payload
            tensors = (images, labels)
            device_rows = ()
        resident_bytes = sum(
            0 if tensor is None else self._tensor_bytes(tensor) for tensor in tensors
        )
        return int(resident_bytes), device_rows

    def _lookahead_gather_recorded(
        self,
        pool: _PersistentPanelProducerPool,
        entry: _LookaheadEntry,
    ) -> tuple[tuple[tuple[str, ...], Any, Any, int], float]:
        """Gather and bind actual row/byte residency before readiness."""

        result = self._lookahead_gather(entry.key, int(entry.parallel_shards))
        payload, _seconds = result
        resident_bytes, _device_rows = self._lookahead_payload_residency(payload)
        with pool.lock:
            if entry.registered and not entry.taken and not entry.stale:
                entry.resident_bytes = int(resident_bytes)
                pool.resident_bytes_current += int(resident_bytes)
                pool.resident_bytes_high_water = max(
                    pool.resident_bytes_high_water,
                    pool.resident_bytes_current,
                )
                # Device-only pin rows remain in the locator index.  The
                # row-source type is resolved during salvage assembly, never
                # by copying bytes into the locator itself.
        return result

    def _promote_speculative_rows_locked(
        self,
        pool: _PersistentPanelProducerPool,
        requested: Sequence[str],
        post_id: int,
    ) -> None:
        """Promote every speculative entry overlapping exact demand rows."""

        promoted: dict[int, _LookaheadEntry] = {}
        promoted_rows: set[str] = set()
        for context_id in requested:
            for locator in tuple(pool.row_index.get(context_id, ())):
                entry = locator.entry
                if (
                    entry.speculative
                    and entry.registered
                    and not entry.taken
                    and not entry.stale
                    and (entry.consumed_rows is None or locator.key_row not in entry.consumed_rows)
                ):
                    promoted[id(entry)] = entry
                    promoted_rows.add(context_id)
        for entry in promoted.values():
            entry.speculative = False
            entry.demand_confirmed = True
            entry.post_id = int(post_id)
            pool.speculative_confirmed += 1
            pool.demand_promotions += 1
            pool.live_order.pop(id(entry), None)
            pool.live_order[id(entry)] = entry
            future = entry.future
            # A queued speculative task can be cancelled and reinserted at
            # exact priority without touching its one residency permit.
            if future is not None and future.cancel():
                entry.future = pool.gather_executor.submit(
                    self._lookahead_gather_recorded,
                    pool,
                    entry,
                    priority=_GATHER_PRIORITY_EXACT,
                )
                pool.queued_speculative_promoted += 1
        pool.demand_promoted_rows += len(promoted_rows)

    def _reconcile_prediction_generation_locked(
        self,
        pool: _PersistentPanelProducerPool,
        *,
        prediction_generation_id: str,
        role: str,
        requested: Sequence[str],
        post_id: int,
    ) -> list[_LookaheadEntry]:
        """Resolve one provider-bound speculative prediction completely.

        Exact-overlap promotion always happens before non-overlap retirement.
        Only entries carrying the exact opaque generation AND physical role
        participate.  Whole entries are detached under the pool lock and
        returned for callback-safe physical cleanup outside it.  Partial
        entries retain their one payload/permit while unmatched locators are
        removed and counted as settled rows.
        """

        generation = str(prediction_generation_id)
        physical_role = str(role)
        requested_set = set(str(value) for value in requested)
        matching = [
            entry
            for entry in tuple(pool.live_order.values())
            if entry.registered
            and not entry.taken
            and not entry.stale
            and entry.origin_speculative
            and entry.prediction_generation_id == generation
            and entry.prediction_role == physical_role
        ]
        pool.prediction_generation_role_mismatch_refusals += sum(
            1
            for entry in pool.live_order.values()
            if entry.registered
            and not entry.taken
            and not entry.stale
            and entry.origin_speculative
            and entry.prediction_generation_id == generation
            and entry.prediction_role != physical_role
        )
        mismatched = sum(
            1
            for entry in pool.live_order.values()
            if entry.registered
            and not entry.taken
            and not entry.stale
            and entry.origin_speculative
            and (
                entry.prediction_generation_id != generation
                or entry.prediction_role != physical_role
            )
        )
        pool.prediction_generation_mismatched_entries_untouched += mismatched
        if not matching:
            pool.prediction_generation_unmatched_noops += 1
            return []

        pool.prediction_generations_reconciled += 1
        pool.prediction_generation_entries_reconciled += len(matching)
        detached: list[_LookaheadEntry] = []
        for entry in matching:
            live_rows = [
                key_row
                for key_row in range(len(entry.key))
                if key_row not in self._lookahead_entry_settled_rows(entry)
            ]
            confirmed_rows = [
                key_row for key_row in live_rows if entry.key[key_row] in requested_set
            ]
            unmatched_rows = [
                key_row for key_row in live_rows if entry.key[key_row] not in requested_set
            ]

            # Promotion precedes retirement, including for partially
            # overlapping chunks whose tuple identity differs from demand.
            if confirmed_rows:
                was_speculative = entry.speculative
                entry.speculative = False
                entry.demand_confirmed = True
                entry.post_id = int(post_id)
                if was_speculative:
                    pool.speculative_confirmed += 1
                    pool.demand_promotions += 1
                pool.demand_promoted_rows += len(confirmed_rows)
                pool.live_order.pop(id(entry), None)
                pool.live_order[id(entry)] = entry
                future = entry.future
                if future is not None and future.cancel():
                    entry.future = pool.gather_executor.submit(
                        self._lookahead_gather_recorded,
                        pool,
                        entry,
                        priority=_GATHER_PRIORITY_EXACT,
                    )
                    pool.queued_speculative_promoted += 1

            if unmatched_rows:
                if entry.generation_retired_rows is None:
                    entry.generation_retired_rows = set()
                for key_row in unmatched_rows:
                    self._remove_row_locator_locked(pool, entry, key_row)
                entry.generation_retired_rows.update(unmatched_rows)
                pool.prediction_generation_rows_retired += len(unmatched_rows)

                if confirmed_rows:
                    pool.prediction_generation_partial_entries += 1

            if not self._lookahead_entry_fully_settled(entry):
                continue
            pool.prediction_generation_entries_retired += 1
            if entry.serving_claims or entry.delivery_holds:
                entry.retire_pending = True
                pool.prediction_generation_deferred_by_claims_or_holds += 1
                continue
            if self._detach_lookahead_entry_locked(pool, entry, stale=True):
                detached.append(entry)
        return detached

    def post_upcoming_look(
        self,
        context_ids: Sequence[str],
        *,
        role: str,
        batch_rows: int | None = None,
        speculative: bool = False,
        prediction_generation_id: str | None = None,
        complete_prediction_resolution: bool = False,
        branch_prediction: bool = False,
    ) -> None:
        """Enqueue a known upcoming look's gathers behind current serving.

        Runtime overlap only: the caller (the provider, at decision start
        or look extension) supplies the already-determined physical source
        list in the order the serve path will request it.  Chunking uses
        the identical ``_chunks``/role-row resolution the serve path uses,
        and the pre-gather runs the identical ``_gather_bank_chunk`` — a
        ready-buffer hit therefore delivers byte-identical host tensors at
        an earlier wall-clock time, nothing else.  Never raises; every
        failure degrades to on-demand gathering with a receipted reason.

        Lifetime is CONSUMPTION-KEYED (t25b): a pre-gathered entry lives
        until it is served (or salvage-exhausted) and is evicted only
        when a NEW post needs its budget slot, oldest-post-first.  The
        lib3t23/lib3t25 receipts measured the previous post-count TTL
        discarding 90-95% of gathered bytes unconsumed — the residency
        bound was always the entry-count semaphore, never the aging rule.

        ``speculative`` marks rung-1 posts whose consumption is uncertain.
        Chunks already resident in the buffer are never gathered twice: an
        determined (non-speculative) post CONFIRMS a matching live entry
        (refreshing its eviction position), while a speculative post
        simply skips chunks that are already resident.  A nonempty opaque
        ``prediction_generation_id`` may bind a speculative post to one
        provider-owned prediction for this physical role.  A later exact post
        repeats that identity and sets ``complete_prediction_resolution`` to
        retire only that generation's proved-unrealized rows.  Omitting the
        identity keeps the entry until it is consumed or its budget slot is needed.
        """

        if self._device_gather is not None:
            if speculative and not branch_prediction and self._device_gather.worker_residency:
                return
            if context_ids:
                if self._device_gather.worker_residency:
                    if str(role) == "category_calibration":
                        self._device_gather.retain(
                            "category_calibration", tuple(int(self._records[str(value)].ordinal) for value in context_ids),
                        )
                    elif _physical_native_panel_role(str(role)) == "terminal":
                        self._device_gather.retain("validation", range(len(self._bank_inputs)))
                    else:
                        self.request(context_ids)
                else:
                    self.install_device_row_order(f"posted:{role}", context_ids)
            return
        pool = self._persistent_producer_pool()
        if pool is None or pool.lookahead_looks <= 0:
            return
        try:
            if not self.optimized_bank:
                pool.record_degradation("lookahead_requires_optimized_bank")
                return
            generation_id = (
                None
                if prediction_generation_id is None
                else str(prediction_generation_id).strip()
            )
            if prediction_generation_id is not None and not generation_id:
                with pool.lock:
                    pool.prediction_generation_invalid_identity_refusals += 1
                return
            if speculative and complete_prediction_resolution:
                with pool.lock:
                    pool.prediction_generation_invalid_resolution_mode_refusals += 1
                return
            if complete_prediction_resolution and generation_id is None:
                with pool.lock:
                    pool.prediction_generation_invalid_identity_refusals += 1
                return
            raw_requested = tuple(str(value) for value in context_ids)
            if (
                not raw_requested
                and not speculative
                and complete_prediction_resolution
                and generation_id is not None
            ):
                # A provider may prove that a prediction resolves to NO next
                # look (for example, the current fit closes).  This narrow
                # control-plane resolution must not make empty scientific
                # panels valid anywhere else.
                requested = ()
            else:
                requested = self._validate_contexts(raw_requested)
            semantic_role = str(role)
            rows = (
                self._resolve_role_rows(
                    semantic_role, _physical_native_panel_role(semantic_role)
                )
                if batch_rows is None
                else int(batch_rows)
            )
            if rows <= 0:
                raise ValueError("posted look batch_rows must be positive")
            chunks = () if not requested else self._chunks(requested, rows)
            # Mirror the serve path's sole-gather sharding so a one-chunk
            # look pre-gathers at the same internal parallelism (bytes are
            # shard-invariant; only the gather's duration differs).
            parallel_shards = int(self.spec.producer_workers) if len(chunks) == 1 else 1
            to_enqueue: list[tuple[str, ...]] = []
            generation_retire: list[_LookaheadEntry] = []
            with pool.lock:
                post_id = pool.post_counter
                pool.post_counter += 1
                if not speculative and complete_prediction_resolution:
                    generation_retire = self._reconcile_prediction_generation_locked(
                        pool,
                        prediction_generation_id=generation_id,
                        role=str(role),
                        requested=requested,
                        post_id=post_id,
                    )
                elif not speculative:
                    # Confirmation is row-addressed, not tuple-addressed:
                    # shifted chunk boundaries still promote every shared
                    # speculative gather ahead of unrelated speculation.
                    self._promote_speculative_rows_locked(pool, requested, post_id)
                for chunk in chunks:
                    key = tuple(chunk)
                    bucket = pool.entries.get(key)
                    resident = self._best_bucket_entry_locked(bucket)
                    if (
                        complete_prediction_resolution
                        and resident is not None
                        and resident.origin_speculative
                        and (
                            resident.prediction_generation_id != generation_id
                            or resident.prediction_role != str(role)
                        )
                    ):
                        # Resolution authority is generation+role scoped.  A
                        # same-key entry owned by another prediction cannot be
                        # silently confirmed; exact supply is enqueued beside
                        # it (or serves on demand if budget-bound).
                        resident = None
                    if resident is None:
                        if complete_prediction_resolution and all(
                            any(
                                locator.entry.registered
                                and not locator.entry.taken
                                and not locator.entry.stale
                                and not locator.entry.retire_pending
                                and locator.entry.demand_confirmed
                                and locator.entry.origin_speculative
                                and locator.entry.prediction_generation_id
                                == generation_id
                                and locator.entry.prediction_role == str(role)
                                and (
                                    locator.entry.consumed_rows is None
                                    or locator.key_row not in locator.entry.consumed_rows
                                )
                                and (
                                    locator.entry.generation_retired_rows is None
                                    or locator.key_row
                                    not in locator.entry.generation_retired_rows
                                )
                                for locator in pool.row_index.get(context_id, ())
                            )
                            for context_id in key
                        ):
                            # Shifted exact chunks are already fully covered by
                            # this generation's promoted row-addressed donors;
                            # enqueueing duplicate bytes would defeat the
                            # reconciliation's memory bound.
                            pool.deduplicated_chunks += 1
                            continue
                        to_enqueue.append(key)
                        continue
                    pool.deduplicated_chunks += 1
                    if speculative:
                        # A speculative re-post never steals a live entry.
                        continue
                    # Exact post confirms the resident entry and refreshes
                    # its eviction position — demand keeps it warm.
                    resident.post_id = post_id
                    resident.demand_confirmed = True
                    if resident.speculative:
                        resident.speculative = False
                        pool.speculative_confirmed += 1
                    if id(resident) in pool.live_order:
                        pool.live_order.pop(id(resident))
                        pool.live_order[id(resident)] = resident
            # Release/cancel whole-generation misses before exact demand asks
            # for permits.  Running gathers retain their permit until their
            # existing completion callback releases the payload safely.
            for entry in generation_retire:
                self._discard_lookahead_entry(pool, entry)
            budget_exhausted = False
            fresh = 0
            for key in to_enqueue:
                if not self._reserve_lookahead_slot(pool, post_id, speculative=bool(speculative)):
                    budget_exhausted = True
                    break
                reservation = int(len(key)) * (
                    int(np.prod(self._bank_inputs.shape[1:])) + 8
                )
                if pool.resident_byte_limit is not None:
                    with pool.lock:
                        byte_room = (
                            pool.reserved_bytes_current + reservation
                            <= pool.resident_byte_limit
                        )
                        if byte_room:
                            pool.reserved_bytes_current += reservation
                            pool.reserved_bytes_high_water = max(
                                pool.reserved_bytes_high_water,
                                pool.reserved_bytes_current,
                            )
                    if not byte_room:
                        pool.release_budget()
                        budget_exhausted = True
                        break
                entry = _LookaheadEntry(
                    key=key,
                    post_id=post_id,
                    speculative=bool(speculative),
                    origin_speculative=bool(speculative),
                    prediction_generation_id=(
                        generation_id if speculative else None
                    ),
                    prediction_role=(
                        str(role)
                        if speculative and generation_id is not None
                        else None
                    ),
                    parallel_shards=int(parallel_shards),
                    reserved_bytes=(
                        reservation if pool.resident_byte_limit is not None else 0
                    ),
                )
                submission_error: BaseException | None = None
                release_failed_submission = False
                with pool.lock:
                    self._register_lookahead_entry_locked(pool, entry)
                    try:
                        entry.future = pool.gather_executor.submit(
                            self._lookahead_gather_recorded,
                            pool,
                            entry,
                            priority=(
                                _GATHER_PRIORITY_SPECULATIVE
                                if speculative
                                else _GATHER_PRIORITY_EXACT
                            ),
                        )
                    except BaseException as error:
                        submission_error = error
                        release_failed_submission = self._detach_lookahead_entry_locked(
                            pool, entry, stale=True
                        )
                if submission_error is not None:
                    if release_failed_submission:
                        pool.release_budget()
                    pool.record_degradation(
                        "lookahead_submit_failed:"
                        f"{type(submission_error).__name__}:"
                        f"{submission_error}"
                    )
                    break
                fresh += 1
            with pool.lock:
                pool.posted_look_count += 1
                pool.enqueued += fresh
                if speculative:
                    pool.speculative_posted += fresh
                if budget_exhausted:
                    pool.budget_exhausted_posts += 1
            if budget_exhausted:
                pool.record_degradation("lookahead_budget_exhausted")
        except Exception as error:  # never fail the caller's arm
            pool.record_degradation(f"post_failed:{type(error).__name__}:{error}")

    def _reserve_lookahead_slot(
        self,
        pool: _PersistentPanelProducerPool,
        post_id: int,
        *,
        speculative: bool,
    ) -> bool:
        """Fund one pre-gather permit, evicting old supply for new demand.

        The budget semaphore is the sole live-ENTRY bound (not a byte
        bound; actual host bytes are receipted).  When it is
        full, the oldest live entry from an OLDER post whose gather is
        not RUNNING (queued → cancellable, done → releasable) is evicted
        to fund the new post — newest demand replaces stalest supply,
        the LRU discipline.  Entries of the posting look itself are
        never victims (a look must not cannibalize its own serve order),
        and a budget fully held by in-flight gathers simply
        back-pressures the post (receipted), exactly as before.
        """

        if pool.try_reserve_budget():
            return True
        victim: _LookaheadEntry | None = None
        with pool.lock:
            candidates = [
                entry
                for entry in pool.live_order.values()
                if entry.post_id < int(post_id)
                and entry.registered
                and not entry.taken
                and not entry.stale
                and entry.serving_claims == 0
                and entry.delivery_holds == 0
                and entry.future is not None
                and not entry.future.running()
                and (not speculative or entry.speculative)
            ]
            if candidates:
                # Exact demand evicts speculation first (queued before done),
                # then old exact supply.  A speculative post may replace
                # only speculation; it can never evict exact demand.
                def eviction_rank(entry: _LookaheadEntry) -> tuple[int, int, int]:
                    return (
                        0 if entry.speculative else 1,
                        0 if self._lookahead_future_state(entry) == "queued" else 1,
                        int(entry.post_id),
                    )

                victim = min(candidates, key=eviction_rank)
            if victim is None:
                if speculative and any(
                    entry.registered and not entry.speculative for entry in pool.live_order.values()
                ):
                    pool.speculative_budget_refusals_preserving_exact += 1
                return False
            self._detach_lookahead_entry_locked(pool, victim, stale=True)
            pool.demand_evicted_chunks += 1
            if not speculative and victim.speculative:
                pool.speculative_evicted_for_demand += 1
        self._discard_lookahead_entry(pool, victim)
        return pool.try_reserve_budget()

    def _lookahead_gather(
        self, chunk: tuple[str, ...], parallel_shards: int
    ) -> tuple[tuple[tuple[str, ...], Any, Any, int], float]:
        """Pre-gather one chunk; return exactly the serve path's payload."""

        gather_started = time.perf_counter()
        payload = self._gather_bank_chunk(chunk, parallel_shards=int(parallel_shards))
        return payload, time.perf_counter() - gather_started

    def _resident_pinned_slice(
        self,
        pool: _PersistentPanelProducerPool,
        entry: _LookaheadEntry,
        want: tuple[str, ...],
        key_rows: Sequence[int],
        payload: Any,
    ) -> _ResidentPinnedSlicePayload | None:
        """Borrow an ordered contiguous view of one resident pinned entry.

        The fast path is deliberately narrow.  Requested rows must form a
        strictly increasing unit-stride interval.  Plain payloads expose
        direct tensor views.  Pin-split payloads rebase the interval's
        existing host/pin positions; their compact host indices must also be
        a unit-stride interval, so the host tensors remain storage-sharing
        views.  The caller must already own the entry's serving claim; this
        method atomically converts that claim to one delivery hold.
        """

        if not key_rows:
            return None
        rows = tuple(int(value) for value in key_rows)
        start = rows[0]
        stop = rows[-1] + 1
        if rows != tuple(range(start, stop)):
            return None
        if tuple(entry.key[start:stop]) != tuple(want):
            return None

        pin_split = None
        if isinstance(payload, _PinSplitPayload):
            owner_images = payload.host_images
            owner_labels = payload.host_labels
            selected_host = tuple(
                (compact_row, int(key_row))
                for compact_row, key_row in enumerate(payload.host_positions)
                if start <= int(key_row) < stop
            )
            selected_pin = tuple(
                (int(key_row), int(payload.pin_slots[pin_row]))
                for pin_row, key_row in enumerate(payload.pin_positions)
                if start <= int(key_row) < stop
            )
            if len(selected_host) + len(selected_pin) != len(want):
                return None
            if selected_host:
                compact_rows = tuple(item[0] for item in selected_host)
                compact_start = compact_rows[0]
                compact_stop = compact_rows[-1] + 1
                if compact_rows != tuple(range(compact_start, compact_stop)):
                    return None
                if owner_images is None or owner_labels is None:
                    return None
                host_images = owner_images[compact_start:compact_stop]
                host_labels = owner_labels[compact_start:compact_stop]
            else:
                host_images = None
                host_labels = None
            pin_split = _PinSplitPayload(
                context_ids=tuple(want),
                host_images=host_images,
                host_labels=host_labels,
                host_positions=tuple(key_row - start for _compact_row, key_row in selected_host),
                pin_positions=tuple(key_row - start for key_row, _slot in selected_pin),
                pin_slots=tuple(slot for _key_row, slot in selected_pin),
                loaded_bytes=(0 if host_images is None else int(host_images.numel())),
                # The owner's compact host digests, rebased to the
                # served interval (pinned rows are certified by the pin bank).
                host_row_digests=(
                    None
                    if not selected_host or payload.host_row_digests is None
                    else np.asarray(payload.host_row_digests, dtype=np.int64)[
                        compact_start:compact_stop
                    ]
                ),
            )
        else:
            try:
                _ids, owner_images, owner_labels, _loaded_bytes = payload
            except (TypeError, ValueError):
                return None
            if owner_images is None or owner_labels is None:
                return None
            host_images = owner_images[start:stop]
            host_labels = owner_labels[start:stop]

        if owner_images is not None or owner_labels is not None:
            if owner_images is None or owner_labels is None:
                return None
            with self._pinned_ring_lock:
                image_state = self._pinned_slot_state.get(id(owner_images))
                label_state = self._pinned_slot_state.get(id(owner_labels))
                eligible = (
                    id(owner_images) in self._pinned_slot_ids
                    and id(owner_labels) in self._pinned_slot_ids
                    and image_state is not None
                    and label_state is not None
                    and bool(image_state.get("leased"))
                    and bool(label_state.get("leased"))
                    and image_state.get("staged_checksum") is not None
                    and label_state.get("staged_checksum") is not None
                )
            if not eligible:
                return None

        # Checksums are slice-scoped: the owner's full-slot checksum cannot
        # certify a view, and a view id is intentionally absent from the
        # pinned-ring registry. Their device disagreement flags settle after
        # delivery, together with the other panel checks.
        image_checksum = None if host_images is None else self._host_delivery_checksum(host_images)
        label_checksum = None if host_labels is None else self._host_delivery_checksum(host_labels)
        with pool.lock:
            if not entry.registered or entry.taken or entry.stale or entry.serving_claims <= 0:
                return None
            entry.delivery_holds += 1
        owner_digests = None if pin_split is not None else _payload_row_digests(payload)
        return _ResidentPinnedSlicePayload(
            context_ids=tuple(want),
            host_images=host_images,
            host_labels=host_labels,
            loaded_bytes=(0 if host_images is None else int(host_images.numel())),
            entry=entry,
            pool=pool,
            source_payload=payload,
            owner_images=owner_images,
            owner_labels=owner_labels,
            image_checksum=(None if image_checksum is None else int(image_checksum)),
            label_checksum=(None if label_checksum is None else int(label_checksum)),
            pin_split=pin_split,
            # The owner's digests restricted to the served interval.
            row_digests=(
                None
                if owner_digests is None
                else np.asarray(owner_digests, dtype=np.int64)[start:stop]
            ),
        )

    def _settle_resident_pinned_slice(
        self,
        borrowed: _ResidentPinnedSlicePayload,
        *,
        ready_event: Any | None,
    ) -> None:
        """Transfer a borrowed slice's lifetime to H2D, exactly once."""

        if borrowed.settled:
            raise RuntimeError("resident pinned slice settled more than once")
        borrowed.settled = True
        release = False
        release_event = None
        with borrowed.pool.lock:
            entry = borrowed.entry
            if entry.delivery_holds <= 0:
                raise RuntimeError("resident pinned slice delivery-hold underflow")
            if ready_event is not None:
                entry.delivery_event = ready_event
            entry.delivery_holds -= 1
            if entry.delivery_holds == 0 and entry.serving_claims == 0:
                if entry.retire_pending or self._lookahead_entry_fully_settled(entry):
                    release = self._detach_lookahead_entry_locked(borrowed.pool, entry, taken=True)
                    release_event = entry.delivery_event
        if release:
            if release_event is not None:
                with self._pinned_ring_lock:
                    for owner in (
                        borrowed.owner_images,
                        borrowed.owner_labels,
                    ):
                        if owner is None:
                            continue
                        state = self._pinned_slot_state.get(id(owner))
                        if state is None or not state.get("leased"):
                            raise RuntimeError(
                                "resident pinned slice owner lost its lease before H2D"
                            )
                        state["event"] = release_event
                # The allocator frees these owners once ``release_event``
                # completes.  Do not also release them here: after the
                # event completes another plane sharing the pool may lease
                # the same slot, and a release now would clear that new
                # lease (its state has no event yet).
            else:
                self._release_lookahead_payload(borrowed.source_payload)
            borrowed.pool.release_budget()

    def _take_lookahead_chunk(
        self,
        pool: _PersistentPanelProducerPool,
        chunk: tuple[str, ...],
    ) -> tuple[tuple[tuple[str, ...], Any, Any, int], float] | None:
        """Consume a ready-buffer entry for ``chunk``; None means on-demand.

        Deadlock guard: this runs on a pool worker, so it only blocks on
        futures that are RUNNING (their worker cannot depend on this one);
        a still-QUEUED future behind other pool work is cancelled and the
        chunk falls back to the on-demand gather.
        """

        key = tuple(chunk)
        with pool.lock:
            bucket = pool.entries.get(key)
            entry = self._best_bucket_entry_locked(bucket)
            if entry is not None:
                self._detach_lookahead_entry_locked(pool, entry, taken=True)
        if entry is None:
            # Exact-key miss.  Boundary-shifted posts (Neyman-moved
            # allocations, slab drift, speculative size guesses) still hold
            # the SAME immutable bank rows — assemble the served chunk from
            # completed resident entries byte-for-byte before giving up.
            # Salvage has not been seen firing in the
            # canaries, so it is OPT-IN
            # (IRG_ROW_SALVAGE=1) rather than dark — campaign arms have
            # no reference masks, so an unexercised byte-sourcing path
            # must not ride by default.  The miss-classification
            # diagnostic (proven live) runs regardless.
            if self._row_salvage_env:
                salvaged = self._try_salvage_chunk(pool, key)
                if salvaged is not None:
                    return salvaged
            if pool.posted_look_count > 0:
                with pool.lock:
                    pool.on_demand_misses += 1
                self._classify_lookahead_miss(pool, key)
            return None
        future = entry.future
        if not future.done() and not future.running() and future.cancel():
            pool.release_budget()
            with pool.lock:
                pool.on_demand_misses += 1
                pool.queued_lookahead_cancelled_for_demand += 1
            pool.record_degradation("lookahead_chunk_still_queued")
            return None
        was_ready = bool(future.done())
        was_running = bool(future.running())
        wait_started = time.perf_counter()
        try:
            payload, gather_seconds = future.result()
        except BaseException as error:
            pool.release_budget()
            with pool.lock:
                pool.on_demand_misses += 1
            pool.record_degradation(f"lookahead_gather_failed:{type(error).__name__}:{error}")
            return None
        waited = time.perf_counter() - wait_started
        pool.release_budget()
        with pool.lock:
            pool.ready_hits += 1
            if was_ready:
                pool.demand_ready_rows += len(key)
            elif was_running:
                pool.demand_waited_rows += len(key)
            if entry.speculative:
                pool.speculative_consumed += 1
            # Residual exposure: seconds the serve path still waited on a
            # not-yet-finished lookahead gather (0.0 when fully hidden).
            pool.exposed_wait_seconds += float(waited)
        return payload, gather_seconds

    def _try_salvage_chunk(
        self,
        pool: _PersistentPanelProducerPool,
        chunk: tuple[str, ...],
    ) -> tuple[Any, float] | None:
        """Assemble a boundary-shifted served chunk from pre-gathered rows.

        The measured residual wait is exact-key misses whose ROWS are
        largely resident (Neyman-moved allocation sizes, slab drift, and
        speculative size guesses shift chunk BOUNDARIES, not identities).
        Bank rows are immutable, so a served chunk assembled from
        COMPLETED entry payloads is byte-for-byte the on-demand gather of
        that chunk.  Host-sourced rows are compacted into a staged host
        tensor; device-pinned rows are represented as pin slots in a fresh
        ``_PinSplitPayload`` and interleaved by ``_device_batch_pinned``.
        All-device salvage therefore allocates no placeholder host tensor.
        Delivery order and grouping are the serve path's — only where the
        bytes came from changes.  Metadata selection and serving claims are
        atomic under the pool lock; no future wait, allocation, tensor copy,
        mmap gather, payload release, or semaphore operation occurs under
        that lock.  Claimed entries cannot be evicted until assembly settles.
        Entries retire (lease + budget returned exactly once) when every row
        is consumed.  Declines (None → plain sharded on-demand gather) when
        no rows are resident or when resident rows are a minority of the
        chunk (the cost rule).
        """

        framework = _require_torch()
        want = tuple(chunk)
        salvage_started = time.perf_counter()
        locator_options: dict[str, tuple[_RowLocator, ...]] = {}
        claim_rows: dict[int, tuple[_LookaheadEntry, set[int]]] = {}
        with pool.lock:
            pool.row_index_requests += len(want)
            selected_locator_ids: set[int] = set()
            for context_id in want:
                eligible = tuple(
                    locator
                    for locator in sorted(
                        pool.row_index.get(context_id, ()),
                        key=lambda candidate: self._lookahead_entry_rank(candidate.entry),
                    )
                    if not locator.entry.taken
                    and not locator.entry.stale
                    and not locator.entry.retire_pending
                    and locator.entry.future is not None
                    and locator.entry.future.done()
                    and (
                        locator.entry.consumed_rows is None
                        or locator.key_row not in locator.entry.consumed_rows
                    )
                    and (
                        locator.entry.claimed_rows is None
                        or locator.key_row not in locator.entry.claimed_rows
                    )
                )
                if not eligible:
                    continue
                locator_options[context_id] = eligible
                # Claim only the first-ranked locator.  Claiming every
                # duplicate would transiently block eviction across the
                # entry-bounded buffer; if this source cannot decode, the
                # serve path falls back instead of widening the claim.
                locator = eligible[0]
                selected_locator_ids.add(id(locator))
                entry = locator.entry
                claimed = claim_rows.setdefault(id(entry), (entry, set()))[1]
                claimed.add(int(locator.key_row))
            for entry, key_rows in claim_rows.values():
                entry.serving_claims += 1
                if entry.claimed_rows is None:
                    entry.claimed_rows = set()
                entry.claimed_rows.update(key_rows)

        # From here through assembly there is no pool lock.  Serving claims
        # keep source payloads resident while completed futures are observed
        # and tensors are copied.
        coverage: dict[str, tuple[_LookaheadEntry, str, int, int]] = {}
        payload_of: dict[int, tuple[Any, Any]] = {}
        raw_payload_of: dict[int, Any] = {}
        rows_lookup_of: dict[int, dict[str, tuple[str, int, int]]] = {}
        retire: list[tuple[_LookaheadEntry, Any]] = []
        host_groups: dict[int, tuple[_LookaheadEntry, list[int], list[int]]] = {}
        consumed_key_rows: dict[int, tuple[_LookaheadEntry, set[int]]] = {}
        host_positions: list[int] = []
        pin_positions: list[int] = []
        pin_slots: list[int] = []
        missing_positions: list[int] = []
        missing: tuple[str, ...] = want
        salvaged_row_count = 0
        pin_salvaged_row_count = 0
        pin_topup_row_count = 0
        speculative_source = False
        assembly_succeeded = False
        assembly_error: BaseException | None = None
        low_coverage_declined = False
        assembled_payload: Any | None = None
        direct_pinned_assembly = False
        direct_pinned_assembly_bytes = 0
        assembly_host_images = None
        assembly_host_labels = None
        borrowed_slice: _ResidentPinnedSlicePayload | None = None
        try:
            for entry, _claimed_rows in claim_rows.values():
                try:
                    payload, _seconds = entry.future.result()
                except BaseException:
                    continue
                raw_payload_of[id(entry)] = payload
                if isinstance(payload, _PinSplitPayload):
                    rows_lookup: dict[str, tuple[str, int, int]] = {}
                    for host_index, key_position in enumerate(payload.host_positions):
                        rows_lookup[payload.context_ids[key_position]] = (
                            "host",
                            int(host_index),
                            int(key_position),
                        )
                    for pin_index, key_position in enumerate(payload.pin_positions):
                        rows_lookup[payload.context_ids[key_position]] = (
                            "pin",
                            int(payload.pin_slots[pin_index]),
                            int(key_position),
                        )
                    images = payload.host_images
                    labels = payload.host_labels
                else:
                    _ids, images, labels, _nbytes = payload
                    rows_lookup = {
                        context_id: ("host", int(row), int(row))
                        for row, context_id in enumerate(entry.key)
                    }
                payload_of[id(entry)] = (images, labels)
                rows_lookup_of[id(entry)] = rows_lookup
            for context_id in want:
                for locator in locator_options.get(context_id, ()):
                    if id(locator) not in selected_locator_ids:
                        continue
                    located = rows_lookup_of.get(id(locator.entry), {}).get(context_id)
                    if located is not None:
                        source_kind, source_row, key_row = located
                        coverage[context_id] = (
                            locator.entry,
                            source_kind,
                            int(source_row),
                            int(key_row),
                        )
                        break
            missing = tuple(context_id for context_id in want if context_id not in coverage)
            if not coverage:
                return None
            if 2 * len(missing) > len(want):
                low_coverage_declined = True
                return None

            # Strict zero-copy joint: every requested row must resolve, in
            # order, to one completed plain pinned entry and form one
            # contiguous source interval.  Anything else follows the proven
            # direct-pinned assembly path below.
            if not missing:
                ordered = tuple(coverage[context_id] for context_id in want)
                source_entries = {id(item[0]) for item in ordered}
                source_kinds = {item[1] for item in ordered}
                key_rows = tuple(int(item[3]) for item in ordered)
                if len(source_entries) == 1 and source_kinds.issubset({"host", "pin"}):
                    source_entry = ordered[0][0]
                    borrowed_slice = self._resident_pinned_slice(
                        pool,
                        source_entry,
                        want,
                        key_rows,
                        raw_payload_of[id(source_entry)],
                    )
                    if borrowed_slice is not None:
                        consumed_key_rows[id(source_entry)] = (
                            source_entry,
                            set(key_rows),
                        )
                        speculative_source = source_entry.speculative
                        salvaged_row_count = len(want)
                        assembled_payload = borrowed_slice
                        assembly_succeeded = True

            topup_images = None
            topup_labels = None
            if borrowed_slice is None and missing:
                # ONE bounded on-demand gather of only the rows no resident
                # source held (a strict minority, by the cost rule above),
                # placed at their served positions — byte-exact by the same
                # immutable-row argument.  Even if a missing row belongs to
                # the pinned class, a missing locator is repaired by exact
                # host top-up rather than by silently inventing coverage.
                topup_images, topup_labels = self._gather_bank_rows(missing, parallel_shards=1)

            host_index_by_position: dict[int, int] = {}
            for position, context_id in enumerate(want if borrowed_slice is None else ()):
                located = coverage.get(context_id)
                if located is None:
                    missing_positions.append(position)
                    host_index_by_position[position] = len(host_positions)
                    host_positions.append(position)
                    if str(context_id) in self._pin_slot_by_context:
                        pin_topup_row_count += 1
                    continue
                entry, source_kind, source_row, key_row = located
                consumed = consumed_key_rows.setdefault(id(entry), (entry, set()))[1]
                consumed.add(int(key_row))
                speculative_source = speculative_source or entry.speculative
                if source_kind == "pin":
                    if self._pin_images is None or self._pin_labels is None:
                        raise RuntimeError("pin row locator observed without a pin bank")
                    pin_positions.append(position)
                    pin_slots.append(int(source_row))
                    pin_salvaged_row_count += 1
                elif source_kind == "host":
                    host_index = len(host_positions)
                    host_index_by_position[position] = host_index
                    host_positions.append(position)
                    group = host_groups.setdefault(id(entry), (entry, [], []))
                    group[1].append(host_index)
                    group[2].append(int(source_row))
                else:
                    raise RuntimeError(f"unknown row salvage source kind {source_kind!r}")
            if borrowed_slice is None:
                salvaged_row_count = len(coverage)

            host_images = None
            host_labels = None
            salvage_image_checksum: int | None = None
            salvage_label_checksum: int | None = None
            salvage_row_digests: np.ndarray | None = None
            if borrowed_slice is None and host_positions:
                reference_images = None
                reference_labels = None
                for entry, _host_indices, _tensor_rows in host_groups.values():
                    images, labels = payload_of[id(entry)]
                    if images is not None and labels is not None:
                        reference_images = images
                        reference_labels = labels
                        break
                if reference_images is None or reference_labels is None:
                    reference_images = topup_images
                    reference_labels = topup_labels
                if reference_images is None or reference_labels is None:
                    raise RuntimeError("host row salvage has no host reference tensor")
                (
                    source_images,
                    source_labels,
                    direct_pinned_assembly,
                ) = self._empty_row_assembly_host(
                    row_count=len(host_positions),
                    reference_images=reference_images,
                    reference_labels=reference_labels,
                )
                assembly_host_images = source_images
                assembly_host_labels = source_labels
                if direct_pinned_assembly:
                    direct_pinned_assembly_bytes = self._tensor_bytes(
                        source_images
                    ) + self._tensor_bytes(source_labels)
                # Vectorized assembly: one index_select/index_copy_ pair per
                # source entry, never a Python loop over resident rows.
                for entry, host_indices, tensor_rows in host_groups.values():
                    images, labels = payload_of[id(entry)]
                    if images is None or labels is None:
                        raise RuntimeError("host row locator observed without host tensors")
                    destination = framework.as_tensor(host_indices, dtype=framework.long)
                    source = framework.as_tensor(tensor_rows, dtype=framework.long)
                    source_images.index_copy_(0, destination, images.index_select(0, source))
                    source_labels.index_copy_(0, destination, labels.index_select(0, source))
                if missing:
                    if topup_images is None or topup_labels is None:
                        raise RuntimeError("missing row top-up was not gathered")
                    for index, position in enumerate(missing_positions):
                        destination = host_index_by_position[position]
                        source_images[destination].copy_(topup_images[index])
                        source_labels[destination] = topup_labels[index]
                (
                    host_images,
                    host_labels,
                    salvage_image_checksum,
                    salvage_label_checksum,
                ) = self._stage_gathered_host_stamped(source_images, source_labels)
                # CLASS3: salvage assembles rows from resident entries.  With
                # the bank row digest table the digests are the
                # table's for the wanted ids, so the assembly itself is
                # certified too; without it they are taken from the ASSEMBLED
                # host rows and certify staging->device only (a
                # documented limit; salvage is opt-in, default OFF).
                if self._delivery_integrity_armed:
                    salvage_row_digests = _lane_g.gather_authority_digests(
                        self,
                        np.fromiter(
                            (int(self._records[str(want[position])].ordinal) for position in host_positions),
                            dtype=np.int64,
                            count=len(host_positions),
                        ),
                    )
                    if salvage_row_digests is None and _lane_g.gather_host_digest_required(self):
                        salvage_row_digests = row_digests_host(host_images)

            if borrowed_slice is not None:
                pass
            elif pin_positions:
                assembled_payload = _PinSplitPayload(
                    context_ids=want,
                    host_images=host_images,
                    host_labels=host_labels,
                    host_positions=tuple(host_positions),
                    pin_positions=tuple(pin_positions),
                    pin_slots=tuple(pin_slots),
                    loaded_bytes=(0 if host_images is None else int(host_images.numel())),
                    host_row_digests=salvage_row_digests,
                    host_image_checksum=salvage_image_checksum,
                    host_label_checksum=salvage_label_checksum,
                )
            else:
                if host_images is None or host_labels is None:
                    raise RuntimeError("host-only salvage produced no host tensor")
                if self._delivery_integrity_armed:
                    assembled_payload = _StagedChunkPayload(
                        want,
                        host_images,
                        host_labels,
                        int(host_images.numel()),
                        row_digests=salvage_row_digests,
                        image_checksum=salvage_image_checksum,
                        label_checksum=salvage_label_checksum,
                    )
                else:
                    assembled_payload = (
                        want,
                        host_images,
                        host_labels,
                        int(host_images.numel()),
                    )
            assembly_succeeded = True
        except BaseException as error:
            assembly_error = error
        finally:
            # Settle claims and consumption atomically, but transfer all
            # physical cleanup out of the pool lock.
            with pool.lock:
                if assembly_succeeded:
                    # A row hit means the located shared row survived all
                    # the way through byte-exact assembly and staging.  A
                    # failed assembly records a degradation but must not
                    # promote tentative locator coverage into a hit.
                    pool.row_index_hits += len(coverage)
                    pool.row_index_misses += len(want) - len(coverage)
                    pool.pin_row_locator_hits += pin_salvaged_row_count
                    pool.pin_row_locator_topup_rows += pin_topup_row_count
                    if direct_pinned_assembly:
                        pool.direct_pinned_assembly_chunks += 1
                        pool.direct_pinned_assembly_rows += len(host_positions)
                        pool.pageable_to_pinned_copy_elided_bytes += int(
                            direct_pinned_assembly_bytes
                        )
                    if borrowed_slice is not None:
                        pool.resident_pinned_slice_chunks += 1
                        pool.resident_pinned_slice_rows += len(want)
                        pool.resident_pinned_slice_host_copy_elided_bytes += (
                            0
                            if borrowed_slice.host_images is None
                            else self._tensor_bytes(borrowed_slice.host_images)
                            + self._tensor_bytes(borrowed_slice.host_labels)
                        )
                        borrowed_pin_split = borrowed_slice.pin_split
                        if borrowed_pin_split is None:
                            pool.resident_pinned_slice_host_rows += len(want)
                        else:
                            pool.resident_pinned_slice_pin_split_chunks += 1
                            pool.resident_pinned_slice_host_rows += len(
                                borrowed_pin_split.host_positions
                            )
                            pool.resident_pinned_slice_pin_rows += len(
                                borrowed_pin_split.pin_positions
                            )
                elif not coverage:
                    pool.row_index_misses += len(want)
                elif low_coverage_declined:
                    pool.row_index_declined_rows += len(want)
                elif assembly_error is not None:
                    pool.row_index_assembly_failed_rows += len(want)
                if low_coverage_declined:
                    pool.salvage_declined_low_coverage += 1
                for entry, claimed_rows in claim_rows.values():
                    consumed = consumed_key_rows.get(id(entry))
                    key_rows = () if consumed is None else consumed[1]
                    if entry.claimed_rows is not None:
                        entry.claimed_rows.difference_update(claimed_rows)
                    entry.serving_claims -= 1
                    if assembly_succeeded:
                        self._remove_consumed_row_locators_locked(pool, entry, key_rows)
                        if entry.consumed_rows is None:
                            entry.consumed_rows = set()
                        entry.consumed_rows.update(key_rows)
                    if (
                        (
                            entry.retire_pending
                            or self._lookahead_entry_fully_settled(entry)
                        )
                        and entry.serving_claims == 0
                        and entry.delivery_holds == 0
                    ):
                        should_release = self._detach_lookahead_entry_locked(
                            pool, entry, taken=True
                        )
                        if should_release:
                            retire.append((entry, raw_payload_of.get(id(entry))))
                if assembly_succeeded:
                    pool.ready_hits += 1
                    pool.salvaged_rows += salvaged_row_count
                    if missing:
                        pool.salvaged_partial_hits += 1
                        pool.topup_rows += len(missing)
                    else:
                        pool.salvaged_hits += 1
                    if speculative_source:
                        pool.speculative_salvaged += 1
            for _entry, payload in retire:
                if payload is not None:
                    self._release_lookahead_payload(payload)
                pool.release_budget()
            if not assembly_succeeded and direct_pinned_assembly:
                self._release_lookahead_payload(
                    (
                        want,
                        assembly_host_images,
                        assembly_host_labels,
                        0,
                    )
                )
            if not assembly_succeeded and borrowed_slice is not None:
                self._settle_resident_pinned_slice(borrowed_slice, ready_event=None)
        if assembly_error is not None:
            pool.record_degradation(
                f"lookahead_row_assembly_failed:{type(assembly_error).__name__}:{assembly_error}"
            )
            return None
        return assembled_payload, time.perf_counter() - salvage_started

    def _device_batch_resident_slice(
        self, borrowed: _ResidentPinnedSlicePayload
    ) -> tuple[
        MeasurementInputBatch,
        int,
        int,
        int,
        Any,
        Any,
        Mapping[str, float],
    ]:
        """Stage a borrowed resident view, then settle its source lifetime."""

        try:
            result = self._device_batch(
                borrowed.context_ids,
                borrowed.host_images,
                borrowed.host_labels,
                pin_split=borrowed.pin_split,
                expected_host_checksums=(
                    borrowed.image_checksum,
                    borrowed.label_checksum,
                ),
                row_digests=borrowed.row_digests,
            )
            ready_event = result[0].ready_event
            if self.device.type == "cuda" and ready_event is None:
                raise RuntimeError("resident pinned slice CUDA staging produced no ready event")
            self._settle_resident_pinned_slice(borrowed, ready_event=ready_event)
            return result
        except BaseException:
            if not borrowed.settled:
                # _device_batch may have enqueued part of the transfer before
                # raising and therefore returned no ready event.  Synchronize
                # the one copy stream before making the backing lease
                # releasable; this is an error-only path, never the fast path.
                if self._copy_stream is not None:
                    self._copy_stream.synchronize()
                self._settle_resident_pinned_slice(borrowed, ready_event=None)
            raise

    def _classify_lookahead_miss(
        self,
        pool: _PersistentPanelProducerPool,
        chunk: tuple[str, ...],
    ) -> None:
        """Receipted miss diagnostic: where did posted and served diverge?

        Bounded work: scans the (budget-bounded) live entries once and
        keeps at most ``_LOOKAHEAD_MISMATCH_SAMPLE_CAP`` detailed samples;
        counters accumulate for every classified miss.  Fail-soft.
        """

        try:
            want = set(chunk)
            with pool.lock:
                covered: set[str] = set()
                best_overlap = 0
                best_width = 0
                live_keys = 0
                for key, bucket in pool.entries.items():
                    if not any(not entry.taken and not entry.stale for entry in bucket):
                        continue
                    live_keys += 1
                    overlap = want.intersection(key)
                    if overlap:
                        covered.update(overlap)
                        if len(overlap) > best_overlap:
                            best_overlap = len(overlap)
                            best_width = len(key)
                if not covered:
                    pool.miss_never_posted += 1
                    classification = "never_posted"
                elif covered == want:
                    pool.miss_covered_salvage_unavailable += 1
                    classification = "covered_salvage_unavailable"
                else:
                    pool.miss_partial_overlap += 1
                    classification = "partial_overlap"
                if len(pool.mismatch_samples) < (_LOOKAHEAD_MISMATCH_SAMPLE_CAP):
                    pool.mismatch_samples.append(
                        {
                            "classification": classification,
                            "served_width": len(chunk),
                            "covered_rows": len(covered),
                            "best_posted_overlap": int(best_overlap),
                            "best_posted_width": int(best_width),
                            "live_posted_keys": int(live_keys),
                            "served_head": tuple(chunk[:2]),
                            "served_tail": tuple(chunk[-2:]),
                        }
                    )
        except Exception:  # diagnostic must never fail serving
            return

    def _evict_all_lookahead_entries(self, pool: _PersistentPanelProducerPool) -> None:
        """Shutdown path: retire every live entry, returning resources."""

        with pool.lock:
            stale: list[_LookaheadEntry] = []
            for entry in tuple(pool.live_order.values()):
                if entry.taken or entry.stale:
                    continue
                if entry.serving_claims or entry.delivery_holds:
                    entry.retire_pending = True
                    continue
                if self._detach_lookahead_entry_locked(pool, entry, stale=True):
                    stale.append(entry)
        for entry in stale:
            self._discard_lookahead_entry(pool, entry)

    def _discard_lookahead_entry(
        self,
        pool: _PersistentPanelProducerPool,
        entry: _LookaheadEntry,
    ) -> None:
        future = entry.future
        if future is None or future.cancel():
            pool.release_budget()
            with pool.lock:
                pool.stale_flushed_chunks += 1
                if entry.speculative:
                    pool.speculative_flushed += 1
            return

        def _release(done_future: Any) -> None:
            try:
                payload, _seconds = done_future.result()
            except BaseException:
                payload = None
            if payload is not None:
                self._release_lookahead_payload(payload)
            pool.release_budget()
            with pool.lock:
                pool.stale_flushed_chunks += 1
                if entry.speculative:
                    pool.speculative_flushed += 1

        future.add_done_callback(_release)

    def _release_unstaged_gather(self, gathered: Any) -> None:
        """Return the leases of a gathered chunk that device staging never took.

        ``gathered`` is one host-pipeline result,
        ``(payload, gather_seconds)``, abandoned when its panel stopped
        early.  A borrowed resident slice is settled without an H2D event
        (as the staging error path does); any other payload returns its
        pinned slots as a never-consumed lookahead chunk does.
        """

        payload, _gather_seconds = gathered
        if isinstance(payload, _ResidentPinnedSlicePayload):
            if not payload.settled:
                self._settle_resident_pinned_slice(payload, ready_event=None)
            return
        self._release_lookahead_payload(payload)

    def _release_lookahead_payload(self, payload: Any) -> None:
        """Return a never-consumed pre-gathered chunk's pinned leases.

        Mirrors the gather fallback's release: a leased slot with no
        recorded H2D event will never reach ``_device_batch``, so without
        this it would stay unavailable until overflow accounting noticed.
        """

        if isinstance(payload, _PinSplitPayload):
            images, labels = payload.host_images, payload.host_labels
        else:
            try:
                _ids, images, labels, _nbytes = payload
            except (TypeError, ValueError):
                return
        with self._pinned_ring_lock:
            for tensor in (images, labels):
                if tensor is None:
                    continue
                state = self._pinned_slot_state.get(id(tensor))
                if state is not None and state.get("event") is None:
                    state["leased"] = False
                    state["staged_checksum"] = None

    # ------------------------------------------------------------------ #
    # On-target device pin
    # ------------------------------------------------------------------ #

    def enter_raw_panel_delivery(self) -> bool:
        """Switch to raw-space uint8 panels (the folded-stem transaction).

        Delivered fp32 tensors carry the exact uint8 codes (cast only —
        the ``mul_/add_`` normalize is elided); the folded first conv owns
        the decode algebra.  Refuses (False) for non-optimized sources,
        whose fallback batches are pre-normalized floats and cannot be
        expressed in raw space — the transaction then degrades the WHOLE
        fold, never a partial one.  Must run before any serving.
        """

        if not self.optimized_bank:
            return False
        self._raw_panel_delivery = True
        return True

    def exit_raw_panel_delivery(self) -> None:
        self._raw_panel_delivery = False

    def record_uint8_folded_stem_binding(self, receipt: Mapping[str, Any]) -> None:
        """Bind the plane receipt to the completed atomic transaction.

        The plane owns delivery-state telemetry while the evaluator owns
        the fold transaction.  Recording the evaluator's completed receipt
        here makes a refusal diagnosable from the canonical data-plane
        receipt without allowing telemetry to disagree with live delivery.
        """

        binding = dict(receipt)
        for field in ("enabled", "engaged"):
            if type(binding.get(field)) is not bool:
                raise TypeError(f"folded-stem binding receipt requires boolean {field!r}")
        if bool(binding["engaged"]) != bool(self._raw_panel_delivery):
            raise ValueError(
                "folded-stem binding engagement disagrees with raw-panel delivery state"
            )
        self._uint8_folded_stem_binding = binding

    def _delivery_inputs(self, probe: Any, uint8_batch: Any, inputs: Any) -> Any:
        binding = self._uint8_folded_stem_binding
        bound = self._raw_panel_delivery and binding is not None and binding.get("raw_input_dtype") == "uint8"
        scoring = bound and getattr(self._delivery_local, "role", None) == "scoring"
        if bound and not scoring:
            # The stem owns the cast and decode algebra. The consumer's
            # existing first/after-use probes still read every raw row.
            return uint8_batch
        # Attribution kernels consume the normalized conv1 source activation.
        # Keep that source and its scale for scoring, while readout forwards
        # use the equivalent folded stem.
        raw = self._raw_panel_delivery and not scoring
        if not raw and self._decode_scale_device is None:
            from mechanistic_systems.measurement.device_step_records import upload_metadata

            self._decode_scale_device = upload_metadata(
                self._decode_scale, self.device, dtype=torch.float32,
            )
            self._decode_bias_device = upload_metadata(
                self._decode_bias, self.device, dtype=torch.float32,
            )
        _lane_g.normalize_with_first_use_digest(
            probe, uint8_batch, inputs, decode_scale=self._decode_scale_device,
            decode_bias=self._decode_bias_device, raw=raw,
        )
        return inputs

    def install_on_target_pin(self, target_class_id: int) -> None:
        """Pin the target-class stratum rows on device, once per arm.

        Built through the normal gather path so the pinned bytes are the
        exact bytes the host path would deliver.  Fail-soft: any failure
        leaves the pin unbuilt with a receipted reason and serving stays
        on the host path.  Idempotent; a second install with a different
        class is refused (receipted), never rebuilt.
        """

        if self._device_gather is not None or not self._on_target_pin_env:
            return
        with self._pin_lock:
            if self._pin_build_attempted:
                if (
                    self._pin_target_class_id is not None
                    and int(target_class_id) != self._pin_target_class_id
                    and len(self._pin_degradations) < _LOOKAHEAD_DEGRADATION_REASON_CAP
                ):
                    # Direct append: _pin_degradation would re-acquire the
                    # (non-reentrant) pin lock held here.
                    self._pin_degradations.append(
                        "pin_target_class_conflict:"
                        f"{self._pin_target_class_id}!="
                        f"{int(target_class_id)}"
                    )
                return
            self._pin_build_attempted = True
            self._pin_target_class_id = int(target_class_id)
        try:
            self._build_on_target_pin()
        except Exception as error:  # never fail the arm
            self._pin_degradation(f"pin_build_failed:{type(error).__name__}:{error}")

    def _pin_degradation(self, reason: str) -> None:
        with self._pin_lock:
            if len(self._pin_degradations) < (_LOOKAHEAD_DEGRADATION_REASON_CAP):
                self._pin_degradations.append(str(reason))

    def _build_on_target_pin(self) -> None:
        framework = _require_torch()
        if not self.optimized_bank:
            self._pin_degradation("pin_requires_optimized_bank")
            return
        labels = getattr(self.source, "_labels", None)
        if labels is None:
            self._pin_degradation("pin_requires_bank_labels")
            return
        target = int(self._pin_target_class_id)
        pinned_ids = tuple(
            str(record.context_id)
            for record in sorted(self.axis.records, key=lambda item: int(item.ordinal))
            if int(labels[int(record.ordinal)]) == target
        )
        if not pinned_ids:
            self._pin_degradation("pin_target_class_absent_from_bank")
            return
        pin_images: Any | None = None
        pin_labels: Any | None = None
        pin_digest_chunks: list[np.ndarray] = []
        pin_digests_complete = True
        offset = 0
        for start in range(0, len(pinned_ids), _PIN_BUILD_CHUNK_ROWS):
            chunk = pinned_ids[start : start + _PIN_BUILD_CHUNK_ROWS]
            chunk_payload = self._gather_bank_chunk(
                chunk, parallel_shards=1, _pin_exempt=True
            )
            _ids, images, chunk_labels, _bytes = chunk_payload
            # CLASS3: the pin bank's row digests come from the same gather
            # that produced its bytes, so pinned rows are certified against
            # the bank exactly like host rows.
            chunk_digests = _payload_row_digests(chunk_payload)
            if chunk_digests is None:
                pin_digests_complete = False
            else:
                pin_digest_chunks.append(np.asarray(chunk_digests, dtype=np.int64))
            if pin_images is None:
                pin_images = framework.empty(
                    (len(pinned_ids), *tuple(int(v) for v in images.shape[1:])),
                    dtype=framework.uint8,
                    device=self.device,
                    memory_format=(
                        framework.channels_last if images.ndim == 4 else framework.contiguous_format
                    ),
                )
                pin_labels = framework.empty(
                    len(pinned_ids),
                    dtype=framework.long,
                    device=self.device,
                )
            pin_images[offset : offset + len(chunk)].copy_(
                images.to(device=self.device, non_blocking=False)
            )
            pin_labels[offset : offset + len(chunk)].copy_(
                chunk_labels.to(device=self.device, non_blocking=False)
            )
            offset += len(chunk)
            # The build leased pinned host slots on CUDA; no H2D event is
            # recorded for them (synchronous copies above), so return the
            # leases explicitly.
            self._release_lookahead_payload((_ids, images, chunk_labels, 0))
        # CLASS3: the pin bank is filled on the
        # building thread's CURRENT stream (the final label ``copy_`` above
        # is still asynchronous there) but read on the COPY stream by the
        # pinned serve path, with no event between the two.  Complete the
        # fill before the pin can serve.  Once per arm; not gated by the
        # integrity kill switch; value-neutral.
        if pin_images is not None and self.device.type == "cuda":
            framework.cuda.current_stream(self.device).synchronize()
        self._pin_images = pin_images
        self._pin_labels = pin_labels
        self._pin_row_digests_host = (
            np.concatenate(pin_digest_chunks)
            if self._delivery_integrity_armed and pin_digests_complete and pin_digest_chunks
            else None
        )
        self._pin_slot_by_context = {context_id: slot for slot, context_id in enumerate(pinned_ids)}
        self._pin_bytes = self._tensor_bytes(pin_images) + self._tensor_bytes(pin_labels)
        self._pin_built = True

    def _pin_serve_active(self) -> bool:
        """CPU degeneracy: the pin serves rows only on a CUDA device."""

        return self._pin_built and self._on_target_pin_env and self.device.type == "cuda"

    def _pin_split_for_chunk(
        self, context_ids: tuple[str, ...]
    ) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Partition a chunk into host rows and device-pinned rows.

        Returns (host_ids, host_positions, pin_positions, pin_slots) with
        positions in chunk-relative order — the assembly's interleave key.
        """

        host_ids: list[str] = []
        host_positions: list[int] = []
        pin_positions: list[int] = []
        pin_slots: list[int] = []
        for position, context_id in enumerate(context_ids):
            slot = self._pin_slot_by_context.get(str(context_id))
            if slot is None:
                host_ids.append(str(context_id))
                host_positions.append(position)
            else:
                pin_positions.append(position)
                pin_slots.append(int(slot))
        return (
            tuple(host_ids),
            tuple(host_positions),
            tuple(pin_positions),
            tuple(pin_slots),
        )

    def _assemble_pinned_uint8(
        self,
        pin_split: _PinSplitPayload,
        host_images: Any,
        *,
        out: Any | None = None,
    ) -> tuple[Any, Any]:
        """Interleave host-gathered and device-pinned rows byte-exactly.

        Device-agnostic (unit-tested on CPU): the output row at position p
        is the host gather's row for host positions and the pin bank's row
        for pinned positions — the EXACT original chunk order.  Returns
        (full uint8 batch, moved host part or None) so the caller can audit
        the H2D bytes against the staged host checksum.
        """

        from mechanistic_systems.measurement.device_step_records import upload_metadata

        framework = _require_torch()
        rows = len(pin_split.context_ids)
        uint8_batch = (
            out
            if out is not None
            else framework.empty(
                (rows, *tuple(int(v) for v in self._pin_images.shape[1:])),
                dtype=framework.uint8,
                device=self.device,
                memory_format=(
                    framework.channels_last
                    if self._pin_images.ndim == 4
                    else framework.contiguous_format
                ),
            )
        )
        moved_images = None
        if host_images is not None and pin_split.host_positions:
            moved_images = host_images.to(
                device=self.device,
                dtype=framework.uint8,
                memory_format=(
                    framework.contiguous_format
                    if self._image_gather_enabled and host_images.is_contiguous()
                    else framework.channels_last
                    if host_images.ndim == 4 else framework.contiguous_format
                ),
                non_blocking=True,
            )
            uint8_batch.index_copy_(
                0,
                upload_metadata(
                    pin_split.host_positions,
                    self.device,
                    dtype=framework.long,
                ),
                moved_images,
            )
        if pin_split.pin_positions:
            uint8_batch.index_copy_(
                0,
                upload_metadata(
                    pin_split.pin_positions,
                    self.device,
                    dtype=framework.long,
                ),
                self._pin_images.index_select(
                    0,
                    upload_metadata(
                        pin_split.pin_slots,
                        self.device,
                        dtype=framework.long,
                    ),
                ),
            )
        return uint8_batch, moved_images

    def _assemble_pinned_labels(
        self,
        pin_split: _PinSplitPayload,
        host_labels: Any,
        *,
        out: Any | None = None,
    ) -> tuple[Any, Any]:
        """Label counterpart of :meth:`_assemble_pinned_uint8`.

        ``out`` is the normalize ring slot's label
        buffer; every position is written (host and pinned positions
        partition the chunk), so reusing it is value-identical.
        """

        from mechanistic_systems.measurement.device_step_records import upload_metadata

        framework = _require_torch()
        labels = (
            out
            if out is not None
            else framework.empty(
                len(pin_split.context_ids),
                dtype=framework.long,
                device=self.device,
            )
        )
        moved_labels = None
        if host_labels is not None and pin_split.host_positions:
            moved_labels = host_labels.to(device=self.device, non_blocking=True).to(
                dtype=framework.long
            )
            labels.index_copy_(
                0,
                upload_metadata(
                    pin_split.host_positions,
                    self.device,
                    dtype=framework.long,
                ),
                moved_labels,
            )
        if pin_split.pin_positions:
            labels.index_copy_(
                0,
                upload_metadata(
                    pin_split.pin_positions,
                    self.device,
                    dtype=framework.long,
                ),
                self._pin_labels.index_select(
                    0,
                    upload_metadata(
                        pin_split.pin_slots,
                        self.device,
                        dtype=framework.long,
                    ),
                ),
            )
        return labels, moved_labels

    def _cache_key(self, context_id: str) -> tuple[str, str]:
        return self._preprocessing_hash, str(context_id)

    @staticmethod
    def _tensor_bytes(value: Any) -> int:
        return int(value.numel() * value.element_size())

    def set_device_cache_bytes(self, byte_limit: int) -> None:
        self.spec = replace(self.spec, device_cache_bytes=int(byte_limit))

    def release_device_cache(self) -> None:
        self.spec = replace(self.spec, device_cache_bytes=0)
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self._device_cache_pages = None
        self._device_cache_free_slots = []
        self._device_cache_row_bytes = 0

    def _ensure_device_cache_pages(self, row: Any) -> None:
        framework = _require_torch()
        if (
            self._device_cache_pages is not None
            or int(self.spec.device_cache_bytes) <= 0
            or self.device.type != "cuda"
            or row.dtype != framework.uint8
        ):
            return
        row_bytes = self._tensor_bytes(row[:1])
        slot_count = min(
            len(self._records),
            process_device_memory_budget_bytes(int(self.spec.device_cache_bytes)) // max(1, row_bytes),
        )
        if slot_count <= 0:
            return
        # CLASS3: the pages are only ever read
        # and written on the copy stream, so allocate them under it; a
        # stage-thread (default-stream) allocation freed by
        # ``release_device_cache`` could otherwise be reused on the default
        # stream while copy-stream work on it is still queued.
        with (
            framework.cuda.stream(self._copy_stream)
            if self._copy_stream is not None
            else nullcontext()
        ):
            self._device_cache_pages = framework.empty(
                (slot_count, *tuple(int(value) for value in row.shape[1:])),
                dtype=framework.uint8,
                device=self.device,
                memory_format=(
                    framework.channels_last if row.ndim == 4 else framework.contiguous_format
                ),
            )
        self._device_cache_free_slots = list(range(slot_count - 1, -1, -1))
        self._device_cache_row_bytes = row_bytes

    def device_cache_active(self) -> bool:
        """True when the device LRU row cache is configured or allocated."""

        return self._device_cache_pages is not None or (
            int(self.spec.device_cache_bytes) > 0 and self.device.type == "cuda"
        )

    def evict_device_cache_contexts(self, context_ids: Sequence[str]) -> int:
        """Drop ``context_ids`` from the device LRU row cache; returns the count.

        CLASS3: a delivery retry must re-read its rows from
        the bank, never re-serve a cached device copy of a row the failed
        attempt may have corrupted.  Called only between physical passes
        (no panel of this plane is being staged).  Value-neutral: an
        evicted row is simply a cache miss on its next delivery.
        """

        removed = 0
        with self._cache_lock:
            for context_id in context_ids:
                slot = self._cache.pop(self._cache_key(str(context_id)), None)
                if slot is None:
                    continue
                self._device_cache_free_slots.append(int(slot))
                self._cache_bytes -= int(self._device_cache_row_bytes)
                removed += 1
        return removed

    def _workspace(self, host_images: Any) -> dict[str, Any] | None:
        framework = _require_torch()
        if self.device.type != "cuda" or host_images.dtype != framework.uint8:
            return None
        key = (
            tuple(int(value) for value in host_images.shape),
            str(host_images.dtype),
            str(self.device),
        )
        ring = self._workspace_rings.get(key)
        if ring is None:
            ring = []
            # ``PooledBackgroundIterator`` permits ``workers + depth`` staged
            # results while the consumer may still own the batch it most
            # recently received.  Device staging uses one worker, so the
            # workspace ring must cover:
            #
            #   one consumer-owned batch
            #   + one actively staged batch
            #   + ``prefetch_depth`` queued batches.
            #
            # A ring of only ``prefetch_depth`` slots can wrap before the
            # consumer records the slot's completion event.  That silently
            # pairs labels from one physical chunk with inputs from a later
            # chunk; class-major terminal panels expose the corruption while
            # randomly ordered training panels can conceal it.
            workspace_slots = int(self.spec.prefetch_depth) + 2
            for _ in range(workspace_slots):
                ring.append(
                    {
                        "uint8": framework.empty(
                            tuple(int(value) for value in host_images.shape),
                            dtype=framework.uint8,
                            device=self.device,
                            memory_format=(
                                framework.contiguous_format
                                if self._image_gather_enabled else framework.channels_last
                            ),
                        ),
                        "float": framework.empty(
                            tuple(int(value) for value in host_images.shape),
                            dtype=framework.float32,
                            device=self.device,
                            memory_format=framework.channels_last,
                        ),
                        "labels": framework.empty(
                            int(host_images.shape[0]),
                            dtype=framework.long,
                            device=self.device,
                        ),
                        "completion": framework.cuda.Event(),
                        "busy": False,
                    }
                )
            self._workspace_rings[key] = ring
            self._workspace_cursors[key] = 0
        cursor = self._workspace_cursors[key]
        slot = ring[cursor]
        self._workspace_cursors[key] = (cursor + 1) % len(ring)
        if bool(slot["busy"]) and self._copy_stream is not None:
            self._copy_stream.wait_event(slot["completion"])
        slot["busy"] = True
        return slot

    def _normalize_ring_slot(self, shape: tuple[int, ...]) -> dict[str, Any]:
        """Lease one persistent normalize buffer pair.

        Ring size is ``prefetch_depth + 2`` — the identical in-flight
        arithmetic the workspace ring documents (one consumer-owned batch
        + one actively staged + ``prefetch_depth`` queued), so a slot is
        never rewritten before its batch leaves the pipeline under the
        plane's own consumption pattern.  On CUDA the slot carries a
        completion event and the copy stream waits on it before reuse
        (the workspace discipline); on CPU (no event contract on
        ``mark_consumed``) the depth bound is the guarantee, and armed
        mode therefore requires consumption-scoped batches — the plane's
        only production consumers.  CLASS3: each slot also
        carries a ``labels`` buffer (int64, one per row) that the pinned
        path assembles into, so pinned-path labels share the slot's
        completion-event lifetime instead of being freshly allocated on the
        copy stream and freed while a consumer-stream clone of them may
        still be pending.  The label buffers (8 B/row) are not counted in
        ``buffers``/``reserved_bytes`` so that receipt stays unchanged.
        """

        framework = _require_torch()
        key = (tuple(int(value) for value in shape), str(self.device))
        with self._normalize_ring_lock:
            ring = self._normalize_rings.get(key)
            if ring is None:
                ring = []
                for _ in range(int(self.spec.prefetch_depth) + 2):
                    uint8_buffer = framework.empty(
                        key[0],
                        dtype=framework.uint8,
                        device=self.device,
                        memory_format=(
                            framework.contiguous_format
                            if self._image_gather_enabled
                            else framework.channels_last
                            if len(key[0]) == 4 else framework.contiguous_format
                        ),
                    )
                    float_buffer = framework.empty(
                        key[0],
                        dtype=framework.float32,
                        device=self.device,
                        memory_format=(
                            framework.channels_last
                            if len(key[0]) == 4
                            else framework.contiguous_format
                        ),
                    )
                    labels_buffer = framework.empty(
                        (int(key[0][0]),),
                        dtype=framework.long,
                        device=self.device,
                    )
                    ring.append(
                        {
                            "uint8": uint8_buffer,
                            "float": float_buffer,
                            "labels": labels_buffer,
                            "completion": (
                                framework.cuda.Event() if self.device.type == "cuda" else None
                            ),
                            "busy": False,
                        }
                    )
                    self._normalize_ring_buffers += 2
                    self._normalize_ring_bytes += self._tensor_bytes(
                        uint8_buffer
                    ) + self._tensor_bytes(float_buffer)
                self._normalize_rings[key] = ring
                self._normalize_ring_cursors[key] = 0
            cursor = self._normalize_ring_cursors[key]
            slot = ring[cursor]
            self._normalize_ring_cursors[key] = (cursor + 1) % len(ring)
            self._normalize_ring_hits += 1
        if bool(slot["busy"]) and slot["completion"] is not None and self._copy_stream is not None:
            self._copy_stream.wait_event(slot["completion"])
        slot["busy"] = True
        return slot

    def _pin_split_expected_digests(self, pin_split: _PinSplitPayload) -> np.ndarray | None:
        """CLASS3 I1 authority for a pin-split chunk, in chunk order.

        Host rows take the payload's compact host digests; pinned rows take
        the pin bank's digests at their slots.  None when either part lacks
        an authority (the chunk is then counted as uncovered, never guessed).
        """

        rows = len(pin_split.context_ids)
        expected = np.empty((rows,), dtype=np.int64)
        if pin_split.host_positions:
            host = pin_split.host_row_digests
            if host is None:
                return None
            host_array = np.asarray(host, dtype=np.int64).reshape(-1)
            if int(host_array.shape[0]) != len(pin_split.host_positions):
                return None
            expected[list(pin_split.host_positions)] = host_array
        if pin_split.pin_positions:
            pin = self._pin_row_digests_host
            if pin is None:
                return None
            expected[list(pin_split.pin_positions)] = pin[list(pin_split.pin_slots)]
        return expected

    def _device_batch_pinned(
        self,
        pin_split: _PinSplitPayload,
        expected_host_checksums: tuple[int | None, int | None] | None = None,
        row_digests: Any = None,
    ) -> tuple[
        MeasurementInputBatch,
        int,
        int,
        int,
        Any,
        Any,
        Mapping[str, float],
    ]:
        """Assemble a chunk whose target rows are device-resident (rung 3).

        Bitwise contract: the delivered float batch equals the host path's
        batch row-for-row — host rows arrive via the same staged-slot H2D,
        pinned rows via ``index_select`` on bytes the pin build gathered
        through the identical host path.  Only WHERE the uint8 bytes came
        from changes.  The LRU device cache is bypassed (the pin replaces
        it for target rows; non-target rows are plain H2D — receipted as
        misses); delivered-byte audits cover the moved host part exactly
        as the full-slot path does.  CLASS3: the assembled uint8 batch is
        row-digested on the copy stream, so the pin interleave and the pin
        bank bytes are certified against the gather too.
        """

        framework = _require_torch()
        host_images = pin_split.host_images
        host_labels = pin_split.host_labels
        # CLASS3 I2: the host part's staged stamps travel on the payload.
        if expected_host_checksums is None and (
            pin_split.host_image_checksum is not None
            or pin_split.host_label_checksum is not None
        ):
            expected_host_checksums = (
                pin_split.host_image_checksum,
                pin_split.host_label_checksum,
            )
        expected_image_checksum = (
            None if expected_host_checksums is None else expected_host_checksums[0]
        )
        expected_label_checksum = (
            None if expected_host_checksums is None else expected_host_checksums[1]
        )
        expected_row_digests = None if self._device_look_delivery_active() else (
            np.asarray(row_digests, dtype=np.int64).reshape(-1)
            if row_digests is not None
            and int(np.asarray(row_digests).size) == len(pin_split.context_ids)
            else self._pin_split_expected_digests(pin_split)
        )
        h2d_bytes = 0 if host_images is None else self._tensor_bytes(host_images)
        ready_event = None
        copy_start_event = None
        copy_end_event = None
        h2d_enqueue_seconds = 0.0
        normalization_enqueue_seconds = 0.0
        stream_context = (
            framework.cuda.stream(self._copy_stream)
            if self._copy_stream is not None
            else nullcontext()
        )
        with stream_context:
            if self._copy_stream is not None and not self._device_look_delivery_active():
                copy_start_event = framework.cuda.Event(enable_timing=True)
                copy_end_event = framework.cuda.Event(enable_timing=True)
                copy_start_event.record(self._copy_stream)
            h2d_started = time.perf_counter()
            pinned_normalize_slot = (
                self._normalize_ring_slot(
                    (
                        len(pin_split.context_ids),
                        *tuple(int(v) for v in self._pin_images.shape[1:]),
                    )
                )
                if self._normalize_buffer_reuse
                else None
            )
            uint8_batch, moved_images = self._assemble_pinned_uint8(
                pin_split,
                host_images,
                out=(None if pinned_normalize_slot is None else pinned_normalize_slot["uint8"]),
            )
            h2d_enqueue_seconds += time.perf_counter() - h2d_started
            if host_images is not None and moved_images is not None:
                # Full asynchronous copy of the staged host part: audit the
                # delivered bytes against the staged checksum (settled at
                # panel end), exactly as the full-slot path does.
                self._queue_delivery_check(
                    host_images,
                    moved_images,
                    "images_h2d",
                    expected_checksum=expected_image_checksum,
                )
            # CLASS3 I1: every assembled row (host-moved and pinned alike).
            delivery_probe = self._queue_row_digest_check(
                pin_split.context_ids,
                expected_row_digests,
                uint8_batch,
                "pinned_chunk",
            )
            inputs = (
                pinned_normalize_slot["float"]
                if pinned_normalize_slot is not None
                else framework.empty_like(
                    uint8_batch,
                    dtype=framework.float32,
                    memory_format=(
                        framework.channels_last
                        if uint8_batch.ndim == 4
                        else framework.contiguous_format
                    ),
                )
            )
            normalization_started = time.perf_counter()
            # See ``_device_batch``.
            inputs = self._delivery_inputs(delivery_probe, uint8_batch, inputs)
            normalization_enqueue_seconds += time.perf_counter() - normalization_started
            labels, moved_labels = self._assemble_pinned_labels(
                pin_split,
                host_labels,
                out=(
                    None
                    if pinned_normalize_slot is None
                    else pinned_normalize_slot.get("labels")
                ),
            )
            if host_labels is not None and moved_labels is not None:
                self._queue_delivery_check(
                    host_labels,
                    moved_labels,
                    "labels_h2d",
                    expected_checksum=expected_label_checksum,
                )
            if self._copy_stream is not None:
                if copy_end_event is not None:
                    copy_end_event.record(self._copy_stream)
                ready_event = framework.cuda.Event()
                ready_event.record(self._copy_stream)
                with self._pinned_ring_lock:
                    for host_tensor in (host_images, host_labels):
                        if host_tensor is None:
                            continue
                        state = self._pinned_slot_state.get(id(host_tensor))
                        if state is not None:
                            # The lease stays held; acquisition frees the
                            # slot once this event reports complete.
                            state["event"] = ready_event
        return (
            MeasurementInputBatch(
                context_ids=pin_split.context_ids,
                inputs=inputs,
                labels=labels,
                input_identity_hash=self._input_identity_hash,
                ready_event=ready_event,
                completion_event=(
                    None if pinned_normalize_slot is None else pinned_normalize_slot["completion"]
                ),
                **({"delivery_probe": delivery_probe} if delivery_probe is not None else {}),
                # CLASS3: a fresh (non-ring) uint8 staging
                # buffer is read on the consumer stream by the use probes.
                **(
                    {"stream_guarded_tensors": (uint8_batch,)}
                    if delivery_probe is not None and pinned_normalize_slot is None
                    else {}
                ),
            ),
            0,
            len(pin_split.host_positions),
            h2d_bytes,
            copy_start_event,
            copy_end_event,
            {
                "device_cache_lookup_seconds": 0.0,
                "device_cache_insert_seconds": 0.0,
                "h2d_enqueue_seconds": float(h2d_enqueue_seconds),
                "normalization_enqueue_seconds": float(normalization_enqueue_seconds),
            },
        )

    def _device_batch(
        self,
        context_ids: tuple[str, ...],
        host_images: Any,
        host_labels: Any,
        pin_split: _PinSplitPayload | None = None,
        expected_host_checksums: tuple[int | None, int | None] | None = None,
        row_digests: Any = None,
    ) -> tuple[
        MeasurementInputBatch,
        int,
        int,
        int,
        Any,
        Any,
        Mapping[str, float],
    ]:
        from mechanistic_systems.measurement.device_step_records import upload_metadata

        if pin_split is not None:
            return self._device_batch_pinned(
                pin_split,
                expected_host_checksums=expected_host_checksums,
                row_digests=row_digests,
            )
        framework = _require_torch()
        # CLASS3 I2: an expectation handed in by the caller came from the
        # payload's own stamp (serve path, resident slice); None means the
        # pre-existing lease-stamped audit.
        # Only the CUDA path binds an explicit copy stream plus readiness
        # and lease-lifetime events.  Apple MPS accepts ``non_blocking=True``
        # for CPU transfers but exposes no corresponding event contract here;
        # in particular, a final labels transfer can outlive its assembled
        # host tensor and deliver stale/uninitialized values.  Keep non-CUDA
        # host transfers synchronous.  This is execution policy only and does
        # not alter delivered bytes or scientific ordering.
        asynchronous_host_transfer = self.device.type == "cuda"
        # Producer stamps follow the payload into device delivery checks.
        # Their flags settle after dispatch; a mismatch ends the run.
        expected_image_checksum = (
            None
            if expected_host_checksums is None or expected_host_checksums[0] is None
            else int(expected_host_checksums[0])
        )
        expected_label_checksum = (
            None
            if expected_host_checksums is None or expected_host_checksums[1] is None
            else int(expected_host_checksums[1])
        )
        self._ensure_device_cache_pages(host_images)
        cache_lookup_started = time.perf_counter()
        hits = 0
        misses: list[int] = []
        hit_slots: dict[int, int] = {}
        with self._cache_lock:
            for index, context_id in enumerate(context_ids):
                key = self._cache_key(context_id)
                slot = self._cache.get(key)
                if slot is None:
                    misses.append(index)
                else:
                    self._cache.move_to_end(key)
                    hits += 1
                    hit_slots[index] = int(slot)
        cache_lookup_seconds = time.perf_counter() - cache_lookup_started

        h2d_bytes = 0
        h2d_enqueue_seconds = 0.0
        normalization_enqueue_seconds = 0.0
        cache_insert_seconds = 0.0
        ready_event = None
        copy_start_event = None
        copy_end_event = None
        delivery_probe = None
        if host_images.dtype == framework.uint8:
            h2d_bytes = 0 if not misses else len(misses) * self._tensor_bytes(host_images[:1])
            stream_context = (
                framework.cuda.stream(self._copy_stream)
                if self._copy_stream is not None
                else nullcontext()
            )
            with stream_context:
                if self._copy_stream is not None and not self._device_look_delivery_active():
                    copy_start_event = framework.cuda.Event(enable_timing=True)
                    copy_end_event = framework.cuda.Event(enable_timing=True)
                    copy_start_event.record(self._copy_stream)
                workspace = self._workspace(host_images)
                normalize_slot = (
                    self._normalize_ring_slot(tuple(int(value) for value in host_images.shape))
                    if workspace is None and self._normalize_buffer_reuse
                    else None
                )
                uint8_batch = (
                    workspace["uint8"]
                    if workspace is not None
                    else (
                        normalize_slot["uint8"]
                        if normalize_slot is not None
                        else framework.empty(
                            tuple(int(value) for value in host_images.shape),
                            dtype=framework.uint8,
                            device=self.device,
                            memory_format=(
                                framework.contiguous_format
                                if self._image_gather_enabled else framework.channels_last
                            ),
                        )
                    )
                )
                cache_disabled_fast_path = (
                    self._device_cache_pages is None and int(self.spec.device_cache_bytes) == 0
                )
                if cache_disabled_fast_path:
                    h2d_started = time.perf_counter()
                    uint8_batch.copy_(
                        host_images,
                        non_blocking=asynchronous_host_transfer,
                    )
                    h2d_enqueue_seconds += time.perf_counter() - h2d_started
                    h2d_bytes = self._tensor_bytes(host_images)
                    # Full-slot asynchronous copy: audit the delivered bytes
                    # against the staged checksum (settled at panel end).
                    self._queue_delivery_check(
                        host_images,
                        uint8_batch,
                        "images_h2d",
                        expected_checksum=expected_image_checksum,
                    )
                elif hit_slots and self._device_cache_pages is not None:
                    destinations = upload_metadata(
                        tuple(hit_slots),
                        self.device,
                        dtype=framework.long,
                    )
                    slots = upload_metadata(
                        tuple(hit_slots[index] for index in hit_slots),
                        self.device,
                        dtype=framework.long,
                    )
                    uint8_batch.index_copy_(
                        0,
                        destinations,
                        self._device_cache_pages.index_select(0, slots),
                    )
                if misses and not cache_disabled_fast_path:
                    h2d_started = time.perf_counter()
                    miss_index_cpu = framework.as_tensor(
                        misses,
                        dtype=framework.long,
                    )
                    missing_host = host_images.index_select(
                        0,
                        miss_index_cpu,
                    )
                    missing_uint8 = missing_host.to(
                        device=self.device,
                        dtype=framework.uint8,
                        memory_format=framework.channels_last,
                        non_blocking=asynchronous_host_transfer,
                    )
                    destinations = upload_metadata(
                        misses,
                        self.device,
                        dtype=framework.long,
                    )
                    uint8_batch.index_copy_(
                        0,
                        destinations,
                        missing_uint8,
                    )
                    h2d_enqueue_seconds += time.perf_counter() - h2d_started
                    if self._device_cache_pages is not None:
                        cache_insert_started = time.perf_counter()
                        cache_capacity = int(self._device_cache_pages.shape[0])
                        cacheable = tuple(
                            range(
                                max(0, len(misses) - cache_capacity),
                                len(misses),
                            )
                        )
                        cache_slots: list[int] = []
                        with self._cache_lock:
                            for local in cacheable:
                                original = misses[local]
                                if not self._device_cache_free_slots:
                                    old_key, slot = self._cache.popitem(last=False)
                                    del old_key
                                    self._cache_bytes -= int(self._device_cache_row_bytes)
                                else:
                                    slot = self._device_cache_free_slots.pop()
                                key = self._cache_key(context_ids[original])
                                self._cache[key] = int(slot)
                                self._cache.move_to_end(key)
                                self._cache_bytes += int(self._device_cache_row_bytes)
                                cache_slots.append(int(slot))
                        # One batched device operation replaces one CUDA
                        # ``copy_`` launch per physical image.  Cache metadata
                        # remains route/context keyed on the CPU, while the
                        # uint8 payload is stored in contiguous device pages.
                        if cache_slots:
                            cache_slot_tensor = upload_metadata(
                                cache_slots,
                                self.device,
                                dtype=framework.long,
                            )
                            cache_source_tensor = upload_metadata(
                                cacheable,
                                self.device,
                                dtype=framework.long,
                            )
                            self._device_cache_pages.index_copy_(
                                0,
                                cache_slot_tensor,
                                missing_uint8.index_select(0, cache_source_tensor),
                            )
                        cache_insert_seconds += time.perf_counter() - cache_insert_started
                # CLASS3 I1: the uint8 batch is complete (H2D rows and cache
                # hits alike); digest it here, on the copy stream, before the
                # normalize copy reads it.  Settled at panel end.
                delivery_probe = self._queue_row_digest_check(
                    context_ids, row_digests, uint8_batch, "chunk"
                )
                inputs = (
                    workspace["float"]
                    if workspace is not None
                    else (
                        normalize_slot["float"]
                        if normalize_slot is not None
                        else framework.empty_like(
                            uint8_batch,
                            dtype=framework.float32,
                            memory_format=framework.channels_last,
                        )
                    )
                )
                normalization_started = time.perf_counter()
                # Bitwise the copy_/mul_/add_ sequence; on CUDA
                # with Triton one kernel reads each byte once and, when the
                # probe wants it, also yields the first-use digest of the
                # bytes it read (the consumer-stream first-use call is then
                # a no-op).  Elsewhere the torch sequence runs unchanged.
                inputs = self._delivery_inputs(delivery_probe, uint8_batch, inputs)
                normalization_enqueue_seconds += time.perf_counter() - normalization_started
                if workspace is not None:
                    labels = workspace["labels"]
                    labels.copy_(
                        host_labels,
                        non_blocking=asynchronous_host_transfer,
                    )
                else:
                    labels = host_labels.to(
                        device=self.device,
                        non_blocking=asynchronous_host_transfer,
                    )
                # The labels copy reads its pinned slot asynchronously on
                # both cache paths; audit the delivered label bytes too.
                self._queue_delivery_check(
                    host_labels,
                    labels,
                    "labels_h2d",
                    expected_checksum=expected_label_checksum,
                )
                if self._copy_stream is not None:
                    if copy_end_event is not None:
                        copy_end_event.record(self._copy_stream)
                    ready_event = framework.cuda.Event()
                    ready_event.record(self._copy_stream)
                    with self._pinned_ring_lock:
                        for host_tensor in (host_images, host_labels):
                            state = self._pinned_slot_state.get(id(host_tensor))
                            if state is not None:
                                # The lease stays held; acquisition frees the
                                # slot once this event reports complete.
                                state["event"] = ready_event
        else:
            # Exploratory JPEG fallback is already normalized float data.
            # CLASS3: these rows carry no bank bytes, so the
            # row-digest audit cannot cover them; count them as uncovered.
            if self._delivery_integrity_armed:
                with self._pinned_ring_lock:
                    self._row_digest_rows_uncovered += len(context_ids)
            h2d_bytes = self._tensor_bytes(host_images)
            inputs = host_images.to(
                device=self.device,
                dtype=framework.float32,
                memory_format=(
                    framework.channels_last
                    if host_images.ndim == 4
                    else framework.contiguous_format
                ),
                non_blocking=asynchronous_host_transfer,
            )
            labels = host_labels.to(
                device=self.device,
                non_blocking=asynchronous_host_transfer,
            )
            h2d_enqueue_seconds = 0.0
            normalization_enqueue_seconds = 0.0
        return (
            MeasurementInputBatch(
                context_ids=context_ids,
                inputs=inputs,
                labels=labels,
                input_identity_hash=self._input_identity_hash,
                ready_event=ready_event,
                completion_event=(
                    workspace["completion"]
                    if host_images.dtype == framework.uint8 and workspace is not None
                    else (
                        normalize_slot["completion"]
                        if host_images.dtype == framework.uint8 and normalize_slot is not None
                        else None
                    )
                ),
                **({"delivery_probe": delivery_probe} if delivery_probe is not None else {}),
                # CLASS3: a fresh (non-ring) uint8 staging
                # buffer is read on the consumer stream by the use probes.
                **(
                    {"stream_guarded_tensors": (uint8_batch,)}
                    if delivery_probe is not None
                    and workspace is None
                    and normalize_slot is None
                    else {}
                ),
            ),
            hits,
            len(misses),
            h2d_bytes,
            copy_start_event,
            copy_end_event,
            {
                "device_cache_lookup_seconds": float(cache_lookup_seconds),
                "device_cache_insert_seconds": float(cache_insert_seconds),
                "h2d_enqueue_seconds": float(h2d_enqueue_seconds),
                "normalization_enqueue_seconds": float(normalization_enqueue_seconds),
            },
        )

    def install_batch_policy(self, policy: Any) -> None:
        """Bind a per-role adaptive batch policy (runtime-only, receipted).

        Physical batch rows only partition the panel stream; they never
        change which contexts are evaluated or what is produced, so a policy
        is execution policy, never scientific identity.  Roles absent from
        the policy keep their spec constants.
        """

        self._batch_policy = policy

    def _resolve_role_rows(
        self, semantic_role: str, physical_role: str | None = None
    ) -> int:
        if physical_role is None:
            physical_role = _physical_native_panel_role(semantic_role)
        if self._batch_policy is not None:
            # A semantic-specific override is maximally precise. Otherwise
            # aliases inherit the adaptive policy of their physical execution
            # bucket instead of falling back to a stale fixed batch size.
            rows = self._batch_policy.rows_for(semantic_role)
            policy_role = semantic_role
            if rows is None and physical_role != semantic_role:
                rows = self._batch_policy.rows_for(physical_role)
                policy_role = physical_role
            if rows is not None:
                self._batch_policy_applied_rows[semantic_role] = int(rows)
                if hasattr(self, "_batch_policy_source_role"):
                    self._batch_policy_source_role[semantic_role] = policy_role
                return int(rows)
        return self.spec.batch_rows(semantic_role)

    def panel_batches(
        self,
        context_ids: Sequence[str],
        *,
        role: str,
        batch_rows: int | None = None,
    ) -> Iterable[MeasurementInputBatch]:
        requested = self._validate_contexts(context_ids)
        resolved_role = str(role)
        physical_role = _physical_native_panel_role(resolved_role)
        rows = (
            self._resolve_role_rows(resolved_role, physical_role)
            if batch_rows is None
            else int(batch_rows)
        )
        if rows <= 0:
            raise ValueError("native panel batch_rows must be positive")
        chunks = self._chunks(requested, rows)
        worker_feed = self._device_gather is not None and self._device_gather.worker_residency
        resident_name = (
            "category_calibration" if resolved_role == "category_calibration" else
            "validation" if physical_role == "terminal" else None
        ) if worker_feed else None
        if self.spec.working_set_prefetch and self.optimized_bank and self._device_gather is None:
            self._prefetch_working_set(requested)

        def iterator() -> Iterable[MeasurementInputBatch]:
            if self._device_gather_run is not None and self._device_gather_run.delivery_identity_error is not None:
                raise self._device_gather_run.delivery_identity_error
            if self._device_gather_run is not None and self._device_gather_run.failure is not None:
                raise self._device_gather_error(self._device_gather_run.failure)
            started = time.perf_counter()
            first_batch_latency = 0.0
            wait_seconds = 0.0
            cache_hits = 0
            cache_misses = 0
            read_bytes = 0
            h2d_bytes = 0
            batch_count = 0
            delivered_batch_sizes: list[int] = []
            delivered = 0
            host_gather_activity = 0.0
            device_stage_activity = 0.0
            cache_lookup_activity = 0.0
            cache_insert_activity = 0.0
            h2d_enqueue_activity = 0.0
            normalization_enqueue_activity = 0.0
            copy_event_pairs: list[tuple[Any, Any]] = []
            # Whether THIS panel was drained; its chunks carry it to a
            # deferred settle (None until the panel ends).
            from mechanistic_systems.inference.device_bce import DEVICE_LOOK_ACTIVE
            active_device_look = bool(DEVICE_LOOK_ACTIVE.get())
            panel_state: dict[str, Any] = {"drained": None, "device_look": active_device_look}

            first_device_batch = None
            if self._device_gather is not None and chunks:
                gather = self._device_gather
                if gather.failure is not None:
                    raise self._device_gather_error(gather.failure)
                try:
                    self._delivery_local.role = physical_role
                    self._delivery_local.panel = panel_state
                    self._delivery_local.resident_name = resident_name
                    # This full physical list also covers terminal passes.
                    # Delivery may get far ahead of the current chunk, up to
                    # the byte bound, without deciding any look's row count.
                    if worker_feed:
                        load_start = _require_torch().cuda.Event(enable_timing=True)
                        load_end = _require_torch().cuda.Event(enable_timing=True)
                        load_start.record(self._copy_stream)
                        if resident_name is not None:
                            before = gather.resident_loaded_bytes
                            retained_rows = (
                                range(len(self._bank_inputs)) if resident_name == "validation" else
                                tuple(int(self._records[value].ordinal) for value in requested)
                            )
                            gather.retain(resident_name, retained_rows)
                            loaded = gather.resident_loaded_bytes - before
                            read_bytes += loaded
                            h2d_bytes += loaded
                        else:
                            before = gather.gathered_rows
                            self.request(requested)
                            loaded = (gather.gathered_rows - before) * gather.row_bytes
                            read_bytes += loaded
                            h2d_bytes += loaded
                        load_end.record(self._copy_stream)
                        if loaded:
                            copy_event_pairs.append((load_start, load_end))
                    else:
                        self.install_device_row_order(f"panel:{resolved_role}", requested)
                    if self._device_gather is not None:
                        first_device_batch = self._device_gather_batch(chunks[0])
                except Exception as error:
                    self._device_gather_failure(error)

            # With the master env gate armed, the producer pool's
            # lifetime is the data plane and pre-gathered ready-buffer
            # chunks are consumed first.  ``persistent_pool`` is None with
            # the gate unset, keeping this path structurally unchanged.
            persistent_pool = self._persistent_producer_pool()
            lookahead_active = (
                persistent_pool is not None
                and persistent_pool.lookahead_looks > 0
                and self.optimized_bank
            )

            def record_integrity_error(error: BaseException) -> None:
                if isinstance(error, (DeliveryIntegrityError, _lane_g.BankRowDigestTableError)):
                    with self._pinned_ring_lock:
                        self._staging_integrity_errors.append(error)

            def timed_gather(chunk: tuple[str, ...]):
                self._delivery_local.panel = panel_state
                if lookahead_active:
                    ready = self._take_lookahead_chunk(persistent_pool, chunk)
                    if ready is not None:
                        return ready
                gather_started = time.perf_counter()
                try:
                    if self.optimized_bank:
                        # Multiple physical batches are already gathered in
                        # parallel by ``host_staged``.  A one-batch look would
                        # otherwise exercise only one producer, so parallelize
                        # bounded shards inside that sole gather.
                        payload = self._gather_bank_chunk(
                            chunk,
                            parallel_shards=(
                                int(self.spec.producer_workers) if len(chunks) == 1 else 1
                            ),
                        )
                    else:
                        payload = self._gather_fallback_chunk(chunk)
                except BaseException as error:
                    record_integrity_error(error)
                    raise
                return payload, time.perf_counter() - gather_started

            if persistent_pool is not None:
                with persistent_pool.lock:
                    persistent_pool.reuse_count += 1
            host_staged = None if self._device_gather is not None else PooledBackgroundIterator(
                iter(chunks),
                timed_gather,
                workers=int(self.spec.producer_workers),
                depth=int(self.spec.prefetch_depth),
                executor=(None if persistent_pool is None else persistent_pool.gather_executor),
                on_discard=self._release_unstaged_gather,
            )

            def stage_device(payload):
                gathered, gather_seconds = payload
                resident_slice = (
                    gathered if isinstance(gathered, _ResidentPinnedSlicePayload) else None
                )
                if resident_slice is not None:
                    ids = gathered.context_ids
                    host_images = gathered.host_images
                    host_labels = gathered.host_labels
                    loaded_bytes = gathered.loaded_bytes
                    pin_split = None
                elif isinstance(gathered, _PinSplitPayload):
                    ids = gathered.context_ids
                    host_images = gathered.host_images
                    host_labels = gathered.host_labels
                    loaded_bytes = gathered.loaded_bytes
                    pin_split = gathered
                else:
                    ids, host_images, host_labels, loaded_bytes = gathered
                    pin_split = None
                # CLASS3: the payload's own stamps (I2) and row digests (I1)
                # ride into device staging; both are None when disarmed and
                # the call is then exactly the pre-existing one.
                payload_stamps = (
                    None if resident_slice is not None or pin_split is not None
                    else _payload_stamps(gathered)
                )
                payload_digests = (
                    None if resident_slice is not None else _payload_row_digests(gathered)
                )
                stage_started = time.perf_counter()
                self._delivery_local.role = physical_role
                self._delivery_local.panel = panel_state
                with self._pinned_ring_lock:
                    input_leases = tuple(
                        (id(tensor), self._pinned_slot_state.get(id(tensor)))
                        for tensor in (host_images, host_labels) if tensor is not None
                    )
                try:
                    (
                        batch,
                        hits,
                        misses,
                        transferred,
                        copy_start,
                        copy_end,
                        device_phase_timings,
                    ) = (
                        self._device_batch_resident_slice(resident_slice)
                        if resident_slice is not None
                        else self._device_batch(
                            ids,
                            host_images,
                            host_labels,
                            pin_split=pin_split,
                            expected_host_checksums=payload_stamps,
                            row_digests=payload_digests,
                        )
                    )
                except BaseException as error:
                    # Keep detected delivery failures for attempt settlement,
                    # including when the consumer failed before this result.
                    record_integrity_error(error)
                    if resident_slice is None:
                        # A failed copy may have read these owners without
                        # returning its completion event. Finish that read
                        # before returning either ordinary or split leases.
                        if self._copy_stream is not None:
                            self._copy_stream.synchronize()
                        with self._pinned_ring_lock:
                            for tensor_id, lease in input_leases:
                                if (lease is not None
                                        and self._pinned_slot_state.get(tensor_id) is lease
                                        and lease.get("event") is None):
                                    lease["leased"] = False
                                    lease["staged_checksum"] = None
                    raise
                return (
                    batch,
                    hits,
                    misses,
                    transferred,
                    loaded_bytes,
                    float(gather_seconds),
                    time.perf_counter() - stage_started,
                    copy_start,
                    copy_end,
                    device_phase_timings,
                )

            def device_iterator():
                nonlocal first_device_batch
                for index, chunk in enumerate(chunks):
                    if index == 0 and first_device_batch is not None:
                        initial = first_device_batch
                        first_device_batch = None
                        yield initial
                        del initial
                        continue
                    try:
                        self._delivery_local.role = physical_role
                        self._delivery_local.panel = panel_state
                        self._delivery_local.resident_name = resident_name
                        yield self._device_gather_batch(chunk)
                    except Exception as error:
                        self._device_gather_failure(error)
                        raise

            device_staged = device_iterator() if self._device_gather is not None else PooledBackgroundIterator(
                    host_staged,
                    stage_device,
                    workers=1,
                    depth=int(self.spec.prefetch_depth),
                    executor=(None if persistent_pool is None else persistent_pool.stage_executor),
                    on_discard_input=self._release_unstaged_gather,
            )
            staged = iter(device_staged)
            exhausted = False
            pass_error: BaseException | None = None
            try:
                while True:
                    wait_started = time.perf_counter()
                    try:
                        (
                            batch,
                            batch_hits,
                            batch_misses,
                            batch_h2d,
                            loaded_bytes,
                            gather_seconds,
                            stage_seconds,
                            copy_start,
                            copy_end,
                            device_phase_timings,
                        ) = next(staged)
                    except StopIteration:
                        # CLASS3: every chunk of THIS panel was staged and
                        # handed to the consumer, so this panel's stage
                        # worker no longer writes into the device rings and
                        # the consumer may re-execute the pass.  (This
                        # does not mean the plane is idle --
                        # lookahead and speculative gathers of OTHER posted
                        # looks may still be staging pinned HOST slots; they
                        # never touch the device rings, and a retry re-serves
                        # through the same lease/stamp discipline.)
                        exhausted = True
                        break
                    wait_seconds += time.perf_counter() - wait_started
                    if batch_count == 0:
                        first_batch_latency = time.perf_counter() - started
                    cache_hits += batch_hits
                    cache_misses += batch_misses
                    read_bytes += loaded_bytes
                    h2d_bytes += batch_h2d
                    host_gather_activity += float(gather_seconds)
                    device_stage_activity += float(stage_seconds)
                    cache_lookup_activity += float(
                        device_phase_timings["device_cache_lookup_seconds"]
                    )
                    cache_insert_activity += float(
                        device_phase_timings["device_cache_insert_seconds"]
                    )
                    h2d_enqueue_activity += float(device_phase_timings["h2d_enqueue_seconds"])
                    normalization_enqueue_activity += float(
                        device_phase_timings["normalization_enqueue_seconds"]
                    )
                    if not active_device_look and copy_start is not None and copy_end is not None:
                        copy_event_pairs.append((copy_start, copy_end))
                    batch_count += 1
                    delivered_batch_sizes.append(len(batch.context_ids))
                    delivered += len(batch.context_ids)
                    if self._device_gather is not None:
                        self._device_gather.delivered_rows += len(batch.context_ids)
                    yield batch
            except BaseException as error:
                pass_error = error
                raise
            finally:
                # A suspended panel can still have queued stages. Join this
                # pass's feeders and stages before checking probes or freeing
                # rings; the persistent executors remain available for retry.
                # A close fault must not replace the pass's own error (an OOM
                # drives the physical-plan backoff); it is attached as a note.
                # A consumer's close (GeneratorExit) carries no error of its
                # own, so there the close fault propagates. A close that timed
                # out always propagates: a stage may still be writing, so the
                # run stops (the pass's error chained) rather than retrying.
                try:
                    device_staged.close()
                except PipelineCloseTimeout as close_error:
                    if pass_error is not None and not isinstance(pass_error, GeneratorExit):
                        close_error.add_note(f"raised while closing after: {pass_error!r}")
                    raise
                except BaseException as close_error:
                    if pass_error is None or isinstance(pass_error, GeneratorExit):
                        raise
                    pass_error.add_note(f"staging pipeline close also failed: {close_error!r}")
                finally:
                    staged.close()
                elapsed = time.perf_counter() - started
                copy_normalize_seconds = 0.0
                # Settle the delivered-bytes audit at the panel's natural
                # synchronization point; any mismatch fails the run here
                # rather than letting corrupted rows shape science.
                # CLASS3: armed, a failed settle still builds
                # and appends this panel's receipt (flagged
                # ``integrity_failed``) before the error propagates, so a
                # discarded attempt is accounted exactly once.  Disarmed,
                # the pre-existing behaviour (raise first) is unchanged.
                settle_error: BaseException | None = None
                panel_state["drained"] = exhausted
                try:
                    if worker_feed:
                        pass
                    elif self._integrity_scope_depth:
                        self._integrity_scope_drained &= exhausted
                    else:
                        receipt_index = self.receipt_cursor()
                        self._verify_delivery_checksums(
                            drained=exhausted, receipt_start=receipt_index,
                            receipt_stop=receipt_index + 1,
                        )
                except BaseException as error:
                    if not self._delivery_integrity_armed:
                        raise
                    settle_error = error
                    if self._device_gather is not None:
                        reason = f"{type(error).__name__}:{error}"
                        if isinstance(error, DeliveryIntegrityError):
                            if self._device_gather_run is not None:
                                self._device_gather_run.record_identity_error(error)
                            else:
                                self._device_gather.delivery_identity_error = error
                                self._device_gather.failure_details = dict(getattr(error, "details", {}) or {})
                        elif self._device_gather_run is not None:
                            self._device_gather_run.record_failure(reason)
                        else:
                            self._device_gather.failure = reason
                            self._device_gather.failure_details = getattr(error, "details", None)
                consumer_seconds = max(0.0, elapsed - wait_seconds)
                memory_breakdown = None
                if os.environ.get("IRG_RESIDENCY_BREAKDOWN", "").strip().lower() in {"1", "true"}:
                    from mechanistic_systems.measurement.memory_probe import tensor_storage_breakdown

                    memory_breakdown = tensor_storage_breakdown({
                        "device_input_cache": self._device_cache_pages,
                        "workspace_rings": self._workspace_rings,
                        "normalize_rings": self._normalize_rings,
                        "on_target_pin": (self._pin_images, self._pin_labels),
                        "pinned_host_rings": self._pinned_host_rings,
                        **({"device_gather": (
                            self._device_gather.pages, self._device_gather.scratch,
                            self._device_gather.labels,
                            self._device_gather.row_indexes, self._device_gather.slot_indexes,
                            self._device_gather.output_indexes, self._device_gather.label_indexes,
                        )} if self._device_gather is not None else {}),
                    }, device=self.device)
                # Honest coverage: only independently measured consumer-side
                # phase timers count toward accounting; the residual is
                # surfaced instead of being absorbed into a derived term that
                # made the fraction identically 1.0.
                measured_consumer_activity = (
                    device_stage_activity
                    + cache_lookup_activity
                    + cache_insert_activity
                    + h2d_enqueue_activity
                    + normalization_enqueue_activity
                )
                accounted = wait_seconds + measured_consumer_activity
                unattributed_consumer = max(0.0, elapsed - accounted)
                receipt = NativePanelReceipt(
                    settings=self._effective_delivery_settings(),
                    role=resolved_role,
                    requested_context_count=len(requested),
                    delivered_context_count=delivered,
                    batch_rows=rows,
                    batch_count=batch_count,
                    delivered_batch_sizes=tuple(delivered_batch_sizes),
                    cache_hits=cache_hits,
                    cache_misses=cache_misses,
                    bank_read_bytes=read_bytes,
                    host_to_device_bytes=h2d_bytes,
                    producer_wait_seconds=wait_seconds,
                    host_gather_activity_seconds=host_gather_activity,
                    device_stage_activity_seconds=device_stage_activity,
                    device_copy_normalize_seconds=copy_normalize_seconds,
                    device_cache_lookup_seconds=cache_lookup_activity,
                    device_cache_insert_seconds=cache_insert_activity,
                    h2d_enqueue_seconds=h2d_enqueue_activity,
                    normalization_enqueue_seconds=(normalization_enqueue_activity),
                    consumer_compute_controller_seconds=consumer_seconds,
                    phase_accounting_fraction=(
                        1.0 if elapsed <= 0.0 else min(1.0, accounted / elapsed)
                    ),
                    measured_consumer_activity_seconds=(measured_consumer_activity),
                    unattributed_consumer_seconds=unattributed_consumer,
                    bank_gather_bytes_per_second=(
                        float(read_bytes) / host_gather_activity
                        if host_gather_activity > 0.0 and read_bytes > 0
                        else None
                    ),
                    host_to_device_bytes_per_second=(
                        float(h2d_bytes) / copy_normalize_seconds
                        if copy_normalize_seconds > 0.0 and h2d_bytes > 0
                        else None
                    ),
                    first_batch_latency_seconds=first_batch_latency,
                    elapsed_seconds=elapsed,
                    panel_hash=stable_digest(
                        {
                            "context_ids": requested,
                            "role": resolved_role,
                            "batch_rows": rows,
                        }
                    ),
                    optimized_bank=self.optimized_bank,
                    source_id=str(getattr(self.source, "source_id", type(self.source).__name__)),
                    integrity_failed=settle_error is not None,
                    device_memory_breakdown=memory_breakdown,
                    gpu_process_count=(gpu_process_count() if gpu_process_count() > 1 else None),
                    device_gather=self._device_gather_receipt(),
                )
                for counters in (
                    self._role_counters[resolved_role],
                    self._physical_role_counters[physical_role],
                ):
                    counters.calls += 1
                    counters.requested_physical_contexts += len(requested)
                    counters.delivered_physical_contexts += delivered
                    counters.input_cache_hits += cache_hits
                    counters.input_cache_misses += cache_misses
                    counters.bank_read_bytes += read_bytes
                    counters.host_to_device_bytes += h2d_bytes
                    counters.producer_wait_seconds += wait_seconds
                    counters.panel_wall_seconds += elapsed
                    counters.first_batch_latency_seconds += first_batch_latency
                    counters.batch_count += batch_count
                    counters.device_cache_lookup_seconds += cache_lookup_activity
                    counters.device_cache_insert_seconds += cache_insert_activity
                    counters.h2d_enqueue_seconds += h2d_enqueue_activity
                    counters.normalization_enqueue_seconds += normalization_enqueue_activity
                with self._receipt_lock:
                    self._receipts.append(receipt)
                    if copy_event_pairs:
                        queue = (self._image_feed_copy_timings if worker_feed else
                                 self._integrity_scope_copy_timings if self._integrity_scope_depth else
                                 self._pending_copy_timings)
                        queue.append((len(self._receipts) - 1, tuple(copy_event_pairs)))
                panel_state["closed"] = True
                if (gpu_process_count() > 1 and not self._integrity_scope_depth
                        and not worker_feed and not active_device_look):
                    self.release_buffer_rings(include_pinned=True, block=False)
                if settle_error is not None:
                    raise settle_error

        return iterator()

    def role_batch_factory(self, role: str):
        resolved_role = str(role)

        def batches(context_ids: tuple[str, ...]):
            return self.panel_batches(context_ids, role=resolved_role)

        batches.native_panel_data_plane = self  # type: ignore[attr-defined]
        batches.native_panel_role = resolved_role  # type: ignore[attr-defined]
        return batches

    def sized_role_batch_factory(self, role: str):
        """Return a factory whose physical batch size is runtime-selected."""

        resolved_role = str(role)

        def batches(context_ids: tuple[str, ...], batch_rows: int):
            return self.panel_batches(
                context_ids,
                role=resolved_role,
                batch_rows=int(batch_rows),
            )

        batches.native_panel_data_plane = self  # type: ignore[attr-defined]
        batches.native_panel_role = resolved_role  # type: ignore[attr-defined]
        return batches

    def receipt_cursor(self) -> int:
        """Return a stable cursor for phase receipts emitted after this call."""

        with self._receipt_lock:
            return len(self._receipts)

    def receipts_since(self, cursor: int) -> tuple[Mapping[str, Any], ...]:
        start = int(cursor)
        with self._receipt_lock:
            if start < 0 or start > len(self._receipts):
                raise ValueError("native panel receipt cursor is invalid")
            return tuple(receipt.to_dict() for receipt in self._receipts[start:])

    @property
    def receipts(self) -> tuple[Mapping[str, Any], ...]:
        self.resolve_copy_timings(block=True)
        with self._receipt_lock:
            return tuple(receipt.to_dict() for receipt in self._receipts)

    def _effective_delivery_settings(self) -> dict[str, str]:
        values = effective_settings()
        values[DEVICE_GATHER_ENV] = "1" if self._device_gather is not None else "0"
        values[_PINNED_POOL_MAX_BYTES_ENV] = str(self._pinned_pool.max_bytes)
        return values

    def runtime_receipt(self) -> dict[str, Any]:
        self.resolve_copy_timings()
        with self._cache_lock:
            resident_bytes = int(self._cache_bytes)
            resident_contexts = len(self._cache)
        device_cache_reserved_bytes = (
            0 if self._device_cache_pages is None else self._tensor_bytes(self._device_cache_pages)
        )
        workspace_reserved_bytes = sum(
            sum(self._tensor_bytes(slot[name]) for name in ("uint8", "float", "labels"))
            for ring in self._workspace_rings.values()
            for slot in ring
        )
        source_receipt = getattr(self.source, "replica_receipt", None)
        source_receipt_hash = (
            None if source_receipt is None else getattr(source_receipt, "content_hash", None)
        )
        rss_raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        process_peak_rss_bytes = rss_raw if platform.system() == "Darwin" else rss_raw * 1024
        peak_hbm_allocated_bytes = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda" and torch.cuda.is_available()
            else 0
        )
        peak_hbm_reserved_bytes = (
            int(torch.cuda.max_memory_reserved(self.device))
            if self.device.type == "cuda" and torch.cuda.is_available()
            else 0
        )
        payload = {
            "schema_name": NATIVE_PANEL_DATA_PLANE_SCHEMA,
            "effective_settings": self._effective_delivery_settings(),
            "optimized_bank": bool(self.optimized_bank),
            "performance_gate_eligible": bool(self.performance_gate_eligible),
            "unoptimized_source_fallback": not self.optimized_bank,
            "source_id": str(getattr(self.source, "source_id", type(self.source).__name__)),
            "source_receipt_hash": source_receipt_hash,
            "node_local_staged": bool(getattr(self.source, "node_local_staged", False)),
            "pinned_host_rings": {
                "event_syncs": int(self._pinned_pool.event_syncs),
                "ring_count": len(self._pinned_host_rings),
                "slot_count": int(sum(len(r) for r in self._pinned_host_rings.values())),
                "reserved_bytes": int(self._pinned_pool.reserved_bytes),
                "max_reserved_bytes": int(self._pinned_pool.max_bytes),
                "allocations": int(self._pinned_pool.allocations),
                "reuses": int(self._pinned_pool.reuses),
                "size_class": (
                    "exact_256_512_images_other_power_of_two_bytes_v2"
                    if self._pinned_pool.max_bytes else "exact_shape_dtype_layout_v1"
                ),
                "pool_enabled": bool(self._pinned_pool.max_bytes),
                "amortization_scope": "worker_device" if self.device.type == "cuda" and self._pinned_pool.max_bytes else "data_plane",
                "gpu_allocation_comparison": "not_measured",
                "grown_slots": int(self._pinned_pool.grown_slots),
                "overflow_allocations": int(self._pinned_pool.overflow_allocations),
                "pageable_fallback_allocations": int(self._pinned_pool.pageable_allocations),
                "unpooled_pinned_allocations": int(self._pinned_pool.unpooled_allocations),
                "pretouched_bytes": int(self._pinned_pool.pretouched_bytes),
                "idle_slots_reclaimed": int(self._pinned_pool.evictions),
                "delivery_checksums_verified": int(self._delivery_checksums_verified),
                "delivery_checksum_mismatches": int(self._delivery_checksum_mismatches),
                "delivery_checksum_contract": _DELIVERY_CHECKSUM_CONTRACT,
                "lease_lifecycle": ("staged_leased_until_h2d_event_complete_v2"),
            },
            "working_set_prefetch": {
                "enabled": bool(self.spec.working_set_prefetch),
                "panels_submitted": int(self._prefetch_panels),
                "rows_submitted": int(self._prefetch_rows_submitted),
            },
            "bank_staging_seconds": float(getattr(self.source, "staging_elapsed_seconds", 0.0)),
            "bank_page_warm_performed": bool(getattr(self.source, "page_warm_performed", False)),
            "bank_page_warm_seconds": float(getattr(self.source, "page_warm_elapsed_seconds", 0.0)),
            "bank_staging_amortization_scope": (
                "shared_node_content_addressed_replica"
                if bool(getattr(self.source, "node_local_staged", False))
                else "not_staged_or_already_local"
            ),
            "bank_shard_executor": {
                "lifetime": "one_process_pool_reused_across_all_panels_v1",
                "workers": max(1, int(self.spec.producer_workers)),
                "parallel_gather_uses": int(self._bank_shard_executor_uses),
            },
            "node_local_bank_staging": dict(
                getattr(self.source, "node_local_bank_staging", None) or {}
            ),
            "axis_hash": self.axis.content_hash,
            "preprocessing_hash": self._preprocessing_hash,
            "model_state_hash": self.model_state_hash,
            "execution_spec": {
                **self.spec.to_runtime_dict(),
                "effective_settings": self._effective_delivery_settings(),
            },
            "device_cache": {
                "resident_bytes": resident_bytes,
                "resident_contexts": resident_contexts,
                "reserved_bytes": int(device_cache_reserved_bytes),
                "byte_limit": int(self.spec.device_cache_bytes),
                "storage": "contiguous_uint8_device_pages_v2",
            },
            "workspace_rings": {
                "ring_count": len(self._workspace_rings),
                "reserved_bytes": int(workspace_reserved_bytes),
                "depth": int(self.spec.prefetch_depth),
                "storage": "preallocated_uint8_float_label_ring_v1",
            },
            "process_peak_rss_bytes": int(process_peak_rss_bytes),
            "peak_hbm_allocated_bytes": peak_hbm_allocated_bytes,
            "peak_hbm_reserved_bytes": peak_hbm_reserved_bytes,
            "roles": {role: counters.to_dict() for role, counters in self._role_counters.items()},
            "physical_roles": {
                role: counters.to_dict()
                for role, counters in self._physical_role_counters.items()
            },
            "semantic_role_to_physical_role": dict(
                _PHYSICAL_ROLE_BY_SEMANTIC_ROLE
            ),
            "batch_policy": (
                None
                if self._batch_policy is None
                else {
                    "policy": self._batch_policy.receipt(),
                    "applied_rows_by_role": dict(sorted(self._batch_policy_applied_rows.items())),
                    "policy_source_role_by_semantic_role": dict(
                        sorted(self._batch_policy_source_role.items())
                    ),
                }
            ),
            "receipt_count": len(self._receipts),
        }
        if gpu_process_count() > 1:
            payload["process_device_memory"] = (
                dict(self._process_device_memory)
                if self._process_device_memory is not None
                else {"process_count": gpu_process_count(), "scope": "cpu"}
            )
            payload["device_cache"]["byte_limit"] = process_device_memory_budget_bytes(
                int(self.spec.device_cache_bytes)
            )
        if self._device_gather_requested:
            payload["device_gather"] = self._device_gather_receipt()
        pool = self._persistent_pool
        payload["persistent_panel_producer"] = {
            "enabled": bool(self._persistent_producer_enabled),
            "persistent_pool": pool is not None,
            "pool_reuse_count": (0 if pool is None else int(pool.reuse_count)),
            "lookahead": {
                "looks_configured": int(self._lookahead_looks),
                "enqueued": 0 if pool is None else int(pool.enqueued),
                "ready_hits": (0 if pool is None else int(pool.ready_hits)),
                "on_demand_misses": (0 if pool is None else int(pool.on_demand_misses)),
                "exposed_wait_seconds_estimate": (
                    0.0 if pool is None else float(pool.exposed_wait_seconds)
                ),
                "posted_looks": (0 if pool is None else int(pool.posted_look_count)),
                "stale_flushed_chunks": (0 if pool is None else int(pool.stale_flushed_chunks)),
                "budget_exhausted_posts": (
                    0 if pool is None else int(pool.budget_exhausted_posts)
                ),
                "resident_chunk_budget": (
                    0 if pool is None else int(pool.lookahead_budget_limit)
                ),
                "resident_chunk_budget_source": (
                    "unarmed" if pool is None else str(pool.lookahead_budget_source)
                ),
                "salvaged_hits": (0 if pool is None else int(pool.salvaged_hits)),
                "salvaged_rows": (0 if pool is None else int(pool.salvaged_rows)),
                "salvaged_partial_hits": (
                    0 if pool is None else int(pool.salvaged_partial_hits)
                ),
                "topup_rows": (0 if pool is None else int(pool.topup_rows)),
                "demand_evicted_chunks": (
                    0 if pool is None else int(pool.demand_evicted_chunks)
                ),
                "scheduling": (
                    {
                        "effective_settings": self._effective_delivery_settings(),
                        "identity": _GATHER_SCHEDULING_IDENTITY,
                        "workers": int(self.spec.producer_workers),
                        "submitted_by_priority": {
                            "demand": 0,
                            "exact": 0,
                            "speculative": 0,
                        },
                        "started_by_priority": {
                            "demand": 0,
                            "exact": 0,
                            "speculative": 0,
                        },
                        "completed_by_priority": {
                            "demand": 0,
                            "exact": 0,
                            "speculative": 0,
                        },
                        "cancelled_before_start": 0,
                        "queue_high_water": 0,
                    }
                    if pool is None
                    else {
                        **pool.gather_executor.receipt(),
                        "effective_settings": self._effective_delivery_settings(),
                    }
                ),
                "demand_promotions": (0 if pool is None else int(pool.demand_promotions)),
                "demand_promoted_rows": (0 if pool is None else int(pool.demand_promoted_rows)),
                "queued_speculative_promoted": (
                    0 if pool is None else int(pool.queued_speculative_promoted)
                ),
                "queued_lookahead_cancelled_for_demand": (
                    0 if pool is None else int(pool.queued_lookahead_cancelled_for_demand)
                ),
                "speculative_evicted_for_demand": (
                    0 if pool is None else int(pool.speculative_evicted_for_demand)
                ),
                "speculative_budget_refusals_preserving_exact": (
                    0
                    if pool is None
                    else int(pool.speculative_budget_refusals_preserving_exact)
                ),
                "demand_ready_rows": (0 if pool is None else int(pool.demand_ready_rows)),
                "demand_waited_rows": (0 if pool is None else int(pool.demand_waited_rows)),
                "row_index": (
                    {
                        "identity": ("context_id_to_shared_entry_row_locator_multimap_v1"),
                        "precedence": ("exact_confirmed>exact>speculative;done>running>queued"),
                        "requests": 0,
                        "hits": 0,
                        "misses": 0,
                        "declined_rows": 0,
                        "assembly_failed_rows": 0,
                        "pin_row_locator_refusals": 0,
                        "pin_row_locator_hits": 0,
                        "pin_row_locator_topup_rows": 0,
                        "direct_pinned_assembly_chunks": 0,
                        "direct_pinned_assembly_rows": 0,
                        "pageable_to_pinned_copy_elided_bytes": 0,
                        "resident_pinned_slice_chunks": 0,
                        "resident_pinned_slice_rows": 0,
                        "resident_pinned_slice_host_copy_elided_bytes": 0,
                        "resident_pinned_slice_pin_split_chunks": 0,
                        "resident_pinned_slice_host_rows": 0,
                        "resident_pinned_slice_pin_rows": 0,
                        "host_pin_split_rows_supported": True,
                        "device_pin_rows_supported": True,
                    }
                    if pool is None
                    else {
                        "identity": ("context_id_to_shared_entry_row_locator_multimap_v1"),
                        "precedence": ("exact_confirmed>exact>speculative;done>running>queued"),
                        "requests": int(pool.row_index_requests),
                        "hits": int(pool.row_index_hits),
                        "misses": int(pool.row_index_misses),
                        "declined_rows": int(pool.row_index_declined_rows),
                        "assembly_failed_rows": int(pool.row_index_assembly_failed_rows),
                        "pin_row_locator_refusals": int(pool.pin_row_locator_refusals),
                        "pin_row_locator_hits": int(pool.pin_row_locator_hits),
                        "pin_row_locator_topup_rows": int(pool.pin_row_locator_topup_rows),
                        "direct_pinned_assembly_chunks": int(
                            pool.direct_pinned_assembly_chunks
                        ),
                        "direct_pinned_assembly_rows": int(pool.direct_pinned_assembly_rows),
                        "pageable_to_pinned_copy_elided_bytes": int(
                            pool.pageable_to_pinned_copy_elided_bytes
                        ),
                        "resident_pinned_slice_chunks": int(pool.resident_pinned_slice_chunks),
                        "resident_pinned_slice_rows": int(pool.resident_pinned_slice_rows),
                        "resident_pinned_slice_host_copy_elided_bytes": int(
                            pool.resident_pinned_slice_host_copy_elided_bytes
                        ),
                        "resident_pinned_slice_pin_split_chunks": int(
                            pool.resident_pinned_slice_pin_split_chunks
                        ),
                        "resident_pinned_slice_host_rows": int(
                            pool.resident_pinned_slice_host_rows
                        ),
                        "resident_pinned_slice_pin_rows": int(
                            pool.resident_pinned_slice_pin_rows
                        ),
                        "host_pin_split_rows_supported": True,
                        "device_pin_rows_supported": True,
                    }
                ),
                "residency": (
                    {
                        "entries_current": 0,
                        "entries_high_water": 0,
                        "row_locators_current": 0,
                        "row_locators_high_water": 0,
                        "host_bytes_current": 0,
                        "host_bytes_high_water": 0,
                        "permit_unit": "entry_not_byte",
                    }
                    if pool is None
                    else {
                        "entries_current": int(pool.resident_entries_current),
                        "entries_high_water": int(pool.resident_entries_high_water),
                        "row_locators_current": int(pool.resident_row_locators_current),
                        "row_locators_high_water": int(pool.resident_row_locators_high_water),
                        "host_bytes_current": int(pool.resident_bytes_current),
                        "host_bytes_high_water": int(pool.resident_bytes_high_water),
                        "permit_unit": "entry_not_byte",
                    }
                ),
                "resident_state": (
                    {
                        "entry_counts": {
                            entry_class: {
                                state: 0
                                for state in (
                                    "done",
                                    "running",
                                    "queued",
                                    "cancelled",
                                )
                            }
                            for entry_class in (
                                "exact_confirmed",
                                "exact",
                                "speculative",
                            )
                        },
                        "permits": {
                            "acquired": 0,
                            "released": 0,
                            "current": 0,
                            "high_water": 0,
                            "limit": 0,
                            "exact_once_balance": True,
                        },
                    }
                    if pool is None
                    else self._lookahead_resident_state_receipt(pool)
                ),
                "salvage_declined_low_coverage": (
                    0 if pool is None else int(pool.salvage_declined_low_coverage)
                ),
                "speculative_salvaged": (0 if pool is None else int(pool.speculative_salvaged)),
                "mismatch_diagnostic": (
                    {}
                    if pool is None
                    else {
                        "miss_never_posted": int(pool.miss_never_posted),
                        "miss_covered_salvage_unavailable": int(
                            pool.miss_covered_salvage_unavailable
                        ),
                        "miss_partial_overlap": int(pool.miss_partial_overlap),
                        "samples": tuple(pool.mismatch_samples),
                    }
                ),
                "deduplicated_chunks": (0 if pool is None else int(pool.deduplicated_chunks)),
                "speculative_extension_enabled": bool(self._speculative_extension_env),
                "speculative_posted": (0 if pool is None else int(pool.speculative_posted)),
                "speculative_consumed": (0 if pool is None else int(pool.speculative_consumed)),
                "speculative_flushed": (0 if pool is None else int(pool.speculative_flushed)),
                "speculative_confirmed": (
                    0 if pool is None else int(pool.speculative_confirmed)
                ),
                "prediction_generation_reconciliation": (
                    {
                        "identity": "role_and_provider_generation_complete_resolution_v1",
                        "generations_reconciled": 0,
                        "entries_reconciled": 0,
                        "entries_retired": 0,
                        "entries_fully_retired_at_resolution": 0,
                        "rows_retired": 0,
                        "partial_entries": 0,
                        "deferred_by_claims_or_holds": 0,
                        "unmatched_noops": 0,
                        "mismatched_generation_entries_untouched": 0,
                        "role_mismatch_refusals": 0,
                        "invalid_identity_refusals": 0,
                        "invalid_resolution_mode_refusals": 0,
                        "legacy_posts_remain_consumption_keyed": True,
                    }
                    if pool is None
                    else {
                        "identity": "role_and_provider_generation_complete_resolution_v1",
                        "generations_reconciled": int(
                            pool.prediction_generations_reconciled
                        ),
                        "entries_reconciled": int(
                            pool.prediction_generation_entries_reconciled
                        ),
                        "entries_retired": int(
                            pool.prediction_generation_entries_retired
                        ),
                        "entries_fully_retired_at_resolution": int(
                            pool.prediction_generation_entries_retired
                        ),
                        "rows_retired": int(pool.prediction_generation_rows_retired),
                        "partial_entries": int(
                            pool.prediction_generation_partial_entries
                        ),
                        "deferred_by_claims_or_holds": int(
                            pool.prediction_generation_deferred_by_claims_or_holds
                        ),
                        "unmatched_noops": int(
                            pool.prediction_generation_unmatched_noops
                        ),
                        "mismatched_generation_entries_untouched": int(
                            pool.prediction_generation_mismatched_entries_untouched
                        ),
                        "role_mismatch_refusals": int(
                            pool.prediction_generation_role_mismatch_refusals
                        ),
                        "invalid_identity_refusals": int(
                            pool.prediction_generation_invalid_identity_refusals
                        ),
                        "invalid_resolution_mode_refusals": int(
                            pool.prediction_generation_invalid_resolution_mode_refusals
                        ),
                        "legacy_posts_remain_consumption_keyed": True,
                    }
                ),
            },
            "degradation_reasons": tuple(
                (
                    *self._persistent_producer_degradations,
                    *(() if pool is None else pool.degradation_reasons),
                )
            ),
            "closed": bool(self._persistent_pool_closed or (pool is not None and pool.closed)),
        }
        with self._normalize_ring_lock:
            payload["panel_normalize_buffer_reuse"] = {
                "enabled": bool(self._normalize_buffer_reuse),
                "buffers": int(self._normalize_ring_buffers),
                "reserved_bytes": int(self._normalize_ring_bytes),
                "hits": int(self._normalize_ring_hits),
                # Each leased slot replaces one uint8 staging and one
                # fp32 normalize allocation.
                "allocations_avoided": int(2 * self._normalize_ring_hits),
            }
        if self._image_gather_enabled:
            payload["image_gather"] = {
                "enabled": True,
                "workers": int(self.spec.producer_workers),
                "direct_chunks": int(self._direct_gather_chunks),
                "direct_bytes": int(self._direct_gather_bytes),
                "bank_page_advice": dict(self._bank_page_advice),
            }
        if self._bank_page_advice_enabled:
            payload["bank_page_advice"] = {"enabled": True, "advice": dict(self._bank_page_advice)}
        if self._deep_image_lookahead:
            payload["deep_image_lookahead"] = {
                "enabled": True,
                "future_looks": 3,
                "branch_first_look": False,
                "branch_first_look_status": "off_pending_gpu_hit_receipt",
                "reserved_pinned_byte_limit": self._pinned_pool.max_bytes or None,
            }
        # Surface the completed atomic bind transaction.  The
        # evaluator receipt carries the exact refusal reason and owner /
        # consumer inventory; raw delivery remains the independent live-
        # state witness for engagement.
        folded_binding = self._uint8_folded_stem_binding
        if self._uint8_folded_stem_env or self._raw_panel_delivery or folded_binding is not None:
            binding_details = (
                {}
                if folded_binding is None
                else {
                    key: value
                    for key, value in folded_binding.items()
                    if key not in {"enabled", "engaged", "compile_mode"}
                }
            )
            if not self._raw_panel_delivery and "reason" not in binding_details:
                binding_details["reason"] = "awaiting_bind_uint8_folded_stem_transaction"
            payload["folded_stem"] = {
                "enabled": bool(
                    self._uint8_folded_stem_env
                    if folded_binding is None
                    else folded_binding["enabled"]
                ),
                "engaged": bool(self._raw_panel_delivery),
                "compile_mode": (
                    None if folded_binding is None else folded_binding.get("compile_mode")
                ),
                **binding_details,
            }
        pin_rows = len(self._pin_slot_by_context)
        gathered_rows = int(self._pin_gathered_rows)
        payload["on_target_device_pin"] = {
            "enabled": bool(self._on_target_pin_env),
            "installed_target_class_id": self._pin_target_class_id,
            "built": bool(self._pin_built),
            "serve_active": bool(self._pin_serve_active()),
            "pin_rows": int(pin_rows),
            "pin_bytes": int(self._pin_bytes),
            "pin_hit_rows": int(self._pin_hit_rows),
            "pin_gathered_rows": gathered_rows,
            "pin_hit_fraction": (
                0.0 if gathered_rows <= 0 else float(self._pin_hit_rows) / float(gathered_rows)
            ),
            "degradation_reasons": tuple(self._pin_degradations),
        }
        with self._pinned_ring_lock:
            guard_receipt = _lane_g.row_digest_guard(self).receipt()
            payload["delivery_integrity"] = {
                "environment": DELIVERY_INTEGRITY_ENVIRONMENT,
                "armed": bool(self._delivery_integrity_armed),
                # Not gated by the kill switch; named here so an armed
                # receipt states which lifetime contract the run used.
                "stream_lifetime_contract": STREAM_LIFETIME_CONTRACT,
                "row_digest": {
                    "algorithm": ROW_DIGEST_ALGORITHM,
                    "chunks_checked": int(self._row_digest_chunks_checked),
                    "rows_checked": int(self._row_digest_rows_checked),
                    "mismatches": int(self._row_digest_mismatches),
                    "mismatches_by_stage": dict(self._row_digest_mismatches_by_stage),
                    "first_use_probes": int(self._row_digest_first_use_probes),
                    "after_use_probes": int(self._row_digest_after_use_probes),
                    "gather_witness_rows": int(self._row_digest_gather_witness_rows),
                    "rows_uncovered": int(self._row_digest_rows_uncovered),
                    "pending_pairs": len(self._row_digest_pairs),
                    "pin_rows_digested": (
                        0
                        if self._pin_row_digests_host is None
                        else int(self._pin_row_digests_host.shape[0])
                    ),
                    "samples": tuple(self._row_digest_mismatch_samples),
                    "lane_g": guard_receipt,
                    "rows_delivered": guard_receipt["rows_delivered"],
                    "rows_discarded": guard_receipt["rows_discarded"],
                    "chunks_discarded": guard_receipt["chunks_discarded"],
                },
                "consumer_label_checks": {
                    name: dict(counts)
                    for name, counts in sorted(self._consumer_label_checks.items())
                },
                "payload_stamp": {
                    "contract": PAYLOAD_STAMP_CONTRACT,
                    "issued": int(self._payload_stamps_issued),
                },
            }
        if self._delivery_integrity_armed:
            guard_receipt = payload["delivery_integrity"]["row_digest"]["lane_g"]
            discarded = guard_receipt["rows_discarded"]
            if discarded and any(
                counts["delivered_physical_contexts"] > 0
                for counts in payload["roles"].values()
            ):
                active_roles = [
                    role for role, counts in payload["roles"].items()
                    if counts["delivered_physical_contexts"] > 0
                ]
                if len(active_roles) != 1 or payload["roles"][active_roles[0]]["delivered_physical_contexts"] < discarded:
                    raise RuntimeError("G discarded rows cannot be assigned to one plane role")
                payload["roles"][active_roles[0]]["delivered_physical_contexts"] -= discarded
        payload["content_hash"] = runtime_receipt_hash(payload, hash_function=stable_digest)
        return payload


__all__ = [
    "DELIVERY_INTEGRITY_ENVIRONMENT",
    "DeliveryChecksumIntegrityError",
    "DeliveryIntegrityError",
    "NATIVE_PANEL_DATA_PLANE_SCHEMA",
    "NATIVE_PANEL_RECEIPT_SCHEMA",
    "NativePanelDataPlane",
    "NativePanelExecutionSpec",
    "NativePanelReceipt",
    "PAYLOAD_STAMP_CONTRACT",
    "ROW_DIGEST_ALGORITHM",
    "RowDigestIntegrityError",
    "STREAM_LIFETIME_CONTRACT",
    "byte_row_digests_device",
    "byte_row_digests_host",
    "delivery_integrity_enabled",
    "digest_words_device",
    "digest_words_host",
    "row_digests_device",
    "row_digests_host",
]
