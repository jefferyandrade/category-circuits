"""The CLASS3 guard's production mode.

Full guard coverage -- every delivered row, all four stages (gather, device
after copy, first use, after use) -- at a cost that fits the single-category
floor.  Three pieces, all computing the SAME v2 row digest the plane defines
(``native_panel_data_plane.ROW_DIGEST_ALGORITHM``), as identical integers:

G1  ``digest_rows_fused_device``: one fused exact-integer kernel (Triton) that
    reads each uint8 row once, in its physical layout (channels_last or
    contiguous), forms the row's logical little-endian 32-bit words, weights
    them with the v2 odd lane weights in uint32 wrap-around arithmetic (the
    digest is defined modulo 2**32, so this is exact), with the weights
    formed once per program for several rows, and reduces per-block partial
    lanes that are folded by an exact int64 sum and ``_digest_finish``.  Off
    CUDA (or without Triton, or after a Triton failure, which is receipted)
    the same call runs the plane's pure-torch reference, which computes the
    identical integers.  The Triton path is [U] until GPU acceptance runs
    ``fused_digest_self_test``.

G2  ``BankRowDigestTable``: the v2 digest of every bank row, computed once per
    bank, bound to the bank's certified sha256, to the node-local staging
    completion receipt and to the bank file's stat, and stored node-locally
    next to the replica (8 bytes per row).  ``expected(row_indices)`` is the
    per-row authority that replaces the gather-time host digest: the data
    plane's gather attaches ``gather_authority_digests`` (the table's entries
    for the ordinals it read) instead of computing ``row_digests_host``, and
    the settle compares every row with ``authority_digests_for`` (the table
    at ordinals re-derived from the context ids, independently of the
    gather) -- the gather stage at full coverage, with no bank re-read.

G3  ``normalize_with_first_use_digest``: the first-use probe fused into the
    uint8 -> float32 normalize, which reads the bytes anyway (same kernel,
    same tile), where the normalize path allows it (CUDA, Triton, a
    per-channel decode); otherwise the exact torch normalize runs and the
    first-use probe stays on the consumer stream.  Without a probe the same
    kernel normalizes alone (one read, one write instead of the three torch
    passes).

G4  ``IRG_READOUT_GUARD_MODE``: ``full`` (default), ``sampled`` (first-use and
    after-use probes on a seeded 1-in-16 sample of chunks; the after-copy
    digest and the table authority stay on every row) and ``disarmed`` (no
    row-digest (I1) work; receipted as disarmed rows, never as uncovered
    ones).  The modes govern the I1 row digests only; the payload stamps
    (I2), the label checks and the slab digests (I3) are unchanged, and
    ``IRG_READOUT_LABEL_INTEGRITY=0`` remains the kill switch for all of
    CLASS3.


"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from operator import attrgetter, itemgetter
from pathlib import Path
from typing import Any

import numpy as np
from .run_products import effective_settings

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - Triton exists only on CUDA builds
    import triton
    import triton.language as tl

    try:
        from triton.language.extra import libdevice
    except Exception:
        from triton.language.extra.cuda import libdevice
    _TRITON_IMPORT_ERROR: str | None = None
except Exception as _error:  # pragma: no cover - platform without Triton
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    libdevice = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = repr(_error)


# ---------------------------------------------------------------------------
# Switches
# ---------------------------------------------------------------------------

READOUT_GUARD_MODE_ENVIRONMENT = "IRG_READOUT_GUARD_MODE"
GUARD_MODE_FULL = "full"
GUARD_MODE_SAMPLED = "sampled"
GUARD_MODE_DISARMED = "disarmed"
GUARD_MODE_INSPECTION = "inspection"
GUARD_MODES = (GUARD_MODE_FULL, GUARD_MODE_SAMPLED, GUARD_MODE_DISARMED)
SAMPLED_USE_PROBE_PERIOD = 16
SAMPLED_USE_PROBE_SEED = 0x20260925_0003
SAMPLED_USE_PROBE_POLICY = "splitmix64_chunk_ordinal_one_in_16_v1"

BANK_ROW_DIGEST_TABLE_ENVIRONMENT = "IRG_BANK_ROW_DIGEST_TABLE"
TABLE_RESOLUTION_AUTO = "auto"
TABLE_RESOLUTION_OFF = "off"
TABLE_RESOLUTION_REQUIRE = "require"

AUTHORITY_TABLE = "bank_row_digest_table"
AUTHORITY_GATHER = "gather_host_digest"


def readout_guard_mode() -> str:
    """``IRG_READOUT_GUARD_MODE``: full (default), sampled or disarmed."""

    raw = str(os.environ.get(READOUT_GUARD_MODE_ENVIRONMENT, "")).strip().lower()
    if not raw:
        return GUARD_MODE_FULL
    if raw not in GUARD_MODES:
        raise ValueError(
            f"{READOUT_GUARD_MODE_ENVIRONMENT} must be one of {GUARD_MODES}, got {raw!r}"
        )
    return raw


def _splitmix64(value: int) -> int:
    value = (int(value) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def sampled_use_probe_selected(chunk_ordinal: int, *, seed: int = SAMPLED_USE_PROBE_SEED) -> bool:
    """Seeded 1-in-16 selection of a chunk's use probes (G4 ``sampled``).

    Keyed by the plane-lifetime ordinal of the queued chunk, so a re-executed
    pass draws fresh selections and a persistent fault cannot hide on a
    chunk that is never sampled.  Deterministic within a run.
    """

    return _splitmix64(int(seed) ^ int(chunk_ordinal)) % SAMPLED_USE_PROBE_PERIOD == 0


# ---------------------------------------------------------------------------
# G1: the fused exact digest
# ---------------------------------------------------------------------------

FUSED_DIGEST_KERNEL_CONTRACT = (
    "triton_uint8_row_digest_v2_physical_tile_uint32_wrap_rows_per_program_block_partials_v2"
)
FUSED_NORMALIZE_CONTRACT = "uint8_to_fp32_mul_rn_then_add_rn_per_channel_v1"
_DIGEST_MASK32 = 0xFFFFFFFF
_DIGEST_MASK31 = 0x7FFFFFFF
_DIGEST_MAX_WORDS = 1 << 30
_BLOCK_WORDS_4 = 256
_BLOCK_WORDS_1 = 1024
_ROWS_PER_PROGRAM = 8
_MAX_FUSED_CHANNELS = 64
_MAX_GRID_DIM = 65535
_FUSED_NUM_WARPS = 4


# The three helpers below are ONE source text evaluated two ways: numpy runs
# them on uint32 arrays in ``emulate_fused_row_digest_kernel`` (tests,
# locally) and Triton compiles them (``triton.jit`` of the same functions)
# on uint32 tensors.  Every operation is 32-bit unsigned with wrap-around,
# that is, arithmetic modulo 2**32.  That is exact for this digest:
#   * both lanes are defined modulo 2**32 (``_digest_finish`` keeps the low
#     32 bits of lane A and 31 of lane B), and (a*b) and (a+b) modulo 2**32
#     depend only on a and b modulo 2**32;
#   * the plane's signed int32 words equal these unsigned words modulo 2**32;
#   * the weights are the plane's ``_digest_weight_values``: its
#     ``& 0xFFFFFFFF`` multiplies ARE uint32 multiplies, its shifts act on
#     values below 2**32 (logical == arithmetic), and lane A's weight
#     2*index+1 < 2**31 never wraps (index < 2**30).
# 32-bit integer multiply-adds are native on the GPU; int64 multiplies are
# emulated (several instructions each), so this is also the cheap form.
def _v2_lane_a_weight(index):
    return index * 2 + 1


def _v2_lane_b_weight(index):
    mixed = (index + 0x2545F491) * 0x5BD1E995
    mixed = mixed ^ (mixed >> 15)
    mixed = mixed * 0x27D4EB2D
    mixed = mixed ^ (mixed >> 13)
    return ((mixed & 0x3FFF) << 1) | 1


def _v2_masked_product(words, weight):
    # uint32 * uint32 keeps the low 32 bits: the plane's (w * x) & 0xFFFFFFFF.
    return words * weight


if triton is not None:  # pragma: no cover - compiled only where Triton exists
    _tl_lane_a_weight = triton.jit(_v2_lane_a_weight)
    _tl_lane_b_weight = triton.jit(_v2_lane_b_weight)
    _tl_masked_product = triton.jit(_v2_masked_product)

    @triton.jit
    def _uint8_row_digest_kernel(
        x_ptr,
        partial_ptr,
        out_ptr,
        scale_ptr,
        bias_ptr,
        rows,
        stride_row,
        stride_c,
        stride_hw,
        out_stride_row,
        out_stride_c,
        out_stride_hw,
        channels,
        hw_size,
        words_per_channel,
        n_blocks,
        BLOCK_WORDS: tl.constexpr,
        CHANNELS_P2: tl.constexpr,
        WORD_BYTES: tl.constexpr,
        ROWS_PER_PROGRAM: tl.constexpr,
        DIGEST: tl.constexpr,
        NORMALIZE: tl.constexpr,
        RAW: tl.constexpr,
    ):
        # One program = one block of BLOCK_WORDS logical words (in every
        # channel) of ROWS_PER_PROGRAM consecutive rows.  Logical byte
        # (c, hw) of a row lives at row*stride_row + c*stride_c +
        # hw*stride_hw for both channels_last (stride_c = 1, stride_hw = C)
        # and contiguous (stride_c = HW, stride_hw = 1) batches; the tile
        # [BLOCK_WORDS, WORD_BYTES, C] is one contiguous physical region in
        # either layout.  Word i of channel c is logical word
        # c*words_per_channel + i (words never straddle channels: the host
        # requires hw_size % WORD_BYTES == 0).  The weights depend only on
        # the word position, so they are formed once per program and reused
        # for its rows.
        block = tl.program_id(0)
        group = tl.program_id(1)
        word = block * BLOCK_WORDS + tl.arange(0, BLOCK_WORDS)
        byte = tl.arange(0, WORD_BYTES)
        channel = tl.arange(0, CHANNELS_P2)
        hw = word[:, None, None] * WORD_BYTES + byte[None, :, None]
        in_row = (hw < hw_size) & (channel[None, None, :] < channels)
        channel64 = channel[None, None, :].to(tl.int64)
        hw64 = hw.to(tl.int64)
        offsets = channel64 * stride_c + hw64 * stride_hw
        if NORMALIZE:
            out_offsets = channel64 * out_stride_c + hw64 * out_stride_hw
            if not RAW:
                scale = tl.load(scale_ptr + channel, mask=channel < channels, other=0.0)
                bias = tl.load(bias_ptr + channel, mask=channel < channels, other=0.0)
                scale3 = tl.broadcast_to(scale[None, None, :], (BLOCK_WORDS, WORD_BYTES, CHANNELS_P2))
                bias3 = tl.broadcast_to(bias[None, None, :], (BLOCK_WORDS, WORD_BYTES, CHANNELS_P2))
        if DIGEST:
            index = (channel[None, :] * words_per_channel + word[:, None]).to(tl.uint32)
            weight_a = _tl_lane_a_weight(index)
            weight_b = _tl_lane_b_weight(index)
            shift = (byte[None, :, None] * 8).to(tl.uint32)
        for step in range(ROWS_PER_PROGRAM):
            row = group * ROWS_PER_PROGRAM + step
            live = row < rows
            row64 = row.to(tl.int64)
            values = tl.load(x_ptr + row64 * stride_row + offsets, mask=in_row & live, other=0)
            if NORMALIZE:
                # G3: the normalize reads this same tile.  Bitwise the torch
                # sequence ``copy_`` (exact uint8 -> fp32), ``mul_`` (fp32
                # RN), ``add_`` (fp32 RN): explicit RN intrinsics, never an
                # FMA contraction.
                scaled = values.to(tl.float32)
                if not RAW:
                    scaled = libdevice.add_rn(libdevice.mul_rn(scaled, scale3), bias3)
                tl.store(
                    out_ptr + row64 * out_stride_row + out_offsets,
                    scaled,
                    mask=in_row & live,
                )
            if DIGEST:
                words = tl.sum(values.to(tl.uint32) << shift, axis=1)
                lane_a = tl.sum(tl.sum(_tl_masked_product(words, weight_a), axis=1), axis=0)
                lane_b = tl.sum(tl.sum(_tl_masked_product(words, weight_b), axis=1), axis=0)
                partial = (row64 * n_blocks + block) * 2
                tl.store(partial_ptr + partial, lane_a.to(tl.int64), mask=live)
                tl.store(partial_ptr + partial + 1, lane_b.to(tl.int64), mask=live)


@dataclass(frozen=True)
class _FusedLayout:
    rows: int
    channels: int
    hw_size: int
    stride_row: int
    stride_c: int
    stride_hw: int
    word_bytes: int

    @property
    def words_per_channel(self) -> int:
        return self.hw_size // self.word_bytes

    @property
    def block_words(self) -> int:
        return _BLOCK_WORDS_4 if self.word_bytes == 4 else _BLOCK_WORDS_1

    @property
    def n_blocks(self) -> int:
        return max(1, -(-self.words_per_channel // self.block_words))

    @property
    def channels_p2(self) -> int:
        return 1 << max(0, int(self.channels - 1).bit_length())


def _merged_layout(shape: Sequence[int], stride: Sequence[int]) -> tuple[int, int, int, int, int] | None:
    """(channels, hw_size, stride_row, stride_c, stride_hw) or None."""

    if len(shape) == 2:
        return 1, int(shape[1]), int(stride[0]), 0, int(stride[1])
    if len(shape) != 4:
        return None
    _rows, channels, height, width = (int(value) for value in shape)
    stride_row, stride_c, stride_h, stride_w = (int(value) for value in stride)
    if height > 1 and width > 1:
        if stride_h != width * stride_w:
            return None
        stride_hw = stride_w
    elif width > 1:
        stride_hw = stride_w
    else:
        stride_hw = stride_h
    return channels, height * width, stride_row, stride_c, stride_hw


def _fused_layout(images: Any) -> _FusedLayout | None:
    """The kernel's addressing for ``images``, or None (reference path)."""

    if torch is None or not torch.is_tensor(images) or images.dtype != torch.uint8:
        return None
    merged = _merged_layout(tuple(images.shape), tuple(images.stride()))
    if merged is None:
        return None
    channels, hw_size, stride_row, stride_c, stride_hw = merged
    rows = int(images.shape[0])
    row_bytes = channels * hw_size
    if rows <= 0 or row_bytes <= 0 or channels > _MAX_FUSED_CHANNELS:
        return None
    word_bytes = 4 if row_bytes % 4 == 0 else 1
    if word_bytes == 4 and hw_size % 4 != 0:
        return None  # a logical word would straddle two channels
    if row_bytes // word_bytes >= _DIGEST_MAX_WORDS:
        return None
    layout = _FusedLayout(rows, channels, hw_size, stride_row, stride_c, stride_hw, word_bytes)
    if layout.n_blocks > _MAX_GRID_DIM:
        return None
    return layout


_TRITON_STATE: dict[str, Any] = {"disabled_reason": None}


def triton_disabled_reason() -> str | None:
    return _TRITON_STATE["disabled_reason"]


def _disable_triton(error: BaseException) -> None:
    # A synchronous Triton compile/launch failure disables the fused kernel.
    # The torch device kernel retains the integer row identity detector and
    # receipts the capability failure. The explicit offline self-test can
    # require that Triton actually executes.
    _TRITON_STATE["disabled_reason"] = f"{type(error).__name__}: {error}"[:500]


def triton_digest_available(device: Any = None) -> bool:
    if triton is None or torch is None or _TRITON_STATE["disabled_reason"] is not None:
        return False
    if not torch.cuda.is_available():
        return False
    if device is None:
        return True
    return torch.device(device).type == "cuda"


def _finish_partials(partials: Any) -> Any:
    lanes = partials.sum(dim=1)
    lane_a = lanes[:, 0].bitwise_and(_DIGEST_MASK32)
    lane_b = lanes[:, 1].bitwise_and(_DIGEST_MASK32)
    return (lane_b.bitwise_and(_DIGEST_MASK31) << 32) | lane_a


def _launch_fused(
    images: Any,
    layout: _FusedLayout,
    *,
    out: Any = None,
    scale: Any = None,
    bias: Any = None,
    raw: bool = False,
    digest: bool = True,
) -> Any:  # pragma: no cover - CUDA only
    normalize = out is not None
    if normalize:
        out_merged = _merged_layout(tuple(out.shape), tuple(out.stride()))
        if out_merged is None:
            raise ValueError("fused normalize output layout is not mergeable")
        _c, _hw, out_stride_row, out_stride_c, out_stride_hw = out_merged
    else:
        out_stride_row = out_stride_c = out_stride_hw = 0
    if not (normalize or digest):
        raise ValueError("the fused kernel must normalize, digest or both")
    rows_per_launch = _MAX_GRID_DIM * _ROWS_PER_PROGRAM
    outputs = []
    with torch.cuda.device(images.device):
        for start in range(0, layout.rows, rows_per_launch):
            stop = min(layout.rows, start + rows_per_launch)
            rows = stop - start
            block_images = images[start:stop]
            partials = (
                torch.empty((rows, layout.n_blocks, 2), dtype=torch.int64, device=images.device)
                if digest
                else torch.empty((1,), dtype=torch.int64, device=images.device)
            )
            block_out = out[start:stop] if normalize else block_images
            vector_scale = scale if (normalize and not raw) else partials
            vector_bias = bias if (normalize and not raw) else partials
            groups = -(-rows // _ROWS_PER_PROGRAM)
            _uint8_row_digest_kernel[(layout.n_blocks, groups)](
                block_images,
                partials,
                block_out,
                vector_scale,
                vector_bias,
                rows,
                layout.stride_row,
                layout.stride_c,
                layout.stride_hw,
                out_stride_row,
                out_stride_c,
                out_stride_hw,
                layout.channels,
                layout.hw_size,
                layout.words_per_channel,
                layout.n_blocks,
                BLOCK_WORDS=layout.block_words,
                CHANNELS_P2=layout.channels_p2,
                WORD_BYTES=layout.word_bytes,
                ROWS_PER_PROGRAM=_ROWS_PER_PROGRAM,
                DIGEST=bool(digest),
                NORMALIZE=bool(normalize),
                RAW=bool(raw),
                num_warps=_FUSED_NUM_WARPS,
            )
            if digest:
                outputs.append(_finish_partials(partials))
    if not digest:
        return None
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs)


def _torch_row_digests_device(images: Any) -> Any:
    """Block-bounded row digests for devices without the Triton kernel."""
    from mechanistic_systems.measurement import native_panel_data_plane as plane_module
    rows = int(images.shape[0])
    if images.dtype != torch.uint8:
        raise TypeError("row digests are defined on uint8 image rows")
    if rows == 0:
        return torch.empty((0,), dtype=torch.int64, device=images.device)
    row_bytes = int(images[:1].numel())
    word_count = row_bytes // 4 if row_bytes % 4 == 0 else row_bytes
    block_rows = max(1, plane_module._DIGEST_DEVICE_BLOCK_BYTES // max(1, 16 * word_count))
    outputs = []
    for start in range(0, rows, block_rows):
        stop = min(rows, start + block_rows)
        flat = images[start:stop].detach().contiguous().reshape(stop - start, -1)
        outputs.append(plane_module.byte_row_digests_device(flat))
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs)


_FUSED_PATH_COUNTS: dict[str, int] = {
    "triton": 0,
    "reference": 0,
    "triton_normalize_digest": 0,
    "triton_normalize_only": 0,
    "torch_normalize": 0,
}
_FUSED_PATH_LOCK = threading.Lock()


def _count_path(key: str) -> None:
    with _FUSED_PATH_LOCK:
        _FUSED_PATH_COUNTS[key] += 1


def fused_path_counts() -> dict[str, int]:
    with _FUSED_PATH_LOCK:
        return dict(_FUSED_PATH_COUNTS)


def digest_rows_fused_device(images: Any, *, force_reference: bool = False) -> Any:
    """G1: the v2 row digest of a uint8 batch ``[rows, ...]`` (int64 tensor).

    CUDA with Triton and a mergeable layout: one fused kernel reads each row
    once in its physical layout.  Anything else: the plane's pure-torch
    reference.  Both return the identical integers (``row_digests_host``).
    Stream-ordered on the caller's current stream; never synchronizes.
    """

    if torch is None:
        raise RuntimeError("the fused row digest requires PyTorch")
    if images.dtype != torch.uint8:
        raise TypeError("row digests are defined on uint8 image rows")
    rows = int(images.shape[0])
    if rows == 0:
        return torch.empty((0,), dtype=torch.int64, device=images.device)
    if not force_reference and images.is_cuda and triton_digest_available(images.device):
        layout = _fused_layout(images)
        if layout is not None:
            try:
                digests = _launch_fused(images.detach(), layout)
            except Exception as error:  # pragma: no cover - CUDA only
                _disable_triton(error)
            else:
                _count_path("triton")
                return digests
    _count_path("reference")
    return _torch_row_digests_device(images)


def emulate_fused_row_digest_kernel(
    images: Any,
    *,
    block_words: int | None = None,
    rows_per_program: int | None = None,
    normalize: tuple[Any, Any, bool] | None = None,
) -> Any:
    """numpy emulation of ``_uint8_row_digest_kernel``'s exact dataflow.

    Same addressing (``_fused_layout`` over the tensor's own strides and
    storage offset), the same program decomposition (blocks of words x
    groups of ``rows_per_program`` rows, the ragged last group masked), the
    same tile, masked loads, uint32 word formation, the SAME weight and
    product helpers (one source text), uint32 wrap-around block sums, the
    same per-block partials and the same finish.  Runs on CPU; the tests use
    it to certify the kernel's algorithm locally (the Triton compilation
    itself is [U] until GPU acceptance).  Returns None where the dispatcher
    would take the reference path.

    ``normalize=(scale, bias, raw)`` also emulates the G3 store (numpy fp32
    multiply then add, each rounded to nearest) and returns
    ``(digests, normalized)`` with ``normalized`` in logical
    ``[rows, C, H*W]`` order.
    """

    tensor = images.detach().to("cpu") if torch.is_tensor(images) else torch.from_numpy(np.asarray(images))
    layout = _fused_layout(tensor)
    if layout is None:
        return None
    untyped = tensor.untyped_storage()
    storage = (
        torch.empty((0,), dtype=torch.uint8)
        .set_(untyped, 0, (int(untyped.nbytes()),), (1,))
        .numpy()
    )
    base = int(tensor.storage_offset())
    words_per_block = int(block_words or layout.block_words)
    rpp = int(rows_per_program or _ROWS_PER_PROGRAM)
    n_blocks = max(1, -(-layout.words_per_channel // words_per_block))
    groups = -(-layout.rows // rpp)
    partials = np.zeros((layout.rows, n_blocks, 2), dtype=np.int64)
    byte = np.arange(layout.word_bytes, dtype=np.int64)
    channel = np.arange(layout.channels_p2, dtype=np.int64)
    normalized = None
    if normalize is not None:
        scale, bias, raw = normalize
        scale_c = np.zeros((layout.channels_p2,), dtype=np.float32)
        bias_c = np.zeros((layout.channels_p2,), dtype=np.float32)
        if not raw:
            scale_c[: layout.channels] = np.asarray(scale, dtype=np.float32).reshape(-1)
            bias_c[: layout.channels] = np.asarray(bias, dtype=np.float32).reshape(-1)
        normalized = np.full((layout.rows, layout.channels, layout.hw_size), np.nan, dtype=np.float32)
    step = np.arange(rpp, dtype=np.int64)
    for block in range(n_blocks):
        word = block * words_per_block + np.arange(words_per_block, dtype=np.int64)
        hw = word[:, None, None] * layout.word_bytes + byte[None, :, None]
        in_row = (hw < layout.hw_size) & (channel[None, None, :] < layout.channels)
        offsets = channel[None, None, :] * layout.stride_c + hw * layout.stride_hw
        index = (channel[None, :] * layout.words_per_channel + word[:, None]).astype(np.uint32)
        weight_a = _v2_lane_a_weight(index)
        weight_b = _v2_lane_b_weight(index)
        shift = (byte[None, :, None] * 8).astype(np.uint32)
        if normalized is not None:
            where_c = np.broadcast_to(channel[None, None, :], in_row.shape)[in_row]
            where_hw = np.broadcast_to(hw, in_row.shape)[in_row]
        for group in range(groups):
            row = group * rpp + step
            live = row < layout.rows
            mask = in_row[None] & live[:, None, None, None]
            address = base + row[:, None, None, None] * layout.stride_row + offsets[None]
            values = np.where(mask, storage[np.where(mask, address, 0)], 0).astype(np.uint8)
            words = (values.astype(np.uint32) << shift[None]).sum(axis=2, dtype=np.uint32)
            lane_a = _v2_masked_product(words, weight_a[None]).sum(axis=(1, 2), dtype=np.uint32)
            lane_b = _v2_masked_product(words, weight_b[None]).sum(axis=(1, 2), dtype=np.uint32)
            partials[row[live], block, 0] = lane_a[live]
            partials[row[live], block, 1] = lane_b[live]
            if normalized is not None:
                scaled = values.astype(np.float32)
                if not raw:
                    scaled = scaled * scale_c[None, None, None, :]
                    scaled = scaled + bias_c[None, None, None, :]
                for position in np.flatnonzero(live):
                    normalized[row[position], where_c, where_hw] = scaled[position][in_row]
    lanes = partials.sum(axis=1) & _DIGEST_MASK32
    digests = ((lanes[:, 1] & _DIGEST_MASK31) << 32) | lanes[:, 0]
    return digests if normalized is None else (digests, normalized)


# ---------------------------------------------------------------------------
# G3: the first-use probe fused into the normalize
# ---------------------------------------------------------------------------

# Test hook: run the G3 control flow off CUDA (torch normalize + reference
# digest of the same bytes in the same call), so the stage semantics of the
# fused first-use probe can be exercised on CPU.  Never set in production.
_FUSED_NORMALIZE_EMULATION = False


def _per_channel_vector(tensor: Any, channels: int, ndim: int) -> Any:
    """``tensor`` as a contiguous fp32 [C] vector when it broadcasts per
    channel (dimension 1) against a ``ndim``-D batch; otherwise None."""

    if tensor is None or not torch.is_tensor(tensor) or tensor.dtype != torch.float32:
        return None
    if int(tensor.numel()) != int(channels):
        return None
    shape = tuple(int(value) for value in tensor.shape)
    if ndim == 4:
        per_channel = shape in {(1, channels, 1, 1), (channels, 1, 1)} or (channels == 1 and len(shape) <= 4)
    elif ndim == 2:
        per_channel = channels == 1 and len(shape) <= 2
    else:
        per_channel = False
    return tensor.reshape(-1).contiguous() if per_channel else None


def fused_normalize_available(
    uint8_batch: Any,
    inputs: Any,
    decode_scale: Any,
    decode_bias: Any,
    *,
    raw: bool,
) -> bool:
    if torch is None or not torch.is_tensor(uint8_batch) or not uint8_batch.is_cuda:
        return False
    if not triton_digest_available(uint8_batch.device):
        return False
    if inputs is None or inputs.dtype != torch.float32 or tuple(inputs.shape) != tuple(uint8_batch.shape):
        return False
    if inputs.device != uint8_batch.device:
        return False
    layout = _fused_layout(uint8_batch)
    if layout is None or _merged_layout(tuple(inputs.shape), tuple(inputs.stride())) is None:
        return False
    if not raw:
        for tensor in (decode_scale, decode_bias):
            if tensor is None or tensor.device != uint8_batch.device:
                return False
            if _per_channel_vector(tensor, layout.channels, uint8_batch.ndim) is None:
                return False
    return True


def normalize_with_first_use_digest(
    probe: Any,
    uint8_batch: Any,
    inputs: Any,
    *,
    decode_scale: Any,
    decode_bias: Any,
    raw: bool,
) -> str:
    """G3: normalize ``uint8_batch`` into ``inputs``; fuse the first-use digest.

    ``probe`` is the delivery probe ``_queue_row_digest_check`` returned (or
    None).  On CUDA with Triton one kernel reads each byte
    once and writes the normalized rows; when the probe wants a first-use
    digest the same kernel also returns the digest of exactly the bytes it
    read, the probe records it as the chunk's first-use digest and the
    consumer-stream first-use call becomes a no-op. Otherwise the torch
    normalize runs (``copy_``, ``mul_``, ``add_``) and the
    first-use probe stays where it was.  Returns ``"fused"`` (normalize and
    digest), ``"fused_normalize"`` (normalize only), ``"emulated"`` (test
    hook) or ``"torch"``.
    """

    record = getattr(probe, "record_fused_first_use", None)
    wants = bool(record is not None and getattr(probe, "wants_first_use", lambda: False)())
    if fused_normalize_available(
        uint8_batch, inputs, decode_scale, decode_bias, raw=raw
    ):  # pragma: no cover - CUDA only
        layout = _fused_layout(uint8_batch)
        try:
            digest = _launch_fused(
                uint8_batch,
                layout,
                out=inputs,
                scale=None if raw else _per_channel_vector(decode_scale, layout.channels, uint8_batch.ndim),
                bias=None if raw else _per_channel_vector(decode_bias, layout.channels, uint8_batch.ndim),
                raw=raw,
                digest=wants,
            )
        except Exception as error:
            _disable_triton(error)
        else:
            if wants:
                _count_path("triton_normalize_digest")
                record(digest, "fused")
                return "fused"
            _count_path("triton_normalize_only")
            return "fused_normalize"
    _count_path("torch_normalize")
    inputs.copy_(uint8_batch)
    if not raw:
        inputs.mul_(decode_scale).add_(decode_bias)
    if wants and _FUSED_NORMALIZE_EMULATION:
        record(digest_rows_fused_device(uint8_batch), "emulated")
        return "emulated"
    return "torch"


# ---------------------------------------------------------------------------
# G2: the bank row digest table
# ---------------------------------------------------------------------------

BANK_ROW_DIGEST_TABLE_SCHEMA = "irg_bank_row_digest_table_v1"
BANK_ROW_DIGEST_TABLE_CONTRACT = (
    "v2_row_digest_of_every_bank_row_bound_to_certified_sha256_staging_receipt_and_file_stat_v1"
)
_TABLE_FILE_PREFIX = "row_digest_table_v2"


class BankRowDigestTableError(RuntimeError):
    """The table cannot serve as the authority for this bank (binding)."""


class BankRowDigestTableIntegrityError(BankRowDigestTableError):
    """The table's own bytes or binding record are corrupt."""


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _digests_sha256(digests: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(digests, dtype="<i8").tobytes()).hexdigest()


def _file_stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
    }


def _npy_layout(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        offset = int(handle.tell())
    if fortran:
        raise BankRowDigestTableError("bank payload is Fortran-ordered")
    return {"shape": [int(value) for value in shape], "dtype": str(dtype), "data_offset": offset}


def _load_staging_identity(staged_receipt: Any) -> dict[str, Any] | None:
    """Normalize a staging receipt to {receipt_hash, replica_key, files, ...}.

    Accepts the ``resolve_staged_bank`` receipt (``node_local_bank_staging``),
    a completion-receipt mapping, or a path to a completion receipt.  The
    completion receipt's own hash is verified by the loader.
    """

    if staged_receipt is None:
        return None
    from mechanistic_systems.measurement import bank_staging

    if isinstance(staged_receipt, (str, Path)):
        completion = bank_staging._load_completion_receipt(Path(staged_receipt))  # noqa: SLF001
        completion_path = str(Path(staged_receipt).resolve())
    elif isinstance(staged_receipt, Mapping):
        if "receipt_hash" in staged_receipt and "files" in staged_receipt:
            completion = dict(staged_receipt)
            completion_path = None
        else:
            path = staged_receipt.get("completion_receipt_path")
            if not path:
                return {
                    "staging_mode": str(staged_receipt.get("staging_mode")),
                    "receipt_hash": None,
                    "replica_key": staged_receipt.get("replica_key"),
                    "completion_receipt_path": None,
                    "files": None,
                }
            completion = bank_staging._load_completion_receipt(Path(path))  # noqa: SLF001
            completion_path = str(Path(path).resolve())
    else:
        raise TypeError("staged_receipt must be a mapping, a path or None")
    return {
        "staging_mode": "staged",
        "receipt_hash": str(completion.get("receipt_hash")),
        "replica_key": completion.get("replica_key"),
        "completion_receipt_path": completion_path,
        "files": completion.get("files"),
    }


def default_table_path(bank_path: str | Path, bank_sha256: str) -> Path:
    """Node-local home: next to the bank replica, keyed by its sha256."""

    bank = Path(bank_path)
    return bank.with_name(f"{bank.stem}.{_TABLE_FILE_PREFIX}.{str(bank_sha256)[:16]}.npy")


def _binding_path(table_path: Path) -> Path:
    return table_path.with_suffix(".json")


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _host_block_digests(block: np.ndarray) -> np.ndarray:
    from mechanistic_systems.measurement import native_panel_data_plane as plane_module

    return plane_module.row_digests_host(np.ascontiguousarray(block))


@dataclass(frozen=True)
class BankRowDigestTable:
    """G2: the v2 digest of every row of one certified uint8 bank."""

    digests: np.ndarray
    binding: Mapping[str, Any]
    path: Path | None = None

    def __post_init__(self) -> None:
        array = np.asarray(self.digests)
        if array.dtype != np.int64 or array.ndim != 1:
            raise BankRowDigestTableIntegrityError("table digests must be a 1-D int64 array")
        if array.flags.writeable:
            array = array.copy()
            array.setflags(write=False)
            object.__setattr__(self, "digests", array)

    @property
    def rows(self) -> int:
        return int(self.digests.shape[0])

    @property
    def bank_sha256(self) -> str | None:
        bank = self.binding.get("bank") or {}
        return bank.get("sha256")

    @property
    def binding_hash(self) -> str:
        return str(self.binding.get("binding_hash", ""))

    def expected(self, row_indices: Any) -> np.ndarray:
        """The authority digests of bank rows ``row_indices`` (int64, in order)."""

        index = np.asarray(row_indices, dtype=np.int64).reshape(-1)
        if index.size and (int(index.min()) < 0 or int(index.max()) >= self.rows):
            raise IndexError(
                f"bank row index out of range for a {self.rows}-row digest table"
            )
        return self.digests[index]

    def summary(self) -> dict[str, Any]:
        bank = self.binding.get("bank") or {}
        staging = self.binding.get("staging") or {}
        return {
            "schema": self.binding.get("schema"),
            "algorithm": self.binding.get("algorithm"),
            "rows": self.rows,
            "bank_sha256": bank.get("sha256"),
            "sha256_source": bank.get("sha256_source"),
            "staging_receipt_hash": staging.get("receipt_hash"),
            "digests_sha256": (self.binding.get("digests") or {}).get("sha256"),
            "binding_hash": self.binding_hash,
            "in_memory": bool(self.binding.get("in_memory", False)),
            "path": None if self.path is None else str(self.path),
        }

    def validate_file_bank(
        self, bank_path: str | Path, certified_sha256: str | None,
        staged_receipt: Any, expected_shape: Sequence[int] | None = None,
    ) -> str:
        """One file and generation contract for load, bind, and cache reuse."""
        if self.binding.get("in_memory"):
            raise BankRowDigestTableError("an in-memory table has no certified file binding")
        from mechanistic_systems.measurement import native_panel_data_plane as plane_module

        binding_without_hash = dict(self.binding)
        recorded_hash = binding_without_hash.pop("binding_hash", None)
        if (self.binding.get("schema") != BANK_ROW_DIGEST_TABLE_SCHEMA
                or self.binding.get("contract") != BANK_ROW_DIGEST_TABLE_CONTRACT
                or self.binding.get("algorithm") != plane_module.ROW_DIGEST_ALGORITHM
                or recorded_hash != _canonical_sha256(binding_without_hash)):
            raise BankRowDigestTableIntegrityError("table binding integrity mismatch")
        digests_record = self.binding.get("digests") or {}
        if (int(digests_record.get("rows", -1)) != self.rows
                or digests_record.get("sha256") != _digests_sha256(self.digests)):
            raise BankRowDigestTableIntegrityError("table digests integrity mismatch")
        bank_record = self.binding.get("bank") or {}
        bank = Path(bank_path).resolve()
        if certified_sha256 is None or bank_record.get("sha256") != str(certified_sha256):
            raise BankRowDigestTableError("table is not bound to this bank's certified sha256")
        staging = _load_staging_identity(staged_receipt)
        bound_staging = self.binding.get("staging") or {}
        if not staging or not staging.get("receipt_hash") or not bound_staging.get("receipt_hash"):
            raise BankRowDigestTableError("certified table requires a staging receipt identity")
        if (bound_staging.get("receipt_hash") != staging.get("receipt_hash")
                or bound_staging.get("replica_key") != staging.get("replica_key")):
            raise BankRowDigestTableError("table is bound to a different staging receipt")
        entry = (staging.get("files") or {}).get("inputs.npy")
        if not isinstance(entry, Mapping) or entry.get("sha256") != certified_sha256:
            raise BankRowDigestTableError("staging receipt has no matching certified inputs")
        stat = _file_stat(bank)
        for key, value in stat.items():
            if int(bank_record.get(key, -1)) != int(value):
                raise BankRowDigestTableError(f"bank file {key} differs from the table binding")
        for key in ("size_bytes", "mtime_ns", "device", "inode"):
            if int(entry.get(key, -1)) != int(stat[key]):
                raise BankRowDigestTableError(f"bank file {key} differs from the staging receipt")
        layout = _npy_layout(bank)
        if layout["shape"] != list(bank_record.get("shape", [])) or layout["dtype"] != "uint8":
            raise BankRowDigestTableError("bank npy layout differs from the table binding")
        if self.rows != int(layout["shape"][0]):
            raise BankRowDigestTableError("table rows differ from the bank rows")
        if expected_shape is not None and [int(v) for v in expected_shape] != layout["shape"]:
            raise BankRowDigestTableError("table bank shape differs from the source")
        return str(staging["receipt_hash"])

    # -- construction ------------------------------------------------------

    @classmethod
    def _finalize_binding(cls, binding: dict[str, Any], digests: np.ndarray) -> dict[str, Any]:
        from mechanistic_systems.measurement import native_panel_data_plane as plane_module

        binding = dict(binding)
        binding["schema"] = BANK_ROW_DIGEST_TABLE_SCHEMA
        binding["contract"] = BANK_ROW_DIGEST_TABLE_CONTRACT
        binding["algorithm"] = plane_module.ROW_DIGEST_ALGORITHM
        binding["digests"] = {"rows": int(digests.shape[0]), "sha256": _digests_sha256(digests)}
        binding.pop("binding_hash", None)
        binding["binding_hash"] = _canonical_sha256(binding)
        return binding

    @classmethod
    def from_array(
        cls,
        inputs: np.ndarray,
        *,
        source_identity: Mapping[str, Any] | None = None,
        block_rows: int = 4096,
    ) -> "BankRowDigestTable":
        """An in-memory table over an array already in RAM (tests, tools).

        Bound only to ``source_identity`` (for example the source replica
        receipt's content hash); never written and never auto-resolved.
        """

        array = np.asarray(inputs)
        if array.dtype != np.uint8 or array.ndim < 2:
            raise BankRowDigestTableError("a bank digest table needs a uint8 [rows, ...] array")
        rows = int(array.shape[0])
        digests = np.empty((rows,), dtype=np.int64)
        for start in range(0, rows, int(block_rows)):
            stop = min(rows, start + int(block_rows))
            digests[start:stop] = _host_block_digests(array[start:stop])
        binding = {
            "in_memory": True,
            "source_identity": dict(source_identity or {}),
            "bank": {
                "shape": [int(value) for value in array.shape],
                "dtype": "uint8",
                "rows": rows,
                "row_bytes": int(np.prod(array.shape[1:])),
                "sha256": None,
                "sha256_source": None,
            },
            "staging": None,
            "build": {"path": "host_numpy", "device": "cpu"},
        }
        return cls(digests, cls._finalize_binding(binding, digests), None)

    @classmethod
    def build(
        cls,
        bank_path: str | Path,
        staged_receipt: Any = None,
        *,
        certified_sha256: str | None = None,
        verify_sha256: bool | None = None,
        device: Any = None,
        block_rows: int = 512,
        out_path: str | Path | None = None,
        write: bool = True,
    ) -> "BankRowDigestTable":
        """Compute the v2 digest of every row of the ``.npy`` bank at ``bank_path``.

        Binding: the bank's sha256 comes from the staging completion receipt
        (whose file stat must still match the bank) or, without one, from
        ``certified_sha256`` verified inline over the file bytes in the same
        read pass (``verify_sha256``, default on when there is no staging
        receipt).  The table is written node-locally next to the bank
        (``default_table_path``) unless ``write=False``.
        """

        bank = Path(bank_path).resolve()
        layout = _npy_layout(bank)
        if layout["dtype"] != "uint8" or len(layout["shape"]) < 2:
            raise BankRowDigestTableError("the bank payload is not a uint8 [rows, ...] array")
        stat_before = _file_stat(bank)
        staging = _load_staging_identity(staged_receipt)
        staged_sha256 = None
        if staging is not None and staging.get("files"):
            entry = (staging.get("files") or {}).get("inputs.npy")
            if not isinstance(entry, Mapping):
                raise BankRowDigestTableError("staging receipt has no inputs.npy file record")
            for key in ("size_bytes", "mtime_ns", "device", "inode"):
                if int(entry.get(key, -1)) != int(stat_before[key]):
                    raise BankRowDigestTableError(
                        f"bank file {key} differs from the staging completion receipt"
                    )
            staged_sha256 = str(entry.get("sha256"))
        if certified_sha256 is not None and staged_sha256 is not None and certified_sha256 != staged_sha256:
            raise BankRowDigestTableError("certified sha256 differs from the staging receipt")
        if verify_sha256 is None:
            verify_sha256 = staged_sha256 is None
        if staged_sha256 is None and certified_sha256 is None and not verify_sha256:
            raise BankRowDigestTableError("no certified sha256 to bind the table to")
        shape = layout["shape"]
        rows = int(shape[0])
        row_bytes = int(np.prod(shape[1:]))
        digests = np.empty((rows,), dtype=np.int64)
        started = time.perf_counter()
        resolved_device = None if device is None else torch.device(device)
        hasher = hashlib.sha256() if verify_sha256 else None
        path_used = _build_digests(
            bank,
            layout,
            digests,
            block_rows=max(1, int(block_rows)),
            device=resolved_device,
            hasher=hasher,
        )
        seconds = time.perf_counter() - started
        stat_after = _file_stat(bank)
        if stat_after != stat_before:
            raise BankRowDigestTableError("bank file changed while the table was built")
        if hasher is not None:
            computed = hasher.hexdigest()
            expected = certified_sha256 or staged_sha256
            if expected is not None and computed != expected:
                raise BankRowDigestTableError(
                    "bank bytes do not match the certified sha256 "
                    f"(computed {computed}, certified {expected})"
                )
            bank_sha256, sha256_source = computed, "verified_inline"
        else:
            bank_sha256 = staged_sha256 or certified_sha256
            sha256_source = (
                "staging_completion_receipt" if staged_sha256 is not None else "caller_certified"
            )
        binding = {
            "in_memory": False,
            "bank": {
                "path": str(bank),
                **stat_before,
                "data_offset": int(layout["data_offset"]),
                "shape": shape,
                "dtype": "uint8",
                "rows": rows,
                "row_bytes": row_bytes,
                "sha256": bank_sha256,
                "sha256_source": sha256_source,
            },
            "staging": None
            if staging is None
            else {
                "receipt_hash": staging.get("receipt_hash"),
                "replica_key": staging.get("replica_key"),
                "completion_receipt_path": staging.get("completion_receipt_path"),
            },
            "build": {
                "path": path_used,
                "device": "cpu" if resolved_device is None else str(resolved_device),
                "seconds": float(seconds),
                "rows_per_second": float(rows / seconds) if seconds > 0 else None,
                "created_unix": float(time.time()),
            },
        }
        binding = cls._finalize_binding(binding, digests)
        table = cls(digests, binding, None)
        if write:
            target = Path(out_path) if out_path is not None else default_table_path(bank, bank_sha256)
            table = table.write(target)
        return table

    def write(self, path: str | Path) -> "BankRowDigestTable":
        if self.binding.get("in_memory"):
            raise BankRowDigestTableError("an in-memory table is never written as an authority")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        buffer = _NpyBuffer()
        np.save(buffer, np.ascontiguousarray(self.digests, dtype="<i8"), allow_pickle=False)
        _write_bytes_atomic(target, buffer.getvalue())
        _write_bytes_atomic(
            _binding_path(target),
            (json.dumps(dict(self.binding), indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        return BankRowDigestTable(self.digests, self.binding, target)

    @classmethod
    def load(
        cls,
        table_path: str | Path,
        *,
        bank_path: str | Path,
        certified_sha256: str | None,
        staged_receipt: Any = None,
        expected_shape: Sequence[int] | None = None,
    ) -> "BankRowDigestTable":
        """Load and verify a table against the bank it must be the authority for.

        Integrity (raises :class:`BankRowDigestTableIntegrityError`): the
        binding record's own hash, the schema/algorithm/contract, and the
        digests' sha256.  Binding (raises :class:`BankRowDigestTableError`):
        the certified sha256, the bank file's stat (size, mtime, device,
        inode -- the file the staging hashed), its npy layout, and the
        staging receipt hash.  Nothing re-reads the bank.
        """

        from mechanistic_systems.measurement import native_panel_data_plane as plane_module

        target = Path(table_path)
        try:
            binding = json.loads(_binding_path(target).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise BankRowDigestTableIntegrityError(f"unreadable table binding: {error}") from error
        recorded = str(binding.pop("binding_hash", ""))
        if recorded != _canonical_sha256(binding):
            raise BankRowDigestTableIntegrityError("table binding hash mismatch")
        binding["binding_hash"] = recorded
        if (
            binding.get("schema") != BANK_ROW_DIGEST_TABLE_SCHEMA
            or binding.get("contract") != BANK_ROW_DIGEST_TABLE_CONTRACT
            or binding.get("algorithm") != plane_module.ROW_DIGEST_ALGORITHM
            or binding.get("in_memory")
        ):
            raise BankRowDigestTableIntegrityError("table schema, contract or algorithm mismatch")
        digests = np.load(target, allow_pickle=False)
        if digests.dtype != np.int64 or digests.ndim != 1:
            raise BankRowDigestTableIntegrityError("table payload is not a 1-D int64 array")
        record = binding.get("digests") or {}
        if int(record.get("rows", -1)) != int(digests.shape[0]) or record.get("sha256") != _digests_sha256(digests):
            raise BankRowDigestTableIntegrityError("table digests sha256 mismatch")
        table = cls(digests, binding, target)
        table.validate_file_bank(bank_path, certified_sha256, staged_receipt, expected_shape)
        return table


class _NpyBuffer:
    """Minimal in-memory binary sink for ``np.save``."""

    def __init__(self) -> None:
        self._parts: list[bytes] = []

    def write(self, data: Any) -> int:
        payload = bytes(data)
        self._parts.append(payload)
        return len(payload)

    def getvalue(self) -> bytes:
        return b"".join(self._parts)


def _build_digests(
    bank: Path,
    layout: Mapping[str, Any],
    digests: np.ndarray,
    *,
    block_rows: int,
    device: Any,
    hasher: Any,
) -> str:
    """Fill ``digests`` in one sequential read of the bank payload."""

    shape = tuple(int(value) for value in layout["shape"])
    rows = shape[0]
    row_shape = shape[1:]
    row_bytes = int(np.prod(row_shape))
    offset = int(layout["data_offset"])
    use_device = device is not None and device.type != "cpu"
    path = (
        "triton_fused"
        if use_device and device.type == "cuda" and triton is not None
        else ("torch_reference_" + device.type if use_device else "host_numpy")
    )
    buffer = np.empty((block_rows * row_bytes,), dtype=np.uint8)
    pending: list[tuple[int, int, Any, Any]] = []
    with bank.open("rb", buffering=0) as handle:
        header = handle.read(offset)
        if len(header) != offset:
            raise BankRowDigestTableError("short read of the bank header")
        if hasher is not None:
            hasher.update(header)
        for start in range(0, rows, block_rows):
            stop = min(rows, start + block_rows)
            count = (stop - start) * row_bytes
            if use_device:
                # A fresh host buffer per block, referenced until its digest
                # is fetched: no reuse hazard whatever the device copy's
                # host-side completion semantics.
                buffer = np.empty((block_rows * row_bytes,), dtype=np.uint8)
            view = memoryview(buffer)[:count]
            done = 0
            while done < count:
                got = handle.readinto(view[done:])
                if not got:
                    raise BankRowDigestTableError("short read of the bank payload")
                done += got
            if hasher is not None:
                hasher.update(view)
            block = buffer[:count].reshape((stop - start, *row_shape))
            if use_device:
                host = torch.from_numpy(block)
                moved = host.to(device=device)
                if moved.ndim == 4:
                    moved = moved.contiguous(memory_format=torch.channels_last)
                pending.append((start, stop, digest_rows_fused_device(moved), host))
                if len(pending) >= 8:
                    for p_start, p_stop, tensor, _host in pending:
                        digests[p_start:p_stop] = tensor.to("cpu").numpy()
                    pending.clear()
            else:
                digests[start:stop] = _host_block_digests(block)
        if hasher is not None and handle.read(1):
            raise BankRowDigestTableError("bank payload has trailing bytes")
    for p_start, p_stop, tensor, _host in pending:
        digests[p_start:p_stop] = tensor.to("cpu").numpy()
    return path


# ---------------------------------------------------------------------------
# Per-plane guard state and table authority
# ---------------------------------------------------------------------------


@dataclass
class RowDigestGuard:
    """Digest-table state attached lazily to one ``NativePanelDataPlane``."""

    mode: str
    table: BankRowDigestTable | None = None
    table_status: str = "unresolved"
    table_resolved: bool = False
    staging_generation: tuple[Any, ...] | None = None
    fallback_reason: str | None = None
    stage_rows: dict[str, int] = field(default_factory=lambda: {
        "gather_vs_bank": 0, "after_copy": 0, "first_use": 0, "after_use": 0,
    })
    chunks_queued: int = 0
    rows_queued: int = 0
    rows_table_authority: int = 0
    rows_gather_authority: int = 0
    rows_witness_compared: int = 0
    rows_disarmed: int = 0
    rows_discarded: int = 0
    chunks_discarded: int = 0
    discards: list[dict[str, Any]] = field(default_factory=list)
    use_probe_chunks_sampled: int = 0
    use_probe_chunks_skipped: int = 0
    first_use_fused: int = 0
    first_use_emulated: int = 0
    first_use_fused_attempts: int = 0
    first_use_emulated_attempts: int = 0
    rows_gather_table_sourced: int = 0
    authority_cache_hits: int = 0
    authority_cache_misses: int = 0
    lock: Any = field(default_factory=threading.Lock, repr=False)
    # ids tuple -> the table's digests for those rows (read-only arrays).
    # A pure function of the key (the plane's records and the table are
    # immutable), so a hit returns exactly what a recomputation would.
    authority_cache: dict = field(default_factory=dict, repr=False)

    def receipt(self) -> dict[str, Any]:
        with self.lock:
            delivered = self.rows_queued - self.rows_discarded
            return {
                "guard_mode": self.mode,
                "effective_settings": effective_settings(),
                "sampled_use_probe_policy": (
                    SAMPLED_USE_PROBE_POLICY if self.mode == GUARD_MODE_SAMPLED else None
                ),
                "authority": AUTHORITY_TABLE if self.table is not None else AUTHORITY_GATHER,
                "table_status": self.table_status,
                "fallback_reason": self.fallback_reason,
                "stage_rows": dict(self.stage_rows),
                "stage_coverage": {
                    stage: {"rows_covered": count, "rows_delivered": delivered,
                            "complete": count == delivered}
                    for stage, count in self.stage_rows.items()
                },
                "table": None if self.table is None else self.table.summary(),
                "fused_digest_contract": FUSED_DIGEST_KERNEL_CONTRACT,
                "triton_available": triton is not None,
                "chunks_queued": self.chunks_queued,
                "rows_queued": self.rows_queued,
                "rows_delivered": delivered,
                "rows_table_authority": self.rows_table_authority,
                "rows_gather_authority": self.rows_gather_authority,
                "rows_witness_compared": self.rows_witness_compared,
                "rows_disarmed": self.rows_disarmed,
                "rows_discarded": self.rows_discarded,
                "chunks_discarded": self.chunks_discarded,
                "discards": tuple(dict(item) for item in self.discards),
                "use_probe_chunks_sampled": self.use_probe_chunks_sampled,
                "use_probe_chunks_skipped": self.use_probe_chunks_skipped,
                "first_use_fused": self.first_use_fused,
                "first_use_emulated": self.first_use_emulated,
                "first_use_fused_attempts": self.first_use_fused_attempts,
                "first_use_emulated_attempts": self.first_use_emulated_attempts,
                "rows_gather_table_sourced": self.rows_gather_table_sourced,
                "authority_cache_hits": self.authority_cache_hits,
                "authority_cache_misses": self.authority_cache_misses,
                "triton_disabled_reason": triton_disabled_reason(),
                "fused_path_counts": fused_path_counts(),
            }


_GUARD_ATTRIBUTE = "_v6_row_digest_guard"
_GUARD_CREATE_LOCK = threading.Lock()


def row_digest_guard(plane: Any) -> RowDigestGuard:
    guard = getattr(plane, _GUARD_ATTRIBUTE, None)
    if guard is not None:
        return guard
    with _GUARD_CREATE_LOCK:
        guard = getattr(plane, _GUARD_ATTRIBUTE, None)
        if guard is None:
            guard = RowDigestGuard(mode=readout_guard_mode())
            setattr(plane, _GUARD_ATTRIBUTE, guard)
    return guard


def configure_guard_for_inspection(plane: Any) -> RowDigestGuard:
    """Opt a direct, non-pipeline consumer into named inspection coverage.

    Call before any chunks are queued. Authority and after-copy digests remain
    mandatory; use probes are optional because the caller only inspects or
    clones the supplied batch, without a pipeline consumption window.
    """

    guard = row_digest_guard(plane)
    with guard.lock:
        if guard.chunks_queued:
            raise RuntimeError("inspection guard mode must be selected before delivery")
        guard.mode = GUARD_MODE_INSPECTION
    return guard


def discard_guard_batch(batch: Any, *, reason: str, device: Any = None) -> None:
    """Release one replaced or abandoned delivery through its guard probe."""

    if not reason or not str(reason).strip():
        raise ValueError("guard discard requires a reason")
    probe = getattr(batch, "delivery_probe", None)
    discard = getattr(probe, "discard", None)
    if discard is None:
        raise ValueError("batch has no active guard discard API")
    completion = getattr(batch, "completion_event", None)
    ready = getattr(batch, "ready_event", None)
    stream = None
    if completion is not None:
        if torch is None or device is None:
            raise ValueError("ring-backed guard discard requires a consumer device")
        resolved = torch.device(device)
        if resolved.type != "cuda":
            raise ValueError("CUDA completion event requires a CUDA consumer device")
        stream = torch.cuda.current_stream(resolved)
        if ready is not None:
            stream.wait_event(ready)
    discard(str(reason))
    if stream is not None:
        completion.record(stream)


def _source_bank_shape(source: Any) -> tuple[int, ...] | None:
    inputs = getattr(source, "_inputs", None)
    shape = getattr(inputs, "shape", None)
    return None if shape is None else tuple(int(value) for value in shape)


def _source_generation(source: Any) -> tuple[Any, ...]:
    staging = _load_staging_identity(getattr(source, "node_local_bank_staging", None)) or {}
    return (
        staging.get("receipt_hash"), staging.get("replica_key"),
        str(getattr(source, "inputs_payload_path", None)),
        getattr(getattr(source, "replica_receipt", None), "inputs_sha256", None),
    )


def _source_requires_file_authority(source: Any) -> bool:
    """Grant RAM authority only when the complete buffer owner is known RAM."""

    receipt = getattr(source, "replica_receipt", None)
    if any((
        getattr(source, "inputs_payload_path", None) is not None,
        getattr(receipt, "inputs_sha256", None) is not None,
        getattr(source, "node_local_bank_staging", None) is not None,
    )):
        return True

    seen: set[int] = set()

    def known_ram(value: Any) -> bool:
        if value is None or id(value) in seen:
            return False
        seen.add(id(value))
        if isinstance(value, np.memmap):
            return False
        if isinstance(value, (bytes, bytearray)):
            return True
        if isinstance(value, np.ndarray):
            base = value.base
            return (bool(value.flags.owndata) and base is None) if base is None else known_ram(base)
        if isinstance(value, memoryview):
            return known_ram(value.obj)
        if isinstance(value, ctypes.Array):
            base = getattr(value, "_b_base_", None)
            objects = getattr(value, "_objects", None)
            if base is not None:
                return known_ram(base)
            if objects:
                return all(known_ram(item) for item in objects.values()) if isinstance(objects, dict) else False
            return bool(getattr(value, "_b_needsfree_", False))
        return False

    return not known_ram(getattr(source, "_inputs", None))


def bind_bank_row_digest_table(plane: Any, table: BankRowDigestTable) -> RowDigestGuard:
    """Bind ``table`` as ``plane``'s row authority (explicit; tests and D2)."""

    source = getattr(plane, "source", None)
    if table.binding.get("in_memory") and _source_requires_file_authority(source):
        raise BankRowDigestTableError("file-backed or certified source rejects in-memory table")
    shape = _source_bank_shape(source)
    if shape is None:
        raise BankRowDigestTableError("the plane's source exposes no indexable bank")
    if table.rows != shape[0]:
        raise BankRowDigestTableError(
            f"table rows ({table.rows}) differ from the bank rows ({shape[0]})"
        )
    bank = table.binding.get("bank") or {}
    if int(bank.get("row_bytes", -1)) != int(np.prod(shape[1:])):
        raise BankRowDigestTableError("table row bytes differ from the bank's")
    receipt = getattr(source, "replica_receipt", None)
    certified = getattr(receipt, "inputs_sha256", None)
    if not table.binding.get("in_memory"):
        bank_path = getattr(source, "inputs_payload_path", None)
        if bank_path is None:
            raise BankRowDigestTableError("source has no certified bank file")
        table.validate_file_bank(
            bank_path, certified, getattr(source, "node_local_bank_staging", None), shape
        )
        generation = _source_generation(source)
    else:
        generation = None
        identity = table.binding.get("source_identity") or {}
        expected_hash = identity.get("replica_receipt_content_hash")
        observed_hash = getattr(receipt, "content_hash", None)
        if expected_hash is not None and expected_hash != observed_hash:
            raise BankRowDigestTableError("in-memory table was built for a different source")
    guard = row_digest_guard(plane)
    with guard.lock:
        if guard.chunks_queued:
            raise BankRowDigestTableError("table cannot be rebound after delivery begins")
        guard.table = table
        guard.table_status = "bound_explicitly"
        guard.table_resolved = True
        guard.staging_generation = generation
        guard.fallback_reason = None
        guard.authority_cache.clear()
    return guard


def in_memory_table_for_source(source: Any) -> BankRowDigestTable:
    if _source_requires_file_authority(source):
        raise BankRowDigestTableError("file-backed or certified source rejects in-memory table")
    receipt = getattr(source, "replica_receipt", None)
    return BankRowDigestTable.from_array(
        np.asarray(source._inputs),  # noqa: SLF001 - bank adapter
        source_identity={"replica_receipt_content_hash": getattr(receipt, "content_hash", None)},
    )


def table_authority_for(plane: Any) -> BankRowDigestTable | None:
    """Resolve and cache the plane's bank authority.

    FULL (default) requires the certified file table, or a complete in-memory
    table for a source with no file-backed bank.  A missing, stale, or
    mismatched file table raises.  Only explicit sampled mode may fall back
    to a gather digest, with the reason in the guard receipt.  Reuse checks
    the staging generation and revalidates the file when it changes.
    """

    guard = row_digest_guard(plane)
    if guard.table_resolved:
        if guard.table is not None and guard.table.binding.get("in_memory") and _source_requires_file_authority(getattr(plane, "source", None)):
            raise BankRowDigestTableError("file-backed or certified source rejects in-memory table")
        if guard.table is not None and not guard.table.binding.get("in_memory"):
            source = getattr(plane, "source", None)
            generation = _source_generation(source)
            if generation != guard.staging_generation:
                guard.table.validate_file_bank(
                    getattr(source, "inputs_payload_path", None),
                    getattr(getattr(source, "replica_receipt", None), "inputs_sha256", None),
                    getattr(source, "node_local_bank_staging", None), _source_bank_shape(source),
                )
                guard.staging_generation = generation
        return guard.table
    with guard.lock:
        if guard.table_resolved:
            if guard.table is not None and guard.table.binding.get("in_memory") and _source_requires_file_authority(getattr(plane, "source", None)):
                raise BankRowDigestTableError("file-backed or certified source rejects in-memory table")
            return guard.table
        raw = str(os.environ.get(BANK_ROW_DIGEST_TABLE_ENVIRONMENT, "")).strip()
        resolution = raw.lower() if raw.lower() in {
            "",
            TABLE_RESOLUTION_AUTO,
            TABLE_RESOLUTION_OFF,
            TABLE_RESOLUTION_REQUIRE,
        } else raw
        table = None
        status = "off"
        strict_authority = guard.mode in (GUARD_MODE_FULL, GUARD_MODE_INSPECTION)
        required = strict_authority or resolution == TABLE_RESOLUTION_REQUIRE or resolution not in {
            "", TABLE_RESOLUTION_AUTO, TABLE_RESOLUTION_OFF,
        }
        if resolution == TABLE_RESOLUTION_OFF and strict_authority:
            mode_name = "FULL" if guard.mode == GUARD_MODE_FULL else "inspection"
            raise BankRowDigestTableError(
                f"{mode_name} requires certified table authority; table resolution is off"
            )
        if resolution != TABLE_RESOLUTION_OFF:
            source = getattr(plane, "source", None)
            bank_path = getattr(source, "inputs_payload_path", None)
            receipt = getattr(source, "replica_receipt", None)
            certified = getattr(receipt, "inputs_sha256", None)
            if not getattr(plane, "optimized_bank", False) or bank_path is None or certified is None:
                status = "unavailable:no_certified_bank_file"
                inputs = getattr(source, "_inputs", None)
                if strict_authority and not _source_requires_file_authority(source) and isinstance(inputs, np.ndarray):
                    table = in_memory_table_for_source(source)
                    status = "bound:in_memory_complete_bank"
                elif required:
                    raise BankRowDigestTableError(f"row digest table {status}")
            else:
                explicit = resolution not in {"", TABLE_RESOLUTION_AUTO, TABLE_RESOLUTION_REQUIRE}
                path = Path(raw) if explicit else default_table_path(bank_path, certified)
                if not path.is_file():
                    status = f"unavailable:missing:{path}"
                    if required:
                        raise BankRowDigestTableError(f"row digest table {status}")
                else:
                    try:
                        table = BankRowDigestTable.load(
                            path,
                            bank_path=bank_path,
                            certified_sha256=certified,
                            staged_receipt=getattr(source, "node_local_bank_staging", None),
                            expected_shape=_source_bank_shape(source),
                        )
                        status = "bound:" + str(path)
                        guard.staging_generation = _source_generation(source)
                    except BankRowDigestTableIntegrityError:
                        raise
                    except BankRowDigestTableError as error:
                        status = f"unavailable:binding:{error}"
                        if required:
                            raise
        guard.table = table
        guard.table_status = status
        guard.fallback_reason = status if table is None and guard.mode != GUARD_MODE_DISARMED else None
        guard.table_resolved = True
        return table


_AUTHORITY_CACHE_ENTRIES = 512
_ORDINAL_OF = attrgetter("ordinal")


def authority_digests_for(plane: Any, table: BankRowDigestTable, context_ids: Sequence[str]) -> np.ndarray:
    """The table's digests for ``context_ids``, by ordinals derived HERE.

    The ordinals come from the plane's own records for these ids,
    independently of any ordinals a gather computed, so a gather (or a
    payload keyed to the wrong ids) that returns self-consistent wrong rows
    is still indicted.  Cached per ids tuple (bounded): the value is a pure
    function of the key, so a hit is exactly a recomputation.
    """

    guard = row_digest_guard(plane)
    key = tuple(map(str, context_ids))
    with guard.lock:
        cached = guard.authority_cache.get(key)
        if cached is not None:
            guard.authority_cache_hits += 1
            return cached
    records = plane._records  # noqa: SLF001 - the plane's immutable axis records
    # C-level lookups (itemgetter/attrgetter): ~0.3 us/row on a 1.28M-record
    # axis against ~0.9 us/row for the equivalent generator expression.
    looked_up = itemgetter(*key)(records) if len(key) > 1 else tuple(records[value] for value in key)
    ordinals = np.fromiter(map(_ORDINAL_OF, looked_up), dtype=np.int64, count=len(key))
    expected = table.expected(ordinals)
    expected.setflags(write=False)
    with guard.lock:
        guard.authority_cache_misses += 1
        if len(guard.authority_cache) >= _AUTHORITY_CACHE_ENTRIES:
            guard.authority_cache.pop(next(iter(guard.authority_cache)))
        guard.authority_cache[key] = expected
    return expected


def gather_authority_digests(plane: Any, ordinals: Any) -> np.ndarray | None:
    """D2 API: the gather's per-row digests taken from the bank table.

    The data plane's gather calls this with the bank ordinals it is about to read
    (in the requested order).  With a bound table (v6, guard not disarmed)
    it returns the table's digests for those ordinals, and the gather
    attaches them to its payload in place of ``row_digests_host`` (the
    95-128 us/row host digest).  Every downstream CLASS3 stage is
    unchanged: the device stages compare the delivered bytes with these
    digests (a gather that read the wrong bytes fails at the after-copy
    stage), and the settle compares them with the table at ordinals derived
    independently from the context ids on every row (a gather or payload
    keyed to the wrong ids fails at the gather stage).  None: no table (or
    disarmed); the gather then asks ``gather_host_digest_required``.
    """

    guard = row_digest_guard(plane)
    if guard.mode == GUARD_MODE_DISARMED:
        return None
    table = table_authority_for(plane)
    if table is None:
        return None
    digests = table.expected(np.asarray(ordinals, dtype=np.int64).reshape(-1))
    with guard.lock:
        guard.rows_gather_table_sourced += int(digests.shape[0])
    return digests


def gather_host_digest_required(plane: Any) -> bool:
    """Whether a gather must still compute the per-row host digest (D2 API).

    False in v6 when the plane's guard is disarmed or a bank row digest
    table is its authority (``expected()`` replaces the host digest); True
    whenever the gather-time digest is the authority.
    The gather calls this instead of digesting unconditionally.
    """

    if row_digest_guard(plane).mode == GUARD_MODE_DISARMED:
        return False
    return table_authority_for(plane) is None


# ---------------------------------------------------------------------------
# Self-test for GPU acceptance, and the read-only bank check
# ---------------------------------------------------------------------------


def _self_test_cases(rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    cases = [
        ("imagenet_rows", rng.integers(0, 256, size=(9, 3, 224, 224), dtype=np.uint8)),
        ("imagenet_all_zero", np.zeros((2, 3, 224, 224), dtype=np.uint8)),
        ("imagenet_all_255", np.full((2, 3, 224, 224), 255, dtype=np.uint8)),
        ("words_3x16x16", rng.integers(0, 256, size=(7, 3, 16, 16), dtype=np.uint8)),
        ("bytes_3x5x5", rng.integers(0, 256, size=(5, 3, 5, 5), dtype=np.uint8)),
        ("masked_lane_3x150x150", rng.integers(0, 256, size=(3, 3, 150, 150), dtype=np.uint8)),
        ("one_channel_1x8x8", rng.integers(0, 256, size=(4, 1, 8, 8), dtype=np.uint8)),
        ("straddling_8x8191x1", rng.integers(0, 256, size=(3, 8, 8191, 1), dtype=np.uint8)),
        ("single_row", rng.integers(0, 256, size=(1, 3, 224, 224), dtype=np.uint8)),
        ("ragged_group_19_rows", rng.integers(0, 256, size=(19, 3, 32, 32), dtype=np.uint8)),
        ("flat_150528", rng.integers(0, 256, size=(5, 150528), dtype=np.uint8)),
        ("flat_odd_75", rng.integers(0, 256, size=(6, 75), dtype=np.uint8)),
        ("flat_odd_13", rng.integers(0, 256, size=(3, 13), dtype=np.uint8)),
    ]
    return cases


def _imagenet_decode(device: Any) -> tuple[Any, Any]:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    scale = (1.0 / (255.0 * std)).view(1, 3, 1, 1).to(device)
    bias = (-mean / std).view(1, 3, 1, 1).to(device)
    return scale, bias


def fused_digest_self_test(
    device: Any = None,
    *,
    seed: int = 20260925,
    timing_rows: int = 512,
    require_triton: bool = False,
) -> dict[str, Any]:
    """G1/G3 acceptance self-test.  Raises on any disagreement.

    For every case (production rows, all-0/all-255 rows, odd widths, a
    ragged last row group, the straddling fallback shape, flat rows) and
    every layout (contiguous, channels_last, a sliced view):
    ``digest_rows_fused_device`` equals ``row_digests_host`` bitwise, and so
    does the numpy kernel emulation.  The G3 emulation's normalized values
    equal the torch ``copy_``/``mul_``/``add_`` sequence bitwise.  On CUDA
    with Triton it also certifies the compiled fused normalize (with and
    without the digest) bitwise against the torch sequence, and times every
    path at production shape (the numbers GPU acceptance reports).
    """

    from mechanistic_systems.measurement import native_panel_data_plane as plane_module

    if torch is None:
        raise RuntimeError("self-test requires PyTorch")
    resolved = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    triton_path = resolved.type == "cuda" and triton_digest_available(resolved)
    if require_triton and not triton_path:
        raise RuntimeError(
            f"Triton fused path unavailable on {resolved} "
            f"(import: {_TRITON_IMPORT_ERROR}; disabled: {triton_disabled_reason()})"
        )
    rng = np.random.default_rng(seed)
    report: dict[str, Any] = {
        "device": str(resolved),
        "triton": None if triton is None else getattr(triton, "__version__", "?"),
        "triton_import_error": _TRITON_IMPORT_ERROR,
        "fused_path": "triton" if triton_path else "reference",
        "kernel_contract": FUSED_DIGEST_KERNEL_CONTRACT,
        "normalize_contract": FUSED_NORMALIZE_CONTRACT,
        "cases": {},
    }
    before = fused_path_counts()
    for name, array in _self_test_cases(rng):
        host = plane_module.row_digests_host(array)
        layouts = {"contiguous": torch.from_numpy(array)}
        if array.ndim == 4:
            layouts["channels_last"] = torch.from_numpy(array).contiguous(memory_format=torch.channels_last)
        if array.shape[0] >= 3:
            layouts["slice_1_to_3"] = layouts.get("channels_last", layouts["contiguous"])[1:3]
        results = {}
        for layout_name, tensor in layouts.items():
            expected = host if not layout_name.startswith("slice") else host[1:3]
            fused = digest_rows_fused_device(tensor.to(resolved)).to("cpu").numpy()
            emulated = emulate_fused_row_digest_kernel(tensor)
            ok = np.array_equal(fused, expected) and (emulated is None or np.array_equal(emulated, expected))
            results[layout_name] = {
                "equal": bool(ok),
                "emulated": emulated is not None,
                "kernel_layout": _fused_layout(tensor) is not None,
            }
            if not ok:
                raise RuntimeError(
                    f"fused digest self-test failed: case {name} layout {layout_name} "
                    f"(fused={fused[:3].tolist()} host={expected[:3].tolist()})"
                )
        report["cases"][name] = results
    row = rng.integers(0, 256, size=(1, 3, 224, 224), dtype=np.uint8)
    flips = np.repeat(row, 64, axis=0)
    positions = rng.integers(0, row.size * 8, size=64)
    flips.reshape(64, -1)[np.arange(64), positions // 8] ^= (1 << (positions % 8)).astype(np.uint8)
    base = plane_module.row_digests_host(row)[0]
    flipped = digest_rows_fused_device(
        torch.from_numpy(flips).contiguous(memory_format=torch.channels_last).to(resolved)
    ).to("cpu").numpy()
    if int((flipped == base).sum()) != 0:
        raise RuntimeError("fused digest self-test failed: a single-bit flip went unseen")
    report["single_bit_flips_detected"] = 64
    report["normalize_emulation"] = _normalize_emulation_self_test(rng)
    if triton_path:  # pragma: no cover - CUDA only
        report["normalize"] = _fused_normalize_self_test(resolved, rng)
        report["timing"] = _fused_timing(resolved, rng, timing_rows)
    after = fused_path_counts()
    report["path_counts"] = {key: after[key] - before.get(key, 0) for key in after}
    report["triton_disabled_reason"] = triton_disabled_reason()
    if triton_path and (report["path_counts"].get("triton", 0) <= 0 or triton_disabled_reason() is not None):
        raise RuntimeError(
            f"fused digest self-test failed: the Triton path did not run ({triton_disabled_reason()})"
        )
    report["passed"] = True
    return report


def _normalize_emulation_self_test(rng: np.random.Generator) -> dict[str, Any]:
    """The G3 kernel's dataflow (numpy emulation) against the torch sequence."""

    from mechanistic_systems.measurement import native_panel_data_plane as plane_module

    array = rng.integers(0, 256, size=(11, 3, 224, 224), dtype=np.uint8)
    array[0] = 0
    array[1] = 255
    batch = torch.from_numpy(array).contiguous(memory_format=torch.channels_last)
    scale, bias = _imagenet_decode("cpu")
    results = {}
    for raw in (False, True):
        reference = torch.empty_like(batch, dtype=torch.float32, memory_format=torch.channels_last)
        reference.copy_(batch)
        if not raw:
            reference.mul_(scale).add_(bias)
        digests, normalized = emulate_fused_row_digest_kernel(
            batch, rows_per_program=4, normalize=(scale, bias, raw)
        )
        expected = reference.contiguous().reshape(array.shape[0], 3, -1).numpy()
        bitwise = bool(np.array_equal(normalized.view(np.int32), expected.view(np.int32)))
        digest_ok = bool(np.array_equal(digests, plane_module.row_digests_host(array)))
        results["raw" if raw else "decoded"] = {"bitwise_equal": bitwise, "digest_equal": digest_ok}
        if not (bitwise and digest_ok):
            raise RuntimeError(f"G3 normalize emulation self-test failed ({'raw' if raw else 'decoded'})")
    return results


def _fused_normalize_self_test(device: Any, rng: np.random.Generator) -> dict[str, Any]:  # pragma: no cover - CUDA only
    from mechanistic_systems.measurement import native_panel_data_plane as plane_module

    array = rng.integers(0, 256, size=(33, 3, 224, 224), dtype=np.uint8)
    array[0] = 0
    array[1] = 255
    uint8_batch = torch.from_numpy(array).to(device).contiguous(memory_format=torch.channels_last)
    scale, bias = _imagenet_decode(device)
    host = plane_module.row_digests_host(array)
    results = {}
    for raw in (False, True):
        reference = torch.empty_like(uint8_batch, dtype=torch.float32, memory_format=torch.channels_last)
        reference.copy_(uint8_batch)
        if not raw:
            reference.mul_(scale).add_(bias)
        layout = _fused_layout(uint8_batch)
        for with_digest in (True, False):
            fused_out = torch.full_like(reference, float("nan"), memory_format=torch.channels_last)
            digest = _launch_fused(
                uint8_batch,
                layout,
                out=fused_out,
                scale=None if raw else _per_channel_vector(scale, 3, 4),
                bias=None if raw else _per_channel_vector(bias, 3, 4),
                raw=raw,
                digest=with_digest,
            )
            torch.cuda.synchronize(device)
            bitwise = bool(torch.equal(reference.view(torch.int32), fused_out.view(torch.int32)))
            digest_ok = (not with_digest) or bool(np.array_equal(digest.to("cpu").numpy(), host))
            key = ("raw" if raw else "decoded") + ("_digest" if with_digest else "_normalize_only")
            results[key] = {"bitwise_equal": bitwise, "digest_equal": digest_ok}
            if not (bitwise and digest_ok):
                raise RuntimeError(f"fused normalize self-test failed ({key})")
    return results


def _fused_timing(device: Any, rng: np.random.Generator, rows: int) -> dict[str, Any]:  # pragma: no cover - CUDA only
    array = rng.integers(0, 256, size=(rows, 3, 224, 224), dtype=np.uint8)
    batch = torch.from_numpy(array).to(device).contiguous(memory_format=torch.channels_last)
    inputs = torch.empty_like(batch, dtype=torch.float32, memory_format=torch.channels_last)
    scale, bias = _imagenet_decode(device)
    layout = _fused_layout(batch)
    vector_scale = _per_channel_vector(scale, 3, 4)
    vector_bias = _per_channel_vector(bias, 3, 4)

    def timed(fn, repeats=20):
        fn()
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        stop.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(stop) / repeats / 1e3

    def torch_normalize():
        inputs.copy_(batch)
        inputs.mul_(scale).add_(bias)

    seconds = {
        "fused_digest": timed(lambda: digest_rows_fused_device(batch)),
        "reference_digest": timed(lambda: digest_rows_fused_device(batch, force_reference=True), repeats=5),
        "torch_normalize": timed(torch_normalize),
        "fused_normalize_only": timed(
            lambda: _launch_fused(batch, layout, out=inputs, scale=vector_scale, bias=vector_bias, digest=False)
        ),
        "fused_normalize_digest": timed(
            lambda: _launch_fused(batch, layout, out=inputs, scale=vector_scale, bias=vector_bias, digest=True)
        ),
    }
    per_row_us = {key: value / rows * 1e6 for key, value in seconds.items()}
    # The full guard's device cost per delivered row, against the same
    # normalize without a guard: the after-copy and after-use digests plus
    # the first-use digest's share of the fused normalize.
    guard_us = 2 * per_row_us["fused_digest"] + max(
        0.0, per_row_us["fused_normalize_digest"] - per_row_us["fused_normalize_only"]
    )
    return {
        "rows": rows,
        "seconds_per_chunk": seconds,
        "us_per_row": per_row_us,
        "full_guard_device_us_per_row": guard_us,
        "normalize_saving_vs_torch_us_per_row": per_row_us["torch_normalize"]
        - per_row_us["fused_normalize_only"],
        "row_bytes": int(batch[0].numel()),
        "digest_effective_gb_per_s": float(batch.numel()) / seconds["fused_digest"] / 1e9,
    }


def check_bank_rows(
    bank_path: str | Path,
    *,
    rows: int = 100_000,
    seed: int = 20260925,
    table_path: str | Path | None = None,
    device: Any = None,
    block_rows: int = 512,
) -> dict[str, Any]:
    """Real-bank check, READ-ONLY: table entries equal host digests.

    Opens the bank read-only (mmap), draws ``rows`` random row indices, and
    compares ``row_digests_host`` of those rows with (a) the table at
    ``table_path`` when given and (b) the table builder's digest path on
    ``device`` (the fused kernel on CUDA).  Writes nothing; prints nothing;
    returns the report.
    """

    from mechanistic_systems.measurement import native_panel_data_plane as plane_module

    bank = np.load(Path(bank_path), mmap_mode="r")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(int(bank.shape[0]), size=min(int(rows), int(bank.shape[0])), replace=False))
    table = None
    if table_path is not None:
        digests = np.load(Path(table_path), mmap_mode="r")
        table = np.asarray(digests[indices], dtype=np.int64)
    resolved = None if device is None else torch.device(device)
    mismatches_table = 0
    mismatches_builder = 0
    started = time.perf_counter()
    for start in range(0, indices.size, int(block_rows)):
        chunk = indices[start : start + int(block_rows)]
        rows_array = np.asarray(bank[chunk])
        host = plane_module.row_digests_host(rows_array)
        if table is not None:
            mismatches_table += int((table[start : start + chunk.size] != host).sum())
        if resolved is not None:
            tensor = torch.from_numpy(rows_array).to(resolved)
            if tensor.ndim == 4:
                tensor = tensor.contiguous(memory_format=torch.channels_last)
            builder = digest_rows_fused_device(tensor).to("cpu").numpy()
            mismatches_builder += int((builder != host).sum())
    return {
        "bank_path": str(bank_path),
        "bank_rows": int(bank.shape[0]),
        "checked_rows": int(indices.size),
        "seed": int(seed),
        "table_path": None if table_path is None else str(table_path),
        "table_mismatches": None if table is None else mismatches_table,
        "builder_device": None if resolved is None else str(resolved),
        "builder_path": None
        if resolved is None
        else ("triton" if resolved.type == "cuda" and triton is not None else "reference"),
        "builder_mismatches": None if resolved is None else mismatches_builder,
        "seconds": float(time.perf_counter() - started),
        "passed": (table is None or mismatches_table == 0)
        and (resolved is None or mismatches_builder == 0),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bank_row_digest_table", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    selftest = commands.add_parser("selftest", help="G1/G3 acceptance self-test (JSON on stdout)")
    selftest.add_argument("--device", default=None)
    selftest.add_argument("--require-triton", action="store_true")
    selftest.add_argument("--timing-rows", type=int, default=512)
    check = commands.add_parser("check-bank", help="read-only §9 test 2 on a real bank")
    check.add_argument("--bank", required=True)
    check.add_argument("--rows", type=int, default=100_000)
    check.add_argument("--seed", type=int, default=20260925)
    check.add_argument("--table", default=None)
    check.add_argument("--device", default=None)
    build = commands.add_parser("build", help="build and store a node-local table")
    build.add_argument("--bank", required=True)
    build.add_argument("--staging-receipt", default=None)
    build.add_argument("--certified-sha256", default=None)
    build.add_argument("--verify-sha256", action="store_true")
    build.add_argument("--device", default=None)
    build.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.command == "selftest":
        report = fused_digest_self_test(
            args.device, timing_rows=args.timing_rows, require_triton=args.require_triton
        )
    elif args.command == "check-bank":
        report = check_bank_rows(
            args.bank, rows=args.rows, seed=args.seed, table_path=args.table, device=args.device
        )
    else:
        table = BankRowDigestTable.build(
            args.bank,
            args.staging_receipt,
            certified_sha256=args.certified_sha256,
            verify_sha256=True if args.verify_sha256 else None,
            device=args.device,
            out_path=args.out,
        )
        report = table.summary()
    report["effective_settings"] = effective_settings()
    json.dump(report, sys.stdout, indent=2, sort_keys=True, default=str)
    sys.stdout.write("\n")
    return 0 if report.get("passed", True) else 1


__all__ = [
    "AUTHORITY_GATHER",
    "AUTHORITY_TABLE",
    "BANK_ROW_DIGEST_TABLE_CONTRACT",
    "BANK_ROW_DIGEST_TABLE_ENVIRONMENT",
    "BANK_ROW_DIGEST_TABLE_SCHEMA",
    "BankRowDigestTable",
    "BankRowDigestTableError",
    "BankRowDigestTableIntegrityError",
    "FUSED_DIGEST_KERNEL_CONTRACT",
    "FUSED_NORMALIZE_CONTRACT",
    "GUARD_MODES",
    "GUARD_MODE_DISARMED",
    "GUARD_MODE_FULL",
    "GUARD_MODE_INSPECTION",
    "GUARD_MODE_SAMPLED",
    "READOUT_GUARD_MODE_ENVIRONMENT",
    "RowDigestGuard",
    "SAMPLED_USE_PROBE_PERIOD",
    "bind_bank_row_digest_table",
    "configure_guard_for_inspection",
    "discard_guard_batch",
    "check_bank_rows",
    "default_table_path",
    "digest_rows_fused_device",
    "emulate_fused_row_digest_kernel",
    "fused_digest_self_test",
    "authority_digests_for",
    "gather_authority_digests",
    "gather_host_digest_required",
    "in_memory_table_for_source",
    "normalize_with_first_use_digest",
    "readout_guard_mode",
    "row_digest_guard",
    "sampled_use_probe_selected",
    "table_authority_for",
    "triton_digest_available",
    "triton_disabled_reason",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
