"""Fused device programs for the terminal calibration banks.

One program per functional fits every category of a bank at once.

* The rows of the fit are ordered once by selection block, so every row tile
  lies inside one block and computes only the lanes that use its rows (15 of
  the 20 selection lanes, the 6 held-out lanes of its block, the one
  deployment lane) instead of masking every lane on every row.
* One kernel pass reads the readout (and the dense logits) once and keeps the
  per-lane sums in registers.  It writes one partial per (row tile, column,
  lane); a sum over tiles closes the pass.  No [cell x row] tensor exists.
* The substituted functional's dense-exclusion offset is formed in the kernel
  from the streaming plan's per-row maximum, shifted sum, argmax and second
  logsumexp (``_DenseExclusionStreamingPlan``).
* Columns whose cells are all inactive leave the launch grid between Newton
  step batches.  One packed status read closes each batch.

The programs fit fixed-offset L2 affine models with one-SE selection and
proper-score identity shrinkage. The latter uses anchored targets, a weighted
chart, guarded projected Newton steps, and an Armijo ladder. Unconverged cells
restart on the device with the extended iteration and line-search budgets;
unresolved columns remain missing. A frozen cell's iterates stay fixed as it
leaves the grid. Row reductions follow the block tiles.

The same programs run on CPU through torch row sums and on CUDA through
Triton kernels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from threading import Lock
from collections import OrderedDict
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:  # pragma: no cover - CUDA hosts only
    if not torch.cuda.is_available():
        raise ImportError("Triton needs CUDA")
    import triton
    import triton.language as tl

    _TRITON = True
except ImportError:
    triton = tl = None  # type: ignore[assignment]
    _TRITON = False


SELECTION = "selection"
HELDOUT = "heldout"
DEPLOYMENT = "deployment"
STATISTICS = 6
TILE_ROWS = 4096
BATCH_STEPS = 8
EXECUTION = "fused_device_program_v63"

# Kernel kinds.
_AFFINE = 0  # eta = a*x + b + offset, binary target, positive-row weight w
_ANCHORED = 1  # eta = beta*(x-mu)/sd + gamma, anchored target and weight
_CHART = 2  # weighted chart moments of the anchored fit

_OFFSET_NONE = 0
_OFFSET_PLAN = 1
_OFFSET_EXPLICIT = 2


# ---------------------------------------------------------------------------
# Rows of one fit: block order and tiles (category-invariant, memoized).


@dataclass(frozen=True)
class FusedRows:
    order: torch.Tensor  # int64 [rows]: axis row of each fit row, block-major
    block_of_row: torch.Tensor  # int64 [rows]
    block_start: tuple[int, ...]
    block_rows: tuple[int, ...]
    tile_start: torch.Tensor  # int32 [tiles]
    tile_stop: torch.Tensor
    tile_block: torch.Tensor
    total_rows: int  # every row of the axis (the ordinary objective's divisor)

    @property
    def rows(self) -> int:
        return int(self.order.numel())

    @property
    def block_count(self) -> int:
        return len(self.block_rows)

    @property
    def tile_count(self) -> int:
        return int(self.tile_start.numel())


_ROWS_MEMO: "OrderedDict[tuple[Any, ...], tuple[Any, FusedRows]]" = OrderedDict()
_ROWS_MEMO_LOCK = Lock()
_ROWS_MEMO_ENTRIES = 4


def fused_rows(layout: Any, device: Any, *, tile_rows: int = TILE_ROWS) -> FusedRows:
    """Block-ordered fit rows of a one-fold selection layout."""

    device = torch.device(device)
    key = (id(layout), str(layout.content_digest), str(device), int(tile_rows))
    with _ROWS_MEMO_LOCK:
        cached = _ROWS_MEMO.get(key)
        if cached is not None and cached[0] is layout:
            _ROWS_MEMO.move_to_end(key)
            return cached[1]
    if layout.fold_count != 1 or not layout.selection_enabled:
        raise ValueError("fused bank fit requires one outer fold with selection blocks")
    blocks = np.asarray(layout.inner_block_ids[0], dtype=np.int64)
    count = int(layout.inner_block_counts[0])
    if count < 2:
        raise ValueError("fused bank fit requires at least two selection blocks")
    fit = np.flatnonzero(blocks >= 0)  # outer-fit rows (validated layout)
    order = fit[np.argsort(blocks[fit], kind="stable")]
    sizes = np.bincount(blocks[fit], minlength=count)[:count]
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.int64)
    tiles = [
        (int(start + offset), int(min(start + size, start + offset + tile_rows)), block)
        for block, (start, size) in enumerate(zip(starts, sizes))
        for offset in range(0, int(size), int(tile_rows))
    ]
    tile_array = np.asarray(tiles, dtype=np.int32).reshape(-1, 3)
    rows = FusedRows(
        order=torch.as_tensor(order, dtype=torch.int64).to(device),
        block_of_row=torch.as_tensor(
            np.repeat(np.arange(count, dtype=np.int64), sizes)
        ).to(device),
        block_start=tuple(int(value) for value in starts),
        block_rows=tuple(int(value) for value in sizes),
        tile_start=torch.as_tensor(tile_array[:, 0].copy()).to(device),
        tile_stop=torch.as_tensor(tile_array[:, 1].copy()).to(device),
        tile_block=torch.as_tensor(tile_array[:, 2].copy()).to(device),
        total_rows=int(blocks.size),
    )
    with _ROWS_MEMO_LOCK:
        _ROWS_MEMO[key] = (layout, rows)
        _ROWS_MEMO.move_to_end(key)
        while len(_ROWS_MEMO) > _ROWS_MEMO_ENTRIES:
            _ROWS_MEMO.popitem(last=False)
    return rows


def lane_table(kind: str, block_count: int, strength_count: int) -> list[list[int]]:
    """Lanes that use the rows of each block, padded to a power of two (min 2).

    Selection lanes are ``block * S + strength`` (block-major) and use
    the rows outside their block; held-out lanes are the same 20 fitted lanes
    plus one identity lane ``B * S + block`` and use the rows of their block;
    the deployment lane uses every row.
    """

    B, S = int(block_count), int(strength_count)
    if kind == SELECTION:
        table = [[b * S + s for b in range(B) if b != k for s in range(S)] for k in range(B)]
    elif kind == HELDOUT:
        table = [[k * S + s for s in range(S)] + [B * S + k] for k in range(B)]
    elif kind == DEPLOYMENT:
        table = [[0] for _ in range(B)]
    else:
        raise ValueError(f"unknown lane table {kind!r}")
    width = max(2, 1 << (max(len(row) for row in table) - 1).bit_length())
    return [row + [-1] * (width - len(row)) for row in table]


# ---------------------------------------------------------------------------
# One fit's inputs in block-ordered rows.


@dataclass
class FusedProblem:
    rows: FusedRows
    values: torch.Tensor  # [columns, rows] readout (float32/float64), fit-row order
    labels: torch.Tensor  # int32 [rows]
    class_ids: torch.Tensor  # int32 [columns]
    positives: torch.Tensor  # float64 [columns, blocks] positive rows per block
    sample_weights: torch.Tensor | None = None  # shared image-design weights, fit-row order
    offset_mode: int = _OFFSET_NONE
    offsets: torch.Tensor | None = None  # [columns, rows] dense logits or offsets
    row_maximum: torch.Tensor | None = None
    row_shifted_sum: torch.Tensor | None = None
    row_argmax: torch.Tensor | None = None
    row_second_lse: torch.Tensor | None = None

    @property
    def columns(self) -> int:
        return int(self.values.shape[0])

    @property
    def device(self) -> torch.device:
        return self.values.device

    @classmethod
    def build(
        cls,
        rows: FusedRows,
        *,
        readout_by_category: Any,
        labels: Any,
        class_ids: Sequence[int],
        plan: Any | None = None,
        fixed_offsets: Any | None = None,
        sample_weights: Any | None = None,
    ) -> "FusedProblem":
        readout = torch.as_tensor(readout_by_category)
        device = readout.device
        if readout.dtype not in (torch.float32, torch.float64):
            readout = readout.to(torch.float32)
        columns = torch.as_tensor(tuple(int(v) for v in class_ids), device=device, dtype=torch.int64)
        order = rows.order.to(device)
        values = readout.index_select(0, columns).index_select(1, order).contiguous()
        label_rows = torch.as_tensor(labels, device=device).reshape(-1).to(torch.int64)
        if int(label_rows.numel()) != rows.total_rows or int(readout.shape[1]) != rows.total_rows:
            raise ValueError("fused bank fit sample axes are misaligned")
        label_rows = label_rows.index_select(0, order)
        B = rows.block_count
        key = label_rows * B + rows.block_of_row.to(device)
        width = max(int(readout.shape[0]), int(columns.max().item()) + 1)
        counts = torch.bincount(key, minlength=width * B)
        counts = torch.nn.functional.pad(counts, (0, (-int(counts.numel())) % B))
        positives = counts.reshape(-1, B).index_select(0, columns).to(torch.float64)
        problem = cls(
            rows=rows,
            values=values,
            labels=label_rows.to(torch.int32).contiguous(),
            class_ids=columns.to(torch.int32).contiguous(),
            positives=positives,
        )
        if sample_weights is not None:
            weights = torch.as_tensor(sample_weights, device=device, dtype=torch.float64).reshape(-1)
            if weights.numel() != rows.total_rows or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()) or not bool(weights.sum() > 0):
                raise ValueError("image sample weights must align with rows and have finite nonnegative mass")
            problem.sample_weights = (weights.index_select(0, order) * (rows.rows / weights.index_select(0, order).sum())).contiguous()
        if plan is not None:
            dense = torch.as_tensor(plan.dense_logits_by_category)
            if dense.dtype not in (torch.float32, torch.float64):
                dense = dense.to(torch.float32)
            problem.offset_mode = _OFFSET_PLAN
            problem.offsets = dense.index_select(0, columns).index_select(1, order).contiguous()
            problem.row_maximum = plan.maximum.index_select(0, order).to(torch.float64).contiguous()
            problem.row_shifted_sum = plan.shifted_sum.index_select(0, order).to(torch.float64).contiguous()
            problem.row_argmax = plan.argmax.index_select(0, order).to(torch.int32).contiguous()
            problem.row_second_lse = plan.second_logsumexp.index_select(0, order).to(torch.float64).contiguous()
        elif fixed_offsets is not None:
            fixed = torch.as_tensor(fixed_offsets, device=device, dtype=torch.float64)
            problem.offset_mode = _OFFSET_EXPLICIT
            problem.offsets = fixed.index_select(0, columns).index_select(1, order).contiguous()
        return problem

    def finite_inputs(self) -> torch.Tensor:
        """Device flag: readout (and explicit offsets) finite."""

        flag = torch.isfinite(self.values).all()
        if self.offset_mode == _OFFSET_EXPLICIT:
            flag = flag & torch.isfinite(self.offsets).all()
        return flag


# ---------------------------------------------------------------------------
# Per-lane row sums: the one kernel of both programs.

if _TRITON:  # pragma: no cover - compiled and run on CUDA hosts only

    @triton.jit
    def _softplus(z):
        # torch.nn.functional.softplus (threshold 20), in a stable form.
        return tl.where(z > 20.0, z, tl.maximum(z, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(z))))

    @triton.jit(do_not_specialize=["row_count", "cells", "active_count"])
    def _bank_rows_kernel(
        values_ptr, aux_ptr, rmax_ptr, rsum_ptr, rarg_ptr, rlse_ptr,
        label_ptr, class_ptr, col_ptr, sample_weight_ptr,
        tile_start_ptr, tile_stop_ptr, tile_block_ptr, lane_ptr,
        a_ptr, b_ptr, w_ptr, mu_ptr, sd_ptr, lam_ptr, eps_ptr, out_ptr,
        row_count, cells, active_count,
        KIND: tl.constexpr, OFFSET: tl.constexpr, GRAD: tl.constexpr,
        OBJECTIVE: tl.constexpr, DESIGN_WEIGHTED: tl.constexpr, LANES: tl.constexpr, BLOCK: tl.constexpr,
    ):
        item = tl.program_id(0)
        tile = tl.program_id(1)
        column = tl.load(col_ptr + item).to(tl.int64)
        target = tl.load(class_ptr + column)
        start = tl.load(tile_start_ptr + tile)
        stop = tl.load(tile_stop_ptr + tile)
        block = tl.load(tile_block_ptr + tile)
        lane = tl.load(lane_ptr + block * LANES + tl.arange(0, LANES))
        lane_ok = lane >= 0
        cell = column * cells + tl.where(lane_ok, lane, 0)
        pa = tl.load(a_ptr + cell, mask=lane_ok, other=0.0)
        pb = tl.load(b_ptr + cell, mask=lane_ok, other=0.0)
        pw = tl.load(w_ptr + cell, mask=lane_ok, other=1.0)
        pmu = tl.load(mu_ptr + cell, mask=lane_ok, other=0.0)
        psd = tl.load(sd_ptr + cell, mask=lane_ok, other=1.0)
        plam = tl.load(lam_ptr + cell, mask=lane_ok, other=0.0)
        eps = tl.load(eps_ptr)
        s0 = tl.zeros([LANES], dtype=tl.float64)
        s1 = tl.zeros([LANES], dtype=tl.float64)
        s2 = tl.zeros([LANES], dtype=tl.float64)
        s3 = tl.zeros([LANES], dtype=tl.float64)
        s4 = tl.zeros([LANES], dtype=tl.float64)
        s5 = tl.zeros([LANES], dtype=tl.float64)
        base = column * row_count
        for r0 in range(start, stop, BLOCK):
            row = r0 + tl.arange(0, BLOCK)
            row_ok = row < stop
            ok = lane_ok[:, None] & row_ok[None, :]
            x = tl.load(values_ptr + base + row, mask=row_ok, other=0.0).to(tl.float64)
            label = tl.load(label_ptr + row, mask=row_ok, other=-1)
            y = (label == target).to(tl.float64)
            xx = x[None, :]
            yy = y[None, :]
            fixed = tl.full((BLOCK,), 0.0, tl.float64)
            if OFFSET == 1:
                dense = tl.load(aux_ptr + base + row, mask=row_ok, other=0.0).to(tl.float64)
                top = tl.load(rmax_ptr + row, mask=row_ok, other=0.0)
                total = tl.load(rsum_ptr + row, mask=row_ok, other=2.0)
                first = tl.load(rarg_ptr + row, mask=row_ok, other=-1)
                second = tl.load(rlse_ptr + row, mask=row_ok, other=0.0)
                exclusion = tl.where(first == target, second, tl.log(total - tl.exp(dense - top)) + top)
                fixed = -exclusion
            elif OFFSET == 2:
                fixed = tl.load(aux_ptr + base + row, mask=row_ok, other=0.0).to(tl.float64)
            # Positive rows carry w, negative rows 1 (exact for y in {0, 1}).
            weight = pw[:, None] * yy + (1.0 - yy)
            if DESIGN_WEIGHTED:
                weight = weight * tl.load(sample_weight_ptr + row, mask=row_ok, other=0.0)[None, :]
            if KIND == 2:
                identity_eta = x + fixed
                p0 = 1.0 / (1.0 + tl.exp(-identity_eta))
                p0 = tl.minimum(tl.maximum(p0, eps), 1.0 - eps)
                g = _softplus(identity_eta) - p0 * identity_eta
                d = xx - pmu[:, None]
                s0 += tl.sum(tl.where(ok, weight, 0.0), axis=1)
                s1 += tl.sum(tl.where(ok, weight * xx, 0.0), axis=1)
                s2 += tl.sum(tl.where(ok, weight * yy, 0.0), axis=1)
                s3 += tl.sum(tl.where(ok, weight * g[None, :], 0.0), axis=1)
                s4 += tl.sum(tl.where(ok, weight * (d * d), 0.0), axis=1)
                s5 += tl.sum(tl.where(ok, weight * (1.0 - yy), 0.0), axis=1)
            else:
                if KIND == 1:
                    design = (xx - pmu[:, None]) / psd[:, None]
                    p0 = 1.0 / (1.0 + tl.exp(-(x + fixed)))
                    p0 = tl.minimum(tl.maximum(p0, eps), 1.0 - eps)
                    goal = (yy + plam[:, None] * p0[None, :]) / (1.0 + plam[:, None])
                    weight = weight * (1.0 + plam[:, None])
                    eta = design * pa[:, None] + pb[:, None] + fixed[None, :]
                else:
                    design = xx
                    goal = yy
                    eta = pa[:, None] * xx + pb[:, None] + fixed[None, :]
                if OBJECTIVE:
                    s0 += tl.sum(tl.where(ok, weight * (_softplus(eta) - goal * eta), 0.0), axis=1)
                if GRAD:
                    p = 1.0 / (1.0 + tl.exp(-eta))
                    residual = weight * (p - goal)
                    curvature = weight * p * (1.0 - p)
                    s1 += tl.sum(tl.where(ok, residual * design, 0.0), axis=1)
                    s2 += tl.sum(tl.where(ok, residual, 0.0), axis=1)
                    s3 += tl.sum(tl.where(ok, curvature * (design * design), 0.0), axis=1)
                    s4 += tl.sum(tl.where(ok, curvature * design, 0.0), axis=1)
                    s5 += tl.sum(tl.where(ok, curvature, 0.0), axis=1)
        out = out_ptr + ((tile.to(tl.int64) * active_count + item) * cells + tl.where(lane_ok, lane, 0)) * 6
        tl.store(out + 0, s0, mask=lane_ok)
        tl.store(out + 1, s1, mask=lane_ok)
        tl.store(out + 2, s2, mask=lane_ok)
        tl.store(out + 3, s3, mask=lane_ok)
        tl.store(out + 4, s4, mask=lane_ok)
        tl.store(out + 5, s5, mask=lane_ok)


class _Sums:
    """Launches one per-lane row-sum pass; counts passes for the receipt."""

    def __init__(self, problem: FusedProblem, *, epsilon: float = 0.0) -> None:
        self.problem = problem
        self.device = problem.device
        self.epsilon = torch.tensor([float(epsilon)], device=self.device, dtype=torch.float64)
        self.tables: dict[tuple[str, int, int], tuple[list[list[int]], torch.Tensor]] = {}
        self.passes = 0
        self.column_passes = 0
        self.triton = bool(_TRITON and problem.values.is_cuda)
        self.chunk_elements = 1 << 22

    def table(self, kind: str, strength_count: int) -> tuple[list[list[int]], torch.Tensor]:
        key = (kind, self.problem.rows.block_count, int(strength_count))
        cached = self.tables.get(key)
        if cached is None:
            host = lane_table(kind, key[1], key[2])
            cached = (host, torch.tensor(host, dtype=torch.int32, device=self.device))
            self.tables[key] = cached
        return cached

    def __call__(
        self,
        *,
        kind: int,
        lanes: tuple[list[list[int]], torch.Tensor],
        cells: int,
        columns: torch.Tensor,
        params: Sequence[torch.Tensor],
        grad: bool,
        objective: bool,
        offsets: bool = True,
    ) -> torch.Tensor:
        """[len(columns), cells, 6] float64 per-lane row sums.

        ``params`` are six float64 [columns_total, cells] tensors
        (a|beta, b|gamma, positive-row weight, mu, sd, anchor strength).
        Unlisted lanes of a column stay zero.
        """

        count = int(columns.numel())
        self.passes += 1
        self.column_passes += count
        if count == 0:
            return torch.zeros((0, cells, STATISTICS), device=self.device, dtype=torch.float64)
        offset_mode = self.problem.offset_mode if offsets else _OFFSET_NONE
        if self.triton:
            return self._triton(kind, lanes[1], cells, columns, params, grad, objective, offset_mode)
        return self._torch(kind, lanes[0], cells, columns, params, grad, objective, offset_mode)

    def _triton(self, kind, table, cells, columns, params, grad, objective, offset_mode):  # pragma: no cover
        problem = self.problem
        rows = problem.rows
        count = int(columns.numel())
        width = int(table.shape[1])
        out = torch.zeros(
            (rows.tile_count, count, cells, STATISTICS), device=self.device, dtype=torch.float64
        )
        dummy = problem.values
        aux = problem.offsets if offset_mode != _OFFSET_NONE else dummy
        plan = offset_mode == _OFFSET_PLAN
        _bank_rows_kernel[(count, rows.tile_count)](
            problem.values,
            aux,
            problem.row_maximum if plan else dummy,
            problem.row_shifted_sum if plan else dummy,
            problem.row_argmax if plan else problem.labels,
            problem.row_second_lse if plan else dummy,
            problem.labels,
            problem.class_ids,
            columns.to(torch.int32).contiguous(),
            problem.sample_weights if problem.sample_weights is not None else dummy,
            rows.tile_start,
            rows.tile_stop,
            rows.tile_block,
            table,
            *[param.contiguous() for param in params],
            self.epsilon,
            out,
            int(rows.rows),
            int(cells),
            count,
            KIND=int(kind),
            OFFSET=int(offset_mode),
            GRAD=bool(grad),
            OBJECTIVE=bool(objective),
            DESIGN_WEIGHTED=problem.sample_weights is not None,
            LANES=width,
            BLOCK=max(32, 1024 // width),
            num_warps=4,
        )
        return out.sum(dim=0)

    def _torch(self, kind, table, cells, columns, params, grad, objective, offset_mode):
        problem = self.problem
        rows = problem.rows
        index = columns.to(device=self.device, dtype=torch.int64)
        count = int(index.numel())
        out = torch.zeros((count, cells, STATISTICS), device=self.device, dtype=torch.float64)
        values = problem.values.index_select(0, index)
        target = problem.class_ids.index_select(0, index).to(torch.int64).unsqueeze(1)
        selected = [param.index_select(0, index) for param in params]
        aux = None if offset_mode == _OFFSET_NONE else problem.offsets.index_select(0, index)
        epsilon = float(self.epsilon.item())
        for block in range(rows.block_count):
            lanes = [lane for lane in table[block] if lane >= 0]
            if not lanes or rows.block_rows[block] == 0:
                continue
            lane_index = torch.tensor(lanes, device=self.device, dtype=torch.int64)
            a, b, w, mu, sd, lam = (param.index_select(1, lane_index).unsqueeze(2) for param in selected)
            low = rows.block_start[block]
            high = low + rows.block_rows[block]
            step = max(32, self.chunk_elements // max(1, count * len(lanes)))
            total = torch.zeros((count, len(lanes), STATISTICS), device=self.device, dtype=torch.float64)
            for start in range(low, high, step):
                stop = min(high, start + step)
                x = values[:, start:stop].to(torch.float64).unsqueeze(1)
                y = (problem.labels[start:stop].to(torch.int64).unsqueeze(0) == target).to(torch.float64).unsqueeze(1)
                weight = torch.where(y > 0.5, w, torch.ones_like(w))
                if problem.sample_weights is not None:
                    weight = weight * problem.sample_weights[start:stop].reshape(1, 1, -1)
                fixed = 0.0
                if offset_mode == _OFFSET_PLAN:
                    dense = aux[:, start:stop].to(torch.float64)
                    top = problem.row_maximum[start:stop].unsqueeze(0)
                    exclusion = torch.log(problem.row_shifted_sum[start:stop].unsqueeze(0) - torch.exp(dense - top)) + top
                    first = problem.row_argmax[start:stop].to(torch.int64).unsqueeze(0) == target
                    exclusion = torch.where(first, problem.row_second_lse[start:stop].unsqueeze(0), exclusion)
                    fixed = (-exclusion).unsqueeze(1)
                elif offset_mode == _OFFSET_EXPLICIT:
                    fixed = aux[:, start:stop].unsqueeze(1)
                if kind == _CHART:
                    identity_eta = x + fixed
                    p0 = torch.sigmoid(identity_eta).clamp(min=epsilon, max=1.0 - epsilon)
                    g = F.softplus(identity_eta) - p0 * identity_eta
                    d = x - mu
                    terms = (weight, weight * x, weight * y, weight * g, weight * (d * d), weight * (1.0 - y))
                    total += torch.stack([term.expand(count, len(lanes), stop - start).sum(dim=2) for term in terms], dim=2)
                    continue
                if kind == _ANCHORED:
                    design = (x - mu) / sd
                    p0 = torch.sigmoid(x + fixed).clamp(min=epsilon, max=1.0 - epsilon)
                    goal = (y + lam * p0) / (1.0 + lam)
                    weight = weight * (1.0 + lam)
                    eta = design * a + b + fixed
                else:
                    design = x
                    goal = y
                    eta = a * x + b + fixed
                zero = torch.zeros((count, len(lanes)), device=self.device, dtype=torch.float64)
                sums = [zero] * STATISTICS
                if objective:
                    sums[0] = (weight * (F.softplus(eta) - goal * eta)).sum(dim=2)
                if grad:
                    p = torch.sigmoid(eta)
                    residual = weight * (p - goal)
                    curvature = weight * p * (1.0 - p)
                    sums[1] = (residual * design).sum(dim=2)
                    sums[2] = residual.sum(dim=2)
                    sums[3] = (curvature * (design * design)).sum(dim=2)
                    sums[4] = (curvature * design).sum(dim=2)
                    sums[5] = curvature.sum(dim=2)
                total += torch.stack([s.expand(count, len(lanes)) for s in sums], dim=2)
            out[:, lane_index] += total
        return out


def _compact(flags: torch.Tensor, count: int) -> torch.Tensor:
    """Positions of the ``count`` true entries of ``flags``, in order."""

    order = torch.argsort((~flags).to(torch.int8), stable=True)
    return order[: int(count)]


def _read(*values: torch.Tensor) -> list[int]:
    """One packed device-to-host read of integer/boolean status values."""

    return [int(v) for v in torch.stack([v.reshape(()).to(torch.int64) for v in values]).cpu().tolist()]


def one_se_selection(fold_losses: torch.Tensor) -> dict[str, torch.Tensor]:
    """The registered one-SE rule over ``[column, block, candidate]`` losses."""

    blocks = int(fold_losses.shape[1])
    means = fold_losses.mean(dim=1)
    errors = fold_losses.std(dim=1, unbiased=True) / math.sqrt(float(blocks))
    best = means.argmin(dim=1)
    threshold = means.gather(1, best.unsqueeze(1)).squeeze(1)
    threshold = threshold + errors.gather(1, best.unsqueeze(1)).squeeze(1)
    eligible = means <= threshold.unsqueeze(1)
    candidates = torch.arange(int(fold_losses.shape[2]), device=fold_losses.device).unsqueeze(0)
    selected = torch.where(eligible, candidates, torch.full_like(candidates, -1)).max(dim=1).values
    return {"means": means, "errors": errors, "best": best, "eligible": eligible, "selected": selected}


def _support(positive: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    return ((positive > 0.0) & (count - positive > 0.0)).all()


# ---------------------------------------------------------------------------
# Substituted functional: fixed-offset L2 Newton, one-SE selection, refit.


def _offset_newton(
    sums: _Sums,
    *,
    lanes: Any,
    cells: int,
    rows_per_lane: torch.Tensor,
    weight: torch.Tensor,
    strength: torch.Tensor,
    minimum_scale: float,
    max_iterations: int,
    tolerance: float,
    ridge: float,
    projected: bool,
    profile: dict[str, int],
) -> tuple[torch.Tensor, ...]:
    """``_fit_fixed_offset_affine_l2_matrix_streamed``'s Newton, compacted."""

    problem = sums.problem
    device = problem.device
    C = problem.columns
    a = torch.ones((C, cells), device=device, dtype=torch.float64)
    b = torch.zeros_like(a)
    iterations = torch.zeros((C, cells), device=device, dtype=torch.int64)
    zero = torch.zeros_like(a)
    columns = torch.arange(C, device=device, dtype=torch.int64)
    count = C
    done = 0
    bad_hessian = torch.zeros((), device=device, dtype=torch.bool)
    bad_step = torch.zeros_like(bad_hessian)
    while count and done < int(max_iterations):
        steps = min(BATCH_STEPS, int(max_iterations) - done)
        moved = torch.zeros((count,), device=device, dtype=torch.bool)
        for _ in range(steps):
            totals = sums(kind=_AFFINE, lanes=lanes, cells=cells, columns=columns,
                          params=(a, b, weight, zero, zero, zero), grad=True, objective=False)
            ac, bc = a.index_select(0, columns), b.index_select(0, columns)
            n = rows_per_lane.index_select(0, columns)
            lam = strength.index_select(0, columns)
            ga = totals[..., 1] / n + 2.0 * lam * (ac - 1.0)
            gb = totals[..., 2] / n + 2.0 * lam * bc
            haa = totals[..., 3] / n + 2.0 * lam
            hab = totals[..., 4] / n
            hbb = totals[..., 5] / n + 2.0 * lam
            residual_a = ga.abs()
            if projected:
                residual_a = torch.where(ac <= minimum_scale, torch.clamp(-ga, min=0.0), residual_a)
            active = torch.maximum(residual_a, gb.abs()) > tolerance
            haa, hbb = haa + ridge, hbb + ridge
            determinant = haa * hbb - hab.square()
            invalid = (determinant <= 0.0) | ~torch.isfinite(determinant)
            bad_hessian |= invalid.any() & active.any()
            update = active & ~invalid
            determinant = torch.where(update, determinant, torch.ones_like(determinant))
            da = (-ga * hbb + gb * hab) / determinant
            db = (ga * hab - haa * gb) / determinant
            if projected:
                bound = (ac <= minimum_scale) & (ga >= 0.0)
                da = torch.where(bound, torch.zeros_like(da), da)
                db = torch.where(bound, -gb / hbb, db)
            next_a = torch.where(update, torch.clamp(ac + da, min=minimum_scale), ac)
            next_b = torch.where(update, bc + db, bc)
            nonfinite = update & (~torch.isfinite(next_a) | ~torch.isfinite(next_b))
            bad_step |= nonfinite.any()
            update = update & ~nonfinite
            a.index_copy_(0, columns, torch.where(update, next_a, ac))
            b.index_copy_(0, columns, torch.where(update, next_b, bc))
            iterations.index_add_(0, columns, update.to(torch.int64))
            moved = update.any(dim=1)
        done += steps
        profile["batches"] += 1
        profile["steps"] += steps
        hessian_flag, step_flag, count = _read(bad_hessian, bad_step, moved.sum())
        if hessian_flag:
            raise ValueError("batched fixed-offset Hessian is not positive definite")
        if step_flag:
            raise ValueError("batched fixed-offset Newton step became nonfinite")
        if count:
            # A cell that did not move is converged and never moves again.
            columns = columns.index_select(0, _compact(moved, count))
    every = torch.arange(C, device=device, dtype=torch.int64)
    totals = sums(kind=_AFFINE, lanes=lanes, cells=cells, columns=every,
                  params=(a, b, weight, zero, zero, zero), grad=True, objective=False)
    ga = totals[..., 1] / rows_per_lane + 2.0 * strength * (a - 1.0)
    gb = totals[..., 2] / rows_per_lane + 2.0 * strength * b
    residual_a = ga.abs()
    if projected:
        residual_a = torch.where(a <= minimum_scale, torch.clamp(-ga, min=0.0), residual_a)
    residual = torch.maximum(residual_a, gb.abs())
    return a, b, iterations, residual


def fit_fixed_offset(
    problem: FusedProblem,
    *,
    strengths: Sequence[float],
    minimum_scale: float,
    max_iterations: int,
    kkt_tolerance: float,
    hessian_ridge: float,
    projected: bool,
) -> dict[str, Any]:
    """Selection, held-out losses, one-SE rule and deployment refit, all columns."""

    device = problem.device
    rows = problem.rows
    C, B, S = problem.columns, rows.block_count, len(strengths)
    if S < 1 or any(not math.isfinite(float(v)) or float(v) <= 0.0 for v in strengths):
        raise ValueError("batched fixed-offset anchors must be positive and finite")
    sums = _Sums(problem)
    lam = torch.as_tensor(tuple(float(v) for v in strengths), device=device, dtype=torch.float64)
    block_rows = torch.as_tensor(rows.block_rows, device=device, dtype=torch.float64)
    total = float(rows.rows)
    positives = problem.positives
    positive_all = positives.sum(dim=1)
    # Selection lanes (block-major): rows outside the lane's block.
    n_selection = (total - block_rows).repeat_interleave(S).unsqueeze(0).expand(C, B * S).contiguous()
    p_selection = (positive_all.unsqueeze(1) - positives).repeat_interleave(S, dim=1)
    w_selection = ((n_selection - p_selection) / p_selection).contiguous()
    lam_selection = lam.repeat(B).unsqueeze(0).expand(C, B * S).contiguous()
    # Held-out lanes: the 20 fitted lanes plus the identity lane of each block.
    n_block = block_rows.unsqueeze(0).expand(C, B)
    w_block = (n_block - positives) / positives
    n_heldout = torch.cat((n_block.repeat_interleave(S, dim=1), n_block), dim=1).contiguous()
    w_heldout = torch.cat((w_block.repeat_interleave(S, dim=1), w_block), dim=1).contiguous()
    n_deploy = torch.full((C, 1), total, device=device, dtype=torch.float64)
    w_deploy = ((n_deploy[:, 0] - positive_all) / positive_all).unsqueeze(1).contiguous()
    finite, support = _read(
        problem.finite_inputs(),
        _support(p_selection, n_selection) & _support(positives, n_block)
        & _support(positive_all, n_deploy[:, 0]),
    )
    if not finite:
        raise ValueError("batched fixed-offset inputs must be finite")
    if not support:
        raise ValueError("batched fixed-offset rows require both binary classes")
    if problem.sample_weights is not None:
        block_mass = torch.stack(tuple(problem.sample_weights[start:start + count].sum()
                                      for start, count in zip(rows.block_start, rows.block_rows)))
        mass = block_mass.sum()
        n_selection = (mass - block_mass).repeat_interleave(S).unsqueeze(0).expand(C, B * S).contiguous()
        held_mass = block_mass.unsqueeze(0).expand(C, B)
        n_heldout = torch.cat((held_mass.repeat_interleave(S, dim=1), held_mass), dim=1).contiguous()
        n_deploy = mass.expand(C, 1).contiguous()
        w_selection = torch.ones_like(w_selection)
        w_heldout = torch.ones_like(w_heldout)
        w_deploy = torch.ones_like(w_deploy)
    options = dict(minimum_scale=float(minimum_scale), max_iterations=int(max_iterations),
                   tolerance=float(kkt_tolerance), ridge=float(hessian_ridge), projected=bool(projected))
    profile = {"batches": 0, "steps": 0}
    selection_profile = dict(profile)
    a, b, iterations, residual = _offset_newton(
        sums, lanes=sums.table(SELECTION, S), cells=B * S, rows_per_lane=n_selection,
        weight=w_selection, strength=lam_selection, profile=selection_profile, **options,
    )
    zero = torch.zeros((C, B * S + B), device=device, dtype=torch.float64)
    a_heldout = torch.cat((a, torch.ones((C, B), device=device, dtype=torch.float64)), dim=1).contiguous()
    b_heldout = torch.cat((b, torch.zeros((C, B), device=device, dtype=torch.float64)), dim=1).contiguous()
    every = torch.arange(C, device=device, dtype=torch.int64)
    heldout = sums(kind=_AFFINE, lanes=sums.table(HELDOUT, S), cells=B * S + B, columns=every,
                   params=(a_heldout, b_heldout, w_heldout, zero, zero, zero), grad=False, objective=True)
    losses = heldout[..., 0] / n_heldout
    fold = torch.empty((C, B, S + 1), device=device, dtype=torch.float64)
    fold[:, :, :S] = losses[:, : B * S].reshape(C, B, S)
    fold[:, :, S] = losses[:, B * S :]
    selection = one_se_selection(fold)
    identity = selection["selected"] == S
    chosen = lam.index_select(0, torch.clamp(selection["selected"], max=S - 1))
    deploy_profile = dict(profile)
    da, db, d_iterations, d_residual = _offset_newton(
        sums, lanes=sums.table(DEPLOYMENT, S), cells=1, rows_per_lane=n_deploy,
        weight=w_deploy, strength=chosen.unsqueeze(1).contiguous(), profile=deploy_profile, **options,
    )
    converged = d_residual[:, 0] <= float(kkt_tolerance)
    scale = torch.where(identity, torch.ones_like(da[:, 0]), da[:, 0])
    offset = torch.where(identity, torch.zeros_like(db[:, 0]), db[:, 0])
    return {
        "scale": scale,
        "offset": offset,
        "converged": torch.where(identity, torch.ones_like(converged), converged),
        "identity_selected": identity,
        "selected_strength": chosen,
        "deployment_gradient_residual": d_residual[:, 0],
        "deployment_iterations": d_iterations[:, 0],
        "selection_iterations": iterations,
        "selection_gradient_residual": residual,
        "fold_losses": fold,
        "execution_profile": {
            "execution": EXECUTION,
            "active_poll_every": BATCH_STEPS,
            "statistics_rows_per_pass": int(rows.rows),
            "row_tiles_per_pass": int(rows.tile_count),
            "kernel_passes": int(sums.passes),
            "column_passes": int(sums.column_passes),
            "selection": selection_profile,
            "deployment": deploy_profile,
            "heldout_score_statistics_pass_count": 1,
            "triton": bool(sums.triton),
        },
    }


# ---------------------------------------------------------------------------
# Ordinary functional: proper-score identity-shrinkage affine.


def _chart(sums: _Sums, *, lanes: Any, cells: int, weight: torch.Tensor, strength: torch.Tensor,
           minimum_scale: float, identifiability: float) -> dict[str, torch.Tensor]:
    """``_fit_batched_post_query_affine_core``'s anchored setup, per lane."""

    problem = sums.problem
    C = problem.columns
    zero = torch.zeros((C, cells), device=problem.device, dtype=torch.float64)
    every = torch.arange(C, device=problem.device, dtype=torch.int64)
    first = sums(kind=_CHART, lanes=lanes, cells=cells, columns=every,
                 params=(zero, zero, weight, zero, zero, zero), grad=False, objective=False)
    weight_sum = first[..., 0].clamp_min(1e-30)
    location = (first[..., 1] / weight_sum).contiguous()
    second = sums(kind=_CHART, lanes=lanes, cells=cells, columns=every,
                  params=(zero, zero, weight, location, zero, zero), grad=False, objective=False)
    raw_spread = torch.sqrt(second[..., 4] / weight_sum)
    identifiable = raw_spread > identifiability
    spread = torch.where(identifiable, raw_spread, torch.ones_like(raw_spread)).contiguous()
    return {
        "location": location,
        "raw_spread": raw_spread,
        "spread": spread,
        "identifiable": identifiable,
        "minimum": spread * float(minimum_scale),
        "anchor_constant": strength * first[..., 3] / (weight_sum if problem.sample_weights is not None else float(problem.rows.total_rows)),
        **({"normalizer": weight_sum} if problem.sample_weights is not None else {}),
        "weighted_positive": first[..., 2],
        "weighted_negative": first[..., 5],
        "weight": weight,
        "strength": strength,
    }


def _anchored_solve(
    sums: _Sums,
    *,
    lanes: Any,
    cells: int,
    chart: dict[str, torch.Tensor],
    frozen: torch.Tensor | None,
    initial_objective: torch.Tensor | None,
    max_iterations: int,
    line_search_limit: int,
    tolerance: float,
    ridge: float,
    objective_tolerance: float,
    profile: dict[str, int],
    substituted_reporting: bool = False,
) -> dict[str, torch.Tensor]:
    """``_guarded_projected_newton`` + ``_newton_final_gates`` + the v6 KKT and
    feasibility gates, over the cells that are not ``frozen``."""

    problem = sums.problem
    device = problem.device
    C = problem.columns
    N = chart.get("normalizer", float(problem.rows.total_rows))
    beta = chart["spread"].clone()
    gamma = chart["location"].clone()
    if substituted_reporting and "initial_gamma" in chart:
        gamma = chart["initial_gamma"].clone()
    probe_beta = beta.clone()
    probe_gamma = gamma.clone()
    weight, location, spread, strength = chart["weight"], chart["location"], chart["spread"], chart["strength"]
    anchor, minimum = chart["anchor_constant"], chart["minimum"]
    live = torch.ones((C, cells), device=device, dtype=torch.bool) if frozen is None else ~frozen
    converged = ~live
    failed = torch.zeros_like(converged)
    iterations = torch.zeros((C, cells), device=device, dtype=torch.int64)
    line_steps = torch.zeros_like(iterations)
    start = (
        torch.full((C, cells), float("nan"), device=device, dtype=torch.float64)
        if initial_objective is None
        else initial_objective.clone()
    )
    boundary_slack = max(float(tolerance), 1e-12)
    floor = max(float(ridge) ** 2, 1e-30)
    if frozen is None:
        count = C
        columns = torch.arange(C, device=device, dtype=torch.int64)
    else:
        live_columns = live.any(dim=1)
        (count,) = _read(live_columns.sum())
        columns = _compact(live_columns, count)
    solve_columns = columns

    def stats(params_beta, params_gamma, cols, *, grad):
        totals = sums(kind=_ANCHORED, lanes=lanes, cells=cells, columns=cols,
                      params=(params_beta, params_gamma, weight, location, spread, strength),
                      grad=grad, objective=True)
        divisor = N.index_select(0, cols).unsqueeze(-1) if problem.sample_weights is not None else N
        return totals / divisor, anchor.index_select(0, cols)

    for iteration in range(1, max(1, int(max_iterations)) + 1):
        if not count:
            break
        totals, anchor_c = stats(beta, gamma, columns, grad=True)
        objective = totals[..., 0] - anchor_c
        gs, go, hss, hso, hoo = (totals[..., k] for k in range(1, 6))
        if initial_objective is None and iteration == 1:
            start.index_copy_(0, columns, objective)
        bc, gc, mc = beta.index_select(0, columns), gamma.index_select(0, columns), minimum.index_select(0, columns)
        at_boundary = bc <= mc + boundary_slack
        if substituted_reporting:
            from mechanistic_systems.backends.device_calibration_fit import _firth_projected_kkt
            kkt = _firth_projected_kkt(bc, mc, gs, go, hss, hoo,
                                      tolerance=tolerance, convergence="fisher_standardized")
        else:
            kkt = torch.maximum(torch.where(at_boundary, torch.clamp(-gs, min=0.0), gs.abs()), go.abs())
        done = converged.index_select(0, columns) | (kkt <= float(tolerance))
        converged.index_copy_(0, columns, done)
        fail_c = failed.index_select(0, columns)
        active = ~(done | fail_c)
        iterations.index_copy_(0, columns, torch.where(active, torch.full_like(kkt, iteration, dtype=torch.int64), iterations.index_select(0, columns)))
        hss = hss + float(ridge)
        hoo = hoo + float(ridge)
        determinant = hss * hoo - hso.square()
        stable = determinant > floor
        determinant = determinant.clamp_min(floor)
        coupled_s = (hoo * gs - hso * go) / determinant
        coupled_o = (hss * go - hso * gs) / determinant
        diagonal_s = gs / hss.clamp_min(float(ridge))
        diagonal_o = go / hoo.clamp_min(float(ridge))
        step_s = torch.where(stable, coupled_s, diagonal_s)
        step_o = torch.where(stable, coupled_o, diagonal_o)
        boundary_active = at_boundary & (gs >= 0.0)
        step_s = torch.where(boundary_active, torch.zeros_like(step_s), step_s)
        step_o = torch.where(boundary_active, diagonal_o, step_o)
        direction_s, direction_o = -step_s, -step_o
        accepted = torch.zeros_like(active)
        steps_c = line_steps.index_select(0, columns)
        local = torch.arange(int(columns.numel()), device=device, dtype=torch.int64)
        for k in range(int(line_search_limit)):
            alpha = 0.5 ** k
            sub_cols = columns.index_select(0, local)
            sb, sg = bc.index_select(0, local), gc.index_select(0, local)
            ps = torch.maximum(sb + alpha * direction_s.index_select(0, local), mc.index_select(0, local))
            po = sg + alpha * direction_o.index_select(0, local)
            probe_beta.index_copy_(0, sub_cols, ps)
            probe_gamma.index_copy_(0, sub_cols, po)
            probe, probe_anchor = stats(probe_beta, probe_gamma, sub_cols, grad=False)
            proposal = probe[..., 0] - probe_anchor
            directional = gs.index_select(0, local) * (ps - sb) + go.index_select(0, local) * (po - sg)
            sufficient = (
                proposal <= objective.index_select(0, local) + 1e-4 * directional + float(objective_tolerance)
            ) & torch.isfinite(proposal)
            take = active.index_select(0, local) & ~accepted.index_select(0, local) & sufficient
            bc.index_copy_(0, local, torch.where(take, ps, sb))
            gc.index_copy_(0, local, torch.where(take, po, sg))
            steps_c.index_copy_(0, local, torch.where(take, torch.full_like(steps_c.index_select(0, local), k + 1), steps_c.index_select(0, local)))
            accepted.index_copy_(0, local, accepted.index_select(0, local) | take)
            pending = (active & ~accepted).any(dim=1)
            survivors = (active & accepted).any(dim=1)
            profile["line_search_passes"] += 1
            waiting, count = _read(pending.sum(), survivors.sum())
            if not waiting:
                break
            local = _compact(pending, waiting)
        failed.index_copy_(0, columns, fail_c | (active & ~accepted))
        beta.index_copy_(0, columns, bc)
        gamma.index_copy_(0, columns, gc)
        line_steps.index_copy_(0, columns, steps_c)
        profile["iterations"] += 1
        if count:
            # A converged or failed cell never moves again.
            columns = columns.index_select(0, _compact(survivors, count))
    totals, anchor_c = stats(beta, gamma, solve_columns, grad=True)
    final_c = totals[..., 0] - anchor_c
    final = torch.full((C, cells), float("nan"), device=device, dtype=torch.float64)
    final.index_copy_(0, solve_columns, final_c)
    gs = torch.zeros_like(final)
    go = torch.zeros_like(final)
    gs.index_copy_(0, solve_columns, totals[..., 1])
    go.index_copy_(0, solve_columns, totals[..., 2])
    at_boundary = beta <= minimum + boundary_slack
    if substituted_reporting:
        from mechanistic_systems.backends.device_calibration_fit import _firth_projected_kkt
        hss, hoo = torch.zeros_like(final), torch.zeros_like(final)
        hss.index_copy_(0, solve_columns, totals[..., 3])
        hoo.index_copy_(0, solve_columns, totals[..., 5])
        kkt = _firth_projected_kkt(beta, minimum, gs, go, hss, hoo,
                                  tolerance=tolerance, convergence="fisher_standardized")
    else:
        kkt = torch.maximum(torch.where(at_boundary, torch.clamp(-gs, min=0.0), gs.abs()), go.abs())
    finite = torch.isfinite(beta) & torch.isfinite(gamma) & torch.isfinite(final) & torch.isfinite(kkt)
    worsened = final > start + float(objective_tolerance)
    within = kkt <= float(tolerance)
    certified = (converged | within) & finite & ~worsened & ~failed
    certified = certified & within & (beta >= minimum)
    return {
        "beta": beta,
        "gamma": gamma,
        "initial_objective": start,
        "final_objective": final,
        "kkt": kkt,
        "iterations": iterations,
        "line_steps": line_steps,
        "finite": finite,
        "converged": certified,
        "objective_worsened": worsened,
        "failed_line_search": failed,
    }


_FALLBACK_FIELDS = (
    "beta", "gamma", "final_objective", "kkt", "iterations", "line_steps",
    "finite", "converged", "objective_worsened", "failed_line_search",
)


def _substituted_reference_initial_gamma(problem, lanes, cells, chart):
    """Weighted-probability starts for constant substituted readouts only."""
    from mechanistic_systems.backends.device_calibration_fit import fixed_offset_mean_probability_intercept

    gamma = chart["location"].clone()
    for column in range(problem.columns):
        if bool(chart["identifiable"][column].all().item()):
            continue
        target = problem.labels.eq(problem.class_ids[column]).to(torch.float64)
        if problem.offset_mode == _OFFSET_PLAN:
            dense = problem.offsets[column].to(torch.float64)
            exclusion = torch.log(problem.row_shifted_sum - torch.exp(dense - problem.row_maximum)) + problem.row_maximum
            exclusion = torch.where(problem.row_argmax == problem.class_ids[column],
                                    problem.row_second_lse, exclusion)
            fixed = -exclusion
        elif problem.offset_mode == _OFFSET_EXPLICIT:
            fixed = problem.offsets[column].to(torch.float64)
        else:
            fixed = torch.zeros_like(target)
        initial_by_blocks = {}
        for cell in range(cells):
            if bool(chart["identifiable"][column, cell].item()):
                continue
            included = tuple(block for block, table in enumerate(lanes[0]) if cell in table)
            if included not in initial_by_blocks:
                valid = torch.zeros_like(target, dtype=torch.bool)
                for block in included:
                    start, count = problem.rows.block_start[block], problem.rows.block_rows[block]
                    valid[start:start + count] = True
                weights = torch.where(target > 0.5, chart["weight"][column, cell], 1.0)
                if problem.sample_weights is not None:
                    weights = weights * problem.sample_weights
                initial_by_blocks[included] = fixed_offset_mean_probability_intercept(
                    target[valid], weights[valid], fixed[valid])
            gamma[column, cell] = initial_by_blocks[included] + chart["location"][column, cell]
    return gamma


def _anchored_stage(
    sums: _Sums,
    *,
    lanes: Any,
    cells: int,
    weight: torch.Tensor,
    strength: torch.Tensor,
    minimum_scale: float,
    max_iterations: int,
    tolerance: float,
    ridge: float,
    objective_tolerance: float,
    reference_tolerance: float,
    fallback_iteration_factor: int,
    fallback_line_search_limit: int,
    line_search_limit: int,
    intercept_only: bool,
    profile: dict[str, int],
    substituted_reporting: bool = False,
) -> dict[str, torch.Tensor]:
    """Fit every (column, lane) cell, restarting unconverged cells from the
    same start with more iterations and the longer line-search ladder."""

    chart = _chart(sums, lanes=lanes, cells=cells, weight=weight, strength=strength,
                   minimum_scale=minimum_scale,
                   identifiability=max(float(tolerance), float(reference_tolerance)))
    options = dict(tolerance=tolerance, ridge=ridge, objective_tolerance=objective_tolerance)
    if substituted_reporting:
        chart["initial_gamma"] = _substituted_reference_initial_gamma(sums.problem, lanes, cells, chart)
        options["substituted_reporting"] = True
    production = _anchored_solve(
        sums, lanes=lanes, cells=cells, chart=chart, frozen=None, initial_objective=None,
        max_iterations=max_iterations, line_search_limit=line_search_limit, profile=profile, **options,
    )
    retry = ~(production["finite"] & production["converged"])
    gate = torch.zeros(retry.shape, device=retry.device, dtype=torch.int8)
    for code, failing in ((4, ~production["converged"]), (3, production["failed_line_search"]),
                          (2, production["objective_worsened"]), (1, ~production["finite"])):
        gate = torch.where(failing, torch.full_like(gate, code), gate)
    status = torch.zeros_like(gate)
    (retried,) = _read(retry.sum())
    result = dict(production)
    rerun = None
    if retried:
        profile["fallback_cells"] += int(retried)
        rerun = _anchored_solve(
            sums, lanes=lanes, cells=cells, chart=chart, frozen=~retry,
            initial_objective=production["initial_objective"],
            max_iterations=max(1, int(max_iterations)) * max(1, int(fallback_iteration_factor)),
            line_search_limit=fallback_line_search_limit, profile=profile, **options,
        )
        adopt = retry & rerun["finite"] & rerun["converged"]
        if substituted_reporting:
            adopt = retry & rerun["finite"]
        for name in _FALLBACK_FIELDS:
            result[name] = torch.where(adopt, rerun[name], production[name])
        rescued = retry & rerun["finite"] & rerun["converged"]
        status = torch.where(rescued, torch.ones_like(status), status)
        status = torch.where(retry & ~rescued, torch.full_like(status, 2), status)
    result["certified"] = result["converged"] & result["finite"]
    result["fallback_status"] = status
    result["production_gate"] = gate
    result["rerun"] = rerun
    result["chart"] = chart
    result["raw_scale"] = result["beta"] / chart["spread"]
    result["raw_offset"] = result["gamma"] - result["raw_scale"] * chart["location"]
    if intercept_only:
        C = sums.problem.columns
        every = torch.arange(C, device=weight.device, dtype=torch.int64)
        null = torch.log(chart["weighted_positive"].clamp_min(1e-30) / chart["weighted_negative"].clamp_min(1e-30))
        totals = sums(kind=_ANCHORED, lanes=lanes, cells=cells, columns=every,
                      params=(torch.zeros_like(null), null.contiguous(), weight, chart["location"],
                              chart["spread"], strength), grad=False, objective=True)
        result["intercept_only_objective"] = totals[..., 0] / chart.get("normalizer", float(sums.problem.rows.total_rows)) - chart["anchor_constant"]
    return result


def fit_ordinary(
    problem: FusedProblem,
    *,
    strengths: Sequence[float],
    minimum_scale: float,
    max_iterations: int,
    kkt_tolerance: float,
    hessian_ridge: float,
    objective_tolerance: float,
    reference_tolerance: float,
    anchor_probability_epsilon: float,
    fallback_iteration_factor: int,
    fallback_line_search_limit: int,
    line_search_limit: int = 30,
    substituted_reporting: bool = False,
) -> dict[str, Any]:
    """The proper-score identity-shrinkage fit of every column at once."""

    device = problem.device
    rows = problem.rows
    C, B, S = problem.columns, rows.block_count, len(strengths)
    if S < 1 or any(not math.isfinite(float(v)) or float(v) <= 0.0 for v in strengths):
        # A positive anchor on every lane is what makes the separation
        # gate vacuous here (separated requires anchor <= 0).
        raise ValueError("proper-score anchor strengths must be positive and finite")
    epsilon = float(anchor_probability_epsilon)
    if not 0.0 < epsilon < 0.5:
        raise ValueError("anchor probability epsilon must lie in (0,0.5)")
    if rows.rows != rows.total_rows:
        # An ordinary layout without every row in its one fold is refused.
        raise ValueError("cross-fit layout requires declared outer fold validity")
    sums = _Sums(problem, epsilon=epsilon)
    lam = torch.as_tensor(tuple(float(v) for v in strengths), device=device, dtype=torch.float64)
    block_rows = torch.as_tensor(rows.block_rows, device=device, dtype=torch.float64)
    total = float(rows.rows)
    positives = problem.positives
    positive_all = positives.sum(dim=1)
    n_selection = (total - block_rows).repeat_interleave(S).unsqueeze(0).expand(C, B * S)
    p_selection = (positive_all.unsqueeze(1) - positives).repeat_interleave(S, dim=1)
    w_selection = ((n_selection - p_selection) / p_selection).contiguous()
    lam_selection = lam.repeat(B).unsqueeze(0).expand(C, B * S).contiguous()
    n_block = block_rows.unsqueeze(0).expand(C, B)
    w_block = (n_block - positives) / positives
    w_heldout = torch.cat((w_block.repeat_interleave(S, dim=1), w_block), dim=1).contiguous()
    n_heldout = torch.cat((n_block.repeat_interleave(S, dim=1), n_block), dim=1).contiguous()
    w_deploy = ((total - positive_all) / positive_all).unsqueeze(1).contiguous()
    finite, support = _read(
        problem.finite_inputs(),
        _support(p_selection, n_selection) & _support(positives, n_block)
        & _support(positive_all, torch.full_like(positive_all, total)),
    )
    if not finite:
        raise ValueError("calibration inputs and targets must be finite")
    if not support:
        raise ValueError("every calibration query requires positive and negative support within every row mask")
    if problem.sample_weights is not None:
        block_mass = torch.stack(tuple(problem.sample_weights[start:start + count].sum()
                                      for start, count in zip(rows.block_start, rows.block_rows)))
        held_mass = block_mass.unsqueeze(0).expand(C, B)
        n_heldout = torch.cat((held_mass.repeat_interleave(S, dim=1), held_mass), dim=1).contiguous()
        w_selection = torch.ones_like(w_selection)
        w_heldout = torch.ones_like(w_heldout)
        w_deploy = torch.ones_like(w_deploy)
    options = dict(
        minimum_scale=float(minimum_scale), max_iterations=int(max_iterations),
        tolerance=float(kkt_tolerance), ridge=float(hessian_ridge),
        objective_tolerance=float(objective_tolerance), reference_tolerance=float(reference_tolerance),
        fallback_iteration_factor=int(fallback_iteration_factor),
        fallback_line_search_limit=int(fallback_line_search_limit),
        line_search_limit=int(line_search_limit),
    )
    if substituted_reporting:
        options["substituted_reporting"] = True
    selection_profile = {"iterations": 0, "line_search_passes": 0, "fallback_cells": 0}
    selection = _anchored_stage(
        sums, lanes=sums.table(SELECTION, S), cells=B * S, weight=w_selection,
        strength=lam_selection, intercept_only=False, profile=selection_profile, **options,
    )
    # Held-out class-balanced BCE of every fitted lane's raw affine and of
    # the identity, on the rows of the lane's block.
    zero = torch.zeros((C, B * S + B), device=device, dtype=torch.float64)
    a_heldout = torch.cat((selection["raw_scale"], torch.ones((C, B), device=device, dtype=torch.float64)), dim=1).contiguous()
    b_heldout = torch.cat((selection["raw_offset"], torch.zeros((C, B), device=device, dtype=torch.float64)), dim=1).contiguous()
    every = torch.arange(C, device=device, dtype=torch.int64)
    heldout = sums(kind=_AFFINE, lanes=sums.table(HELDOUT, S), cells=B * S + B, columns=every,
                   params=(a_heldout, b_heldout, w_heldout, zero, zero, zero), grad=False,
                   objective=True)
    losses = heldout[..., 0] / n_heldout
    fold = torch.empty((C, B, S + 1), device=device, dtype=torch.float64)
    fold[:, :, :S] = losses[:, : B * S].reshape(C, B, S)
    fold[:, :, S] = losses[:, B * S :]
    chosen = one_se_selection(fold)
    selection_missing = ~selection["certified"].all(dim=1)
    selected = torch.where(selection_missing, torch.zeros_like(chosen["selected"]), chosen["selected"])
    identity = selected == S
    strength = lam.index_select(0, torch.clamp(selected, max=S - 1))
    (deploy_needed,) = _read((~identity if substituted_reporting else ~(identity | selection_missing)).any())
    deployment = None
    deploy_profile = {"iterations": 0, "line_search_passes": 0, "fallback_cells": 0}
    if deploy_needed:
        deployment = _anchored_stage(
            sums, lanes=sums.table(DEPLOYMENT, S), cells=1, weight=w_deploy,
            strength=strength.unsqueeze(1).contiguous(), intercept_only=True,
            profile=deploy_profile, **options,
        )
        deployment_missing = ~identity & ~selection_missing & ~deployment["certified"][:, 0]
        if substituted_reporting:
            deployment_missing = ~identity & ~deployment["certified"][:, 0]
        scale = deployment["raw_scale"][:, 0]
        offset = deployment["raw_offset"][:, 0]
    else:
        deployment_missing = torch.zeros_like(identity)
        scale = torch.ones((C,), device=device, dtype=torch.float64)
        offset = torch.zeros_like(scale)
    missing = selection_missing | deployment_missing
    scale = torch.where(identity, torch.ones_like(scale), scale)
    offset = torch.where(identity, torch.zeros_like(offset), offset)
    if not substituted_reporting:
        scale = torch.where(missing, torch.full_like(scale, float("nan")), scale)
        offset = torch.where(missing, torch.full_like(offset, float("nan")), offset)
    return {
        "scale": scale,
        "offset": offset,
        "identity_selected": identity,
        "selected_index": selected,
        "selected_strength": strength,
        "selection_missing": selection_missing,
        "deployment_missing": deployment_missing,
        "missing": missing,
        "fold_losses": fold,
        "selection_rule": chosen,
        "selection": selection,
        "deployment": deployment,
        "execution_profile": {
            "execution": EXECUTION,
            "statistics_rows_per_pass": int(rows.rows),
            "row_tiles_per_pass": int(rows.tile_count),
            "kernel_passes": int(sums.passes),
            "column_passes": int(sums.column_passes),
            "selection": selection_profile,
            "deployment": deploy_profile,
            "triton": bool(sums.triton),
        },
    }


__all__ = [
    "EXECUTION",
    "FusedProblem",
    "FusedRows",
    "fit_fixed_offset",
    "fit_ordinary",
    "fused_rows",
    "lane_table",
    "one_se_selection",
]
