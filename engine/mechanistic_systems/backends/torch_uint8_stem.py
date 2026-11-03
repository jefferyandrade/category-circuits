"""One-pass uint8 stem staging in channels-last, with tensor-core channel padding."""

from __future__ import annotations

from functools import lru_cache

import torch

try:  # Triton is present on the CUDA target, optional for CPU execution.
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


# Four-channel float32 NHWC stems avoid doubling the three-channel AlexNet
# workload while preserving cuDNN's tensor-core layout on the H200.
STEM_CHANNEL_MULTIPLE = 4


@lru_cache(maxsize=1)
def _pad_kernel():
    if triton is None:
        raise RuntimeError("CUDA uint8 stem staging requires Triton")

    @triton.jit
    def pad_convert(Input, Pad, Output, N: tl.constexpr, C: tl.constexpr,
                    H: tl.constexpr, W: tl.constexpr, PH: tl.constexpr,
                    PW: tl.constexpr, PC: tl.constexpr, SN: tl.constexpr,
                    SC: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
                    BLOCK: tl.constexpr):
        # Large later looks can exceed 2**31 padded float elements.
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        c = i % PC
        x = (i // PC) % (W + 2 * PW) - PW
        y = (i // (PC * (W + 2 * PW))) % (H + 2 * PH) - PH
        n = i // (PC * (W + 2 * PW) * (H + 2 * PH))
        valid = n < N
        interior = valid & (c < C) & (x >= 0) & (x < W) & (y >= 0) & (y < H)
        raw = tl.load(Input + n * SN + c * SC + y * SH + x * SW,
                      mask=interior, other=0).to(tl.float32)
        edge = tl.load(Pad + c, mask=valid & (c < C), other=0)
        value = tl.where(interior, raw, edge)
        tl.store(Output + i, value, mask=valid)

    return pad_convert


@torch.library.custom_op("irg::uint8_stem_pad", mutates_args=())
def uint8_stem_pad(inputs: torch.Tensor, pad_value: torch.Tensor,
                   padding_h: int, padding_w: int, channels: int) -> torch.Tensor:
    """Write the sole float input buffer directly; extra channels are zero.

    The CUDA implementation fuses conversion, spatial edge fill and NHWC
    layout. The CPU implementation is the same map expressed in Torch for
    CPU execution and focused numerical tests.
    """
    if inputs.ndim != 4 or inputs.dtype != torch.uint8:
        raise ValueError("uint8 stem staging requires NCHW-shaped uint8 images")
    n, c, h, w = inputs.shape
    if channels < c or padding_h < 0 or padding_w < 0:
        raise ValueError("invalid stem padding")
    if pad_value.numel() != c or pad_value.device != inputs.device:
        raise ValueError("stem padding constants must match the input channels and device")
    output = torch.empty(
        (n, channels, h + 2 * padding_h, w + 2 * padding_w),
        dtype=torch.float32, device=inputs.device, memory_format=torch.channels_last,
    )
    if inputs.device.type == "cuda":
        _pad_kernel()[(triton.cdiv(output.numel(), 1024),)](
            inputs, pad_value, output, n, c, h, w, padding_h, padding_w,
            channels, *inputs.stride(), BLOCK=1024,
        )
    else:
        output.zero_()
        output[:, :c].copy_(pad_value.reshape(1, c, 1, 1))
        output[:, :c, padding_h:padding_h + h, padding_w:padding_w + w].copy_(inputs)
    return output


@uint8_stem_pad.register_fake
def _fake_uint8_stem_pad(inputs, pad_value, padding_h, padding_w, channels):
    n, _, h, w = inputs.shape
    return torch.empty(
        (n, channels, h + 2 * padding_h, w + 2 * padding_w),
        dtype=torch.float32, device=inputs.device, memory_format=torch.channels_last,
    )


@uint8_stem_pad.register_vmap
def _vmap_uint8_stem_pad(info, in_dims, inputs, pad_value, padding_h, padding_w, channels):
    if in_dims[1] is not None:
        raise ValueError("stem decode constants must be shared across lanes")
    if in_dims[0] is None:
        return uint8_stem_pad(inputs, pad_value, padding_h, padding_w, channels), None
    images = inputs.movedim(in_dims[0], 0)
    lanes, rows = images.shape[:2]
    result = uint8_stem_pad(images.flatten(0, 1), pad_value, padding_h, padding_w, channels)
    return result.reshape(lanes, rows, *result.shape[1:]), 0


def pad_stem_weight(weight: torch.Tensor) -> torch.Tensor:
    """Pad only the physical channel axis; the executable route axis stays C."""
    channels = ((weight.shape[1] + STEM_CHANNEL_MULTIPLE - 1)
                // STEM_CHANNEL_MULTIPLE) * STEM_CHANNEL_MULTIPLE
    # Permute/contiguous also works inside vmap, whose BatchedTensor does not
    # admit contiguous(memory_format=channels_last).
    padded = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, channels - weight.shape[1]))
    return padded.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)


def prepare_stem_components(weight: torch.Tensor, pad: torch.Tensor) -> dict[str, torch.Tensor]:
    """Pack two BF16 components of a float32 weight along the output axis.

    Raw uint8 values are exactly representable in BF16. A second weight
    component retains the float32 weight's residual, and the spatial-edge
    residual preserves fractional decoding constants. All sums use float32.
    """
    high = weight.to(torch.bfloat16)
    low = (weight - high.float()).to(torch.bfloat16)
    outputs = weight.shape[0]
    return {
        "execution_stem_weight_hi": high.permute(2, 3, 1, 0).contiguous().reshape(-1, outputs),
        "execution_stem_weight_lo": low.permute(2, 3, 1, 0).contiguous().reshape(-1, outputs),
        "execution_stem_pad_residual": (pad - pad.to(torch.bfloat16).float()).to(torch.bfloat16),
    }


@lru_cache(maxsize=1)
def _bf16_pad_kernel():
    if triton is None:
        raise RuntimeError("CUDA uint8 stem convolution requires Triton")

    @triton.jit
    def convert(Input, Pad, Output, N: tl.constexpr, C: tl.constexpr,
                H: tl.constexpr, W: tl.constexpr, PH: tl.constexpr,
                PW: tl.constexpr, PC: tl.constexpr, LANES: tl.constexpr,
                SL: tl.constexpr, SN: tl.constexpr, SC: tl.constexpr,
                SH: tl.constexpr, SW: tl.constexpr, PL: tl.constexpr,
                BLOCK: tl.constexpr):
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        c = i % PC
        x = (i // PC) % (W + 2 * PW) - PW
        y = (i // (PC * (W + 2 * PW))) % (H + 2 * PH) - PH
        image = i // (PC * (W + 2 * PW) * (H + 2 * PH))
        lane, n = image // N, image % N
        valid = lane < LANES
        interior = valid & (c < C) & (x >= 0) & (x < W) & (y >= 0) & (y < H)
        raw = tl.load(Input + lane * SL + n * SN + c * SC + y * SH + x * SW,
                      mask=interior, other=0).to(tl.float32)
        edge = tl.load(Pad + lane * PL + c, mask=valid & (c < C), other=0)
        tl.store(Output + i, tl.where(interior, raw, edge), mask=valid)

    return convert


@lru_cache(maxsize=1)
def _bf16_conv_kernel():
    if triton is None:
        raise RuntimeError("CUDA uint8 stem convolution requires Triton")

    @triton.jit
    def convolution(Input, High, Low, Bias, PadResidual, Output,
                    N: tl.constexpr, C: tl.constexpr, PC: tl.constexpr,
                    CO: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
                    OH: tl.constexpr, OW: tl.constexpr, KH: tl.constexpr,
                    KW: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
                    DH: tl.constexpr, DW: tl.constexpr, PH: tl.constexpr,
                    PW: tl.constexpr, TOP: tl.constexpr, BOTTOM: tl.constexpr,
                    LEFT: tl.constexpr, RIGHT: tl.constexpr, XL: tl.constexpr,
                    HL: tl.constexpr, HK: tl.constexpr, HO: tl.constexpr,
                    LL: tl.constexpr, LK: tl.constexpr, LO: tl.constexpr,
                    BL: tl.constexpr, BS: tl.constexpr, PL: tl.constexpr,
                    PS: tl.constexpr, BM: tl.constexpr,
                    BN: tl.constexpr, BK: tl.constexpr, LANES: tl.constexpr,
                    WIDE: tl.constexpr, CORRECT: tl.constexpr):
        lane = tl.program_id(2) if LANES != 1 else 0
        if WIDE and LANES != 1:
            lane = lane.to(tl.int64)
        block_index = tl.program_id(0).to(tl.int64) if WIDE else tl.program_id(0)
        index = block_index * BM + tl.arange(0, BM)
        outputs = tl.program_id(1) * BN + tl.arange(0, BN)
        if WIDE:
            outputs = outputs.to(tl.int64)
        if CORRECT:
            top_size, bottom_size = TOP * OW, BOTTOM * OW
            middle_cols = LEFT + RIGHT
            edge_size = top_size + bottom_size + (OH - TOP - BOTTOM) * middle_cols
            batch, edge_index = index // edge_size, index % edge_size
            lower = edge_index - top_size
            middle = edge_index - top_size - bottom_size
            # With only top/bottom edges, the middle section is empty.
            columns = tl.maximum(middle_cols, 1)
            middle_x = middle % columns
            oy = tl.where(edge_index < top_size, edge_index // OW,
                 tl.where(edge_index < top_size + bottom_size,
                          OH - BOTTOM + lower // OW, TOP + middle // columns))
            ox = tl.where(edge_index < top_size, edge_index % OW,
                 tl.where(edge_index < top_size + bottom_size, lower % OW,
                          tl.where(middle_x < LEFT, middle_x,
                                   OW - RIGHT + middle_x - LEFT)))
            valid = index < N * edge_size
        else:
            batch, oy, ox = index // (OH * OW), (index // OW) % OH, index % OW
            valid = index < N * OH * OW
        row = (batch * OH + oy) * OW + ox if CORRECT else index
        reduction = tl.arange(0, BK)
        if WIDE:
            reduction = reduction.to(tl.int64)
        result = tl.zeros((BM, BN), tl.float32)
        for block in range(tl.cdiv(PC * KH * KW, BK)):
            k = block * BK + reduction
            channel, ky, kx = k % PC, k // (PC * KW), (k // PC) % KW
            valid_k = k < PC * KH * KW
            iy = oy[:, None] * SH + ky[None, :] * DH
            ix = ox[:, None] * SW + kx[None, :] * DW
            if CORRECT:
                residual = tl.load(PadResidual + lane * PL + channel * PS,
                                   mask=channel < C, other=0)
                edge = (iy < PH) | (iy >= IH - PH) | (ix < PW) | (ix >= IW - PW)
                values = tl.where(edge & valid[:, None] & valid_k[None, :],
                                  residual[None, :], 0).to(tl.bfloat16)
            else:
                values = tl.load(Input + lane * XL + ((batch[:, None] * IH + iy) * IW + ix)
                                 * PC + channel[None, :],
                                 mask=valid[:, None] & valid_k[None, :], other=0)
            weight_mask = valid_k[:, None] & (outputs[None, :] < CO)
            high = tl.load(High + lane * HL + k[:, None] * HK + outputs[None, :] * HO,
                           mask=weight_mask, other=0)
            low = tl.load(Low + lane * LL + k[:, None] * LK + outputs[None, :] * LO,
                          mask=weight_mask, other=0)
            result = tl.dot(values, high, result)
            result = tl.dot(values, low, result)
        output_pointer = Output + ((lane * N * OH * OW + row[:, None]) * CO
                                   + outputs[None, :])
        mask = valid[:, None] & (outputs[None, :] < CO)
        if CORRECT:
            result += tl.load(output_pointer, mask=mask, other=0)
        else:
            result += tl.load(Bias + lane * BL + outputs * BS, mask=outputs < CO, other=0)[None, :]
        tl.store(output_pointer, result, mask=mask)

    return convolution


def _lane_count(inputs, weight, bias, pad_value, weight_hi, weight_lo, pad_residual):
    counts = [value.shape[0] for value, rank in (
        (inputs, 4), (weight, 4), (bias, 1), (pad_value, 1),
        (weight_hi, 2), (weight_lo, 2), (pad_residual, 1),
    ) if value.ndim == rank + 1]
    if not counts:
        return 1, False
    lanes = counts[0]
    if any(count != lanes for count in counts):
        raise ValueError("uint8 stem lane dimensions must agree")
    return lanes, True


@torch.library.custom_op("irg::uint8_stem_conv", mutates_args=())
def uint8_stem_conv(inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                    pad_value: torch.Tensor, weight_hi: torch.Tensor,
                    weight_lo: torch.Tensor, pad_residual: torch.Tensor,
                    stride_h: int, stride_w: int, padding_h: int, padding_w: int,
                    dilation_h: int, dilation_w: int) -> torch.Tensor:
    """Float32 stem convolution over exact integer images and paired weights.

    Candidate lanes occupy a launch-grid dimension. The original float32
    weight is retained as the differentiation argument; the prepared BF16
    pair is its execution representation, refreshed with every route update.
    """
    if inputs.device.type != "cuda" or inputs.dtype != torch.uint8 or weight.dtype != torch.float32:
        raise ValueError("the fused stem requires CUDA uint8 inputs and float32 weights")
    lanes, batched = _lane_count(inputs, weight, bias, pad_value, weight_hi, weight_lo, pad_residual)
    n, c = inputs.shape[-4:-2]
    pc = weight.shape[-3]
    staged = _stage_stem_inputs(inputs, pad_value, pc, lanes, padding_h, padding_w)
    return _launch_stem_convolution(staged, weight, bias, weight_hi, weight_lo, pad_residual,
                                    n, c, lanes, batched, stride_h, stride_w, padding_h,
                                    padding_w, dilation_h, dilation_w)


def _stage_stem_inputs(inputs, pad_value, pc, lanes, padding_h, padding_w):
    """Write the BF16 execution buffer once; shared lane inputs share storage."""
    n, c, h, w = inputs.shape[-4:]
    ih, iw = h + 2 * padding_h, w + 2 * padding_w
    input_lane_stride = inputs.stride(0) if inputs.ndim == 5 else 0
    pad_lane_stride = pad_value.stride(0) if pad_value.ndim == 2 else 0
    stage_lanes = lanes if input_lane_stride or pad_lane_stride else 1
    staged = torch.empty((stage_lanes * n, pc, ih, iw), device=inputs.device,
                         dtype=torch.bfloat16, memory_format=torch.channels_last)
    _bf16_pad_kernel()[(triton.cdiv(staged.numel(), 1024),)](
        inputs, pad_value, staged, n, c, h, w, padding_h, padding_w, pc, stage_lanes,
        input_lane_stride, *inputs.stride()[-4:], pad_lane_stride, BLOCK=1024,
    )
    return staged


def _launch_stem_convolution(staged, weight, bias, weight_hi, weight_lo, pad_residual,
                              n, c, lanes, batched, stride_h, stride_w, padding_h,
                              padding_w, dilation_h, dilation_w):
    """Launch the same main/boundary core independently of input conversion."""
    co, pc, kh, kw = weight.shape[-4:]
    ih, iw = staged.shape[-2:]
    oh = (ih - dilation_h * (kh - 1) - 1) // stride_h + 1
    ow = (iw - dilation_w * (kw - 1) - 1) // stride_w + 1
    if oh < 1 or ow < 1:
        raise ValueError("the stem kernel exceeds its padded input")
    stage_lanes = staged.shape[0] // n
    output = torch.empty((lanes * n, co, oh, ow), device=staged.device,
                         dtype=torch.float32, memory_format=torch.channels_last)
    top, left = min(oh, triton.cdiv(padding_h, stride_h)), min(ow, triton.cdiv(padding_w, stride_w))
    bottom = oh - max(top, triton.cdiv(ih - padding_h - (kh - 1) * dilation_h, stride_h))
    right = ow - max(left, triton.cdiv(iw - padding_w - (kw - 1) * dilation_w, stride_w))
    bottom, right = max(0, bottom), max(0, right)
    edge_rows = (top + bottom) * ow + (oh - top - bottom) * (left + right)
    # Coordinate width depends only on address bounds. Large resident looks
    # retain int64 pointers without taxing the ordinary H200 look sizes.
    maximum_address = max(
        staged.numel(), output.numel(),
        *(1 + sum((size - 1) * stride for size, stride in zip(value.shape, value.stride()))
          for value in (weight_hi, weight_lo, bias, pad_residual)),
    )
    wide = maximum_address >= 2 ** 31
    bm, bn, bk = 128, min(64, max(16, triton.next_power_of_2(co))), 16
    kernel_args = (
        staged, weight_hi, weight_lo, bias, pad_residual, output, n, c, pc, co,
        ih, iw, oh, ow, kh, kw, stride_h, stride_w, dilation_h, dilation_w,
        padding_h, padding_w, top, bottom, left, right,
        staged.stride(0) * n if stage_lanes != 1 else 0,
        weight_hi.stride(0) if weight_hi.ndim == 3 else 0, *weight_hi.stride()[-2:],
        weight_lo.stride(0) if weight_lo.ndim == 3 else 0, *weight_lo.stride()[-2:],
        bias.stride(0) if bias.ndim == 2 else 0, bias.stride(-1),
        pad_residual.stride(0) if pad_residual.ndim == 2 else 0,
        pad_residual.stride(-1), bm, bn, bk,
    )
    _bf16_conv_kernel()[(triton.cdiv(n * oh * ow, bm), triton.cdiv(co, bn), lanes)](
        *kernel_args, LANES=lanes, WIDE=wide, CORRECT=False, num_warps=4, num_stages=3,
    )
    if edge_rows:
        _bf16_conv_kernel()[(triton.cdiv(n * edge_rows, bm), triton.cdiv(co, bn), lanes)](
            *kernel_args, LANES=lanes, WIDE=wide, CORRECT=True, num_warps=4, num_stages=3,
        )
    return output.view(lanes, n, co, oh, ow) if batched else output


@uint8_stem_conv.register_fake
def _fake_uint8_stem_conv(inputs, weight, bias, pad_value, weight_hi, weight_lo,
                          pad_residual, stride_h, stride_w, padding_h, padding_w,
                          dilation_h, dilation_w):
    lanes, batched = _lane_count(inputs, weight, bias, pad_value, weight_hi, weight_lo, pad_residual)
    n, _, h, w = inputs.shape[-4:]
    co, _, kh, kw = weight.shape[-4:]
    oh = (h + 2 * padding_h - dilation_h * (kh - 1) - 1) // stride_h + 1
    ow = (w + 2 * padding_w - dilation_w * (kw - 1) - 1) // stride_w + 1
    output = torch.empty((lanes * n, co, oh, ow), device=inputs.device,
                         dtype=torch.float32, memory_format=torch.channels_last)
    return output.view(lanes, n, co, oh, ow) if batched else output


@uint8_stem_conv.register_vmap
def _vmap_uint8_stem_conv(info, in_dims, inputs, weight, bias, pad_value, weight_hi,
                          weight_lo, pad_residual, stride_h, stride_w, padding_h,
                          padding_w, dilation_h, dilation_w):
    values = (inputs, weight, bias, pad_value, weight_hi, weight_lo, pad_residual)
    mapped = tuple(value.movedim(dim, 0) if dim is not None else value
                   for value, dim in zip(values, in_dims[:7]))
    result = uint8_stem_conv(*mapped, stride_h, stride_w, padding_h, padding_w,
                             dilation_h, dilation_w)
    return result, 0


def _stem_conv_context(ctx, inputs, output):
    raw, weight, bias, pad, _, _, _, *geometry = inputs
    ctx.save_for_backward(raw, weight, pad)
    ctx.geometry, ctx.bias_rank = geometry, bias.ndim
    ctx.batched = output.ndim == 5


def _stem_conv_backward(ctx, output_grad):
    raw, weight, pad = ctx.saved_tensors
    sh, sw, ph, pw, dh, dw = ctx.geometry
    gradient = output_grad if ctx.batched else output_grad.unsqueeze(0)
    lanes, n, co, oh, ow = gradient.shape
    c, h, w = raw.shape[-3:]
    _, pc, kh, kw = weight.shape[-4:]
    raw_lanes = raw if raw.ndim == 5 else raw.unsqueeze(0).expand(lanes, *raw.shape)
    pad_lanes = pad if pad.ndim == 2 else pad.unsqueeze(0).expand(lanes, *pad.shape)
    # Backward differentiates the original float32 convolution. This avoids
    # forcing float32 route/weight derivatives through a BF16 gradient tensor.
    raw_float = torch.nn.functional.pad(raw_lanes.float(), (pw, pw, ph, ph, 0, pc - c))
    ys = torch.arange(h + 2 * ph, device=raw.device)
    xs = torch.arange(w + 2 * pw, device=raw.device)
    interior = ((ys >= ph) & (ys < ph + h))[:, None] & ((xs >= pw) & (xs < pw + w))[None, :]
    constants = torch.nn.functional.pad(pad_lanes, (0, pc - c))[:, None, :, None, None]
    full = torch.where(interior[None, None, None], raw_float, constants)
    grouped_inputs = full.movedim(0, 1).reshape(n, lanes * pc, h + 2 * ph, w + 2 * pw)
    grouped_gradient = gradient.movedim(0, 1).reshape(n, lanes * co, oh, ow)
    weight_gradient = bias_gradient = pad_gradient = None
    if ctx.needs_input_grad[1]:
        value = torch.nn.grad.conv2d_weight(grouped_inputs, (lanes * co, pc, kh, kw),
                                          grouped_gradient, (sh, sw), 0, (dh, dw), lanes)
        value = value.reshape(lanes, co, pc, kh, kw)
        weight_gradient = value if weight.ndim == 5 else value.sum(dim=0)
    if ctx.needs_input_grad[2]:
        value = gradient.sum(dim=(1, 3, 4))
        bias_gradient = value if ctx.bias_rank == 2 else value.sum(dim=0)
    if ctx.needs_input_grad[3]:
        weights = weight if weight.ndim == 5 else weight.unsqueeze(0).expand(lanes, *weight.shape)
        value = torch.nn.grad.conv2d_input(grouped_inputs.shape, weights.reshape(lanes * co, pc, kh, kw),
                                         grouped_gradient, (sh, sw), 0, (dh, dw), lanes)
        value = value.reshape(n, lanes, pc, h + 2 * ph, w + 2 * pw).movedim(1, 0)
        value = (value[:, :, :c] * (~interior)[None, None, None]).sum(dim=(1, 3, 4))
        pad_gradient = value if pad.ndim == 2 else value.sum(dim=0)
    return (None, weight_gradient, bias_gradient, pad_gradient, None, None, None,
            None, None, None, None, None, None)


uint8_stem_conv.register_autograd(_stem_conv_backward, setup_context=_stem_conv_context)
