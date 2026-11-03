"""Experimental exact-row linear contractions; import only on admitted CUDA.

No matrix product factoring and no atomics. GPU compile/execution qualification
is separate from this source proposal. The two eager FP32 products and the
FP64 gate/negation are explicit PTX instructions, including their rounding.
"""
import triton
import triton.language as tl


@triton.jit
def _row(cotangent, activation, weight, gate):
    rounded = tl.inline_asm_elementwise(
        "{ .reg .f32 tmp; mul.rn.f32 tmp, $1, $2; mul.rn.f32 $0, tmp, $3; }",
        constraints="=f,f,f,f", args=[cotangent, activation, weight],
        dtype=tl.float32, is_pure=True, pack=1,
    )
    return tl.inline_asm_elementwise(
        "{ .reg .f64 tmp; mul.rn.f64 tmp, $1, $2; neg.f64 $0, tmp; }",
        constraints="=d,d,d", args=[rounded.to(tl.float64), gate],
        dtype=tl.float64, is_pure=True, pack=1,
    )


@triton.jit
def _add_preserving_zero(left, right):
    # A padding zero is an exact no-op even if a rounded add could flush a
    # subnormal other operand. Only nonzero/nonzero sums consume error budget.
    return tl.where(left == 0.0, right, tl.where(right == 0.0, left, left + right))


@triton.jit
def rows_kernel(X, W, C, G, O, B: tl.constexpr, S: tl.constexpr,
                FIRST, FCOUNT: tl.constexpr,
                ROFF, X0: tl.constexpr, X1: tl.constexpr,
                W0: tl.constexpr, W1: tl.constexpr,
                C0: tl.constexpr, C1: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < B * FCOUNT * S
    s = i % S
    f = (i // S) % FCOUNT + FIRST
    b = i // (FCOUNT * S)
    x = tl.load(X + b * X0 + s * X1, valid, other=0.0)
    w = tl.load(W + f * W0 + s * W1, valid, other=0.0)
    c = tl.load(C + b * C0 + f * C1, valid, other=0.0)
    g = tl.load(G + ROFF + f * S + s, valid, other=0.0)
    tl.store(O + i, _row(c, x, w, g), valid)


@triton.jit
def forward_kernel(X, W, C, G, Z, P, B: tl.constexpr, S: tl.constexpr,
                   F: tl.constexpr, LANES: tl.constexpr, CHUNKS: tl.constexpr,
                   ROFF, X0: tl.constexpr, X1: tl.constexpr,
                   W0: tl.constexpr, W1: tl.constexpr,
                   C0: tl.constexpr, C1: tl.constexpr,
                   BLOCK_S: tl.constexpr, BLOCK_B: tl.constexpr,
                   BLOCK_F: tl.constexpr, ABSOLUTE: tl.constexpr):
    f = tl.program_id(0) * BLOCK_F + tl.arange(0, BLOCK_F)
    b = tl.program_id(1) * BLOCK_B + tl.arange(0, BLOCK_B)
    part_lane = tl.program_id(2)
    chunk = part_lane // LANES
    lane = part_lane % LANES
    s = chunk * BLOCK_S + tl.arange(0, BLOCK_S)
    x = tl.load(X + b[:, None] * X0 + s[None, :] * X1,
                (b[:, None] < B) & (s[None, :] < S), other=0.0)
    w = tl.load(W + f[:, None] * W0 + s[None, :] * W1,
                (f[:, None] < F) & (s[None, :] < S), other=0.0)
    c = tl.load(C + b[:, None] * C0 + f[None, :] * C1,
                (b[:, None] < B) & (f[None, :] < F), other=0.0)
    routes = ROFF + f[:, None] * S + s[None, :]
    fs_valid = (f[:, None] < F) & (s[None, :] < S)
    g = tl.load(G + routes, fs_valid, other=0.0)
    a = _row(c[:, :, None], x[:, None, :], w[None, :, :], g[None, :, :])
    z = tl.load(Z + routes * LANES + lane, fs_valid, other=0.0)
    if ABSOLUTE:
        a = tl.abs(a)
        z = tl.abs(z)
    valid = (b[:, None, None] < B) & fs_valid[None, :, :]
    terms = tl.where(valid, a * z[None, :, :], 0.0)
    total = tl.reduce(terms, axis=2, combine_fn=_add_preserving_zero)
    offset = ((b[:, None] * F + f[None, :]) * LANES + lane) * CHUNKS + chunk
    tl.store(P + offset, total, (b[:, None] < B) & (f[None, :] < F))


@triton.jit
def forward_reduce_kernel(P, O, B: tl.constexpr, F: tl.constexpr,
                          LANES: tl.constexpr, CHUNKS: tl.constexpr,
                          GLOBAL_F: tl.constexpr, CONTEXT_START, FOFF,
                          BLOCK_CELLS: tl.constexpr, BLOCK_CHUNKS: tl.constexpr):
    cell = tl.program_id(0) * BLOCK_CELLS + tl.arange(0, BLOCK_CELLS)
    chunk = tl.arange(0, BLOCK_CHUNKS)
    partial = tl.load(P + cell[:, None] * CHUNKS + chunk[None, :],
        (cell[:, None] < B * F * LANES) & (chunk[None, :] < CHUNKS), other=0.0)
    total = tl.reduce(partial, axis=1, combine_fn=_add_preserving_zero)
    lane = cell % LANES
    f = (cell // LANES) % F
    b = cell // (LANES * F)
    target = ((CONTEXT_START + b) * GLOBAL_F + FOFF + f) * LANES + lane
    tl.store(O + target, total, cell < B * F * LANES)


@triton.jit
def adjoint_kernel(X, W, C, G, Y, O, B: tl.constexpr, S: tl.constexpr,
                   F: tl.constexpr, LANES: tl.constexpr, GLOBAL_F: tl.constexpr,
                   CONTEXT_START, FOFF, ROFF,
                   X0: tl.constexpr, X1: tl.constexpr,
                   W0: tl.constexpr, W1: tl.constexpr,
                   C0: tl.constexpr, C1: tl.constexpr,
                   BLOCK_B: tl.constexpr, BLOCK_S: tl.constexpr,
                   BLOCK_F: tl.constexpr, ABSOLUTE: tl.constexpr):
    tile = tl.program_id(0)
    lane = tl.program_id(1)
    chunks = tl.cdiv(S, BLOCK_S)
    f = (tile // chunks) * BLOCK_F + tl.arange(0, BLOCK_F)
    s = (tile % chunks) * BLOCK_S + tl.arange(0, BLOCK_S)
    b = tl.arange(0, BLOCK_B)
    x = tl.load(X + b[:, None] * X0 + s[None, :] * X1,
                (b[:, None] < B) & (s[None, :] < S), other=0.0)
    w = tl.load(W + f[:, None] * W0 + s[None, :] * W1,
                (f[:, None] < F) & (s[None, :] < S), other=0.0)
    c = tl.load(C + b[:, None] * C0 + f[None, :] * C1,
                (b[:, None] < B) & (f[None, :] < F), other=0.0)
    routes = ROFF + f[:, None] * S + s[None, :]
    fs_valid = (f[:, None] < F) & (s[None, :] < S)
    g = tl.load(G + routes, fs_valid, other=0.0)
    a = _row(c[:, :, None], x[:, None, :], w[None, :, :], g[None, :, :])
    y = tl.load(Y + ((CONTEXT_START + b[:, None]) * GLOBAL_F + FOFF + f[None, :]) * LANES + lane,
                (b[:, None] < B) & (f[None, :] < F), other=0.0)
    if ABSOLUTE:
        a = tl.abs(a)
        y = tl.abs(y)
    valid = (b[:, None, None] < B) & fs_valid[None, :, :]
    terms = tl.where(valid, a * y[:, :, None], 0.0)
    total = tl.reduce(terms, axis=0, combine_fn=_add_preserving_zero)
    out = O + routes * LANES + lane
    # One packet kernel at a time on the same stream; each output route is
    # written once per launch. No atomics and no simultaneous packet writers.
    previous = tl.load(out, fs_valid, other=0.0)
    tl.store(out, previous + total, fs_valid)
