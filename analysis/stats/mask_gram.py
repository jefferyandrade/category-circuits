"""The Gram matrix of circuit masks: how many kept weights each pair of circuits shares, counted exactly.

overlap_nulls.py uses it for the overlap within each weight sign (Fig. 4D).
"""
from __future__ import annotations

import numpy as np


class PackedRuns:
    """The masks of R circuits on one layer, packed with np.packbits (default bit order), and the layer's number of
    weights N. N matters when it is not a multiple of 8: the padding bits at the end are never counted."""

    def __init__(self, bits, N):
        self.bits = np.asarray(bits)
        self.N = int(N)
        if self.N < 0 or self.bits.ndim != 2 or self.bits.dtype != np.uint8:
            raise ValueError("PackedRuns needs a uint8 matrix and nonnegative N")
        if self.bits.shape[1] != (self.N + 7) // 8:
            raise ValueError("packed width does not match N")


def gram(X, chunk_routes=1048576):
    """The R x R matrix of exact intersection counts of R masks, unpacking one chunk of weights at a time.

    X is an (R, N) bool array or PackedRuns. Each float32 product counts fewer than 2**24 weights, so every
    intermediate value is an exact integer; the chunks are summed in int64.
    """
    if isinstance(chunk_routes, (bool, np.bool_)) or int(chunk_routes) != chunk_routes:
        raise ValueError("chunk_routes must be an integer")
    chunk_routes = int(chunk_routes)
    if not 0 < chunk_routes < 2**24:
        raise ValueError("chunk_routes must be between 1 and 2**24 - 1")
    packed = isinstance(X, PackedRuns)
    if packed:
        R, N = X.bits.shape[0], X.N
    else:
        X = np.asarray(X)
        if X.ndim != 2 or X.dtype != np.bool_:
            raise ValueError("X must be an (R, N) bool array or PackedRuns")
        R, N = X.shape
    result = np.zeros((R, R), dtype=np.int64)
    for start in range(0, N, chunk_routes):
        stop = min(start + chunk_routes, N)
        if packed:
            byte_start, byte_stop = start // 8, (stop + 7) // 8
            bits = np.unpackbits(X.bits[:, byte_start:byte_stop], axis=1)
            offset = start - 8 * byte_start
            chunk = bits[:, offset:offset + stop - start].astype(np.float32)
            del bits
        else:
            chunk = X[:, start:stop].astype(np.float32)
        result += (chunk @ chunk.T).astype(np.int64)
        del chunk
    return result
