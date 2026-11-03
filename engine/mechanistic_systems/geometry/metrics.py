"""Numerical contracts for finite-dimensional response metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def validate_metric(
    metric: Sequence[Sequence[float]] | np.ndarray,
    *,
    psd_tolerance: float = 1e-8,
) -> np.ndarray:
    """Validate and return a symmetric positive-semidefinite metric.

    Tiny negative eigenvalues inside ``psd_tolerance`` receive the historical
    diagonal correction.  This preserves existing numerical behavior; a caller
    requiring a decision-bearing correction receipt must wrap this primitive
    with the repository tolerance registry rather than infer exact PSD input.
    """

    matrix = np.asarray(metric, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("quadratic metric must be a nonempty square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("quadratic metric must be finite")
    symmetric = 0.5 * (matrix + matrix.T)
    if np.max(np.abs(matrix - matrix.T)) > 1e-8 * (1.0 + np.max(np.abs(matrix))):
        raise ValueError("quadratic metric is not symmetric within tolerance")
    minimum = float(np.min(np.linalg.eigvalsh(symmetric)))
    if minimum < -abs(float(psd_tolerance)):
        raise ValueError(
            f"quadratic metric is not PSD: minimum eigenvalue {minimum:.3e}"
        )
    if minimum < 0.0:
        symmetric = symmetric + (-minimum + 1e-12) * np.eye(symmetric.shape[0])
    return symmetric


__all__ = ["validate_metric"]
