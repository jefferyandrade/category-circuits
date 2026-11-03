"""Certified bounded linear algebra for modern-v1 decision shells.

An ordinary floating eigensolve can propose a strong-convexity value, but
``nextafter(eigvalsh(...))`` is not a mathematical lower bound because the
eigensolver's backward error can exceed one ULP. This module certifies a
proposal by exact rational LDL decomposition of the FP64 shell coordinates.
The work is bounded by the registered shell dimension (eight in production),
never by the route axis.
"""

from __future__ import annotations

from fractions import Fraction
import math

import numpy as np


DEFAULT_MAXIMUM_CERTIFIED_DIMENSION = 8


def _fraction_upper_float(value: Fraction) -> float:
    if value < 0:
        raise ValueError("upper conversion expects a nonnegative fraction")
    rounded = float(value)
    if not math.isfinite(rounded):
        raise ValueError("certified bound is outside binary64")
    if Fraction.from_float(rounded) < value:
        rounded = math.nextafter(rounded, math.inf)
    if not math.isfinite(rounded) or Fraction.from_float(rounded) < value:
        raise ValueError("certified bound is outside binary64")
    return rounded


def _require_f64_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must be one two-dimensional float64 array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _require_optional_f64_vector(
    value: np.ndarray | None,
    *,
    dimension: int,
    name: str,
) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.shape != (dimension,) or array.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must be one float64 vector of shell dimension")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _exact_operator(
    factor: np.ndarray,
    *,
    residual_direction: np.ndarray | None,
    diagonal_shift: float,
) -> tuple[tuple[Fraction, ...], ...]:
    shell = factor.shape[1]
    factor_fraction = tuple(
        tuple(
            Fraction.from_float(float(factor[row, column]))
            for column in range(shell)
        )
        for row in range(factor.shape[0])
    )
    residual_fraction = (
        None
        if residual_direction is None
        else tuple(Fraction.from_float(float(value)) for value in residual_direction)
    )
    shift = Fraction.from_float(diagonal_shift)
    matrix: list[list[Fraction]] = [
        [Fraction(0) for _ in range(shell)] for _ in range(shell)
    ]
    for left in range(shell):
        for right in range(left, shell):
            entry = sum(
                (row[left] * row[right] for row in factor_fraction),
                Fraction(0),
            )
            if residual_fraction is not None:
                entry += residual_fraction[left] * residual_fraction[right]
            if left == right:
                entry += shift
            matrix[left][right] = entry
            matrix[right][left] = entry
    return tuple(tuple(row) for row in matrix)


def _exact_shift_is_positive_definite(
    matrix: tuple[tuple[Fraction, ...], ...],
    lower_bound: Fraction,
) -> bool:
    """Certify ``matrix - lower_bound I`` by exact unpivoted LDL."""

    dimension = len(matrix)
    lower = [[Fraction(0) for _ in range(dimension)] for _ in range(dimension)]
    diagonal: list[Fraction] = []
    for row in range(dimension):
        pivot = matrix[row][row] - lower_bound
        pivot -= sum(
            (
                lower[row][prior] * lower[row][prior] * diagonal[prior]
                for prior in range(row)
            ),
            Fraction(0),
        )
        if pivot <= 0:
            return False
        diagonal.append(pivot)
        lower[row][row] = Fraction(1)
        for following in range(row + 1, dimension):
            numerator = matrix[following][row]
            numerator -= sum(
                (
                    lower[following][prior]
                    * lower[row][prior]
                    * diagonal[prior]
                    for prior in range(row)
                ),
                Fraction(0),
            )
            lower[following][row] = numerator / pivot
    return True


def certified_frobenius_norm_upper_bound(value: np.ndarray) -> float:
    """Return an exact-coordinate outward Frobenius/Euclidean norm bound."""

    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64) or array.ndim not in {1, 2}:
        raise TypeError("norm input must be one float64 vector or matrix")
    if not np.isfinite(array).all():
        raise ValueError("norm input must contain only finite values")
    squared = sum(
        (
            Fraction.from_float(float(item))
            * Fraction.from_float(float(item))
            for item in array.reshape(-1)
        ),
        Fraction(0),
    )
    if not squared:
        return 0.0
    squared_upper = _fraction_upper_float(squared)
    root = math.sqrt(squared_upper)
    if Fraction.from_float(root) * Fraction.from_float(root) < squared:
        root = math.nextafter(root, math.inf)
    if not math.isfinite(root) or (
        Fraction.from_float(root) * Fraction.from_float(root) < squared
    ):
        raise ValueError("certified norm is outside binary64")
    return root


def certified_gram_perturbation_norm_upper_bound(
    midpoint_factor_norm_upper: float,
    radius_factor_norm_upper: float,
) -> float:
    """Outward bound ``2 ||H|| ||E|| + ||E||^2`` from FP64 bounds."""

    values = (midpoint_factor_norm_upper, radius_factor_norm_upper)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in values
    ):
        raise ValueError("Gram perturbation norms must be finite and nonnegative")
    midpoint = Fraction.from_float(float(midpoint_factor_norm_upper))
    radius = Fraction.from_float(float(radius_factor_norm_upper))
    return _fraction_upper_float(2 * midpoint * radius + radius * radius)


def certified_gram_operator_min_eigenvalue_lower_bound(
    factor: np.ndarray,
    *,
    residual_direction: np.ndarray | None = None,
    diagonal_shift: float = 0.0,
    maximum_dimension: int = DEFAULT_MAXIMUM_CERTIFIED_DIMENSION,
) -> float:
    """Certify a lower eigenvalue bound for a bounded Gram-plus-residual shell.

    The mathematical operator is ``H.T H + vv.T + rho I`` over the exact real
    values represented by the supplied FP64 coordinates. A floating symmetric
    eigensolve proposes a useful value; exact rational LDL either certifies it
    or contracts it. Rank-deficient and unresolved cases safely return zero.
    """

    matrix_factor = _require_f64_matrix(factor, name="restricted factor")
    shell = int(matrix_factor.shape[1])
    if shell <= 0:
        raise ValueError("certified operator requires a nonempty shell")
    if isinstance(maximum_dimension, bool) or not isinstance(maximum_dimension, int):
        raise TypeError("maximum_dimension must be one integer")
    if maximum_dimension <= 0 or shell > maximum_dimension:
        raise ValueError("certified operator exceeds its registered shell dimension")
    residual = _require_optional_f64_vector(
        residual_direction,
        dimension=shell,
        name="residual direction",
    )
    if isinstance(diagonal_shift, bool) or not isinstance(
        diagonal_shift, (int, float)
    ):
        raise TypeError("diagonal_shift must be numeric")
    shift = float(diagonal_shift)
    if not math.isfinite(shift) or shift < 0.0:
        raise ValueError("diagonal_shift must be finite and nonnegative")

    exact = _exact_operator(
        matrix_factor,
        residual_direction=residual,
        diagonal_shift=shift,
    )
    if not _exact_shift_is_positive_definite(exact, Fraction(0)):
        return 0.0

    approximate = matrix_factor.T @ matrix_factor
    if residual is not None:
        approximate = approximate + np.outer(residual, residual)
    if shift:
        approximate = approximate + shift * np.eye(shell, dtype=np.float64)
    proposal = float(np.min(np.linalg.eigvalsh(approximate)))
    if not math.isfinite(proposal) or proposal <= 0.0:
        return 0.0

    candidate = math.nextafter(proposal, 0.0)
    for _ in range(64):
        if candidate <= 0.0:
            return 0.0
        if _exact_shift_is_positive_definite(
            exact,
            Fraction.from_float(candidate),
        ):
            return candidate
        candidate = math.nextafter(candidate * 0.5, 0.0)
    return 0.0


__all__ = [
    "DEFAULT_MAXIMUM_CERTIFIED_DIMENSION",
    "certified_frobenius_norm_upper_bound",
    "certified_gram_perturbation_norm_upper_bound",
    "certified_gram_operator_min_eigenvalue_lower_bound",
]
