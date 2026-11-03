"""Workflow-neutral regularized nuisance-quotient metric operators.

The operator is matrix-free when constructed from aligned route/nuisance
factors.  It forms only the low-dimensional nuisance Gram matrix and computes
route-space diagonals, submatrices, and products on demand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .metrics import validate_metric


IdentificationPolicy = Literal[
    "require_raw_identification",
    "allow_regularized_solve",
]


@dataclass(frozen=True, slots=True)
class PseudoinverseSpectrum:
    """Spectrum actually retained by ``numpy.linalg.pinv`` for a PSD system."""

    dimension: int
    effective_rank: int
    condition_number: float
    singular_cutoff: float
    maximum_singular_value: float

    @property
    def full_rank(self) -> bool:
        return self.effective_rank == self.dimension


def pseudoinverse_spectrum(
    matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    rcond: float,
) -> PseudoinverseSpectrum:
    """Return rank diagnostics using the exact relative cutoff used by ``pinv``.

    The input is a symmetric positive-semidefinite Gram matrix.  Its singular
    values therefore equal its nonnegative eigenvalues, and NumPy retains a
    singular direction exactly when it is strictly larger than
    ``rcond * maximum_singular_value``.
    """

    if not math.isfinite(float(rcond)) or float(rcond) < 0.0:
        raise ValueError("pseudoinverse rcond must be finite and nonnegative")
    system = validate_metric(matrix)
    eigenvalues = np.linalg.eigvalsh(system)
    maximum = float(np.max(eigenvalues))
    cutoff = float(rcond) * maximum
    retained = eigenvalues[eigenvalues > cutoff]
    rank = int(retained.size)
    condition = (
        float(maximum / float(np.min(retained)))
        if rank == system.shape[0] and rank > 0
        else float("inf")
    )
    return PseudoinverseSpectrum(
        dimension=int(system.shape[0]),
        effective_rank=rank,
        condition_number=condition,
        singular_cutoff=cutoff,
        maximum_singular_value=maximum,
    )


@dataclass(frozen=True)
class QuotientDiagnostics:
    """Raw-identification and regularized-solve diagnostics.

    ``effective_rank`` and ``condition_number`` retain their historical role
    as aliases for the system actually pseudoinverted.  Promotion/refusal is
    governed by the explicit raw and regularized fields plus
    ``identification_policy``.
    """

    nuisance_dimension: int
    effective_rank: int
    condition_number: float
    regularization: float
    refused: bool
    reason: str
    raw_effective_rank: int = 0
    raw_condition_number: float = float("inf")
    raw_singular_cutoff: float = 0.0
    regularized_effective_rank: int = 0
    regularized_condition_number: float = float("inf")
    regularized_singular_cutoff: float = 0.0
    raw_identified: bool = False
    regularized_solve_usable: bool = False
    identification_policy: str = "require_raw_identification"
    regularization_created_rank: bool = False

    def to_dict(self) -> dict[str, int | float | bool | str | None]:
        return {
            "nuisance_dimension": int(self.nuisance_dimension),
            "effective_rank": int(self.effective_rank),
            "condition_number": (
                float(self.condition_number)
                if math.isfinite(self.condition_number)
                else None
            ),
            "condition_number_is_finite": math.isfinite(self.condition_number),
            "regularization": float(self.regularization),
            "refused": bool(self.refused),
            "reason": str(self.reason),
            "raw_effective_rank": int(self.raw_effective_rank),
            "raw_condition_number": (
                float(self.raw_condition_number)
                if math.isfinite(self.raw_condition_number)
                else None
            ),
            "raw_condition_number_is_finite": math.isfinite(self.raw_condition_number),
            "raw_singular_cutoff": float(self.raw_singular_cutoff),
            "regularized_effective_rank": int(self.regularized_effective_rank),
            "regularized_condition_number": (
                float(self.regularized_condition_number)
                if math.isfinite(self.regularized_condition_number)
                else None
            ),
            "regularized_condition_number_is_finite": math.isfinite(
                self.regularized_condition_number
            ),
            "regularized_singular_cutoff": float(self.regularized_singular_cutoff),
            "raw_identified": bool(self.raw_identified),
            "regularized_solve_usable": bool(self.regularized_solve_usable),
            "identification_policy": str(self.identification_policy),
            "regularization_created_rank": bool(self.regularization_created_rank),
        }


class IntrinsicMetricOperator:
    """Schur-complement operator profiling permitted nuisance directions."""

    def __init__(
        self,
        g_gg: Sequence[Sequence[float]] | np.ndarray,
        g_gtheta: Sequence[Sequence[float]] | np.ndarray,
        g_thetatheta: Sequence[Sequence[float]] | np.ndarray,
        *,
        regularization: float = 1e-8,
        rcond: float = 1e-10,
        maximum_condition_number: float = 1e12,
        maximum_dense_routes: int = 8192,
        identification_policy: IdentificationPolicy = "require_raw_identification",
    ) -> None:
        fixed = validate_metric(g_gg)
        cross = np.asarray(g_gtheta, dtype=np.float64)
        nuisance_gram = validate_metric(g_thetatheta)
        if cross.shape != (fixed.shape[0], nuisance_gram.shape[0]):
            raise ValueError("route-nuisance block has an incompatible shape")
        if not np.isfinite(cross).all():
            raise ValueError("route-nuisance block must be finite")
        self._route_factor: np.ndarray | None = None
        self._nuisance_factor: np.ndarray | None = None
        self._sample_count: int | None = None
        self._g_gg: np.ndarray | None = fixed
        self._g_gtheta: np.ndarray | None = cross
        self._g_thetatheta = nuisance_gram
        self._route_dimension = int(fixed.shape[0])
        self.maximum_dense_routes = int(maximum_dense_routes)
        self._initialize_nuisance(
            regularization=regularization,
            rcond=rcond,
            maximum_condition_number=maximum_condition_number,
            identification_policy=identification_policy,
        )
        # Dense-block construction is intended for bounded tests/screened
        # regions, so validate its Schur complement immediately.
        intrinsic = self._dense_from_blocks()
        minimum = float(np.min(np.linalg.eigvalsh(intrinsic)))
        if minimum < -1e-7:
            raise ValueError(
                "Schur complement is materially non-PSD; joint metric blocks are inconsistent"
            )

    def _initialize_nuisance(
        self,
        *,
        regularization: float,
        rcond: float,
        maximum_condition_number: float,
        identification_policy: IdentificationPolicy,
    ) -> None:
        self.regularization = float(regularization)
        self.rcond = float(rcond)
        self.maximum_condition_number = float(maximum_condition_number)
        self.identification_policy = str(identification_policy)
        if (
            not math.isfinite(self.regularization)
            or not math.isfinite(self.rcond)
            or not math.isfinite(self.maximum_condition_number)
            or self.regularization < 0.0
            or self.rcond < 0.0
            or self.maximum_condition_number <= 0.0
        ):
            raise ValueError("quotient regularization parameters are invalid")
        if self.identification_policy not in {
            "require_raw_identification",
            "allow_regularized_solve",
        }:
            raise ValueError("quotient identification_policy is invalid")
        if self.maximum_dense_routes <= 0:
            raise ValueError("maximum_dense_routes must be positive")
        nuisance = self._g_thetatheta + self.regularization * np.eye(
            self._g_thetatheta.shape[0]
        )
        raw = pseudoinverse_spectrum(self._g_thetatheta, rcond=self.rcond)
        regularized = pseudoinverse_spectrum(nuisance, rcond=self.rcond)
        raw_identified = (
            raw.full_rank
            and math.isfinite(raw.condition_number)
            and raw.condition_number <= self.maximum_condition_number
        )
        regularized_usable = (
            regularized.full_rank
            and math.isfinite(regularized.condition_number)
            and regularized.condition_number <= self.maximum_condition_number
        )
        refused = (
            not raw_identified
            if self.identification_policy == "require_raw_identification"
            else not regularized_usable
        )
        if raw_identified:
            reason = "quotient operator is identified"
        elif self.identification_policy == "require_raw_identification":
            reason = (
                "raw nuisance block is rank-deficient at the pseudoinverse cutoff"
                if not raw.full_rank
                else "raw nuisance block exceeds maximum condition number"
            )
        elif regularized_usable:
            reason = (
                "regularized solve is usable under explicit policy; raw nuisance "
                "identification is not established"
            )
        else:
            reason = (
                "regularized nuisance block is rank-deficient at the pseudoinverse cutoff"
                if not regularized.full_rank
                else "regularized nuisance block exceeds maximum condition number"
            )
        self.diagnostics = QuotientDiagnostics(
            nuisance_dimension=self._g_thetatheta.shape[0],
            effective_rank=regularized.effective_rank,
            condition_number=regularized.condition_number,
            regularization=self.regularization,
            refused=refused,
            reason=reason,
            raw_effective_rank=raw.effective_rank,
            raw_condition_number=raw.condition_number,
            raw_singular_cutoff=raw.singular_cutoff,
            regularized_effective_rank=regularized.effective_rank,
            regularized_condition_number=regularized.condition_number,
            regularized_singular_cutoff=regularized.singular_cutoff,
            raw_identified=raw_identified,
            regularized_solve_usable=regularized_usable,
            identification_policy=self.identification_policy,
            regularization_created_rank=(
                regularized.effective_rank > raw.effective_rank
            ),
        )
        self._nuisance_inverse = np.linalg.pinv(nuisance, rcond=self.rcond)

    @classmethod
    def from_factors(
        cls,
        route_factor: Sequence[Sequence[float]] | np.ndarray,
        nuisance_factor: Sequence[Sequence[float]] | np.ndarray,
        *,
        regularization: float = 1e-8,
        rcond: float = 1e-10,
        maximum_condition_number: float = 1e12,
        maximum_dense_routes: int = 8192,
        identification_policy: IdentificationPolicy = "require_raw_identification",
    ) -> "IntrinsicMetricOperator":
        route = np.asarray(route_factor, dtype=np.float64)
        nuisance = np.asarray(nuisance_factor, dtype=np.float64)
        if (
            route.ndim != 2
            or nuisance.ndim != 2
            or route.shape[0] != nuisance.shape[0]
            or route.shape[0] == 0
        ):
            raise ValueError(
                "route and nuisance factors must be aligned nonempty matrices"
            )
        if route.shape[1] == 0 or nuisance.shape[1] == 0:
            raise ValueError(
                "route and nuisance factors must have nonempty feature axes"
            )
        if not np.isfinite(route).all() or not np.isfinite(nuisance).all():
            raise ValueError("route and nuisance factors must be finite")
        self = cls.__new__(cls)
        self._route_factor = route
        self._nuisance_factor = nuisance
        self._sample_count = int(route.shape[0])
        self._g_gg = None
        self._g_gtheta = None
        self._g_thetatheta = validate_metric(nuisance.T @ nuisance / self._sample_count)
        self._route_dimension = int(route.shape[1])
        self.maximum_dense_routes = int(maximum_dense_routes)
        self._initialize_nuisance(
            regularization=regularization,
            rcond=rcond,
            maximum_condition_number=maximum_condition_number,
            identification_policy=identification_policy,
        )
        return self

    @property
    def shape(self) -> tuple[int, int]:
        return (self._route_dimension, self._route_dimension)

    @property
    def factor_backed(self) -> bool:
        return self._route_factor is not None

    @property
    def g_thetatheta(self) -> np.ndarray:
        return self._g_thetatheta.copy()

    @property
    def g_gg(self) -> np.ndarray:
        if self._g_gg is not None:
            return self._g_gg.copy()
        return self.fixed_submatrix(np.arange(self._route_dimension, dtype=np.int64))

    @property
    def g_gtheta(self) -> np.ndarray:
        if self._g_gtheta is not None:
            return self._g_gtheta.copy()
        assert (
            self._route_factor is not None
            and self._nuisance_factor is not None
            and self._sample_count is not None
        )
        return self._route_factor.T @ self._nuisance_factor / self._sample_count

    def _check_ready(self) -> None:
        if self.diagnostics.refused:
            raise RuntimeError(self.diagnostics.reason)

    def _validate_indices(self, indices: Sequence[int] | np.ndarray) -> np.ndarray:
        selected = np.asarray(indices, dtype=np.int64).reshape(-1)
        if (
            np.any(selected < 0)
            or np.any(selected >= self._route_dimension)
            or np.unique(selected).size != selected.size
        ):
            raise ValueError("route indices are invalid")
        return selected

    def _dense_from_blocks(self) -> np.ndarray:
        assert self._g_gg is not None and self._g_gtheta is not None
        intrinsic = (
            self._g_gg - self._g_gtheta @ self._nuisance_inverse @ self._g_gtheta.T
        )
        intrinsic = 0.5 * (intrinsic + intrinsic.T)
        minimum = float(np.min(np.linalg.eigvalsh(intrinsic)))
        if minimum < 0.0 and minimum >= -1e-7:
            intrinsic = intrinsic + (-minimum + 1e-12) * np.eye(intrinsic.shape[0])
        return intrinsic

    def fixed_submatrix(self, indices: Sequence[int] | np.ndarray) -> np.ndarray:
        selected = self._validate_indices(indices)
        if selected.size == 0:
            return np.zeros((0, 0), dtype=np.float64)
        if self._g_gg is not None:
            return self._g_gg[np.ix_(selected, selected)].copy()
        assert self._route_factor is not None and self._sample_count is not None
        factor = self._route_factor[:, selected]
        return validate_metric(factor.T @ factor / self._sample_count)

    def cross_for_indices(self, indices: Sequence[int] | np.ndarray) -> np.ndarray:
        selected = self._validate_indices(indices)
        if selected.size == 0:
            return np.zeros((0, self._g_thetatheta.shape[0]), dtype=np.float64)
        if self._g_gtheta is not None:
            return self._g_gtheta[selected].copy()
        assert (
            self._route_factor is not None
            and self._nuisance_factor is not None
            and self._sample_count is not None
        )
        return (
            self._route_factor[:, selected].T
            @ self._nuisance_factor
            / self._sample_count
        )

    def submatrix(self, indices: Sequence[int] | np.ndarray) -> np.ndarray:
        self._check_ready()
        selected = self._validate_indices(indices)
        fixed = self.fixed_submatrix(selected)
        cross = self.cross_for_indices(selected)
        intrinsic = fixed - cross @ self._nuisance_inverse @ cross.T
        intrinsic = 0.5 * (intrinsic + intrinsic.T)
        minimum = (
            float(np.min(np.linalg.eigvalsh(intrinsic))) if intrinsic.size else 0.0
        )
        if minimum < -1e-7:
            raise ValueError("screened Schur complement is materially non-PSD")
        if minimum < 0.0:
            intrinsic = intrinsic + (-minimum + 1e-12) * np.eye(intrinsic.shape[0])
        return intrinsic

    def diagonal(self, *, chunk_size: int = 16384) -> np.ndarray:
        self._check_ready()
        if self._g_gg is not None and self._g_gtheta is not None:
            correction = np.einsum(
                "ij,jk,ik->i", self._g_gtheta, self._nuisance_inverse, self._g_gtheta
            )
            return np.maximum(np.diag(self._g_gg) - correction, 0.0)
        assert (
            self._route_factor is not None
            and self._nuisance_factor is not None
            and self._sample_count is not None
        )
        size = self._route_dimension
        result = np.empty(size, dtype=np.float64)
        step = max(1, int(chunk_size))
        for start in range(0, size, step):
            stop = min(size, start + step)
            route = self._route_factor[:, start:stop]
            fixed = np.mean(route * route, axis=0)
            cross = route.T @ self._nuisance_factor / self._sample_count
            correction = np.einsum("ij,jk,ik->i", cross, self._nuisance_inverse, cross)
            result[start:stop] = np.maximum(fixed - correction, 0.0)
        return result

    def dense(
        self,
        *,
        refuse_ill_conditioned: bool = True,
        maximum_routes: int | None = None,
    ) -> np.ndarray:
        if refuse_ill_conditioned:
            self._check_ready()
        limit = (
            self.maximum_dense_routes if maximum_routes is None else int(maximum_routes)
        )
        if self._route_dimension > limit:
            raise RuntimeError(
                f"refusing dense {self._route_dimension}x{self._route_dimension} quotient metric; "
                "use diagonal(), submatrix(), or matvec()"
            )
        if self._g_gg is not None:
            return self._dense_from_blocks().copy()
        return self.submatrix(np.arange(self._route_dimension, dtype=np.int64))

    def matvec(self, vector: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=np.float64).reshape(-1)
        if values.size != self.shape[1]:
            raise ValueError("intrinsic metric vector has incompatible dimension")
        self._check_ready()
        if self._g_gg is not None and self._g_gtheta is not None:
            return self._g_gg @ values - self._g_gtheta @ (
                self._nuisance_inverse @ (self._g_gtheta.T @ values)
            )
        assert (
            self._route_factor is not None
            and self._nuisance_factor is not None
            and self._sample_count is not None
        )
        response = self._route_factor @ values
        nuisance_score = self._nuisance_factor.T @ response / self._sample_count
        nuisance_shift = self._nuisance_inverse @ nuisance_score
        residual = response - self._nuisance_factor @ nuisance_shift
        return self._route_factor.T @ residual / self._sample_count

    def fixed_damage(self, vector: Sequence[float] | np.ndarray) -> float:
        values = np.asarray(vector, dtype=np.float64).reshape(-1)
        if values.size != self.shape[1]:
            raise ValueError("route action has incompatible dimension")
        if self._g_gg is not None:
            return float(values @ self._g_gg @ values)
        assert self._route_factor is not None
        response = self._route_factor @ values
        return float(np.mean(response * response))

    def intrinsic_damage(self, vector: Sequence[float] | np.ndarray) -> float:
        values = np.asarray(vector, dtype=np.float64).reshape(-1)
        return float(values @ self.matvec(values))

    def _sparse_response(
        self,
        indices: Sequence[int] | np.ndarray,
        values: Sequence[float] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        selected = self._validate_indices(indices)
        coefficients = (
            -np.ones(selected.size, dtype=np.float64)
            if values is None
            else np.asarray(values, dtype=np.float64).reshape(-1)
        )
        if coefficients.size != selected.size or not np.isfinite(coefficients).all():
            raise ValueError("sparse route coefficients are invalid")
        if self._route_factor is not None:
            return selected, self._route_factor[:, selected] @ coefficients
        dense = np.zeros(self._route_dimension, dtype=np.float64)
        dense[selected] = coefficients
        return selected, dense

    def fixed_damage_sparse(
        self,
        indices: Sequence[int] | np.ndarray,
        values: Sequence[float] | np.ndarray | None = None,
    ) -> float:
        selected, response = self._sparse_response(indices, values)
        if self._route_factor is not None:
            return float(np.mean(response * response))
        return self.fixed_damage(response)

    def intrinsic_damage_sparse(
        self,
        indices: Sequence[int] | np.ndarray,
        values: Sequence[float] | np.ndarray | None = None,
    ) -> float:
        self._check_ready()
        selected, response = self._sparse_response(indices, values)
        if selected.size == 0:
            return 0.0
        if self._route_factor is None:
            return self.intrinsic_damage(response)
        assert self._nuisance_factor is not None and self._sample_count is not None
        fixed = float(np.mean(response * response))
        score = self._nuisance_factor.T @ response / self._sample_count
        correction = float(score @ self._nuisance_inverse @ score)
        return max(0.0, fixed - correction)

    def absorbable_damage(self, vector: Sequence[float] | np.ndarray) -> float:
        return max(0.0, self.fixed_damage(vector) - self.intrinsic_damage(vector))

    def absorbable_damage_sparse(
        self,
        indices: Sequence[int] | np.ndarray,
        values: Sequence[float] | np.ndarray | None = None,
    ) -> float:
        return max(
            0.0,
            self.fixed_damage_sparse(indices, values)
            - self.intrinsic_damage_sparse(indices, values),
        )

    def optimal_nuisance_shift(
        self, vector: Sequence[float] | np.ndarray
    ) -> np.ndarray:
        values = np.asarray(vector, dtype=np.float64).reshape(-1)
        if values.size != self.shape[1]:
            raise ValueError("route action has incompatible dimension")
        self._check_ready()
        if self._g_gtheta is not None:
            return -self._nuisance_inverse @ self._g_gtheta.T @ values
        assert (
            self._route_factor is not None
            and self._nuisance_factor is not None
            and self._sample_count is not None
        )
        score = (
            self._nuisance_factor.T @ (self._route_factor @ values) / self._sample_count
        )
        return -self._nuisance_inverse @ score

    def optimal_nuisance_shift_sparse(
        self,
        indices: Sequence[int] | np.ndarray,
        values: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        self._check_ready()
        selected, response = self._sparse_response(indices, values)
        if selected.size == 0:
            return np.zeros(self._g_thetatheta.shape[0], dtype=np.float64)
        if self._route_factor is None:
            return self.optimal_nuisance_shift(response)
        assert self._nuisance_factor is not None and self._sample_count is not None
        score = self._nuisance_factor.T @ response / self._sample_count
        return -self._nuisance_inverse @ score


__all__ = [
    "IdentificationPolicy",
    "IntrinsicMetricOperator",
    "PseudoinverseSpectrum",
    "QuotientDiagnostics",
    "pseudoinverse_spectrum",
]
