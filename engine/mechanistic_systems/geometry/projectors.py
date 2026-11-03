"""Basis-invariant stable spectral projectors and transport diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.geometry.metrics import validate_metric


@dataclass(frozen=True)
class StableProjector:
    cluster_id: str
    eigenvalue_min: float
    eigenvalue_max: float
    multiplicity: int
    external_gap: float
    stable: bool
    basis: np.ndarray
    projector_hash: str

    def __post_init__(self) -> None:
        basis = np.asarray(self.basis, dtype=np.float64)
        if basis.ndim != 2 or basis.shape[1] != int(self.multiplicity):
            raise ValueError("stable projector basis has an invalid shape")
        if not np.allclose(
            basis.T @ basis, np.eye(basis.shape[1]), atol=1e-8, rtol=1e-8
        ):
            raise ValueError("stable projector basis must be orthonormal")
        object.__setattr__(self, "basis", basis.copy())
        expected = content_hash(self.projector, prefix="irg-projector-")
        if self.projector_hash != expected:
            raise ValueError("stable projector hash mismatch")

    @property
    def projector(self) -> np.ndarray:
        return self.basis @ self.basis.T

    @property
    def participation(self) -> np.ndarray:
        return np.diag(self.projector).copy()


@dataclass(frozen=True)
class ProjectorMatch:
    source_cluster_id: str
    target_cluster_id: str
    maximum_principal_angle: float
    mean_squared_overlap: float
    accepted: bool


def _cluster_slices(
    eigenvalues_descending: np.ndarray,
    *,
    relative_gap: float,
    absolute_gap: float,
) -> list[slice]:
    starts = [0]
    for index in range(len(eigenvalues_descending) - 1):
        left = float(eigenvalues_descending[index])
        right = float(eigenvalues_descending[index + 1])
        gap = left - right
        scale = max(abs(left), abs(right), 1.0)
        if gap > max(absolute_gap, relative_gap * scale):
            starts.append(index + 1)
    starts.append(len(eigenvalues_descending))
    return [slice(left, right) for left, right in zip(starts, starts[1:])]


def stable_spectral_projectors(
    metric: Sequence[Sequence[float]] | np.ndarray,
    *,
    relative_gap: float = 1e-3,
    absolute_gap: float = 1e-8,
    noise_floor: float = 1e-6,
    minimum_eigenvalue: float = 1e-12,
) -> tuple[StableProjector, ...]:
    matrix = validate_metric(metric)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues > float(minimum_eigenvalue)
    eigenvalues = eigenvalues[positive]
    eigenvectors = eigenvectors[:, positive]
    if eigenvalues.size == 0:
        return ()
    clusters: list[StableProjector] = []
    slices = _cluster_slices(
        eigenvalues,
        relative_gap=float(relative_gap),
        absolute_gap=float(absolute_gap),
    )
    for cluster_index, cluster_slice in enumerate(slices):
        values = eigenvalues[cluster_slice]
        basis = eigenvectors[:, cluster_slice]
        left_index = cluster_slice.start
        right_index = cluster_slice.stop
        gaps: list[float] = []
        if left_index > 0:
            gaps.append(float(eigenvalues[left_index - 1] - eigenvalues[left_index]))
        if right_index < len(eigenvalues):
            gaps.append(float(eigenvalues[right_index - 1] - eigenvalues[right_index]))
        external_gap = (
            min(gaps) if gaps else float(max(np.max(np.abs(eigenvalues)), 1.0))
        )
        projector = basis @ basis.T
        clusters.append(
            StableProjector(
                cluster_id=f"cluster-{cluster_index:03d}",
                eigenvalue_min=float(np.min(values)),
                eigenvalue_max=float(np.max(values)),
                multiplicity=int(len(values)),
                external_gap=external_gap,
                stable=bool(external_gap >= float(noise_floor)),
                basis=basis,
                projector_hash=content_hash(projector, prefix="irg-projector-"),
            )
        )
    return tuple(clusters)


def principal_angles(left_basis: np.ndarray, right_basis: np.ndarray) -> np.ndarray:
    left = np.asarray(left_basis, dtype=np.float64)
    right = np.asarray(right_basis, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("principal-angle bases must share ambient dimension")
    singular = np.linalg.svd(left.T @ right, compute_uv=False)
    singular = np.clip(singular, 0.0, 1.0)
    return np.arccos(singular)


def match_projectors(
    source: Sequence[StableProjector],
    target: Sequence[StableProjector],
    *,
    maximum_angle: float = 0.35,
) -> tuple[ProjectorMatch, ...]:
    remaining = set(range(len(target)))
    matches: list[ProjectorMatch] = []
    for source_projector in source:
        candidates: list[tuple[float, float, int]] = []
        for index in remaining:
            target_projector = target[index]
            angles = principal_angles(source_projector.basis, target_projector.basis)
            max_angle_value = float(np.max(angles)) if angles.size else math.pi / 2.0
            overlap = float(
                np.trace(source_projector.projector @ target_projector.projector)
                / max(source_projector.multiplicity, target_projector.multiplicity, 1)
            )
            candidates.append((max_angle_value, -overlap, index))
        if not candidates:
            continue
        angle, negative_overlap, index = min(candidates)
        remaining.remove(index)
        target_projector = target[index]
        matches.append(
            ProjectorMatch(
                source_cluster_id=source_projector.cluster_id,
                target_cluster_id=target_projector.cluster_id,
                maximum_principal_angle=angle,
                mean_squared_overlap=-negative_overlap,
                accepted=bool(
                    source_projector.stable
                    and target_projector.stable
                    and source_projector.multiplicity == target_projector.multiplicity
                    and angle <= float(maximum_angle)
                ),
            )
        )
    return tuple(matches)


def residualize_route_direction(
    factor: Sequence[Sequence[float]] | np.ndarray,
    direction: Sequence[float] | np.ndarray,
) -> np.ndarray:
    rows = np.asarray(factor, dtype=np.float64)
    vector = np.asarray(direction, dtype=np.float64).reshape(-1)
    if rows.ndim != 2 or rows.shape[1] != vector.size or rows.shape[0] == 0:
        raise ValueError("factor and route direction have incompatible shapes")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-15:
        raise ValueError("route direction must be nonzero")
    unit = vector / norm
    return rows - np.outer(rows @ unit, unit)


__all__ = [
    "ProjectorMatch",
    "StableProjector",
    "match_projectors",
    "principal_angles",
    "residualize_route_direction",
    "stable_spectral_projectors",
]
