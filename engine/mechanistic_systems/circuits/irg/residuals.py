"""Multi-fidelity correction and measured local-model validity."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .contracts import ModelValidityDecision, content_hash


@dataclass(frozen=True)
class ResidualObservation:
    candidate_id: str
    state_hash: str
    predicted_quality_loss: float
    actual_quality_loss: float
    features: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.state_hash:
            raise ValueError(
                "residual observation requires candidate_id and state_hash"
            )
        for name in ("predicted_quality_loss", "actual_quality_loss"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        normalized = {str(key): float(value) for key, value in self.features.items()}
        if any(not math.isfinite(value) for value in normalized.values()):
            raise ValueError("residual features must be finite")
        object.__setattr__(self, "features", normalized)

    @property
    def residual(self) -> float:
        return float(self.actual_quality_loss - self.predicted_quality_loss)


@dataclass(frozen=True)
class ResidualPrediction:
    corrected_quality_loss: float
    standard_error: float
    predicted_residual: float
    model_id: str


@dataclass(frozen=True)
class PairedMultiFidelityEstimate:
    corrected_mean: float
    standard_error: float
    residual_mean: float
    residual_variance: float
    paired_count: int


@dataclass(frozen=True)
class PredictiveValidityReceipt:
    """Partition-local evidence for deciding whether one response chart survives.

    Exact-mask oracle values remain the commit authority elsewhere.  This
    receipt has the narrower, guidance-only role of comparing the chart's
    pre-registered predictions with exact action means for candidates born
    under one parent state and one geometric partition.
    """

    parent_state_hash: str
    parent_mask_hash: str
    partition_id: str
    selected_candidate_id: str
    candidate_ids: tuple[str, ...]
    predicted_changes: tuple[float, ...]
    actual_changes: tuple[float, ...]
    actual_standard_errors: tuple[float, ...]
    oracle_receipt_hashes: tuple[str, ...]
    decision: ModelValidityDecision
    schema: str = "irg_predictive_validity_receipt_v1"

    def __post_init__(self) -> None:
        identities = (
            self.parent_state_hash,
            self.parent_mask_hash,
            self.partition_id,
            self.selected_candidate_id,
            self.schema,
        )
        if any(not value for value in identities):
            raise ValueError("predictive validity receipt requires complete identity")
        candidate_ids = tuple(str(value) for value in self.candidate_ids)
        predictions = tuple(float(value) for value in self.predicted_changes)
        actual = tuple(float(value) for value in self.actual_changes)
        standard_errors = tuple(float(value) for value in self.actual_standard_errors)
        receipt_hashes = tuple(str(value) for value in self.oracle_receipt_hashes)
        size = len(candidate_ids)
        if (
            size == 0
            or len(set(candidate_ids)) != size
            or self.selected_candidate_id not in candidate_ids
            or len(predictions) != size
            or len(actual) != size
            or len(standard_errors) != size
            or len(receipt_hashes) != size
        ):
            raise ValueError("predictive validity evidence axes are not aligned")
        if (
            any(not math.isfinite(value) for value in predictions)
            or any(not math.isfinite(value) for value in actual)
            or any(not math.isfinite(value) or value < 0.0 for value in standard_errors)
            or any(not value for value in receipt_hashes)
        ):
            raise ValueError("predictive validity receipt contains invalid evidence")
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "predicted_changes", predictions)
        object.__setattr__(self, "actual_changes", actual)
        object.__setattr__(self, "actual_standard_errors", standard_errors)
        object.__setattr__(self, "oracle_receipt_hashes", receipt_hashes)

    @property
    def content_hash(self) -> str:
        return content_hash(
            {
                **asdict(self),
                "decision": asdict(self.decision),
            },
            prefix="irg-predictive-validity-",
        )


def paired_control_variate_estimate(
    low_fidelity_population_mean: float,
    low_fidelity_paired: Sequence[float] | np.ndarray,
    high_fidelity_paired: Sequence[float] | np.ndarray,
) -> PairedMultiFidelityEstimate:
    low = np.asarray(low_fidelity_paired, dtype=np.float64).reshape(-1)
    high = np.asarray(high_fidelity_paired, dtype=np.float64).reshape(-1)
    if (
        low.shape != high.shape
        or low.size == 0
        or not np.isfinite(low).all()
        or not np.isfinite(high).all()
    ):
        raise ValueError(
            "control-variate estimate requires aligned finite paired values"
        )
    residual = high - low
    variance = float(np.var(residual, ddof=1)) if residual.size >= 2 else 0.0
    return PairedMultiFidelityEstimate(
        corrected_mean=float(low_fidelity_population_mean + np.mean(residual)),
        standard_error=math.sqrt(variance / residual.size),
        residual_mean=float(np.mean(residual)),
        residual_variance=variance,
        paired_count=int(residual.size),
    )


class RobustResidualModel:
    """Small Huber-IRLS model for exact-minus-geometric action residuals."""

    def __init__(
        self,
        *,
        ridge: float = 1e-8,
        huber_delta: float = 1.5,
        maximum_iterations: int = 50,
    ) -> None:
        self.ridge = float(ridge)
        self.huber_delta = float(huber_delta)
        self.maximum_iterations = int(maximum_iterations)
        if self.ridge < 0.0 or self.huber_delta <= 0.0 or self.maximum_iterations <= 0:
            raise ValueError("residual model hyperparameters are invalid")
        self._observations: list[ResidualObservation] = []
        self._feature_names: tuple[str, ...] = ()
        self._coefficients: np.ndarray | None = None
        self._covariance: np.ndarray | None = None
        self._residual_scale: float = float("inf")

    @property
    def observation_count(self) -> int:
        return len(self._observations)

    @property
    def fitted(self) -> bool:
        return self._coefficients is not None

    @property
    def model_id(self) -> str:
        names = ",".join(self._feature_names)
        return f"huber_residual(intercept,predicted,{names})"

    def add(self, observation: ResidualObservation) -> None:
        self._observations.append(observation)
        self._fit()

    def extend(self, observations: Sequence[ResidualObservation]) -> None:
        self._observations.extend(observations)
        self._fit()

    def _design_row(
        self, observation: ResidualObservation, names: tuple[str, ...]
    ) -> list[float]:
        return [
            1.0,
            float(observation.predicted_quality_loss),
            *[float(observation.features.get(name, 0.0)) for name in names],
        ]

    def _fit(self) -> None:
        if not self._observations:
            self._coefficients = None
            self._covariance = None
            self._residual_scale = float("inf")
            return
        names = tuple(
            sorted({name for item in self._observations for name in item.features})
        )
        x = np.asarray(
            [self._design_row(item, names) for item in self._observations],
            dtype=np.float64,
        )
        y = np.asarray([item.residual for item in self._observations], dtype=np.float64)
        coefficients = np.linalg.solve(
            x.T @ x + self.ridge * np.eye(x.shape[1]),
            x.T @ y,
        )
        weights = np.ones_like(y)
        for _ in range(self.maximum_iterations):
            residuals = y - x @ coefficients
            median = float(np.median(residuals))
            mad = float(np.median(np.abs(residuals - median)))
            scale = max(
                1.4826 * mad, float(np.sqrt(np.mean(residuals * residuals))), 1e-12
            )
            standardized = np.abs(residuals) / (self.huber_delta * scale)
            new_weights = np.ones_like(standardized)
            np.divide(
                1.0,
                standardized,
                out=new_weights,
                where=standardized > 1.0,
            )
            weighted_x = x * np.sqrt(new_weights)[:, None]
            weighted_y = y * np.sqrt(new_weights)
            updated = np.linalg.solve(
                weighted_x.T @ weighted_x + self.ridge * np.eye(x.shape[1]),
                weighted_x.T @ weighted_y,
            )
            if np.linalg.norm(updated - coefficients) <= 1e-10 * (
                1.0 + np.linalg.norm(coefficients)
            ):
                coefficients = updated
                weights = new_weights
                break
            coefficients = updated
            weights = new_weights
        residuals = y - x @ coefficients
        dof = max(1, len(y) - x.shape[1])
        variance = float(np.sum(weights * residuals * residuals) / dof)
        gram = (x * np.sqrt(weights)[:, None]).T @ (x * np.sqrt(weights)[:, None])
        covariance = variance * np.linalg.pinv(gram + self.ridge * np.eye(x.shape[1]))
        self._feature_names = names
        self._coefficients = coefficients
        self._covariance = covariance
        self._residual_scale = math.sqrt(max(variance, 0.0))

    def predict(
        self,
        predicted_quality_loss: float,
        *,
        features: Mapping[str, float] | None = None,
    ) -> ResidualPrediction:
        prediction = float(predicted_quality_loss)
        if not math.isfinite(prediction):
            raise ValueError("predicted_quality_loss must be finite")
        if not self.fitted:
            return ResidualPrediction(
                corrected_quality_loss=prediction,
                standard_error=float("inf"),
                predicted_residual=0.0,
                model_id="unfitted_residual_identity",
            )
        row = np.asarray(
            [
                1.0,
                prediction,
                *[
                    float((features or {}).get(name, 0.0))
                    for name in self._feature_names
                ],
            ],
            dtype=np.float64,
        )
        residual = float(row @ self._coefficients)
        parameter_variance = float(row @ self._covariance @ row)
        standard_error = math.sqrt(
            max(parameter_variance + self._residual_scale**2, 0.0)
        )
        return ResidualPrediction(
            corrected_quality_loss=prediction + residual,
            standard_error=standard_error,
            predicted_residual=residual,
            model_id=self.model_id,
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "observation_count": self.observation_count,
            "fitted": self.fitted,
            "feature_names": list(self._feature_names),
            "coefficients": (
                None if self._coefficients is None else self._coefficients.tolist()
            ),
            "residual_scale": self._residual_scale,
            "model_id": self.model_id if self.fitted else "unfitted_residual_identity",
        }


class ModelValidityController:
    """Choose scale/refresh from measured actual-versus-predicted remainder."""

    def __init__(
        self,
        *,
        maximum_relative_remainder: float = 0.5,
        maximum_absolute_remainder: float = 0.05,
        minimum_observations: int = 2,
        shrink_factor: float = 0.5,
    ) -> None:
        self.maximum_relative_remainder = float(maximum_relative_remainder)
        self.maximum_absolute_remainder = float(maximum_absolute_remainder)
        self.minimum_observations = int(minimum_observations)
        self.shrink_factor = float(shrink_factor)
        if (
            self.maximum_relative_remainder < 0.0
            or self.maximum_absolute_remainder < 0.0
            or self.minimum_observations <= 0
            or not 0.0 < self.shrink_factor < 1.0
        ):
            raise ValueError("model validity thresholds are invalid")

    def assess(
        self,
        predicted_changes: Sequence[float] | np.ndarray,
        actual_changes: Sequence[float] | np.ndarray | None,
        *,
        actual_standard_errors: Sequence[float] | np.ndarray | None = None,
        uncertainty_z_value: float = 2.0,
    ) -> ModelValidityDecision:
        predicted = np.asarray(predicted_changes, dtype=np.float64).reshape(-1)
        if predicted.size == 0 or not np.isfinite(predicted).all():
            raise ValueError("model validity requires finite predicted changes")
        predicted_rms = float(np.sqrt(np.mean(predicted * predicted)))
        if actual_changes is None:
            return ModelValidityDecision(
                action="shrink",
                reason="no exact action outcome is available to validate the local model",
                recommended_scale=self.shrink_factor,
                predicted_rms=predicted_rms,
                actual_rms=None,
                remainder_rms=None,
            )
        actual = np.asarray(actual_changes, dtype=np.float64).reshape(-1)
        if actual.shape != predicted.shape or not np.isfinite(actual).all():
            raise ValueError(
                "actual changes must be finite and aligned with predictions"
            )
        if actual_standard_errors is None:
            standard_errors = np.zeros_like(actual)
        else:
            standard_errors = np.asarray(
                actual_standard_errors, dtype=np.float64
            ).reshape(-1)
            if (
                standard_errors.shape != predicted.shape
                or not np.isfinite(standard_errors).all()
                or np.any(standard_errors < 0.0)
            ):
                raise ValueError(
                    "actual standard errors must be finite, nonnegative, and aligned"
                )
        z_value = float(uncertainty_z_value)
        if not math.isfinite(z_value) or z_value < 0.0:
            raise ValueError("uncertainty_z_value must be finite and nonnegative")
        actual_rms = float(np.sqrt(np.mean(actual * actual)))
        remainder = actual - predicted
        remainder_rms = float(np.sqrt(np.mean(remainder * remainder)))
        conservative_remainder = np.abs(remainder) + z_value * standard_errors
        conservative_remainder_rms = float(
            np.sqrt(np.mean(conservative_remainder * conservative_remainder))
        )
        relative = conservative_remainder_rms / max(predicted_rms, 1e-12)
        diagnostics = {
            "relative_remainder": relative,
            "point_remainder_rms": remainder_rms,
            "conservative_remainder_rms": conservative_remainder_rms,
            "maximum_actual_standard_error": float(
                np.max(standard_errors, initial=0.0)
            ),
            "uncertainty_z_value": z_value,
            "observation_count": int(predicted.size),
            "maximum_relative_remainder": self.maximum_relative_remainder,
            "maximum_absolute_remainder": self.maximum_absolute_remainder,
        }
        if predicted.size < self.minimum_observations:
            return ModelValidityDecision(
                action="shrink",
                reason="insufficient paired exact outcomes for model reuse",
                recommended_scale=self.shrink_factor,
                predicted_rms=predicted_rms,
                actual_rms=actual_rms,
                remainder_rms=remainder_rms,
                diagnostics=diagnostics,
            )
        if (
            conservative_remainder_rms > self.maximum_absolute_remainder
            and relative > self.maximum_relative_remainder
        ):
            return ModelValidityDecision(
                action="refresh",
                reason="measured nonlinear remainder exceeds both absolute and relative limits",
                recommended_scale=self.shrink_factor,
                predicted_rms=predicted_rms,
                actual_rms=actual_rms,
                remainder_rms=conservative_remainder_rms,
                diagnostics=diagnostics,
            )
        if relative > self.maximum_relative_remainder:
            return ModelValidityDecision(
                action="shrink",
                reason="relative local-model remainder is too large",
                recommended_scale=self.shrink_factor,
                predicted_rms=predicted_rms,
                actual_rms=actual_rms,
                remainder_rms=conservative_remainder_rms,
                diagnostics=diagnostics,
            )
        return ModelValidityDecision(
            action="reuse",
            reason="actual-versus-predicted remainder is calibrated",
            recommended_scale=min(1.0 / self.shrink_factor, 2.0),
            predicted_rms=predicted_rms,
            actual_rms=actual_rms,
            remainder_rms=conservative_remainder_rms,
            diagnostics=diagnostics,
        )


__all__ = [
    "ModelValidityController",
    "PairedMultiFidelityEstimate",
    "PredictiveValidityReceipt",
    "ResidualObservation",
    "ResidualPrediction",
    "RobustResidualModel",
    "paired_control_variate_estimate",
]
