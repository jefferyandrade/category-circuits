"""Analytic logit-coordinate trust budgets for canonical modern-v1 pruning.

Corrected v1 attribution is a raw target-logit/evidence object.  The direct
paired oracle is a BCE object.  This module bridges those units without using
oracle outcomes to tune the proposal geometry.

For a positive affine calibration

    z_cal = a * z_raw + b,

binary BCE is 1-Lipschitz in the calibrated logit ``z_cal``.  A registered
dense-baseline BCE increment budget ``B`` therefore induces the proposal-only
raw-logit radii

    tau_F_raw = eta_F * B / a
    tau_L_raw = eta_L * B / a
    tau_Q_raw = tau_L_raw ** 2.

The offset ``b`` fixes the calibrated origin but does not affect route
displacements.  These radii only define a current-chart trust region; the
direct paired oracle continues to own every mask decision.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

from .contracts import content_hash


GEOMETRIC_TRUST_BUDGET_POLICY_SCHEMA: Final = (
    "partition_free_v1_geometric_trust_budget_policy_v1"
)
GEOMETRIC_TRUST_BUDGET_SCHEMA: Final = (
    "partition_free_v1_geometric_trust_budgets_v1"
)
GEOMETRIC_TRUST_BUDGET_EQUATION_ID: Final = (
    "raw_target_logit_tau_equals_calibrated_logit_B_over_affine_slope_v1"
)
GEOMETRIC_TRUST_BUDGET_AUTHORITY: Final = (
    "proposal_trust_only_direct_paired_dense_baseline_oracle_owns_acceptance"
)
LOSS_RESCALING_LAW: Final = (
    "under positive affine logit calibration z_cal=a*z_raw+b: offsets do not "
    "change displacements; raw tau_F and tau_L scale as 1/a; raw tau_Q scales "
    "as 1/a^2; action identities are invariant when route geometry and budgets "
    "are transported together"
)
GEOMETRIC_COORDINATE_RELATIVE_TOLERANCE: Final = 1e-12
GEOMETRIC_COORDINATE_MAX_ULPS: Final = 16


def same_geometric_coordinate(observed: object, expected: object) -> bool:
    """Compare coordinate values without introducing an absolute unit floor.

    The explicit zero guard is important: a positive analytic radius may never
    round-trip as zero merely because its declared unit happens to be small.
    The relative term is covariant under positive coordinate rescaling, while
    the ULP term admits only representational noise at the values being
    compared rather than noise measured in an unrelated unit-scale constant.
    """

    if isinstance(observed, bool) or isinstance(expected, bool):
        return False
    try:
        left = float(observed)
        right = float(expected)
    except (TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(left) or not math.isfinite(right):
        return False
    if left == right:
        return True
    if (left == 0.0) != (right == 0.0):
        return False
    scale = max(abs(left), abs(right))
    relative_allowance = GEOMETRIC_COORDINATE_RELATIVE_TOLERANCE * scale
    ulp_allowance = GEOMETRIC_COORDINATE_MAX_ULPS * max(
        math.ulp(left),
        math.ulp(right),
    )
    return abs(left - right) <= max(relative_allowance, ulp_allowance)


def _nonempty(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _finite_positive(value: object, label: str) -> float:
    result = _finite_nonnegative(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _freeze(value: Any, label: str) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise TypeError(f"{label} keys must be nonempty strings")
            frozen[key] = _freeze(item, f"{label}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(
            _freeze(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a nonfinite float")
        return float(value)
    if isinstance(value, (str, bytes, int, bool)) or value is None:
        return value
    raise TypeError(f"{label} contains unsupported value {type(value)!r}")


@dataclass(frozen=True, slots=True)
class V1GeometricTrustBudgetPolicy:
    """Dimensionless campaign coordinates for one chart's trust radii."""

    policy_id: str
    finsler_to_damage_multiplier: float = 1.0
    metric_length_to_damage_multiplier: float = 1.0
    source_receipt_hash: str = "unregistered-geometric-budget-policy-source"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    equation_id: str = GEOMETRIC_TRUST_BUDGET_EQUATION_ID
    schema: str = GEOMETRIC_TRUST_BUDGET_POLICY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _nonempty(self.policy_id, "policy_id"))
        object.__setattr__(
            self,
            "finsler_to_damage_multiplier",
            _finite_positive(
                self.finsler_to_damage_multiplier,
                "finsler_to_damage_multiplier",
            ),
        )
        object.__setattr__(
            self,
            "metric_length_to_damage_multiplier",
            _finite_positive(
                self.metric_length_to_damage_multiplier,
                "metric_length_to_damage_multiplier",
            ),
        )
        object.__setattr__(
            self,
            "source_receipt_hash",
            _nonempty(self.source_receipt_hash, "source_receipt_hash"),
        )
        object.__setattr__(self, "metadata", _freeze(self.metadata, "metadata"))
        if self.equation_id != GEOMETRIC_TRUST_BUDGET_EQUATION_ID:
            raise ValueError("unsupported geometric trust-budget equation")
        if self.schema != GEOMETRIC_TRUST_BUDGET_POLICY_SCHEMA:
            raise ValueError("unsupported geometric trust-budget policy schema")

    @property
    def policy_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-geometric-trust-budget-policy-",
        )

    def derive(
        self,
        *,
        maximum_dense_baseline_mean_bce_increment: float,
        damage_target_derivation_hash: str,
        current_mask_hash: str,
        raw_logit_to_calibrated_logit_slope: float = 1.0,
    ) -> "V1GeometricTrustBudgets":
        """Derive one fresh chart's raw-logit radii from the BCE budget."""

        damage = _finite_nonnegative(
            maximum_dense_baseline_mean_bce_increment,
            "maximum_dense_baseline_mean_bce_increment",
        )
        slope = _finite_positive(
            raw_logit_to_calibrated_logit_slope,
            "raw_logit_to_calibrated_logit_slope",
        )
        calibrated_path = self.finsler_to_damage_multiplier * damage
        calibrated_length = self.metric_length_to_damage_multiplier * damage
        path = calibrated_path / slope
        length = calibrated_length / slope
        return V1GeometricTrustBudgets(
            policy_hash=self.policy_hash,
            damage_target_derivation_hash=_nonempty(
                damage_target_derivation_hash,
                "damage_target_derivation_hash",
            ),
            current_mask_hash=_nonempty(current_mask_hash, "current_mask_hash"),
            maximum_dense_baseline_mean_bce_increment=damage,
            finsler_path_budget=path,
            metric_length_budget=length,
            quadratic_trust_energy_budget=length * length,
            finsler_to_damage_multiplier=self.finsler_to_damage_multiplier,
            metric_length_to_damage_multiplier=(
                self.metric_length_to_damage_multiplier
            ),
            raw_logit_to_calibrated_logit_slope=slope,
            calibrated_logit_finsler_path_budget=calibrated_path,
            calibrated_logit_metric_length_budget=calibrated_length,
        )


@dataclass(frozen=True, slots=True)
class V1GeometricTrustBudgets:
    """Typed proposal radii for one exact current-mask chart."""

    policy_hash: str
    damage_target_derivation_hash: str
    current_mask_hash: str
    maximum_dense_baseline_mean_bce_increment: float
    finsler_path_budget: float
    metric_length_budget: float
    quadratic_trust_energy_budget: float
    finsler_to_damage_multiplier: float
    metric_length_to_damage_multiplier: float
    raw_logit_to_calibrated_logit_slope: float = 1.0
    calibrated_logit_finsler_path_budget: float | None = None
    calibrated_logit_metric_length_budget: float | None = None
    acceptance_authority: str = GEOMETRIC_TRUST_BUDGET_AUTHORITY
    loss_rescaling_law: str = LOSS_RESCALING_LAW
    equation_id: str = GEOMETRIC_TRUST_BUDGET_EQUATION_ID
    schema: str = GEOMETRIC_TRUST_BUDGET_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "policy_hash",
            "damage_target_derivation_hash",
            "current_mask_hash",
            "acceptance_authority",
            "loss_rescaling_law",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        for name in (
            "maximum_dense_baseline_mean_bce_increment",
            "finsler_path_budget",
            "metric_length_budget",
            "quadratic_trust_energy_budget",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name),
            )
        for name in (
            "finsler_to_damage_multiplier",
            "metric_length_to_damage_multiplier",
            "raw_logit_to_calibrated_logit_slope",
        ):
            object.__setattr__(
                self,
                name,
                _finite_positive(getattr(self, name), name),
            )
        expected_calibrated_path = _finite_nonnegative(
            self.finsler_to_damage_multiplier
            * self.maximum_dense_baseline_mean_bce_increment,
            "derived calibrated-logit finsler path",
        )
        expected_calibrated_length = _finite_nonnegative(
            self.metric_length_to_damage_multiplier
            * self.maximum_dense_baseline_mean_bce_increment,
            "derived calibrated-logit metric length",
        )
        calibrated_path = (
            expected_calibrated_path
            if self.calibrated_logit_finsler_path_budget is None
            else _finite_nonnegative(
                self.calibrated_logit_finsler_path_budget,
                "calibrated_logit_finsler_path_budget",
            )
        )
        calibrated_length = (
            expected_calibrated_length
            if self.calibrated_logit_metric_length_budget is None
            else _finite_nonnegative(
                self.calibrated_logit_metric_length_budget,
                "calibrated_logit_metric_length_budget",
            )
        )
        object.__setattr__(
            self,
            "calibrated_logit_finsler_path_budget",
            calibrated_path,
        )
        object.__setattr__(
            self,
            "calibrated_logit_metric_length_budget",
            calibrated_length,
        )
        expected_path = _finite_nonnegative(
            calibrated_path / self.raw_logit_to_calibrated_logit_slope,
            "derived raw-logit finsler path",
        )
        expected_length = _finite_nonnegative(
            calibrated_length / self.raw_logit_to_calibrated_logit_slope,
            "derived raw-logit metric length",
        )
        expected_energy = _finite_nonnegative(
            expected_length * expected_length,
            "derived raw-logit quadratic energy",
        )
        if calibrated_path > 0.0 and expected_path == 0.0:
            raise ValueError("raw-logit finsler path underflowed to zero")
        if calibrated_length > 0.0 and expected_length == 0.0:
            raise ValueError("raw-logit metric length underflowed to zero")
        if expected_length > 0.0 and expected_energy == 0.0:
            raise ValueError("raw-logit quadratic energy underflowed to zero")
        for observed, expected, label in (
            (
                calibrated_path,
                expected_calibrated_path,
                "calibrated-logit finsler path",
            ),
            (
                calibrated_length,
                expected_calibrated_length,
                "calibrated-logit metric length",
            ),
        ):
            if not same_geometric_coordinate(observed, expected):
                raise ValueError(f"{label} budget does not satisfy its derivation law")
        for observed, expected, label in (
            (self.finsler_path_budget, expected_path, "raw-logit finsler path"),
            (self.metric_length_budget, expected_length, "raw-logit metric length"),
            (
                self.quadratic_trust_energy_budget,
                expected_energy,
                "raw-logit quadratic energy",
            ),
        ):
            if not same_geometric_coordinate(observed, expected):
                raise ValueError(f"{label} budget does not satisfy its derivation law")
        if self.acceptance_authority != GEOMETRIC_TRUST_BUDGET_AUTHORITY:
            raise PermissionError("geometric trust budgets cannot accept masks")
        if self.loss_rescaling_law != LOSS_RESCALING_LAW:
            raise ValueError("unsupported geometric trust-budget scaling law")
        if self.equation_id != GEOMETRIC_TRUST_BUDGET_EQUATION_ID:
            raise ValueError("unsupported geometric trust-budget equation")
        if self.schema != GEOMETRIC_TRUST_BUDGET_SCHEMA:
            raise ValueError("unsupported geometric trust-budget schema")

    @property
    def budget_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-geometric-trust-budgets-",
        )

    def rescale_loss(self, scale: float) -> "V1GeometricTrustBudgets":
        """Return the exact covariant budget object under positive loss scale."""

        factor = _finite_positive(scale, "loss scale")
        return V1GeometricTrustBudgets(
            policy_hash=self.policy_hash,
            damage_target_derivation_hash=self.damage_target_derivation_hash,
            current_mask_hash=self.current_mask_hash,
            maximum_dense_baseline_mean_bce_increment=(
                factor * self.maximum_dense_baseline_mean_bce_increment
            ),
            finsler_path_budget=factor * self.finsler_path_budget,
            metric_length_budget=factor * self.metric_length_budget,
            quadratic_trust_energy_budget=(
                factor * factor * self.quadratic_trust_energy_budget
            ),
            finsler_to_damage_multiplier=self.finsler_to_damage_multiplier,
            metric_length_to_damage_multiplier=(
                self.metric_length_to_damage_multiplier
            ),
            raw_logit_to_calibrated_logit_slope=(
                self.raw_logit_to_calibrated_logit_slope
            ),
            calibrated_logit_finsler_path_budget=(
                factor * self.calibrated_logit_finsler_path_budget
            ),
            calibrated_logit_metric_length_budget=(
                factor * self.calibrated_logit_metric_length_budget
            ),
        )


__all__ = [
    "GEOMETRIC_COORDINATE_MAX_ULPS",
    "GEOMETRIC_COORDINATE_RELATIVE_TOLERANCE",
    "GEOMETRIC_TRUST_BUDGET_AUTHORITY",
    "GEOMETRIC_TRUST_BUDGET_EQUATION_ID",
    "LOSS_RESCALING_LAW",
    "V1GeometricTrustBudgetPolicy",
    "V1GeometricTrustBudgets",
    "same_geometric_coordinate",
]
