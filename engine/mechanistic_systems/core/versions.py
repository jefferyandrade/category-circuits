"""Immutable circuit-engine versions and opt-in v2 selection policy."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from contracts.hashing import stable_digest
from mechanistic_systems.contracts import (
    CANONICAL_PRUNING_CALIBRATION_ALGORITHM,
    RATE_DISTORTION_V2,
    RISK_AWARE_ROUTE_GATE_V2,
    ROUTE_GATE_V1,
    SAMPLED_PATH_V1_CALIBRATION_ALGORITHM,
    SAMPLED_PATH_FRONTIER_V1,
    CircuitPlan,
)


CALIBRATION_FAMILIES = (
    "none",
    "shared_affine",
    "classwise_diagonal_affine",
)


@dataclass(frozen=True)
class AlgorithmVersion:
    algorithm_id: str
    measurement_version: str
    immutable: bool
    promotion_status: str


@dataclass(frozen=True)
class CircuitPipelineVersion:
    """Explicit composition of measurement, identification, and calibration.

    The sampled-path objective and its nuisance-calibration estimator are
    independently versioned scientific objects.  Calling both compositions
    merely "v1" obscures a real estimand change, so reports and receipts use
    this product identity.
    """

    measurement_version: str
    identification_version: str
    calibration_version: str
    calibration_mode: str
    composition_class: str

    @property
    def content_hash(self) -> str:
        return stable_digest(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "measurement_version": self.measurement_version,
            "identification_version": self.identification_version,
            "calibration_version": self.calibration_version,
            "calibration_mode": self.calibration_mode,
            "composition_class": self.composition_class,
        }
        if include_hash:
            payload["pipeline_composition_hash"] = self.content_hash
        return payload


def compose_circuit_pipeline_version(
    *,
    measurement_version: str,
    identification_version: str,
    calibration_version: str,
    calibration_mode: str,
) -> CircuitPipelineVersion:
    """Classify one exact scientific pipeline without architecture dispatch."""

    measurement = str(measurement_version)
    identification = str(identification_version)
    calibration = str(calibration_version)
    mode = str(calibration_mode)
    if not all((measurement, identification, calibration, mode)):
        raise ValueError("pipeline composition requires all component versions")

    if (
        measurement == ROUTE_GATE_V1
        and identification == SAMPLED_PATH_FRONTIER_V1
        and calibration == CANONICAL_PRUNING_CALIBRATION_ALGORITHM
    ):
        composition_class = "canonical_default"
    elif (
        measurement == ROUTE_GATE_V1
        and identification == SAMPLED_PATH_FRONTIER_V1
        and calibration == SAMPLED_PATH_V1_CALIBRATION_ALGORITHM
    ):
        composition_class = "historical_calibration_compatibility"
    else:
        composition_class = "experimental_or_nonbaseline"

    return CircuitPipelineVersion(
        measurement_version=measurement,
        identification_version=identification,
        calibration_version=calibration,
        calibration_mode=mode,
        composition_class=composition_class,
    )


ROUTE_GATE_BASELINE = AlgorithmVersion(
    algorithm_id=ROUTE_GATE_V1,
    measurement_version=ROUTE_GATE_V1,
    immutable=True,
    promotion_status="canonical_baseline",
)
SAMPLED_PATH_BASELINE = AlgorithmVersion(
    algorithm_id=SAMPLED_PATH_FRONTIER_V1,
    measurement_version=ROUTE_GATE_V1,
    immutable=True,
    promotion_status="canonical_baseline",
)
RISK_AWARE_RATE_DISTORTION = AlgorithmVersion(
    algorithm_id=RATE_DISTORTION_V2,
    measurement_version=RISK_AWARE_ROUTE_GATE_V2,
    immutable=False,
    promotion_status="experimental_requires_validity_and_pareto",
)
EVIDENCE_SAMPLED_PATH = AlgorithmVersion(
    algorithm_id="evidence_foundation_sampled_path_v1",
    measurement_version=ROUTE_GATE_V1,
    immutable=False,
    promotion_status=(
        "promoted_twice_canary_20260720_jobs_33649767_and_33655045"
    ),
)
CONTINUOUS_CANONICAL_DEFAULT = AlgorithmVersion(
    algorithm_id="evidence_foundation_continuous_v2",
    measurement_version=ROUTE_GATE_V1,
    immutable=False,
    promotion_status="historical_exploratory_default_superseded_2026_07_28",
)
BCE_PRECISION_DEFAULT = AlgorithmVersion(
    algorithm_id="evidence_foundation_bce_precision_v1",
    measurement_version=ROUTE_GATE_V1,
    immutable=False,
    promotion_status="exploratory_default_by_decision_2026_07_28",
)


def canonical_identification_default() -> str:
    """The identification engine new campaigns use unless they opt out."""

    return "evidence_foundation_bce_precision_v1"


def fixed16_replication_mode() -> dict[str, Any]:
    """Exact historical replication: the immutable fixed-16 baseline.

    Selecting this composition reproduces the pre-evidence engine
    byte-for-byte under the sealed contracts — sampled_path_frontier_v1 at
    a single sixteen-per-class stage over the frozen four-block window.
    """

    return {
        "algorithm": "statistical_sequential_symmetric_frontier",
        "algorithm_version": SAMPLED_PATH_FRONTIER_V1,
        "measurement_version": ROUTE_GATE_V1,
        "sample_stages_per_class": (16,),
        "score_images_per_class": 16,
        "mode": "historical_fixed16_replication",
    }


@dataclass(frozen=True)
class CalibrationSelection:
    family: str
    best_family: str
    best_mean_loss: float
    best_standard_error: float
    eligible_families: tuple[str, ...]
    rule: str = "simplest_within_one_se"


def select_simplest_calibration_within_one_se(
    losses_by_family: Mapping[str, Sequence[float]],
) -> CalibrationSelection:
    """Select nuisance-calibration complexity using extraction-only folds."""

    if tuple(losses_by_family) != CALIBRATION_FAMILIES:
        raise ValueError(
            "calibration families must be ordered none, shared_affine, classwise_diagonal_affine"
        )
    summaries: dict[str, tuple[float, float]] = {}
    for family in CALIBRATION_FAMILIES:
        values = np.asarray(losses_by_family[family], dtype=np.float64)
        if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
            raise ValueError("calibration selection requires at least two finite inner-fold losses")
        summaries[family] = (
            float(np.mean(values)),
            float(np.std(values, ddof=1) / math.sqrt(values.size)),
        )
    best = min(CALIBRATION_FAMILIES, key=lambda family: (summaries[family][0], CALIBRATION_FAMILIES.index(family)))
    threshold = summaries[best][0] + summaries[best][1]
    eligible = tuple(family for family in CALIBRATION_FAMILIES if summaries[family][0] <= threshold)
    selected = eligible[0]
    return CalibrationSelection(
        family=selected,
        best_family=best,
        best_mean_loss=summaries[best][0],
        best_standard_error=summaries[best][1],
        eligible_families=eligible,
    )


def validate_algorithm_version(plan: CircuitPlan) -> dict[str, Any]:
    """Return the explicit scientific version for a validated circuit plan."""

    plan.validate(claim_bearing=plan.status == "frozen").require_valid()
    algorithm = str(plan.search_policy.get("algorithm", ""))
    if algorithm == "statistical_sequential_symmetric_frontier":
        declared = str(plan.search_policy.get("algorithm_version", SAMPLED_PATH_FRONTIER_V1))
        if declared != SAMPLED_PATH_FRONTIER_V1:
            raise ValueError("the production sampled-path baseline is immutable")
        calibration_version = str(
            plan.search_policy.get("calibration_optimization_contract", "")
        )
        calibration_mode = str(
            plan.search_policy.get("target_calibration_mode", "")
            or plan.search_policy.get("calibration_mode", "")
        )
        if not calibration_mode and calibration_version == (
            SAMPLED_PATH_V1_CALIBRATION_ALGORITHM
        ):
            calibration_mode = "c2_shared_diagonal_affine"
        composition = compose_circuit_pipeline_version(
            measurement_version=str(
                plan.search_policy.get("measurement_version", ROUTE_GATE_V1)
            ),
            identification_version=SAMPLED_PATH_FRONTIER_V1,
            calibration_version=calibration_version,
            calibration_mode=calibration_mode,
        )
        return {
            "algorithm_version": SAMPLED_PATH_FRONTIER_V1,
            **composition.to_dict(),
            "immutable": True,
            "immutability_scope": "measurement_and_frontier_components",
            "historical_full_control_requires": (
                "historical_calibration_compatibility plus a sealed historical "
                "MeasurementArtifact and matched Study input operator"
            ),
        }
    if algorithm == "risk_aware_rate_distortion_v2":
        return {
            "algorithm_version": RATE_DISTORTION_V2,
            "measurement_version": RISK_AWARE_ROUTE_GATE_V2,
            "immutable": False,
            "promotion_gate": "estimator_validity_stability_and_model_mean_pareto",
        }
    if algorithm == "evidence_foundation_continuous_v2":
        return {
            "algorithm_version": "evidence_foundation_continuous_v2",
            "measurement_version": str(
                plan.search_policy.get("measurement_version", ROUTE_GATE_V1)
            ),
            "immutable": False,
            "confidence_rule": "alpha_spending_over_information_fraction",
            "canonical_status": "canonical_default_by_decision_2026_07_20",
            "declared_budget_contract": (
                "spends its declared image budget to maximize certified "
                "pruning under (distortion budget, run alpha); iso-runtime "
                "vs fixed16 with substantially deeper certified frontiers "
                "(canary job 33655045)"
            ),
            "known_caveat": (
                "balanced-BCE scalar can credit target-attenuation as "
                "improvement; two-sided positive/negative distortion "
                "budgets are the registered follow-up"
            ),
            "replication": "use fixed16_replication_mode() for the exact "
            "historical engine",
        }
    if algorithm == "evidence_foundation_bce_precision_v1":
        return {
            "algorithm_version": "evidence_foundation_bce_precision_v1",
            "measurement_version": str(
                plan.search_policy.get("measurement_version", ROUTE_GATE_V1)
            ),
            "immutable": False,
            "confidence_rule": (
                "paired crossfit heldout BCE Student-t intervals with "
                "Bonferroni protection over frozen planned looks"
            ),
            "scientific_inputs": ("confidence_level", "sensitivity_tau"),
            "decision_rule": (
                "accept iff upper<=tau; reject iff lower>tau; otherwise "
                "sample another nested prefix; exhaustion means no-prune"
            ),
            "normalization": "raw_BCE_increase_per_actual_route_removed",
            "calibration": "registered_BCE_optimizer_backend",
            "canonical_status": "exploratory_default_by_decision_2026_07_28",
            "known_caveat": (
                "raw BCE is unbounded, so this is not a distribution-free "
                "or campaign-global claim-bearing guarantee"
            ),
        }
    if algorithm == "evidence_foundation_sampled_path_v1":
        return {
            "algorithm_version": "evidence_foundation_sampled_path_v1",
            "measurement_version": str(
                plan.search_policy.get("measurement_version", ROUTE_GATE_V1)
            ),
            "immutable": False,
            "confidence_rule": "one_sided_moderated_t_alpha_ledger",
            "promotion_gate": (
                "canary_data_efficiency_and_heldout_quality_pareto_vs_fixed16_baseline"
            ),
        }
    return {
        "algorithm_version": str(plan.search_policy.get("algorithm_version", algorithm)),
        "measurement_version": str(plan.search_policy.get("measurement_version", "unspecified")),
        "immutable": False,
    }


__all__ = [
    "AlgorithmVersion",
    "CALIBRATION_FAMILIES",
    "CalibrationSelection",
    "CircuitPipelineVersion",
    "BCE_PRECISION_DEFAULT",
    "CONTINUOUS_CANONICAL_DEFAULT",
    "EVIDENCE_SAMPLED_PATH",
    "canonical_identification_default",
    "fixed16_replication_mode",
    "RISK_AWARE_RATE_DISTORTION",
    "ROUTE_GATE_BASELINE",
    "SAMPLED_PATH_BASELINE",
    "compose_circuit_pipeline_version",
    "select_simplest_calibration_within_one_se",
    "validate_algorithm_version",
]
