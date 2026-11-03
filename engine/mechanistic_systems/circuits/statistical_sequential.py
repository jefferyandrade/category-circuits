"""Frozen semantics for ``sampled_path_frontier_v1`` circuit identification.

The native Mechanistic Systems runner owns claim-bearing candidate evaluation
and frontier execution. Historical oracle invocation is an administrative
CircuitFlow concern. The sampled-path/frontier design remains v1 while its
calibration estimator is an independently versioned, content-hashed policy.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from contracts.hashing import stable_digest
from mechanistic_systems.contracts import (
    CANONICAL_PRUNING_CALIBRATION_ALGORITHM,
    CANONICAL_TARGET_CALIBRATION_MODE,
    CircuitPlan,
    PROPER_SCORE_SHRINKAGE_AFFINE_V2,
    PRODUCTION_CIRCUIT_SEARCH_ALGORITHM,
    SAMPLED_PATH_GROUP_REFERENCE_POLICY,
    SAMPLED_PATH_V1_CALIBRATION_ALGORITHM,
)

LEGACY_STATISTICAL_SEQUENTIAL_PROFILE_SCHEMA = (
    "mechanistic_statistical_sequential_symmetric_frontier_profile_v2"
)
PRE_BCE_PRECISION_PROFILE_SCHEMA = (
    "mechanistic_statistical_sequential_symmetric_frontier_profile_v3"
)
STATISTICAL_SEQUENTIAL_PROFILE_SCHEMA = (
    "mechanistic_statistical_sequential_symmetric_frontier_profile_v4"
)
PRUNING_DENSITY_SCHEDULE_SEMANTICS = (
    "historical_family_specific_multiplicative_decay_v1"
)
SCALAR_CALIBRATION_TARGET_TYPES = frozenset(
    {
        "class_logit",
        "linear_probe_category_direction",
        "linear_probe_one_vs_rest_log_odds",
        "natural_metric_category_direction",
    }
)


def _is_scalar_calibration_target(target_type: Any) -> bool:
    """Recognize the sealed scalar-query families supported by C2 calibration."""

    return str(target_type).strip().lower() in SCALAR_CALIBRATION_TARGET_TYPES


def _resolved_precision_policy(grid_units: float) -> dict[str, Any]:
    """Mirror the frozen bounded-sequential precision compilation."""

    value = float(grid_units)
    if not 0.01 <= value <= 1.0:
        raise ValueError("statistical_precision_grid_units must lie in [0.01, 1]")
    looseness = min(1.0, max(0.0, (value - 0.01) / (0.15 - 0.01)))
    posterior_probability = 0.95 - (0.95 - 0.50) * looseness
    if value <= 0.0200000001:
        stages = (4, 8, 16, 32, 60, 120)
    elif value <= 0.0500000001:
        stages = (4, 8, 16, 32, 60)
    elif value <= 0.1000000001:
        stages = (4, 8, 16, 32)
    else:
        stages = (4, 8, 16)
    return {
        "posterior_probability": float(posterior_probability),
        "max_posterior_boundary_width": value,
        "sample_stages_per_class": stages,
        "posterior_regret_tolerance": value / (2.0 * 1.96 * (2.0 * 3.141592653589793) ** 0.5),
    }


def _view_by_role(config: Any, role: str) -> Any | None:
    for view in tuple(getattr(config.dataset, "views", ()) or ()):
        if str(getattr(view, "role", "")) == role:
            return view
    return None


@dataclass(frozen=True)
class StatisticalSequentialProfile:
    """Claim-relevant semantics extracted from one production config."""

    frontier_policy: str
    sensitivity_tau: float
    candidate_evaluation_mode: str
    statistical_estimand: str
    statistical_sampling_design: str
    statistical_inference_engine: str
    statistical_precision_grid_units: float
    sample_stages_per_class: tuple[int, ...]
    crossfit_fold_count: int
    statistical_sampling_seed: int
    statistical_panel_policy: str
    statistical_panel_execution: str
    statistical_image_bank_mode: str
    statistical_full_extract_resolve: bool
    finite_population_correction: bool
    design_weighted_calibration: bool
    variance_prior_strength: float
    frontier_window_size: int
    initial_panel_image_budget: int
    min_positive_images: int
    min_negative_categories: int
    readout_policy: str
    readout_layer: str
    preserve_readout_layer: bool
    calibration_mode: str
    calibration_backend: str
    binary_loss_weighting: str
    candidate_evaluation_fusion: bool
    forward_replay_policy: str
    custom_kernel_policy: str
    kernel_backend: str
    exactness_contract: str
    layer_group_names: tuple[str, ...]
    reservoir_role: str
    reservoir_format: str
    reservoir_scope: str
    reservoir_max_per_class: int | None
    validation_role: str
    validation_format: str
    validation_max_per_class: int | None
    validation_require_exact_per_class: bool
    bce_precision_confidence_level: float = 0.95
    posterior_accept_probability: float = 0.95
    posterior_reject_probability: float = 0.95
    max_posterior_boundary_width: float = 0.02
    posterior_regret_tolerance: float = 0.0
    panel_image_budget_sequence: tuple[int, ...] = (2048,)
    statistical_panel_batch_multiple: int = 256
    min_neg_images_per_category: int = 1
    min_neg_images_per_new_category: int = 2
    max_negative_categories: int = 999
    calibration_enforce_positive_scale: bool = True
    calibration_iterations: int = 64
    calibration_learning_rate: float = 0.25
    calibration_min_scale: float = 1e-8
    calibration_optimization_contract: str = (
        SAMPLED_PATH_V1_CALIBRATION_ALGORITHM
    )
    scalar_target_semantics: bool = True
    fully_pruned_anchor_loss: float = 1.384909
    branch_sign_convention: str = "nonnegative_positive_negative_strict"
    mask_retention_rounding: str = "ceil_per_layer_per_sign_branch"
    score_tie_policy: str = "stable_flat_route_order"
    partition_schedule: str = "symmetric_full_turns"
    initial_partition: str = "negative"
    sampling_path: str = "stochastic_class_indexed_nested_panels"
    paired_comparison: str = "common_random_numbers"
    crossfit_calibration: str = "fit_complement_score_heldout_swap"
    slope_numerator: str = "paired_crossfit_risk_delta"
    denominator_policy: str = (
        "requested_sign_density_delta_times_total_layer_route_count"
    )
    final_validation: str = "untouched_full_validation_split"
    object_currency: str = (
        "symmetric_frontier_requested_signed_density_x_full_layer_count"
    )
    density_axis_mode: str = "legacy_old_sparse"
    density_axis_decay: float = 0.9
    target_preservation_linear_decay: float = 0.85
    target_preservation_conv_decay: float = 0.9
    target_preservation_linear_target_final_density: float = 1e-4
    target_preservation_conv_target_final_density: float = 5e-2
    density_schedule_semantics: str = PRUNING_DENSITY_SCHEDULE_SEMANTICS
    # Which readout columns the crossfit calibration consumes:
    # "all" = every estimable column (the cross-category response profile is
    # a first-class output of the same batched fit); "target_only" = the
    # decision column alone (pinned automatically for the sealed v1
    # calibration contract); or an explicit tuple of class ids (fail-closed).
    # The pruning DECISION is always the target class column.
    readout_columns: str | tuple[int, ...] = "all"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": STATISTICAL_SEQUENTIAL_PROFILE_SCHEMA,
            **asdict(self),
        }

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "StatisticalSequentialProfile":
        """Load one sealed v1 profile without consulting a legacy config."""

        schema_name = str(payload.get("schema_name", ""))
        if schema_name not in {
            LEGACY_STATISTICAL_SEQUENTIAL_PROFILE_SCHEMA,
            PRE_BCE_PRECISION_PROFILE_SCHEMA,
            STATISTICAL_SEQUENTIAL_PROFILE_SCHEMA,
        }:
            raise ValueError("unsupported statistical sequential profile schema")
        field_names = {item.name for item in fields(cls)}
        unknown = sorted(set(payload) - field_names - {"schema_name"})
        if unknown:
            raise ValueError(
                "statistical sequential profile contains unknown fields: "
                + ",".join(unknown)
            )
        values = {name: payload[name] for name in field_names if name in payload}
        if isinstance(values.get("readout_columns"), list):
            values["readout_columns"] = tuple(
                int(item) for item in values["readout_columns"]
            )
        if "calibration_optimization_contract" not in values:
            # Profiles serialized before optimizer identity was explicit are
            # historical v1 artifacts. Never reinterpret them under a newer
            # default during migration.
            values["calibration_optimization_contract"] = (
                SAMPLED_PATH_V1_CALIBRATION_ALGORITHM
            )
        for name in (
            "sample_stages_per_class",
            "layer_group_names",
            "panel_image_budget_sequence",
        ):
            raw = values.get(name)
            if raw is not None:
                if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
                    raise TypeError(f"{name} must be an ordered sequence")
                converter = str if name == "layer_group_names" else int
                values[name] = tuple(converter(value) for value in raw)
        try:
            result = cls(**values)
        except TypeError as error:
            raise ValueError(
                "statistical sequential profile is missing required fields"
            ) from error
        result.validate()
        return result

    @classmethod
    def load(cls, path: str | Path) -> "StatisticalSequentialProfile":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def write(self, path: str | Path) -> Path:
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target

    @property
    def content_hash(self) -> str:
        return stable_digest(self.to_dict())

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        historical_compatibility: bool = False,
    ) -> "StatisticalSequentialProfile":
        runtime = config.runtime
        sensitivity = config.sensitivity
        metrics = config.metrics
        grid = config.grid
        reservoir = _view_by_role(config, "statistical_reservoir")
        validation = _view_by_role(config, "validation")
        compute_forward = dict(getattr(runtime, "compute_forward_kwargs", {}) or {})
        explicit_calibration_contract = compute_forward.get(
            "calibration_optimization_contract"
        )
        calibration_contract = str(
            explicit_calibration_contract
            or (
                SAMPLED_PATH_V1_CALIBRATION_ALGORITHM
                if historical_compatibility
                else CANONICAL_PRUNING_CALIBRATION_ALGORITHM
            )
        )
        calibration_mode = str(
            getattr(runtime, "target_calibration_mode", "")
            or CANONICAL_TARGET_CALIBRATION_MODE
        )
        if (
            historical_compatibility
            and calibration_contract == SAMPLED_PATH_V1_CALIBRATION_ALGORITHM
        ):
            calibration_mode = "c2_shared_diagonal_affine"
        precision = float(
            getattr(runtime, "statistical_precision_grid_units", float("nan"))
        )
        resolved_precision = _resolved_precision_policy(precision)
        budget_raw = getattr(runtime, "panel_image_budget_sequence", (2048,))
        if isinstance(budget_raw, str):
            panel_budgets = tuple(
                int(item) for item in budget_raw.split(",") if item.strip()
            )
        else:
            panel_budgets = tuple(int(item) for item in budget_raw)
        return cls(
            frontier_policy=str(getattr(sensitivity, "frontier_policy", "")),
            sensitivity_tau=float(getattr(sensitivity, "sensitivity_tau", float("nan"))),
            candidate_evaluation_mode=str(
                getattr(runtime, "candidate_evaluation_mode", "")
            ),
            statistical_estimand=str(getattr(runtime, "statistical_estimand", "")),
            statistical_sampling_design=str(
                getattr(runtime, "statistical_sampling_design", "")
            ),
            statistical_inference_engine=str(
                getattr(runtime, "statistical_inference_engine", "")
            ),
            statistical_precision_grid_units=precision,
            sample_stages_per_class=tuple(
                int(value) for value in resolved_precision["sample_stages_per_class"]
            ),
            crossfit_fold_count=int(getattr(runtime, "crossfit_fold_count", 0)),
            statistical_sampling_seed=int(
                getattr(runtime, "statistical_sampling_seed", -1)
            ),
            statistical_panel_policy=str(
                getattr(runtime, "statistical_panel_policy", "")
            ),
            statistical_panel_execution=str(
                getattr(runtime, "statistical_panel_execution", "")
            ),
            statistical_image_bank_mode=str(
                getattr(runtime, "statistical_image_bank_mode", "")
            ),
            statistical_full_extract_resolve=bool(
                getattr(runtime, "statistical_full_extract_resolve", True)
            ),
            finite_population_correction=bool(
                getattr(runtime, "finite_population_correction", False)
            ),
            design_weighted_calibration=bool(
                getattr(runtime, "design_weighted_calibration", False)
            ),
            variance_prior_strength=float(
                getattr(runtime, "statistical_variance_prior_strength", 0.0)
            ),
            frontier_window_size=int(
                getattr(runtime, "statistical_frontier_window_size", 0)
            ),
            initial_panel_image_budget=int(
                getattr(runtime, "initial_panel_image_budget", 0)
            ),
            min_positive_images=int(getattr(runtime, "min_positive_images", 0)),
            min_negative_categories=int(
                getattr(runtime, "min_negative_categories", 0)
            ),
            readout_policy=str(getattr(runtime, "readout_policy", "")),
            readout_layer=str(getattr(runtime, "readout_layer", "") or ""),
            preserve_readout_layer=bool(
                getattr(runtime, "preserve_readout_layer", False)
            ),
            calibration_mode=calibration_mode,
            calibration_backend=str(compute_forward.get("affine_backend", "")),
            binary_loss_weighting=str(
                getattr(metrics, "binary_loss_weighting", "")
            ),
            candidate_evaluation_fusion=bool(
                getattr(runtime, "candidate_evaluation_fusion", False)
            ),
            forward_replay_policy=str(
                getattr(runtime, "forward_replay_policy", "")
            ),
            custom_kernel_policy=str(getattr(runtime, "custom_kernel_policy", "")),
            kernel_backend=str(getattr(runtime, "kernel_backend", "")),
            exactness_contract=str(getattr(runtime, "exactness_contract", "")),
            layer_group_names=tuple(
                str(value) for value in getattr(runtime, "layer_group_names", ())
            ),
            reservoir_role=str(getattr(reservoir, "role", "")),
            reservoir_format=str(getattr(reservoir, "format", "")),
            reservoir_scope=str(
                getattr(runtime, "statistical_reservoir_scope", "")
            ),
            reservoir_max_per_class=(
                None
                if reservoir is None or getattr(reservoir, "max_per_class", None) is None
                else int(reservoir.max_per_class)
            ),
            validation_role=str(getattr(validation, "role", "")),
            validation_format=str(getattr(validation, "format", "")),
            validation_max_per_class=(
                None
                if validation is None or getattr(validation, "max_per_class", None) is None
                else int(validation.max_per_class)
            ),
            validation_require_exact_per_class=bool(
                getattr(validation, "require_exact_per_class", False)
            ),
            bce_precision_confidence_level=float(
                getattr(runtime, "bce_precision_confidence_level", 0.95)
            ),
            posterior_accept_probability=float(
                resolved_precision["posterior_probability"]
            ),
            posterior_reject_probability=float(
                resolved_precision["posterior_probability"]
            ),
            max_posterior_boundary_width=float(
                resolved_precision["max_posterior_boundary_width"]
            ),
            posterior_regret_tolerance=float(
                resolved_precision["posterior_regret_tolerance"]
            ),
            panel_image_budget_sequence=panel_budgets or (2048,),
            statistical_panel_batch_multiple=int(
                getattr(runtime, "statistical_panel_batch_multiple", 256) or 256
            ),
            min_neg_images_per_category=int(
                getattr(runtime, "min_neg_images_per_category", 1) or 1
            ),
            min_neg_images_per_new_category=int(
                getattr(runtime, "min_neg_images_per_new_category", 2) or 2
            ),
            max_negative_categories=int(
                getattr(runtime, "max_negative_categories", 999) or 999
            ),
            calibration_enforce_positive_scale=bool(
                compute_forward.get("enforce_positive_scale", True)
            ),
            calibration_iterations=int(compute_forward.get("iters", 64) or 64),
            calibration_learning_rate=float(compute_forward.get("lr", 0.25) or 0.25),
            calibration_min_scale=float(compute_forward.get("ws_min", 1e-8) or 1e-8),
            calibration_optimization_contract=calibration_contract,
            scalar_target_semantics=_is_scalar_calibration_target(
                getattr(runtime, "target_type", "class_logit")
            ),
            density_axis_mode=str(getattr(grid, "density_axis_mode", "")),
            density_axis_decay=float(getattr(grid, "density_axis_decay", float("nan"))),
            target_preservation_linear_decay=float(
                getattr(grid, "target_preservation_linear_decay", float("nan"))
            ),
            target_preservation_conv_decay=float(
                getattr(grid, "target_preservation_conv_decay", float("nan"))
            ),
            target_preservation_linear_target_final_density=float(
                getattr(
                    grid,
                    "target_preservation_linear_target_final_density",
                    float("nan"),
                )
            ),
            target_preservation_conv_target_final_density=float(
                getattr(
                    grid,
                    "target_preservation_conv_target_final_density",
                    float("nan"),
                )
            ),
            density_schedule_semantics=PRUNING_DENSITY_SCHEDULE_SEMANTICS,
        )

    def validate(self) -> None:
        issues: list[str] = []
        required = {
            "frontier_policy": "level0_symmetric_frontier",
            "candidate_evaluation_mode": "statistical_sequential",
            "statistical_estimand": "crossfit_risk",
            "statistical_sampling_design": "cost_optimal_two_stage_neyman",
            "statistical_inference_engine": "bayesian_hierarchical_risk",
            "statistical_panel_policy": "cost_optimal_regret_panel",
            "statistical_panel_execution": "single_forward_crossfit_reuse",
            "statistical_image_bank_mode": "cpu_resident_uint8_class_indexed",
            "readout_policy": "fixed",
            "calibration_backend": "optimized",
            "binary_loss_weighting": "pos_weighted",
            "exactness_contract": "lossless_exact_no_nonvanishing_approximation",
            "reservoir_role": "statistical_reservoir",
            "reservoir_format": "imagefolder",
            "reservoir_scope": "entire_train_split",
            "validation_role": "validation",
            "validation_format": "imagefolder",
        }
        for field_name, expected in required.items():
            if str(getattr(self, field_name)) != expected:
                issues.append(f"{field_name}_invalid")
        if not 0.0 < self.sensitivity_tau < float("inf"):
            issues.append("sensitivity_tau_invalid")
        if not 0.5 < self.bce_precision_confidence_level < 1.0:
            issues.append("bce_precision_confidence_level_invalid")
        if not 0.0 < self.statistical_precision_grid_units <= 1.0:
            issues.append("statistical_precision_grid_units_invalid")
        if self.crossfit_fold_count < 2:
            issues.append("crossfit_fold_count_invalid")
        if (
            not self.sample_stages_per_class
            or tuple(sorted(set(self.sample_stages_per_class)))
            != self.sample_stages_per_class
            or any(value <= 0 for value in self.sample_stages_per_class)
        ):
            issues.append("sample_stages_per_class_invalid")
        if self.statistical_full_extract_resolve:
            issues.append("full_extract_resolution_forbidden")
        if not self.finite_population_correction:
            issues.append("finite_population_correction_disabled")
        if not self.design_weighted_calibration:
            issues.append("design_weighted_calibration_disabled")
        if self.variance_prior_strength <= 0.0:
            issues.append("variance_prior_strength_invalid")
        try:
            precision_policy = _resolved_precision_policy(
                self.statistical_precision_grid_units
            )
        except ValueError:
            precision_policy = None
            issues.append("statistical_precision_policy_invalid")
        if precision_policy is not None:
            expected_stages = tuple(precision_policy["sample_stages_per_class"])
            if self.sample_stages_per_class != expected_stages:
                issues.append("sample_stages_not_compiled_from_precision")
            for field_name in (
                "posterior_accept_probability",
                "posterior_reject_probability",
            ):
                if not abs(
                    float(getattr(self, field_name))
                    - float(precision_policy["posterior_probability"])
                ) <= 1e-12:
                    issues.append(f"{field_name}_not_compiled_from_precision")
            if not abs(
                self.max_posterior_boundary_width
                - float(precision_policy["max_posterior_boundary_width"])
            ) <= 1e-12:
                issues.append("max_posterior_boundary_width_not_compiled_from_precision")
            if not abs(
                self.posterior_regret_tolerance
                - float(precision_policy["posterior_regret_tolerance"])
            ) <= 1e-12:
                issues.append("posterior_regret_tolerance_not_compiled_from_precision")
        if self.frontier_window_size <= 0:
            issues.append("frontier_window_size_invalid")
        if self.initial_panel_image_budget <= 0:
            issues.append("initial_panel_image_budget_invalid")
        if self.min_positive_images <= 0 or self.min_negative_categories <= 0:
            issues.append("panel_minimums_invalid")
        if (
            not self.panel_image_budget_sequence
            or any(value <= 0 for value in self.panel_image_budget_sequence)
            or self.statistical_panel_batch_multiple <= 0
            or self.min_neg_images_per_category <= 0
            or self.min_neg_images_per_new_category <= 0
            or self.max_negative_categories < self.min_negative_categories
        ):
            issues.append("cost_optimal_panel_contract_invalid")
        if not self.readout_layer or not self.preserve_readout_layer:
            issues.append("scored_readout_not_fixed")
        if not self.candidate_evaluation_fusion:
            issues.append("candidate_evaluation_fusion_disabled")
        if not self.forward_replay_policy:
            issues.append("forward_replay_policy_missing")
        if "exact" not in self.custom_kernel_policy:
            issues.append("custom_kernel_exact_fallback_contract_missing")
        if not self.kernel_backend:
            issues.append("kernel_backend_missing")
        common_calibration_invalid = (
            not self.calibration_enforce_positive_scale
            or self.calibration_iterations != 64
            or abs(self.calibration_min_scale - 1e-8) > 1e-15
            or not self.scalar_target_semantics
        )
        if self.calibration_optimization_contract == (
            SAMPLED_PATH_V1_CALIBRATION_ALGORITHM
        ):
            if (
                common_calibration_invalid
                or abs(self.calibration_learning_rate - 0.25) > 1e-12
                or self.calibration_mode != "c2_shared_diagonal_affine"
            ):
                issues.append("historical_scalar_calibration_contract_invalid")
        elif self.calibration_optimization_contract == (
            PROPER_SCORE_SHRINKAGE_AFFINE_V2
        ):
            if (
                common_calibration_invalid
                or self.calibration_mode != CANONICAL_TARGET_CALIBRATION_MODE
            ):
                issues.append("canonical_proper_score_calibration_contract_invalid")
        else:
            issues.append("calibration_optimization_contract_invalid")
        if abs(self.fully_pruned_anchor_loss - 1.384909) > 1e-12:
            issues.append("fully_pruned_anchor_loss_invalid")
        if self.branch_sign_convention != "nonnegative_positive_negative_strict":
            issues.append("branch_sign_convention_invalid")
        if self.mask_retention_rounding != "ceil_per_layer_per_sign_branch":
            issues.append("mask_retention_rounding_invalid")
        if self.score_tie_policy != "stable_flat_route_order":
            issues.append("score_tie_policy_invalid")
        if not self.layer_group_names or len(set(self.layer_group_names)) != len(
            self.layer_group_names
        ):
            issues.append("layer_groups_empty_or_nonunique")
        if self.reservoir_max_per_class is not None:
            issues.append("reservoir_is_not_full_train_split")
        if self.validation_max_per_class is None or not self.validation_require_exact_per_class:
            issues.append("validation_split_not_exact")
        if self.density_axis_mode not in {
            "target_preservation_layerwise",
            "legacy_old_sparse",
            "old_sparse",
            "legacy_sparse",
        }:
            issues.append("density_axis_mode_not_historical_sampled_path")
        expected_density_values = {
            "density_axis_decay": (self.density_axis_decay, (0.9,)),
            "target_preservation_linear_decay": (
                self.target_preservation_linear_decay,
                (0.85,),
            ),
            # The symmetric conv grid (decay 0.85, floor 1e-4, matching
            # linear) is registered alongside the historical conv grid;
            # any other numeral remains a refusal, not a warning.
            "target_preservation_conv_decay": (
                self.target_preservation_conv_decay,
                (0.9, 0.85),
            ),
            "target_preservation_linear_target_final_density": (
                self.target_preservation_linear_target_final_density,
                (1e-4,),
            ),
            "target_preservation_conv_target_final_density": (
                self.target_preservation_conv_target_final_density,
                (5e-2, 1e-4),
            ),
        }
        for field_name, (observed, allowed) in expected_density_values.items():
            if not any(
                abs(float(observed) - float(expected)) <= 1e-12
                for expected in allowed
            ):
                issues.append(f"{field_name}_not_historical")
        if self.density_schedule_semantics != PRUNING_DENSITY_SCHEDULE_SEMANTICS:
            issues.append("density_schedule_semantics_mismatch")
        if issues:
            raise ValueError("invalid statistical sequential production config: " + ";".join(issues))


def validate_plan_against_statistical_profile(
    plan: CircuitPlan,
    profile: StatisticalSequentialProfile,
) -> None:
    plan.validate(claim_bearing=plan.status == "frozen").require_valid()
    profile.validate()
    issues: list[str] = []
    policy = dict(plan.search_policy)
    if str(policy.get("algorithm", "")) != PRODUCTION_CIRCUIT_SEARCH_ALGORITHM:
        issues.append("plan_algorithm_not_statistical_sequential")
    comparisons = {
        "frontier_policy": profile.frontier_policy,
        "candidate_evaluation_mode": profile.candidate_evaluation_mode,
        "statistical_estimand": profile.statistical_estimand,
        "statistical_sampling_design": profile.statistical_sampling_design,
        "statistical_inference_engine": profile.statistical_inference_engine,
        "statistical_panel_execution": profile.statistical_panel_execution,
        "crossfit_fold_count": profile.crossfit_fold_count,
        "sample_stages_per_class": list(profile.sample_stages_per_class),
        "sensitivity_threshold": profile.sensitivity_tau,
        "denominator_policy": profile.denominator_policy,
        "initial_partition": profile.initial_partition,
        "partition_schedule": profile.partition_schedule,
        "reference_anchor_policy": SAMPLED_PATH_GROUP_REFERENCE_POLICY,
        "calibration_optimization_contract": (
            profile.calibration_optimization_contract
        ),
    }
    for key, expected in comparisons.items():
        observed = policy.get(key)
        if key == "sample_stages_per_class":
            observed = list(observed or ())
        if observed != expected:
            issues.append(f"plan_profile_mismatch:{key}")
    if (
        plan.optimization_view.view_hash == plan.evaluation_view.view_hash
        or plan.optimization_view.ids_hash == plan.evaluation_view.ids_hash
    ):
        issues.append("optimization_and_final_validation_views_overlap")
    production_readout_layer = str(
        plan.notes.get("production_readout_layer_id", "")
    )
    if (
        profile.readout_layer not in set(plan.fixed_measurement_layer_ids)
        and production_readout_layer != profile.readout_layer
    ):
        issues.append("scored_readout_missing_from_fixed_measurement_layers")
    if str(plan.response_distortion.get("metric", "")) not in {
        "crossfit_pos_weighted_bce",
        "per_candidate_affine_calibrated_balanced_bce",
    }:
        issues.append("response_distortion_not_crossfit_pos_weighted_bce")
    if issues:
        raise ValueError("CircuitPlan differs from production profile: " + ";".join(issues))


__all__ = [
    "LEGACY_STATISTICAL_SEQUENTIAL_PROFILE_SCHEMA",
    "PRUNING_DENSITY_SCHEDULE_SEMANTICS",
    "STATISTICAL_SEQUENTIAL_PROFILE_SCHEMA",
    "StatisticalSequentialProfile",
    "validate_plan_against_statistical_profile",
]
