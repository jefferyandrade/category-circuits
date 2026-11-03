"""Untouched final validation for parity-promoted sampled-path circuits."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from contracts.hashing import stable_digest
from mechanistic_systems.contracts import QuerySpec
from mechanistic_systems.evidence import (
    IIDSamplingReceipt,
    iid_physical_executor_identity_hash,
    iid_physical_provider_identity_hash,
)
from mechanistic_systems.measurement.contexts import ContextAxis

from .preservation import balanced_binary_bce
from .sampled_path import SampledComparisonRequest, sampled_path_mask_hash
from .sampled_torch import TorchSampledPathEvaluationBackend
from .torch_evaluator import ReadoutResponse, TorchCircuitEvaluator


NATIVE_SAMPLED_PATH_FINAL_VALIDATION_SCHEMA = (
    "mechanistic_native_sampled_path_final_validation_v1"
)
GLOCAL_NATIVE_SAMPLED_PATH_FINAL_VALIDATION_SCHEMA = (
    "mechanistic_native_sampled_path_final_validation_glocal_v3"
)
GLOCAL_TERMINAL_PARITY_POLICY = "glocal_v3_locked_binary_decision_parity"


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _terminal_validation_lock_hash(
    *,
    plan_hash: str,
    profile_hash: str,
    promotion_receipt_hash: str,
    optimization_axis_hash: str,
    evaluation_axis_hash: str,
    query_ids: Sequence[str],
    selected_mask_hashes: Mapping[str, str],
    calibration_fit_hashes: Mapping[str, Mapping[str, str]],
    parity_tolerance: float,
    terminal_alpha: float,
    terminal_sampling_design: str,
    terminal_population_size: int | None,
    iid_sampling_receipt_hash: str | None,
    calibration_policy: str,
    evaluation_policy: str,
) -> str:
    return stable_digest(
        {
            "policy": GLOCAL_TERMINAL_PARITY_POLICY,
            "plan_hash": str(plan_hash),
            "profile_hash": str(profile_hash),
            "promotion_receipt_hash": str(promotion_receipt_hash),
            "optimization_axis_hash": str(optimization_axis_hash),
            "evaluation_axis_hash": str(evaluation_axis_hash),
            "query_ids": tuple(str(value) for value in query_ids),
            "selected_mask_hashes": {
                str(query_id): str(selected_mask_hashes[str(query_id)])
                for query_id in query_ids
            },
            "calibration_fit_hashes": {
                str(query_id): {
                    str(role): str(value)
                    for role, value in sorted(
                        calibration_fit_hashes[str(query_id)].items()
                    )
                }
                for query_id in query_ids
            },
            "parity_tolerance": float(parity_tolerance),
            "terminal_alpha": float(terminal_alpha),
            "terminal_sampling_design": str(terminal_sampling_design),
            "terminal_population_size": (
                None
                if terminal_population_size is None
                else int(terminal_population_size)
            ),
            "iid_sampling_receipt_hash": (
                None
                if iid_sampling_receipt_hash is None
                else str(iid_sampling_receipt_hash)
            ),
            "calibration_policy": str(calibration_policy),
            "evaluation_policy": str(evaluation_policy),
            "outcomes_visible_at_lock": False,
        }
    )


def _query_class_id(query: QuerySpec) -> int:
    value = query.evaluator.get(
        "class_id",
        query.evaluator.get("unit", query.evaluator.get("coordinate")),
    )
    if value is None:
        raise ValueError("sampled-path final validation requires class queries")
    return int(value)


def _state_accuracy(
    logits: np.ndarray, labels: np.ndarray, class_id: int
) -> dict[str, float]:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    positive = np.asarray(labels).reshape(-1).astype(np.int64, copy=False) == int(
        class_id
    )
    predicted = values >= 0.0
    tp = int(np.count_nonzero(predicted & positive))
    fn = int(np.count_nonzero((~predicted) & positive))
    tn = int(np.count_nonzero((~predicted) & (~positive)))
    fp = int(np.count_nonzero(predicted & (~positive)))
    sensitivity = float(tp / max(tp + fn, 1))
    specificity = float(tn / max(tn + fp, 1))
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "youden_j": sensitivity + specificity - 1.0,
    }


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    result = np.empty_like(values)
    nonnegative = values >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponential = np.exp(values[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return result


def _balanced_brier_rows(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    target_class_id: int,
) -> tuple[np.ndarray, float]:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    positive = np.asarray(labels).reshape(-1).astype(np.int64, copy=False) == int(
        target_class_id
    )
    if values.shape != positive.shape or not np.isfinite(values).all():
        raise ValueError("Brier logits and labels are incompatible")
    if not np.any(positive) or np.all(positive):
        raise ValueError("class-balanced Brier risk requires both binary strata")
    probabilities = _sigmoid(values)
    rows = (probabilities - positive.astype(np.float64)) ** 2
    risk = 0.5 * (float(np.mean(rows[positive])) + float(np.mean(rows[~positive])))
    return rows, risk


def _clopper_pearson_upper(successes: int, count: int, alpha: float) -> float:
    """One-sided exact binomial upper confidence endpoint."""

    x, n = int(successes), int(count)
    if n <= 0 or x < 0 or x > n or not 0.0 < float(alpha) < 1.0:
        raise ValueError("Clopper-Pearson inputs are invalid")
    if x == n:
        return 1.0
    from scipy.stats import beta

    return float(beta.ppf(1.0 - float(alpha), x + 1, n - x))


def _terminal_parity_summary(
    dense_logits: np.ndarray,
    selected_logits: np.ndarray,
    labels: np.ndarray,
    *,
    class_id: int,
    alpha: float,
    sampling_design: str,
) -> dict[str, Any]:
    dense = np.asarray(dense_logits, dtype=np.float64).reshape(-1) >= 0.0
    selected = np.asarray(selected_logits, dtype=np.float64).reshape(-1) >= 0.0
    raw_labels = np.asarray(labels).reshape(-1).astype(np.int64, copy=False)
    if dense.shape != selected.shape or dense.shape != raw_labels.shape:
        raise ValueError("terminal parity axes do not align")
    disagreement = dense != selected
    positive = raw_labels == int(class_id)
    if not np.any(positive) or np.all(positive):
        raise ValueError("terminal parity requires positive and negative strata")
    design = str(sampling_design)
    if design not in {"finite_census", "iid_fixed_n"}:
        raise ValueError(
            "terminal parity sampling_design must be finite_census or iid_fixed_n"
        )
    strata: dict[str, dict[str, Any]] = {}
    upper_values = []
    for name, mask in (("positive", positive), ("negative", ~positive)):
        count = int(np.count_nonzero(mask))
        errors = int(np.count_nonzero(disagreement[mask]))
        rate = float(errors / count)
        upper = (
            rate
            if design == "finite_census"
            else _clopper_pearson_upper(errors, count, float(alpha) / 2.0)
        )
        strata[name] = {
            "count": count,
            "disagreements": errors,
            "rate": rate,
            "upper": upper,
            "weight": 0.5,
        }
        upper_values.append(upper)
    return {
        "decision_rule_id": "calibrated_target_logit_ge_zero",
        "sampling_design": design,
        "alpha": float(alpha),
        "disagreement_count": int(np.count_nonzero(disagreement)),
        "context_count": int(disagreement.size),
        "unweighted_disagreement_rate": float(np.mean(disagreement)),
        "balanced_disagreement_rate": 0.5
        * (strata["positive"]["rate"] + strata["negative"]["rate"]),
        "balanced_disagreement_upper": 0.5 * sum(upper_values),
        "strata": strata,
    }


@dataclass(frozen=True)
class NativeSampledPathFinalValidationReceipt:
    """Content-addressed proof that final masks saw only untouched outcomes."""

    plan_hash: str
    profile_hash: str
    promotion_receipt_hash: str
    optimization_axis_hash: str
    evaluation_axis_hash: str
    query_ids: tuple[str, ...]
    rows: tuple[Mapping[str, Any], ...]
    issues: tuple[str, ...] = tuple()
    calibration_policy: str = "state_specific_full_sampled_optimization_panel"
    evaluation_policy: str = "frozen_affine_untouched_full_validation_axis"
    terminal_alpha: float | None = None
    parity_tolerance: float | None = None
    parity_policy: str = "unconfigured_shadow"
    terminal_sampling_design: str | None = None
    terminal_population_size: int | None = None
    iid_sampling_receipt: IIDSamplingReceipt | None = None
    validation_lock_hash: str | None = None
    validation_lock_open_count: int | None = None
    schema_name: str = NATIVE_SAMPLED_PATH_FINAL_VALIDATION_SCHEMA

    @property
    def passed(self) -> bool:
        return not self.issues

    @property
    def preservation_by_query(self) -> dict[str, float]:
        return {
            str(row["query_id"]): float(row["target_preservation"]) for row in self.rows
        }

    @property
    def parity_configured(self) -> bool:
        return self.parity_tolerance is not None

    @property
    def parity_passed(self) -> bool:
        return bool(
            self.parity_configured
            and not self.issues
            and all(bool(row.get("terminal_parity_passed", False)) for row in self.rows)
        )

    @property
    def glocal_claim_ready(self) -> bool:
        return bool(self.passed and self.parity_passed)

    def validate(self) -> None:
        if self.schema_name not in {
            NATIVE_SAMPLED_PATH_FINAL_VALIDATION_SCHEMA,
            GLOCAL_NATIVE_SAMPLED_PATH_FINAL_VALIDATION_SCHEMA,
        }:
            raise ValueError("unsupported sampled-path final validation schema")
        hashes = (
            self.plan_hash,
            self.profile_hash,
            self.promotion_receipt_hash,
            self.optimization_axis_hash,
            self.evaluation_axis_hash,
        )
        if any(not _is_sha256(value) for value in hashes):
            raise ValueError("sampled-path final validation requires content hashes")
        if not self.query_ids or len(set(self.query_ids)) != len(self.query_ids):
            raise ValueError("sampled-path final validation query axis is invalid")
        if tuple(str(row.get("query_id", "")) for row in self.rows) != self.query_ids:
            raise ValueError(
                "sampled-path final validation rows differ from query axis"
            )
        for row in self.rows:
            if int(row.get("evaluation_context_count", 0)) <= 1:
                raise ValueError("sampled-path final validation axis is empty")
            for key in (
                "dense_validation_canonical_loss",
                "selected_validation_canonical_loss",
                "fully_pruned_validation_canonical_loss",
                "selected_validation_canonical_loss_delta_from_unpruned",
                "selected_route_density",
            ):
                if not math.isfinite(float(row.get(key, float("nan")))):
                    raise ValueError(
                        f"sampled-path final validation {key} is non-finite"
                    )
            calibrations = dict(row.get("calibrations", {}) or {})
            if set(calibrations) != {"dense", "selected", "fully_pruned"}:
                raise ValueError(
                    "sampled-path final validation calibration states differ"
                )
            panel_hashes = {
                str(dict(value).get("panel_hash", ""))
                for value in calibrations.values()
            }
            if len(panel_hashes) != 1 or len(next(iter(panel_hashes), "")) != 64:
                raise ValueError(
                    "final validation states must use one common calibration panel"
                )
            if any(
                bool(dict(value).get("validation_refit", True))
                for value in calibrations.values()
            ):
                raise ValueError("untouched final validation may not refit calibration")
            if self.parity_configured:
                for key in (
                    "dense_validation_balanced_brier",
                    "selected_validation_balanced_brier",
                    "selected_validation_balanced_brier_delta",
                    "terminal_balanced_decision_disagreement",
                    "terminal_balanced_decision_disagreement_upper",
                ):
                    if not math.isfinite(float(row.get(key, float("nan")))):
                        raise ValueError(
                            f"sampled-path final validation {key} is non-finite"
                        )
                if bool(row.get("terminal_parity_passed", False)) != (
                    float(row["terminal_balanced_decision_disagreement_upper"])
                    <= float(self.parity_tolerance) + 1e-15
                ):
                    raise ValueError("terminal parity verdict disagrees with its bound")
        if self.parity_configured:
            if self.schema_name != GLOCAL_NATIVE_SAMPLED_PATH_FINAL_VALIDATION_SCHEMA:
                raise ValueError("configured glocal parity requires the glocal schema")
            if (
                self.terminal_alpha is None
                or not 0.0 < float(self.terminal_alpha) < 1.0
                or not 0.0 <= float(self.parity_tolerance) <= 1.0
            ):
                raise ValueError(
                    "configured terminal parity has invalid alpha/tolerance"
                )
            if self.parity_policy != GLOCAL_TERMINAL_PARITY_POLICY:
                raise ValueError(
                    "configured terminal parity uses an unsupported policy"
                )
            if self.terminal_sampling_design not in {"finite_census", "iid_fixed_n"}:
                raise ValueError(
                    "configured terminal parity requires a sampling design"
                )
            if self.terminal_sampling_design == "finite_census":
                if self.terminal_population_size is None or any(
                    int(self.terminal_population_size)
                    != int(row["evaluation_context_count"])
                    for row in self.rows
                ):
                    raise ValueError(
                        "finite-census terminal parity requires the declared population size"
                    )
                if self.iid_sampling_receipt is not None:
                    raise ValueError(
                        "finite-census terminal parity cannot carry an IID receipt"
                    )
            else:
                if self.terminal_population_size is not None:
                    raise ValueError(
                        "iid terminal parity may not declare a census size"
                    )
                if not isinstance(
                    self.iid_sampling_receipt,
                    IIDSamplingReceipt,
                ):
                    raise ValueError(
                        "iid terminal parity requires a typed sampling receipt"
                    )
                expected_draw_ids_hash = stable_digest(
                    {"draw_ids": self.iid_sampling_receipt.draw_ids}
                )
                if any(
                    int(row["evaluation_context_count"])
                    != len(self.iid_sampling_receipt.draws)
                    or str(row.get("terminal_draw_ids_hash", ""))
                    != expected_draw_ids_hash
                    or str(row.get("iid_support_distribution_hash", ""))
                    != self.iid_sampling_receipt.support_distribution_hash
                    or not _is_sha256(
                        row.get("terminal_iid_physical_evidence_hash", "")
                    )
                    for row in self.rows
                ):
                    raise ValueError(
                        "iid terminal rows differ from their physical draw contract"
                    )
            if not _is_sha256(self.validation_lock_hash):
                raise ValueError(
                    "configured terminal parity requires a validation lock"
                )
            if self.validation_lock_open_count != 1:
                raise ValueError("terminal validation lock must be opened exactly once")
            selected_mask_hashes: dict[str, str] = {}
            calibration_fit_hashes: dict[str, dict[str, str]] = {}
            for row in self.rows:
                query_id = str(row["query_id"])
                selected_mask_hash = str(row.get("selected_mask_hash", ""))
                if not _is_sha256(selected_mask_hash):
                    raise ValueError("terminal validation row has an invalid mask hash")
                selected_mask_hashes[query_id] = selected_mask_hash
                if row.get("terminal_audit_visible_to_search") is not False:
                    raise ValueError("terminal audit was visible to adaptive search")
                terminal = dict(row.get("terminal_parity", {}) or {})
                if terminal.get(
                    "sampling_design"
                ) != self.terminal_sampling_design or not math.isclose(
                    float(terminal.get("alpha", float("nan"))),
                    float(self.terminal_alpha),
                    abs_tol=1e-15,
                ):
                    raise ValueError(
                        "terminal row differs from the locked audit design"
                    )
                calibration_fit_hashes[query_id] = {
                    str(role): stable_digest(dict(metadata))
                    for role, metadata in dict(row["calibrations"]).items()
                }
            expected_lock_hash = _terminal_validation_lock_hash(
                plan_hash=self.plan_hash,
                profile_hash=self.profile_hash,
                promotion_receipt_hash=self.promotion_receipt_hash,
                optimization_axis_hash=self.optimization_axis_hash,
                evaluation_axis_hash=self.evaluation_axis_hash,
                query_ids=self.query_ids,
                selected_mask_hashes=selected_mask_hashes,
                calibration_fit_hashes=calibration_fit_hashes,
                parity_tolerance=float(self.parity_tolerance),
                terminal_alpha=float(self.terminal_alpha),
                terminal_sampling_design=self.terminal_sampling_design,
                terminal_population_size=self.terminal_population_size,
                iid_sampling_receipt_hash=(
                    None
                    if self.iid_sampling_receipt is None
                    else self.iid_sampling_receipt.content_hash
                ),
                calibration_policy=self.calibration_policy,
                evaluation_policy=self.evaluation_policy,
            )
            if self.validation_lock_hash != expected_lock_hash:
                raise ValueError(
                    "terminal validation lock differs from the locked state and audit coordinates"
                )
        else:
            if self.schema_name != NATIVE_SAMPLED_PATH_FINAL_VALIDATION_SCHEMA:
                raise ValueError(
                    "unconfigured parity must use the legacy validation schema"
                )
            if any(
                value is not None
                for value in (
                    self.terminal_alpha,
                    self.terminal_sampling_design,
                    self.terminal_population_size,
                    self.iid_sampling_receipt,
                    self.validation_lock_hash,
                    self.validation_lock_open_count,
                )
            ):
                raise ValueError(
                    "legacy final validation may not carry glocal lock fields"
                )
            if self.parity_policy != "unconfigured_shadow":
                raise ValueError("legacy final validation has an invalid parity policy")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = {
            "schema_name": self.schema_name,
            "plan_hash": self.plan_hash,
            "profile_hash": self.profile_hash,
            "promotion_receipt_hash": self.promotion_receipt_hash,
            "optimization_axis_hash": self.optimization_axis_hash,
            "evaluation_axis_hash": self.evaluation_axis_hash,
            "query_ids": list(self.query_ids),
            "rows": [dict(row) for row in self.rows],
            "issues": list(self.issues),
            "passed": self.passed,
            "calibration_policy": self.calibration_policy,
            "evaluation_policy": self.evaluation_policy,
        }
        if self.parity_configured:
            payload.update(
                {
                    "parity_configured": True,
                    "parity_passed": self.parity_passed,
                    "glocal_claim_ready": self.glocal_claim_ready,
                    "terminal_alpha": self.terminal_alpha,
                    "parity_tolerance": self.parity_tolerance,
                    "parity_policy": self.parity_policy,
                    "terminal_sampling_design": self.terminal_sampling_design,
                    "terminal_population_size": self.terminal_population_size,
                    "iid_sampling_receipt": (
                        None
                        if self.iid_sampling_receipt is None
                        else self.iid_sampling_receipt.to_dict()
                    ),
                    "validation_lock_hash": self.validation_lock_hash,
                    "validation_lock_open_count": self.validation_lock_open_count,
                }
            )
        return payload

    @property
    def content_hash(self) -> str:
        return stable_digest(self.to_dict())

    def require_passed(
        self,
        *,
        plan_hash: str | None = None,
        profile_hash: str | None = None,
        promotion_receipt_hash: str | None = None,
        query_ids: Sequence[str] | None = None,
    ) -> None:
        self.validate()
        if not self.passed:
            raise ValueError(
                "sampled-path final validation failed: " + ";".join(self.issues)
            )
        expected = (
            ("plan", plan_hash, self.plan_hash),
            ("profile", profile_hash, self.profile_hash),
            (
                "promotion receipt",
                promotion_receipt_hash,
                self.promotion_receipt_hash,
            ),
        )
        for label, requested, observed in expected:
            if requested is not None and str(requested) != observed:
                raise ValueError(
                    f"final validation {label} differs from circuit extraction"
                )
        if (
            query_ids is not None
            and tuple(str(value) for value in query_ids) != self.query_ids
        ):
            raise ValueError(
                "final validation query axis differs from circuit extraction"
            )

    def require_glocal_claim_ready(self, **expected: Any) -> None:
        self.require_passed(**expected)
        if not self.parity_configured:
            raise ValueError("glocal terminal parity tolerance was not configured")
        if not self.parity_passed:
            raise ValueError("glocal terminal parity certificate did not pass")

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "NativeSampledPathFinalValidationReceipt":
        schema_name = str(payload.get("schema_name", ""))
        if schema_name not in {
            NATIVE_SAMPLED_PATH_FINAL_VALIDATION_SCHEMA,
            GLOCAL_NATIVE_SAMPLED_PATH_FINAL_VALIDATION_SCHEMA,
        }:
            raise ValueError("unsupported sampled-path final validation schema")
        receipt = cls(
            plan_hash=str(payload.get("plan_hash", "")),
            profile_hash=str(payload.get("profile_hash", "")),
            promotion_receipt_hash=str(payload.get("promotion_receipt_hash", "")),
            optimization_axis_hash=str(payload.get("optimization_axis_hash", "")),
            evaluation_axis_hash=str(payload.get("evaluation_axis_hash", "")),
            query_ids=tuple(str(value) for value in payload.get("query_ids", ())),
            rows=tuple(dict(row) for row in payload.get("rows", ())),
            issues=tuple(str(value) for value in payload.get("issues", ())),
            calibration_policy=str(payload.get("calibration_policy", "")),
            evaluation_policy=str(payload.get("evaluation_policy", "")),
            terminal_alpha=(
                None
                if payload.get("terminal_alpha") is None
                else float(payload["terminal_alpha"])
            ),
            parity_tolerance=(
                None
                if payload.get("parity_tolerance") is None
                else float(payload["parity_tolerance"])
            ),
            parity_policy=str(payload.get("parity_policy", "unconfigured_shadow")),
            terminal_sampling_design=(
                None
                if payload.get("terminal_sampling_design") is None
                else str(payload["terminal_sampling_design"])
            ),
            terminal_population_size=(
                None
                if payload.get("terminal_population_size") is None
                else int(payload["terminal_population_size"])
            ),
            iid_sampling_receipt=(
                None
                if payload.get("iid_sampling_receipt") is None
                else IIDSamplingReceipt.from_mapping(
                    dict(payload["iid_sampling_receipt"])
                )
            ),
            validation_lock_hash=(
                None
                if payload.get("validation_lock_hash") is None
                else str(payload["validation_lock_hash"])
            ),
            validation_lock_open_count=(
                None
                if payload.get("validation_lock_open_count") is None
                else int(payload["validation_lock_open_count"])
            ),
            schema_name=schema_name,
        )
        receipt.validate()
        if bool(payload.get("passed", receipt.passed)) != receipt.passed:
            raise ValueError("sampled-path final validation pass flag is inconsistent")
        return receipt


def _execute_readout(
    evaluator: TorchCircuitEvaluator,
    source: Any,
    axis: ContextAxis,
    masks: Mapping[str, np.ndarray],
    *,
    iid_sampling_receipt: IIDSamplingReceipt | None = None,
) -> ReadoutResponse:
    if iid_sampling_receipt is None:
        physical_context_ids = axis.context_ids
        draw_take = np.arange(len(physical_context_ids), dtype=np.int64)
    else:
        if iid_sampling_receipt.draw_ids != axis.context_ids:
            raise ValueError(
                "IID evaluation axis must enumerate the unique realized draw IDs"
            )
        physical_context_ids = iid_sampling_receipt.unique_source_unit_ids
        source_position = {
            source_unit_id: position
            for position, source_unit_id in enumerate(physical_context_ids)
        }
        draw_take = np.asarray(
            [
                source_position[source_unit_id]
                for source_unit_id in iid_sampling_receipt.source_unit_ids
            ],
            dtype=np.int64,
        )
    response = evaluator.readout_response_from_batches(
        source.panel_batches(physical_context_ids),
        compiled_mask=evaluator.compile_mask(masks),
    )
    if response.context_ids != physical_context_ids:
        raise ValueError(
            "final validation execution changed its frozen physical source axis"
        )
    expanded_labels = np.asarray(response.labels, dtype=np.int64)[draw_take]
    expected_labels = np.asarray(
        [record.label for record in axis.records], dtype=np.int64
    )
    if not np.array_equal(expanded_labels, expected_labels):
        raise ValueError("final validation execution changed its frozen context axis")
    if iid_sampling_receipt is not None:
        return ReadoutResponse(
            context_ids=axis.context_ids,
            values=np.asarray(response.values)[draw_take],
            labels=expanded_labels,
        )
    return response


def validate_native_sampled_path_final_masks(
    *,
    plan_hash: str,
    profile_hash: str,
    promotion_receipt_hash: str,
    backend: TorchSampledPathEvaluationBackend,
    evaluator: TorchCircuitEvaluator,
    queries: Sequence[QuerySpec],
    selected_masks_by_query: Mapping[str, Mapping[str, np.ndarray]],
    optimization_axis: ContextAxis,
    evaluation_axis: ContextAxis,
    evaluation_source: Any,
    sample_stage_per_class: int,
    fold_count: int,
    random_seed: int,
    parity_tolerance: float | None = None,
    terminal_alpha: float = 0.01,
    terminal_sampling_design: str | None = None,
    terminal_population_size: int | None = None,
    iid_sampling_receipt: IIDSamplingReceipt | None = None,
) -> NativeSampledPathFinalValidationReceipt:
    """Fit state-specific extraction affines and score untouched outcomes once."""

    query_axis = tuple(queries)
    query_ids = tuple(query.query_id for query in query_axis)
    if tuple(str(value) for value in selected_masks_by_query) != query_ids:
        raise ValueError("final validation masks differ from the ordered query axis")
    optimization_axis.validate()
    evaluation_axis.validate()
    if parity_tolerance is None:
        if (
            terminal_sampling_design is not None
            or terminal_population_size is not None
            or iid_sampling_receipt is not None
        ):
            raise ValueError(
                "terminal audit design is only legal with an explicit parity tolerance"
            )
    else:
        if terminal_sampling_design not in {"finite_census", "iid_fixed_n"}:
            raise ValueError(
                "configured terminal parity requires an explicit finite_census "
                "or iid_fixed_n sampling design"
            )
        if terminal_sampling_design == "finite_census":
            if terminal_population_size is None or int(terminal_population_size) != len(
                evaluation_axis.records
            ):
                raise ValueError(
                    "finite_census requires an explicit population size equal "
                    "to the complete evaluation axis"
                )
            if iid_sampling_receipt is not None:
                raise ValueError(
                    "finite_census terminal parity cannot carry an IID receipt"
                )
        else:
            if terminal_population_size is not None:
                raise ValueError(
                    "iid_fixed_n terminal parity may not declare a census size"
                )
            if not isinstance(iid_sampling_receipt, IIDSamplingReceipt):
                raise ValueError(
                    "iid_fixed_n terminal parity requires a typed sampling receipt"
                )
            if iid_sampling_receipt.draw_ids != evaluation_axis.context_ids:
                raise ValueError(
                    "iid sampling receipt differs from the unique evaluation draw axis"
                )
            selector = dict(evaluation_axis.selector)
            if (
                str(selector.get("iid_context_population_hash", ""))
                != iid_sampling_receipt.context_population_hash
                or str(selector.get("iid_support_distribution_hash", ""))
                != iid_sampling_receipt.support_distribution_hash
            ):
                raise ValueError(
                    "iid evaluation axis differs from its frozen population/risk support"
                )
            if (
                iid_physical_provider_identity_hash(evaluation_source)
                != iid_sampling_receipt.provider_identity_hash
            ):
                raise ValueError(
                    "iid sampling receipt provider identity differs from the "
                    "physical evaluation source"
                )
            if (
                iid_physical_executor_identity_hash(backend)
                != iid_sampling_receipt.executor_identity_hash
            ):
                raise ValueError(
                    "iid sampling receipt executor identity differs from the "
                    "physical evaluation backend"
                )
            source_axis = getattr(evaluation_source, "context_axis", None)
            if source_axis is None:
                source_axis = getattr(evaluation_source, "axis", None)
            source_axis_ids = set(getattr(source_axis, "context_ids", ()))
            if set(iid_sampling_receipt.support_source_unit_ids) - source_axis_ids:
                raise ValueError(
                    "iid sampling receipt references source units unavailable from "
                    "the physical provider"
                )
    if optimization_axis.content_hash == evaluation_axis.content_hash:
        raise ValueError("final validation must be disjoint from circuit extraction")
    context_overlap = set(optimization_axis.context_ids).intersection(
        evaluation_axis.context_ids
    )
    if iid_sampling_receipt is not None:
        context_overlap.update(
            set(optimization_axis.context_ids).intersection(
                iid_sampling_receipt.unique_source_unit_ids
            )
        )
    if context_overlap:
        preview = ",".join(sorted(context_overlap)[:3])
        raise ValueError(
            "final validation context IDs overlap circuit extraction"
            + (f": {preview}" if preview else "")
        )
    expected_layers: tuple[str, ...] | None = None
    dense_masks: dict[str, np.ndarray] | None = None
    fully_pruned_masks: dict[str, np.ndarray] | None = None
    normalized_masks: dict[str, dict[str, np.ndarray]] = {}
    for query_id in query_ids:
        selected = {
            str(layer): np.asarray(mask, dtype=bool)
            for layer, mask in selected_masks_by_query[query_id].items()
        }
        layer_axis = tuple(selected)
        if expected_layers is None:
            expected_layers = layer_axis
            dense_masks = {
                layer: np.ones_like(mask, dtype=bool)
                for layer, mask in selected.items()
            }
            fully_pruned_masks = {
                layer: np.zeros_like(mask, dtype=bool)
                for layer, mask in selected.items()
            }
        elif layer_axis != expected_layers:
            raise ValueError("final validation masks do not share one route-block axis")
        normalized_masks[query_id] = selected
    assert dense_masks is not None and fully_pruned_masks is not None
    calibration_policy = "state_specific_full_sampled_optimization_panel"
    evaluation_policy = "frozen_affine_untouched_full_validation_axis"
    reference_calibrations: dict[
        str,
        tuple[
            Any,
            Any,
            Any,
        ],
    ] = {}
    for query in query_axis:
        query_id = query.query_id
        request = SampledComparisonRequest(
            query_id=query_id,
            layer_id="__final_validation__",
            partition_id="reference",
            previous_density=1.0,
            proposed_density=1.0,
            stage_index=0,
            stage_per_class=int(sample_stage_per_class),
            proposal_index=0,
            random_seed=int(random_seed),
            fold_count=int(fold_count),
        )
        dense_calibration = backend.fit_reference_calibration(
            dense_masks, request, role="dense"
        )
        panel = dense_calibration.panel
        selected_calibration = backend.fit_reference_calibration(
            normalized_masks[query_id],
            request,
            role="selected",
            panel=panel,
        )
        fully_calibration = backend.fit_reference_calibration(
            fully_pruned_masks,
            request,
            role="fully_pruned",
            panel=panel,
        )
        reference_calibrations[query_id] = (
            dense_calibration,
            selected_calibration,
            fully_calibration,
        )
    selected_mask_hashes = {
        query_id: sampled_path_mask_hash(normalized_masks[query_id])
        for query_id in query_ids
    }
    calibration_fit_hashes = {
        query_id: {
            calibration.role: stable_digest(dict(calibration.metadata))
            for calibration in reference_calibrations[query_id]
        }
        for query_id in query_ids
    }
    validation_lock_hash = (
        _terminal_validation_lock_hash(
            plan_hash=str(plan_hash),
            profile_hash=str(profile_hash),
            promotion_receipt_hash=str(promotion_receipt_hash),
            optimization_axis_hash=optimization_axis.content_hash,
            evaluation_axis_hash=evaluation_axis.content_hash,
            query_ids=query_ids,
            selected_mask_hashes=selected_mask_hashes,
            calibration_fit_hashes=calibration_fit_hashes,
            parity_tolerance=float(parity_tolerance),
            terminal_alpha=float(terminal_alpha),
            terminal_sampling_design=str(terminal_sampling_design),
            terminal_population_size=terminal_population_size,
            iid_sampling_receipt_hash=(
                None
                if iid_sampling_receipt is None
                else iid_sampling_receipt.content_hash
            ),
            calibration_policy=calibration_policy,
            evaluation_policy=evaluation_policy,
        )
        if parity_tolerance is not None
        else None
    )

    dense_response = _execute_readout(
        evaluator,
        evaluation_source,
        evaluation_axis,
        dense_masks,
        iid_sampling_receipt=iid_sampling_receipt,
    )
    fully_response = _execute_readout(
        evaluator,
        evaluation_source,
        evaluation_axis,
        fully_pruned_masks,
        iid_sampling_receipt=iid_sampling_receipt,
    )
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    for query in query_axis:
        query_id = query.query_id
        selected_masks = normalized_masks[query_id]
        (
            dense_calibration,
            selected_calibration,
            fully_calibration,
        ) = reference_calibrations[query_id]
        panel = dense_calibration.panel
        selected_response = _execute_readout(
            evaluator,
            evaluation_source,
            evaluation_axis,
            selected_masks,
            iid_sampling_receipt=iid_sampling_receipt,
        )
        class_id = _query_class_id(query)
        dense_logits = backend.calibrated_target_logits(
            dense_response, dense_calibration
        )
        selected_logits = backend.calibrated_target_logits(
            selected_response, selected_calibration
        )
        fully_logits = backend.calibrated_target_logits(
            fully_response, fully_calibration
        )
        labels = np.asarray(dense_response.labels, dtype=np.int64)
        dense_loss = balanced_binary_bce(dense_logits, labels, target_class_id=class_id)
        selected_loss = balanced_binary_bce(
            selected_logits, labels, target_class_id=class_id
        )
        fully_loss = balanced_binary_bce(fully_logits, labels, target_class_id=class_id)
        _dense_brier_rows, dense_brier = _balanced_brier_rows(
            dense_logits, labels, target_class_id=class_id
        )
        _selected_brier_rows, selected_brier = _balanced_brier_rows(
            selected_logits, labels, target_class_id=class_id
        )
        _fully_brier_rows, fully_brier = _balanced_brier_rows(
            fully_logits, labels, target_class_id=class_id
        )
        parity = (
            _terminal_parity_summary(
                dense_logits,
                selected_logits,
                labels,
                class_id=class_id,
                alpha=float(terminal_alpha),
                sampling_design=str(terminal_sampling_design),
            )
            if parity_tolerance is not None
            else None
        )
        parity_passed = bool(
            parity is not None
            and float(parity["balanced_disagreement_upper"])
            <= float(parity_tolerance) + 1e-15
        )
        denominator = float(fully_loss - dense_loss)
        raw_delta = float(selected_loss - dense_loss)
        normalized_damage = (
            float(raw_delta / denominator) if denominator > 1e-12 else None
        )
        target_preservation = (
            float(1.0 - normalized_damage)
            if normalized_damage is not None
            else float("nan")
        )
        if denominator <= 1e-12:
            issues.append(f"{query_id}:dense_readout_not_better_than_fully_pruned")
        if parity_tolerance is not None and not parity_passed:
            issues.append(f"{query_id}:terminal_decision_parity_exceeds_tolerance")
        retained = sum(int(np.count_nonzero(mask)) for mask in selected_masks.values())
        total = sum(int(mask.size) for mask in selected_masks.values())
        row = {
            "query_id": query_id,
            "class_id": class_id,
            "optimization_context_count": len(panel.context_ids),
            "evaluation_context_count": len(evaluation_axis.records),
            "dense_mask_hash": sampled_path_mask_hash(dense_masks),
            "selected_mask_hash": sampled_path_mask_hash(selected_masks),
            "fully_pruned_mask_hash": sampled_path_mask_hash(fully_pruned_masks),
            "selected_retained_route_count": retained,
            "route_universe_count": total,
            "selected_route_density": float(retained / max(total, 1)),
            "dense_validation_canonical_loss": float(dense_loss),
            "selected_validation_canonical_loss": float(selected_loss),
            "fully_pruned_validation_canonical_loss": float(fully_loss),
            "selected_validation_canonical_loss_delta_from_unpruned": raw_delta,
            "selected_validation_damage_denominator": denominator,
            "selected_validation_normalized_bce_damage": normalized_damage,
            "target_preservation": target_preservation,
            "target_readout_baseline_valid": bool(denominator > 1e-12),
            "dense_accuracy": _state_accuracy(dense_logits, labels, class_id),
            "selected_accuracy": _state_accuracy(selected_logits, labels, class_id),
            "fully_pruned_accuracy": _state_accuracy(fully_logits, labels, class_id),
            "calibrations": {
                "dense": dict(dense_calibration.metadata),
                "selected": dict(selected_calibration.metadata),
                "fully_pruned": dict(fully_calibration.metadata),
            },
        }
        if parity_tolerance is not None:
            row.update(
                {
                    "dense_validation_balanced_brier": float(dense_brier),
                    "selected_validation_balanced_brier": float(selected_brier),
                    "fully_pruned_validation_balanced_brier": float(fully_brier),
                    "selected_validation_balanced_brier_delta": float(
                        selected_brier - dense_brier
                    ),
                    "paired_brier_delta_range": [-1.0, 1.0],
                    "terminal_balanced_decision_disagreement": float(
                        parity["balanced_disagreement_rate"]  # type: ignore[index]
                    ),
                    "terminal_balanced_decision_disagreement_upper": float(
                        parity["balanced_disagreement_upper"]  # type: ignore[index]
                    ),
                    "terminal_parity_passed": bool(parity_passed),
                    "terminal_parity": parity,
                    "terminal_audit_visible_to_search": False,
                    **(
                        {
                            "terminal_draw_ids_hash": stable_digest(
                                {"draw_ids": iid_sampling_receipt.draw_ids}
                            ),
                            "iid_support_distribution_hash": (
                                iid_sampling_receipt.support_distribution_hash
                            ),
                            "terminal_iid_physical_evidence_hash": stable_digest(
                                {
                                    "schema_name": (
                                        "native_iid_terminal_physical_evidence_v1"
                                    ),
                                    "iid_sampling_receipt_hash": (
                                        iid_sampling_receipt.content_hash
                                    ),
                                    "draw_ids": iid_sampling_receipt.draw_ids,
                                    "source_unit_ids": (
                                        iid_sampling_receipt.source_unit_ids
                                    ),
                                    "labels": labels.tolist(),
                                    "dense_decisions": (
                                        np.asarray(dense_logits) >= 0.0
                                    ).tolist(),
                                    "selected_decisions": (
                                        np.asarray(selected_logits) >= 0.0
                                    ).tolist(),
                                }
                            ),
                        }
                        if iid_sampling_receipt is not None
                        else {}
                    ),
                }
            )
        rows.append(row)
    receipt = NativeSampledPathFinalValidationReceipt(
        plan_hash=str(plan_hash),
        profile_hash=str(profile_hash),
        promotion_receipt_hash=str(promotion_receipt_hash),
        optimization_axis_hash=optimization_axis.content_hash,
        evaluation_axis_hash=evaluation_axis.content_hash,
        query_ids=query_ids,
        rows=tuple(rows),
        issues=tuple(issues),
        terminal_alpha=(
            float(terminal_alpha) if parity_tolerance is not None else None
        ),
        parity_tolerance=(
            float(parity_tolerance) if parity_tolerance is not None else None
        ),
        parity_policy=(
            GLOCAL_TERMINAL_PARITY_POLICY
            if parity_tolerance is not None
            else "unconfigured_shadow"
        ),
        terminal_sampling_design=(
            str(terminal_sampling_design) if parity_tolerance is not None else None
        ),
        terminal_population_size=(
            int(terminal_population_size)
            if parity_tolerance is not None and terminal_population_size is not None
            else None
        ),
        iid_sampling_receipt=(
            iid_sampling_receipt if parity_tolerance is not None else None
        ),
        validation_lock_hash=validation_lock_hash,
        validation_lock_open_count=(1 if parity_tolerance is not None else None),
        calibration_policy=calibration_policy,
        evaluation_policy=evaluation_policy,
        schema_name=(
            GLOCAL_NATIVE_SAMPLED_PATH_FINAL_VALIDATION_SCHEMA
            if parity_tolerance is not None
            else NATIVE_SAMPLED_PATH_FINAL_VALIDATION_SCHEMA
        ),
    )
    receipt.validate()
    return receipt


__all__ = [
    "GLOCAL_TERMINAL_PARITY_POLICY",
    "GLOCAL_NATIVE_SAMPLED_PATH_FINAL_VALIDATION_SCHEMA",
    "NATIVE_SAMPLED_PATH_FINAL_VALIDATION_SCHEMA",
    "NativeSampledPathFinalValidationReceipt",
    "validate_native_sampled_path_final_masks",
]
