"""Immutable metric-budget bindings for the tractable v1 raw-logit controller.

The D/selection radius is a local parent-relative raw target-logit half-energy
radius.  This module freezes how that radius is registered before the
trajectory begins.  It also records the separate final dense-relative RMS gate
without evaluating or authorizing that gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from ..modern_v1_physical_metric import BALANCED_MEASURE

SCHEMA = "tractable_v1_raw_logit_metric_budget_binding_v2"
HASH_PREFIX = "tractable-v1-raw-logit-metric-budget-v2-"
AUTHORITY = "diagnostic_trust_radius_binding_only_no_loss_or_commit_authority"
SOURCE_EXPLICIT_RAW = "explicit_raw_logit_rms_radius"
SOURCE_V0_BCE = "v0_bce_conversion"
LOSS_NORMALIZATION_V0_POS_WEIGHT = "v0_pos_weight_mean_registered_counts_against_balanced_measure"
LOSS_NORMALIZATION_NOT_USED = "not_used_explicit_raw_logit_radius"
BUDGET_EPOCH = "pretrajectory_metric_budget_epoch_before_D_selection"
FINAL_DENSE_RMS_METRIC = "function_preservation.rms_logit_displacement"
FINAL_DENSE_REFERENCE = "all_live_routes_original_loaded_layer_and_readout_biases"
FINAL_GATE_REGISTERED = "registered_final_dense_relative_raw_logit_rms_ball"
FINAL_GATE_OPEN = "unregistered_final_dense_relative_rms_gate_open_no_function_ball_claim"


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def stable_payload_hash(payload: Mapping[str, Any], *, prefix: str = "tractable-v1-payload-") -> str:
    return prefix + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _finite_positive(value: Any, *, name: str) -> float:
    if type(value) is bool:
        raise ValueError(f"{name} must be finite and positive")
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return out


def _finite_or_none(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    if type(value) is bool:
        raise ValueError(f"{name} must be finite")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _nonnegative_int_or_none(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _positive_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _count_pair_or_none(values: Sequence[int] | None, *, name: str) -> tuple[int, int] | None:
    if values is None:
        return None
    raw = tuple(values)
    if len(raw) != 2:
        raise ValueError(f"{name} must contain target and complement counts")
    return (_positive_int(raw[0], name=f"{name} target"), _positive_int(raw[1], name=f"{name} complement"))


def _nonnegative_count_pair(values: Sequence[int], *, name: str) -> tuple[int, int]:
    raw = tuple(values)
    if len(raw) != 2:
        raise ValueError(f"{name} must contain target and complement counts")
    out = []
    for value in raw:
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} counts must be nonnegative integers")
        out.append(value)
    return (out[0], out[1])


def v0_loss_normalization_fraction(target_count: int, complement_count: int) -> Fraction:
    nt = _positive_int(target_count, name="normalization target count")
    nc = _positive_int(complement_count, name="normalization complement count")
    return Fraction(2 * nc, nt + nc)


def v0_loss_normalization(target_count: int, complement_count: int) -> float:
    """Exact v0 pos-weight-mean factor relative to the balanced two-stratum mean."""

    return float(v0_loss_normalization_fraction(target_count, complement_count))


@dataclass(frozen=True, slots=True)
class DenseRelativeFunctionBallRequirement:
    """Separate final dense-relative raw-logit RMS requirement.

    This requirement is recorded at budget time so the trajectory cannot later
    advertise a scientific function-ball claim without a named final replay
    gate.  It is not evaluated by the local D trust-radius binding.
    """

    rms_radius: float | None
    requirement_kind: str = FINAL_GATE_OPEN
    metric: str = FINAL_DENSE_RMS_METRIC
    dense_reference: str = FINAL_DENSE_REFERENCE
    measure: str = BALANCED_MEASURE
    opens_after_train_frontier_freeze: bool = True
    not_a_bce_ceiling: bool = True
    evaluated_by_metric_budget_binding: bool = False
    terminal_validation_authority: bool = False
    production_authority: bool = False
    schema: str = "tractable_v1_final_dense_relative_function_ball_requirement_v1"

    def __post_init__(self) -> None:
        radius = None if self.rms_radius is None else _finite_positive(self.rms_radius, name="final dense-relative RMS radius")
        expected_kind = FINAL_GATE_OPEN if radius is None else FINAL_GATE_REGISTERED
        if self.requirement_kind != expected_kind:
            raise ValueError("final dense-relative requirement kind does not match radius presence")
        if self.metric != FINAL_DENSE_RMS_METRIC or self.dense_reference != FINAL_DENSE_REFERENCE or self.measure != BALANCED_MEASURE:
            raise ValueError("final dense-relative function-ball coordinates changed")
        for name in ("opens_after_train_frontier_freeze", "not_a_bce_ceiling"):
            if getattr(self, name) is not True:
                raise ValueError(f"{name} must be true")
        for name in ("evaluated_by_metric_budget_binding", "terminal_validation_authority", "production_authority"):
            if getattr(self, name) is not False:
                raise ValueError(f"{name} must be false")
        object.__setattr__(self, "rms_radius", radius)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "requirement_kind": self.requirement_kind,
            "rms_radius": self.rms_radius,
            "metric": self.metric,
            "dense_reference": self.dense_reference,
            "measure": self.measure,
            "opens_after_train_frontier_freeze": self.opens_after_train_frontier_freeze,
            "not_a_bce_ceiling": self.not_a_bce_ceiling,
            "evaluated_by_metric_budget_binding": self.evaluated_by_metric_budget_binding,
            "terminal_validation_authority": self.terminal_validation_authority,
            "production_authority": self.production_authority,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "DenseRelativeFunctionBallRequirement":
        data = dict(payload)
        if data.get("schema") != cls.__dataclass_fields__["schema"].default:  # type: ignore[attr-defined]
            raise ValueError("final dense-relative requirement schema mismatch")
        return cls(
            rms_radius=data.get("rms_radius"),
            requirement_kind=str(data.get("requirement_kind")),
            metric=str(data.get("metric")),
            dense_reference=str(data.get("dense_reference")),
            measure=str(data.get("measure")),
            opens_after_train_frontier_freeze=data.get("opens_after_train_frontier_freeze"),
            not_a_bce_ceiling=data.get("not_a_bce_ceiling"),
            evaluated_by_metric_budget_binding=data.get("evaluated_by_metric_budget_binding"),
            terminal_validation_authority=data.get("terminal_validation_authority"),
            production_authority=data.get("production_authority"),
        )


@dataclass(frozen=True, slots=True)
class RawLogitMetricBudgetBinding:
    target_class: int
    bce_ceiling: float
    radius_source_kind: str
    raw_logit_rms_radius: float | None
    dense_affine_scale: float | None
    dense_affine_offset: float | None
    dense_affine_loss: float | None
    dense_affine_iterations: int | None
    dense_affine_positive_scale: bool | None
    dense_calibration_payload_hash: str | None
    normalization_counts: tuple[int, int] | None
    config_hash: str
    population_identity_hash: str
    route_axis_hash: str
    final_dense_relative_requirement: DenseRelativeFunctionBallRequirement | Mapping[str, Any]
    loss_normalization_kind: str = LOSS_NORMALIZATION_NOT_USED
    loss_normalization_fraction: tuple[int, int] | None = None
    loss_normalization: float | None = None
    tau_L: float = 0.0
    tau_Q: float = 0.0
    response_half_energy_radius: float = 0.0
    response_rms_radius: float = 0.0
    balanced_population_measure: str = BALANCED_MEASURE
    budget_epoch: str = BUDGET_EPOCH
    k_counts_at_budget_epoch: tuple[int, int] = (0, 0)
    o_counts_at_budget_epoch: tuple[int, int] = (0, 0)
    local_parent_relative_trust_region: bool = True
    calibration_refit_may_change_radius: bool = False
    ko_count_growth_may_change_radius: bool = False
    final_dense_relative_ball_authority: bool = False
    loss_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    global_optimality_authority: bool = False
    authority: str = AUTHORITY
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if type(self.target_class) is not int or self.target_class < 0:
            raise ValueError("target_class must be a nonnegative integer")
        budget = _finite_positive(self.bce_ceiling, name="BCE ceiling")
        if self.radius_source_kind not in (SOURCE_EXPLICIT_RAW, SOURCE_V0_BCE):
            raise ValueError("unknown metric budget source kind")
        raw_radius = _finite_or_none(self.raw_logit_rms_radius, name="raw-logit RMS radius")
        scale = _finite_or_none(self.dense_affine_scale, name="dense affine scale")
        offset = _finite_or_none(self.dense_affine_offset, name="dense affine offset")
        loss = _finite_or_none(self.dense_affine_loss, name="dense affine loss")
        iterations = _nonnegative_int_or_none(self.dense_affine_iterations, name="dense affine iterations")
        normalization_counts = _count_pair_or_none(self.normalization_counts, name="normalization counts")
        if isinstance(self.final_dense_relative_requirement, DenseRelativeFunctionBallRequirement):
            final_req = self.final_dense_relative_requirement
        else:
            final_req = DenseRelativeFunctionBallRequirement.from_json(self.final_dense_relative_requirement)
        if self.radius_source_kind == SOURCE_EXPLICIT_RAW:
            if raw_radius is None or raw_radius <= 0.0:
                raise ValueError("explicit raw-logit metric budget requires a finite positive raw RMS radius")
            expected_loss_kind = LOSS_NORMALIZATION_NOT_USED
            expected_fraction = None
            expected_norm = None
            tau_l = raw_radius
        else:
            if normalization_counts is None:
                raise ValueError("v0 BCE conversion requires declared normalization counts")
            if self.dense_affine_positive_scale is not True or scale is None or scale <= 0.0:
                raise ValueError("v0 BCE conversion requires a frozen positive dense calibration")
            frac = v0_loss_normalization_fraction(*normalization_counts)
            expected_loss_kind = LOSS_NORMALIZATION_V0_POS_WEIGHT
            expected_fraction = (frac.numerator, frac.denominator)
            expected_norm = float(frac)
            tau_l = budget / (expected_norm * scale)
        tau_q = tau_l * tau_l
        half = 0.5 * tau_q
        if not all(math.isfinite(v) and v > 0.0 for v in (tau_l, tau_q, half)):
            raise ValueError("derived raw-logit radius is not finite positive")
        if self.loss_normalization_kind != expected_loss_kind:
            raise ValueError("loss normalization kind is inconsistent with radius source")
        if self.loss_normalization_fraction not in (None, expected_fraction):
            raise ValueError("loss normalization coordinates are inconsistent with radius source")
        if self.loss_normalization not in (None, expected_norm):
            raise ValueError("loss normalization coordinates are inconsistent with radius source")
        for observed, expected, label in (
            (self.tau_L, tau_l, "tau_L"),
            (self.tau_Q, tau_q, "tau_Q"),
            (self.response_half_energy_radius, half, "response_half_energy_radius"),
            (self.response_rms_radius, tau_l, "response_rms_radius"),
        ):
            if observed != 0.0 and observed != expected:
                raise ValueError(f"{label} is inconsistent with radius source")
        for name in ("config_hash", "population_identity_hash", "route_axis_hash"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a nonempty source hash")
        if self.balanced_population_measure != BALANCED_MEASURE or self.budget_epoch != BUDGET_EPOCH:
            raise ValueError("metric budget semantic coordinates changed")
        for name in ("local_parent_relative_trust_region",):
            if getattr(self, name) is not True:
                raise ValueError(f"{name} must be true")
        for name in (
            "calibration_refit_may_change_radius",
            "ko_count_growth_may_change_radius",
            "final_dense_relative_ball_authority",
            "loss_oracle_authority",
            "mask_commit_authority",
            "terminal_authority",
            "global_optimality_authority",
        ):
            if getattr(self, name) is not False:
                raise ValueError(f"{name} must be false")
        if self.authority != AUTHORITY or self.schema != SCHEMA:
            raise ValueError("metric budget authority/schema mismatch")
        object.__setattr__(self, "bce_ceiling", budget)
        object.__setattr__(self, "raw_logit_rms_radius", raw_radius)
        object.__setattr__(self, "dense_affine_scale", scale)
        object.__setattr__(self, "dense_affine_offset", offset)
        object.__setattr__(self, "dense_affine_loss", loss)
        object.__setattr__(self, "dense_affine_iterations", iterations)
        object.__setattr__(self, "normalization_counts", normalization_counts)
        object.__setattr__(self, "loss_normalization_fraction", expected_fraction)
        object.__setattr__(self, "loss_normalization", expected_norm)
        object.__setattr__(self, "tau_L", tau_l)
        object.__setattr__(self, "tau_Q", tau_q)
        object.__setattr__(self, "response_half_energy_radius", half)
        object.__setattr__(self, "response_rms_radius", tau_l)
        object.__setattr__(self, "k_counts_at_budget_epoch", _nonnegative_count_pair(self.k_counts_at_budget_epoch, name="K epoch"))
        object.__setattr__(self, "o_counts_at_budget_epoch", _nonnegative_count_pair(self.o_counts_at_budget_epoch, name="O epoch"))
        object.__setattr__(self, "final_dense_relative_requirement", final_req)
        self.assert_integrity()

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "target_class": self.target_class,
            "bce_ceiling": self.bce_ceiling,
            "radius_source_kind": self.radius_source_kind,
            "raw_logit_rms_radius": self.raw_logit_rms_radius,
            "dense_affine_scale": self.dense_affine_scale,
            "dense_affine_offset": self.dense_affine_offset,
            "dense_affine_loss": self.dense_affine_loss,
            "dense_affine_iterations": self.dense_affine_iterations,
            "dense_affine_positive_scale": self.dense_affine_positive_scale,
            "dense_calibration_payload_hash": self.dense_calibration_payload_hash,
            "normalization_counts": self.normalization_counts,
            "loss_normalization_kind": self.loss_normalization_kind,
            "loss_normalization_fraction": self.loss_normalization_fraction,
            "loss_normalization": self.loss_normalization,
            "tau_L": self.tau_L,
            "tau_Q": self.tau_Q,
            "response_half_energy_radius": self.response_half_energy_radius,
            "response_rms_radius": self.response_rms_radius,
            "balanced_population_measure": self.balanced_population_measure,
            "budget_epoch": self.budget_epoch,
            "k_counts_at_budget_epoch": self.k_counts_at_budget_epoch,
            "o_counts_at_budget_epoch": self.o_counts_at_budget_epoch,
            "config_hash": self.config_hash,
            "population_identity_hash": self.population_identity_hash,
            "route_axis_hash": self.route_axis_hash,
            "final_dense_relative_requirement": self.final_dense_relative_requirement.to_json(),
            "local_parent_relative_trust_region": self.local_parent_relative_trust_region,
            "calibration_refit_may_change_radius": self.calibration_refit_may_change_radius,
            "ko_count_growth_may_change_radius": self.ko_count_growth_may_change_radius,
            "final_dense_relative_ball_authority": self.final_dense_relative_ball_authority,
            "loss_oracle_authority": self.loss_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "global_optimality_authority": self.global_optimality_authority,
            "authority": self.authority,
        }

    @property
    def budget_hash(self) -> str:
        return HASH_PREFIX + hashlib.sha256(canonical_json_bytes(self._payload())).hexdigest()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self.budget_hash
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("raw-logit metric budget hash mismatch")
        return observed

    def assert_runtime_matches(
        self,
        *,
        target_class: int,
        bce_ceiling: float,
        radius_source_kind: str,
        config_hash: str,
        population_identity_hash: str,
        route_axis_hash: str,
    ) -> None:
        self.assert_integrity()
        if self.target_class != target_class:
            raise ValueError("raw-logit metric budget target_class does not match this trajectory")
        if self.bce_ceiling != float(bce_ceiling):
            raise ValueError("raw-logit metric budget BCE ceiling does not match this trajectory")
        if self.radius_source_kind != radius_source_kind:
            raise ValueError("raw-logit metric budget source kind does not match this trajectory")
        if self.config_hash != config_hash:
            raise ValueError("raw-logit metric budget config hash does not match this trajectory")
        if self.population_identity_hash != population_identity_hash:
            raise ValueError("raw-logit metric budget population identity does not match this trajectory")
        if self.route_axis_hash != route_axis_hash:
            raise ValueError("raw-logit metric budget route axis does not match this trajectory")

    def to_json(self) -> dict[str, Any]:
        payload = dict(self._payload())
        payload["budget_hash"] = self.budget_hash
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "RawLogitMetricBudgetBinding":
        data = dict(payload)
        expected = data.pop("budget_hash", None)
        if type(expected) is not str or not expected:
            raise ValueError("raw-logit metric budget requires issued budget_hash")
        if data.get("schema") != SCHEMA:
            raise ValueError("raw-logit metric budget schema mismatch")
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"raw-logit metric budget has unknown fields: {sorted(unknown)}")
        observed = HASH_PREFIX + hashlib.sha256(canonical_json_bytes(data)).hexdigest()
        if observed != expected:
            raise ValueError("raw-logit metric budget hash mismatch")
        kwargs = {key: data[key] for key in data if key in allowed}
        if "normalization_counts" in kwargs and kwargs["normalization_counts"] is not None:
            kwargs["normalization_counts"] = tuple(kwargs["normalization_counts"])
        if "loss_normalization_fraction" in kwargs and kwargs["loss_normalization_fraction"] is not None:
            kwargs["loss_normalization_fraction"] = tuple(kwargs["loss_normalization_fraction"])
        if "k_counts_at_budget_epoch" in kwargs:
            kwargs["k_counts_at_budget_epoch"] = tuple(kwargs["k_counts_at_budget_epoch"])
        if "o_counts_at_budget_epoch" in kwargs:
            kwargs["o_counts_at_budget_epoch"] = tuple(kwargs["o_counts_at_budget_epoch"])
        binding = cls(**kwargs)
        binding.assert_integrity(expected)
        return binding


def make_final_dense_relative_requirement(rms_radius: float | None) -> DenseRelativeFunctionBallRequirement:
    return DenseRelativeFunctionBallRequirement(
        rms_radius=rms_radius,
        requirement_kind=(FINAL_GATE_OPEN if rms_radius is None else FINAL_GATE_REGISTERED),
    )


def freeze_raw_logit_metric_budget(
    *,
    target_class: int,
    bce_ceiling: float,
    radius_source_kind: str,
    raw_logit_rms_radius: float | None,
    dense_calibration: Any | None,
    normalization_counts: tuple[int, int] | None,
    config_hash: str,
    population_identity_hash: str,
    route_axis_hash: str,
    final_dense_relative_rms_radius: float | None,
    k_counts_at_budget_epoch: tuple[int, int] = (0, 0),
    o_counts_at_budget_epoch: tuple[int, int] = (0, 0),
) -> RawLogitMetricBudgetBinding:
    cal_payload = None
    if dense_calibration is not None:
        if hasattr(dense_calibration, "to_json"):
            cal_payload = dense_calibration.to_json()
        elif isinstance(dense_calibration, Mapping):
            cal_payload = dict(dense_calibration)
        else:
            raise ValueError("dense calibration must expose to_json() or a mapping payload")
    if radius_source_kind == SOURCE_V0_BCE and cal_payload is None:
        raise ValueError("v0 BCE conversion requires dense calibration")
    dense_positive = None if cal_payload is None else cal_payload.get("positive_scale")
    return RawLogitMetricBudgetBinding(
        target_class=target_class,
        bce_ceiling=bce_ceiling,
        radius_source_kind=radius_source_kind,
        raw_logit_rms_radius=raw_logit_rms_radius,
        dense_affine_scale=(None if cal_payload is None else cal_payload.get("scale")),
        dense_affine_offset=(None if cal_payload is None else cal_payload.get("offset")),
        dense_affine_loss=(None if cal_payload is None else cal_payload.get("loss")),
        dense_affine_iterations=(None if cal_payload is None else cal_payload.get("iterations")),
        dense_affine_positive_scale=dense_positive,
        dense_calibration_payload_hash=(None if cal_payload is None else stable_payload_hash(cal_payload, prefix="tractable-v1-dense-calibration-")),
        normalization_counts=normalization_counts,
        config_hash=config_hash,
        population_identity_hash=population_identity_hash,
        route_axis_hash=route_axis_hash,
        final_dense_relative_requirement=make_final_dense_relative_requirement(final_dense_relative_rms_radius),
        loss_normalization_kind=(LOSS_NORMALIZATION_V0_POS_WEIGHT if radius_source_kind == SOURCE_V0_BCE else LOSS_NORMALIZATION_NOT_USED),
        k_counts_at_budget_epoch=k_counts_at_budget_epoch,
        o_counts_at_budget_epoch=o_counts_at_budget_epoch,
    )
