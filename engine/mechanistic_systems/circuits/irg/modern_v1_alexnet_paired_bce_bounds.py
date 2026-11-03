"""Analytic finite supports for AlexNet's paired dense-relative BCE oracle.

The oracle observes candidate-minus-dense Bernoulli log-loss rows after each
mask receives its own validation-independent positive affine calibration.  This
module transports the exact, mask-specific raw target-logit intervals through
those two frozen affine maps and through the two class-balanced BCE strata.

Every scalar arithmetic operation that can round is widened by one binary64
ULP in the required direction.  The resulting target/complement supports are
therefore conservative inputs to the anytime finite-population acquisition;
they are not empirical calibration, clipping, or a change of estimand.
"""

from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass, field, fields
from decimal import Decimal, localcontext
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Mapping

import numpy as np

from .contracts import content_hash
from .modern_v1_alexnet_envelope import (
    V1AlexNetRawTargetLogitIntervalCertificate,
)
from .modern_v1_controller import V1FrozenAffineCalibration


PAIRED_BCE_SUPPORT_BOUND_SCHEMA: Final = (
    "partition_free_v1_alexnet_paired_bce_support_bound_v1"
)
CALIBRATED_BCE_INTERVAL_SCHEMA: Final = (
    "partition_free_v1_alexnet_calibrated_bce_interval_v1"
)
STRATUM_SUPPORT_BOUND_SCHEMA: Final = (
    "partition_free_v1_alexnet_paired_bce_stratum_support_bound_v1"
)
PAIRED_BCE_SUPPORT_BOUND_ALGORITHM_ID: Final = (
    "mask_specific_raw_logit_interval_positive_affine_softplus_difference_v1"
)
PAIRED_BCE_OUTWARD_ROUNDING_ID: Final = (
    "binary64_outward_arithmetic_decimal192_softplus_v2"
)
PAIRED_BCE_SUPPORT_BOUND_METADATA_KEY: Final = (
    "paired_bce_support_bound_certificate"
)
PAIRED_BCE_SUPPORT_BOUND_HASH_METADATA_KEY: Final = (
    "paired_bce_support_bound_certificate_hash"
)
PAIRED_BCE_SUPPORT_BOUND_MODE_METADATA_KEY: Final = (
    "paired_bce_support_bound_mode"
)
FINITE_PARTIAL_AUTHORITY_MODE: Final = (
    "analytic_finite_partial_authority_mask_specific_v1"
)
CENSUS_ONLY_COMPATIBILITY_MODE: Final = (
    "census_only_compatibility_no_analytic_alexnet_owners_v1"
)

_CALIBRATED_PREFIX: Final = "partition-free-v1-alexnet-calibrated-bce-interval-"
_STRATUM_PREFIX: Final = "partition-free-v1-alexnet-paired-bce-stratum-bound-"
_CERTIFICATE_PREFIX: Final = "partition-free-v1-alexnet-paired-bce-bound-"
_RAW_CERTIFICATE_HASH_FIELD: Final = "certificate_hash"
_CERTIFICATE_HASH_FIELD: Final = "certificate_hash"
_STRATA: Final = ("target", "complement")
_SOFTPLUS_DECIMAL_PRECISION: Final = 192
_SOFTPLUS_DECIMAL_EMAX: Final = 999999


def _identity(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be finite binary64")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite binary64")
    return result


def _ordered_nonvacuous(lower: object, upper: object, name: str) -> tuple[float, float]:
    lo = _finite(lower, f"{name} lower")
    hi = _finite(upper, f"{name} upper")
    if not lo < hi:
        raise ValueError(f"{name} must have finite positive width")
    span = hi - lo
    if not math.isfinite(span) or span < sys.float_info.min:
        raise ValueError(f"{name} is numerically vacuous")
    return lo, hi


def _next_down(value: object, name: str, *, nonnegative: bool = False) -> float:
    scalar = _finite(value, name)
    widened = float(np.nextafter(np.float64(scalar), np.float64(-np.inf)))
    if nonnegative and widened < 0.0:
        # BCE is mathematically nonnegative.  When the correctly rounded value
        # is +0, the next lower binary64 number is negative; zero is the tighter
        # rigorous lower bound.
        widened = 0.0
    return _finite(widened, f"outward lower {name}")


def _next_up(value: object, name: str) -> float:
    scalar = _finite(value, name)
    widened = float(np.nextafter(np.float64(scalar), np.float64(np.inf)))
    return _finite(widened, f"outward upper {name}")


def _mul_down(left: float, right: float, name: str) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.float64(left) * np.float64(right)
    return _next_down(value, name)


def _mul_up(left: float, right: float, name: str) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.float64(left) * np.float64(right)
    return _next_up(value, name)


def _add_down(left: float, right: float, name: str) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.float64(left) + np.float64(right)
    return _next_down(value, name)


def _add_up(left: float, right: float, name: str) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.float64(left) + np.float64(right)
    return _next_up(value, name)


def _subtract_down(left: float, right: float, name: str) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.float64(left) - np.float64(right)
    return _next_down(value, name)


def _subtract_up(left: float, right: float, name: str) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.float64(left) - np.float64(right)
    return _next_up(value, name)


def _softplus_decimal(value: float, name: str) -> Decimal:
    """High-precision softplus oracle before binary64 outward transport."""

    scalar = _finite(value, name)
    with localcontext() as context:
        context.prec = _SOFTPLUS_DECIMAL_PRECISION
        context.Emax = _SOFTPLUS_DECIMAL_EMAX
        context.Emin = -_SOFTPLUS_DECIMAL_EMAX
        x = context.create_decimal_from_float(scalar)
        one = Decimal(1)
        if x > 0:
            result = x + (one + (-x).exp(context=context)).ln(context=context)
        else:
            result = (one + x.exp(context=context)).ln(context=context)
    if not result.is_finite() or result < 0:
        raise ValueError(f"{name} softplus is not finite and nonnegative")
    return result


def _next_down_decimal(value: Decimal, name: str, *, nonnegative: bool = False) -> float:
    widened = float(np.nextafter(np.float64(float(value)), np.float64(-np.inf)))
    if nonnegative and widened < 0.0:
        widened = 0.0
    return _finite(widened, f"outward lower {name}")


def _next_up_decimal(value: Decimal, name: str) -> float:
    widened = float(np.nextafter(np.float64(float(value)), np.float64(np.inf)))
    return _finite(widened, f"outward upper {name}")


def _softplus_down(value: float, name: str) -> float:
    return _next_down_decimal(
        _softplus_decimal(value, name),
        name,
        nonnegative=True,
    )


def _softplus_up(value: float, name: str) -> float:
    return _next_up_decimal(_softplus_decimal(value, name), name)


@lru_cache(maxsize=1)
def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def _raw_certificate_payload(
    certificate: V1AlexNetRawTargetLogitIntervalCertificate,
) -> dict[str, Any]:
    if type(certificate) is not V1AlexNetRawTargetLogitIntervalCertificate:
        raise TypeError("raw target-logit owner requires the exact certificate type")
    certificate_hash = certificate.assert_integrity()
    payload = {
        item.name: getattr(certificate, item.name)
        for item in fields(V1AlexNetRawTargetLogitIntervalCertificate)
        if item.init
    }
    payload[_RAW_CERTIFICATE_HASH_FIELD] = certificate_hash
    return payload


def _raw_certificate_from_payload(
    value: object,
) -> V1AlexNetRawTargetLogitIntervalCertificate:
    if not isinstance(value, Mapping):
        raise TypeError("raw target-logit certificate payload must be a mapping")
    constructor_names = {
        item.name
        for item in fields(V1AlexNetRawTargetLogitIntervalCertificate)
        if item.init
    }
    expected = {*constructor_names, _RAW_CERTIFICATE_HASH_FIELD}
    if set(value) != expected:
        raise ValueError("raw target-logit certificate payload fields differ")
    payload = dict(value)
    expected_hash = _identity(
        payload.pop(_RAW_CERTIFICATE_HASH_FIELD),
        "raw target-logit certificate hash",
    )
    result = V1AlexNetRawTargetLogitIntervalCertificate(**payload)
    result.assert_integrity(expected_hash)
    return result


@dataclass(frozen=True, slots=True)
class V1AlexNetCalibratedBCEIntervalCertificate:
    """One raw-logit interval transported through one frozen affine map."""

    role: str
    mask_hash: str
    raw_target_logit_certificate_hash: str
    frozen_calibration_hash: str
    calibration_hash: str
    calibration_state_id: str
    calibration_source_hash: str
    calibration_context_axis_hash: str
    source_label_map_hash: str
    fit_receipt_hash: str
    query_hash: str
    model_state_hash: str
    target_class_id: int
    affine_scale: float
    affine_offset: float
    raw_target_logit_lower: float
    raw_target_logit_upper: float
    calibrated_target_logit_lower: float
    calibrated_target_logit_upper: float
    target_bce_lower: float
    target_bce_upper: float
    complement_bce_lower: float
    complement_bce_upper: float
    verifier_source_hash: str
    algorithm_id: str = PAIRED_BCE_SUPPORT_BOUND_ALGORITHM_ID
    outward_rounding_id: str = PAIRED_BCE_OUTWARD_ROUNDING_ID
    schema: str = CALIBRATED_BCE_INTERVAL_SCHEMA
    _content_hash_cache: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.role not in {"dense_reference", "candidate"}:
            raise ValueError("calibrated BCE interval has an invalid mask role")
        for name in (
            "mask_hash",
            "raw_target_logit_certificate_hash",
            "frozen_calibration_hash",
            "calibration_hash",
            "calibration_state_id",
            "calibration_source_hash",
            "calibration_context_axis_hash",
            "source_label_map_hash",
            "fit_receipt_hash",
            "query_hash",
            "model_state_hash",
            "verifier_source_hash",
        ):
            _identity(getattr(self, name), name)
        if isinstance(self.target_class_id, bool) or int(self.target_class_id) < 0:
            raise ValueError("target_class_id must be a nonnegative integer")
        scale = _finite(self.affine_scale, "affine scale")
        if scale <= 0.0:
            raise ValueError("affine scale must be finite and strictly positive")
        _finite(self.affine_offset, "affine offset")
        raw_lower = _finite(self.raw_target_logit_lower, "raw target-logit lower")
        raw_upper = _finite(self.raw_target_logit_upper, "raw target-logit upper")
        if raw_lower > raw_upper:
            raise ValueError("raw target-logit interval is not ordered")
        _ordered_nonvacuous(
            self.calibrated_target_logit_lower,
            self.calibrated_target_logit_upper,
            "calibrated target-logit interval",
        )
        target_lower, target_upper = _ordered_nonvacuous(
            self.target_bce_lower, self.target_bce_upper, "target BCE interval"
        )
        complement_lower, complement_upper = _ordered_nonvacuous(
            self.complement_bce_lower,
            self.complement_bce_upper,
            "complement BCE interval",
        )
        if target_lower < 0.0 or complement_lower < 0.0:
            raise ValueError("BCE interval cannot have a negative lower endpoint")
        if target_upper <= 0.0 or complement_upper <= 0.0:
            raise ValueError("BCE interval must have a positive upper endpoint")
        if (
            self.algorithm_id != PAIRED_BCE_SUPPORT_BOUND_ALGORITHM_ID
            or self.outward_rounding_id != PAIRED_BCE_OUTWARD_ROUNDING_ID
            or self.schema != CALIBRATED_BCE_INTERVAL_SCHEMA
        ):
            raise ValueError("calibrated BCE interval identity is unsupported")

    def _hash_payload(self) -> dict[str, Any]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(type(self))
            if item.init
        }

    @property
    def content_hash(self) -> str:
        observed = content_hash(self._hash_payload(), prefix=_CALIBRATED_PREFIX)
        cached = self._content_hash_cache
        if cached is not None and cached != observed:
            raise ValueError("calibrated BCE interval certificate was tampered")
        object.__setattr__(self, "_content_hash_cache", observed)
        return observed

    def to_payload(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {**self._hash_payload(), _CERTIFICATE_HASH_FIELD: self.content_hash}
        )

    @classmethod
    def from_payload(
        cls, value: object
    ) -> V1AlexNetCalibratedBCEIntervalCertificate:
        if not isinstance(value, Mapping):
            raise TypeError("calibrated BCE interval payload must be a mapping")
        names = {item.name for item in fields(cls) if item.init}
        if set(value) != {*names, _CERTIFICATE_HASH_FIELD}:
            raise ValueError("calibrated BCE interval payload fields differ")
        payload = dict(value)
        expected_hash = _identity(
            payload.pop(_CERTIFICATE_HASH_FIELD), "calibrated BCE interval hash"
        )
        result = cls(**payload)
        if result.content_hash != expected_hash:
            raise ValueError("calibrated BCE interval hash does not replay")
        return result


@dataclass(frozen=True, slots=True)
class V1AlexNetPairedBCEStratumSupportBound:
    """Finite pointwise support for one class-balanced oracle stratum."""

    stratum_id: str
    stratum_label: int
    dense_calibrated_bce_interval_hash: str
    candidate_calibrated_bce_interval_hash: str
    dense_bce_lower: float
    dense_bce_upper: float
    candidate_bce_lower: float
    candidate_bce_upper: float
    paired_loss_lower_bound: float
    paired_loss_upper_bound: float
    verifier_source_hash: str
    algorithm_id: str = PAIRED_BCE_SUPPORT_BOUND_ALGORITHM_ID
    outward_rounding_id: str = PAIRED_BCE_OUTWARD_ROUNDING_ID
    finite_partial_authority: bool = True
    bound_frozen_before_oracle_rows: bool = True
    schema: str = STRATUM_SUPPORT_BOUND_SCHEMA
    _bound_hash_cache: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.stratum_id not in _STRATA:
            raise ValueError("paired-BCE support stratum must be target or complement")
        expected_label = 1 if self.stratum_id == "target" else 0
        if isinstance(self.stratum_label, bool) or int(self.stratum_label) != expected_label:
            raise ValueError("paired-BCE stratum label differs from its identity")
        for name in (
            "dense_calibrated_bce_interval_hash",
            "candidate_calibrated_bce_interval_hash",
            "verifier_source_hash",
        ):
            _identity(getattr(self, name), name)
        dense_lower, dense_upper = _ordered_nonvacuous(
            self.dense_bce_lower, self.dense_bce_upper, "dense BCE support"
        )
        candidate_lower, candidate_upper = _ordered_nonvacuous(
            self.candidate_bce_lower,
            self.candidate_bce_upper,
            "candidate BCE support",
        )
        if dense_lower < 0.0 or candidate_lower < 0.0:
            raise ValueError("BCE support cannot be negative")
        paired_lower, paired_upper = _ordered_nonvacuous(
            self.paired_loss_lower_bound,
            self.paired_loss_upper_bound,
            "candidate-minus-dense paired BCE support",
        )
        if not (
            paired_lower <= candidate_lower - dense_upper
            and paired_upper >= candidate_upper - dense_lower
        ):
            raise ValueError("paired BCE endpoints do not enclose interval subtraction")
        if (
            self.algorithm_id != PAIRED_BCE_SUPPORT_BOUND_ALGORITHM_ID
            or self.outward_rounding_id != PAIRED_BCE_OUTWARD_ROUNDING_ID
            or self.finite_partial_authority is not True
            or self.bound_frozen_before_oracle_rows is not True
            or self.schema != STRATUM_SUPPORT_BOUND_SCHEMA
        ):
            raise ValueError("paired-BCE stratum support identity is unsupported")

    def _hash_payload(self) -> dict[str, Any]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(type(self))
            if item.init
        }

    @property
    def bound_certificate_hash(self) -> str:
        observed = content_hash(self._hash_payload(), prefix=_STRATUM_PREFIX)
        cached = self._bound_hash_cache
        if cached is not None and cached != observed:
            raise ValueError("paired-BCE stratum support certificate was tampered")
        object.__setattr__(self, "_bound_hash_cache", observed)
        return observed

    def to_payload(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                **self._hash_payload(),
                _CERTIFICATE_HASH_FIELD: self.bound_certificate_hash,
            }
        )

    @classmethod
    def from_payload(cls, value: object) -> V1AlexNetPairedBCEStratumSupportBound:
        if not isinstance(value, Mapping):
            raise TypeError("paired-BCE stratum support payload must be a mapping")
        names = {item.name for item in fields(cls) if item.init}
        if set(value) != {*names, _CERTIFICATE_HASH_FIELD}:
            raise ValueError("paired-BCE stratum support payload fields differ")
        payload = dict(value)
        expected_hash = _identity(
            payload.pop(_CERTIFICATE_HASH_FIELD), "paired-BCE stratum support hash"
        )
        result = cls(**payload)
        if result.bound_certificate_hash != expected_hash:
            raise ValueError("paired-BCE stratum support hash does not replay")
        return result


@dataclass(frozen=True, slots=True)
class V1AlexNetPairedBCESupportBoundCertificate:
    """Complete two-stratum finite-support certificate for one candidate."""

    dense_raw_target_logit_certificate: V1AlexNetRawTargetLogitIntervalCertificate
    candidate_raw_target_logit_certificate: V1AlexNetRawTargetLogitIntervalCertificate
    dense_frozen_calibration: V1FrozenAffineCalibration
    candidate_frozen_calibration: V1FrozenAffineCalibration
    dense_calibrated_bce_interval: V1AlexNetCalibratedBCEIntervalCertificate
    candidate_calibrated_bce_interval: V1AlexNetCalibratedBCEIntervalCertificate
    target_stratum_bound: V1AlexNetPairedBCEStratumSupportBound
    complement_stratum_bound: V1AlexNetPairedBCEStratumSupportBound
    verifier_source_hash: str
    algorithm_id: str = PAIRED_BCE_SUPPORT_BOUND_ALGORITHM_ID
    outward_rounding_id: str = PAIRED_BCE_OUTWARD_ROUNDING_ID
    mask_specific: bool = True
    bound_frozen_before_oracle_rows: bool = True
    observed_oracle_rows_consumed: bool = False
    terminal_outcomes_opened: bool = False
    schema: str = PAIRED_BCE_SUPPORT_BOUND_SCHEMA
    _certificate_hash_cache: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        dense_raw = self.dense_raw_target_logit_certificate
        candidate_raw = self.candidate_raw_target_logit_certificate
        if type(dense_raw) is not V1AlexNetRawTargetLogitIntervalCertificate or type(
            candidate_raw
        ) is not V1AlexNetRawTargetLogitIntervalCertificate:
            raise TypeError("paired-BCE support requires exact raw-logit certificates")
        dense_raw.assert_integrity()
        candidate_raw.assert_integrity()
        if type(self.dense_frozen_calibration) is not V1FrozenAffineCalibration or type(
            self.candidate_frozen_calibration
        ) is not V1FrozenAffineCalibration:
            raise TypeError("paired-BCE support requires exact frozen calibrations")
        if type(
            self.dense_calibrated_bce_interval
        ) is not V1AlexNetCalibratedBCEIntervalCertificate or type(
            self.candidate_calibrated_bce_interval
        ) is not V1AlexNetCalibratedBCEIntervalCertificate:
            raise TypeError("paired-BCE support requires exact calibrated intervals")
        if type(
            self.target_stratum_bound
        ) is not V1AlexNetPairedBCEStratumSupportBound or type(
            self.complement_stratum_bound
        ) is not V1AlexNetPairedBCEStratumSupportBound:
            raise TypeError("paired-BCE support requires exact stratum certificates")
        _identity(self.verifier_source_hash, "verifier source hash")
        if dense_raw.current_mask_hash == candidate_raw.current_mask_hash:
            raise ValueError("paired-BCE support requires distinct dense/candidate masks")
        common_raw_names = (
            "architecture_id",
            "topology_hash",
            "route_axis_hash",
            "route_axis_content_hash",
            "route_count",
            "model_identity_hash",
            "model_state_hash",
            "query_contract_hash",
            "target_class_id",
            "preprocessing_hash",
            "input_box_hash",
            "graph_semantics_hash",
            "arithmetic_profile_hash",
            "verifier_source_hash",
            "algorithm_id",
            "outward_rounding_id",
        )
        for name in common_raw_names:
            if getattr(dense_raw, name) != getattr(candidate_raw, name):
                raise ValueError(f"dense/candidate raw-logit owners differ on {name}")
        dense_frozen = self.dense_frozen_calibration
        candidate_frozen = self.candidate_frozen_calibration
        if (
            dense_frozen.mask_hash != dense_raw.current_mask_hash
            or candidate_frozen.mask_hash != candidate_raw.current_mask_hash
        ):
            raise ValueError("raw-logit masks differ from frozen calibration masks")
        for name in (
            "query_id",
            "query_hash",
            "target_class_id",
            "calibration_context_axis_hash",
            "source_label_map_hash",
            "model_state_hash",
            "calibration_schema",
        ):
            if getattr(dense_frozen, name) != getattr(candidate_frozen, name):
                raise ValueError(f"dense/candidate calibration owners differ on {name}")
        if (
            dense_frozen.query_hash != dense_raw.query_contract_hash
            or dense_frozen.model_state_hash != dense_raw.model_state_hash
            or dense_frozen.target_class_id != dense_raw.target_class_id
        ):
            raise ValueError("frozen calibration differs from raw-logit owner identities")

        expected_dense = _build_calibrated_interval(
            role="dense_reference", raw=dense_raw, calibration=dense_frozen
        )
        expected_candidate = _build_calibrated_interval(
            role="candidate", raw=candidate_raw, calibration=candidate_frozen
        )
        if self.dense_calibrated_bce_interval != expected_dense:
            raise ValueError("dense calibrated BCE interval is not reproducible")
        if self.candidate_calibrated_bce_interval != expected_candidate:
            raise ValueError("candidate calibrated BCE interval is not reproducible")
        expected_target = _build_stratum_bound(
            stratum_id="target", dense=expected_dense, candidate=expected_candidate
        )
        expected_complement = _build_stratum_bound(
            stratum_id="complement",
            dense=expected_dense,
            candidate=expected_candidate,
        )
        if self.target_stratum_bound != expected_target:
            raise ValueError("target-stratum paired BCE support is not reproducible")
        if self.complement_stratum_bound != expected_complement:
            raise ValueError("complement-stratum paired BCE support is not reproducible")
        if (
            self.algorithm_id != PAIRED_BCE_SUPPORT_BOUND_ALGORITHM_ID
            or self.outward_rounding_id != PAIRED_BCE_OUTWARD_ROUNDING_ID
            or self.mask_specific is not True
            or self.bound_frozen_before_oracle_rows is not True
            or self.observed_oracle_rows_consumed is not False
            or self.terminal_outcomes_opened is not False
            or self.schema != PAIRED_BCE_SUPPORT_BOUND_SCHEMA
        ):
            raise PermissionError("paired-BCE support crossed its pre-outcome authority")

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "dense_raw_target_logit_certificate": _raw_certificate_payload(
                self.dense_raw_target_logit_certificate
            ),
            "candidate_raw_target_logit_certificate": _raw_certificate_payload(
                self.candidate_raw_target_logit_certificate
            ),
            "dense_frozen_calibration": dict(
                self.dense_frozen_calibration.to_payload()
            ),
            "candidate_frozen_calibration": dict(
                self.candidate_frozen_calibration.to_payload()
            ),
            "dense_calibrated_bce_interval": dict(
                self.dense_calibrated_bce_interval.to_payload()
            ),
            "candidate_calibrated_bce_interval": dict(
                self.candidate_calibrated_bce_interval.to_payload()
            ),
            "target_stratum_bound": dict(self.target_stratum_bound.to_payload()),
            "complement_stratum_bound": dict(
                self.complement_stratum_bound.to_payload()
            ),
            "verifier_source_hash": self.verifier_source_hash,
            "algorithm_id": self.algorithm_id,
            "outward_rounding_id": self.outward_rounding_id,
            "mask_specific": self.mask_specific,
            "bound_frozen_before_oracle_rows": self.bound_frozen_before_oracle_rows,
            "observed_oracle_rows_consumed": self.observed_oracle_rows_consumed,
            "terminal_outcomes_opened": self.terminal_outcomes_opened,
            "schema": self.schema,
        }

    @property
    def certificate_hash(self) -> str:
        observed = content_hash(self._hash_payload(), prefix=_CERTIFICATE_PREFIX)
        cached = self._certificate_hash_cache
        if cached is not None and cached != observed:
            raise ValueError("paired-BCE support certificate was tampered")
        object.__setattr__(self, "_certificate_hash_cache", observed)
        return observed

    @property
    def stratum_bound_certificate_hashes(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "target": self.target_stratum_bound.bound_certificate_hash,
                "complement": self.complement_stratum_bound.bound_certificate_hash,
            }
        )

    def stratum_bound(self, stratum_id: str) -> V1AlexNetPairedBCEStratumSupportBound:
        if stratum_id == "target":
            return self.target_stratum_bound
        if stratum_id == "complement":
            return self.complement_stratum_bound
        raise KeyError("paired-BCE support stratum must be target or complement")

    def to_payload(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {**self._hash_payload(), _CERTIFICATE_HASH_FIELD: self.certificate_hash}
        )

    @classmethod
    def from_payload(
        cls, value: object
    ) -> V1AlexNetPairedBCESupportBoundCertificate:
        if not isinstance(value, Mapping):
            raise TypeError("paired-BCE support certificate payload must be a mapping")
        expected = {
            "dense_raw_target_logit_certificate",
            "candidate_raw_target_logit_certificate",
            "dense_frozen_calibration",
            "candidate_frozen_calibration",
            "dense_calibrated_bce_interval",
            "candidate_calibrated_bce_interval",
            "target_stratum_bound",
            "complement_stratum_bound",
            "verifier_source_hash",
            "algorithm_id",
            "outward_rounding_id",
            "mask_specific",
            "bound_frozen_before_oracle_rows",
            "observed_oracle_rows_consumed",
            "terminal_outcomes_opened",
            "schema",
            _CERTIFICATE_HASH_FIELD,
        }
        if set(value) != expected:
            raise ValueError("paired-BCE support certificate payload fields differ")
        payload = dict(value)
        expected_hash = _identity(
            payload.pop(_CERTIFICATE_HASH_FIELD), "paired-BCE support certificate hash"
        )
        result = cls(
            dense_raw_target_logit_certificate=_raw_certificate_from_payload(
                payload.pop("dense_raw_target_logit_certificate")
            ),
            candidate_raw_target_logit_certificate=_raw_certificate_from_payload(
                payload.pop("candidate_raw_target_logit_certificate")
            ),
            dense_frozen_calibration=V1FrozenAffineCalibration.from_payload(
                payload.pop("dense_frozen_calibration")
            ),
            candidate_frozen_calibration=V1FrozenAffineCalibration.from_payload(
                payload.pop("candidate_frozen_calibration")
            ),
            dense_calibrated_bce_interval=(
                V1AlexNetCalibratedBCEIntervalCertificate.from_payload(
                    payload.pop("dense_calibrated_bce_interval")
                )
            ),
            candidate_calibrated_bce_interval=(
                V1AlexNetCalibratedBCEIntervalCertificate.from_payload(
                    payload.pop("candidate_calibrated_bce_interval")
                )
            ),
            target_stratum_bound=V1AlexNetPairedBCEStratumSupportBound.from_payload(
                payload.pop("target_stratum_bound")
            ),
            complement_stratum_bound=(
                V1AlexNetPairedBCEStratumSupportBound.from_payload(
                    payload.pop("complement_stratum_bound")
                )
            ),
            **payload,
        )
        if result.certificate_hash != expected_hash:
            raise ValueError("paired-BCE support certificate hash does not replay")
        return result


def _build_calibrated_interval(
    *,
    role: str,
    raw: V1AlexNetRawTargetLogitIntervalCertificate,
    calibration: V1FrozenAffineCalibration,
) -> V1AlexNetCalibratedBCEIntervalCertificate:
    raw.assert_integrity()
    scale = _finite(calibration.scale, "affine scale")
    offset = _finite(calibration.offset, "affine offset")
    if scale <= 0.0:
        raise ValueError("affine scale must be finite and strictly positive")
    scaled_lower = _mul_down(
        scale, raw.raw_target_logit_lower, f"{role} affine lower multiplication"
    )
    scaled_upper = _mul_up(
        scale, raw.raw_target_logit_upper, f"{role} affine upper multiplication"
    )
    calibrated_lower = _add_down(
        scaled_lower, offset, f"{role} affine lower addition"
    )
    calibrated_upper = _add_up(
        scaled_upper, offset, f"{role} affine upper addition"
    )
    _ordered_nonvacuous(
        calibrated_lower, calibrated_upper, f"{role} calibrated target logit"
    )
    # y=1: softplus(-c), decreasing in c.  y=0: softplus(c), increasing.
    target_lower = _softplus_down(
        -calibrated_upper, f"{role} target BCE lower softplus"
    )
    target_upper = _softplus_up(
        -calibrated_lower, f"{role} target BCE upper softplus"
    )
    complement_lower = _softplus_down(
        calibrated_lower, f"{role} complement BCE lower softplus"
    )
    complement_upper = _softplus_up(
        calibrated_upper, f"{role} complement BCE upper softplus"
    )
    return V1AlexNetCalibratedBCEIntervalCertificate(
        role=role,
        mask_hash=raw.current_mask_hash,
        raw_target_logit_certificate_hash=raw.certificate_hash,
        frozen_calibration_hash=calibration.frozen_calibration_hash,
        calibration_hash=calibration.calibration_hash,
        calibration_state_id=calibration.calibration_state_id,
        calibration_source_hash=calibration.calibration_source_hash,
        calibration_context_axis_hash=calibration.calibration_context_axis_hash,
        source_label_map_hash=calibration.source_label_map_hash,
        fit_receipt_hash=calibration.fit_receipt_hash,
        query_hash=calibration.query_hash,
        model_state_hash=calibration.model_state_hash,
        target_class_id=calibration.target_class_id,
        affine_scale=scale,
        affine_offset=offset,
        raw_target_logit_lower=raw.raw_target_logit_lower,
        raw_target_logit_upper=raw.raw_target_logit_upper,
        calibrated_target_logit_lower=calibrated_lower,
        calibrated_target_logit_upper=calibrated_upper,
        target_bce_lower=target_lower,
        target_bce_upper=target_upper,
        complement_bce_lower=complement_lower,
        complement_bce_upper=complement_upper,
        verifier_source_hash=_source_sha256(),
    )


def _build_stratum_bound(
    *,
    stratum_id: str,
    dense: V1AlexNetCalibratedBCEIntervalCertificate,
    candidate: V1AlexNetCalibratedBCEIntervalCertificate,
) -> V1AlexNetPairedBCEStratumSupportBound:
    if stratum_id == "target":
        dense_lower, dense_upper = dense.target_bce_lower, dense.target_bce_upper
        candidate_lower = candidate.target_bce_lower
        candidate_upper = candidate.target_bce_upper
        label = 1
    elif stratum_id == "complement":
        dense_lower = dense.complement_bce_lower
        dense_upper = dense.complement_bce_upper
        candidate_lower = candidate.complement_bce_lower
        candidate_upper = candidate.complement_bce_upper
        label = 0
    else:
        raise KeyError("paired-BCE support stratum must be target or complement")
    paired_lower = _subtract_down(
        candidate_lower,
        dense_upper,
        f"{stratum_id} candidate-minus-dense lower subtraction",
    )
    paired_upper = _subtract_up(
        candidate_upper,
        dense_lower,
        f"{stratum_id} candidate-minus-dense upper subtraction",
    )
    _ordered_nonvacuous(
        paired_lower, paired_upper, f"{stratum_id} paired BCE support"
    )
    return V1AlexNetPairedBCEStratumSupportBound(
        stratum_id=stratum_id,
        stratum_label=label,
        dense_calibrated_bce_interval_hash=dense.content_hash,
        candidate_calibrated_bce_interval_hash=candidate.content_hash,
        dense_bce_lower=dense_lower,
        dense_bce_upper=dense_upper,
        candidate_bce_lower=candidate_lower,
        candidate_bce_upper=candidate_upper,
        paired_loss_lower_bound=paired_lower,
        paired_loss_upper_bound=paired_upper,
        verifier_source_hash=_source_sha256(),
    )


def build_v1_alexnet_paired_bce_support_bound_certificate(
    *,
    dense_raw_target_logit_certificate: V1AlexNetRawTargetLogitIntervalCertificate,
    candidate_raw_target_logit_certificate: V1AlexNetRawTargetLogitIntervalCertificate,
    dense_frozen_calibration: V1FrozenAffineCalibration,
    candidate_frozen_calibration: V1FrozenAffineCalibration,
) -> V1AlexNetPairedBCESupportBoundCertificate:
    """Compose a replayable two-stratum support from exact analytic owners."""

    dense_interval = _build_calibrated_interval(
        role="dense_reference",
        raw=dense_raw_target_logit_certificate,
        calibration=dense_frozen_calibration,
    )
    candidate_interval = _build_calibrated_interval(
        role="candidate",
        raw=candidate_raw_target_logit_certificate,
        calibration=candidate_frozen_calibration,
    )
    result = V1AlexNetPairedBCESupportBoundCertificate(
        dense_raw_target_logit_certificate=dense_raw_target_logit_certificate,
        candidate_raw_target_logit_certificate=candidate_raw_target_logit_certificate,
        dense_frozen_calibration=dense_frozen_calibration,
        candidate_frozen_calibration=candidate_frozen_calibration,
        dense_calibrated_bce_interval=dense_interval,
        candidate_calibrated_bce_interval=candidate_interval,
        target_stratum_bound=_build_stratum_bound(
            stratum_id="target", dense=dense_interval, candidate=candidate_interval
        ),
        complement_stratum_bound=_build_stratum_bound(
            stratum_id="complement",
            dense=dense_interval,
            candidate=candidate_interval,
        ),
        verifier_source_hash=_source_sha256(),
    )
    # Materialize all nested hashes now so later mutation cannot establish a
    # new first-seen cache value.
    _ = result.certificate_hash
    _ = result.stratum_bound_certificate_hashes
    return result


def paired_bce_support_bound_certificate_from_freeze_metadata(
    metadata: Mapping[str, Any],
    *,
    required: bool = False,
) -> V1AlexNetPairedBCESupportBoundCertificate | None:
    """Recover the exact analytic certificate embedded in a calibration freeze."""

    if not isinstance(metadata, Mapping):
        raise TypeError("candidate calibration metadata must be a mapping")
    payload = metadata.get(PAIRED_BCE_SUPPORT_BOUND_METADATA_KEY)
    declared_hash = metadata.get(PAIRED_BCE_SUPPORT_BOUND_HASH_METADATA_KEY)
    mode = metadata.get(PAIRED_BCE_SUPPORT_BOUND_MODE_METADATA_KEY)
    if payload is None and declared_hash is None:
        if required:
            raise ValueError("candidate calibration freeze lacks its analytic BCE bound")
        if mode not in (None, CENSUS_ONLY_COMPATIBILITY_MODE):
            raise ValueError("candidate calibration freeze has an invalid bound mode")
        return None
    if payload is None or declared_hash is None:
        raise ValueError("paired-BCE support payload/hash must be co-present")
    if mode != FINITE_PARTIAL_AUTHORITY_MODE:
        raise ValueError("analytic paired-BCE support has an invalid authority mode")
    result = V1AlexNetPairedBCESupportBoundCertificate.from_payload(payload)
    if result.certificate_hash != _identity(
        declared_hash, "paired-BCE support certificate hash"
    ):
        raise ValueError("paired-BCE support certificate hash differs from metadata")
    return result


__all__ = [
    "CENSUS_ONLY_COMPATIBILITY_MODE",
    "FINITE_PARTIAL_AUTHORITY_MODE",
    "PAIRED_BCE_SUPPORT_BOUND_HASH_METADATA_KEY",
    "PAIRED_BCE_SUPPORT_BOUND_METADATA_KEY",
    "PAIRED_BCE_SUPPORT_BOUND_MODE_METADATA_KEY",
    "V1AlexNetCalibratedBCEIntervalCertificate",
    "V1AlexNetPairedBCEStratumSupportBound",
    "V1AlexNetPairedBCESupportBoundCertificate",
    "build_v1_alexnet_paired_bce_support_bound_certificate",
    "paired_bce_support_bound_certificate_from_freeze_metadata",
]
