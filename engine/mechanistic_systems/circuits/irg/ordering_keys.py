"""Registered ordering-key registry for the partition commensurability theory.

At first registration the removal cost is ``c_r == 1`` for every key, so the
receipt isolates key semantics from cost normalization.  Keys construct
candidate orders ONLY — they never carry commit authority; every commit
remains certified by exact paired calibrated-BCE evidence.

The registry (values per route, with a declared estimator interval):

* ``abs_raw``              — K_inc, the incumbent activity envelope
  ``A_r = E_x||H^{1/2} G_r(x)||``; byte-identical to the default order source.
* ``k1_expected_damage``   — ``|mu_r|``; interval from the
  finite-population-corrected mean standard error.
* ``k2_damage_energy``     — ``mu_r^2 + sigma_r^2`` = the measured
  global-head second moment; interval from the measured
  second-moment standard error directly.
* ``k3_whitened_salience`` — ``mu~_r^2 / C~_rr``; interval by the
  first-order delta method (formula below).  Under the declared SCALAR
  response semantics the metric ``H`` is a positive scalar and
  cancels in the ratio, so the whitened ratio equals ``mu_r^2 / C_rr``
  with ``C_rr`` the per-route contextual variance of the global head.

K3 delta-method interval, exactly as registered:

    Var(K3) ~= (2 mu~/C~_rr)^2 Var(mu~)
             + (mu~^2/C~_rr^2)^2 Var(C~_rr)
             - 2 (2 mu~/C~_rr)(mu~^2/C~_rr^2) Cov(mu~, C~_rr)

``Var(mu~)`` is read from the measured (R,3,3) mean-estimator-covariance
tensor (global-head diagonal).  ``Var(C~_rr)`` and ``Cov(mu~, C~_rr)`` are
not directly measured; they are constructed by first-order propagation
through ``C = m2 - mu^2`` from the measured second-moment standard error
and the same estimator-covariance diagonal, with the unmeasured
mean/second-moment estimator cross-covariance set to zero:

    Var(C)     = Var(m2) + 4 mu^2 Var(mu)
    Cov(mu, C) = -2 mu Var(mu)

which is algebraically the exact first-order delta method in the measured
``(mu, m2)`` coordinates with ``Cov(mu_hat, m2_hat) = 0``.

Degenerate-route convention (declared): where ``C_rr <= 0`` (an exactly
constant contribution — in practice only dead routes with ``mu = 0``),
K3's value and interval are 0, matching the tensor's existing
``effective_context_participation`` degenerate-division convention.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping

import numpy as np

ORDERING_KEY_REGISTRY_VERSION = "partition_commensurability_key_registry_v1"

# Registration order: incumbent first, then the theory keys K1/K2/K3.
V1_ORDERING_KEYS = (
    "abs_raw",
    "k1_expected_damage",
    "k2_damage_energy",
    "k3_whitened_salience",
)

# Field statistics the widened pilot return carries beyond the historical
# (local_mean, global_mean, global_activity_mean) triple.  Names double as
# geometry-cache payload array keys.
ORDERING_FIELD_STATISTIC_NAMES = (
    "global_mean_standard_error",
    "global_contextual_variance",
    "global_second_moment",
    "global_second_moment_standard_error",
    "global_mean_estimator_variance",
    "global_activity_standard_error",
)

_NONNEGATIVE_FIELDS = (
    "global_mean_standard_error",
    "global_contextual_variance",
    "global_second_moment",
    "global_second_moment_standard_error",
    "global_mean_estimator_variance",
    "global_activity",
    "global_activity_standard_error",
)


@dataclass(frozen=True, slots=True)
class GlobalHeadOrderingField:
    """Per-route global-head statistics consumed by the registered keys.

    Every array is flat on the canonical route axis.  ``global_mean`` and
    ``global_mean_standard_error`` are the fused, finite-population-
    corrected head-mean statistics; the second-moment/estimator-variance
    entries come from the finalize-time tripartite tensor (global-head
    slice); ``global_activity`` is the incumbent envelope.
    """

    global_mean: np.ndarray
    global_mean_standard_error: np.ndarray
    global_contextual_variance: np.ndarray
    global_second_moment: np.ndarray
    global_second_moment_standard_error: np.ndarray
    global_mean_estimator_variance: np.ndarray
    global_activity: np.ndarray
    global_activity_standard_error: np.ndarray

    def __post_init__(self) -> None:
        reference: np.ndarray | None = None
        for item in fields(self):
            value = np.asarray(getattr(self, item.name), dtype=np.float64)
            value = value.reshape(-1)
            if value.size <= 0 or not np.isfinite(value).all():
                raise ValueError(
                    f"ordering field {item.name} must be finite and nonempty"
                )
            if item.name in _NONNEGATIVE_FIELDS and np.any(value < 0.0):
                raise ValueError(
                    f"ordering field {item.name} must be nonnegative"
                )
            if reference is not None and value.shape != reference.shape:
                raise ValueError(
                    "ordering field arrays must share one route axis"
                )
            reference = value
            object.__setattr__(self, item.name, value)

    @property
    def route_count(self) -> int:
        return int(self.global_mean.size)

    @classmethod
    def from_statistics(
        cls,
        *,
        global_mean: np.ndarray,
        global_activity: np.ndarray,
        statistics: Mapping[str, np.ndarray],
    ) -> "GlobalHeadOrderingField":
        """Assemble the field from the widened pilot/cache payload."""

        missing = tuple(
            name
            for name in ORDERING_FIELD_STATISTIC_NAMES
            if name not in statistics
        )
        if missing:
            raise ValueError(
                f"ordering field statistics are incomplete: missing {missing!r}"
            )
        return cls(
            global_mean=np.asarray(global_mean, dtype=np.float64),
            global_activity=np.asarray(global_activity, dtype=np.float64),
            **{
                name: np.asarray(statistics[name], dtype=np.float64)
                for name in ORDERING_FIELD_STATISTIC_NAMES
            },
        )


def ordering_key_statistic(
    key: str,
    field: GlobalHeadOrderingField,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-route ``(value, standard_error)`` for one registered key.

    Values are nonnegative for every registered key, matching the order
    certificate's score contract; ``c_r == 1`` throughout.
    """

    if key not in V1_ORDERING_KEYS:
        raise ValueError(f"unregistered ordering key {key!r}")
    if key == "abs_raw":
        # K_inc: the incumbent envelope, byte-identical when selected.
        return (
            field.global_activity.copy(),
            field.global_activity_standard_error.copy(),
        )
    if key == "k1_expected_damage":
        return (
            np.abs(field.global_mean),
            field.global_mean_standard_error.copy(),
        )
    if key == "k2_damage_energy":
        # (D1) exact: E[G_r^2] = mu_r^2 + sigma_r^2, the measured second
        # moment, with its directly measured standard error.
        return (
            field.global_second_moment.copy(),
            field.global_second_moment_standard_error.copy(),
        )
    # k3_whitened_salience — see the module docstring for the registered
    # formula and the declared degenerate-route convention.
    mean = field.global_mean
    contextual = field.global_contextual_variance
    mean_variance = field.global_mean_estimator_variance
    second_variance = field.global_second_moment_standard_error**2
    positive = contextual > 0.0
    zeros = np.zeros_like(mean)
    value = np.divide(
        mean * mean, contextual, out=zeros.copy(), where=positive
    )
    variance_c = second_variance + 4.0 * mean * mean * mean_variance
    covariance_mu_c = -2.0 * mean * mean_variance
    gradient_mu = np.divide(
        2.0 * mean, contextual, out=zeros.copy(), where=positive
    )
    gradient_c = np.divide(
        mean * mean,
        contextual * contextual,
        out=zeros.copy(),
        where=positive,
    )
    variance = (
        gradient_mu * gradient_mu * mean_variance
        + gradient_c * gradient_c * variance_c
        - 2.0 * gradient_mu * gradient_c * covariance_mu_c
    )
    return value, np.sqrt(np.maximum(variance, 0.0))


def ordering_key_interval(
    key: str,
    field: GlobalHeadOrderingField,
    *,
    critical_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-route ``(value, lower, upper)`` at one critical value.

    The lower bound clamps at zero exactly as the order certificate's
    scope receipt does for its nonnegative score axis.
    """

    critical = float(critical_value)
    if not np.isfinite(critical) or critical <= 0.0:
        raise ValueError("ordering key critical value must be positive")
    value, standard_error = ordering_key_statistic(key, field)
    lower = np.maximum(value - critical * standard_error, 0.0)
    upper = value + critical * standard_error
    return value, lower, upper


__all__ = [
    "GlobalHeadOrderingField",
    "ORDERING_FIELD_STATISTIC_NAMES",
    "ORDERING_KEY_REGISTRY_VERSION",
    "V1_ORDERING_KEYS",
    "ordering_key_interval",
    "ordering_key_statistic",
]
