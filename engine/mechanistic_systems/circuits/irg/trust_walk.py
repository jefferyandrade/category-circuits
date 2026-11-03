"""Non-executing compatibility contract for the retired IRG-v1 trust walk.

The partition-alternating workflow implementation is byte-preserved in archive
wave ``20260810_retired_irg_v1_trust_walk_form``.  Modern ``irg/v1`` is a
different, unregistered unified-queue design.  No execution or replay callable
is exposed from this live module.

The registered queue-v0 pilot still constructs :class:`TrustWalkChart` as an
internal response-field container in protected historical helper code.  The
current v0 path never invokes the retired stepper, so the data contract and its
immutable algorithm label remain here until that protected carrier is sealed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from .contracts import GeometricPartitioningSpec
from .trust_geometry import GlobalTrustGeometryState


TRUST_WALK_ALGORITHM_VERSION = "irg_v1_trust_walk_partition_alternating_v1"
TRUST_WALK_STEP_RECEIPT_SCHEMA = "v1_trust_walk_step_receipt_v1"
TRUST_WALK_CHART_RECEIPT_SCHEMA = "v1_trust_walk_chart_receipt_v1"
TRUST_WALK_RUN_RECEIPT_SCHEMA = "v1_trust_walk_receipt_v1"
RETIRED_TRUST_WALK_ARCHIVE_WAVE = "20260810_retired_irg_v1_trust_walk_form"
RETIRED_TRUST_WALK_AUTHORITY_CEILING = (
    "compatibility_data_contract_only_no_execution_replay_or_commit_authority"
)


@dataclass(frozen=True)
class TrustWalkPolicy:
    """Frozen historical constants retained only for protected-data parsing."""

    tau_growth: float = 2.0
    tau_shrink: float = 0.5
    rho_faithful_window: tuple[float, float] = (0.25, 4.0)
    eta_step: float = 0.5
    action_scale: float = 1.0
    chart_cap: int = 32
    maximum_walk_proposals: int = 10000
    panel_growth: str = "variance_forecast_v1"
    panel_margin_fraction: float = 0.5
    panel_maximum_contexts: int = 4096
    stall_semantics: str = "zero_commits_in_turn_v0"

    def validate(self) -> None:
        growth = float(self.tau_growth)
        shrink = float(self.tau_shrink)
        window = tuple(float(value) for value in self.rho_faithful_window)
        if not math.isfinite(growth) or growth <= 1.0:
            raise ValueError("tau_growth must be finite and exceed 1")
        if not math.isfinite(shrink) or not 0.0 < shrink < 1.0:
            raise ValueError("tau_shrink must lie in (0,1)")
        if (
            len(window) != 2
            or not all(math.isfinite(value) for value in window)
            or not 0.0 < window[0] <= 1.0 <= window[1]
        ):
            raise ValueError("rho_faithful_window must bracket one positively")
        if not math.isfinite(float(self.eta_step)) or not 0.0 < self.eta_step <= 1.0:
            raise ValueError("eta_step must lie in (0,1]")
        if int(self.chart_cap) < 1 or int(self.maximum_walk_proposals) < 1:
            raise ValueError("historical trust-walk caps must be positive")
        if not 0.0 < float(self.panel_margin_fraction) < 1.0:
            raise ValueError("panel_margin_fraction must lie in (0,1)")


@dataclass(frozen=True)
class TrustWalkChart:
    """Compatibility-only chart payload still referenced by protected v0 code."""

    chart_index: int
    anchor_mask_hash: str
    partitioning: GeometricPartitioningSpec
    partition_order: Mapping[str, np.ndarray]
    key_id: str
    key_values: np.ndarray
    key_standard_errors: np.ndarray
    global_mean: np.ndarray
    trust_state: GlobalTrustGeometryState
    rademacher_global_factor: np.ndarray
    horizon: Mapping[str, int]
    field_receipt: Mapping[str, Any]
    measurement_class_id: str

    def __post_init__(self) -> None:
        key_values = np.asarray(self.key_values, dtype=np.float64)
        key_standard_errors = np.asarray(
            self.key_standard_errors,
            dtype=np.float64,
        )
        global_mean = np.asarray(self.global_mean, dtype=np.float64)
        rademacher = np.asarray(
            self.rademacher_global_factor,
            dtype=np.float64,
        )
        route_count = int(key_values.size)
        if (
            key_values.shape != (route_count,)
            or key_standard_errors.shape != (route_count,)
            or global_mean.shape != (route_count,)
            or rademacher.ndim != 2
            or rademacher.shape[1] != route_count
        ):
            raise ValueError("trust-walk chart arrays do not share the route axis")
        if set(self.horizon) != set(self.partition_order):
            raise ValueError("trust-walk chart horizon does not cover its partitions")
        for partition_id, order in self.partition_order.items():
            horizon = int(self.horizon[partition_id])
            if not 0 <= horizon <= len(order):
                raise ValueError(
                    f"trust-walk chart horizon is invalid for {partition_id}"
                )
        if not self.anchor_mask_hash:
            raise ValueError("trust-walk chart requires an anchor mask hash")
        if self.trust_state.measured_mask_hash != self.anchor_mask_hash:
            raise ValueError(
                "trust-walk chart trust state does not match its anchor mask"
            )
        object.__setattr__(self, "key_values", key_values)
        object.__setattr__(self, "key_standard_errors", key_standard_errors)
        object.__setattr__(self, "global_mean", global_mean)
        object.__setattr__(self, "rademacher_global_factor", rademacher)


__all__ = [
    "RETIRED_TRUST_WALK_ARCHIVE_WAVE",
    "RETIRED_TRUST_WALK_AUTHORITY_CEILING",
    "TRUST_WALK_ALGORITHM_VERSION",
    "TRUST_WALK_CHART_RECEIPT_SCHEMA",
    "TRUST_WALK_RUN_RECEIPT_SCHEMA",
    "TRUST_WALK_STEP_RECEIPT_SCHEMA",
    "TrustWalkChart",
    "TrustWalkPolicy",
]
