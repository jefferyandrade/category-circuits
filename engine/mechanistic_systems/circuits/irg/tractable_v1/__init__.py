"""Tractable native v1 lane: exact factored geometry, fast operators, trust-region control.

This subpackage is the runtime-tractable realization of the canonical modern-v1
geometric pruning method.  It keeps
the registered estimands — corrected Aumann–Shapley prices, literal-J response
geometry, signed native-fiber work, the raw-target-logit query, balanced
target/complement strata, the BCE-derived trust radii — and changes only how
they are evaluated:

* rows are exact real numbers evaluated in operator form (one GEMM or unfolded
  GEMM per layer) instead of materialized rounded fp32 row tiles;
* the response Gram is assembled exactly in context space (B x B) instead of a
  materialized N x r factor;
* first-order solvers use spectral-norm step certificates and run in seconds;
* every accepted action is replayed exactly and its actual-versus-predicted
  damage ratio controls the trust radius.

Authority: DIAGNOSTIC / EXPLORATORY.  Nothing here issues production, commit,
or Pareto authority; receipts are labelled
``authority: diagnostic_tractable_lane`` and are scored from files only.
"""

from __future__ import annotations

TRACTABLE_V1_LANE_ID = "modern_v1_tractable_native_lane_v1"
TRACTABLE_V1_AUTHORITY = "diagnostic_tractable_lane"

__all__ = ["TRACTABLE_V1_LANE_ID", "TRACTABLE_V1_AUTHORITY"]
