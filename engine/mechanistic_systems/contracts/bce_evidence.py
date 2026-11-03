"""Typed fixed-calibration BCE measurement evidence.

These records bind paired hard-mask observations and calibration precision
refusals. They do not select actions, stop a workflow, or authorize a commit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from mechanistic_systems.contracts.sampled_evidence import CrossfitPanel, PANEL_AXIS_CHAIN_KEY
from mechanistic_systems.inference.sequential_bce import (
    CalibrationImprovementProfile,
    RunningMoment,
)


# Provider metadata contract (``CrossfitPanel.allocation_metadata`` keys) for
# the sealed without-replacement oracle law.  The provider lane imports these
# names; a panel that carries none of them is legacy with-replacement
# evidence and is decided exactly as before v4.
ORACLE_METADATA_SAMPLING_WITH_REPLACEMENT = "sampling_with_replacement"
ORACLE_METADATA_FPC_APPLIED = "finite_population_correction_applied"
ORACLE_METADATA_SAMPLING_LAW = "oracle_sampling_law"
ORACLE_METADATA_POPULATION_BY_STRATUM = "oracle_population_by_stratum"
ORACLE_METADATA_DRAWN_BY_STRATUM = "oracle_drawn_by_stratum"
ORACLE_METADATA_EXHAUSTED_STRATA = "oracle_exhausted_strata"
ORACLE_METADATA_REMAINING_POPULATION = "oracle_remaining_population"
ORACLE_METADATA_CENSUS_TERMINAL_REACHED = "oracle_census_terminal_reached"
ORACLE_METADATA_CENSUS_COMPLETE = "oracle_census_complete"
ORACLE_METADATA_CENSUS_STRATA_COUNT = "oracle_census_strata_count"
ORACLE_METADATA_CENSUS_EXACT_WEIGHT = "oracle_census_exact_weight"
ORACLE_METADATA_PANEL_AXIS_CHAIN = PANEL_AXIS_CHAIN_KEY
ORACLE_METADATA_KEYS = (
    ORACLE_METADATA_SAMPLING_WITH_REPLACEMENT,
    ORACLE_METADATA_FPC_APPLIED,
    ORACLE_METADATA_SAMPLING_LAW,
    ORACLE_METADATA_POPULATION_BY_STRATUM,
    ORACLE_METADATA_DRAWN_BY_STRATUM,
    ORACLE_METADATA_EXHAUSTED_STRATA,
    ORACLE_METADATA_REMAINING_POPULATION,
    ORACLE_METADATA_CENSUS_TERMINAL_REACHED,
    ORACLE_METADATA_CENSUS_COMPLETE,
    ORACLE_METADATA_CENSUS_STRATA_COUNT,
    ORACLE_METADATA_CENSUS_EXACT_WEIGHT,
    ORACLE_METADATA_PANEL_AXIS_CHAIN,
)


ORACLE_FINITE_POPULATION_CENSUS_RECEIPT_V1 = (
    "native_oracle_finite_population_census_receipt_v1"
)


class CalibrationPrecisionHorizonExhausted(RuntimeError):
    """A candidate calibration remained unresolved at its compute horizon.

    This is a typed, fail-closed decision outcome rather than a runtime
    failure.  The committed parent calibration remains valid; the caller must
    not evaluate or commit the unresolved candidate, but may shrink/bisect the
    proposed pruning action.
    """

    def __init__(self, receipt: Mapping[str, Any]) -> None:
        frozen = dict(receipt)
        if frozen.get(
            "schema"
        ) != "native_bce_calibration_precision_horizon_receipt_v1" or not frozen.get(
            "mask_hash"
        ):
            raise ValueError(
                "calibration precision exhaustion requires a typed receipt"
            )
        self.receipt = frozen
        super().__init__(
            "candidate calibration precision remained unresolved at its "
            "declared compute horizon"
        )


class OracleFinitePopulationCensusExhausted(RuntimeError):
    """The sealed without-replacement oracle has no rows left to draw.

    Sibling of :class:`CalibrationPrecisionHorizonExhausted`.  Under the
    sealed without-replacement law every proposal-adjudication stratum is a
    finite population; once all of them are at census the oracle cannot add
    rows.  A provider that cannot relabel its previous evidence as the exact
    census (for example under a refinable estimand whose calibration terms
    still move) raises this typed outcome instead of a bare error.  The
    frontier converts it into an ``unresolved`` decision with reason
    ``finite_population_census_exhausted_unresolved``; nothing is committed.

    The receipt must carry ``schema ==
    ORACLE_FINITE_POPULATION_CENSUS_RECEIPT_V1``.  Providers are expected to
    include ``oracle_population_by_stratum`` and ``oracle_drawn_by_stratum``
    so the exhaustion is auditable from the receipt alone.
    """

    def __init__(self, receipt: Mapping[str, Any]) -> None:
        frozen = dict(receipt)
        if frozen.get("schema") != ORACLE_FINITE_POPULATION_CENSUS_RECEIPT_V1:
            raise ValueError(
                "oracle finite-population census exhaustion requires a typed receipt"
            )
        self.receipt = frozen
        super().__init__(
            "sealed without-replacement oracle exhausted every stratum's finite "
            "population without resolving the decision"
        )


@dataclass(frozen=True)
class FixedCalibrationPairedBCEEvidence:
    """Paired BCE increments under a frozen independently fitted calibration.

    Calibration and adjudication may either use disjoint finite populations or
    independent IID streams over the same frozen empirical distribution. The
    state-specific affine fits are frozen before proposal outcomes are drawn.
    ``CrossfitPanel`` carries unique draw identities; its fold IDs are
    compatibility bookkeeping rather than fit ownership.  Under fold-owned
    providers (``calibration_fold_ownership``), fit ownership is
    instead assigned per row by the source-context calibration fold: each
    row is scored by the fold fit trained without that row's fold.  The
    panel fold IDs remain walk bookkeeping either way.
    """

    panel: CrossfitPanel
    paired_bce_increment: np.ndarray
    committed_mask_hash: str
    candidate_mask_hash: str
    committed_calibration_state_id: str
    candidate_calibration_state_id: str
    stratified_moments: Mapping[str, Mapping[str, float | int]] = field(
        default_factory=dict
    )
    bce_loss_upper_bound: float | None = None
    committed_calibration_improvement_bound: float = 0.0
    candidate_calibration_improvement_bound: float = 0.0
    committed_calibration_profile: CalibrationImprovementProfile = field(
        default_factory=CalibrationImprovementProfile
    )
    candidate_calibration_profile: CalibrationImprovementProfile = field(
        default_factory=CalibrationImprovementProfile
    )
    joint_precision_allocation_receipts: tuple[Mapping[str, Any], ...] = ()
    activation_flip_counts: Mapping[str, int] = field(default_factory=dict)
    activation_flip_context_count: int | None = None
    activation_coverage: Mapping[str, Any] = field(default_factory=dict)
    candidate_preactivation_percentiles: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict
    )
    preactivation_percentile_coverage: Mapping[str, Any] = field(default_factory=dict)
    dense_anchored_readings: Mapping[str, Any] = field(default_factory=dict)
    paired_bce_increment_observations: np.ndarray | None = None
    calibration_contexts_disjoint: bool = True
    calibration_stream_independent: bool = True
    common_random_numbers: bool = True
    # (names, counts [S], means [1,S], M2 [1,S]); values remain on the
    # acquisition device.  Host counts describe the known sample layout.
    device_moments: Any | None = None

    def validate(self, *, fold_count: int, device_count_transfers=None) -> None:
        self.panel.validate(fold_count=fold_count)
        if self.device_moments is not None:
            names, counts, means, m2 = self.device_moments
            known = self.panel.allocation_metadata.get("delivered_stratum_counts", {})
            if (len(names) != len(known) or set(names) != set(known)
                    or sum(known.values()) != len(self.panel.context_ids)
                    or counts.shape != (len(names),) or means.shape != (1, len(names))
                    or m2.shape != means.shape or str(means.dtype) != "torch.float64"
                    or str(m2.dtype) != "torch.float64" or str(counts.dtype) != "torch.float64"
                    or counts.device != means.device or m2.device != means.device
                    or self.bce_loss_upper_bound is not None):
                raise ValueError("device BCE moments differ from the declared sample layout")
            import torch
            message = "device BCE tensor counts differ from declared stratum counts"
            declared = [known[name] for name in names]
            if counts.device.type == 'cpu':
                if not torch.equal(counts, torch.tensor(declared, dtype=counts.dtype)):
                    raise ValueError(message)
            else:
                from mechanistic_systems.inference.device_bce import HostTransferBuffer, defer_device_check
                transfers = (HostTransferBuffer(counts.device) if device_count_transfers is None
                             else device_count_transfers)
                expected = transfers.copy('declared_stratum_counts', declared)
                failed = (counts != expected).any()
                if not defer_device_check(failed,lambda:ValueError(message)):
                    torch._assert_async(~failed,message)
            # Numerical validity is checked by the compiled look and returned
            # with its control code, without a separate device read.
            values = np.zeros(0, dtype=np.float64)
            moments = {}
            evidence_count = len(self.panel.context_ids)
        else:
            values = np.asarray(self.paired_bce_increment, dtype=np.float64).reshape(-1)
            moments = {
                str(key): RunningMoment.from_mapping(value)
                for key, value in dict(self.stratified_moments).items()
            }
            evidence_count = (int(values.size) if values.size
                              else sum(moment.count for moment in moments.values()))
        if evidence_count != len(self.panel.context_ids) or (
            values.size and not np.isfinite(values).all()
        ):
            raise ValueError(
                "fixed-calibration BCE increments must cover the complete panel"
            )
        if values.size == 0 and not moments and self.device_moments is None:
            raise ValueError(
                "fixed-calibration BCE evidence lacks rows or sufficient statistics"
            )
        if (
            not self.calibration_contexts_disjoint
            and not self.calibration_stream_independent
        ):
            raise ValueError(
                "fixed-calibration BCE evidence requires disjoint rows or an "
                "independent IID calibration stream"
            )
        if not self.common_random_numbers:
            raise ValueError("paired BCE evidence requires common random numbers")
        if (
            not self.committed_calibration_state_id
            or not self.candidate_calibration_state_id
        ):
            raise ValueError("paired BCE evidence lacks calibration identity")
        if len(self.committed_mask_hash) != 64 or len(self.candidate_mask_hash) != 64:
            raise ValueError("paired BCE evidence requires immutable mask identities")
        if self.bce_loss_upper_bound is not None and (
            not math.isfinite(float(self.bce_loss_upper_bound))
            or float(self.bce_loss_upper_bound) <= 0.0
            or (values.size == 0 and moments)
            or (
                values.size
                and np.max(np.abs(values), initial=0.0)
                > float(self.bce_loss_upper_bound) + 1e-10
            )
        ):
            raise ValueError(
                "bounded paired BCE evidence requires retained rows within its bound"
            )
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in (
                self.committed_calibration_improvement_bound,
                self.candidate_calibration_improvement_bound,
            )
        ):
            raise ValueError(
                "calibration improvement bounds must be finite and nonnegative"
            )


__all__ = [
    "CalibrationPrecisionHorizonExhausted",
    "FixedCalibrationPairedBCEEvidence",
    "ORACLE_FINITE_POPULATION_CENSUS_RECEIPT_V1",
    "OracleFinitePopulationCensusExhausted",
]
