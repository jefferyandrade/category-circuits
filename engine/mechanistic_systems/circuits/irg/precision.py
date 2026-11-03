"""Independent, predictable sample-size policies for canonical IRG v1/v2.

The canonical v1/v2 runtime does not require one monolithic acquisition
optimizer.  Each physical estimand instead owns a stopping/allocation rule
matched to its scientific role:

* calibration rows resolve calibration-induced active-contrast uncertainty;
* whole-route response rows stabilize the active frontier/screen/trust action;
* exact hard-mask rows resolve the paired BCE accept/reject boundary.

Every next packet is a function only of the frozen design and observations
available before that packet.  The policies determine acquisition size; they
never replace exact hard-mask adjudication or narrow another modality's
interval.
"""

from __future__ import annotations

import math
import dataclasses
from dataclasses import dataclass
from statistics import NormalDist
from typing import Mapping, Sequence

import numpy as np

from mechanistic_systems.evidence import staged_alpha
from mechanistic_systems.inference.sequential_bce import (
    IID_WITH_REPLACEMENT_SAMPLING_ID,
    ORACLE_SAMPLING_LAWS,
)
from .contracts import content_hash


RESPONSE_FIELD_ENVELOPE_RECEIPT_V1 = (
    "hard_partition_response_envelope_receipt_v1"
)


class ResponseFieldPrecisionEnvelopeExhausted(RuntimeError):
    """A hard-partition response field remained unresolved when it stopped.

    This is a typed, FAIL-CLOSED decision outcome, not a runtime failure. It
    replaces a bare ``RuntimeError`` that carried only a message: the field is
    still refused for every downstream pruning workflow -- exactly as before --
    but the run now emits the evidence it accumulated instead of discarding it.

    It is raised for BOTH unresolved terminals: a sealed census exhausted
    without resolution, and a declared acquisition envelope reached first.
    ``stopping_reason`` in the receipt distinguishes them, and ``binding_bar``
    names which obligation was still outstanding.

    The exception type is deliberate. The three call sites unpack this
    function's tuple straight into a cache, and no consumer of the returned
    partition fields checks ``decision_resolved``; returning an unresolved
    field would therefore require auditing every consumer and would silently
    admit it wherever one was missed. Raising preserves the fail-closed
    guarantee by construction.
    """

    def __init__(self, receipt: Mapping[str, object]) -> None:
        frozen = dict(receipt)
        if frozen.get("schema") != RESPONSE_FIELD_ENVELOPE_RECEIPT_V1:
            raise ValueError(
                "response-field envelope exhaustion requires a typed receipt"
            )
        self.receipt = frozen
        super().__init__(
            "hard-partition response acquisition stopped without resolving "
            f"{frozen.get('required_partition_views')!r} and the "
            "global-activity complete executable proposal-mask stability at "
            "the declared tolerances "
            f"(stopping_reason={frozen.get('stopping_reason')!r}, "
            f"binding_bar={frozen.get('binding_bar')!r}, "
            f"delivered={frozen.get('delivered_cumulative_count')!r} of "
            f"{frozen.get('population_count')!r}); "
            "no pruning workflow may consume the field"
        )


def simultaneous_normal_critical_value(
    confidence_level: float,
    *,
    family_size: int = 1,
) -> float:
    """Two-sided Bonferroni critical value for a frozen finite family."""

    confidence = float(confidence_level)
    family = int(family_size)
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence_level must lie in (0.5,1)")
    if family <= 0:
        raise ValueError("family_size must be positive")
    tail = (1.0 - confidence) / (2.0 * family)
    return float(NormalDist().inv_cdf(1.0 - tail))


@dataclass(frozen=True, slots=True)
class IndependentSampleSizePolicy:
    """Operational packet controls compiled from one confidence requirement."""

    confidence_level: float = 0.95
    calibration_minimum_packet: int = 32
    # Operational fail-safe only. IID calibration stops at its precision
    # obligation or the global runtime/resource envelope; this large ceiling
    # prevents an accidentally unbounded run when no external envelope was
    # supplied. It is not a sample-size target or population size.
    calibration_maximum_contexts_without_support_cap: int = 65536
    response_initial_packet: int = 32
    response_packet: int = 128
    response_stable_looks: int = 2
    response_maximum_contexts_per_stratum: int = 512
    # Maximum activity mass whose retained/pruned membership may change at
    # the controller's audited score boundary. This is an operational
    # approximation tolerance for proposal construction; unlike confidence
    # it does not represent an error probability.
    maximum_prune_order_uncertainty_fraction: float = 0.001
    # Sign labels form a coarse proposal partition rather than a ranking or
    # commit authority. Preserve the historical unresolved sign-activity
    # tolerance independently of the stricter active-boundary obligation.
    maximum_partition_sign_uncertainty_fraction: float = 0.05
    oracle_initial_packet: int = 32
    oracle_minimum_packet: int = 16
    oracle_maximum_packet: int = 256
    oracle_maximum_contexts_per_candidate: int | None = 4096
    precision_only_iid: bool = False
    oracle_minimum_numerical_gap: float = 1e-8
    # Operational fail-safe only, in the same class as
    # ``calibration_maximum_contexts_without_support_cap`` above: a wall-clock
    # envelope for ONE hard-partition response-field acquisition. It is not a
    # sample-size target, not a statistical numeral, and it never enters any
    # confidence statement or tolerance.
    #
    # WHY: the response-field loop's only terminals are resolution and sealed
    # census exhaustion, and its cost per look grows with the panel it is
    # doubling. MEASURED on q504/cp/local_attribution_sign: the sign bar was
    # satisfied from look 3, the order bar never was, and looks 3-10 spent
    # ~3 h growing the panel 768 -> 49,664 at 0.072-0.084 s/context before the
    # job was killed at its 12 h wall having written nothing at all -- no
    # receipt, no partial field, no diagnosis. Arming this envelope converts
    # that outcome into a disclosed unresolved certificate that names the
    # binding bar. ``None`` (the default) reproduces prior behaviour exactly.
    response_acquisition_envelope_seconds: float | None = None
    # Gate the prune order on the CONSEQUENCE PRICE against its own derived
    # floor, instead of the activity-mass fraction against
    # ``maximum_prune_order_uncertainty_fraction``.
    #
    # WHY THIS EXISTS. The mass measure charges a mask change by the ACTIVITY of
    # the routes involved, so its smallest nonzero step in a scope is one
    # route's share. MEASURED at coarse granularity: `features.stage1.conv`
    # holds 192 routes and its largest carries 142% of the whole 0.001 budget,
    # so a single boundary swap cannot respect the tolerance and the bar is
    # unsatisfiable by arithmetic rather than by sampling. Pricing the same
    # ambiguity by how much DAMAGE it could cause gives that scope the smallest
    # price in the field, because a pair of near-tied routes costs nothing to
    # order either way.
    #
    # Default False so every existing arm is byte-identical; arming it is an
    # explicit act and changes what the census certifies about, which is a
    # preregistration-affecting change rather than a knob.
    prune_order_consequence_gated: bool = False
    # Proposal-adjudication oracle sampling law (v4 decision law).
    # ``independent_empirical_iid_with_replacement_v1`` reproduces every
    # historical arm byte for byte; the sealed without-replacement id makes
    # each class stratum a finite population revealed as a nested sealed
    # permutation prefix, so the paired oracle terminates at census. The law
    # changes what the interval certifies (exact finite-population
    # correction), so it is part of ``acquisition_policy_hash``.
    oracle_sampling_law: str = IID_WITH_REPLACEMENT_SAMPLING_ID
    # Operational fail-safe only, the exact sibling of
    # ``response_acquisition_envelope_seconds`` for ONE proposal's paired
    # oracle acquisition: a wall-clock envelope after which the next look is
    # declined (never the first look) and the proposal returns a typed
    # ``unresolved`` decision with a disclosure receipt. It is not a sample
    # size, enters no confidence statement, and ``None`` reproduces prior
    # behaviour exactly.
    oracle_acquisition_envelope_seconds: float | None = None
    # The census's response-panel sizing is ENROLLED: the sign forecast sizes
    # the panel with a one-packet fresh floor per look (a hard
    # fresh-allocation constraint).  The acquisition schedule is covered by
    # acquisition_policy_hash; arms under a different census sizing are not
    # replicates on census-sizing-dependent claims.

    def __post_init__(self) -> None:
        if not 0.5 < float(self.confidence_level) < 1.0:
            raise ValueError("confidence_level must lie in (0.5,1)")
        integer_fields = (
            "calibration_minimum_packet",
            "calibration_maximum_contexts_without_support_cap",
            "response_initial_packet",
            "response_packet",
            "response_stable_looks",
            "response_maximum_contexts_per_stratum",
            "oracle_initial_packet",
            "oracle_minimum_packet",
            "oracle_maximum_packet",
        )
        if any(int(getattr(self, name)) <= 0 for name in integer_fields):
            raise ValueError("sample-size packet and horizon controls must be positive")
        if int(self.oracle_minimum_packet) > int(self.oracle_maximum_packet):
            raise ValueError("oracle minimum packet may not exceed maximum packet")
        if (
            self.oracle_maximum_contexts_per_candidate is not None
            and int(self.oracle_initial_packet)
            > int(self.oracle_maximum_contexts_per_candidate)
        ):
            raise ValueError("oracle initial packet exceeds its context horizon")
        if int(self.response_initial_packet) > int(
            self.response_maximum_contexts_per_stratum
        ):
            raise ValueError(
                "response initial packet exceeds its per-stratum horizon"
            )
        gap = float(self.oracle_minimum_numerical_gap)
        if not math.isfinite(gap) or gap <= 0.0:
            raise ValueError("oracle_minimum_numerical_gap must be finite and positive")
        order_tolerance = float(
            self.maximum_prune_order_uncertainty_fraction
        )
        if (
            not math.isfinite(order_tolerance)
            or not 0.0 <= order_tolerance <= 1.0
        ):
            raise ValueError(
                "maximum_prune_order_uncertainty_fraction must lie in [0,1]"
            )
        sign_tolerance = float(
            self.maximum_partition_sign_uncertainty_fraction
        )
        if (
            not math.isfinite(sign_tolerance)
            or not 0.0 <= sign_tolerance <= 1.0
        ):
            raise ValueError(
                "maximum_partition_sign_uncertainty_fraction must lie in [0,1]"
            )
        envelope = self.response_acquisition_envelope_seconds
        if envelope is not None and (
            not math.isfinite(float(envelope)) or float(envelope) <= 0.0
        ):
            raise ValueError(
                "response_acquisition_envelope_seconds must be positive "
                "and finite when declared"
            )
        if str(self.oracle_sampling_law) not in ORACLE_SAMPLING_LAWS:
            raise ValueError(
                "oracle_sampling_law must be one of "
                f"{ORACLE_SAMPLING_LAWS!r}, got {self.oracle_sampling_law!r}"
            )
        oracle_envelope = self.oracle_acquisition_envelope_seconds
        if oracle_envelope is not None and (
            not math.isfinite(float(oracle_envelope))
            or float(oracle_envelope) <= 0.0
        ):
            raise ValueError(
                "oracle_acquisition_envelope_seconds must be positive "
                "and finite when declared"
            )


    @property
    def receipt(self) -> Mapping[str, object]:
        """Every declared field, so the hash below covers all of them.

        Built by reflection over the dataclass rather than by listing names: a
        field added later and forgotten here would produce two policies that
        differ in behaviour and agree in hash, which is the one failure this
        object exists to prevent.
        """

        return {
            "policy_id": "irg_independent_sample_size_policy_v1",
            **{
                item.name: getattr(self, item.name)
                for item in dataclasses.fields(self)
            },
        }

    @property
    def acquisition_policy_hash(self) -> str:
        """Bind an arm's receipt to the exact acquisition policy that made it.

        The fleet varies knobs that change WHAT IS CERTIFIED, not merely how
        fast it is reached -- consequence gating changes the order criterion's
        basis, the response envelope changes what an unresolved arm discloses,
        and the decay schedule changes the audited cut ladder. Two arms differing
        in any of them are not two samples of one condition, and nothing in a
        receipt currently says so: the arm id encodes family/query/granularity/
        direction and stops there.

        Prefixed so a hash can never be mistaken for a route-axis or mask hash
        when it appears as a bare string in a comparison.
        """

        return content_hash(self.receipt, prefix="acquisition-policy-")


@dataclass(frozen=True, slots=True)
class OracleSequentialPrecisionPolicy:
    """Predict the next exact-mask packet needed to resolve a BCE boundary.

    The target uses the paired empirical variance and the distance from the
    frozen threshold.  It is allocation-only: the actual accept/reject verdict
    remains the planned-look confidence interval in ``AdaptiveBoundarySearch``.
    Near-zero observed gaps therefore request the remaining horizon rather
    than manufacturing a decision.
    """

    confidence_level: float = 0.95
    initial_packet: int = 32
    minimum_packet: int = 16
    maximum_packet: int = 256
    maximum_contexts_per_candidate: int | None = 4096
    planned_looks: int | None = 32
    precision_only_iid: bool = False
    minimum_numerical_gap: float = 1e-8

    def __post_init__(self) -> None:
        if not 0.5 < float(self.confidence_level) < 1.0:
            raise ValueError("confidence_level must lie in (0.5,1)")
        if min(
            int(self.initial_packet),
            int(self.minimum_packet),
            int(self.maximum_packet),
        ) <= 0:
            raise ValueError("oracle packet controls must be positive")
        if self.precision_only_iid:
            if (
                self.maximum_contexts_per_candidate is not None
                or self.planned_looks is not None
            ):
                raise ValueError(
                    "precision-only IID oracle cannot declare sample/look horizons"
                )
        elif (
            self.maximum_contexts_per_candidate is None
            or int(self.maximum_contexts_per_candidate) <= 0
            or self.planned_looks is None
            or int(self.planned_looks) <= 0
        ):
            raise ValueError("oracle horizon and planned looks must be positive")
        if int(self.minimum_packet) > int(self.maximum_packet):
            raise ValueError("oracle minimum packet may not exceed maximum packet")
        if (
            self.maximum_contexts_per_candidate is not None
            and int(self.initial_packet) > int(self.maximum_contexts_per_candidate)
        ):
            raise ValueError("oracle initial packet exceeds its context horizon")
        gap = float(self.minimum_numerical_gap)
        if not math.isfinite(gap) or gap <= 0.0:
            raise ValueError("minimum_numerical_gap must be finite and positive")

    def alpha_for_look(self, look_index: int) -> float:
        index = int(look_index)
        if index < 0:
            raise ValueError("look_index must be nonnegative")
        total = 1.0 - float(self.confidence_level)
        if self.precision_only_iid:
            return total / float((index + 1) * (index + 2))
        assert self.planned_looks is not None
        return float(staged_alpha(total, planned_looks=int(self.planned_looks)))

    def allocation_critical_value(self, look_index: int = 0) -> float:
        # The inference primitive uses a Student-t interval.  A normal
        # approximation is used only to forecast sample demand and is slightly
        # enlarged by the planned-look alpha split.
        return simultaneous_normal_critical_value(
            1.0 - self.alpha_for_look(look_index),
            family_size=1,
        )

    def target_total(
        self,
        values: Sequence[float] | np.ndarray,
        *,
        threshold: float,
    ) -> int:
        observations = np.asarray(values, dtype=np.float64).reshape(-1)
        if not np.isfinite(observations).all():
            raise ValueError("oracle precision observations must be finite")
        count = int(observations.size)
        horizon = (
            None
            if self.maximum_contexts_per_candidate is None
            else int(self.maximum_contexts_per_candidate)
        )
        if count == 0:
            return (
                int(self.initial_packet)
                if horizon is None
                else min(int(self.initial_packet), horizon)
            )
        if horizon is not None and count >= horizon:
            return horizon
        if count < 2:
            target = max(
                int(self.initial_packet), count + int(self.minimum_packet)
            )
            return target if horizon is None else min(horizon, target)
        variance = max(float(observations.var(ddof=1)), 0.0)
        mean = float(observations.mean())
        gap = abs(mean - float(threshold))
        if gap <= float(self.minimum_numerical_gap):
            target = max(
                count + int(self.minimum_packet),
                2 * count,
            )
            return target if horizon is None else min(horizon, target)
        if variance <= 0.0:
            return count
        critical = float(self.allocation_critical_value())
        target = int(math.ceil(variance * (critical / gap) ** 2))
        target = max(count, target)
        return target if horizon is None else min(horizon, target)

    def target_total_from_estimate(
        self,
        *,
        count: int,
        mean: float,
        standard_error: float,
        threshold: float,
    ) -> int:
        """Forecast total rows from the active aggregate contrast estimate.

        This is the appropriate path for unequal stratified Neyman panels:
        ``mean`` and ``standard_error`` already project the two stratum means
        through their frozen risk weights.
        """

        observed = int(count)
        horizon = (
            None
            if self.maximum_contexts_per_candidate is None
            else int(self.maximum_contexts_per_candidate)
        )
        if observed <= 0:
            return (
                int(self.initial_packet)
                if horizon is None
                else min(int(self.initial_packet), horizon)
            )
        if horizon is not None and observed >= horizon:
            return horizon
        estimate = float(mean)
        error = float(standard_error)
        if not math.isfinite(estimate) or error < 0.0:
            raise ValueError("oracle aggregate estimate is invalid")
        if not math.isfinite(error):
            target = max(
                int(self.initial_packet), observed + int(self.minimum_packet)
            )
            return target if horizon is None else min(horizon, target)
        gap = abs(estimate - float(threshold))
        if gap <= float(self.minimum_numerical_gap):
            target = max(
                observed + int(self.minimum_packet),
                2 * observed,
            )
            return target if horizon is None else min(horizon, target)
        if error <= 0.0:
            return observed
        critical = float(self.allocation_critical_value())
        target = int(
            math.ceil(
                float(observed) * (critical * error / gap) ** 2
            )
        )
        target = max(observed, target)
        return target if horizon is None else min(horizon, target)

    def next_packet_from_estimates(
        self,
        estimates_by_candidate: Mapping[str, tuple[int, float, float]],
        *,
        candidate_ids: Sequence[str],
        threshold: float,
    ) -> int:
        identifiers = tuple(str(value) for value in candidate_ids)
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError("oracle packet planning requires unique candidates")
        required = 0
        for candidate_id in identifiers:
            count, mean, standard_error = estimates_by_candidate.get(
                candidate_id, (0, 0.0, math.inf)
            )
            target = self.target_total_from_estimate(
                count=int(count),
                mean=float(mean),
                standard_error=float(standard_error),
                threshold=float(threshold),
            )
            required = max(required, target - int(count))
        if required <= 0:
            return 0
        requested = max(int(self.minimum_packet), int(required))
        return (
            requested
            if self.precision_only_iid
            else min(int(self.maximum_packet), requested)
        )

    def next_packet(
        self,
        observations_by_candidate: Mapping[
            str, Sequence[float] | np.ndarray
        ],
        *,
        candidate_ids: Sequence[str],
        threshold: float,
    ) -> int:
        identifiers = tuple(str(value) for value in candidate_ids)
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError("oracle packet planning requires unique candidates")
        required = 0
        for candidate_id in identifiers:
            values = np.asarray(
                observations_by_candidate.get(candidate_id, ()),
                dtype=np.float64,
            ).reshape(-1)
            current = int(values.size)
            target = self.target_total(values, threshold=threshold)
            required = max(required, target - current)
        if required <= 0:
            return 0
        requested = max(int(self.minimum_packet), int(required))
        return (
            requested
            if self.precision_only_iid
            else min(int(self.maximum_packet), requested)
        )

    @property
    def receipt(self) -> Mapping[str, object]:
        return {
            "policy_id": "paired_bce_boundary_variance_gap_allocation_v2",
            "confidence_level": float(self.confidence_level),
            "first_look_alpha": float(self.alpha_for_look(0)),
            "allocation_critical_value": float(
                self.allocation_critical_value()
            ),
            "initial_packet": int(self.initial_packet),
            "minimum_packet": int(self.minimum_packet),
            "maximum_packet": int(self.maximum_packet),
            "maximum_contexts_per_candidate": (
                None
                if self.maximum_contexts_per_candidate is None
                else int(self.maximum_contexts_per_candidate)
            ),
            "planned_looks": (
                None if self.planned_looks is None else int(self.planned_looks)
            ),
            "precision_only_iid": bool(self.precision_only_iid),
            "minimum_numerical_gap": float(self.minimum_numerical_gap),
            "allocation_only": True,
            "stratified_aggregate_estimates_supported": True,
        }


__all__ = [
    "IndependentSampleSizePolicy",
    "OracleSequentialPrecisionPolicy",
    "simultaneous_normal_critical_value",
]
