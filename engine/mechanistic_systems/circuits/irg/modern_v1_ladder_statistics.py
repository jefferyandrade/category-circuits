"""Anytime-valid finite-population certification of a frozen v1 action ladder.

Corrected-attribution discovery may choose a nested structural ladder, but it
may not certify the ladder with the same outcomes.  This module implements the
causal seam

``discovery -> ladder freeze -> fresh scalar projection acquisition``.

For every frozen endpoint ``S`` and fresh context ``x`` the runtime supplies
the signed binary64 raw-target-logit projection center ``C_S(x)``.  Production
records also bind a numerical radius enclosing the corresponding exact-real
sum.  The certifier derives conservative pointwise upper observations for

``Y_S(x) = |C_S(x)|`` and ``Z_S(x) = C_S(x)^2``

from the center/radius interval.  Historical v1 point records retain their
registered point estimand for exact replay.  A pre-outcome analytic route
envelope gives the deterministic bounds
``Y_S <= sum_{j in S} U_j`` and ``Z_S <= (sum_{j in S} U_j)^2``.  Each
endpoint/role/stratum cell is then handled by the repository's bounded
sampling-without-replacement betting confidence sequence.  Bonferroni family
allocation is paid once from a trajectory-wide, nonrecycling alpha ledger;
arbitrary inspections and predictable interleaving of sealed stratum prefixes
do not spend a new look-wise alpha.  Numerical enclosures preserve only the
upper F/Q confidence authority used for mask admission: lower bounds are
withdrawn, and even a completed population scan is not called an exact census
unless the nonlinear transforms are point-exact.

The resulting decision has proposal authority only.  It can choose the
largest endpoint whose simultaneous upper bounds fit supplied F/Q budgets; it
can never authorize an irreversible mask commit or substitute for the paired
BCE oracle.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
import hashlib
import math
from pathlib import Path
from typing import Final, Literal

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import (
    HardMask,
    RemovedRouteIndexSet,
)
from mechanistic_systems.evidence.finite_population import (
    WOR_BETTING_CS_METHOD,
    BoundedWoRBettingCS,
)
from mechanistic_systems.evidence.glocal_contracts import (
    content_hash as sampling_content_hash,
)

from .modern_v1_candidate_composition import V1PackedRemovalAction
from .modern_v1_alexnet_fq_registration import (
    V1AlexNetFQCausalGenerationFrameReceipt,
    V1AlexNetFQCausalRoleSamplingStreamReceipt,
    V1AlexNetFQDiscoveryPrefixFreezeReceipt,
)
from .modern_v1_proposal_statistics import (
    ANALYTIC_ENVELOPE_DERIVATION_KIND,
    FIXED_TRAJECTORY_FWER_POLICY,
    SEALED_WITHOUT_REPLACEMENT_SAMPLING_LAW,
    V1CorrectedAttributionEnvelopeCertificate,
)


FROZEN_LADDER_ACTION_SCHEMA: Final = "modern_v1_frozen_ladder_action_v1"
FROZEN_CANDIDATE_LADDER_SCHEMA: Final = "modern_v1_frozen_candidate_ladder_v1"
FROZEN_LADDER_PACKED_ACTION_SCHEMA: Final = "modern_v1_frozen_ladder_action_v2"
FROZEN_PACKED_CANDIDATE_LADDER_SCHEMA: Final = "modern_v1_frozen_candidate_ladder_v2"
FROZEN_LADDER_CERTIFICATION_DESIGN_SCHEMA: Final = (
    "modern_v1_frozen_ladder_certification_design_v1"
)
FROZEN_LADDER_ENDPOINT_ENVELOPE_CERTIFICATE_SCHEMA: Final = (
    "modern_v1_frozen_ladder_endpoint_envelope_certificate_v1"
)
FROZEN_LADDER_PROJECTION_RECORD_SCHEMA: Final = (
    "modern_v1_frozen_ladder_projection_record_v1"
)
FROZEN_LADDER_ENCLOSED_PROJECTION_RECORD_SCHEMA: Final = (
    "modern_v1_frozen_ladder_projection_record_v2"
)
FROZEN_LADDER_CELL_INTERVAL_SCHEMA: Final = "modern_v1_frozen_ladder_cell_interval_v1"
FROZEN_LADDER_ENCLOSED_CELL_INTERVAL_SCHEMA: Final = (
    "modern_v1_frozen_ladder_cell_interval_v2"
)
FROZEN_LADDER_ENDPOINT_INTERVAL_SCHEMA: Final = (
    "modern_v1_frozen_ladder_endpoint_interval_v1"
)
FROZEN_LADDER_ENCLOSED_ENDPOINT_INTERVAL_SCHEMA: Final = (
    "modern_v1_frozen_ladder_endpoint_interval_v2"
)
FROZEN_LADDER_CERTIFICATION_RECEIPT_SCHEMA: Final = (
    "modern_v1_frozen_ladder_certification_receipt_v1"
)
FROZEN_LADDER_ENCLOSED_CERTIFICATION_RECEIPT_SCHEMA: Final = (
    "modern_v1_frozen_ladder_certification_receipt_v2"
)
FQ_PROPOSAL_BUDGET_SCHEMA: Final = "modern_v1_fq_proposal_budget_v1"
FROZEN_LADDER_DECISION_SCHEMA: Final = "modern_v1_frozen_ladder_decision_v1"

F_ROLE: Final = "F_absolute_raw_target_logit"
Q_ROLE: Final = "Q_squared_raw_target_logit"
CERTIFICATION_ROLES: Final = (F_ROLE, Q_ROLE)
F_UNITS: Final = "absolute_raw_target_logit"
Q_UNITS: Final = "squared_raw_target_logit"
SQRT_Q_UNITS: Final = "absolute_raw_target_logit"
PROPOSAL_AUTHORITY: Final = "proposal_fq_certification_only"
ZERO_ENVELOPE_METHOD: Final = "analytic_zero_envelope_constant_v1"
NUMERICALLY_ENCLOSED_WOR_UPPER_CS_METHOD: Final = (
    "bounded_wor_numerically_enclosed_upper_observation_v1"
)
EXACT_POINT_OBSERVATION_SEMANTICS: Final = (
    "registered_signed_projection_point_observation_v1"
)
NUMERICALLY_ENCLOSED_UPPER_OBSERVATION_SEMANTICS: Final = (
    "exact_real_projection_functional_upper_envelope_observation_v1"
)
UNIFORM_CELL_ALPHA_POLICY: Final = (
    "uniform_bonferroni_endpoint_role_stratum_nonrecycling_v1"
)
ROUTE_SUM_ENDPOINT_ENVELOPE_KIND: Final = "singleton_route_sum_endpoint_envelope_v1"
ACTION_SPECIFIC_ENDPOINT_ENVELOPE_KIND: Final = (
    "action_specific_frozen_endpoint_projection_envelope_v1"
)
ACTION_ROUTE_SUM_ENDPOINT_ENVELOPE_KIND: Final = (
    "action_explicit_route_sum_endpoint_envelope_v1"
)
ENDPOINT_ENVELOPE_DERIVATION_KINDS: Final = (
    ACTION_SPECIFIC_ENDPOINT_ENVELOPE_KIND,
    ACTION_ROUTE_SUM_ENDPOINT_ENVELOPE_KIND,
)
FROZEN_LADDER_ROUTE_SUM_ENDPOINT_ENVELOPE_PROOF_SCHEMA: Final = (
    "modern_v1_frozen_ladder_route_sum_endpoint_envelope_proof_v1"
)
_ENDPOINT_ENVELOPE_PROOF_PREFIX: Final = (
    "modern-v1-frozen-ladder-route-sum-endpoint-envelope-proof-"
)
_PACKED_LADDER_SCAN_BYTE_CHUNK_SIZE: Final = 1 << 16
_LEGACY_LADDER_INDEX_CHUNK_SIZE: Final = 1 << 18

DecisionStatus = Literal[
    "certified_largest_endpoint",
    "refuse_no_certified_endpoint",
    "refuse_inconsistent_confidence_sequence",
]


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be one nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be one nonnegative integer")
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be one positive integer")
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _finite_unit_interval(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie strictly inside (0,1)")
    return result


def _assert_content_integrity(
    owner: object,
    *,
    payload: object,
    cache_name: str,
    label: str,
    expected_hash: str | None = None,
) -> str:
    observed = content_hash(payload)
    cached = getattr(owner, cache_name)
    if cached is not None and observed != cached:
        raise ValueError(f"{label} was tampered after hashing")
    if expected_hash is not None and observed != expected_hash:
        raise ValueError(f"{label} hash does not match")
    object.__setattr__(owner, cache_name, observed)
    return observed


def _coordinate_payload(value: object) -> dict[str, object]:
    return {
        name: getattr(value, name)
        for name in (
            "campaign_contract_hash",
            "model_identity_hash",
            "query_contract_hash",
            "preprocessing_hash",
            "current_mask_hash",
            "raw_logit_stream_source_hash",
            "sampling_law_hash",
            "proposal_generation",
            "route_count",
            "route_axis_hash",
        )
    }


def _validate_coordinates(value: object) -> None:
    for name in (
        "campaign_contract_hash",
        "model_identity_hash",
        "query_contract_hash",
        "preprocessing_hash",
        "current_mask_hash",
        "raw_logit_stream_source_hash",
        "sampling_law_hash",
        "route_axis_hash",
    ):
        object.__setattr__(value, name, _identity(getattr(value, name), name=name))
    object.__setattr__(
        value,
        "proposal_generation",
        _nonnegative_int(
            getattr(value, "proposal_generation"), name="proposal generation"
        ),
    )
    object.__setattr__(
        value,
        "route_count",
        _positive_int(getattr(value, "route_count"), name="route count"),
    )


def _coordinates_match(left: object, right: object) -> bool:
    return _coordinate_payload(left) == _coordinate_payload(right)


def _direct_hard_mask_hash(mask: HardMask, *, name: str) -> str:
    """Validate and hash live packed bytes without trusting a cached property."""

    if type(mask) is not HardMask:
        raise TypeError(f"{name} must use the exact HardMask type")
    route_count = _positive_int(mask.route_count, name=f"{name} route count")
    route_axis_hash = _identity(
        mask.route_axis_hash,
        name=f"{name} route axis hash",
    )
    packed = mask.packed_bits
    if type(packed) is not bytes or len(packed) != (route_count + 7) // 8:
        raise ValueError(f"{name} packed byte length differs from its route count")
    padding = len(packed) * 8 - route_count
    if padding and packed[-1] & ((1 << padding) - 1):
        raise ValueError(f"{name} contains nonzero padding bits")
    return content_hash(
        {
            "route_axis_hash": route_axis_hash,
            "route_count": route_count,
            "packed_bits": packed,
        },
        prefix="irg-mask-",
    )


def _stream_removed_index_count_and_hash(
    *,
    child: HardMask,
    parent: HardMask,
) -> tuple[int, str]:
    """Content-address one packed delta while retaining only a bounded slab."""

    dtype_code = "u32" if child.route_count <= np.iinfo(np.uint32).max else "u64"
    digest = hashlib.sha256()
    count = 0
    for chunk in child.iter_removed_index_chunks_from(
        parent,
        packed_byte_chunk_size=_PACKED_LADDER_SCAN_BYTE_CHUNK_SIZE,
    ):
        digest.update(memoryview(chunk).cast("B"))
        count += int(chunk.size)
    return (
        count,
        RemovedRouteIndexSet.index_hash_from_encoded_digest(
            route_count=child.route_count,
            count=count,
            dtype_code=dtype_code,
            encoded_sha256=digest.hexdigest(),
        ),
    )


def _axis_pairs(
    value: Sequence[tuple[str, Sequence[str]]], *, name: str
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    result = tuple(
        (
            _identity(axis_name, name=f"{name} axis name"),
            tuple(_identity(item, name=f"{name} item") for item in tuple(axis_values)),
        )
        for axis_name, axis_values in tuple(value)
    )
    if not result or len({axis_name for axis_name, _ in result}) != len(result):
        raise ValueError(f"{name} must have unique nonempty axes")
    return result


def _weight_pairs(
    value: Sequence[tuple[str, float]], *, expected_names: tuple[str, ...]
) -> tuple[tuple[str, float], ...]:
    result = tuple(
        (_identity(name, name="stratum weight name"), float(weight))
        for name, weight in tuple(value)
    )
    if tuple(name for name, _ in result) != expected_names:
        raise ValueError("stratum weight axis differs from the analytic certificate")
    if any(not math.isfinite(weight) or weight <= 0.0 for _, weight in result):
        raise ValueError("stratum weights must be finite and positive")
    total = math.fsum(weight for _, weight in result)
    if not math.isclose(total, 1.0, abs_tol=1e-12, rel_tol=0.0):
        raise ValueError("stratum weights must sum to one")
    return tuple((name, weight / total) for name, weight in result)


def _string_axis(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(_identity(value, name=f"{name} item") for value in tuple(values))
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique and nonempty")
    return result


def _endpoint_bound_rows(
    values: Sequence[Sequence[float]],
    *,
    stratum_count: int,
    endpoint_count: int,
) -> tuple[tuple[float, ...], ...]:
    rows = tuple(
        tuple(
            _finite_nonnegative(value, name="endpoint absolute upper bound")
            for value in tuple(row)
        )
        for row in tuple(values)
    )
    if len(rows) != stratum_count or any(len(row) != endpoint_count for row in rows):
        raise ValueError("endpoint envelope bounds must align to strata and actions")
    return rows


def _conservative_nonnegative_block_sum(values: np.ndarray) -> float:
    """Fast upward enclosure for a finite sum of nonnegative binary64s.

    NumPy reduction is substantially cheaper than Python iteration on the
    ImageNet-scale route axis.  The ordinary ``gamma_(n-1)`` floating-point
    error bound is deliberately used even though NumPy's pairwise reduction
    is normally much tighter.  A separate absolute allowance covers
    subnormal flush-to-zero execution, and one final outward rounding keeps
    the result an upper envelope rather than a numerical estimate.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError("route envelope blocks must be finite and nonnegative")
    count = int(array.size)
    if count == 0:
        return 0.0
    magnitude_bits = np.bitwise_and(
        array.view(np.uint64),
        np.uint64(0x7FFF_FFFF_FFFF_FFFF),
    )
    positive_count = int(np.count_nonzero(magnitude_bits))
    if positive_count == 0:
        return 0.0
    if count == 1:
        # No arithmetic is needed, so the registered binary64 upper bound is
        # already an exact enclosure even when it is subnormal.
        return float(array[0])
    reduced = float(np.sum(array, dtype=np.float64))
    if not math.isfinite(reduced):
        raise ValueError("route envelope sum overflowed binary64")
    unit_roundoff = float(np.finfo(np.float64).eps) / 2.0
    product = (count - 1) * unit_roundoff
    if product >= 0.5:
        raise ValueError("route envelope axis is too large for certified summation")
    gamma = math.nextafter(product / (1.0 - product), math.inf)
    denominator = math.nextafter(1.0 - gamma, -math.inf)
    if denominator <= 0.0:
        raise ValueError("route envelope summation error bound is undefined")
    upper = reduced / denominator
    # The relative gamma bound assumes no underflow.  Add one minimum-normal
    # allowance per nonzero input so the certificate remains outward under a
    # flush-to-zero/denormals-are-zero execution mode as well.  This term is
    # numerically immaterial for ordinary envelopes and deliberately
    # conservative for pathological subnormal-only blocks.
    underflow_allowance = math.nextafter(
        positive_count * float(np.finfo(np.float64).tiny),
        math.inf,
    )
    upper += underflow_allowance
    if not math.isfinite(upper):
        raise ValueError("route envelope upper sum overflowed binary64")
    return math.nextafter(upper, math.inf)


def _outward_nonnegative_add(left: float, right: float) -> float:
    if left == 0.0:
        return right
    if right == 0.0:
        return left
    result = left + right
    if not math.isfinite(result):
        raise ValueError("endpoint envelope sum overflowed binary64")
    return math.nextafter(result, math.inf)


def _outward_nonnegative_square(value: float) -> float:
    if value == 0.0:
        return 0.0
    if value > math.sqrt(float(np.finfo(np.float64).max)):
        raise ValueError("squared endpoint envelope overflowed binary64")
    result = value * value
    if not math.isfinite(result):
        raise ValueError("squared endpoint envelope overflowed binary64")
    return math.nextafter(result, math.inf)


def _outward_projection_functional_bounds(
    *, center: float, radius: float, analytic_absolute_bound: float
) -> tuple[float, float, float, float]:
    """Enclose ``|x|`` and ``x**2`` for ``x in [center-radius, center+radius]``.

    The analytic endpoint envelope is an independent exact-real bound.  It is
    therefore sound (and tighter) to intersect the numerical interval with
    that support before constructing the pointwise F/Q upper observations.
    Every nontrivial arithmetic boundary is rounded outwards.
    """

    center_f = float(center)
    radius_f = _finite_nonnegative(radius, name="projection numerical radius")
    bound = _finite_nonnegative(
        analytic_absolute_bound,
        name="analytic endpoint absolute bound",
    )
    if not math.isfinite(center_f):
        raise ValueError("projection center must be finite")

    if radius_f == 0.0:
        absolute = abs(center_f)
        if absolute > bound:
            raise ValueError(
                "signed endpoint projection violated its analytic endpoint envelope"
            )
        squared = center_f * center_f
        squared_bound = _outward_nonnegative_square(bound)
        if not math.isfinite(squared) or squared > squared_bound:
            raise ValueError(
                "squared endpoint projection violated its analytic endpoint envelope"
            )
        if squared == 0.0:
            return absolute, absolute, 0.0, 0.0
        return (
            absolute,
            absolute,
            max(0.0, math.nextafter(squared, -math.inf)),
            min(squared_bound, math.nextafter(squared, math.inf)),
        )

    raw_lower = center_f - radius_f
    raw_upper = center_f + radius_f
    lower = (
        -math.inf
        if not math.isfinite(raw_lower)
        else math.nextafter(raw_lower, -math.inf)
    )
    upper = (
        math.inf
        if not math.isfinite(raw_upper)
        else math.nextafter(raw_upper, math.inf)
    )
    support_lower = max(-bound, lower)
    support_upper = min(bound, upper)
    if support_lower > support_upper:
        raise ValueError(
            "projection numerical enclosure is disjoint from its analytic endpoint envelope"
        )

    if support_lower <= 0.0 <= support_upper:
        f_lower = 0.0
    else:
        nearest = min(abs(support_lower), abs(support_upper))
        f_lower = max(0.0, math.nextafter(nearest, -math.inf))
    farthest = max(abs(support_lower), abs(support_upper))
    f_upper = min(bound, math.nextafter(farthest, math.inf))
    q_lower = (
        0.0
        if f_lower == 0.0
        else max(0.0, math.nextafter(f_lower * f_lower, -math.inf))
    )
    q_bound = _outward_nonnegative_square(bound)
    q_upper = min(q_bound, _outward_nonnegative_square(f_upper))
    return f_lower, f_upper, q_lower, q_upper


def _route_sum_endpoint_bounds(
    *,
    ladder: "V1FrozenCandidateLadder",
    certificate: V1CorrectedAttributionEnvelopeCertificate,
) -> tuple[tuple[float, ...], ...]:
    """Outward-rounded singleton-route envelope sums over frozen endpoints."""

    stratum_names = tuple(certificate.stratum_names)
    endpoint_bounds: list[list[float]] = [[] for _ in range(len(stratum_names))]
    running = [0.0 for _ in stratum_names]
    envelope = certificate.absolute_upper_envelope_by_stratum
    for action in ladder.actions:
        for route_index in ladder.iter_shell_index_chunks(
            endpoint_index=action.endpoint_index
        ):
            for stratum_index in range(len(stratum_names)):
                block = _conservative_nonnegative_block_sum(
                    envelope[stratum_index, route_index]
                )
                running[stratum_index] = _outward_nonnegative_add(
                    running[stratum_index], block
                )
        for stratum_index in range(len(stratum_names)):
            _outward_nonnegative_square(running[stratum_index])
            endpoint_bounds[stratum_index].append(running[stratum_index])
    return tuple(tuple(row) for row in endpoint_bounds)


def _nonoverspending_uniform_cell_alpha(*, allocation: float, cell_count: int) -> float:
    result = allocation / cell_count
    while result * cell_count > allocation:
        result = math.nextafter(result, 0.0)
    if not 0.0 < result < 1.0:
        raise ValueError("family allocation is too small for binary64 cell alpha")
    return result


def _weighted_cell_projection(
    cells: Sequence["V1FrozenLadderCellInterval"],
    *,
    weights: Mapping[str, float],
) -> tuple[float, float]:
    if all(item.full_census and not item.empty for item in cells):
        census = math.fsum(
            weights[item.stratum_id] * float(item.sample_mean) for item in cells
        )
        return census, census
    lower = (
        math.fsum(weights[item.stratum_id] * item.lower for item in cells)
        if all(item.lower_confidence_authority for item in cells)
        else 0.0
    )
    upper = math.fsum(weights[item.stratum_id] * item.upper for item in cells)
    if lower > 0.0:
        lower = max(0.0, math.nextafter(lower, -math.inf))
    if upper > 0.0:
        upper = math.nextafter(upper, math.inf)
    return lower, upper


def _sqrt_interval(lower: float, upper: float) -> tuple[float, float]:
    if lower == upper:
        value = math.sqrt(max(0.0, lower))
        return value, value
    sqrt_lower = math.sqrt(max(0.0, lower))
    sqrt_upper = math.sqrt(max(0.0, upper))
    if sqrt_lower > 0.0:
        sqrt_lower = max(0.0, math.nextafter(sqrt_lower, -math.inf))
    if sqrt_upper > 0.0:
        sqrt_upper = math.nextafter(sqrt_upper, math.inf)
    return sqrt_lower, sqrt_upper


@dataclass(frozen=True, slots=True)
class V1FrozenLadderAction:
    """One typed endpoint in a strictly nested irreversible route family.

    Schema v1 retains the historical explicit route-order prefix.  Schema v2
    binds one exact packed child mask plus its cumulative parent-to-child
    delta identity.  The v2 representation is constant-size in the number of
    removed routes and never expands that delta into a Python tuple.
    """

    endpoint_index: int
    action_id: str
    removed_route_indices: tuple[int, ...] = ()
    schema: str = FROZEN_LADDER_ACTION_SCHEMA
    _action_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )
    # This field is deliberately appended after every v1 slot.  The custom
    # pickle state adapter below can therefore load historical v1 objects
    # without reordering any previously serialized coordinate.
    packed_removal_action: V1PackedRemovalAction | None = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        endpoint = _positive_int(self.endpoint_index, name="endpoint index")
        action_id = _identity(self.action_id, name="action id")
        raw_routes = tuple(self.removed_route_indices)
        routes = tuple(
            _nonnegative_int(route, name="removed route index") for route in raw_routes
        )
        packed = self.packed_removal_action
        if packed is None:
            if not routes or len(routes) != len(set(routes)):
                raise ValueError(
                    "a legacy frozen action requires unique nonempty route indices"
                )
            if self.schema != FROZEN_LADDER_ACTION_SCHEMA:
                raise ValueError("legacy frozen ladder action schema is invalid")
        else:
            if type(packed) is not V1PackedRemovalAction:
                raise TypeError("a packed frozen action requires V1PackedRemovalAction")
            if routes:
                raise ValueError(
                    "a packed frozen action cannot retain an explicit route tuple"
                )
            if self.schema not in {
                FROZEN_LADDER_ACTION_SCHEMA,
                FROZEN_LADDER_PACKED_ACTION_SCHEMA,
            }:
                raise ValueError("packed frozen ladder action schema is invalid")
            # The historical schema is the constructor default.  Presence of
            # the exact packed action is an unambiguous request for schema v2.
            object.__setattr__(self, "schema", FROZEN_LADDER_PACKED_ACTION_SCHEMA)
            _direct_hard_mask_hash(
                packed.resulting_mask,
                name="packed frozen action child mask",
            )
        object.__setattr__(self, "endpoint_index", endpoint)
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "removed_route_indices", routes)
        self.assert_integrity()

    def _hash_payload(self) -> dict[str, object]:
        packed = getattr(self, "packed_removal_action", None)
        if packed is None:
            # Exact historical v1 payload: do not add a representation tag or
            # a null packed field, because either would rewrite replay hashes.
            return {
                "schema": self.schema,
                "endpoint_index": self.endpoint_index,
                "action_id": self.action_id,
                "removed_route_indices": self.removed_route_indices,
            }
        child = packed.resulting_mask
        return {
            "schema": self.schema,
            "endpoint_index": self.endpoint_index,
            "action_id": self.action_id,
            "packed_action_hash": packed.action_hash,
            "resulting_mask_hash": _direct_hard_mask_hash(
                child,
                name="packed frozen action child mask",
            ),
            "route_axis_hash": child.route_axis_hash,
            "route_count": child.route_count,
            "removed_count": packed.removed_count,
            "removed_route_index_hash": packed.removed_route_index_hash,
        }

    @property
    def is_packed(self) -> bool:
        return getattr(self, "packed_removal_action", None) is not None

    @property
    def cumulative_removed_count(self) -> int:
        packed = getattr(self, "packed_removal_action", None)
        return (
            len(self.removed_route_indices) if packed is None else packed.removed_count
        )

    @property
    def resulting_mask(self) -> HardMask:
        packed = getattr(self, "packed_removal_action", None)
        if packed is None:
            raise ValueError("a legacy tuple action has no bound resulting mask")
        return packed.resulting_mask

    @property
    def action_hash(self) -> str:
        cached = self._action_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        return _assert_content_integrity(
            self,
            payload=self._hash_payload(),
            cache_name="_action_hash",
            label="frozen ladder action",
            expected_hash=expected_hash,
        )

    def __getstate__(self) -> list[object]:
        return [getattr(self, item.name) for item in fields(type(self))]

    def __setstate__(self, state: Sequence[object]) -> None:
        names = tuple(item.name for item in fields(type(self)))
        values = list(state)
        if len(values) == len(names) - 1:
            # Historical v1 state predates the trailing packed action slot.
            values.append(None)
        if len(values) != len(names):
            raise ValueError("frozen ladder action pickle state has invalid arity")
        for name, value in zip(names, values, strict=True):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class V1FrozenCandidateLadder:
    """Discovery-selected ladder frozen before any certification outcome."""

    campaign_contract_hash: str
    model_identity_hash: str
    query_contract_hash: str
    preprocessing_hash: str
    current_mask_hash: str
    raw_logit_stream_source_hash: str
    sampling_law_hash: str
    proposal_generation: int
    route_count: int
    route_axis_hash: str
    actions: tuple[V1FrozenLadderAction, ...]
    discovery_context_ids: tuple[str, ...]
    discovery_draw_ids: tuple[str, ...]
    discovery_evidence_hash: str
    freeze_receipt_hash: str
    frozen_before_certification_outcomes: bool = True
    schema: str = FROZEN_CANDIDATE_LADDER_SCHEMA
    _ladder_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )
    # Appended after all v1 slots for historical pickle-state compatibility.
    parent_mask: HardMask | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _validate_coordinates(self)
        actions = tuple(self.actions)
        if not actions:
            raise ValueError("a frozen candidate ladder requires at least one action")
        if any(type(action) is not V1FrozenLadderAction for action in actions):
            raise TypeError("frozen candidate ladder actions must use the typed action")
        if tuple(action.endpoint_index for action in actions) != tuple(
            range(1, len(actions) + 1)
        ):
            raise ValueError(
                "frozen ladder endpoint indices must be contiguous from one"
            )
        if len({action.action_id for action in actions}) != len(actions):
            raise ValueError("frozen ladder action identities must be unique")
        representation_axis = tuple(action.is_packed for action in actions)
        if len(set(representation_axis)) != 1:
            raise ValueError("a frozen ladder cannot mix tuple and packed actions")
        packed_representation = representation_axis[0]
        parent = self.parent_mask
        if packed_representation:
            if type(parent) is not HardMask:
                raise TypeError("a packed frozen ladder requires its exact parent mask")
            if self.schema not in {
                FROZEN_CANDIDATE_LADDER_SCHEMA,
                FROZEN_PACKED_CANDIDATE_LADDER_SCHEMA,
            }:
                raise ValueError("packed frozen candidate ladder schema is invalid")
            object.__setattr__(
                self,
                "schema",
                FROZEN_PACKED_CANDIDATE_LADDER_SCHEMA,
            )
            if (
                parent.route_count != self.route_count
                or parent.route_axis_hash != self.route_axis_hash
            ):
                raise ValueError("packed ladder parent differs from its route axis")
            if (
                _direct_hard_mask_hash(
                    parent,
                    name="packed frozen ladder parent mask",
                )
                != self.current_mask_hash
            ):
                raise ValueError("packed ladder parent differs from current_mask_hash")
            previous_mask = parent
            previous_count = 0
            for action in actions:
                action.assert_integrity()
                packed_action = action.packed_removal_action
                if packed_action is None:  # pragma: no cover - narrowed above.
                    raise RuntimeError("packed ladder representation changed")
                child = packed_action.resulting_mask
                if (
                    child.route_count != self.route_count
                    or child.route_axis_hash != self.route_axis_hash
                ):
                    raise ValueError("packed action left the registered route axis")
                cumulative_count, cumulative_hash = (
                    _stream_removed_index_count_and_hash(
                        child=child,
                        parent=parent,
                    )
                )
                if (
                    cumulative_count != packed_action.removed_count
                    or cumulative_hash != packed_action.removed_route_index_hash
                ):
                    raise ValueError(
                        "packed frozen action delta identity differs from its parent"
                    )
                shell_count = sum(
                    int(chunk.size)
                    for chunk in child.iter_removed_index_chunks_from(
                        previous_mask,
                        packed_byte_chunk_size=(_PACKED_LADDER_SCAN_BYTE_CHUNK_SIZE),
                    )
                )
                if shell_count <= 0:
                    raise ValueError(
                        "packed frozen candidate actions must be strictly nested"
                    )
                if shell_count != cumulative_count - previous_count:
                    raise ValueError(
                        "packed frozen action cumulative count is inconsistent"
                    )
                previous_mask = child
                previous_count = cumulative_count
        else:
            if parent is not None:
                raise ValueError("a legacy tuple ladder cannot bind a packed parent")
            if self.schema != FROZEN_CANDIDATE_LADDER_SCHEMA:
                raise ValueError("legacy frozen candidate ladder schema is invalid")
            previous: tuple[int, ...] = ()
            for action in actions:
                action.assert_integrity()
                routes = action.removed_route_indices
                if any(route >= self.route_count for route in routes):
                    raise ValueError("frozen action left the registered route axis")
                if len(routes) <= len(previous) or routes[: len(previous)] != previous:
                    raise ValueError(
                        "frozen candidate actions must be strict prefixes "
                        "of one route order"
                    )
                previous = routes

        contexts = tuple(
            _identity(value, name="discovery context id")
            for value in self.discovery_context_ids
        )
        draws = tuple(
            _identity(value, name="discovery draw id")
            for value in self.discovery_draw_ids
        )
        if not contexts or len(contexts) != len(draws):
            raise ValueError(
                "ladder freeze requires aligned nonempty discovery ledgers"
            )
        if len(contexts) != len(set(contexts)) or len(draws) != len(set(draws)):
            raise ValueError("ladder discovery ledgers cannot repeat identities")
        object.__setattr__(
            self,
            "discovery_evidence_hash",
            _identity(self.discovery_evidence_hash, name="discovery evidence hash"),
        )
        object.__setattr__(
            self,
            "freeze_receipt_hash",
            _identity(self.freeze_receipt_hash, name="freeze receipt hash"),
        )
        if self.frozen_before_certification_outcomes is not True:
            raise ValueError(
                "the candidate ladder must freeze before certification outcomes"
            )
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "discovery_context_ids", contexts)
        object.__setattr__(self, "discovery_draw_ids", draws)
        self.assert_integrity()

    @property
    def action_hashes(self) -> tuple[str, ...]:
        return tuple(action.assert_integrity() for action in self.actions)

    @property
    def is_packed(self) -> bool:
        return self.actions[0].is_packed

    @property
    def endpoint_removed_counts(self) -> tuple[int, ...]:
        return tuple(action.cumulative_removed_count for action in self.actions)

    @property
    def endpoint_masks(self) -> tuple[HardMask, ...]:
        if not self.is_packed:
            raise ValueError("a legacy tuple ladder has no bound endpoint masks")
        return tuple(action.resulting_mask for action in self.actions)

    def endpoint_mask(self, *, endpoint_index: int) -> HardMask:
        endpoint = _positive_int(endpoint_index, name="endpoint index")
        if endpoint > len(self.actions):
            raise ValueError("endpoint index left the frozen ladder")
        return self.actions[endpoint - 1].resulting_mask

    def iter_shell_index_chunks(
        self,
        *,
        endpoint_index: int,
        legacy_index_chunk_size: int = _LEGACY_LADDER_INDEX_CHUNK_SIZE,
        packed_byte_chunk_size: int = _PACKED_LADDER_SCAN_BYTE_CHUNK_SIZE,
    ) -> Iterator[np.ndarray]:
        """Yield one disjoint endpoint shell in bounded sorted index chunks.

        Packed schema v2 delegates directly to the canonical hard-mask delta
        scanner.  Legacy schema v1 streams the already-materialized route
        prefix suffix without constructing a second endpoint-sized tuple.
        """

        endpoint = _positive_int(endpoint_index, name="endpoint index")
        if endpoint > len(self.actions):
            raise ValueError("endpoint index left the frozen ladder")
        if self.is_packed:
            parent = (
                self.parent_mask
                if endpoint == 1
                else self.endpoint_mask(endpoint_index=endpoint - 1)
            )
            if parent is None:  # pragma: no cover - constructor invariant.
                raise RuntimeError("packed ladder lost its parent mask")
            yield from self.endpoint_mask(
                endpoint_index=endpoint
            ).iter_removed_index_chunks_from(
                parent,
                packed_byte_chunk_size=_positive_int(
                    packed_byte_chunk_size,
                    name="packed byte chunk size",
                ),
            )
            return

        chunk_size = _positive_int(
            legacy_index_chunk_size,
            name="legacy index chunk size",
        )
        action = self.actions[endpoint - 1]
        routes = action.removed_route_indices
        start = (
            0
            if endpoint == 1
            else len(self.actions[endpoint - 2].removed_route_indices)
        )
        dtype = np.dtype(
            "<u4" if self.route_count <= np.iinfo(np.uint32).max else "<u8"
        )
        for offset in range(start, len(routes), chunk_size):
            stop = min(len(routes), offset + chunk_size)
            chunk = np.fromiter(
                (routes[index] for index in range(offset, stop)),
                dtype=dtype,
                count=stop - offset,
            )
            chunk.setflags(write=False)
            yield chunk

    def _hash_payload(self) -> dict[str, object]:
        payload = {
            "schema": self.schema,
            **_coordinate_payload(self),
            "action_hashes": self.action_hashes,
            "discovery_context_ids": self.discovery_context_ids,
            "discovery_draw_ids": self.discovery_draw_ids,
            "discovery_evidence_hash": self.discovery_evidence_hash,
            "freeze_receipt_hash": self.freeze_receipt_hash,
            "frozen_before_certification_outcomes": (
                self.frozen_before_certification_outcomes
            ),
        }
        if self.is_packed:
            parent = self.parent_mask
            if parent is None:  # pragma: no cover - constructor invariant.
                raise RuntimeError("packed ladder lost its parent mask")
            payload["parent_mask_hash"] = _direct_hard_mask_hash(
                parent,
                name="packed frozen ladder parent mask",
            )
            payload["endpoint_removed_counts"] = self.endpoint_removed_counts
        return payload

    @property
    def ladder_hash(self) -> str:
        cached = self._ladder_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        return _assert_content_integrity(
            self,
            payload=self._hash_payload(),
            cache_name="_ladder_hash",
            label="frozen candidate ladder",
            expected_hash=expected_hash,
        )

    def __getstate__(self) -> list[object]:
        return [getattr(self, item.name) for item in fields(type(self))]

    def __setstate__(self, state: Sequence[object]) -> None:
        names = tuple(item.name for item in fields(type(self)))
        values = list(state)
        if len(values) == len(names) - 1:
            # Historical v1 state predates the trailing parent-mask slot.
            values.append(None)
        if len(values) != len(names):
            raise ValueError("frozen candidate ladder pickle state has invalid arity")
        for name, value in zip(names, values, strict=True):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class V1FrozenLadderEndpointEnvelopeCertificate:
    """Pre-outcome scalar bounds for already-frozen ladder endpoints.

    The singleton route envelope remains the sound fallback.  This certificate
    is the production hook: the current route-sum producer freezes explicit
    action endpoint bounds under a typed proof identity, while a sharper
    action-specific abstract interpreter can later bound the whole frozen
    endpoint projection directly, including cancellations that a sum of
    singleton route bounds necessarily loses.
    """

    campaign_contract_hash: str
    model_identity_hash: str
    query_contract_hash: str
    preprocessing_hash: str
    current_mask_hash: str
    raw_logit_stream_source_hash: str
    sampling_law_hash: str
    proposal_generation: int
    route_count: int
    route_axis_hash: str
    ladder_hash: str
    endpoint_action_hashes: tuple[str, ...]
    stratum_names: tuple[str, ...]
    endpoint_absolute_upper_bounds_by_stratum: tuple[tuple[float, ...], ...]
    verifier_source_hash: str
    proof_artifact_hash: str
    derivation_kind: str = ACTION_SPECIFIC_ENDPOINT_ENVELOPE_KIND
    issued_before_certification_outcomes: bool = True
    score_order_authority: bool = False
    bce_mask_commit_authority: bool = False
    schema: str = FROZEN_LADDER_ENDPOINT_ENVELOPE_CERTIFICATE_SCHEMA
    _certificate_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _validate_coordinates(self)
        object.__setattr__(
            self, "ladder_hash", _identity(self.ladder_hash, name="ladder hash")
        )
        actions = _string_axis(
            self.endpoint_action_hashes, name="endpoint action hashes"
        )
        strata = _string_axis(self.stratum_names, name="endpoint envelope strata")
        bounds = _endpoint_bound_rows(
            self.endpoint_absolute_upper_bounds_by_stratum,
            stratum_count=len(strata),
            endpoint_count=len(actions),
        )
        for name in ("verifier_source_hash", "proof_artifact_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if self.derivation_kind not in ENDPOINT_ENVELOPE_DERIVATION_KINDS:
            raise ValueError("unsupported frozen-ladder endpoint envelope derivation")
        if self.issued_before_certification_outcomes is not True:
            raise ValueError("endpoint envelope must be issued before outcomes")
        if self.score_order_authority is not False:
            raise ValueError("endpoint envelope cannot grant score-order authority")
        if self.bce_mask_commit_authority is not False:
            raise ValueError("endpoint envelope cannot grant BCE commit authority")
        if self.schema != FROZEN_LADDER_ENDPOINT_ENVELOPE_CERTIFICATE_SCHEMA:
            raise ValueError("unsupported endpoint envelope certificate schema")
        object.__setattr__(self, "endpoint_action_hashes", actions)
        object.__setattr__(self, "stratum_names", strata)
        object.__setattr__(self, "endpoint_absolute_upper_bounds_by_stratum", bounds)
        self.assert_integrity()

    def _hash_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            **_coordinate_payload(self),
            "ladder_hash": self.ladder_hash,
            "endpoint_action_hashes": self.endpoint_action_hashes,
            "stratum_names": self.stratum_names,
            "endpoint_absolute_upper_bounds_by_stratum": (
                self.endpoint_absolute_upper_bounds_by_stratum
            ),
            "verifier_source_hash": self.verifier_source_hash,
            "proof_artifact_hash": self.proof_artifact_hash,
            "derivation_kind": self.derivation_kind,
            "issued_before_certification_outcomes": (
                self.issued_before_certification_outcomes
            ),
            "score_order_authority": self.score_order_authority,
            "bce_mask_commit_authority": self.bce_mask_commit_authority,
        }
        return payload

    @property
    def certificate_hash(self) -> str:
        cached = self._certificate_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        return _assert_content_integrity(
            self,
            payload=self._hash_payload(),
            cache_name="_certificate_hash",
            label="frozen ladder endpoint envelope certificate",
            expected_hash=expected_hash,
        )


@dataclass(frozen=True, slots=True)
class V1FrozenLadderCertificationDesign:
    """Pre-outcome design and trajectory-wide alpha allocation for one ladder."""

    design_id: str
    ladder: V1FrozenCandidateLadder
    envelope_certificate: V1CorrectedAttributionEnvelopeCertificate = field(
        repr=False, compare=False
    )
    certification_context_ids_by_stratum: tuple[tuple[str, tuple[str, ...]], ...]
    stratum_weights: tuple[tuple[str, float], ...]
    certification_context_population_hash: str
    certification_stream_commitment_hash: str
    projection_kernel_source_hash: str
    trajectory_alpha_ledger_hash: str
    trajectory_alpha_ledger_generation: int
    trajectory_familywise_alpha: float
    trajectory_alpha_committed_before_generation: float
    generation_alpha_allocation: float
    ladder_redesign_generation: int
    generation_alpha_committed_before_redesign: float
    redesign_alpha_allocation: float
    endpoint_envelope_certificate: V1FrozenLadderEndpointEnvelopeCertificate | None = (
        field(default=None, repr=False, compare=False)
    )
    causal_generation_frame: V1AlexNetFQCausalGenerationFrameReceipt | None = field(
        default=None, repr=False, compare=False
    )
    causal_discovery_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt | None = field(
        default=None, repr=False, compare=False
    )
    causal_discovery_prefix_freeze: V1AlexNetFQDiscoveryPrefixFreezeReceipt | None = field(
        default=None, repr=False, compare=False
    )
    causal_certification_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt | None = field(
        default=None, repr=False, compare=False
    )
    issued_before_certification_outcomes: bool = True
    sampling_law_kind: str = SEALED_WITHOUT_REPLACEMENT_SAMPLING_LAW
    trajectory_alpha_policy_kind: str = FIXED_TRAJECTORY_FWER_POLICY
    family_alpha_policy_kind: str = UNIFORM_CELL_ALPHA_POLICY
    proposal_authority: str = PROPOSAL_AUTHORITY
    bce_mask_commit_authority: bool = False
    schema: str = FROZEN_LADDER_CERTIFICATION_DESIGN_SCHEMA
    endpoint_absolute_upper_bounds_by_stratum: tuple[tuple[float, ...], ...] = field(
        init=False, repr=False
    )
    endpoint_envelope_derivation_kind: str = field(init=False)
    sealed_population_order_hashes: tuple[str, ...] = field(init=False)
    _envelope_certificate_hash: str = field(init=False, repr=False, compare=False)
    _endpoint_envelope_certificate_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )
    _design_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "design_id", _identity(self.design_id, name="design id")
        )
        if type(self.ladder) is not V1FrozenCandidateLadder:
            raise TypeError("ladder certification requires a typed frozen ladder")
        self.ladder.assert_integrity()
        certificate = self.envelope_certificate
        if type(certificate) is not V1CorrectedAttributionEnvelopeCertificate:
            raise TypeError("ladder certification requires the analytic envelope type")
        certificate_hash = certificate.assert_integrity()
        if certificate.derivation_kind != ANALYTIC_ENVELOPE_DERIVATION_KIND:
            raise ValueError(
                "ladder certification refuses a nonanalytic route envelope"
            )
        if not _coordinates_match(self.ladder, certificate):
            raise ValueError(
                "analytic envelope coordinates differ from the frozen ladder"
            )
        endpoint_certificate = self.endpoint_envelope_certificate
        endpoint_certificate_hash: str | None = None
        if endpoint_certificate is not None:
            if (
                type(endpoint_certificate)
                is not V1FrozenLadderEndpointEnvelopeCertificate
            ):
                raise TypeError(
                    "endpoint envelope requires the typed ladder endpoint certificate"
                )
            endpoint_certificate_hash = endpoint_certificate.assert_integrity()
            if not _coordinates_match(self.ladder, endpoint_certificate):
                raise ValueError(
                    "endpoint envelope coordinates differ from the frozen ladder"
                )
            if (
                endpoint_certificate.ladder_hash != self.ladder.ladder_hash
                or endpoint_certificate.endpoint_action_hashes
                != self.ladder.action_hashes
            ):
                raise ValueError(
                    "endpoint envelope action axis differs from the frozen ladder"
                )

        expected = _axis_pairs(
            self.certification_context_ids_by_stratum,
            name="certification context ledger",
        )
        stratum_names = tuple(certificate.stratum_names)
        if endpoint_certificate is not None and (
            endpoint_certificate.stratum_names != stratum_names
        ):
            raise ValueError(
                "endpoint envelope strata differ from the analytic certificate"
            )
        if tuple(name for name, _ in expected) != stratum_names:
            raise ValueError(
                "certification context strata differ from the analytic certificate"
            )
        if any(not contexts for _, contexts in expected):
            raise ValueError("every certification stratum must be nonempty")
        flat_expected = tuple(
            context for _, contexts in expected for context in contexts
        )
        if len(flat_expected) != len(set(flat_expected)):
            raise ValueError("certification context identities must be globally unique")
        causal_receipts = (
            self.causal_generation_frame,
            self.causal_discovery_stream,
            self.causal_discovery_prefix_freeze,
            self.causal_certification_stream,
        )
        causal_full_frame = all(value is not None for value in causal_receipts)
        if any(value is not None for value in causal_receipts) and not causal_full_frame:
            raise ValueError(
                "causal ladder certification receipts must be supplied atomically"
            )
        if causal_full_frame:
            frame = self.causal_generation_frame
            discovery_stream = self.causal_discovery_stream
            discovery_freeze = self.causal_discovery_prefix_freeze
            certification_stream = self.causal_certification_stream
            if (
                type(frame) is not V1AlexNetFQCausalGenerationFrameReceipt
                or type(discovery_stream)
                is not V1AlexNetFQCausalRoleSamplingStreamReceipt
                or type(discovery_freeze)
                is not V1AlexNetFQDiscoveryPrefixFreezeReceipt
                or type(certification_stream)
                is not V1AlexNetFQCausalRoleSamplingStreamReceipt
            ):
                raise TypeError(
                    "causal ladder certification requires exact D/C receipt types"
                )
            assert frame is not None
            assert discovery_stream is not None
            assert discovery_freeze is not None
            assert certification_stream is not None
            frame.assert_integrity()
            discovery_stream.assert_integrity()
            discovery_freeze.assert_integrity()
            certification_stream.assert_integrity()
            d_counts = discovery_freeze.cumulative_stratum_counts
            d_prefixes = (
                discovery_stream.target_assignment_receipt.generated_order[
                    : d_counts[0]
                ],
                discovery_stream.complement_assignment_receipt.generated_order[
                    : d_counts[1]
                ],
            )
            if (
                discovery_stream.role != "discovery"
                or certification_stream.role != "certification"
                or discovery_stream.generation_frame_hash != frame.receipt_hash
                or discovery_freeze.generation_frame_hash != frame.receipt_hash
                or discovery_freeze.discovery_stream_hash
                != discovery_stream.receipt_hash
                or certification_stream.generation_frame_hash != frame.receipt_hash
                or certification_stream.discovery_prefix_freeze_hash
                != discovery_freeze.receipt_hash
                or certification_stream.selected_shell_hash
                != discovery_freeze.selected_shell_hash
                or discovery_freeze.selected_shell_hash
                != self.ladder.assert_integrity()
                or set(self.ladder.discovery_context_ids)
                != set((*d_prefixes[0], *d_prefixes[1]))
                or expected != certification_stream.context_ids_by_stratum
                or self.certification_stream_commitment_hash
                != certification_stream.receipt_hash
            ):
                raise ValueError(
                    "causal ladder certification receipts differ from frozen D/C rows"
                )
        elif set(flat_expected).intersection(self.ladder.discovery_context_ids):
            raise ValueError(
                "certification contexts must be fresh and disjoint from discovery"
            )
        populations = tuple(len(contexts) for _, contexts in expected)
        if populations != tuple(certificate.population_stratum_counts):
            raise ValueError(
                "certification context counts differ from the analytic certificate"
            )
        order_hashes = tuple(
            sampling_content_hash({"stratum_id": name, "context_ids": contexts})
            for name, contexts in expected
        )
        if order_hashes != tuple(certificate.sealed_population_order_hashes):
            raise ValueError(
                "certification context order differs from the analytic certificate"
            )
        expected_population_hash = content_hash(
            {"certification_context_ids_by_stratum": expected}
        )
        if self.certification_context_population_hash != expected_population_hash:
            raise ValueError(
                "certification population hash differs from its expected ledgers"
            )
        weights = _weight_pairs(self.stratum_weights, expected_names=stratum_names)

        for name in (
            "certification_stream_commitment_hash",
            "projection_kernel_source_hash",
            "trajectory_alpha_ledger_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        ledger_generation = _nonnegative_int(
            self.trajectory_alpha_ledger_generation,
            name="trajectory alpha ledger generation",
        )
        if ledger_generation != self.ladder.proposal_generation:
            raise ValueError(
                "trajectory alpha ledger generation must equal proposal generation"
            )
        redesign_generation = _nonnegative_int(
            self.ladder_redesign_generation, name="ladder redesign generation"
        )
        trajectory_alpha = _finite_unit_interval(
            self.trajectory_familywise_alpha, name="trajectory familywise alpha"
        )
        committed_before_generation = _finite_nonnegative(
            self.trajectory_alpha_committed_before_generation,
            name="trajectory alpha committed before generation",
        )
        generation_allocation = _finite_unit_interval(
            self.generation_alpha_allocation, name="generation alpha allocation"
        )
        committed_before_redesign = _finite_nonnegative(
            self.generation_alpha_committed_before_redesign,
            name="generation alpha committed before redesign",
        )
        redesign_allocation = _finite_unit_interval(
            self.redesign_alpha_allocation, name="redesign alpha allocation"
        )
        tolerance = 32.0 * math.ulp(
            max(
                1.0,
                trajectory_alpha,
                committed_before_generation,
                generation_allocation,
                committed_before_redesign,
                redesign_allocation,
            )
        )
        if (
            committed_before_generation + generation_allocation
            > trajectory_alpha + tolerance
        ):
            raise ValueError(
                "generation allocation exceeds the nonrecycling trajectory ledger"
            )
        if (
            committed_before_redesign + redesign_allocation
            > generation_allocation + tolerance
        ):
            raise ValueError(
                "ladder redesign allocation exceeds its generation alpha ledger"
            )
        if self.issued_before_certification_outcomes is not True:
            raise ValueError("certification design must be issued before outcomes")
        if self.sampling_law_kind != SEALED_WITHOUT_REPLACEMENT_SAMPLING_LAW:
            raise ValueError("ladder certification requires sealed WoR sampling")
        if self.trajectory_alpha_policy_kind != FIXED_TRAJECTORY_FWER_POLICY:
            raise ValueError("ladder certification requires nonrecycling path alpha")
        if self.family_alpha_policy_kind != UNIFORM_CELL_ALPHA_POLICY:
            raise ValueError("unsupported frozen-ladder family alpha allocation")
        if self.proposal_authority != PROPOSAL_AUTHORITY:
            raise ValueError("ladder certification has proposal authority only")
        if self.bce_mask_commit_authority is not False:
            raise ValueError(
                "ladder certification can never grant BCE commit authority"
            )
        if self.schema != FROZEN_LADDER_CERTIFICATION_DESIGN_SCHEMA:
            raise ValueError("frozen ladder certification design schema is invalid")

        if endpoint_certificate is None:
            bound_rows = _route_sum_endpoint_bounds(
                ladder=self.ladder,
                certificate=certificate,
            )
            derivation_kind = ROUTE_SUM_ENDPOINT_ENVELOPE_KIND
        else:
            bound_rows = endpoint_certificate.endpoint_absolute_upper_bounds_by_stratum
            for row in bound_rows:
                for value in row:
                    _outward_nonnegative_square(value)
            derivation_kind = endpoint_certificate.derivation_kind

        object.__setattr__(self, "certification_context_ids_by_stratum", expected)
        object.__setattr__(self, "stratum_weights", weights)
        object.__setattr__(self, "sealed_population_order_hashes", order_hashes)
        object.__setattr__(
            self,
            "endpoint_absolute_upper_bounds_by_stratum",
            bound_rows,
        )
        object.__setattr__(
            self,
            "endpoint_envelope_derivation_kind",
            derivation_kind,
        )
        object.__setattr__(self, "_envelope_certificate_hash", certificate_hash)
        object.__setattr__(
            self,
            "_endpoint_envelope_certificate_hash",
            endpoint_certificate_hash,
        )
        object.__setattr__(
            self, "trajectory_alpha_ledger_generation", ledger_generation
        )
        object.__setattr__(self, "ladder_redesign_generation", redesign_generation)
        object.__setattr__(self, "trajectory_familywise_alpha", trajectory_alpha)
        object.__setattr__(
            self,
            "trajectory_alpha_committed_before_generation",
            committed_before_generation,
        )
        object.__setattr__(self, "generation_alpha_allocation", generation_allocation)
        object.__setattr__(
            self,
            "generation_alpha_committed_before_redesign",
            committed_before_redesign,
        )
        object.__setattr__(self, "redesign_alpha_allocation", redesign_allocation)
        self.assert_integrity()

    @property
    def envelope_certificate_hash(self) -> str:
        return self._envelope_certificate_hash

    @property
    def family_cell_count(self) -> int:
        return (
            len(self.ladder.actions)
            * len(CERTIFICATION_ROLES)
            * len(self.certification_context_ids_by_stratum)
        )

    @property
    def cell_alpha(self) -> float:
        return _nonoverspending_uniform_cell_alpha(
            allocation=self.redesign_alpha_allocation,
            cell_count=self.family_cell_count,
        )

    @property
    def stratum_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.certification_context_ids_by_stratum)

    @property
    def population_sizes(self) -> tuple[int, ...]:
        return tuple(
            len(contexts) for _, contexts in self.certification_context_ids_by_stratum
        )

    def endpoint_bound(self, *, stratum_index: int, endpoint_index: int) -> float:
        return self.endpoint_absolute_upper_bounds_by_stratum[stratum_index][
            endpoint_index - 1
        ]

    def _hash_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "design_id": self.design_id,
            "ladder_hash": self.ladder.assert_integrity(),
            **_coordinate_payload(self.ladder),
            "envelope_certificate_hash": self.envelope_certificate.assert_integrity(),
            "endpoint_envelope_certificate_hash": (
                None
                if self.endpoint_envelope_certificate is None
                else self.endpoint_envelope_certificate.assert_integrity()
            ),
            "endpoint_envelope_derivation_kind": (
                self.endpoint_envelope_derivation_kind
            ),
            "certification_context_ids_by_stratum": (
                self.certification_context_ids_by_stratum
            ),
            "sealed_population_order_hashes": self.sealed_population_order_hashes,
            "stratum_weights": self.stratum_weights,
            "certification_context_population_hash": (
                self.certification_context_population_hash
            ),
            "certification_stream_commitment_hash": (
                self.certification_stream_commitment_hash
            ),
            "projection_kernel_source_hash": self.projection_kernel_source_hash,
            "endpoint_absolute_upper_bounds_by_stratum": (
                self.endpoint_absolute_upper_bounds_by_stratum
            ),
            "trajectory_alpha_ledger_hash": self.trajectory_alpha_ledger_hash,
            "trajectory_alpha_ledger_generation": (
                self.trajectory_alpha_ledger_generation
            ),
            "trajectory_familywise_alpha": self.trajectory_familywise_alpha,
            "trajectory_alpha_committed_before_generation": (
                self.trajectory_alpha_committed_before_generation
            ),
            "generation_alpha_allocation": self.generation_alpha_allocation,
            "ladder_redesign_generation": self.ladder_redesign_generation,
            "generation_alpha_committed_before_redesign": (
                self.generation_alpha_committed_before_redesign
            ),
            "redesign_alpha_allocation": self.redesign_alpha_allocation,
            "family_cell_count": self.family_cell_count,
            "cell_alpha": self.cell_alpha,
            "issued_before_certification_outcomes": (
                self.issued_before_certification_outcomes
            ),
            "sampling_law_kind": self.sampling_law_kind,
            "trajectory_alpha_policy_kind": self.trajectory_alpha_policy_kind,
            "family_alpha_policy_kind": self.family_alpha_policy_kind,
            "proposal_authority": self.proposal_authority,
            "bce_mask_commit_authority": self.bce_mask_commit_authority,
        }
        # Preserve the byte-identical legacy design hash when no causal graph is
        # present.  The typed graph becomes part of the authority-bearing hash
        # only for the new full-population D/C role.
        if self.causal_generation_frame is not None:
            assert self.causal_discovery_stream is not None
            assert self.causal_discovery_prefix_freeze is not None
            assert self.causal_certification_stream is not None
            payload.update(
                {
                    "causal_full_frame_role_schema": (
                        "partition_free_v1_fq_causal_full_frame_role_v1"
                    ),
                    "causal_generation_frame_receipt_hash": (
                        self.causal_generation_frame.receipt_hash
                    ),
                    "causal_discovery_stream_receipt_hash": (
                        self.causal_discovery_stream.receipt_hash
                    ),
                    "causal_discovery_prefix_freeze_hash": (
                        self.causal_discovery_prefix_freeze.receipt_hash
                    ),
                    "causal_certification_stream_receipt_hash": (
                        self.causal_certification_stream.receipt_hash
                    ),
                }
            )
        return payload

    @property
    def design_hash(self) -> str:
        cached = self._design_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.ladder.assert_integrity()
        certificate_hash = self.envelope_certificate.assert_integrity()
        if certificate_hash != self._envelope_certificate_hash:
            raise ValueError("analytic envelope identity changed after design freeze")
        if self.endpoint_envelope_certificate is None:
            if self._endpoint_envelope_certificate_hash is not None:
                raise ValueError(
                    "endpoint envelope identity changed after design freeze"
                )
        else:
            endpoint_hash = self.endpoint_envelope_certificate.assert_integrity()
            if endpoint_hash != self._endpoint_envelope_certificate_hash:
                raise ValueError(
                    "endpoint envelope identity changed after design freeze"
                )
        return _assert_content_integrity(
            self,
            payload=self._hash_payload(),
            cache_name="_design_hash",
            label="frozen ladder certification design",
            expected_hash=expected_hash,
        )


def build_v1_frozen_ladder_route_sum_endpoint_envelope(
    *,
    ladder: V1FrozenCandidateLadder,
    envelope_certificate: V1CorrectedAttributionEnvelopeCertificate,
) -> V1FrozenLadderEndpointEnvelopeCertificate:
    """Freeze action endpoint bounds from an analytic singleton-route envelope.

    This is the first production-tractable endpoint certificate producer.  It
    does not claim cancellation-aware direct endpoint interpretation; instead,
    it gives the existing sound route-wise envelope a first-class
    action-specific proof identity.  Downstream F/Q certification can then
    consume frozen endpoint bounds without certifying or replaying any global
    57M-route order.
    """

    if type(ladder) is not V1FrozenCandidateLadder:
        raise TypeError("endpoint envelope construction requires a typed ladder")
    if type(envelope_certificate) is not V1CorrectedAttributionEnvelopeCertificate:
        raise TypeError(
            "endpoint envelope construction requires the analytic route envelope"
        )
    ladder_hash = ladder.assert_integrity()
    envelope_hash = envelope_certificate.assert_integrity()
    if envelope_certificate.derivation_kind != ANALYTIC_ENVELOPE_DERIVATION_KIND:
        raise ValueError("endpoint envelope refuses a nonanalytic route envelope")
    if not _coordinates_match(ladder, envelope_certificate):
        raise ValueError("route envelope coordinates differ from the frozen ladder")
    bound_rows = _route_sum_endpoint_bounds(
        ladder=ladder,
        certificate=envelope_certificate,
    )
    verifier_source_hash = _source_sha256()
    proof_payload = {
        "schema": FROZEN_LADDER_ROUTE_SUM_ENDPOINT_ENVELOPE_PROOF_SCHEMA,
        "derivation_kind": ACTION_ROUTE_SUM_ENDPOINT_ENVELOPE_KIND,
        "ladder_hash": ladder_hash,
        "endpoint_action_hashes": ladder.action_hashes,
        "route_envelope_certificate_hash": envelope_hash,
        "route_envelope_proof_artifact_hash": (
            envelope_certificate.proof_artifact_hash
        ),
        "route_count": ladder.route_count,
        "route_axis_hash": ladder.route_axis_hash,
        "stratum_names": tuple(envelope_certificate.stratum_names),
        "endpoint_absolute_upper_bounds_by_stratum": bound_rows,
        "verifier_source_hash": verifier_source_hash,
        "score_order_authority": False,
        "bce_mask_commit_authority": False,
    }
    proof_artifact_hash = content_hash(
        proof_payload,
        prefix=_ENDPOINT_ENVELOPE_PROOF_PREFIX,
    )
    return V1FrozenLadderEndpointEnvelopeCertificate(
        **_coordinate_payload(ladder),
        ladder_hash=ladder_hash,
        endpoint_action_hashes=ladder.action_hashes,
        stratum_names=tuple(envelope_certificate.stratum_names),
        endpoint_absolute_upper_bounds_by_stratum=bound_rows,
        verifier_source_hash=verifier_source_hash,
        proof_artifact_hash=proof_artifact_hash,
        derivation_kind=ACTION_ROUTE_SUM_ENDPOINT_ENVELOPE_KIND,
    )


@dataclass(frozen=True, slots=True)
class V1FrozenLadderProjectionRecord:
    """One fresh context row projected onto every frozen endpoint.

    Schema v1 is the historical point-record replay surface.  Schema v2 binds
    one nonnegative numerical radius to every binary64 center; the exact-real
    projection is asserted only to lie inside that enclosure.  A present
    radius axis selects v2 even when every radius is zero, so the ABI cannot
    silently discard a production numerical proof.
    """

    design_hash: str
    ladder_hash: str
    stratum_id: str
    context_id: str
    draw_id: str
    projection_proof_hash: str
    endpoint_action_hashes: tuple[str, ...]
    signed_projection_by_endpoint: tuple[float, ...]
    schema: str = FROZEN_LADDER_PROJECTION_RECORD_SCHEMA
    _record_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )
    # Appended after every v1 slot so historical slot-pickle states can be
    # replayed through the explicit zero-radius adapter below.
    numerical_radius_by_endpoint: tuple[float, ...] = field(
        default=(),
        repr=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "design_hash",
            "ladder_hash",
            "stratum_id",
            "context_id",
            "draw_id",
            "projection_proof_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        actions = tuple(
            _identity(value, name="endpoint action hash")
            for value in self.endpoint_action_hashes
        )
        projections = tuple(
            float(value) for value in self.signed_projection_by_endpoint
        )
        if not actions or len(actions) != len(set(actions)):
            raise ValueError("projection row requires unique endpoint identities")
        if len(projections) != len(actions) or any(
            not math.isfinite(value) for value in projections
        ):
            raise ValueError("projection row requires one finite value per endpoint")
        radii = tuple(
            _finite_nonnegative(value, name="projection numerical radius")
            for value in self.numerical_radius_by_endpoint
        )
        if radii:
            if len(radii) != len(actions):
                raise ValueError(
                    "projection row requires one numerical radius per endpoint"
                )
            if self.schema not in {
                FROZEN_LADDER_PROJECTION_RECORD_SCHEMA,
                FROZEN_LADDER_ENCLOSED_PROJECTION_RECORD_SCHEMA,
            }:
                raise ValueError("enclosed projection record schema is invalid")
            object.__setattr__(
                self,
                "schema",
                FROZEN_LADDER_ENCLOSED_PROJECTION_RECORD_SCHEMA,
            )
        elif self.schema != FROZEN_LADDER_PROJECTION_RECORD_SCHEMA:
            raise ValueError("legacy point projection record schema is invalid")
        object.__setattr__(self, "endpoint_action_hashes", actions)
        object.__setattr__(self, "signed_projection_by_endpoint", projections)
        object.__setattr__(self, "numerical_radius_by_endpoint", radii)
        self.assert_integrity()

    def _hash_payload(self) -> dict[str, object]:
        payload = {
            "schema": self.schema,
            "design_hash": self.design_hash,
            "ladder_hash": self.ladder_hash,
            "stratum_id": self.stratum_id,
            "context_id": self.context_id,
            "draw_id": self.draw_id,
            "projection_proof_hash": self.projection_proof_hash,
            "endpoint_action_hashes": self.endpoint_action_hashes,
            "signed_projection_by_endpoint": self.signed_projection_by_endpoint,
        }
        if self.schema == FROZEN_LADDER_ENCLOSED_PROJECTION_RECORD_SCHEMA:
            payload["numerical_radius_by_endpoint"] = self.numerical_radius_by_endpoint
        return payload

    @property
    def has_numerical_enclosure(self) -> bool:
        return self.schema == FROZEN_LADDER_ENCLOSED_PROJECTION_RECORD_SCHEMA

    @property
    def effective_numerical_radius_by_endpoint(self) -> tuple[float, ...]:
        if self.has_numerical_enclosure:
            return self.numerical_radius_by_endpoint
        return (0.0,) * len(self.endpoint_action_hashes)

    @property
    def record_hash(self) -> str:
        cached = self._record_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        return _assert_content_integrity(
            self,
            payload=self._hash_payload(),
            cache_name="_record_hash",
            label="frozen ladder projection record",
            expected_hash=expected_hash,
        )

    def __getstate__(self) -> list[object]:
        return [getattr(self, item.name) for item in fields(type(self))]

    def __setstate__(self, state: Sequence[object]) -> None:
        names = tuple(item.name for item in fields(type(self)))
        values = list(state)
        if len(values) == len(names) - 1:
            # Historical v1 point records predate the trailing radius axis.
            values.append(())
        if len(values) != len(names):
            raise ValueError("projection record pickle state has invalid arity")
        for name, value in zip(names, values, strict=True):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class V1FrozenLadderCellInterval:
    endpoint_index: int
    action_hash: str
    role: str
    units: str
    stratum_id: str
    population_size: int
    sample_size: int
    sample_mean: float | None
    lower: float
    upper: float
    value_upper_bound: float
    alpha: float
    full_census: bool
    empty: bool
    sealed_order_hash: str
    method: str
    evidence_snapshot_hash: str
    schema: str = FROZEN_LADDER_CELL_INTERVAL_SCHEMA
    _interval_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )
    # v2-only authority metadata, appended for v1 slot-pickle compatibility.
    population_census_complete: bool | None = field(default=None, repr=False)
    lower_confidence_authority: bool = True
    upper_confidence_authority: bool = True
    observation_semantics: str = EXACT_POINT_OBSERVATION_SEMANTICS

    def __post_init__(self) -> None:
        endpoint = _positive_int(self.endpoint_index, name="endpoint index")
        for name in (
            "action_hash",
            "stratum_id",
            "sealed_order_hash",
            "method",
            "evidence_snapshot_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if self.role not in CERTIFICATION_ROLES:
            raise ValueError("unknown frozen-ladder certification role")
        expected_units = F_UNITS if self.role == F_ROLE else Q_UNITS
        if self.units != expected_units:
            raise ValueError("frozen-ladder cell units differ from its role")
        population = _positive_int(self.population_size, name="population size")
        sample = _nonnegative_int(self.sample_size, name="sample size")
        if sample > population:
            raise ValueError("cell sample size exceeds its population")
        bound = _finite_nonnegative(self.value_upper_bound, name="cell upper bound")
        alpha = _finite_unit_interval(self.alpha, name="cell alpha")
        lower = float(self.lower)
        upper = float(self.upper)
        if not (math.isfinite(lower) and math.isfinite(upper)):
            raise ValueError("cell confidence limits must be finite")
        if lower < -1e-12 or upper > bound + 1e-12:
            raise ValueError("cell confidence interval left its analytic bounds")
        mean = None if self.sample_mean is None else float(self.sample_mean)
        if sample == 0 and mean is not None:
            raise ValueError("an unobserved cell cannot have a sample mean")
        if sample > 0 and (
            mean is None
            or not math.isfinite(mean)
            or not -1e-12 <= mean <= bound + 1e-12
        ):
            raise ValueError("observed cell requires a bounded finite sample mean")
        if self.empty != (lower > upper):
            raise ValueError("cell empty flag differs from its interval")
        population_complete = (
            sample == population
            if self.population_census_complete is None
            else self.population_census_complete
        )
        if type(population_complete) is not bool or population_complete != (
            sample == population
        ):
            raise ValueError(
                "cell population census flag differs from its sample count"
            )
        if not all(
            type(value) is bool
            for value in (
                self.full_census,
                self.lower_confidence_authority,
                self.upper_confidence_authority,
            )
        ):
            raise ValueError("cell confidence authority flags must be booleans")
        if self.full_census and (
            mean is None
            or not math.isclose(lower, mean, abs_tol=1e-12, rel_tol=0.0)
            or not math.isclose(upper, mean, abs_tol=1e-12, rel_tol=0.0)
        ):
            raise ValueError("a census cell must collapse to its exact sample mean")
        if self.method not in {
            WOR_BETTING_CS_METHOD,
            ZERO_ENVELOPE_METHOD,
            NUMERICALLY_ENCLOSED_WOR_UPPER_CS_METHOD,
        }:
            raise ValueError("unsupported frozen-ladder confidence method")
        if self.method == ZERO_ENVELOPE_METHOD and (
            bound != 0.0 or lower != 0.0 or upper != 0.0 or (mean not in {None, 0.0})
        ):
            raise ValueError("zero-envelope cells must remain identically zero")
        semantics = _identity(
            self.observation_semantics,
            name="cell observation semantics",
        )
        if self.schema == FROZEN_LADDER_CELL_INTERVAL_SCHEMA:
            if (
                self.full_census != population_complete
                or not self.lower_confidence_authority
                or not self.upper_confidence_authority
                or semantics != EXACT_POINT_OBSERVATION_SEMANTICS
                or self.method == NUMERICALLY_ENCLOSED_WOR_UPPER_CS_METHOD
            ):
                raise ValueError("legacy point cell authority is inconsistent")
        elif self.schema == FROZEN_LADDER_ENCLOSED_CELL_INTERVAL_SCHEMA:
            if (
                self.full_census
                or self.lower_confidence_authority
                or not self.upper_confidence_authority
                or semantics != NUMERICALLY_ENCLOSED_UPPER_OBSERVATION_SEMANTICS
                or self.method != NUMERICALLY_ENCLOSED_WOR_UPPER_CS_METHOD
                or (not self.empty and lower != 0.0)
            ):
                raise ValueError("numerically enclosed cell must be upper-only")
            if population_complete and (
                mean is None
                or not math.isclose(upper, mean, abs_tol=1e-12, rel_tol=0.0)
            ):
                raise ValueError(
                    "an enclosed population census must retain its surrogate mean upper"
                )
        else:
            raise ValueError("frozen ladder cell interval schema is invalid")
        object.__setattr__(self, "endpoint_index", endpoint)
        object.__setattr__(self, "population_size", population)
        object.__setattr__(self, "sample_size", sample)
        object.__setattr__(self, "sample_mean", mean)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "value_upper_bound", bound)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "population_census_complete", population_complete)
        object.__setattr__(self, "observation_semantics", semantics)
        self.assert_integrity()

    def _hash_payload(self) -> dict[str, object]:
        payload = {
            name: getattr(self, name)
            for name in (
                "schema",
                "endpoint_index",
                "action_hash",
                "role",
                "units",
                "stratum_id",
                "population_size",
                "sample_size",
                "sample_mean",
                "lower",
                "upper",
                "value_upper_bound",
                "alpha",
                "full_census",
                "empty",
                "sealed_order_hash",
                "method",
                "evidence_snapshot_hash",
            )
        }
        if self.schema == FROZEN_LADDER_ENCLOSED_CELL_INTERVAL_SCHEMA:
            payload.update(
                {
                    "population_census_complete": self.population_census_complete,
                    "lower_confidence_authority": self.lower_confidence_authority,
                    "upper_confidence_authority": self.upper_confidence_authority,
                    "observation_semantics": self.observation_semantics,
                }
            )
        return payload

    @property
    def interval_hash(self) -> str:
        cached = self._interval_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        return _assert_content_integrity(
            self,
            payload=self._hash_payload(),
            cache_name="_interval_hash",
            label="frozen ladder cell interval",
            expected_hash=expected_hash,
        )

    def __getstate__(self) -> list[object]:
        return [getattr(self, item.name) for item in fields(type(self))]

    def __setstate__(self, state: Sequence[object]) -> None:
        names = tuple(item.name for item in fields(type(self)))
        values = list(state)
        legacy_defaults: tuple[object, ...] = (
            None,
            True,
            True,
            EXACT_POINT_OBSERVATION_SEMANTICS,
        )
        if len(values) < len(names):
            missing = len(names) - len(values)
            if missing > len(legacy_defaults):
                raise ValueError("cell interval pickle state has invalid arity")
            values.extend(legacy_defaults[-missing:])
        if len(values) != len(names):
            raise ValueError("cell interval pickle state has invalid arity")
        for name, value in zip(names, values, strict=True):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class V1FrozenLadderEndpointInterval:
    endpoint_index: int
    action_hash: str
    f_lower: float
    f_upper: float
    q_lower: float
    q_upper: float
    sqrt_q_lower: float
    sqrt_q_upper: float
    empty: bool
    full_census: bool
    f_units: str = F_UNITS
    q_units: str = Q_UNITS
    sqrt_q_units: str = SQRT_Q_UNITS
    schema: str = FROZEN_LADDER_ENDPOINT_INTERVAL_SCHEMA
    _interval_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )
    # v2-only aggregate authority metadata, appended after every v1 slot.
    population_census_complete: bool | None = field(default=None, repr=False)
    lower_confidence_authority: bool = True
    upper_confidence_authority: bool = True
    observation_semantics: str = EXACT_POINT_OBSERVATION_SEMANTICS

    def __post_init__(self) -> None:
        endpoint = _positive_int(self.endpoint_index, name="endpoint index")
        object.__setattr__(
            self, "action_hash", _identity(self.action_hash, name="action hash")
        )
        values = {
            name: float(getattr(self, name))
            for name in (
                "f_lower",
                "f_upper",
                "q_lower",
                "q_upper",
                "sqrt_q_lower",
                "sqrt_q_upper",
            )
        }
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("endpoint confidence limits must be finite")
        if values["f_lower"] < -1e-12 or values["q_lower"] < -1e-12:
            raise ValueError("F/Q intervals cannot leave their nonnegative support")
        if not self.empty and (
            values["f_lower"] > values["f_upper"]
            or values["q_lower"] > values["q_upper"]
        ):
            raise ValueError("nonempty endpoint has an inverted F/Q interval")
        if not math.isclose(
            values["sqrt_q_lower"],
            math.sqrt(max(0.0, values["q_lower"])),
            abs_tol=1e-12,
            rel_tol=1e-12,
        ) or not math.isclose(
            values["sqrt_q_upper"],
            math.sqrt(max(0.0, values["q_upper"])),
            abs_tol=1e-12,
            rel_tol=1e-12,
        ):
            raise ValueError("thermodynamic-length interval must be sqrt(Q)")
        if (self.f_units, self.q_units, self.sqrt_q_units) != (
            F_UNITS,
            Q_UNITS,
            SQRT_Q_UNITS,
        ):
            raise ValueError("endpoint F/Q coordinate units are invalid")
        population_complete = (
            self.full_census
            if self.population_census_complete is None
            else self.population_census_complete
        )
        if not all(
            type(value) is bool
            for value in (
                self.full_census,
                population_complete,
                self.lower_confidence_authority,
                self.upper_confidence_authority,
            )
        ):
            raise ValueError("endpoint confidence authority flags must be booleans")
        semantics = _identity(
            self.observation_semantics,
            name="endpoint observation semantics",
        )
        if self.schema == FROZEN_LADDER_ENDPOINT_INTERVAL_SCHEMA:
            if (
                self.full_census != population_complete
                or not self.lower_confidence_authority
                or not self.upper_confidence_authority
                or semantics != EXACT_POINT_OBSERVATION_SEMANTICS
            ):
                raise ValueError("legacy point endpoint authority is inconsistent")
        elif self.schema == FROZEN_LADDER_ENCLOSED_ENDPOINT_INTERVAL_SCHEMA:
            if (
                self.full_census
                or self.lower_confidence_authority
                or not self.upper_confidence_authority
                or semantics != NUMERICALLY_ENCLOSED_UPPER_OBSERVATION_SEMANTICS
                or any(
                    value != 0.0
                    for value in (
                        values["f_lower"],
                        values["q_lower"],
                        values["sqrt_q_lower"],
                    )
                )
            ):
                raise ValueError("numerically enclosed endpoint must be upper-only")
        else:
            raise ValueError("frozen ladder endpoint interval schema is invalid")
        object.__setattr__(self, "endpoint_index", endpoint)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "population_census_complete", population_complete)
        object.__setattr__(self, "observation_semantics", semantics)
        self.assert_integrity()

    def _hash_payload(self) -> dict[str, object]:
        payload = {
            name: getattr(self, name)
            for name in (
                "schema",
                "endpoint_index",
                "action_hash",
                "f_lower",
                "f_upper",
                "q_lower",
                "q_upper",
                "sqrt_q_lower",
                "sqrt_q_upper",
                "empty",
                "full_census",
                "f_units",
                "q_units",
                "sqrt_q_units",
            )
        }
        if self.schema == FROZEN_LADDER_ENCLOSED_ENDPOINT_INTERVAL_SCHEMA:
            payload.update(
                {
                    "population_census_complete": self.population_census_complete,
                    "lower_confidence_authority": self.lower_confidence_authority,
                    "upper_confidence_authority": self.upper_confidence_authority,
                    "observation_semantics": self.observation_semantics,
                }
            )
        return payload

    @property
    def interval_hash(self) -> str:
        cached = self._interval_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        return _assert_content_integrity(
            self,
            payload=self._hash_payload(),
            cache_name="_interval_hash",
            label="frozen ladder endpoint interval",
            expected_hash=expected_hash,
        )

    def __getstate__(self) -> list[object]:
        return [getattr(self, item.name) for item in fields(type(self))]

    def __setstate__(self, state: Sequence[object]) -> None:
        names = tuple(item.name for item in fields(type(self)))
        values = list(state)
        legacy_defaults: tuple[object, ...] = (
            None,
            True,
            True,
            EXACT_POINT_OBSERVATION_SEMANTICS,
        )
        if len(values) < len(names):
            missing = len(names) - len(values)
            if missing > len(legacy_defaults):
                raise ValueError("endpoint interval pickle state has invalid arity")
            values.extend(legacy_defaults[-missing:])
        if len(values) != len(names):
            raise ValueError("endpoint interval pickle state has invalid arity")
        for name, value in zip(names, values, strict=True):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class V1FrozenLadderCertificationReceipt:
    design_hash: str
    ladder_hash: str
    envelope_certificate_hash: str
    campaign_contract_hash: str
    model_identity_hash: str
    query_contract_hash: str
    preprocessing_hash: str
    current_mask_hash: str
    raw_logit_stream_source_hash: str
    sampling_law_hash: str
    proposal_generation: int
    route_count: int
    route_axis_hash: str
    ladder_redesign_generation: int
    certification_context_population_hash: str
    sealed_population_order_hashes: tuple[str, ...]
    stratum_weights: tuple[tuple[str, float], ...]
    observed_context_ids_by_stratum: tuple[tuple[str, tuple[str, ...]], ...]
    observed_draw_ids_by_stratum: tuple[tuple[str, tuple[str, ...]], ...]
    projection_records: tuple[V1FrozenLadderProjectionRecord, ...]
    cell_intervals: tuple[V1FrozenLadderCellInterval, ...]
    endpoint_intervals: tuple[V1FrozenLadderEndpointInterval, ...]
    trajectory_alpha_ledger_hash: str
    trajectory_familywise_alpha: float
    trajectory_alpha_committed_before_generation: float
    generation_alpha_allocation: float
    generation_alpha_committed_before_redesign: float
    redesign_alpha_allocation: float
    family_cell_count: int
    cell_alpha: float
    proposal_authority: str = PROPOSAL_AUTHORITY
    bce_mask_commit_authority: bool = False
    terminal_validation_seen: bool = False
    family_alpha_policy_kind: str = UNIFORM_CELL_ALPHA_POLICY
    schema: str = FROZEN_LADDER_CERTIFICATION_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "design_hash",
            "ladder_hash",
            "envelope_certificate_hash",
            "certification_context_population_hash",
            "trajectory_alpha_ledger_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        _validate_coordinates(self)
        redesign = _nonnegative_int(
            self.ladder_redesign_generation, name="ladder redesign generation"
        )
        order_hashes = tuple(
            _identity(value, name="sealed population order hash")
            for value in self.sealed_population_order_hashes
        )
        observed_contexts = _axis_pairs(
            self.observed_context_ids_by_stratum, name="observed context ledger"
        )
        observed_draws = _axis_pairs(
            self.observed_draw_ids_by_stratum, name="observed draw ledger"
        )
        names = tuple(name for name, _ in observed_contexts)
        if tuple(name for name, _ in observed_draws) != names:
            raise ValueError("observed context and draw axes differ")
        if len(order_hashes) != len(names):
            raise ValueError("sealed-order axis differs from observed strata")
        weights = _weight_pairs(self.stratum_weights, expected_names=names)
        if any(
            len(contexts) != len(draws)
            for (_, contexts), (_, draws) in zip(
                observed_contexts, observed_draws, strict=True
            )
        ):
            raise ValueError("observed context and draw ledgers must align")
        flat_contexts = tuple(
            context for _, contexts in observed_contexts for context in contexts
        )
        flat_draws = tuple(draw for _, draws in observed_draws for draw in draws)
        if len(flat_contexts) != len(set(flat_contexts)) or len(flat_draws) != len(
            set(flat_draws)
        ):
            raise ValueError("observed certification ledgers cannot repeat identities")

        records = tuple(self.projection_records)
        if any(
            type(record) is not V1FrozenLadderProjectionRecord for record in records
        ):
            raise TypeError("receipt projection rows must use the typed record")
        for record in records:
            record.assert_integrity()
            if (
                record.design_hash != self.design_hash
                or record.ladder_hash != self.ladder_hash
            ):
                raise ValueError(
                    "projection record was recontextualized across designs"
                )
            if record.stratum_id not in names:
                raise ValueError("projection record names an unregistered stratum")
        enclosed_records = any(record.has_numerical_enclosure for record in records)
        if enclosed_records:
            if self.schema not in {
                FROZEN_LADDER_CERTIFICATION_RECEIPT_SCHEMA,
                FROZEN_LADDER_ENCLOSED_CERTIFICATION_RECEIPT_SCHEMA,
            }:
                raise ValueError("enclosed certification receipt schema is invalid")
            object.__setattr__(
                self,
                "schema",
                FROZEN_LADDER_ENCLOSED_CERTIFICATION_RECEIPT_SCHEMA,
            )
        elif self.schema != FROZEN_LADDER_CERTIFICATION_RECEIPT_SCHEMA:
            raise ValueError("legacy point certification receipt schema is invalid")
        record_contexts = {
            name: tuple(
                record.context_id for record in records if record.stratum_id == name
            )
            for name in names
        }
        record_draws = {
            name: tuple(
                record.draw_id for record in records if record.stratum_id == name
            )
            for name in names
        }
        if any(
            record_contexts[name] != contexts for name, contexts in observed_contexts
        ) or any(record_draws[name] != draws for name, draws in observed_draws):
            raise ValueError("projection record history differs from observed ledgers")

        cells = tuple(self.cell_intervals)
        endpoints = tuple(self.endpoint_intervals)
        if not endpoints or any(
            type(item) is not V1FrozenLadderEndpointInterval for item in endpoints
        ):
            raise TypeError("receipt requires typed endpoint intervals")
        if any(type(item) is not V1FrozenLadderCellInterval for item in cells):
            raise TypeError("receipt requires typed cell intervals")
        for item in (*cells, *endpoints):
            item.assert_integrity()
        endpoint_axis = tuple(item.endpoint_index for item in endpoints)
        if endpoint_axis != tuple(range(1, len(endpoints) + 1)):
            raise ValueError("receipt endpoint interval axis must be contiguous")
        endpoint_action_hashes = tuple(item.action_hash for item in endpoints)
        if any(
            record.endpoint_action_hashes != endpoint_action_hashes
            for record in records
        ):
            raise ValueError("projection record endpoint axis differs from the receipt")
        expected_cell_axis = tuple(
            (endpoint.endpoint_index, endpoint.action_hash, role, stratum)
            for endpoint in endpoints
            for role in CERTIFICATION_ROLES
            for stratum in names
        )
        observed_cell_axis = tuple(
            (item.endpoint_index, item.action_hash, item.role, item.stratum_id)
            for item in cells
        )
        if observed_cell_axis != expected_cell_axis:
            raise ValueError("receipt cell interval axis is not canonical")
        observed_count_by_stratum = dict(
            (name, len(contexts)) for name, contexts in observed_contexts
        )
        for cell in cells:
            if cell.sample_size != observed_count_by_stratum[cell.stratum_id]:
                raise ValueError("cell sample count differs from its observed ledger")
        population_by_stratum = {
            name: {cell.population_size for cell in cells if cell.stratum_id == name}
            for name in names
        }
        if any(len(values) != 1 for values in population_by_stratum.values()):
            raise ValueError("cell population sizes differ within a stratum")

        cell_map = {
            (cell.endpoint_index, cell.role, cell.stratum_id): cell for cell in cells
        }
        weight_map = dict(weights)
        for endpoint in endpoints:
            f_cells = tuple(
                cell_map[(endpoint.endpoint_index, F_ROLE, name)] for name in names
            )
            q_cells = tuple(
                cell_map[(endpoint.endpoint_index, Q_ROLE, name)] for name in names
            )
            f_projection = _weighted_cell_projection(f_cells, weights=weight_map)
            q_projection = _weighted_cell_projection(q_cells, weights=weight_map)
            expected_aggregate = (*f_projection, *q_projection)
            observed_aggregate = (
                endpoint.f_lower,
                endpoint.f_upper,
                endpoint.q_lower,
                endpoint.q_upper,
            )
            if any(
                not math.isclose(left, right, abs_tol=1e-12, rel_tol=1e-12)
                for left, right in zip(
                    expected_aggregate, observed_aggregate, strict=True
                )
            ):
                raise ValueError("endpoint F/Q interval differs from its stratum cells")
            if endpoint.empty != any(item.empty for item in (*f_cells, *q_cells)):
                raise ValueError("endpoint empty flag differs from its stratum cells")
            if endpoint.full_census != all(
                item.full_census for item in (*f_cells, *q_cells)
            ):
                raise ValueError("endpoint census flag differs from its stratum cells")
            if endpoint.population_census_complete != all(
                item.population_census_complete for item in (*f_cells, *q_cells)
            ):
                raise ValueError(
                    "endpoint population census flag differs from its stratum cells"
                )
            expected_lower_authority = all(
                item.lower_confidence_authority for item in (*f_cells, *q_cells)
            )
            if (
                endpoint.lower_confidence_authority != expected_lower_authority
                or endpoint.upper_confidence_authority
                != all(item.upper_confidence_authority for item in (*f_cells, *q_cells))
            ):
                raise ValueError(
                    "endpoint confidence authority differs from its stratum cells"
                )
            expected_semantics = (
                EXACT_POINT_OBSERVATION_SEMANTICS
                if expected_lower_authority
                else NUMERICALLY_ENCLOSED_UPPER_OBSERVATION_SEMANTICS
            )
            if endpoint.observation_semantics != expected_semantics:
                raise ValueError(
                    "endpoint observation semantics differ from its stratum cells"
                )

        family_cells = _positive_int(self.family_cell_count, name="family cell count")
        if family_cells != len(cells):
            raise ValueError("family cell count differs from the interval family")
        cell_alpha = _finite_unit_interval(self.cell_alpha, name="cell alpha")
        redesign_alpha = _finite_unit_interval(
            self.redesign_alpha_allocation, name="redesign alpha allocation"
        )
        if not math.isclose(
            cell_alpha * family_cells,
            redesign_alpha,
            abs_tol=32.0 * math.ulp(max(1.0, redesign_alpha)),
            rel_tol=0.0,
        ):
            raise ValueError("cell alpha allocations do not exhaust redesign alpha")
        if cell_alpha * family_cells > redesign_alpha:
            raise ValueError("cell alpha allocations overspend redesign alpha")
        if any(
            not math.isclose(item.alpha, cell_alpha, abs_tol=1e-15, rel_tol=0.0)
            for item in cells
        ):
            raise ValueError("receipt cells differ from the family alpha allocation")
        for name in (
            "trajectory_familywise_alpha",
            "generation_alpha_allocation",
        ):
            object.__setattr__(
                self,
                name,
                _finite_unit_interval(getattr(self, name), name=name),
            )
        for name in (
            "trajectory_alpha_committed_before_generation",
            "generation_alpha_committed_before_redesign",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name),
            )
        if self.proposal_authority != PROPOSAL_AUTHORITY:
            raise ValueError("ladder receipt has proposal authority only")
        if self.bce_mask_commit_authority is not False:
            raise ValueError("ladder receipt cannot grant BCE commit authority")
        if self.terminal_validation_seen is not False:
            raise ValueError("proposal certification must remain validation independent")
        if self.family_alpha_policy_kind != UNIFORM_CELL_ALPHA_POLICY:
            raise ValueError("receipt family alpha policy is invalid")
        expected_schema = (
            FROZEN_LADDER_ENCLOSED_CERTIFICATION_RECEIPT_SCHEMA
            if enclosed_records
            else FROZEN_LADDER_CERTIFICATION_RECEIPT_SCHEMA
        )
        if self.schema != expected_schema:
            raise ValueError("frozen ladder certification receipt schema is invalid")

        object.__setattr__(self, "ladder_redesign_generation", redesign)
        object.__setattr__(self, "sealed_population_order_hashes", order_hashes)
        object.__setattr__(self, "stratum_weights", weights)
        object.__setattr__(self, "observed_context_ids_by_stratum", observed_contexts)
        object.__setattr__(self, "observed_draw_ids_by_stratum", observed_draws)
        object.__setattr__(self, "projection_records", records)
        object.__setattr__(self, "cell_intervals", cells)
        object.__setattr__(self, "endpoint_intervals", endpoints)
        object.__setattr__(self, "family_cell_count", family_cells)
        object.__setattr__(self, "cell_alpha", cell_alpha)
        object.__setattr__(self, "redesign_alpha_allocation", redesign_alpha)
        self.assert_integrity()

    @property
    def sample_size(self) -> int:
        return len(self.projection_records)

    @property
    def full_census(self) -> bool:
        return all(item.full_census for item in self.endpoint_intervals)

    @property
    def population_census_complete(self) -> bool:
        return all(
            bool(item.population_census_complete) for item in self.endpoint_intervals
        )

    @property
    def lower_confidence_authority(self) -> bool:
        return all(item.lower_confidence_authority for item in self.endpoint_intervals)

    @property
    def upper_confidence_authority(self) -> bool:
        return all(item.upper_confidence_authority for item in self.endpoint_intervals)

    @property
    def empty(self) -> bool:
        return any(item.empty for item in self.endpoint_intervals)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "design_hash": self.design_hash,
            "ladder_hash": self.ladder_hash,
            "envelope_certificate_hash": self.envelope_certificate_hash,
            **_coordinate_payload(self),
            "ladder_redesign_generation": self.ladder_redesign_generation,
            "certification_context_population_hash": (
                self.certification_context_population_hash
            ),
            "sealed_population_order_hashes": self.sealed_population_order_hashes,
            "stratum_weights": self.stratum_weights,
            "observed_context_ids_by_stratum": self.observed_context_ids_by_stratum,
            "observed_draw_ids_by_stratum": self.observed_draw_ids_by_stratum,
            "projection_record_hashes": tuple(
                item.assert_integrity() for item in self.projection_records
            ),
            "cell_interval_hashes": tuple(
                item.assert_integrity() for item in self.cell_intervals
            ),
            "endpoint_interval_hashes": tuple(
                item.assert_integrity() for item in self.endpoint_intervals
            ),
            "trajectory_alpha_ledger_hash": self.trajectory_alpha_ledger_hash,
            "trajectory_familywise_alpha": self.trajectory_familywise_alpha,
            "trajectory_alpha_committed_before_generation": (
                self.trajectory_alpha_committed_before_generation
            ),
            "generation_alpha_allocation": self.generation_alpha_allocation,
            "generation_alpha_committed_before_redesign": (
                self.generation_alpha_committed_before_redesign
            ),
            "redesign_alpha_allocation": self.redesign_alpha_allocation,
            "family_cell_count": self.family_cell_count,
            "cell_alpha": self.cell_alpha,
            "proposal_authority": self.proposal_authority,
            "bce_mask_commit_authority": self.bce_mask_commit_authority,
            "terminal_validation_seen": self.terminal_validation_seen,
            "family_alpha_policy_kind": self.family_alpha_policy_kind,
        }

    @property
    def receipt_hash(self) -> str:
        cached = self._receipt_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        return _assert_content_integrity(
            self,
            payload=self._hash_payload(),
            cache_name="_receipt_hash",
            label="frozen ladder certification receipt",
            expected_hash=expected_hash,
        )


@dataclass(frozen=True, slots=True)
class V1FQProposalBudget:
    campaign_contract_hash: str
    model_identity_hash: str
    query_contract_hash: str
    preprocessing_hash: str
    current_mask_hash: str
    raw_logit_stream_source_hash: str
    sampling_law_hash: str
    proposal_generation: int
    route_count: int
    route_axis_hash: str
    f_upper_budget: float
    q_upper_budget: float
    f_units: str = F_UNITS
    q_units: str = Q_UNITS
    proposal_authority: str = PROPOSAL_AUTHORITY
    bce_mask_commit_authority: bool = False
    schema: str = FQ_PROPOSAL_BUDGET_SCHEMA
    _budget_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _validate_coordinates(self)
        f_budget = _finite_nonnegative(self.f_upper_budget, name="F upper budget")
        q_budget = _finite_nonnegative(self.q_upper_budget, name="Q upper budget")
        if (self.f_units, self.q_units) != (F_UNITS, Q_UNITS):
            raise ValueError("F/Q proposal budget units are invalid")
        if self.proposal_authority != PROPOSAL_AUTHORITY:
            raise ValueError("F/Q budget has proposal authority only")
        if self.bce_mask_commit_authority is not False:
            raise ValueError("F/Q budget cannot grant BCE commit authority")
        if self.schema != FQ_PROPOSAL_BUDGET_SCHEMA:
            raise ValueError("F/Q proposal budget schema is invalid")
        object.__setattr__(self, "f_upper_budget", f_budget)
        object.__setattr__(self, "q_upper_budget", q_budget)
        self.assert_integrity()

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            **_coordinate_payload(self),
            "f_upper_budget": self.f_upper_budget,
            "q_upper_budget": self.q_upper_budget,
            "f_units": self.f_units,
            "q_units": self.q_units,
            "proposal_authority": self.proposal_authority,
            "bce_mask_commit_authority": self.bce_mask_commit_authority,
        }

    @property
    def budget_hash(self) -> str:
        cached = self._budget_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        return _assert_content_integrity(
            self,
            payload=self._hash_payload(),
            cache_name="_budget_hash",
            label="F/Q proposal budget",
            expected_hash=expected_hash,
        )


@dataclass(frozen=True, slots=True)
class V1FrozenLadderDecision:
    receipt_hash: str
    budget_hash: str
    design_hash: str
    ladder_hash: str
    status: DecisionStatus
    endpoint_outcomes: tuple[tuple[int, str, float, float, bool, bool, bool], ...]
    selected_endpoint_index: int | None
    selected_action_hash: str | None
    reason: str
    proposal_authority: str = PROPOSAL_AUTHORITY
    bce_mask_commit_authority: bool = False
    oracle_required_before_mask_commit: bool = True
    schema: str = FROZEN_LADDER_DECISION_SCHEMA
    _decision_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in ("receipt_hash", "budget_hash", "design_hash", "ladder_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if self.status not in {
            "certified_largest_endpoint",
            "refuse_no_certified_endpoint",
            "refuse_inconsistent_confidence_sequence",
        }:
            raise ValueError("unknown frozen-ladder decision status")
        outcomes = tuple(self.endpoint_outcomes)
        if not outcomes:
            raise ValueError("frozen-ladder decision requires endpoint outcomes")
        normalized: list[tuple[int, str, float, float, bool, bool, bool]] = []
        for position, row in enumerate(outcomes, start=1):
            if len(row) != 7:
                raise ValueError("endpoint outcome rows require seven fields")
            endpoint, action_hash, f_upper, q_upper, f_ok, q_ok, certified = row
            endpoint_i = _positive_int(endpoint, name="endpoint outcome index")
            if endpoint_i != position:
                raise ValueError("endpoint outcomes must be contiguous from one")
            action = _identity(action_hash, name="endpoint action hash")
            f_value = _finite_nonnegative(f_upper, name="endpoint F upper")
            q_value = _finite_nonnegative(q_upper, name="endpoint Q upper")
            if not all(type(flag) is bool for flag in (f_ok, q_ok, certified)):
                raise ValueError("endpoint outcome decisions must be booleans")
            if certified != (f_ok and q_ok):
                raise ValueError("endpoint certification must require both F and Q")
            normalized.append(
                (endpoint_i, action, f_value, q_value, f_ok, q_ok, certified)
            )
        certified_rows = [row for row in normalized if row[-1]]
        if self.status == "certified_largest_endpoint":
            if not certified_rows:
                raise ValueError("certified decision has no certified endpoint")
            largest = certified_rows[-1]
            if (
                self.selected_endpoint_index != largest[0]
                or self.selected_action_hash != largest[1]
            ):
                raise ValueError(
                    "decision did not select the largest certified endpoint"
                )
        elif (
            self.selected_endpoint_index is not None
            or self.selected_action_hash is not None
        ):
            raise ValueError("a refusal cannot name a selected endpoint")
        object.__setattr__(
            self, "reason", _identity(self.reason, name="decision reason")
        )
        if self.proposal_authority != PROPOSAL_AUTHORITY:
            raise ValueError("frozen-ladder decision has proposal authority only")
        if self.bce_mask_commit_authority is not False:
            raise ValueError("frozen-ladder decision cannot grant BCE commit authority")
        if self.oracle_required_before_mask_commit is not True:
            raise ValueError("the paired oracle remains mandatory before mask commit")
        if self.schema != FROZEN_LADDER_DECISION_SCHEMA:
            raise ValueError("frozen-ladder decision schema is invalid")
        object.__setattr__(self, "endpoint_outcomes", tuple(normalized))
        self.assert_integrity()

    def _hash_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "receipt_hash",
                "budget_hash",
                "design_hash",
                "ladder_hash",
                "status",
                "endpoint_outcomes",
                "selected_endpoint_index",
                "selected_action_hash",
                "reason",
                "proposal_authority",
                "bce_mask_commit_authority",
                "oracle_required_before_mask_commit",
            )
        }

    @property
    def decision_hash(self) -> str:
        cached = self._decision_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        return _assert_content_integrity(
            self,
            payload=self._hash_payload(),
            cache_name="_decision_hash",
            label="frozen ladder decision",
            expected_hash=expected_hash,
        )


class V1FrozenLadderCertifier:
    """Mutable acquisition state; every emitted evidence object is immutable."""

    def __init__(self, design: V1FrozenLadderCertificationDesign) -> None:
        if type(design) is not V1FrozenLadderCertificationDesign:
            raise TypeError("frozen-ladder certifier requires its typed design")
        # Construction of the frozen design already established its content
        # identity.  Full re-hashing of a million-context order is reserved
        # for replay/authority boundaries, not repeated on every scalar row.
        design.design_hash
        self.design = design
        self._records: list[V1FrozenLadderProjectionRecord] = []
        self._contexts: dict[str, list[str]] = {
            name: [] for name in design.stratum_names
        }
        self._draws: dict[str, list[str]] = {name: [] for name in design.stratum_names}
        self._seen_draws: set[str] = set()
        self._sequences: dict[tuple[int, str, str], BoundedWoRBettingCS] = {}
        self._upper_only_cells: set[tuple[int, str, str]] = set()
        expected_by_name = dict(design.certification_context_ids_by_stratum)
        for endpoint in design.ladder.actions:
            for role in CERTIFICATION_ROLES:
                for stratum_index, stratum in enumerate(design.stratum_names):
                    y_bound = design.endpoint_bound(
                        stratum_index=stratum_index,
                        endpoint_index=endpoint.endpoint_index,
                    )
                    bound = (
                        y_bound
                        if role == F_ROLE
                        else _outward_nonnegative_square(y_bound)
                    )
                    if bound == 0.0:
                        continue
                    self._sequences[(endpoint.endpoint_index, role, stratum)] = (
                        BoundedWoRBettingCS(
                            stratum_id=stratum,
                            population_size=len(expected_by_name[stratum]),
                            alpha=design.cell_alpha,
                            lower_bound=0.0,
                            upper_bound=bound,
                            sealed_context_order=expected_by_name[stratum],
                            sealed_order_hash=design.sealed_population_order_hashes[
                                stratum_index
                            ],
                            claim_bearing=True,
                        )
                    )

    @property
    def sample_size(self) -> int:
        return len(self._records)

    @property
    def projection_records(self) -> tuple[V1FrozenLadderProjectionRecord, ...]:
        return tuple(self._records)

    def record(
        self,
        *,
        stratum_id: str,
        context_id: str,
        draw_id: str,
        signed_projection_by_action: Mapping[str, float],
        projection_proof_hash: str,
        numerical_radius_by_action: Mapping[str, float] | None = None,
    ) -> V1FrozenLadderProjectionRecord:
        """Atomically append the next sealed context in one chosen stratum.

        Historical callers may omit ``numerical_radius_by_action`` and retain
        the registered v1 point estimand.  Production projection must provide
        the full radius axis.  Its confidence sequences observe conservative
        pointwise F/Q *upper envelopes*; this preserves upper-budget authority
        while deliberately relinquishing lower-bound and exact-census claims.
        """

        stratum = _identity(stratum_id, name="stratum id")
        context = _identity(context_id, name="context id")
        draw = _identity(draw_id, name="draw id")
        proof_hash = _identity(projection_proof_hash, name="projection proof hash")
        expected_by_name = dict(self.design.certification_context_ids_by_stratum)
        if stratum not in expected_by_name:
            raise ValueError("projection row names an unregistered stratum")
        observed_count = len(self._contexts[stratum])
        expected_order = expected_by_name[stratum]
        if observed_count >= len(expected_order):
            raise ValueError("projection row exceeds its finite stratum population")
        expected_context = expected_order[observed_count]
        if context != expected_context:
            raise ValueError(
                f"observed context differs from sealed prefix: expected "
                f"{expected_context!r}, received {context!r}"
            )
        if context in self.design.ladder.discovery_context_ids:
            raise ValueError("discovery context cannot become certification evidence")
        if draw in self._seen_draws or draw in self.design.ladder.discovery_draw_ids:
            raise ValueError("certification draws must be fresh and globally unique")

        action_hashes = self.design.ladder.action_hashes
        if set(signed_projection_by_action) != set(action_hashes) or len(
            signed_projection_by_action
        ) != len(action_hashes):
            raise ValueError(
                "projection row axis differs from the frozen action ladder"
            )
        projections = tuple(
            float(signed_projection_by_action[key]) for key in action_hashes
        )
        if any(not math.isfinite(value) for value in projections):
            raise ValueError("signed endpoint projections must be finite")
        enclosed = numerical_radius_by_action is not None
        if enclosed:
            if set(numerical_radius_by_action) != set(action_hashes) or len(
                numerical_radius_by_action
            ) != len(action_hashes):
                raise ValueError(
                    "projection numerical-radius axis differs from the frozen action ladder"
                )
            radii = tuple(
                _finite_nonnegative(
                    numerical_radius_by_action[key],
                    name="projection numerical radius",
                )
                for key in action_hashes
            )
        else:
            radii = (0.0,) * len(action_hashes)
        stratum_index = self.design.stratum_names.index(stratum)
        cell_updates: list[
            tuple[tuple[int, str, str], BoundedWoRBettingCS, float, bool]
        ] = []
        for endpoint, projection, radius in zip(
            self.design.ladder.actions, projections, radii, strict=True
        ):
            bound = self.design.endpoint_bound(
                stratum_index=stratum_index,
                endpoint_index=endpoint.endpoint_index,
            )
            if enclosed:
                f_lower, f_upper, q_lower, q_upper = (
                    _outward_projection_functional_bounds(
                        center=projection,
                        radius=radius,
                        analytic_absolute_bound=bound,
                    )
                )
                role_values = (
                    (F_ROLE, f_upper, f_lower != f_upper),
                    (Q_ROLE, q_upper, q_lower != q_upper),
                )
            else:
                if abs(projection) > bound:
                    raise ValueError(
                        "signed endpoint projection violated its analytic endpoint envelope"
                    )
                squared = projection * projection
                squared_bound = _outward_nonnegative_square(bound)
                if not math.isfinite(squared) or squared > squared_bound:
                    raise ValueError(
                        "squared endpoint projection violated its analytic endpoint envelope"
                    )
                role_values = (
                    (F_ROLE, abs(projection), False),
                    (Q_ROLE, squared, False),
                )
            for role, value, upper_only in role_values:
                key = (endpoint.endpoint_index, role, stratum)
                sequence = self._sequences.get((endpoint.endpoint_index, role, stratum))
                if sequence is not None:
                    if not 0.0 <= value <= sequence.upper_bound:
                        raise ValueError(
                            "numerically enclosed F/Q observation left its analytic support"
                        )
                    cell_updates.append((key, sequence, value, upper_only))
        record = V1FrozenLadderProjectionRecord(
            design_hash=self.design.design_hash,
            ladder_hash=self.design.ladder.ladder_hash,
            stratum_id=stratum,
            context_id=context,
            draw_id=draw,
            projection_proof_hash=proof_hash,
            endpoint_action_hashes=action_hashes,
            signed_projection_by_endpoint=projections,
            numerical_radius_by_endpoint=radii if enclosed else (),
        )

        # Every possible refusal above precedes mutation.  The following
        # updates therefore form one atomic logical row across all cells.
        for key, sequence, value, upper_only in cell_updates:
            sequence.record(value, context_id=context)
            if upper_only:
                self._upper_only_cells.add(key)
        self._contexts[stratum].append(context)
        self._draws[stratum].append(draw)
        self._seen_draws.add(draw)
        self._records.append(record)
        return record

    def _cell_interval(
        self,
        *,
        endpoint: V1FrozenLadderAction,
        role: str,
        stratum_index: int,
        stratum: str,
    ) -> V1FrozenLadderCellInterval:
        y_bound = self.design.endpoint_bound(
            stratum_index=stratum_index,
            endpoint_index=endpoint.endpoint_index,
        )
        bound = y_bound if role == F_ROLE else _outward_nonnegative_square(y_bound)
        count = len(self._contexts[stratum])
        population = self.design.population_sizes[stratum_index]
        sequence = self._sequences.get((endpoint.endpoint_index, role, stratum))
        if sequence is None:
            snapshot_payload = {
                "method": ZERO_ENVELOPE_METHOD,
                "stratum_id": stratum,
                "population_size": population,
                "sample_size": count,
                "observed_context_ids": tuple(self._contexts[stratum]),
            }
            return V1FrozenLadderCellInterval(
                endpoint_index=endpoint.endpoint_index,
                action_hash=endpoint.action_hash,
                role=role,
                units=F_UNITS if role == F_ROLE else Q_UNITS,
                stratum_id=stratum,
                population_size=population,
                sample_size=count,
                sample_mean=None if count == 0 else 0.0,
                lower=0.0,
                upper=0.0,
                value_upper_bound=0.0,
                alpha=self.design.cell_alpha,
                full_census=count == population,
                empty=False,
                sealed_order_hash=self.design.sealed_population_order_hashes[
                    stratum_index
                ],
                method=ZERO_ENVELOPE_METHOD,
                evidence_snapshot_hash=content_hash(snapshot_payload),
            )
        snapshot = sequence.interval()
        upper_only = (
            endpoint.endpoint_index,
            role,
            stratum,
        ) in self._upper_only_cells
        if upper_only:
            evidence_snapshot_hash = content_hash(
                {
                    "source_betting_snapshot_hash": snapshot.content_hash,
                    "source_method": snapshot.method,
                    "reported_method": NUMERICALLY_ENCLOSED_WOR_UPPER_CS_METHOD,
                    "lower_confidence_authority": False,
                    "upper_confidence_authority": True,
                    "population_census_complete": snapshot.full_census,
                    "exact_real_census_claimed": False,
                }
            )
            return V1FrozenLadderCellInterval(
                endpoint_index=endpoint.endpoint_index,
                action_hash=endpoint.action_hash,
                role=role,
                units=F_UNITS if role == F_ROLE else Q_UNITS,
                stratum_id=stratum,
                population_size=snapshot.population_size,
                sample_size=snapshot.sample_size,
                sample_mean=snapshot.sample_mean,
                lower=0.0 if not snapshot.empty else snapshot.lower,
                upper=snapshot.upper,
                value_upper_bound=bound,
                alpha=snapshot.alpha,
                full_census=False,
                empty=snapshot.empty,
                sealed_order_hash=snapshot.sealed_order_hash,
                method=NUMERICALLY_ENCLOSED_WOR_UPPER_CS_METHOD,
                evidence_snapshot_hash=evidence_snapshot_hash,
                schema=FROZEN_LADDER_ENCLOSED_CELL_INTERVAL_SCHEMA,
                population_census_complete=snapshot.full_census,
                lower_confidence_authority=False,
                upper_confidence_authority=True,
                observation_semantics=(
                    NUMERICALLY_ENCLOSED_UPPER_OBSERVATION_SEMANTICS
                ),
            )
        return V1FrozenLadderCellInterval(
            endpoint_index=endpoint.endpoint_index,
            action_hash=endpoint.action_hash,
            role=role,
            units=F_UNITS if role == F_ROLE else Q_UNITS,
            stratum_id=stratum,
            population_size=snapshot.population_size,
            sample_size=snapshot.sample_size,
            sample_mean=snapshot.sample_mean,
            lower=snapshot.lower,
            upper=snapshot.upper,
            value_upper_bound=bound,
            alpha=snapshot.alpha,
            full_census=snapshot.full_census,
            empty=snapshot.empty,
            sealed_order_hash=snapshot.sealed_order_hash,
            method=snapshot.method,
            evidence_snapshot_hash=snapshot.content_hash,
        )

    def receipt(self) -> V1FrozenLadderCertificationReceipt:
        """Emit a simultaneous immutable receipt at the current stopping time."""

        # Receipt construction uses the frozen cached identity on the hot
        # path.  ``verify_frozen_ladder_certification_receipt`` performs the
        # full design and evidence replay before any decision can issue.
        self.design.design_hash
        cells = tuple(
            self._cell_interval(
                endpoint=endpoint,
                role=role,
                stratum_index=stratum_index,
                stratum=stratum,
            )
            for endpoint in self.design.ladder.actions
            for role in CERTIFICATION_ROLES
            for stratum_index, stratum in enumerate(self.design.stratum_names)
        )
        cell_map = {
            (item.endpoint_index, item.role, item.stratum_id): item for item in cells
        }
        weights = dict(self.design.stratum_weights)
        endpoints: list[V1FrozenLadderEndpointInterval] = []
        for endpoint in self.design.ladder.actions:
            f_cells = tuple(
                cell_map[(endpoint.endpoint_index, F_ROLE, stratum)]
                for stratum in self.design.stratum_names
            )
            q_cells = tuple(
                cell_map[(endpoint.endpoint_index, Q_ROLE, stratum)]
                for stratum in self.design.stratum_names
            )
            f_lower, f_upper = _weighted_cell_projection(f_cells, weights=weights)
            q_lower, q_upper = _weighted_cell_projection(q_cells, weights=weights)
            lower_authority = all(
                item.lower_confidence_authority for item in (*f_cells, *q_cells)
            )
            upper_authority = all(
                item.upper_confidence_authority for item in (*f_cells, *q_cells)
            )
            if not lower_authority:
                # The upper-envelope path is deliberately one-sided.  Zero is
                # the only authority-free lower support bound admissible here.
                f_lower = 0.0
                q_lower = 0.0
            sqrt_q_lower, sqrt_q_upper = _sqrt_interval(q_lower, q_upper)
            population_complete = all(
                bool(item.population_census_complete) for item in (*f_cells, *q_cells)
            )
            endpoints.append(
                V1FrozenLadderEndpointInterval(
                    endpoint_index=endpoint.endpoint_index,
                    action_hash=endpoint.action_hash,
                    f_lower=f_lower,
                    f_upper=f_upper,
                    q_lower=q_lower,
                    q_upper=q_upper,
                    sqrt_q_lower=sqrt_q_lower,
                    sqrt_q_upper=sqrt_q_upper,
                    empty=any(item.empty for item in (*f_cells, *q_cells)),
                    full_census=all(item.full_census for item in (*f_cells, *q_cells)),
                    schema=(
                        FROZEN_LADDER_ENDPOINT_INTERVAL_SCHEMA
                        if lower_authority
                        else FROZEN_LADDER_ENCLOSED_ENDPOINT_INTERVAL_SCHEMA
                    ),
                    population_census_complete=population_complete,
                    lower_confidence_authority=lower_authority,
                    upper_confidence_authority=upper_authority,
                    observation_semantics=(
                        EXACT_POINT_OBSERVATION_SEMANTICS
                        if lower_authority
                        else NUMERICALLY_ENCLOSED_UPPER_OBSERVATION_SEMANTICS
                    ),
                )
            )
        coordinates = _coordinate_payload(self.design.ladder)
        return V1FrozenLadderCertificationReceipt(
            design_hash=self.design.design_hash,
            ladder_hash=self.design.ladder.ladder_hash,
            envelope_certificate_hash=self.design.envelope_certificate_hash,
            **coordinates,
            ladder_redesign_generation=self.design.ladder_redesign_generation,
            certification_context_population_hash=(
                self.design.certification_context_population_hash
            ),
            sealed_population_order_hashes=self.design.sealed_population_order_hashes,
            stratum_weights=self.design.stratum_weights,
            observed_context_ids_by_stratum=tuple(
                (name, tuple(self._contexts[name]))
                for name in self.design.stratum_names
            ),
            observed_draw_ids_by_stratum=tuple(
                (name, tuple(self._draws[name])) for name in self.design.stratum_names
            ),
            projection_records=tuple(self._records),
            cell_intervals=cells,
            endpoint_intervals=tuple(endpoints),
            trajectory_alpha_ledger_hash=self.design.trajectory_alpha_ledger_hash,
            trajectory_familywise_alpha=self.design.trajectory_familywise_alpha,
            trajectory_alpha_committed_before_generation=(
                self.design.trajectory_alpha_committed_before_generation
            ),
            generation_alpha_allocation=self.design.generation_alpha_allocation,
            generation_alpha_committed_before_redesign=(
                self.design.generation_alpha_committed_before_redesign
            ),
            redesign_alpha_allocation=self.design.redesign_alpha_allocation,
            family_cell_count=self.design.family_cell_count,
            cell_alpha=self.design.cell_alpha,
        )


def verify_frozen_ladder_certification_receipt(
    *,
    design: V1FrozenLadderCertificationDesign,
    receipt: V1FrozenLadderCertificationReceipt,
    expected_receipt_hash: str | None = None,
) -> str:
    """Replay raw scalar rows and verify a receipt in its original chart."""

    if (
        type(design) is not V1FrozenLadderCertificationDesign
        or type(receipt) is not V1FrozenLadderCertificationReceipt
    ):
        raise TypeError("receipt replay requires typed design and receipt objects")
    design_hash = design.assert_integrity()
    observed_hash = receipt.assert_integrity(expected_receipt_hash)
    if receipt.design_hash != design_hash:
        raise ValueError("certification receipt belongs to another design")
    if receipt.ladder_hash != design.ladder.ladder_hash or not _coordinates_match(
        receipt, design.ladder
    ):
        raise ValueError("certification receipt was recontextualized across ladders")
    if (
        receipt.envelope_certificate_hash != design.envelope_certificate_hash
        or receipt.certification_context_population_hash
        != design.certification_context_population_hash
        or receipt.sealed_population_order_hashes
        != design.sealed_population_order_hashes
    ):
        raise ValueError("certification receipt changed its population or envelope")
    replay = V1FrozenLadderCertifier(design)
    for record in receipt.projection_records:
        replay.record(
            stratum_id=record.stratum_id,
            context_id=record.context_id,
            draw_id=record.draw_id,
            signed_projection_by_action=dict(
                zip(
                    record.endpoint_action_hashes,
                    record.signed_projection_by_endpoint,
                    strict=True,
                )
            ),
            projection_proof_hash=record.projection_proof_hash,
            numerical_radius_by_action=(
                dict(
                    zip(
                        record.endpoint_action_hashes,
                        record.numerical_radius_by_endpoint,
                        strict=True,
                    )
                )
                if record.has_numerical_enclosure
                else None
            ),
        )
    replayed = replay.receipt()
    if replayed.receipt_hash != observed_hash:
        raise ValueError("certification receipt does not replay from its scalar rows")
    return observed_hash


def select_largest_certified_endpoint(
    *,
    design: V1FrozenLadderCertificationDesign,
    receipt: V1FrozenLadderCertificationReceipt,
    budget: V1FQProposalBudget,
) -> V1FrozenLadderDecision:
    """Choose the largest simultaneously certified endpoint, or refuse."""

    if (
        type(design) is not V1FrozenLadderCertificationDesign
        or type(receipt) is not V1FrozenLadderCertificationReceipt
        or type(budget) is not V1FQProposalBudget
    ):
        raise TypeError(
            "largest-endpoint selection requires typed design, receipt, and budget"
        )
    receipt_hash = verify_frozen_ladder_certification_receipt(
        design=design,
        receipt=receipt,
    )
    budget_hash = budget.assert_integrity()
    if not _coordinates_match(receipt, budget):
        raise ValueError("F/Q budget coordinates differ from the certification receipt")
    outcomes = tuple(
        (
            item.endpoint_index,
            item.action_hash,
            item.f_upper,
            item.q_upper,
            bool(item.f_upper <= budget.f_upper_budget),
            bool(item.q_upper <= budget.q_upper_budget),
            bool(
                not item.empty
                and item.f_upper <= budget.f_upper_budget
                and item.q_upper <= budget.q_upper_budget
            ),
        )
        for item in receipt.endpoint_intervals
    )
    if receipt.empty:
        status: DecisionStatus = "refuse_inconsistent_confidence_sequence"
        selected_endpoint = None
        selected_action = None
        reason = "at least one simultaneous F/Q confidence sequence is empty"
    else:
        certified = tuple(row for row in outcomes if row[-1])
        if certified:
            status = "certified_largest_endpoint"
            selected_endpoint = certified[-1][0]
            selected_action = certified[-1][1]
            reason = (
                "largest frozen endpoint whose simultaneous F and Q upper bounds "
                "fit the supplied proposal budgets"
            )
        else:
            status = "refuse_no_certified_endpoint"
            selected_endpoint = None
            selected_action = None
            reason = "no frozen endpoint fits both simultaneous proposal budgets"
    return V1FrozenLadderDecision(
        receipt_hash=receipt_hash,
        budget_hash=budget_hash,
        design_hash=receipt.design_hash,
        ladder_hash=receipt.ladder_hash,
        status=status,
        endpoint_outcomes=outcomes,
        selected_endpoint_index=selected_endpoint,
        selected_action_hash=selected_action,
        reason=reason,
    )


__all__ = [
    "ACTION_ROUTE_SUM_ENDPOINT_ENVELOPE_KIND",
    "ACTION_SPECIFIC_ENDPOINT_ENVELOPE_KIND",
    "CERTIFICATION_ROLES",
    "ENDPOINT_ENVELOPE_DERIVATION_KINDS",
    "F_ROLE",
    "F_UNITS",
    "FROZEN_LADDER_ENCLOSED_CERTIFICATION_RECEIPT_SCHEMA",
    "FROZEN_LADDER_ENCLOSED_CELL_INTERVAL_SCHEMA",
    "FROZEN_LADDER_ENCLOSED_ENDPOINT_INTERVAL_SCHEMA",
    "FROZEN_LADDER_ENCLOSED_PROJECTION_RECORD_SCHEMA",
    "FROZEN_LADDER_ROUTE_SUM_ENDPOINT_ENVELOPE_PROOF_SCHEMA",
    "PROPOSAL_AUTHORITY",
    "NUMERICALLY_ENCLOSED_UPPER_OBSERVATION_SEMANTICS",
    "NUMERICALLY_ENCLOSED_WOR_UPPER_CS_METHOD",
    "Q_ROLE",
    "Q_UNITS",
    "ROUTE_SUM_ENDPOINT_ENVELOPE_KIND",
    "SQRT_Q_UNITS",
    "UNIFORM_CELL_ALPHA_POLICY",
    "V1FQProposalBudget",
    "V1FrozenCandidateLadder",
    "V1FrozenLadderAction",
    "V1FrozenLadderCellInterval",
    "V1FrozenLadderCertificationDesign",
    "V1FrozenLadderCertificationReceipt",
    "V1FrozenLadderCertifier",
    "V1FrozenLadderDecision",
    "V1FrozenLadderEndpointEnvelopeCertificate",
    "V1FrozenLadderEndpointInterval",
    "V1FrozenLadderProjectionRecord",
    "build_v1_frozen_ladder_route_sum_endpoint_envelope",
    "select_largest_certified_endpoint",
    "verify_frozen_ladder_certification_receipt",
]
