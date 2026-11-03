"""Production source-native join for the complete block-support raw union.

The reference V2/V3 join deliberately cannot cross this boundary.  This
module accepts only the production source-native live-cover and live-family
receipt classes, one of each for every registered alias in its exact order.
It binds their physical, context, KKT, pair-axis, source, and rectangle-count
identities to the existing P-excluded within-shell proof.

The resulting V4 scalar is still *raw*: the D-fixed and residual preferred-
plane deficits have not been debited.  It may therefore serve only as the raw
union parent of a later canonical block-observation envelope.  It grants no
generic observation, population-separation, full-U, FQR, C/K/O, commit,
terminal, or Pareto authority.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
import math
from numbers import Real
from typing import Final, NoReturn, Sequence

from .contracts import content_hash
from .modern_v1_signed_work_block_support_complete_alternate_union import (
    V1BlockSupportPExcludedWithinShellRawSlopeReceipt,
)
from .modern_v1_signed_work_block_support_alternate_union_solver_cuda import (
    V1BlockSupportWithinShellCUDAExecutionReceipt,
)
from .modern_v1_signed_work_block_support_domain_binding import (
    V1BlockSupportAliasCoverBindingV2,
    V1BlockSupportBoundWitnessSource,
    V1BlockSupportCanonicalBlockContextIdentity,
)
from .modern_v1_signed_work_block_support_outside_pair_production import (
    OUTSIDE_PAIR_LIVE_COVER_AUTHORITY_V2_SCHEMA,
    OUTSIDE_PAIR_LIVE_COVER_AUTHORITY_V2_SCOPE,
    V1BlockSupportOutsidePairLiveCoverAuthorityReceiptV2,
)
from .modern_v1_signed_work_block_support_outside_pair_live_family import (
    OUTSIDE_PAIR_LIVE_FAMILY_V3_AUTHORITY,
    OUTSIDE_PAIR_LIVE_FAMILY_V3_EQUATION,
    OUTSIDE_PAIR_LIVE_FAMILY_V3_SCHEMA,
    V1BlockSupportOutsidePairLiveFamilyReceiptV3,
)
from .modern_v1_signed_work_block_support_source_cuda import (
    V1SignedWorkBlockSupportExecutionMode,
    V1SignedWorkBlockSupportSourceReceipt,
    V1SignedWorkBlockSupportSourceRole,
)
from .modern_v1_signed_work_physical_alternate_domain_cuda import (
    V1SignedWorkPhysicalAlternateBlockSupportBindingReceipt,
)
from .modern_v1_signed_work_variable_shell_outside_exclusion_cuda import (
    V1SignedWorkVariableShellOutsidePairRectangle,
)


COMPLETE_COVER_V3_SCHEMA: Final = (
    "partition_free_v1_block_support_complete_constraint_cover_binding_v3"
)
COMPLETE_RAW_V4_SCHEMA: Final = (
    "partition_free_v1_block_support_complete_union_raw_slope_v4"
)
COMPLETE_COVER_V3_AUTHORITY: Final = (
    "production_source_native_complete_physical_cover_only_no_canonical_block_"
    "observation_full_U_FQR_C_K_O_commit_terminal_or_pareto_authority_v3"
)
COMPLETE_RAW_V4_AUTHORITY: Final = (
    "production_source_native_complete_raw_union_for_future_canonical_block_"
    "observation_input_only_no_statistical_separation_full_U_FQR_C_K_O_commit_"
    "terminal_or_pareto_authority_v4"
)

_PREFIX: Final = "partition-free-v1-block-support-complete-union-production-"
_COVER_TOKEN: Final = object()
_RAW_TOKEN: Final = object()
_WITHIN_SHELL_CONTEXT_PREFIX: Final = (
    "partition-free-v1-block-support-within-shell-device-block-context-axis-"
)


class V1BlockSupportCompleteUnionProductionRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_SOURCE = "stale_source"
    SOURCE_MISMATCH = "source_mismatch"
    CONTEXT_MISMATCH = "context_mismatch"
    P_MISMATCH = "P_mismatch"
    ALIAS_AXIS_MISMATCH = "alias_axis_mismatch"
    KKT_MISMATCH = "KKT_mismatch"
    PAIR_AXIS_MISMATCH = "pair_axis_mismatch"
    RECTANGLE_MISMATCH = "rectangle_mismatch"
    CUDA_CUSTODY_REQUIRED = "cuda_custody_required"
    NUMERICAL_REFUSAL = "numerical_refusal"
    AUTHORITY_VIOLATION = "authority_violation"
    RECEIPT_TAMPERING = "receipt_tampering"


class V1BlockSupportCompleteUnionProductionRefusal(RuntimeError):
    def __init__(
        self,
        code: V1BlockSupportCompleteUnionProductionRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1BlockSupportCompleteUnionProductionRefusalCode:
            raise TypeError("production complete-union refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1BlockSupportCompleteUnionProductionRefusalCode,
    message: str,
) -> NoReturn:
    raise V1BlockSupportCompleteUnionProductionRefusal(code, message)


def _payload(value: object) -> dict[str, object]:
    return {
        item.name: (
            getattr(value, item.name).value
            if isinstance(getattr(value, item.name), Enum)
            else getattr(value, item.name)
        )
        for item in fields(value)
        if not item.name.startswith("_") and item.name != "receipt_hash"
    }


def _raw_payload(
    value: "V1BlockSupportCompleteUnionRawSlopeReceiptV4",
) -> dict[str, object]:
    """Encode the theorem's exact empty-rectangle ``+inf`` sentinel safely."""

    payload = _payload(value)
    normalized_candidates = []
    for candidate in payload["candidate_axis"]:
        coordinates = list(candidate)
        numeric = coordinates[4]
        if isinstance(numeric, float) and math.isinf(numeric):
            coordinates[4] = "+inf" if numeric > 0.0 else "-inf"
        normalized_candidates.append(tuple(coordinates))
    payload["candidate_axis"] = tuple(normalized_candidates)
    return payload


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1BlockSupportCompleteUnionProductionRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _refuse(
            V1BlockSupportCompleteUnionProductionRefusalCode.INVALID_TYPE,
            f"{name} must be an exact integer no smaller than {minimum}",
        )
    return value


def _real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1BlockSupportCompleteUnionProductionRefusalCode.INVALID_TYPE,
            f"{name} must be a real scalar",
        )
    result = float(value)
    if math.isnan(result):
        _refuse(
            V1BlockSupportCompleteUnionProductionRefusalCode.NUMERICAL_REFUSAL,
            f"{name} may not be NaN",
        )
    return result


def _closed(values: Sequence[object], *, owner: str) -> None:
    if any(type(value) is not bool or value for value in values):
        _refuse(
            V1BlockSupportCompleteUnionProductionRefusalCode.AUTHORITY_VIOLATION,
            f"{owner} attempted to open downstream authority",
        )


def _within_shell_context_axis_hash(context_ids: Sequence[str]) -> str:
    return content_hash(tuple(context_ids), prefix=_WITHIN_SHELL_CONTEXT_PREFIX)


@dataclass(frozen=True, slots=True)
class V1BlockSupportCompleteConstraintCoverBindingReceiptV3:
    """Exact production cover over within-shell plus every live rectangle."""

    bound_registration_receipt_hash: str
    preoutcome_domain_receipt_hash: str
    canonical_block_context_receipt_hash: str
    canonical_context_tuple_hash: str
    block_support_context_axis_hash: str
    block_index: int
    block_id: str
    structural_domain_receipt_hash: str
    physical_block_support_binding_receipt_hash: str
    d_fixed_support_receipt_hash: str
    residual_support_receipt_hash: str
    within_shell_execution_receipt_hash: str
    p_excluded_within_receipt_hash: str
    preferred_action_hash: str
    preferred_action_binding_hash: str
    preferred_removed_route_index_hash: str
    preferred_selected_count: int
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    resident_device_identity: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    d_weight_numerator: int
    residual_weight_numerator: int
    population_weight_denominator: int
    ordered_alias_binding_hashes: tuple[str, ...]
    ordered_resource_alias_hashes: tuple[str, ...]
    ordered_live_cover_authority_receipt_hashes: tuple[str, ...]
    ordered_live_family_receipt_hashes: tuple[str, ...]
    ordered_alias_cardinality_axis: tuple[tuple[str, str, str, int, int, int, int], ...]
    ordered_source_pair_kkt_axis: tuple[tuple[str, ...], ...]
    ordered_rectangle_count_axis: tuple[
        tuple[int, int, int, int, int, int, int, int, int], ...
    ]
    receipt_hash: str = field(init=False)
    _issuer_token: InitVar[object] = None
    exact_registered_alias_order: bool = True
    every_alias_present_exactly_once: bool = True
    exact_context_and_physical_source_join: bool = True
    exact_P_excluded_receipt_join: bool = True
    exact_live_cover_family_join: bool = True
    exact_KKT_and_source_native_pair_axis_join: bool = True
    both_outside_rectangles_bound_per_alias: bool = True
    complete_constraint_cover_source_bound: bool = True
    production_cuda_custody: bool = True
    legacy_hash_equivalence_used: bool = False
    canonical_block_observation_raw_union_authority: bool = False
    canonical_block_observation_authority: bool = False
    population_separation_authority: bool = False
    full_u_authority: bool = False
    fqr_authority: bool = False
    c_authority: bool = False
    k_authority: bool = False
    o_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = COMPLETE_COVER_V3_AUTHORITY
    schema: str = COMPLETE_COVER_V3_SCHEMA
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _COVER_TOKEN:
            raise TypeError("production complete-cover-v3 receipts are issuer-only")
        for item in fields(self):
            if (
                not item.name.startswith("_")
                and item.name.endswith("_hash")
                and item.name != "receipt_hash"
            ):
                object.__setattr__(
                    self,
                    item.name,
                    _text(getattr(self, item.name), name=item.name),
                )
        for name in ("block_id", "resident_device_identity", "authority_scope"):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name=name),
            )
        for name in (
            "block_index",
            "preferred_selected_count",
            "proposal_generation",
            "route_count",
            "d_weight_numerator",
            "residual_weight_numerator",
            "population_weight_denominator",
        ):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    name=name,
                    minimum=(
                        1
                        if name
                        in {
                            "preferred_selected_count",
                            "route_count",
                            "population_weight_denominator",
                        }
                        else 0
                    ),
                ),
            )
        aliases = tuple(self.ordered_alias_binding_hashes)
        alias_hashes = tuple(self.ordered_resource_alias_hashes)
        covers = tuple(self.ordered_live_cover_authority_receipt_hashes)
        families = tuple(self.ordered_live_family_receipt_hashes)
        cardinalities = tuple(self.ordered_alias_cardinality_axis)
        sources = tuple(self.ordered_source_pair_kkt_axis)
        counts = tuple(self.ordered_rectangle_count_axis)
        axes = (alias_hashes, covers, families, cardinalities, sources, counts)
        if (
            not aliases
            or any(len(axis) != len(aliases) for axis in axes)
            or any(
                len(set(axis)) != len(axis)
                for axis in (aliases, alias_hashes, covers, families)
            )
            or any(
                type(value) is not str or not value
                for axis in (aliases, alias_hashes, covers, families)
                for value in axis
            )
            or any(
                type(value) is not tuple or len(value) != 7 for value in cardinalities
            )
            or any(type(value) is not tuple or len(value) != 20 for value in sources)
            or any(type(value) is not tuple or len(value) != 9 for value in counts)
        ):
            _refuse(
                V1BlockSupportCompleteUnionProductionRefusalCode.ALIAS_AXIS_MISMATCH,
                "production alias/cover/family source axes are empty, duplicated, or misaligned",
            )
        normalized_cardinalities = []
        for item in cardinalities:
            branch, resource, cardinality, rank, fixed, shell, total = item
            normalized = (
                _text(branch, name="alias branch"),
                _text(resource, name="resource alias id"),
                _text(cardinality, name="cardinality alias id"),
                _integer(rank, name="registered FD rank", minimum=1),
                _integer(fixed, name="fixed outside count"),
                _integer(shell, name="shell prefix count"),
                _integer(total, name="total removed count"),
            )
            if normalized[6] != normalized[4] + normalized[5]:
                _refuse(
                    V1BlockSupportCompleteUnionProductionRefusalCode.ALIAS_AXIS_MISMATCH,
                    "one production alias cardinality does not close",
                )
            normalized_cardinalities.append(normalized)
        normalized_sources = []
        for source_axis in sources:
            normalized_sources.append(
                tuple(
                    _text(value, name="source/pair/KKT identity")
                    for value in source_axis
                )
            )
        normalized_counts = []
        for count_axis in counts:
            normalized = tuple(
                _integer(value, name="rectangle count") for value in count_axis
            )
            (
                route_count,
                scan_count,
                first_p_count,
                first_q_count,
                second_p_count,
                second_q_count,
                first_count,
                second_count,
                complete_count,
            ) = normalized
            if (
                route_count != self.route_count
                or scan_count != route_count
                or first_count != first_p_count * first_q_count
                or second_count != second_p_count * second_q_count
                or complete_count != first_count + second_count
            ):
                _refuse(
                    V1BlockSupportCompleteUnionProductionRefusalCode.RECTANGLE_MISMATCH,
                    "one production rectangle census is incomplete",
                )
            normalized_counts.append(normalized)
        object.__setattr__(self, "ordered_alias_binding_hashes", aliases)
        object.__setattr__(self, "ordered_resource_alias_hashes", alias_hashes)
        object.__setattr__(
            self,
            "ordered_live_cover_authority_receipt_hashes",
            covers,
        )
        object.__setattr__(self, "ordered_live_family_receipt_hashes", families)
        object.__setattr__(
            self,
            "ordered_alias_cardinality_axis",
            tuple(normalized_cardinalities),
        )
        object.__setattr__(
            self,
            "ordered_source_pair_kkt_axis",
            tuple(normalized_sources),
        )
        object.__setattr__(
            self,
            "ordered_rectangle_count_axis",
            tuple(normalized_counts),
        )
        required = (
            self.exact_registered_alias_order,
            self.every_alias_present_exactly_once,
            self.exact_context_and_physical_source_join,
            self.exact_P_excluded_receipt_join,
            self.exact_live_cover_family_join,
            self.exact_KKT_and_source_native_pair_axis_join,
            self.both_outside_rectangles_bound_per_alias,
            self.complete_constraint_cover_source_bound,
            self.production_cuda_custody,
        )
        if any(type(value) is not bool or value is not True for value in required):
            _refuse(
                V1BlockSupportCompleteUnionProductionRefusalCode.AUTHORITY_VIOLATION,
                "production complete cover lost a mandatory source invariant",
            )
        _closed(
            (
                self.legacy_hash_equivalence_used,
                self.canonical_block_observation_raw_union_authority,
                self.canonical_block_observation_authority,
                self.population_separation_authority,
                self.full_u_authority,
                self.fqr_authority,
                self.c_authority,
                self.k_authority,
                self.o_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="production complete cover v3",
        )
        if (
            not self.resident_device_identity.startswith("cuda")
            or self.authority_scope != COMPLETE_COVER_V3_AUTHORITY
            or self.schema != COMPLETE_COVER_V3_SCHEMA
        ):
            _refuse(
                V1BlockSupportCompleteUnionProductionRefusalCode.CUDA_CUSTODY_REQUIRED,
                "production complete-cover device or semantics changed",
            )
        observed = content_hash(_payload(self), prefix=_PREFIX + "cover-integrity-")
        if self._integrity_seal and self._integrity_seal != observed:
            _refuse(
                V1BlockSupportCompleteUnionProductionRefusalCode.RECEIPT_TAMPERING,
                "production complete-cover-v3 receipt was mutated",
            )
        object.__setattr__(self, "_integrity_seal", observed)
        object.__setattr__(
            self,
            "receipt_hash",
            content_hash(_payload(self), prefix=_PREFIX + "cover-"),
        )

    def assert_integrity(self) -> str:
        observed = content_hash(_payload(self), prefix=_PREFIX + "cover-integrity-")
        expected = content_hash(_payload(self), prefix=_PREFIX + "cover-")
        if observed != self._integrity_seal or expected != self.receipt_hash:
            _refuse(
                V1BlockSupportCompleteUnionProductionRefusalCode.RECEIPT_TAMPERING,
                "production complete-cover-v3 receipt no longer replays",
            )
        return expected


@dataclass(frozen=True, slots=True)
class V1BlockSupportCompleteUnionRawSlopeReceiptV4:
    """One-ULP outward raw minimum over the source-bound complete cover."""

    complete_cover_v3_receipt_hash: str
    p_excluded_within_receipt_hash: str
    ordered_alias_binding_hashes: tuple[str, ...]
    ordered_live_cover_authority_receipt_hashes: tuple[str, ...]
    ordered_live_family_receipt_hashes: tuple[str, ...]
    candidate_axis: tuple[tuple[str, int, str, bool, float, tuple[int, ...]], ...]
    winning_domain: str
    winning_alias_ordinal: int
    winning_rectangle_ordinal: int
    winning_source_hash: str
    winning_physical_index_axis: tuple[int, ...]
    winning_raw_lower_input: float
    complete_union_raw_lower_outward: float
    finite_candidate_count: int
    empty_rectangle_count: int
    receipt_hash: str = field(init=False)
    _issuer_token: InitVar[object] = None
    exact_P_exclusion_joined: bool = True
    exact_source_bound_complete_cover_joined: bool = True
    exact_live_cover_family_axes_joined: bool = True
    both_rectangles_replayed_for_every_alias: bool = True
    exact_empty_rectangle_vacuity_only: bool = True
    deterministic_registered_alias_ties: bool = True
    downward_exactly_one_ulp: bool = True
    d_plane_deficit_debited: bool = False
    residual_plane_deficit_debited: bool = False
    production_cuda_custody: bool = True
    canonical_block_observation_raw_union_authority: bool = True
    canonical_block_observation_authority: bool = False
    population_separation_authority: bool = False
    full_u_authority: bool = False
    fqr_authority: bool = False
    c_authority: bool = False
    k_authority: bool = False
    o_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = COMPLETE_RAW_V4_AUTHORITY
    schema: str = COMPLETE_RAW_V4_SCHEMA
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _RAW_TOKEN:
            raise TypeError("production complete-union-v4 receipts are issuer-only")
        for name in (
            "complete_cover_v3_receipt_hash",
            "p_excluded_within_receipt_hash",
            "winning_domain",
            "winning_source_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        aliases = tuple(self.ordered_alias_binding_hashes)
        covers = tuple(self.ordered_live_cover_authority_receipt_hashes)
        families = tuple(self.ordered_live_family_receipt_hashes)
        candidates = tuple(self.candidate_axis)
        if (
            not aliases
            or len(aliases) != len(covers)
            or len(aliases) != len(families)
            or len(set(aliases)) != len(aliases)
            or len(set(covers)) != len(covers)
            or len(set(families)) != len(families)
            or any(
                type(value) is not str or not value
                for axis in (aliases, covers, families)
                for value in axis
            )
            or len(candidates) != 1 + 2 * len(aliases)
        ):
            _refuse(
                V1BlockSupportCompleteUnionProductionRefusalCode.ALIAS_AXIS_MISMATCH,
                "V4 alias/cover/family candidate axes are incomplete",
            )
        object.__setattr__(self, "ordered_alias_binding_hashes", aliases)
        object.__setattr__(
            self,
            "ordered_live_cover_authority_receipt_hashes",
            covers,
        )
        object.__setattr__(self, "ordered_live_family_receipt_hashes", families)
        allowed_domains = {
            "P_excluded_within": 0,
            "add_outside": 1,
            "drop_outside_add_inside": 2,
        }
        normalized_candidates = []
        for index, candidate in enumerate(candidates):
            if type(candidate) is not tuple or len(candidate) != 6:
                _refuse(
                    V1BlockSupportCompleteUnionProductionRefusalCode.INVALID_TYPE,
                    "V4 candidates must be exact six-coordinate tuples",
                )
            domain, alias_ordinal, source_hash, empty, value, physical_axis = candidate
            domain = _text(domain, name="candidate domain")
            source_hash = _text(source_hash, name="candidate source hash")
            alias_ordinal = _integer(
                alias_ordinal,
                name="candidate alias ordinal",
                minimum=-1,
            )
            if domain not in allowed_domains or type(empty) is not bool:
                _refuse(
                    V1BlockSupportCompleteUnionProductionRefusalCode.RECTANGLE_MISMATCH,
                    "V4 candidate domain or empty marker is invalid",
                )
            numeric = _real(value, name="candidate raw lower")
            if type(physical_axis) is not tuple:
                _refuse(
                    V1BlockSupportCompleteUnionProductionRefusalCode.INVALID_TYPE,
                    "candidate physical witness must be an exact tuple",
                )
            physical = tuple(
                _integer(item, name="candidate physical index")
                for item in physical_axis
            )
            if len(physical) not in {0, 2}:
                _refuse(
                    V1BlockSupportCompleteUnionProductionRefusalCode.INVALID_TYPE,
                    "candidate physical witness is empty or one ordered pair",
                )
            if empty:
                if numeric != math.inf or physical:
                    _refuse(
                        V1BlockSupportCompleteUnionProductionRefusalCode.NUMERICAL_REFUSAL,
                        "only a sealed witness-free empty rectangle may carry +inf",
                    )
            elif not math.isfinite(numeric):
                _refuse(
                    V1BlockSupportCompleteUnionProductionRefusalCode.NUMERICAL_REFUSAL,
                    "every nonempty V4 candidate must be finite",
                )
            if index == 0:
                if (
                    domain != "P_excluded_within"
                    or alias_ordinal != -1
                    or source_hash != self.p_excluded_within_receipt_hash
                    or empty
                    or physical
                ):
                    _refuse(
                        V1BlockSupportCompleteUnionProductionRefusalCode.P_MISMATCH,
                        "first V4 candidate must be the finite P-excluded within union",
                    )
            else:
                expected_alias = (index - 1) // 2
                expected_domain = (
                    "add_outside" if (index - 1) % 2 == 0 else "drop_outside_add_inside"
                )
                if (
                    alias_ordinal != expected_alias
                    or domain != expected_domain
                    or source_hash != aliases[expected_alias]
                ):
                    _refuse(
                        V1BlockSupportCompleteUnionProductionRefusalCode.ALIAS_AXIS_MISMATCH,
                        "V4 candidates do not follow within/alias/rectangle order",
                    )
            normalized_candidates.append(
                (domain, alias_ordinal, source_hash, empty, numeric, physical)
            )
        object.__setattr__(self, "candidate_axis", tuple(normalized_candidates))
        winning_alias = _integer(
            self.winning_alias_ordinal,
            name="winning alias ordinal",
            minimum=-1,
        )
        winning_rectangle = _integer(
            self.winning_rectangle_ordinal,
            name="winning rectangle ordinal",
        )
        if type(self.winning_physical_index_axis) is not tuple:
            _refuse(
                V1BlockSupportCompleteUnionProductionRefusalCode.INVALID_TYPE,
                "winning physical witness must be an exact tuple",
            )
        winning_physical = tuple(
            _integer(value, name="winning physical index")
            for value in self.winning_physical_index_axis
        )
        winning_input = _real(
            self.winning_raw_lower_input,
            name="winning raw lower input",
        )
        outward = _real(
            self.complete_union_raw_lower_outward,
            name="complete union raw lower outward",
        )
        if not math.isfinite(winning_input) or not math.isfinite(outward):
            _refuse(
                V1BlockSupportCompleteUnionProductionRefusalCode.NUMERICAL_REFUSAL,
                "V4 winning and outward raw lowers must be finite",
            )
        finite_count = _integer(
            self.finite_candidate_count,
            name="finite candidate count",
            minimum=1,
        )
        empty_count = _integer(
            self.empty_rectangle_count,
            name="empty rectangle count",
        )
        finite = tuple(value for value in normalized_candidates if not value[3])
        expected_winner = min(
            finite,
            key=lambda value: (
                value[4],
                value[1],
                allowed_domains[value[0]],
                value[5],
                value[2],
            ),
        )
        expected_outward = math.nextafter(expected_winner[4], -math.inf)
        if (
            not math.isfinite(expected_outward)
            or finite_count != len(finite)
            or empty_count != sum(value[3] for value in normalized_candidates[1:])
            or self.winning_domain != expected_winner[0]
            or winning_alias != expected_winner[1]
            or winning_rectangle != allowed_domains[expected_winner[0]]
            or self.winning_source_hash != expected_winner[2]
            or winning_physical != expected_winner[5]
            or (winning_alias >= 0 and len(winning_physical) != 2)
            or winning_input != expected_winner[4]
            or outward != expected_outward
        ):
            _refuse(
                V1BlockSupportCompleteUnionProductionRefusalCode.NUMERICAL_REFUSAL,
                "V4 winner, tie order, counts, or exact one-ULP enclosure does not replay",
            )
        object.__setattr__(self, "winning_alias_ordinal", winning_alias)
        object.__setattr__(self, "winning_rectangle_ordinal", winning_rectangle)
        object.__setattr__(self, "winning_physical_index_axis", winning_physical)
        object.__setattr__(self, "winning_raw_lower_input", winning_input)
        object.__setattr__(self, "complete_union_raw_lower_outward", outward)
        object.__setattr__(self, "finite_candidate_count", finite_count)
        object.__setattr__(self, "empty_rectangle_count", empty_count)
        required = (
            self.exact_P_exclusion_joined,
            self.exact_source_bound_complete_cover_joined,
            self.exact_live_cover_family_axes_joined,
            self.both_rectangles_replayed_for_every_alias,
            self.exact_empty_rectangle_vacuity_only,
            self.deterministic_registered_alias_ties,
            self.downward_exactly_one_ulp,
            self.production_cuda_custody,
            self.canonical_block_observation_raw_union_authority,
        )
        if any(type(value) is not bool or value is not True for value in required):
            _refuse(
                V1BlockSupportCompleteUnionProductionRefusalCode.AUTHORITY_VIOLATION,
                "V4 raw union lost a mandatory production invariant",
            )
        _closed(
            (
                self.d_plane_deficit_debited,
                self.residual_plane_deficit_debited,
                self.canonical_block_observation_authority,
                self.population_separation_authority,
                self.full_u_authority,
                self.fqr_authority,
                self.c_authority,
                self.k_authority,
                self.o_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="production complete raw union v4",
        )
        if (
            self.authority_scope != COMPLETE_RAW_V4_AUTHORITY
            or self.schema != COMPLETE_RAW_V4_SCHEMA
        ):
            _refuse(
                V1BlockSupportCompleteUnionProductionRefusalCode.AUTHORITY_VIOLATION,
                "unsupported production complete-union-v4 semantics",
            )
        observed = content_hash(_raw_payload(self), prefix=_PREFIX + "raw-integrity-")
        if self._integrity_seal and self._integrity_seal != observed:
            _refuse(
                V1BlockSupportCompleteUnionProductionRefusalCode.RECEIPT_TAMPERING,
                "production complete-union-v4 receipt was mutated",
            )
        object.__setattr__(self, "_integrity_seal", observed)
        object.__setattr__(
            self,
            "receipt_hash",
            content_hash(_raw_payload(self), prefix=_PREFIX + "raw-"),
        )

    def assert_integrity(self) -> str:
        observed = content_hash(_raw_payload(self), prefix=_PREFIX + "raw-integrity-")
        expected = content_hash(_raw_payload(self), prefix=_PREFIX + "raw-")
        if observed != self._integrity_seal or expected != self.receipt_hash:
            _refuse(
                V1BlockSupportCompleteUnionProductionRefusalCode.RECEIPT_TAMPERING,
                "production complete-union-v4 receipt no longer replays",
            )
        return expected


def _validate_source_join(
    *,
    bound_witness_source: V1BlockSupportBoundWitnessSource,
    canonical_context_identity: V1BlockSupportCanonicalBlockContextIdentity,
    physical_binding: V1SignedWorkPhysicalAlternateBlockSupportBindingReceipt,
    d_fixed_source: V1SignedWorkBlockSupportSourceReceipt,
    residual_source: V1SignedWorkBlockSupportSourceReceipt,
    within_shell_execution: V1BlockSupportWithinShellCUDAExecutionReceipt,
    p_excluded_within: V1BlockSupportPExcludedWithinShellRawSlopeReceipt,
) -> object:
    try:
        bound_witness_source.assert_live()
        canonical_context_identity.assert_integrity()
        physical_binding.assert_integrity()
        d_fixed_source.assert_integrity()
        residual_source.assert_integrity()
        within_shell_execution.assert_integrity()
        p_excluded_within.assert_integrity()
    except (RuntimeError, PermissionError, TypeError, ValueError) as error:
        _refuse(
            V1BlockSupportCompleteUnionProductionRefusalCode.STALE_SOURCE,
            f"one production complete-cover parent is stale: {error}",
        )
    pre = bound_witness_source.preoutcome_domain.receipt
    structural = bound_witness_source.preoutcome_domain.structural_receipt
    bound = bound_witness_source.receipt
    if (
        canonical_context_identity.bound_registration_receipt_hash != bound.receipt_hash
        or canonical_context_identity.preoutcome_domain_receipt_hash != pre.receipt_hash
        or canonical_context_identity.block_index != p_excluded_within.block_index
        or canonical_context_identity.sealed_source_block_id
        != p_excluded_within.block_id
        or p_excluded_within.block_context_axis_hash
        != _within_shell_context_axis_hash(canonical_context_identity.context_ids)
        or p_excluded_within.block_context_count
        != len(canonical_context_identity.context_ids)
        or p_excluded_within.block_support_registration_hash
        != bound_witness_source.registration.registration_hash
        or residual_source.context_axis_hash
        != canonical_context_identity.block_support_context_axis_hash
        or residual_source.block_ordinal != canonical_context_identity.block_index
    ):
        _refuse(
            V1BlockSupportCompleteUnionProductionRefusalCode.CONTEXT_MISMATCH,
            "canonical context, P-excluded block, and residual source differ",
        )
    expected_cell_axis = tuple(
        zip(
            within_shell_execution.action_cell_hash_axis,
            within_shell_execution.cell_receipt_hash_axis,
            strict=True,
        )
    )
    if (
        within_shell_execution.production_CUDA_execution is not True
        or within_shell_execution.CPU_reference_test_mode is not False
        or within_shell_execution.device_type != "cuda"
        or within_shell_execution.complete_registered_within_shell_family_solved
        is not True
        or within_shell_execution.solved_cell_count
        != within_shell_execution.registered_cell_count
        or within_shell_execution.solved_cell_count != len(expected_cell_axis)
        or p_excluded_within.within_shell_execution_receipt_hash
        != within_shell_execution.receipt_hash
        or p_excluded_within.complete_cell_receipt_axis != expected_cell_axis
        or p_excluded_within.registered_cell_count
        != within_shell_execution.registered_cell_count
        or p_excluded_within.registry_receipt_hash
        != within_shell_execution.registry_receipt_hash
        or p_excluded_within.block_support_registration_hash
        != within_shell_execution.block_support_registration_hash
        or p_excluded_within.block_index != within_shell_execution.block_index
        or within_shell_execution.block_index != canonical_context_identity.block_index
        or within_shell_execution.variable_shell_source_registration_hash
        != pre.source_registration_hash
        or within_shell_execution.d_fixed_support_receipt_hash
        != d_fixed_source.receipt_hash
        or within_shell_execution.residual_support_receipt_hash
        != residual_source.receipt_hash
        or within_shell_execution.preferred_action_binding_hash
        != pre.preferred_action_binding_hash
        or within_shell_execution.preferred_action_hash != pre.preferred_action_hash
        or within_shell_execution.route_count != pre.route_count
    ):
        _refuse(
            V1BlockSupportCompleteUnionProductionRefusalCode.CUDA_CUSTODY_REQUIRED,
            "P-excluded within-shell parent is not the exact complete production CUDA execution",
        )
    if (
        d_fixed_source.source_role is not V1SignedWorkBlockSupportSourceRole.D_FIXED
        or residual_source.source_role
        is not V1SignedWorkBlockSupportSourceRole.RESIDUAL_BLOCK
        or physical_binding.d_fixed_support_receipt_hash != d_fixed_source.receipt_hash
        or physical_binding.residual_support_receipt_hash
        != residual_source.receipt_hash
        or physical_binding.source_registration_hash != pre.source_registration_hash
        or d_fixed_source.source_registration_hash != pre.source_registration_hash
        or residual_source.source_registration_hash != pre.source_registration_hash
        or physical_binding.route_count != pre.route_count
        or d_fixed_source.route_count != pre.route_count
        or residual_source.route_count != pre.route_count
        or p_excluded_within.physical_block_support_binding_receipt_hash
        != physical_binding.receipt_hash
        or p_excluded_within.d_fixed_support_receipt_hash != d_fixed_source.receipt_hash
        or p_excluded_within.residual_support_receipt_hash
        != residual_source.receipt_hash
        or p_excluded_within.structural_domain_receipt_hash != structural.receipt_hash
        or physical_binding.structural_domain_receipt_hash != structural.receipt_hash
    ):
        _refuse(
            V1BlockSupportCompleteUnionProductionRefusalCode.SOURCE_MISMATCH,
            "physical, D-fixed, residual, structural, and P receipts do not join",
        )
    common = (
        "model_identity_hash",
        "chart_identity_hash",
        "current_mask_hash",
        "proposal_generation",
        "route_axis_hash",
        "route_id_axis_hash",
        "native_fiber_axis_receipt_hash",
        "native_fiber_measure_hash",
        "registered_d_prefix_hash",
        "registered_d_w2_prefix_hash",
    )
    if any(
        getattr(physical_binding, name) != getattr(pre, name)
        or getattr(d_fixed_source, name) != getattr(pre, name)
        or getattr(residual_source, name) != getattr(pre, name)
        for name in common
    ):
        _refuse(
            V1BlockSupportCompleteUnionProductionRefusalCode.SOURCE_MISMATCH,
            "model/chart/mask/route/native-fiber/D-prefix identities differ",
        )
    if (
        p_excluded_within.preferred_action_hash != pre.preferred_action_hash
        or p_excluded_within.preferred_action_binding_hash
        != pre.preferred_action_binding_hash
        or physical_binding.preferred_action_hash != pre.preferred_action_hash
        or physical_binding.preferred_action_binding_hash
        != pre.preferred_action_binding_hash
        or physical_binding.preferred_resulting_mask_hash
        != pre.preferred_resulting_mask_hash
        or physical_binding.preferred_removed_route_index_hash
        != pre.preferred_removed_route_index_hash
        or physical_binding.preferred_selected_count != pre.preferred_selected_count
    ):
        _refuse(
            V1BlockSupportCompleteUnionProductionRefusalCode.P_MISMATCH,
            "preferred P custody differs across production cover parents",
        )
    if (
        pre.production_cuda_source_bound is not True
        or pre.private_reference_test_only is not False
        or d_fixed_source.execution_mode
        is not V1SignedWorkBlockSupportExecutionMode.CUDA_PRODUCTION
        or residual_source.execution_mode
        is not V1SignedWorkBlockSupportExecutionMode.CUDA_PRODUCTION
        or physical_binding.resident_device_identity
        != f"cuda:{within_shell_execution.device_index}"
    ):
        _refuse(
            V1BlockSupportCompleteUnionProductionRefusalCode.CUDA_CUSTODY_REQUIRED,
            "production complete cover requires exact CUDA parent custody",
        )
    return pre


def _assert_live_receipt_integrity(
    *,
    alias_ordinal: int,
    alias: V1BlockSupportAliasCoverBindingV2,
    cover: V1BlockSupportOutsidePairLiveCoverAuthorityReceiptV2,
    family: V1BlockSupportOutsidePairLiveFamilyReceiptV3,
) -> None:
    try:
        alias.assert_integrity()
        cover.assert_integrity()
        family.assert_integrity()
    except (RuntimeError, PermissionError, TypeError, ValueError) as error:
        _refuse(
            V1BlockSupportCompleteUnionProductionRefusalCode.STALE_SOURCE,
            f"production live triple {alias_ordinal} is stale: {error}",
        )


def _validate_family_rectangle_result(
    *,
    alias_ordinal: int,
    family: V1BlockSupportOutsidePairLiveFamilyReceiptV3,
) -> None:
    """Replay sealed-empty values and the source family's deterministic winner."""

    rectangles = (
        (
            V1SignedWorkVariableShellOutsidePairRectangle.ADD_OUTSIDE,
            0,
            family.add_outside_rectangle_pair_count,
            family.add_outside_rectangle_empty,
            _real(
                family.add_outside_pair_lower_outward,
                name=f"live family {alias_ordinal} add-outside lower",
            ),
        ),
        (
            V1SignedWorkVariableShellOutsidePairRectangle.DROP_OUTSIDE_ADD_INSIDE,
            1,
            family.drop_outside_add_inside_rectangle_pair_count,
            family.drop_outside_add_inside_rectangle_empty,
            _real(
                family.drop_outside_add_inside_pair_lower_outward,
                name=f"live family {alias_ordinal} drop/add lower",
            ),
        ),
    )
    if any(type(empty) is not bool for _, _, _, empty, _ in rectangles):
        _refuse(
            V1BlockSupportCompleteUnionProductionRefusalCode.INVALID_TYPE,
            f"live family {alias_ordinal} rectangle seals must be exact booleans",
        )
    for rectangle, _order, pair_count, empty, lower in rectangles:
        if empty is not (pair_count == 0):
            _refuse(
                V1BlockSupportCompleteUnionProductionRefusalCode.RECTANGLE_MISMATCH,
                f"live family {alias_ordinal}/{rectangle.value} has an inconsistent empty seal",
            )
        if empty:
            if lower != math.inf:
                _refuse(
                    V1BlockSupportCompleteUnionProductionRefusalCode.RECTANGLE_MISMATCH,
                    f"sealed empty family rectangle {alias_ordinal}/{rectangle.value} is not +inf",
                )
        elif not math.isfinite(lower):
            _refuse(
                V1BlockSupportCompleteUnionProductionRefusalCode.NUMERICAL_REFUSAL,
                f"nonempty family rectangle {alias_ordinal}/{rectangle.value} is not finite",
            )

    optional_coordinates = (
        family.worst_pair_preferred_selected_index,
        family.worst_pair_preferred_unselected_index,
        family.worst_pair_ordinal,
    )
    worst_lower = _real(
        family.worst_pair_lower_outward,
        name=f"live family {alias_ordinal} worst-pair lower",
    )
    nonempty = tuple(value for value in rectangles if not value[3])
    if not nonempty:
        if (
            family.vacuous_no_outside_pair is not True
            or family.worst_pair_rectangle is not None
            or any(value is not None for value in optional_coordinates)
            or worst_lower != math.inf
        ):
            _refuse(
                V1BlockSupportCompleteUnionProductionRefusalCode.RECTANGLE_MISMATCH,
                f"vacuous live family {alias_ordinal} carries a fake worst pair",
            )
        return

    expected = min(nonempty, key=lambda value: (value[4], value[1]))
    if (
        family.vacuous_no_outside_pair is not False
        or family.worst_pair_rectangle is not expected[0]
        or worst_lower != expected[4]
        or any(type(value) is not int or value < 0 for value in optional_coordinates)
    ):
        _refuse(
            V1BlockSupportCompleteUnionProductionRefusalCode.RECTANGLE_MISMATCH,
            f"live family {alias_ordinal} worst pair does not replay the rectangle minimum",
        )
    selected, unselected, pair_ordinal = optional_coordinates
    assert selected is not None and unselected is not None and pair_ordinal is not None
    first_count = family.add_outside_rectangle_pair_count
    ordinal_start = 0 if expected[1] == 0 else first_count
    ordinal_stop = ordinal_start + expected[2]
    if (
        selected >= family.route_count
        or unselected >= family.route_count
        or selected == unselected
        or not ordinal_start <= pair_ordinal < ordinal_stop
    ):
        _refuse(
            V1BlockSupportCompleteUnionProductionRefusalCode.PAIR_AXIS_MISMATCH,
            f"live family {alias_ordinal} worst-pair coordinates leave their bound pair axis",
        )


def bind_v1_block_support_complete_constraint_cover_production_v3(
    *,
    bound_witness_source: V1BlockSupportBoundWitnessSource,
    canonical_context_identity: V1BlockSupportCanonicalBlockContextIdentity,
    physical_binding: V1SignedWorkPhysicalAlternateBlockSupportBindingReceipt,
    d_fixed_source: V1SignedWorkBlockSupportSourceReceipt,
    residual_source: V1SignedWorkBlockSupportSourceReceipt,
    within_shell_execution: V1BlockSupportWithinShellCUDAExecutionReceipt,
    p_excluded_within: V1BlockSupportPExcludedWithinShellRawSlopeReceipt,
    alias_bindings: Sequence[V1BlockSupportAliasCoverBindingV2],
    outside_live_cover_authorities: Sequence[
        V1BlockSupportOutsidePairLiveCoverAuthorityReceiptV2
    ],
    outside_live_families: Sequence[V1BlockSupportOutsidePairLiveFamilyReceiptV3],
) -> V1BlockSupportCompleteConstraintCoverBindingReceiptV3:
    """Bind every registered alias to exact production live CUDA custody."""

    exact_types = (
        type(bound_witness_source) is V1BlockSupportBoundWitnessSource,
        type(canonical_context_identity) is V1BlockSupportCanonicalBlockContextIdentity,
        type(physical_binding)
        is V1SignedWorkPhysicalAlternateBlockSupportBindingReceipt,
        type(d_fixed_source) is V1SignedWorkBlockSupportSourceReceipt,
        type(residual_source) is V1SignedWorkBlockSupportSourceReceipt,
        type(within_shell_execution) is V1BlockSupportWithinShellCUDAExecutionReceipt,
        type(p_excluded_within) is V1BlockSupportPExcludedWithinShellRawSlopeReceipt,
    )
    if not all(exact_types):
        raise TypeError("production complete-cover-v3 requires exact parent types")
    aliases = tuple(alias_bindings)
    covers = tuple(outside_live_cover_authorities)
    families = tuple(outside_live_families)
    if (
        any(type(value) is not V1BlockSupportAliasCoverBindingV2 for value in aliases)
        or any(
            type(value) is not V1BlockSupportOutsidePairLiveCoverAuthorityReceiptV2
            for value in covers
        )
        or any(
            type(value) is not V1BlockSupportOutsidePairLiveFamilyReceiptV3
            for value in families
        )
    ):
        raise TypeError(
            "production complete-cover-v3 axes require exact alias/live-cover/live-family types"
        )
    pre = _validate_source_join(
        bound_witness_source=bound_witness_source,
        canonical_context_identity=canonical_context_identity,
        physical_binding=physical_binding,
        d_fixed_source=d_fixed_source,
        residual_source=residual_source,
        within_shell_execution=within_shell_execution,
        p_excluded_within=p_excluded_within,
    )
    expected_aliases = bound_witness_source.preoutcome_domain.alias_bindings
    if (
        aliases != expected_aliases
        or not aliases
        or len({value.binding_hash for value in aliases}) != len(aliases)
    ):
        _refuse(
            V1BlockSupportCompleteUnionProductionRefusalCode.ALIAS_AXIS_MISMATCH,
            "aliases must equal the complete registered axis in exact order",
        )
    if len(covers) != len(aliases) or len(families) != len(aliases):
        _refuse(
            V1BlockSupportCompleteUnionProductionRefusalCode.ALIAS_AXIS_MISMATCH,
            "every registered alias requires one live cover and one live family",
        )
    cardinality_axis = []
    source_axis = []
    count_axis = []
    device_identity: str | None = None
    registration = bound_witness_source.registration
    for ordinal, (alias, cover, family) in enumerate(
        zip(aliases, covers, families, strict=True)
    ):
        _assert_live_receipt_integrity(
            alias_ordinal=ordinal,
            alias=alias,
            cover=cover,
            family=family,
        )
        if (
            cover.schema != OUTSIDE_PAIR_LIVE_COVER_AUTHORITY_V2_SCHEMA
            or cover.authority_scope != OUTSIDE_PAIR_LIVE_COVER_AUTHORITY_V2_SCOPE
            or cover.exact_bound_witness_source_join is not True
            or cover.exact_canonical_context_join is not True
            or cover.exact_v2_live_resource_alias_join is not True
            or cover.exact_physical_live_provider_join is not True
            or cover.exact_current_live_cover_join is not True
            or cover.all_four_rectangle_axes_bound is not True
            or cover.legacy_diagnostic_values_excluded is not True
            or cover.one_alias_raw_outside_family_authority is not True
            or cover.complete_union_authority is not False
            or cover.statistical_authority is not False
            or family.schema != OUTSIDE_PAIR_LIVE_FAMILY_V3_SCHEMA
            or family.equation != OUTSIDE_PAIR_LIVE_FAMILY_V3_EQUATION
            or family.authority_scope != OUTSIDE_PAIR_LIVE_FAMILY_V3_AUTHORITY
            or family.production_cuda_execution is not True
            or family.exact_live_cover_authority_v2_join is not True
            or family.exact_problem_final_KKT_dual_v2_join is not True
            or family.complete_two_rectangle_pair_cover_scanned is not True
            or family.complete_route_scan is not True
            or family.legacy_pair_cover_authority_used is not False
            or family.legacy_live_cover_receipt_hash_authority_used is not False
            or family.caller_source_hash_authority_used is not False
            or family.caller_raw_operator_or_multiplier_used is not False
            or family.one_alias_raw_outside_family_authority is not True
            or family.complete_union_authority is not False
            or family.statistical_authority is not False
            or cover.d_weight_numerator != registration.d_weight_numerator
            or cover.residual_weight_numerator != registration.residual_weight_numerator
            or cover.population_weight_denominator
            != registration.population_weight_denominator
            or not cover.resident_device_identity.startswith("cuda")
            or family.resident_device_identity != cover.resident_device_identity
            or cover.resident_device_identity
            != physical_binding.resident_device_identity
        ):
            _refuse(
                V1BlockSupportCompleteUnionProductionRefusalCode.CUDA_CUSTODY_REQUIRED,
                f"live triple {ordinal} lacks exact source-native CUDA custody",
            )
        if device_identity is None:
            device_identity = cover.resident_device_identity
        elif cover.resident_device_identity != device_identity:
            _refuse(
                V1BlockSupportCompleteUnionProductionRefusalCode.CUDA_CUSTODY_REQUIRED,
                "registered live covers do not share one CUDA device identity",
            )
        if (
            cover.resource_alias_hash != alias.resource_alias_hash
            or family.resource_alias_hash != cover.resource_alias_hash
            or cover.branch_id != alias.branch.value
            or family.branch_id != cover.branch_id
            or cover.registered_fd_rank != alias.registered_fd_rank
            or family.registered_fd_rank != cover.registered_fd_rank
            or cover.resource_alias_id != alias.resource_alias_id
            or family.resource_alias_id != cover.resource_alias_id
            or cover.cardinality_alias_id != alias.cardinality_alias_id
            or family.cardinality_alias_id != cover.cardinality_alias_id
            or cover.phase_a_source_receipt_hash != alias.source_receipt_hash
            or family.phase_a_source_receipt_hash != cover.phase_a_source_receipt_hash
            or cover.phase_a_source_replay_hash != alias.source_replay_hash
            or family.phase_a_source_replay_hash != cover.phase_a_source_replay_hash
            or cover.source_resource_receipt_hash != alias.source_resource_receipt_hash
            or family.source_resource_receipt_hash != cover.source_resource_receipt_hash
            or cover.resource_coordinate_hash != alias.resource_coordinate_hash
            or family.resource_coordinate_hash != cover.resource_coordinate_hash
            or cover.phase_a_resource_cell_id != alias.phase_a_resource_cell_id
            or family.phase_a_resource_cell_id != cover.phase_a_resource_cell_id
            or cover.phase_a_invariance_receipt_hash
            != alias.phase_a_invariance_receipt_hash
            or family.phase_a_invariance_receipt_hash
            != cover.phase_a_invariance_receipt_hash
            or cover.fixed_outside_removed_count != alias.fixed_outside_removed_count
            or family.fixed_outside_removed_count != cover.fixed_outside_removed_count
            or cover.shell_top_k_count != alias.shell_top_k_count
            or family.shell_top_k_count != cover.shell_top_k_count
            or cover.hard_total_removed_count != alias.hard_total_removed_count
            or family.hard_total_removed_count != cover.hard_total_removed_count
            or cover.registered_shell_resource_floor
            != alias.live_resource_alias.registered_shell_resource_floor
        ):
            _refuse(
                V1BlockSupportCompleteUnionProductionRefusalCode.ALIAS_AXIS_MISMATCH,
                f"live triple {ordinal} differs from its registered alias",
            )
        if (
            cover.bound_witness_registration_receipt_hash
            != bound_witness_source.receipt.receipt_hash
            or family.bound_witness_registration_receipt_hash
            != cover.bound_witness_registration_receipt_hash
            or cover.preoutcome_domain_receipt_hash != pre.receipt_hash
            or family.preoutcome_domain_receipt_hash
            != cover.preoutcome_domain_receipt_hash
            or cover.canonical_block_context_receipt_hash
            != canonical_context_identity.receipt_hash
            or family.canonical_block_context_receipt_hash
            != cover.canonical_block_context_receipt_hash
            or cover.block_index != canonical_context_identity.block_index
            or family.block_index != cover.block_index
            or cover.block_id != canonical_context_identity.sealed_source_block_id
            or family.block_id != cover.block_id
            or cover.canonical_context_tuple_hash
            != canonical_context_identity.canonical_context_tuple_hash
            or family.canonical_context_tuple_hash != cover.canonical_context_tuple_hash
            or cover.block_support_context_axis_hash
            != canonical_context_identity.block_support_context_axis_hash
            or family.block_support_context_axis_hash
            != cover.block_support_context_axis_hash
        ):
            _refuse(
                V1BlockSupportCompleteUnionProductionRefusalCode.CONTEXT_MISMATCH,
                f"live triple {ordinal} differs from its canonical block context",
            )
        if (
            cover.physical_block_support_binding_receipt_hash
            != physical_binding.receipt_hash
            or family.physical_block_support_binding_receipt_hash
            != physical_binding.receipt_hash
            or cover.structural_domain_receipt_hash
            != physical_binding.structural_domain_receipt_hash
            or family.structural_domain_receipt_hash
            != physical_binding.structural_domain_receipt_hash
            or family.d_fixed_support_receipt_hash != d_fixed_source.receipt_hash
            or family.residual_support_receipt_hash != residual_source.receipt_hash
            or cover.source_registration_hash != pre.source_registration_hash
            or family.source_registration_hash != pre.source_registration_hash
        ):
            _refuse(
                V1BlockSupportCompleteUnionProductionRefusalCode.SOURCE_MISMATCH,
                f"live triple {ordinal} differs from physical/support sources",
            )
        common = (
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "proposal_generation",
            "route_axis_hash",
            "route_id_axis_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
        )
        if any(
            getattr(cover, name) != getattr(pre, name)
            or getattr(family, name) != getattr(pre, name)
            for name in common
        ):
            _refuse(
                V1BlockSupportCompleteUnionProductionRefusalCode.SOURCE_MISMATCH,
                f"live triple {ordinal} changed model/chart/mask/generation/route/prefix identity",
            )
        if (
            cover.preferred_action_binding_hash != pre.preferred_action_binding_hash
            or cover.preferred_action_hash != pre.preferred_action_hash
            or cover.preferred_removed_route_index_hash
            != pre.preferred_removed_route_index_hash
            or cover.preferred_selected_count != pre.preferred_selected_count
            or family.preferred_action_binding_hash != pre.preferred_action_binding_hash
            or family.preferred_action_hash != pre.preferred_action_hash
            or family.preferred_removed_route_index_hash
            != pre.preferred_removed_route_index_hash
            or family.preferred_selected_count != pre.preferred_selected_count
        ):
            _refuse(
                V1BlockSupportCompleteUnionProductionRefusalCode.P_MISMATCH,
                f"live triple {ordinal} changed global comparison P",
            )
        cover_family_pair_exact = (
            "live_cover_source_native_payload_hash",
            "source_native_alias_commitment_hash",
            "source_native_pair_axis_commitment_hash",
            "fixed_membership_binding_hash",
            "parent_live_membership_identity_hash",
            "selectable_membership_identity_hash",
            "add_outside_drop_axis_identity_hash",
            "add_outside_add_axis_identity_hash",
            "drop_outside_add_inside_drop_axis_identity_hash",
            "drop_outside_add_inside_add_axis_identity_hash",
            "rounding_anchor_action_coordinate_hash",
            "rounding_anchor_removed_route_index_hash",
            "rounding_anchor_subset_identity_hash",
        )
        if family.live_cover_authority_receipt_hash != cover.receipt_hash:
            _refuse(
                V1BlockSupportCompleteUnionProductionRefusalCode.SOURCE_MISMATCH,
                f"live family {ordinal} names another cover-authority parent",
            )
        if any(
            getattr(family, name) != getattr(cover, name)
            for name in (
                "kkt_state_identity_hash",
                "kkt_problem_hash",
                "kkt_policy_hash",
                "kkt_branch_receipt_hash",
                "kkt_eligible_mask_identity_hash",
                "kkt_required_mask_identity_hash",
            )
        ):
            _refuse(
                V1BlockSupportCompleteUnionProductionRefusalCode.KKT_MISMATCH,
                f"live family {ordinal} differs from its cover/KKT source axis",
            )
        if any(
            getattr(family, name) != getattr(cover, name)
            for name in cover_family_pair_exact
        ):
            _refuse(
                V1BlockSupportCompleteUnionProductionRefusalCode.PAIR_AXIS_MISMATCH,
                f"live family {ordinal} differs from its source-native pair axis",
            )
        if (
            cover.route_count != pre.route_count
            or family.route_count != pre.route_count
            or cover.complete_route_scan_count != pre.route_count
            or family.live_cover_complete_route_scan_count != pre.route_count
            or family.complete_route_coefficient_scan_count != pre.route_count
            or family.add_outside_droppable_count != cover.add_outside_droppable_count
            or family.add_outside_addable_count != cover.add_outside_addable_count
            or family.drop_outside_add_inside_droppable_count
            != cover.drop_outside_add_inside_droppable_count
            or family.drop_outside_add_inside_addable_count
            != cover.drop_outside_add_inside_addable_count
            or family.add_outside_rectangle_pair_count
            != cover.add_outside_rectangle_pair_count
            or family.drop_outside_add_inside_rectangle_pair_count
            != cover.drop_outside_add_inside_rectangle_pair_count
            or family.complete_pair_cover_count != cover.complete_pair_count
            or cover.complete_pair_count
            != cover.add_outside_rectangle_pair_count
            + cover.drop_outside_add_inside_rectangle_pair_count
            or family.add_outside_rectangle_empty
            is not (cover.add_outside_rectangle_pair_count == 0)
            or family.drop_outside_add_inside_rectangle_empty
            is not (cover.drop_outside_add_inside_rectangle_pair_count == 0)
            or family.vacuous_no_outside_pair is not (cover.complete_pair_count == 0)
        ):
            _refuse(
                V1BlockSupportCompleteUnionProductionRefusalCode.RECTANGLE_MISMATCH,
                f"live family {ordinal} does not replay both cover rectangles",
            )
        _validate_family_rectangle_result(
            alias_ordinal=ordinal,
            family=family,
        )
        cardinality_axis.append(
            (
                alias.branch.value,
                alias.resource_alias_id,
                alias.cardinality_alias_id,
                alias.registered_fd_rank,
                alias.fixed_outside_removed_count,
                alias.shell_top_k_count,
                alias.hard_total_removed_count,
            )
        )
        source_axis.append(
            (
                cover.phase_a_source_receipt_hash,
                cover.phase_a_source_replay_hash,
                cover.live_cover_source_native_payload_hash,
                cover.source_native_alias_commitment_hash,
                cover.source_native_pair_axis_commitment_hash,
                cover.kkt_state_identity_hash,
                cover.kkt_problem_hash,
                cover.kkt_policy_hash,
                cover.kkt_branch_receipt_hash,
                cover.kkt_eligible_mask_identity_hash,
                cover.kkt_required_mask_identity_hash,
                family.production_dual_source_receipt_hash,
                family.least_work_problem_hash,
                family.source_problem_hash,
                family.signed_work_operator_receipt_hash,
                family.resource_axis_hash,
                family.capacity_operator_source_hash,
                family.resource_multiplier_source_hash,
                family.capacity_multiplier_source_hash,
                family.operator_scales_hash,
            )
        )
        count_axis.append(
            (
                cover.route_count,
                cover.complete_route_scan_count,
                cover.add_outside_droppable_count,
                cover.add_outside_addable_count,
                cover.drop_outside_add_inside_droppable_count,
                cover.drop_outside_add_inside_addable_count,
                cover.add_outside_rectangle_pair_count,
                cover.drop_outside_add_inside_rectangle_pair_count,
                cover.complete_pair_count,
            )
        )
    assert device_identity is not None
    return V1BlockSupportCompleteConstraintCoverBindingReceiptV3(
        bound_registration_receipt_hash=bound_witness_source.receipt.receipt_hash,
        preoutcome_domain_receipt_hash=pre.receipt_hash,
        canonical_block_context_receipt_hash=canonical_context_identity.receipt_hash,
        canonical_context_tuple_hash=(
            canonical_context_identity.canonical_context_tuple_hash
        ),
        block_support_context_axis_hash=(
            canonical_context_identity.block_support_context_axis_hash
        ),
        block_index=canonical_context_identity.block_index,
        block_id=canonical_context_identity.sealed_source_block_id,
        structural_domain_receipt_hash=physical_binding.structural_domain_receipt_hash,
        physical_block_support_binding_receipt_hash=physical_binding.receipt_hash,
        d_fixed_support_receipt_hash=d_fixed_source.receipt_hash,
        residual_support_receipt_hash=residual_source.receipt_hash,
        within_shell_execution_receipt_hash=within_shell_execution.receipt_hash,
        p_excluded_within_receipt_hash=p_excluded_within.receipt_hash,
        preferred_action_hash=pre.preferred_action_hash,
        preferred_action_binding_hash=pre.preferred_action_binding_hash,
        preferred_removed_route_index_hash=pre.preferred_removed_route_index_hash,
        preferred_selected_count=pre.preferred_selected_count,
        model_identity_hash=pre.model_identity_hash,
        chart_identity_hash=pre.chart_identity_hash,
        current_mask_hash=pre.current_mask_hash,
        proposal_generation=pre.proposal_generation,
        route_axis_hash=pre.route_axis_hash,
        route_id_axis_hash=pre.route_id_axis_hash,
        route_count=pre.route_count,
        resident_device_identity=device_identity,
        native_fiber_axis_receipt_hash=pre.native_fiber_axis_receipt_hash,
        native_fiber_measure_hash=pre.native_fiber_measure_hash,
        registered_d_prefix_hash=pre.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=pre.registered_d_w2_prefix_hash,
        d_weight_numerator=registration.d_weight_numerator,
        residual_weight_numerator=registration.residual_weight_numerator,
        population_weight_denominator=registration.population_weight_denominator,
        ordered_alias_binding_hashes=tuple(value.binding_hash for value in aliases),
        ordered_resource_alias_hashes=tuple(
            value.resource_alias_hash for value in covers
        ),
        ordered_live_cover_authority_receipt_hashes=tuple(
            value.receipt_hash for value in covers
        ),
        ordered_live_family_receipt_hashes=tuple(
            value.receipt_hash for value in families
        ),
        ordered_alias_cardinality_axis=tuple(cardinality_axis),
        ordered_source_pair_kkt_axis=tuple(source_axis),
        ordered_rectangle_count_axis=tuple(count_axis),
        _issuer_token=_COVER_TOKEN,
    )


def join_v1_block_support_complete_union_raw_lower_production_v4(
    *,
    complete_cover: V1BlockSupportCompleteConstraintCoverBindingReceiptV3,
    p_excluded_within: V1BlockSupportPExcludedWithinShellRawSlopeReceipt,
    outside_live_families: Sequence[V1BlockSupportOutsidePairLiveFamilyReceiptV3],
) -> V1BlockSupportCompleteUnionRawSlopeReceiptV4:
    """Take the deterministic raw minimum without debiting plane deficits."""

    if (
        type(complete_cover)
        is not V1BlockSupportCompleteConstraintCoverBindingReceiptV3
        or type(p_excluded_within)
        is not V1BlockSupportPExcludedWithinShellRawSlopeReceipt
    ):
        raise TypeError("production complete-union-v4 requires exact V3/P types")
    families = tuple(outside_live_families)
    if any(
        type(value) is not V1BlockSupportOutsidePairLiveFamilyReceiptV3
        for value in families
    ):
        raise TypeError("production complete-union-v4 requires exact live families")
    try:
        complete_cover.assert_integrity()
        p_excluded_within.assert_integrity()
        for family in families:
            family.assert_integrity()
    except (RuntimeError, PermissionError, TypeError, ValueError) as error:
        _refuse(
            V1BlockSupportCompleteUnionProductionRefusalCode.STALE_SOURCE,
            f"one production complete-union-v4 parent is stale: {error}",
        )
    if (
        complete_cover.complete_constraint_cover_source_bound is not True
        or complete_cover.production_cuda_custody is not True
        or complete_cover.canonical_block_observation_authority is not False
    ):
        _refuse(
            V1BlockSupportCompleteUnionProductionRefusalCode.AUTHORITY_VIOLATION,
            "V4 requires the exact source-bound production V3 cover",
        )
    if complete_cover.p_excluded_within_receipt_hash != p_excluded_within.receipt_hash:
        _refuse(
            V1BlockSupportCompleteUnionProductionRefusalCode.P_MISMATCH,
            "V4 received another P-excluded within receipt",
        )
    family_hashes = tuple(value.receipt_hash for value in families)
    if family_hashes != complete_cover.ordered_live_family_receipt_hashes:
        _refuse(
            V1BlockSupportCompleteUnionProductionRefusalCode.ALIAS_AXIS_MISMATCH,
            "V4 live-family axis differs from its ordered production V3 cover",
        )
    p_lower = _real(
        p_excluded_within.p_excluded_within_shell_raw_lower,
        name="P-excluded within raw lower",
    )
    if not math.isfinite(p_lower):
        _refuse(
            V1BlockSupportCompleteUnionProductionRefusalCode.NUMERICAL_REFUSAL,
            "P-excluded within raw lower must be finite",
        )
    candidates: list[tuple[str, int, str, bool, float, tuple[int, ...]]] = [
        (
            "P_excluded_within",
            -1,
            p_excluded_within.receipt_hash,
            False,
            p_lower,
            (),
        )
    ]
    finite: list[tuple[float, int, int, tuple[int, ...], str, str]] = [
        (
            p_lower,
            -1,
            0,
            (),
            "P_excluded_within",
            p_excluded_within.receipt_hash,
        )
    ]
    empty_count = 0
    for ordinal, (alias_hash, family) in enumerate(
        zip(
            complete_cover.ordered_alias_binding_hashes,
            families,
            strict=True,
        )
    ):
        rectangles = (
            (
                "add_outside",
                V1SignedWorkVariableShellOutsidePairRectangle.ADD_OUTSIDE,
                family.add_outside_rectangle_empty,
                family.add_outside_rectangle_pair_count,
                family.add_outside_pair_lower_outward,
            ),
            (
                "drop_outside_add_inside",
                V1SignedWorkVariableShellOutsidePairRectangle.DROP_OUTSIDE_ADD_INSIDE,
                family.drop_outside_add_inside_rectangle_empty,
                family.drop_outside_add_inside_rectangle_pair_count,
                family.drop_outside_add_inside_pair_lower_outward,
            ),
        )
        for rectangle_order, (
            domain,
            rectangle,
            empty,
            pair_count,
            value,
        ) in enumerate(rectangles, start=1):
            numeric = _real(value, name=f"live rectangle {ordinal}/{domain}")
            if empty is not (pair_count == 0):
                _refuse(
                    V1BlockSupportCompleteUnionProductionRefusalCode.RECTANGLE_MISMATCH,
                    f"live rectangle {ordinal}/{domain} has an inconsistent empty seal",
                )
            physical_indices: tuple[int, ...] = ()
            if empty:
                if numeric != math.inf:
                    _refuse(
                        V1BlockSupportCompleteUnionProductionRefusalCode.RECTANGLE_MISMATCH,
                        f"sealed empty rectangle {ordinal}/{domain} is not +inf",
                    )
                empty_count += 1
            else:
                if not math.isfinite(numeric):
                    _refuse(
                        V1BlockSupportCompleteUnionProductionRefusalCode.NUMERICAL_REFUSAL,
                        f"nonempty rectangle {ordinal}/{domain} lacks a finite lower",
                    )
                if family.worst_pair_rectangle is rectangle:
                    if (
                        family.worst_pair_preferred_selected_index is None
                        or family.worst_pair_preferred_unselected_index is None
                    ):
                        _refuse(
                            V1BlockSupportCompleteUnionProductionRefusalCode.RECTANGLE_MISMATCH,
                            f"winning family rectangle {ordinal}/{domain} lost its pair witness",
                        )
                    physical_indices = (
                        family.worst_pair_preferred_selected_index,
                        family.worst_pair_preferred_unselected_index,
                    )
                finite.append(
                    (
                        numeric,
                        ordinal,
                        rectangle_order,
                        physical_indices,
                        domain,
                        alias_hash,
                    )
                )
            candidates.append(
                (domain, ordinal, alias_hash, empty, numeric, physical_indices)
            )
    winner = min(
        finite,
        key=lambda value: (
            value[0],
            value[1],
            value[2],
            value[3],
            value[5],
        ),
    )
    if winner[1] >= 0 and len(winner[3]) != 2:
        _refuse(
            V1BlockSupportCompleteUnionProductionRefusalCode.RECTANGLE_MISMATCH,
            "winning outside rectangle is not its family's deterministic worst pair",
        )
    outward = math.nextafter(winner[0], -math.inf)
    if not math.isfinite(outward) or outward >= winner[0]:
        _refuse(
            V1BlockSupportCompleteUnionProductionRefusalCode.NUMERICAL_REFUSAL,
            "production complete-union exact one-ULP downward step failed",
        )
    return V1BlockSupportCompleteUnionRawSlopeReceiptV4(
        complete_cover_v3_receipt_hash=complete_cover.receipt_hash,
        p_excluded_within_receipt_hash=p_excluded_within.receipt_hash,
        ordered_alias_binding_hashes=complete_cover.ordered_alias_binding_hashes,
        ordered_live_cover_authority_receipt_hashes=(
            complete_cover.ordered_live_cover_authority_receipt_hashes
        ),
        ordered_live_family_receipt_hashes=family_hashes,
        candidate_axis=tuple(candidates),
        winning_domain=winner[4],
        winning_alias_ordinal=winner[1],
        winning_rectangle_ordinal=winner[2],
        winning_source_hash=winner[5],
        winning_physical_index_axis=winner[3],
        winning_raw_lower_input=winner[0],
        complete_union_raw_lower_outward=outward,
        finite_candidate_count=len(finite),
        empty_rectangle_count=empty_count,
        _issuer_token=_RAW_TOKEN,
    )


def require_v1_block_support_complete_union_v4_canonical_observation_raw_union_authority(
    receipt: V1BlockSupportCompleteUnionRawSlopeReceiptV4,
) -> V1BlockSupportCompleteUnionRawSlopeReceiptV4:
    """Open only the V4 raw-union input gate for a future observation issuer."""

    if type(receipt) is not V1BlockSupportCompleteUnionRawSlopeReceiptV4:
        raise TypeError("canonical raw-union gate requires the exact production V4")
    receipt.assert_integrity()
    if (
        receipt.production_cuda_custody is not True
        or receipt.canonical_block_observation_raw_union_authority is not True
        or receipt.canonical_block_observation_authority is not False
        or receipt.population_separation_authority is not False
        or receipt.d_plane_deficit_debited is not False
        or receipt.residual_plane_deficit_debited is not False
    ):
        _refuse(
            V1BlockSupportCompleteUnionProductionRefusalCode.AUTHORITY_VIOLATION,
            "V4 may authorize only an undepleted canonical-observation raw-union input",
        )
    return receipt


__all__ = (
    "COMPLETE_COVER_V3_AUTHORITY",
    "COMPLETE_COVER_V3_SCHEMA",
    "COMPLETE_RAW_V4_AUTHORITY",
    "COMPLETE_RAW_V4_SCHEMA",
    "V1BlockSupportCompleteConstraintCoverBindingReceiptV3",
    "V1BlockSupportCompleteUnionProductionRefusal",
    "V1BlockSupportCompleteUnionProductionRefusalCode",
    "V1BlockSupportCompleteUnionRawSlopeReceiptV4",
    "bind_v1_block_support_complete_constraint_cover_production_v3",
    "join_v1_block_support_complete_union_raw_lower_production_v4",
    "require_v1_block_support_complete_union_v4_canonical_observation_raw_union_authority",
)
