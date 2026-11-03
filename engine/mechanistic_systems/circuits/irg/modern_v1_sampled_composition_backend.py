"""Physical-rank sampled-composition backend joints.

This module is a narrow backend seam for adaptive-rank work.  It deliberately
does not own C certification, calibration, oracle access, or mask commits.  Its
authority stops at executor-side, source-bound replay:

* bind a rank-explicit FD design;
* bind a registered-D transaction that replays the identical raw prefix from
  look 1 under rank-local cache keys; and
* execute one physical rank through FD uncertainty, both KKT branches, and
  hard F/Q replay for every rounded rung.

The authority-inert adaptive loop below is an adapter over the existing
``modern_v1_fqr_adaptive_rank`` policy/stability/refusal authority.  It may
name those typed certificate hashes, checkpoint and evict physical geometry,
and stop at the first source-stable rank or typed no-C refusal; it does not
grant C/K/O/mask authority by itself.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Final, Literal

from .contracts import content_hash
from .modern_v1_fqr_adaptive_rank import (
    V1FQRAdaptiveRankNoCRefusal,
    V1FQRAdaptiveRankPolicy,
    V1FQRAdaptiveRankReplayLedger,
    V1FQRAdaptiveRankReplayRecord,
    V1FQRAdaptiveRankRefusal,
    V1FQRAdaptiveRankRefusalCode,
    V1FQRAdaptiveRankStabilityCertificate,
    certify_v1_fqr_adaptive_rank_stability,
)


SAMPLED_COMPOSITION_PHYSICAL_FD_DESIGN_SCHEMA: Final = (
    "partition_free_v1_sampled_composition_physical_fd_design_v1"
)
SAMPLED_COMPOSITION_REGISTERED_D_TRANSACTION_SCHEMA: Final = (
    "partition_free_v1_sampled_composition_registered_d_physical_rank_transaction_v1"
)
SAMPLED_COMPOSITION_REGISTERED_D_PREFIX_BINDING_SCHEMA: Final = (
    "partition_free_v1_sampled_composition_registered_d_prefix_binding_v1"
)
SAMPLED_COMPOSITION_FD_UNCERTAINTY_SCHEMA: Final = (
    "partition_free_v1_sampled_composition_physical_rank_fd_uncertainty_v1"
)
SAMPLED_COMPOSITION_D_REPLAY_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_sampled_composition_physical_rank_d_replay_v1"
)
SAMPLED_COMPOSITION_KKT_BRANCH_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_sampled_composition_physical_rank_kkt_branch_v1"
)
SAMPLED_COMPOSITION_HARD_FQ_REPLAY_SCHEMA: Final = (
    "partition_free_v1_sampled_composition_physical_rank_hard_fq_replay_v1"
)
SAMPLED_COMPOSITION_ONE_RANK_EXECUTION_SCHEMA: Final = (
    "partition_free_v1_sampled_composition_one_physical_rank_execution_v1"
)
SAMPLED_COMPOSITION_RANK_STABILITY_SCHEMA: Final = (
    "partition_free_v1_sampled_composition_rank_local_stability_v1"
)
SAMPLED_COMPOSITION_RANK_CHECKPOINT_SCHEMA: Final = (
    "partition_free_v1_sampled_composition_rank_checkpoint_v1"
)
SAMPLED_COMPOSITION_CALLBACK_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_sampled_composition_rank_callback_receipt_v1"
)
SAMPLED_COMPOSITION_ADAPTIVE_LOOP_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_sampled_composition_adaptive_physical_rank_loop_v1"
)
SAMPLED_COMPOSITION_NO_C_REFUSAL_SCHEMA: Final = (
    "partition_free_v1_sampled_composition_no_c_max_rank_ambiguity_refusal_v1"
)
SAMPLED_COMPOSITION_ONE_RANK_AUTHORITY_SCOPE: Final = (
    "one_physical_rank_D_to_FD_uncertainty_two_KKT_branches_all_hard_FQ_rungs_"
    "no_C_K_O_mask_commit_terminal_or_pareto_authority_v1"
)
SAMPLED_COMPOSITION_ADAPTIVE_LOOP_AUTHORITY_SCOPE: Final = (
    "authority_inert_adaptive_physical_rank_replay_until_stable_or_no_C_refusal_"
    "no_C_K_O_mask_commit_terminal_or_pareto_authority_v1"
)

SPECTRAL_KKT_BRANCH: Final = "spectral"
DIAGONAL_L1_KKT_BRANCH: Final = "diagonal_l1"
REQUIRED_KKT_BRANCHES: Final[tuple[str, str]] = (
    SPECTRAL_KKT_BRANCH,
    DIAGONAL_L1_KKT_BRANCH,
)

_DESIGN_PREFIX: Final = "partition-free-v1-sampled-composition-physical-fd-design-"
_TRANSACTION_PREFIX: Final = (
    "partition-free-v1-sampled-composition-registered-D-transaction-"
)
_D_PREFIX_BINDING_PREFIX: Final = (
    "partition-free-v1-sampled-composition-registered-D-prefix-binding-"
)
_RANK_INVARIANT_PREFIX: Final = "partition-free-v1-registered-D-rank-invariant-prefix-"
_CACHE_KEY_PREFIX: Final = (
    "partition-free-v1-sampled-composition-rank-local-D-cache-key-"
)
_FD_UNCERTAINTY_PREFIX: Final = "partition-free-v1-sampled-composition-fd-uncertainty-"
_D_REPLAY_PREFIX: Final = "partition-free-v1-sampled-composition-D-replay-"
_HARD_RUNG_PREFIX: Final = "partition-free-v1-sampled-composition-hard-rung-"
_HARD_MASK_PREFIX: Final = "partition-free-v1-sampled-composition-hard-mask-"
_ROUNDING_PREFIX: Final = (
    "partition-free-v1-sampled-composition-deterministic-rounding-prefix-"
)
_KKT_BRANCH_PREFIX: Final = "partition-free-v1-sampled-composition-kkt-branch-"
_FQ_REPLAY_PREFIX: Final = "partition-free-v1-sampled-composition-hard-FQ-replay-"
_COMPACT_BINDING_PREFIX: Final = (
    "partition-free-v1-sampled-composition-one-rank-compact-binding-"
)
_ONE_RANK_PREFIX: Final = "partition-free-v1-sampled-composition-one-rank-execution-"
_RANK_STABILITY_PREFIX: Final = "partition-free-v1-sampled-composition-rank-stability-"
_RANK_CHECKPOINT_PREFIX: Final = (
    "partition-free-v1-sampled-composition-rank-checkpoint-"
)
_CALLBACK_RECEIPT_PREFIX: Final = "partition-free-v1-sampled-composition-rank-callback-"
_NO_C_REFUSAL_PREFIX: Final = "partition-free-v1-sampled-composition-no-C-refusal-"
_ADAPTIVE_LOOP_PREFIX: Final = (
    "partition-free-v1-sampled-composition-adaptive-rank-loop-"
)

KKTBranchId = Literal["spectral", "diagonal_l1"]
RankStabilityDecision = Literal[
    "residual_stable",
    "decision_stable",
    "promote",
    "ambiguous",
]
RankCallbackKind = Literal["checkpoint", "eviction"]
RankCheckpointEvent = Literal["rank_completed", "rank_promoted", "rank_finalized"]
AdaptiveLoopStatus = Literal["stable", "no_c_refusal"]


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return str(value)


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be one positive integer")
    return int(value)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be one nonnegative integer")
    return int(value)


def _nonnegative_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be finite and nonnegative")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _immutable_payload_mapping(
    value: Mapping[str, object],
    *,
    name: str,
) -> tuple[tuple[str, object], ...]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be one nonempty mapping")
    rows: list[tuple[str, object]] = []
    for key, payload_value in value.items():
        rows.append((_identity(key, name=f"{name} key"), payload_value))
    rows.sort(key=lambda item: item[0])
    return tuple(rows)


def _immutable_payload_rows(
    value: object,
    *,
    name: str,
) -> tuple[tuple[str, object], ...]:
    if isinstance(value, Mapping):
        return _immutable_payload_mapping(value, name=name)
    rows = tuple(value)  # type: ignore[arg-type]
    keys: list[str] = []
    normalized: list[tuple[str, object]] = []
    for row in rows:
        if not isinstance(row, tuple) or len(row) != 2:
            raise ValueError(f"{name} must contain key/value rows")
        key = _identity(row[0], name=f"{name} key")
        keys.append(key)
        normalized.append((key, row[1]))
    if keys and (len(keys) != len(set(keys)) or tuple(sorted(keys)) != tuple(keys)):
        raise ValueError(f"{name} rows must be sorted with unique keys")
    return tuple(normalized)


def _strict_rank_ladder(value: Sequence[int], *, name: str) -> tuple[int, ...]:
    ranks = tuple(_positive_int(rank, name=f"{name} rank") for rank in value)
    if not ranks or any(after <= before for before, after in zip(ranks, ranks[1:])):
        raise ValueError(f"{name} must be one strictly increasing rank ladder")
    return ranks


def _optional_identity(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _identity(value, name=name)


def sampled_composition_rounding_prefix_hash(
    *,
    rounding_key_axis_hash: str,
    ordered_removed_route_indices: Sequence[int],
) -> str:
    """Commit one exact prefix of the registered deterministic rounding order."""

    indices = tuple(
        _nonnegative_int(value, name="removed physical route index")
        for value in ordered_removed_route_indices
    )
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("rounding prefix must contain unique physical route indices")
    return content_hash(
        {
            "schema": "partition_free_v1_sampled_composition_rounding_prefix_v1",
            "rounding_key_axis_hash": _identity(
                rounding_key_axis_hash,
                name="rounding key axis hash",
            ),
            "ordered_removed_route_indices": indices,
        },
        prefix=_ROUNDING_PREFIX,
    )


def sampled_composition_hard_mask_hash(
    *,
    current_mask_hash: str,
    route_axis_hash: str,
    removed_route_indices: Sequence[int],
) -> str:
    """Commit the set-valued hard mask induced by one rounding prefix."""

    indices = tuple(
        _nonnegative_int(value, name="removed physical route index")
        for value in removed_route_indices
    )
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("hard mask requires unique removed physical route indices")
    return content_hash(
        {
            "schema": "partition_free_v1_sampled_composition_hard_mask_v1",
            "current_mask_hash": _identity(current_mask_hash, name="current mask hash"),
            "route_axis_hash": _identity(route_axis_hash, name="route axis hash"),
            "removed_route_indices": tuple(sorted(indices)),
        },
        prefix=_HARD_MASK_PREFIX,
    )


def _branch_id(value: object, *, name: str = "KKT branch") -> KKTBranchId:
    branch = _identity(value, name=name)
    if branch not in REQUIRED_KKT_BRANCHES:
        raise ValueError(f"{name} must be one canonical KKT branch")
    return branch  # type: ignore[return-value]


@dataclass(frozen=True)
class V1SampledCompositionRegisteredDPrefixBinding:
    """Typed pre-C binding for the raw D prefix shared by every FD rank."""

    rank_invariant_raw_prefix_coordinate: tuple[object, ...]
    registered_look_count: int
    literal_j_source_receipt_hash: str
    current_mask_hash: str
    route_axis_hash: str
    route_count: int
    proposal_generation: int
    cumulative_stratum_counts: tuple[int, ...]
    D_context_axis_hash: str
    D_draw_axis_hash: str
    D_source_measurement_axis_hash: str
    D_weight_hash: str
    C_outcomes_seen: bool = False
    schema: str = SAMPLED_COMPOSITION_REGISTERED_D_PREFIX_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SAMPLED_COMPOSITION_REGISTERED_D_PREFIX_BINDING_SCHEMA:
            raise ValueError("registered-D prefix binding schema is invalid")
        coordinate = self.rank_invariant_raw_prefix_coordinate
        if not isinstance(coordinate, tuple) or not coordinate:
            raise ValueError("rank-invariant raw-prefix coordinate must be nonempty")
        object.__setattr__(
            self,
            "registered_look_count",
            _positive_int(self.registered_look_count, name="registered look count"),
        )
        for name, label in (
            ("literal_j_source_receipt_hash", "literal-J source receipt hash"),
            ("current_mask_hash", "current mask hash"),
            ("route_axis_hash", "route axis hash"),
            ("D_context_axis_hash", "D context axis hash"),
            ("D_draw_axis_hash", "D draw axis hash"),
            ("D_source_measurement_axis_hash", "D source measurement axis hash"),
            ("D_weight_hash", "D weight hash"),
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=label))
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="route count"),
        )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        counts = tuple(
            _positive_int(value, name="cumulative D stratum count")
            for value in self.cumulative_stratum_counts
        )
        if not counts:
            raise ValueError("registered-D prefix binding needs a stratum-count axis")
        object.__setattr__(self, "cumulative_stratum_counts", counts)
        if self.C_outcomes_seen is not False:
            raise PermissionError(
                "registered-D prefix binding cannot contain C outcomes"
            )

    @property
    def rank_invariant_raw_prefix_hash(self) -> str:
        return content_hash(
            self.rank_invariant_raw_prefix_coordinate,
            prefix=_RANK_INVARIANT_PREFIX,
        )

    @property
    def binding_hash(self) -> str:
        return content_hash(self, prefix=_D_PREFIX_BINDING_PREFIX)

    def assert_matches_record(self, record: V1FQRAdaptiveRankReplayRecord) -> None:
        if type(record) is not V1FQRAdaptiveRankReplayRecord:
            raise TypeError("registered-D binding requires an exact FQR replay record")
        observed = (
            record.current_mask_hash,
            record.route_axis_hash,
            record.route_count,
            record.proposal_generation,
            record.cumulative_stratum_counts,
            record.D_context_axis_hash,
            record.D_draw_axis_hash,
            record.D_source_measurement_axis_hash,
            record.D_weight_hash,
        )
        expected = (
            self.current_mask_hash,
            self.route_axis_hash,
            self.route_count,
            self.proposal_generation,
            self.cumulative_stratum_counts,
            self.D_context_axis_hash,
            self.D_draw_axis_hash,
            self.D_source_measurement_axis_hash,
            self.D_weight_hash,
        )
        if observed != expected:
            raise ValueError("source FQR replay record drifted from registered-D axes")


@dataclass(frozen=True)
class V1SampledCompositionPhysicalFDDesign:
    """One rank-explicit FD design, distinct from the max admitted rank."""

    physical_fd_rank: int
    maximum_identifiable_fd_rank: int
    fd_design_hash: str
    fd_design_semantics: str = "rank_explicit_physical_fd_design_v1"
    schema: str = SAMPLED_COMPOSITION_PHYSICAL_FD_DESIGN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SAMPLED_COMPOSITION_PHYSICAL_FD_DESIGN_SCHEMA:
            raise ValueError("sampled composition physical FD design schema is invalid")
        physical = _positive_int(self.physical_fd_rank, name="physical FD rank")
        maximum = _positive_int(
            self.maximum_identifiable_fd_rank,
            name="maximum identifiable FD rank",
        )
        if physical > maximum:
            raise ValueError("physical FD rank exceeds maximum identifiable FD rank")
        object.__setattr__(self, "physical_fd_rank", physical)
        object.__setattr__(self, "maximum_identifiable_fd_rank", maximum)
        object.__setattr__(
            self,
            "fd_design_hash",
            _identity(self.fd_design_hash, name="FD design hash"),
        )
        object.__setattr__(
            self,
            "fd_design_semantics",
            _identity(self.fd_design_semantics, name="FD design semantics"),
        )

    @property
    def design_hash(self) -> str:
        return content_hash(self._payload(), prefix=_DESIGN_PREFIX)

    def rank_local_cache_key(
        self,
        *,
        rank_invariant_raw_prefix_hash: str,
        literal_j_source_receipt_hash: str,
    ) -> str:
        return content_hash(
            {
                "schema": "partition_free_v1_sampled_composition_rank_cache_key_v1",
                "rank_invariant_raw_prefix_hash": _identity(
                    rank_invariant_raw_prefix_hash,
                    name="rank-invariant raw-prefix hash",
                ),
                "literal_j_source_receipt_hash": _identity(
                    literal_j_source_receipt_hash,
                    name="literal-J source receipt hash",
                ),
                "physical_fd_rank": self.physical_fd_rank,
                "maximum_identifiable_fd_rank": self.maximum_identifiable_fd_rank,
                "fd_design_hash": self.fd_design_hash,
                "design_hash": self.design_hash,
            },
            prefix=_CACHE_KEY_PREFIX,
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "physical_fd_rank": self.physical_fd_rank,
            "maximum_identifiable_fd_rank": self.maximum_identifiable_fd_rank,
            "fd_design_hash": self.fd_design_hash,
            "fd_design_semantics": self.fd_design_semantics,
        }


@dataclass(frozen=True)
class V1SampledCompositionRegisteredDTransaction:
    """Registered-D replay request for exactly one physical FD rank."""

    design: V1SampledCompositionPhysicalFDDesign
    rank_invariant_raw_prefix_coordinate: tuple[object, ...]
    registered_look_count: int
    literal_j_source_receipt_hash: str
    source_replay_axis_binding_hash: str | None = None
    first_look_index: int = 1
    schema: str = SAMPLED_COMPOSITION_REGISTERED_D_TRANSACTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SAMPLED_COMPOSITION_REGISTERED_D_TRANSACTION_SCHEMA:
            raise ValueError("sampled composition transaction schema is invalid")
        if type(self.design) is not V1SampledCompositionPhysicalFDDesign:
            raise TypeError(
                "sampled composition transaction needs a physical FD design"
            )
        if self.first_look_index != 1:
            raise ValueError("registered-D physical-rank replay must restart at look 1")
        if not isinstance(self.rank_invariant_raw_prefix_coordinate, tuple) or not (
            self.rank_invariant_raw_prefix_coordinate
        ):
            raise ValueError("rank-invariant raw-prefix coordinate must be nonempty")
        object.__setattr__(
            self,
            "registered_look_count",
            _positive_int(self.registered_look_count, name="registered look count"),
        )
        object.__setattr__(
            self,
            "literal_j_source_receipt_hash",
            _identity(
                self.literal_j_source_receipt_hash,
                name="literal-J source receipt hash",
            ),
        )
        object.__setattr__(
            self,
            "source_replay_axis_binding_hash",
            _optional_identity(
                self.source_replay_axis_binding_hash,
                name="source replay-axis binding hash",
            ),
        )

    @property
    def last_look_index(self) -> int:
        return self.first_look_index + self.registered_look_count - 1

    @property
    def physical_fd_rank(self) -> int:
        return self.design.physical_fd_rank

    @property
    def fd_design_hash(self) -> str:
        return self.design.fd_design_hash

    @property
    def rank_invariant_raw_prefix_hash(self) -> str:
        return content_hash(
            self.rank_invariant_raw_prefix_coordinate,
            prefix=_RANK_INVARIANT_PREFIX,
        )

    @property
    def transaction_hash(self) -> str:
        return content_hash(self._payload(), prefix=_TRANSACTION_PREFIX)

    def rank_local_cache_key(self, *, look_index: int) -> str:
        look = _positive_int(look_index, name="registered look index")
        if look < self.first_look_index or look > self.last_look_index:
            raise ValueError("registered look index is outside this transaction")
        return content_hash(
            {
                "schema": "partition_free_v1_sampled_composition_rank_look_cache_v1",
                "rank_invariant_raw_prefix_hash": self.rank_invariant_raw_prefix_hash,
                "physical_fd_rank": self.physical_fd_rank,
                "fd_design_hash": self.fd_design_hash,
                "design_hash": self.design.design_hash,
                "look_index": look,
                "literal_j_source_receipt_hash": self.literal_j_source_receipt_hash,
                "source_replay_axis_binding_hash": (
                    self.source_replay_axis_binding_hash
                ),
            },
            prefix=_CACHE_KEY_PREFIX,
        )

    @property
    def replay_look_indices(self) -> tuple[int, ...]:
        return tuple(range(self.first_look_index, self.last_look_index + 1))

    @property
    def rank_local_cache_keys(self) -> tuple[str, ...]:
        return tuple(
            self.rank_local_cache_key(look_index=look)
            for look in self.replay_look_indices
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "design": self.design._payload(),
            "rank_invariant_raw_prefix_coordinate": (
                self.rank_invariant_raw_prefix_coordinate
            ),
            "rank_invariant_raw_prefix_hash": self.rank_invariant_raw_prefix_hash,
            "registered_look_count": self.registered_look_count,
            "first_look_index": self.first_look_index,
            "last_look_index": self.last_look_index,
            "literal_j_source_receipt_hash": self.literal_j_source_receipt_hash,
            "source_replay_axis_binding_hash": self.source_replay_axis_binding_hash,
        }


@dataclass(frozen=True)
class V1SampledCompositionFDUncertaintyReceipt:
    """Deterministic/sampling FD uncertainty for one physical rank."""

    physical_fd_rank: int
    rank_invariant_raw_prefix_hash: str
    uncertainty_components: tuple[tuple[str, object], ...]
    schema: str = SAMPLED_COMPOSITION_FD_UNCERTAINTY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SAMPLED_COMPOSITION_FD_UNCERTAINTY_SCHEMA:
            raise ValueError("sampled composition FD uncertainty schema is invalid")
        object.__setattr__(
            self,
            "physical_fd_rank",
            _positive_int(self.physical_fd_rank, name="physical FD rank"),
        )
        object.__setattr__(
            self,
            "rank_invariant_raw_prefix_hash",
            _identity(
                self.rank_invariant_raw_prefix_hash,
                name="rank-invariant raw-prefix hash",
            ),
        )
        rows = tuple(self.uncertainty_components)
        if not rows:
            raise ValueError("FD uncertainty components must be nonempty")
        keys = tuple(key for key, _value in rows)
        if len(keys) != len(set(keys)) or tuple(sorted(keys)) != keys:
            raise ValueError("FD uncertainty components must have sorted unique keys")
        for key, value in rows:
            _identity(key, name="FD uncertainty component")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                _nonnegative_finite(value, name=f"FD uncertainty component {key}")
        object.__setattr__(self, "uncertainty_components", rows)

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_FD_UNCERTAINTY_PREFIX)


@dataclass(frozen=True)
class V1SampledCompositionDReplayReceipt:
    """Receipt that the identical registered-D prefix was reexecuted."""

    transaction_hash: str
    physical_fd_rank: int
    rank_invariant_raw_prefix_hash: str
    replayed_look_indices: tuple[int, ...]
    rank_local_cache_keys: tuple[str, ...]
    rank_local_cache_content_hashes: tuple[str, ...]
    fd_design_hash: str
    fd_snapshot_hash: str
    fd_uncertainty: V1SampledCompositionFDUncertaintyReceipt
    literal_j_source_receipt_hash: str
    schema: str = SAMPLED_COMPOSITION_D_REPLAY_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SAMPLED_COMPOSITION_D_REPLAY_RECEIPT_SCHEMA:
            raise ValueError("sampled composition D replay schema is invalid")
        object.__setattr__(
            self,
            "transaction_hash",
            _identity(self.transaction_hash, name="transaction hash"),
        )
        object.__setattr__(
            self,
            "physical_fd_rank",
            _positive_int(self.physical_fd_rank, name="physical FD rank"),
        )
        object.__setattr__(
            self,
            "rank_invariant_raw_prefix_hash",
            _identity(
                self.rank_invariant_raw_prefix_hash,
                name="rank-invariant raw-prefix hash",
            ),
        )
        looks = tuple(
            _positive_int(value, name="replayed look index")
            for value in self.replayed_look_indices
        )
        if not looks or looks != tuple(range(1, len(looks) + 1)):
            raise ValueError("D replay receipt must restart at look 1 without gaps")
        object.__setattr__(self, "replayed_look_indices", looks)
        keys = tuple(
            _identity(value, name="rank-local cache key")
            for value in self.rank_local_cache_keys
        )
        if len(keys) != len(looks):
            raise ValueError("rank-local cache key count differs from replayed looks")
        if len(set(keys)) != len(keys):
            raise ValueError("rank-local cache keys must be unique")
        object.__setattr__(self, "rank_local_cache_keys", keys)
        cache_contents = tuple(
            _identity(value, name="rank-local cache content hash")
            for value in self.rank_local_cache_content_hashes
        )
        if len(cache_contents) != len(looks):
            raise ValueError(
                "rank-local cache content hash count differs from replayed looks"
            )
        object.__setattr__(self, "rank_local_cache_content_hashes", cache_contents)
        object.__setattr__(
            self,
            "fd_design_hash",
            _identity(self.fd_design_hash, name="FD design hash"),
        )
        object.__setattr__(
            self,
            "fd_snapshot_hash",
            _identity(self.fd_snapshot_hash, name="FD snapshot hash"),
        )
        if type(self.fd_uncertainty) is not V1SampledCompositionFDUncertaintyReceipt:
            raise TypeError("D replay receipt needs an FD uncertainty receipt")
        if self.fd_uncertainty.physical_fd_rank != self.physical_fd_rank:
            raise ValueError("D replay FD uncertainty names another physical FD rank")
        if (
            self.fd_uncertainty.rank_invariant_raw_prefix_hash
            != self.rank_invariant_raw_prefix_hash
        ):
            raise ValueError("D replay FD uncertainty names another raw-prefix hash")
        object.__setattr__(
            self,
            "literal_j_source_receipt_hash",
            _identity(
                self.literal_j_source_receipt_hash,
                name="literal-J source receipt hash",
            ),
        )

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_D_REPLAY_PREFIX)


@dataclass(frozen=True)
class V1SampledCompositionHardRung:
    """One auditable prefix increment of a deterministic rounded ladder."""

    branch_id: KKTBranchId
    rung_id: str
    hard_mask_hash: str
    registered_resource_count: int
    packed_action_hash: str
    added_physical_route_indices: tuple[int, ...]
    rounding_prefix_hash: str
    parent_rung_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "branch_id", _branch_id(self.branch_id))
        object.__setattr__(self, "rung_id", _identity(self.rung_id, name="rung ID"))
        object.__setattr__(
            self,
            "hard_mask_hash",
            _identity(self.hard_mask_hash, name="hard mask hash"),
        )
        object.__setattr__(
            self,
            "registered_resource_count",
            _positive_int(
                self.registered_resource_count,
                name="registered resource count",
            ),
        )
        object.__setattr__(
            self,
            "packed_action_hash",
            _identity(self.packed_action_hash, name="packed action hash"),
        )
        additions = tuple(
            _nonnegative_int(value, name="added physical route index")
            for value in self.added_physical_route_indices
        )
        if not additions or len(additions) != len(set(additions)):
            raise ValueError("hard rung additions must be nonempty and unique")
        object.__setattr__(self, "added_physical_route_indices", additions)
        object.__setattr__(
            self,
            "rounding_prefix_hash",
            _identity(self.rounding_prefix_hash, name="rounding prefix hash"),
        )
        object.__setattr__(
            self,
            "parent_rung_hash",
            _optional_identity(self.parent_rung_hash, name="parent rung hash"),
        )

    @property
    def rung_hash(self) -> str:
        return content_hash(self, prefix=_HARD_RUNG_PREFIX)


@dataclass(frozen=True)
class V1SampledCompositionKKTBranchReceipt:
    """One branch-local relaxed KKT receipt plus all rounded hard rungs."""

    branch_id: KKTBranchId
    relaxed_solution_hash: str
    kkt_optimality_receipt_hash: str
    current_mask_hash: str
    route_axis_hash: str
    route_count: int
    deterministic_rounding_policy_hash: str
    rounding_key_axis_hash: str
    legality_receipt_hash: str
    source_fqr_hard_ladder_hash: str
    source_fqr_branch_decision_receipt_hash: str
    hard_rungs: tuple[V1SampledCompositionHardRung, ...]
    schema: str = SAMPLED_COMPOSITION_KKT_BRANCH_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SAMPLED_COMPOSITION_KKT_BRANCH_RECEIPT_SCHEMA:
            raise ValueError("sampled composition KKT branch schema is invalid")
        branch = _branch_id(self.branch_id)
        object.__setattr__(self, "branch_id", branch)
        object.__setattr__(
            self,
            "relaxed_solution_hash",
            _identity(self.relaxed_solution_hash, name="relaxed solution hash"),
        )
        object.__setattr__(
            self,
            "kkt_optimality_receipt_hash",
            _identity(
                self.kkt_optimality_receipt_hash,
                name="KKT optimality receipt hash",
            ),
        )
        for name, label in (
            ("current_mask_hash", "current mask hash"),
            ("route_axis_hash", "route axis hash"),
            ("deterministic_rounding_policy_hash", "rounding policy hash"),
            ("rounding_key_axis_hash", "rounding key axis hash"),
            ("legality_receipt_hash", "legality receipt hash"),
            ("source_fqr_hard_ladder_hash", "source FQR hard-ladder hash"),
            (
                "source_fqr_branch_decision_receipt_hash",
                "source FQR branch decision receipt hash",
            ),
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=label))
        routes = _positive_int(self.route_count, name="route count")
        object.__setattr__(self, "route_count", routes)
        rungs = tuple(self.hard_rungs)
        if not rungs:
            raise ValueError("KKT branch must emit at least one hard rung")
        if any(type(rung) is not V1SampledCompositionHardRung for rung in rungs):
            raise TypeError("KKT branch contains a noncanonical hard rung")
        if any(rung.branch_id != branch for rung in rungs):
            raise ValueError("KKT branch hard rung names another branch")
        rung_ids = tuple(rung.rung_id for rung in rungs)
        if len(rung_ids) != len(set(rung_ids)):
            raise ValueError("KKT branch hard rung IDs must be unique")
        mask_hashes = tuple(rung.hard_mask_hash for rung in rungs)
        if len(mask_hashes) != len(set(mask_hashes)):
            raise ValueError("KKT branch hard mask hashes must be unique")
        cumulative_order: list[int] = []
        previous: V1SampledCompositionHardRung | None = None
        for rung in rungs:
            if rung.parent_rung_hash != (
                None if previous is None else previous.rung_hash
            ):
                raise ValueError("KKT branch rung parent lineage is not contiguous")
            if any(value >= routes for value in rung.added_physical_route_indices):
                raise ValueError("KKT branch rung contains an out-of-range route")
            if set(cumulative_order).intersection(rung.added_physical_route_indices):
                raise ValueError("KKT branch rounding prefix removes a route twice")
            cumulative_order.extend(rung.added_physical_route_indices)
            if rung.registered_resource_count != len(cumulative_order):
                raise ValueError(
                    "KKT branch registered resource count does not equal its prefix"
                )
            expected_prefix = sampled_composition_rounding_prefix_hash(
                rounding_key_axis_hash=self.rounding_key_axis_hash,
                ordered_removed_route_indices=cumulative_order,
            )
            if rung.rounding_prefix_hash != expected_prefix:
                raise ValueError("KKT branch rounding-prefix lineage is not replayable")
            expected_mask = sampled_composition_hard_mask_hash(
                current_mask_hash=self.current_mask_hash,
                route_axis_hash=self.route_axis_hash,
                removed_route_indices=cumulative_order,
            )
            if rung.hard_mask_hash != expected_mask:
                raise ValueError("KKT branch hard mask does not replay its prefix")
            previous = rung
        object.__setattr__(self, "hard_rungs", rungs)

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_KKT_BRANCH_PREFIX)


@dataclass(frozen=True)
class V1SampledCompositionHardFQReplayReceipt:
    """Complete hard F/Q replay for one frozen rung."""

    branch_id: KKTBranchId
    rung_id: str
    hard_mask_hash: str
    source_rung_hash: str
    source_branch_receipt_hash: str
    source_d_replay_receipt_hash: str
    registered_resource_count: int
    f_upper: float
    q_upper: float
    schema: str = SAMPLED_COMPOSITION_HARD_FQ_REPLAY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SAMPLED_COMPOSITION_HARD_FQ_REPLAY_SCHEMA:
            raise ValueError("sampled composition hard F/Q replay schema is invalid")
        object.__setattr__(self, "branch_id", _branch_id(self.branch_id))
        object.__setattr__(self, "rung_id", _identity(self.rung_id, name="rung ID"))
        object.__setattr__(
            self,
            "hard_mask_hash",
            _identity(self.hard_mask_hash, name="hard mask hash"),
        )
        object.__setattr__(
            self,
            "source_rung_hash",
            _identity(self.source_rung_hash, name="source rung hash"),
        )
        object.__setattr__(
            self,
            "source_branch_receipt_hash",
            _identity(
                self.source_branch_receipt_hash,
                name="source branch receipt hash",
            ),
        )
        object.__setattr__(
            self,
            "registered_resource_count",
            _positive_int(
                self.registered_resource_count,
                name="registered resource count",
            ),
        )
        object.__setattr__(
            self,
            "source_d_replay_receipt_hash",
            _identity(
                self.source_d_replay_receipt_hash,
                name="source D replay receipt hash",
            ),
        )
        object.__setattr__(
            self,
            "f_upper",
            _nonnegative_finite(self.f_upper, name="F upper replay bound"),
        )
        object.__setattr__(
            self,
            "q_upper",
            _nonnegative_finite(self.q_upper, name="Q upper replay bound"),
        )

    @property
    def replay_hash(self) -> str:
        return content_hash(self, prefix=_FQ_REPLAY_PREFIX)


@dataclass(frozen=True)
class V1SampledCompositionOneRankExecutionReceipt:
    """Compact no-authority receipt for one physical-rank execution."""

    transaction_hash: str
    physical_fd_rank: int
    rank_invariant_raw_prefix_hash: str
    d_replay: V1SampledCompositionDReplayReceipt
    branch_receipts: tuple[V1SampledCompositionKKTBranchReceipt, ...]
    hard_fq_replays: tuple[V1SampledCompositionHardFQReplayReceipt, ...]
    literal_j_source_receipt_hash: str
    authority_scope: str = SAMPLED_COMPOSITION_ONE_RANK_AUTHORITY_SCOPE
    C_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    adaptive_rank_loop_authority: bool = False
    schema: str = SAMPLED_COMPOSITION_ONE_RANK_EXECUTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SAMPLED_COMPOSITION_ONE_RANK_EXECUTION_SCHEMA:
            raise ValueError("sampled composition one-rank execution schema is invalid")
        for name in (
            "C_authority",
            "calibration_authority",
            "oracle_authority",
            "mask_commit_authority",
            "adaptive_rank_loop_authority",
        ):
            if getattr(self, name) is not False:
                raise PermissionError("one-rank sampled backend cannot grant authority")
        object.__setattr__(
            self,
            "transaction_hash",
            _identity(self.transaction_hash, name="transaction hash"),
        )
        object.__setattr__(
            self,
            "physical_fd_rank",
            _positive_int(self.physical_fd_rank, name="physical FD rank"),
        )
        object.__setattr__(
            self,
            "rank_invariant_raw_prefix_hash",
            _identity(
                self.rank_invariant_raw_prefix_hash,
                name="rank-invariant raw-prefix hash",
            ),
        )
        if type(self.d_replay) is not V1SampledCompositionDReplayReceipt:
            raise TypeError("one-rank execution needs one D replay receipt")
        if self.d_replay.transaction_hash != self.transaction_hash:
            raise ValueError("one-rank execution D replay names another transaction")
        if self.d_replay.physical_fd_rank != self.physical_fd_rank:
            raise ValueError("one-rank execution D replay names another rank")
        if (
            self.d_replay.rank_invariant_raw_prefix_hash
            != self.rank_invariant_raw_prefix_hash
        ):
            raise ValueError("one-rank execution D replay names another raw prefix")
        if (
            self.d_replay.literal_j_source_receipt_hash
            != self.literal_j_source_receipt_hash
        ):
            raise ValueError(
                "one-rank execution D replay names another literal-J source"
            )
        branches = tuple(self.branch_receipts)
        if tuple(branch.branch_id for branch in branches) != REQUIRED_KKT_BRANCHES:
            raise ValueError("one-rank execution must contain both KKT branches")
        replays = tuple(self.hard_fq_replays)
        expected_rungs = tuple(
            (branch.branch_id, rung.rung_id, rung.hard_mask_hash)
            for branch in branches
            for rung in branch.hard_rungs
        )
        observed_rungs = tuple(
            (replay.branch_id, replay.rung_id, replay.hard_mask_hash)
            for replay in replays
        )
        if observed_rungs != expected_rungs:
            raise ValueError(
                "one-rank execution must replay every hard F/Q rung in "
                "deterministic branch/rung order"
            )
        if len(set(observed_rungs)) != len(observed_rungs):
            raise ValueError("one-rank execution contains duplicate F/Q replays")
        branch_hashes = {branch.branch_id: branch.receipt_hash for branch in branches}
        if len(set(branch_hashes.values())) != len(branches):
            raise ValueError("one-rank execution contains duplicate KKT branches")
        for replay in replays:
            if replay.source_d_replay_receipt_hash != self.d_replay.receipt_hash:
                raise ValueError("one-rank execution F/Q replay names another D replay")
            if replay.source_branch_receipt_hash != branch_hashes[replay.branch_id]:
                raise ValueError(
                    "one-rank execution F/Q replay names another branch receipt"
                )
        object.__setattr__(self, "branch_receipts", branches)
        object.__setattr__(self, "hard_fq_replays", replays)
        object.__setattr__(
            self,
            "literal_j_source_receipt_hash",
            _identity(
                self.literal_j_source_receipt_hash,
                name="literal-J source receipt hash",
            ),
        )

    @property
    def compact_binding_hash(self) -> str:
        return content_hash(
            {
                "schema": "partition_free_v1_sampled_composition_one_rank_compact_binding_v1",
                "transaction_hash": self.transaction_hash,
                "physical_fd_rank": self.physical_fd_rank,
                "rank_invariant_raw_prefix_hash": self.rank_invariant_raw_prefix_hash,
                "d_replay_receipt_hash": self.d_replay.receipt_hash,
                "fd_uncertainty_receipt_hash": (
                    self.d_replay.fd_uncertainty.receipt_hash
                ),
                "branch_receipt_hashes": tuple(
                    branch.receipt_hash for branch in self.branch_receipts
                ),
                "hard_fq_replay_hashes": tuple(
                    replay.replay_hash for replay in self.hard_fq_replays
                ),
                "literal_j_source_receipt_hash": (self.literal_j_source_receipt_hash),
                "authority_scope": self.authority_scope,
            },
            prefix=_COMPACT_BINDING_PREFIX,
        )

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "compact_binding_hash": self.compact_binding_hash,
                "authority_scope": self.authority_scope,
            },
            prefix=_ONE_RANK_PREFIX,
        )


@dataclass(frozen=True)
class V1SampledCompositionSourceFQRRankEvaluation:
    """Transport for exact canonical FQR ledger and certificate/refusal objects."""

    replay_ledger: V1FQRAdaptiveRankReplayLedger
    outcome: V1FQRAdaptiveRankStabilityCertificate | V1FQRAdaptiveRankNoCRefusal

    def __post_init__(self) -> None:
        if type(self.replay_ledger) is not V1FQRAdaptiveRankReplayLedger:
            raise TypeError("sampled adapter requires an exact FQR replay ledger")
        if type(self.outcome) not in (
            V1FQRAdaptiveRankStabilityCertificate,
            V1FQRAdaptiveRankNoCRefusal,
        ):
            raise TypeError("sampled adapter requires an exact FQR certificate/refusal")
        policy_hash = self.replay_ledger.policy.assert_integrity()
        ledger_hash = self.replay_ledger.assert_integrity()
        if self.outcome.policy_hash != policy_hash:
            raise ValueError("source FQR outcome names another policy")
        outcome_ledger_hash = (
            self.outcome.replay_ledger_hash
            if type(self.outcome) is V1FQRAdaptiveRankStabilityCertificate
            else self.outcome.ledger_hash
        )
        if outcome_ledger_hash != ledger_hash:
            raise ValueError("source FQR outcome names another replay ledger")

    @property
    def outcome_hash(self) -> str:
        if type(self.outcome) is V1FQRAdaptiveRankStabilityCertificate:
            return self.outcome.certificate_hash
        return self.outcome.refusal_hash


@dataclass(frozen=True)
class V1SampledCompositionRankStabilityReceipt:
    """Authority-inert binding to a canonically revalidated FQR outcome."""

    physical_fd_rank: int
    rank_invariant_raw_prefix_hash: str
    source_replay_axis_binding_hash: str
    one_rank_execution_receipt_hash: str
    decision: RankStabilityDecision
    source_fqr_policy_hash: str
    source_fqr_replay_ledger_hash: str
    source_fqr_executed_rank_prefix: tuple[int, ...]
    source_fqr_replay_record_hashes: tuple[str, ...]
    source_fqr_frozen_replay_record_hash: str
    current_mask_hash: str
    route_axis_hash: str
    route_count: int
    proposal_generation: int
    source_fqr_stability_certificate_hash: str | None = None
    source_fqr_refusal_hash: str | None = None
    source_fqr_refusal_code: V1FQRAdaptiveRankRefusalCode | None = None
    source_fqr_rank_local_decision_certificate_hash: str | None = None
    diagnostic_payload: tuple[tuple[str, object], ...] = ()
    schema: str = SAMPLED_COMPOSITION_RANK_STABILITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SAMPLED_COMPOSITION_RANK_STABILITY_SCHEMA:
            raise ValueError("sampled composition rank stability schema is invalid")
        object.__setattr__(
            self,
            "physical_fd_rank",
            _positive_int(self.physical_fd_rank, name="physical FD rank"),
        )
        for name, label in (
            ("rank_invariant_raw_prefix_hash", "rank-invariant raw-prefix hash"),
            ("source_replay_axis_binding_hash", "source replay-axis binding hash"),
            ("one_rank_execution_receipt_hash", "one-rank execution receipt hash"),
            ("source_fqr_policy_hash", "source FQR adaptive-rank policy hash"),
            (
                "source_fqr_replay_ledger_hash",
                "source FQR adaptive-rank replay ledger hash",
            ),
            (
                "source_fqr_frozen_replay_record_hash",
                "source FQR frozen replay record hash",
            ),
            ("current_mask_hash", "current mask hash"),
            ("route_axis_hash", "route axis hash"),
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=label))
        decision = _identity(self.decision, name="rank stability decision")
        if decision not in ("decision_stable", "promote", "ambiguous"):
            raise ValueError("rank stability decision is not canonical")
        object.__setattr__(self, "decision", decision)
        ranks = _strict_rank_ladder(
            self.source_fqr_executed_rank_prefix,
            name="source FQR executed rank prefix",
        )
        hashes = tuple(
            _identity(value, name="source FQR replay record hash")
            for value in self.source_fqr_replay_record_hashes
        )
        if (
            len(hashes) != len(ranks)
            or ranks[-1] != self.physical_fd_rank
            or hashes[-1] != self.source_fqr_frozen_replay_record_hash
        ):
            raise ValueError(
                "sampled rank stability has an incomplete source rank axis"
            )
        object.__setattr__(self, "source_fqr_executed_rank_prefix", ranks)
        object.__setattr__(self, "source_fqr_replay_record_hashes", hashes)
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="route count"),
        )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        stable_hash = _optional_identity(
            self.source_fqr_stability_certificate_hash,
            name="source FQR adaptive-rank stability certificate hash",
        )
        refusal_hash = _optional_identity(
            self.source_fqr_refusal_hash,
            name="source FQR adaptive-rank refusal hash",
        )
        rank_decision_hash = _optional_identity(
            self.source_fqr_rank_local_decision_certificate_hash,
            name="source FQR rank-local decision certificate hash",
        )
        if self.is_stable:
            if (
                stable_hash is None
                or refusal_hash is not None
                or self.source_fqr_refusal_code is not None
                or rank_decision_hash is None
            ):
                raise ValueError("stable sampled rank must bind one source certificate")
        else:
            if stable_hash is not None or refusal_hash is None:
                raise ValueError("unstable sampled rank must bind one source refusal")
            if type(self.source_fqr_refusal_code) is not V1FQRAdaptiveRankRefusalCode:
                raise TypeError("unstable sampled rank must preserve its refusal code")
            expected = (
                V1FQRAdaptiveRankRefusalCode.RANK_PROMOTION_REQUIRED
                if decision == "promote"
                else V1FQRAdaptiveRankRefusalCode.DECISION_FAMILY_AMBIGUOUS
            )
            if self.source_fqr_refusal_code is not expected:
                raise ValueError(
                    "sampled rank decision laundered its source refusal code"
                )
        object.__setattr__(self, "source_fqr_stability_certificate_hash", stable_hash)
        object.__setattr__(self, "source_fqr_refusal_hash", refusal_hash)
        object.__setattr__(
            self,
            "source_fqr_rank_local_decision_certificate_hash",
            rank_decision_hash,
        )
        object.__setattr__(
            self,
            "diagnostic_payload",
            _immutable_payload_rows(
                self.diagnostic_payload,
                name="rank stability diagnostic payload",
            ),
        )

    @property
    def is_stable(self) -> bool:
        return self.decision == "decision_stable"

    @property
    def source_fqr_outcome_hash(self) -> str:
        if self.is_stable:
            assert self.source_fqr_stability_certificate_hash is not None
            return self.source_fqr_stability_certificate_hash
        assert self.source_fqr_refusal_hash is not None
        return self.source_fqr_refusal_hash

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_RANK_STABILITY_PREFIX)


@dataclass(frozen=True)
class V1SampledCompositionRankCheckpointReceipt:
    """Restartable checkpoint for one physical-rank replay boundary."""

    event: RankCheckpointEvent
    physical_fd_rank: int
    rank_invariant_raw_prefix_hash: str
    source_replay_axis_binding_hash: str
    transaction_hash: str
    rank_local_cache_keys: tuple[str, ...]
    rank_local_cache_content_hashes: tuple[str, ...]
    live_geometry_hashes: tuple[str, ...]
    one_rank_execution_receipt_hash: str
    stability_receipt_hash: str
    source_fqr_policy_hash: str
    source_fqr_outcome_hash: str
    previous_checkpoint_hash: str | None = None
    schema: str = SAMPLED_COMPOSITION_RANK_CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SAMPLED_COMPOSITION_RANK_CHECKPOINT_SCHEMA:
            raise ValueError("sampled composition checkpoint schema is invalid")
        event = _identity(self.event, name="rank checkpoint event")
        if event not in ("rank_completed", "rank_promoted", "rank_finalized"):
            raise ValueError("rank checkpoint event is not canonical")
        object.__setattr__(self, "event", event)
        object.__setattr__(
            self,
            "physical_fd_rank",
            _positive_int(self.physical_fd_rank, name="physical FD rank"),
        )
        object.__setattr__(
            self,
            "rank_invariant_raw_prefix_hash",
            _identity(
                self.rank_invariant_raw_prefix_hash,
                name="rank-invariant raw-prefix hash",
            ),
        )
        object.__setattr__(
            self,
            "source_replay_axis_binding_hash",
            _identity(
                self.source_replay_axis_binding_hash,
                name="source replay-axis binding hash",
            ),
        )
        object.__setattr__(
            self,
            "transaction_hash",
            _identity(self.transaction_hash, name="transaction hash"),
        )
        keys = tuple(
            _identity(value, name="rank-local cache key")
            for value in self.rank_local_cache_keys
        )
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("checkpoint rank-local cache keys must be unique")
        object.__setattr__(self, "rank_local_cache_keys", keys)
        contents = tuple(
            _identity(value, name="rank-local cache content hash")
            for value in self.rank_local_cache_content_hashes
        )
        if len(contents) != len(keys):
            raise ValueError("checkpoint cache contents do not cover every cache key")
        object.__setattr__(self, "rank_local_cache_content_hashes", contents)
        live = tuple(
            _identity(value, name="live rank-local geometry hash")
            for value in self.live_geometry_hashes
        )
        if not live or len(live) != len(set(live)):
            raise ValueError(
                "checkpoint live geometry hashes must be nonempty and unique"
            )
        object.__setattr__(self, "live_geometry_hashes", live)
        object.__setattr__(
            self,
            "one_rank_execution_receipt_hash",
            _identity(
                self.one_rank_execution_receipt_hash,
                name="one-rank execution receipt hash",
            ),
        )
        object.__setattr__(
            self,
            "stability_receipt_hash",
            _identity(self.stability_receipt_hash, name="stability receipt hash"),
        )
        object.__setattr__(
            self,
            "source_fqr_policy_hash",
            _identity(
                self.source_fqr_policy_hash,
                name="source FQR adaptive-rank policy hash",
            ),
        )
        object.__setattr__(
            self,
            "source_fqr_outcome_hash",
            _identity(
                self.source_fqr_outcome_hash,
                name="source FQR stability/refusal outcome hash",
            ),
        )
        object.__setattr__(
            self,
            "previous_checkpoint_hash",
            _optional_identity(
                self.previous_checkpoint_hash,
                name="previous checkpoint hash",
            ),
        )

    @property
    def restart_replay_hash(self) -> str:
        return content_hash(
            {
                "schema": "partition_free_v1_sampled_composition_rank_restart_v1",
                "event": self.event,
                "physical_fd_rank": self.physical_fd_rank,
                "rank_invariant_raw_prefix_hash": self.rank_invariant_raw_prefix_hash,
                "source_replay_axis_binding_hash": (
                    self.source_replay_axis_binding_hash
                ),
                "transaction_hash": self.transaction_hash,
                "rank_local_cache_keys": self.rank_local_cache_keys,
                "rank_local_cache_content_hashes": (
                    self.rank_local_cache_content_hashes
                ),
                "live_geometry_hashes": self.live_geometry_hashes,
                "one_rank_execution_receipt_hash": self.one_rank_execution_receipt_hash,
                "stability_receipt_hash": self.stability_receipt_hash,
                "source_fqr_policy_hash": self.source_fqr_policy_hash,
                "source_fqr_outcome_hash": self.source_fqr_outcome_hash,
                "previous_checkpoint_hash": self.previous_checkpoint_hash,
            },
            prefix=_RANK_CHECKPOINT_PREFIX,
        )

    @property
    def checkpoint_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "restart_replay_hash": self.restart_replay_hash,
            },
            prefix=_RANK_CHECKPOINT_PREFIX,
        )


@dataclass(frozen=True)
class V1SampledCompositionRankCallbackReceipt:
    """Evidence that checkpoint or eviction callback observed the frozen boundary."""

    callback_kind: RankCallbackKind
    source_checkpoint_receipt_hash: str
    physical_fd_rank: int
    rank_invariant_raw_prefix_hash: str
    rank_local_cache_keys: tuple[str, ...]
    rank_local_cache_content_hashes: tuple[str, ...]
    observed_live_geometry_hashes: tuple[str, ...]
    restart_replay_hash: str
    persisted_checkpoint_hash: str | None = None
    durable_payload_hash: str | None = None
    evicted_geometry_hashes: tuple[str, ...] = ()
    remaining_rank_local_geometry_hashes: tuple[str, ...] = ()
    action_verified: bool = True
    schema: str = SAMPLED_COMPOSITION_CALLBACK_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SAMPLED_COMPOSITION_CALLBACK_RECEIPT_SCHEMA:
            raise ValueError("sampled composition callback schema is invalid")
        kind = _identity(self.callback_kind, name="rank callback kind")
        if kind not in ("checkpoint", "eviction"):
            raise ValueError("rank callback kind is not canonical")
        object.__setattr__(self, "callback_kind", kind)
        object.__setattr__(
            self,
            "source_checkpoint_receipt_hash",
            _identity(
                self.source_checkpoint_receipt_hash,
                name="source checkpoint receipt hash",
            ),
        )
        object.__setattr__(
            self,
            "physical_fd_rank",
            _positive_int(self.physical_fd_rank, name="physical FD rank"),
        )
        object.__setattr__(
            self,
            "rank_invariant_raw_prefix_hash",
            _identity(
                self.rank_invariant_raw_prefix_hash,
                name="rank-invariant raw-prefix hash",
            ),
        )
        keys = tuple(
            _identity(value, name="rank-local cache key")
            for value in self.rank_local_cache_keys
        )
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("callback rank-local cache keys must be unique")
        object.__setattr__(self, "rank_local_cache_keys", keys)
        contents = tuple(
            _identity(value, name="callback cache content hash")
            for value in self.rank_local_cache_content_hashes
        )
        if len(contents) != len(keys):
            raise ValueError("callback cache contents do not cover every cache key")
        object.__setattr__(self, "rank_local_cache_content_hashes", contents)
        observed = tuple(
            _identity(value, name="observed live geometry hash")
            for value in self.observed_live_geometry_hashes
        )
        if not observed or len(observed) != len(set(observed)):
            raise ValueError("callback observed geometry must be nonempty and unique")
        object.__setattr__(self, "observed_live_geometry_hashes", observed)
        object.__setattr__(
            self,
            "restart_replay_hash",
            _identity(self.restart_replay_hash, name="restart replay hash"),
        )
        persisted = _optional_identity(
            self.persisted_checkpoint_hash,
            name="persisted checkpoint hash",
        )
        durable = _optional_identity(
            self.durable_payload_hash,
            name="durable checkpoint payload hash",
        )
        evicted = tuple(
            _identity(value, name="evicted geometry hash")
            for value in self.evicted_geometry_hashes
        )
        remaining = tuple(
            _identity(value, name="remaining geometry hash")
            for value in self.remaining_rank_local_geometry_hashes
        )
        if self.action_verified is not True:
            raise PermissionError("rank callback did not verify its state transition")
        if kind == "checkpoint":
            if persisted is None or durable is None or evicted or remaining != observed:
                raise ValueError(
                    "checkpoint callback lacks durable live-state evidence"
                )
        else:
            if persisted is not None or durable is not None:
                raise ValueError(
                    "eviction callback cannot claim checkpoint persistence"
                )
            if evicted != observed or remaining:
                raise ValueError(
                    "eviction callback did not remove all rank-local geometry"
                )
        object.__setattr__(self, "persisted_checkpoint_hash", persisted)
        object.__setattr__(self, "durable_payload_hash", durable)
        object.__setattr__(self, "evicted_geometry_hashes", evicted)
        object.__setattr__(self, "remaining_rank_local_geometry_hashes", remaining)

    @property
    def callback_hash(self) -> str:
        return content_hash(self, prefix=_CALLBACK_RECEIPT_PREFIX)


@dataclass(frozen=True)
class V1SampledCompositionNoCRefusalReceipt:
    """Typed sampled-backend no-C refusal derived from FQR max-rank ambiguity."""

    maximum_physical_fd_rank: int
    rank_invariant_raw_prefix_hash: str
    last_one_rank_execution_receipt_hash: str
    last_stability_receipt_hash: str
    source_fqr_policy_hash: str
    source_fqr_no_c_refusal_hash: str
    source_fqr_refusal_code: V1FQRAdaptiveRankRefusalCode
    reason: str
    C_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    schema: str = SAMPLED_COMPOSITION_NO_C_REFUSAL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SAMPLED_COMPOSITION_NO_C_REFUSAL_SCHEMA:
            raise ValueError("sampled composition no-C refusal schema is invalid")
        for name in (
            "C_authority",
            "calibration_authority",
            "oracle_authority",
            "mask_commit_authority",
        ):
            if getattr(self, name) is not False:
                raise PermissionError("sampled no-C refusal cannot grant authority")
        object.__setattr__(
            self,
            "maximum_physical_fd_rank",
            _positive_int(
                self.maximum_physical_fd_rank, name="maximum physical FD rank"
            ),
        )
        object.__setattr__(
            self,
            "rank_invariant_raw_prefix_hash",
            _identity(
                self.rank_invariant_raw_prefix_hash,
                name="rank-invariant raw-prefix hash",
            ),
        )
        object.__setattr__(
            self,
            "last_one_rank_execution_receipt_hash",
            _identity(
                self.last_one_rank_execution_receipt_hash,
                name="last one-rank execution receipt hash",
            ),
        )
        object.__setattr__(
            self,
            "last_stability_receipt_hash",
            _identity(
                self.last_stability_receipt_hash,
                name="last stability receipt hash",
            ),
        )
        object.__setattr__(
            self,
            "source_fqr_policy_hash",
            _identity(
                self.source_fqr_policy_hash,
                name="source FQR adaptive-rank policy hash",
            ),
        )
        object.__setattr__(
            self,
            "source_fqr_no_c_refusal_hash",
            _identity(
                self.source_fqr_no_c_refusal_hash,
                name="source FQR no-C refusal hash",
            ),
        )
        if self.source_fqr_refusal_code is not (
            V1FQRAdaptiveRankRefusalCode.DECISION_FAMILY_AMBIGUOUS
        ):
            raise ValueError("sampled max-rank no-C receipt changed its refusal code")
        object.__setattr__(
            self, "reason", _identity(self.reason, name="refusal reason")
        )

    @property
    def refusal_hash(self) -> str:
        return content_hash(self, prefix=_NO_C_REFUSAL_PREFIX)


@dataclass(frozen=True)
class V1SampledCompositionAdaptiveRankLoopReceipt:
    """No-C sampled adapter over a contiguous physical-rank replay prefix."""

    rank_ladder: tuple[int, ...]
    rank_invariant_raw_prefix_hash: str
    source_replay_axis_binding_hash: str
    source_fqr_policy_hash: str
    status: AdaptiveLoopStatus
    executions: tuple[V1SampledCompositionOneRankExecutionReceipt, ...]
    stability_receipts: tuple[V1SampledCompositionRankStabilityReceipt, ...]
    rank_checkpoint_receipts: tuple[V1SampledCompositionRankCheckpointReceipt, ...]
    checkpoint_callback_receipts: tuple[V1SampledCompositionRankCallbackReceipt, ...]
    eviction_callback_receipts: tuple[V1SampledCompositionRankCallbackReceipt, ...]
    selected_physical_fd_rank: int | None = None
    no_c_refusal: V1SampledCompositionNoCRefusalReceipt | None = None
    resumed_from_checkpoint_hash: str | None = None
    restart_checkpoint_verified: bool = False
    authority_scope: str = SAMPLED_COMPOSITION_ADAPTIVE_LOOP_AUTHORITY_SCOPE
    C_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    schema: str = SAMPLED_COMPOSITION_ADAPTIVE_LOOP_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SAMPLED_COMPOSITION_ADAPTIVE_LOOP_RECEIPT_SCHEMA:
            raise ValueError("sampled composition adaptive loop schema is invalid")
        for name in (
            "C_authority",
            "calibration_authority",
            "oracle_authority",
            "mask_commit_authority",
        ):
            if getattr(self, name) is not False:
                raise PermissionError("sampled adaptive loop cannot grant authority")
        ranks = _strict_rank_ladder(self.rank_ladder, name="rank ladder")
        object.__setattr__(self, "rank_ladder", ranks)
        object.__setattr__(
            self,
            "rank_invariant_raw_prefix_hash",
            _identity(
                self.rank_invariant_raw_prefix_hash,
                name="rank-invariant raw-prefix hash",
            ),
        )
        object.__setattr__(
            self,
            "source_replay_axis_binding_hash",
            _identity(
                self.source_replay_axis_binding_hash,
                name="source replay-axis binding hash",
            ),
        )
        object.__setattr__(
            self,
            "source_fqr_policy_hash",
            _identity(
                self.source_fqr_policy_hash,
                name="source FQR adaptive-rank policy hash",
            ),
        )
        status = _identity(self.status, name="adaptive loop status")
        if status not in ("stable", "no_c_refusal"):
            raise ValueError("adaptive loop status is not canonical")
        object.__setattr__(self, "status", status)
        executions = tuple(self.executions)
        stability = tuple(self.stability_receipts)
        checkpoints = tuple(self.rank_checkpoint_receipts)
        checkpoint_callbacks = tuple(self.checkpoint_callback_receipts)
        eviction_callbacks = tuple(self.eviction_callback_receipts)
        if (
            not executions
            or len(stability) != len(executions)
            or len(checkpoints) != len(executions)
            or len(checkpoint_callbacks) != len(executions)
            or len(eviction_callbacks) != len(executions)
        ):
            raise ValueError("adaptive loop receipt has an incomplete rank axis")
        if any(
            type(value) is not V1SampledCompositionOneRankExecutionReceipt
            for value in executions
        ):
            raise TypeError("adaptive loop executions must be exact receipts")
        if any(
            type(value) is not V1SampledCompositionRankStabilityReceipt
            for value in stability
        ):
            raise TypeError("adaptive loop stability axis must be exact receipts")
        if any(
            type(value) is not V1SampledCompositionRankCheckpointReceipt
            for value in checkpoints
        ):
            raise TypeError("adaptive loop checkpoints must be exact receipts")
        if any(
            type(value) is not V1SampledCompositionRankCallbackReceipt
            for value in checkpoint_callbacks + eviction_callbacks
        ):
            raise TypeError("adaptive loop callbacks must be exact receipts")
        execution_ranks = tuple(value.physical_fd_rank for value in executions)
        if execution_ranks != ranks[: len(execution_ranks)]:
            raise ValueError("adaptive loop did not execute a contiguous rank prefix")
        if any(value.is_stable for value in stability[:-1]):
            raise ValueError("adaptive loop continued after first stable FQR source")
        previous_checkpoint_hash: str | None = None
        for execution, rank_stability, checkpoint, checkpoint_cb, eviction_cb in zip(
            executions,
            stability,
            checkpoints,
            checkpoint_callbacks,
            eviction_callbacks,
            strict=True,
        ):
            if (
                execution.rank_invariant_raw_prefix_hash
                != self.rank_invariant_raw_prefix_hash
                or rank_stability.rank_invariant_raw_prefix_hash
                != self.rank_invariant_raw_prefix_hash
                or checkpoint.rank_invariant_raw_prefix_hash
                != self.rank_invariant_raw_prefix_hash
                or checkpoint_cb.rank_invariant_raw_prefix_hash
                != self.rank_invariant_raw_prefix_hash
                or eviction_cb.rank_invariant_raw_prefix_hash
                != self.rank_invariant_raw_prefix_hash
                or rank_stability.source_replay_axis_binding_hash
                != self.source_replay_axis_binding_hash
                or checkpoint.source_replay_axis_binding_hash
                != self.source_replay_axis_binding_hash
            ):
                raise ValueError("adaptive loop raw-prefix drift detected")
            if (
                rank_stability.source_fqr_policy_hash != self.source_fqr_policy_hash
                or checkpoint.source_fqr_policy_hash != self.source_fqr_policy_hash
            ):
                raise ValueError("adaptive loop FQR policy drift detected")
            if (
                rank_stability.physical_fd_rank != execution.physical_fd_rank
                or rank_stability.one_rank_execution_receipt_hash
                != execution.receipt_hash
                or checkpoint.physical_fd_rank != execution.physical_fd_rank
                or checkpoint.transaction_hash != execution.transaction_hash
                or checkpoint.rank_local_cache_keys
                != execution.d_replay.rank_local_cache_keys
                or checkpoint.rank_local_cache_content_hashes
                != execution.d_replay.rank_local_cache_content_hashes
                or checkpoint.one_rank_execution_receipt_hash != execution.receipt_hash
                or checkpoint.stability_receipt_hash != rank_stability.receipt_hash
                or checkpoint.source_fqr_outcome_hash
                != rank_stability.source_fqr_outcome_hash
                or checkpoint.previous_checkpoint_hash != previous_checkpoint_hash
            ):
                raise ValueError("adaptive loop checkpoint/restart binding drifted")
            for callback, expected_kind in (
                (checkpoint_cb, "checkpoint"),
                (eviction_cb, "eviction"),
            ):
                if (
                    callback.callback_kind != expected_kind
                    or callback.source_checkpoint_receipt_hash
                    != checkpoint.checkpoint_hash
                    or callback.physical_fd_rank != execution.physical_fd_rank
                    or callback.rank_local_cache_keys
                    != execution.d_replay.rank_local_cache_keys
                    or callback.rank_local_cache_content_hashes
                    != execution.d_replay.rank_local_cache_content_hashes
                    or callback.observed_live_geometry_hashes
                    != checkpoint.live_geometry_hashes
                    or callback.restart_replay_hash != checkpoint.restart_replay_hash
                ):
                    raise ValueError("adaptive loop callback receipt drifted")
            previous_checkpoint_hash = checkpoint.checkpoint_hash
        final_execution = executions[-1]
        final_stability = stability[-1]
        selected = self.selected_physical_fd_rank
        refusal = self.no_c_refusal
        if status == "stable":
            if not final_stability.is_stable:
                raise ValueError("stable adaptive loop lacks stable FQR source")
            selected_rank = _positive_int(
                selected,
                name="selected physical FD rank",
            )
            if selected_rank != final_execution.physical_fd_rank or refusal is not None:
                raise ValueError("stable adaptive loop selected the wrong rank")
            object.__setattr__(self, "selected_physical_fd_rank", selected_rank)
        else:
            if final_stability.is_stable:
                raise ValueError("no-C refusal cannot wrap a stable FQR source")
            if selected is not None:
                raise ValueError("no-C refusal cannot select a physical rank")
            if final_execution.physical_fd_rank != ranks[-1]:
                raise ValueError("no-C refusal must exhaust the admitted rank ladder")
            if type(refusal) is not V1SampledCompositionNoCRefusalReceipt:
                raise TypeError("no-C adaptive loop needs a typed refusal receipt")
            if (
                refusal.maximum_physical_fd_rank != final_execution.physical_fd_rank
                or refusal.rank_invariant_raw_prefix_hash
                != self.rank_invariant_raw_prefix_hash
                or refusal.last_one_rank_execution_receipt_hash
                != final_execution.receipt_hash
                or refusal.last_stability_receipt_hash != final_stability.receipt_hash
                or refusal.source_fqr_policy_hash != self.source_fqr_policy_hash
                or refusal.source_fqr_no_c_refusal_hash
                != final_stability.source_fqr_outcome_hash
                or refusal.source_fqr_refusal_code
                is not V1FQRAdaptiveRankRefusalCode.DECISION_FAMILY_AMBIGUOUS
            ):
                raise ValueError("adaptive loop no-C refusal drifted from max rank")
        resumed = _optional_identity(
            self.resumed_from_checkpoint_hash,
            name="resumed-from checkpoint hash",
        )
        if self.restart_checkpoint_verified is not (resumed is not None):
            raise ValueError("adaptive loop restart verification state is inconsistent")
        if resumed is not None and resumed not in {
            value.checkpoint_hash for value in checkpoints
        }:
            raise ValueError("adaptive loop did not replay its restart checkpoint")
        object.__setattr__(self, "resumed_from_checkpoint_hash", resumed)
        object.__setattr__(self, "executions", executions)
        object.__setattr__(self, "stability_receipts", stability)
        object.__setattr__(self, "rank_checkpoint_receipts", checkpoints)
        object.__setattr__(
            self,
            "checkpoint_callback_receipts",
            checkpoint_callbacks,
        )
        object.__setattr__(self, "eviction_callback_receipts", eviction_callbacks)

    @property
    def source_fqr_outcome_hash(self) -> str:
        if self.status == "stable":
            return self.stability_receipts[-1].source_fqr_outcome_hash
        assert self.no_c_refusal is not None
        return self.no_c_refusal.source_fqr_no_c_refusal_hash

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "rank_ladder": self.rank_ladder,
                "rank_invariant_raw_prefix_hash": self.rank_invariant_raw_prefix_hash,
                "source_replay_axis_binding_hash": (
                    self.source_replay_axis_binding_hash
                ),
                "source_fqr_policy_hash": self.source_fqr_policy_hash,
                "status": self.status,
                "execution_receipt_hashes": tuple(
                    execution.receipt_hash for execution in self.executions
                ),
                "stability_receipt_hashes": tuple(
                    stability.receipt_hash for stability in self.stability_receipts
                ),
                "checkpoint_hashes": tuple(
                    checkpoint.checkpoint_hash
                    for checkpoint in self.rank_checkpoint_receipts
                ),
                "checkpoint_callback_hashes": tuple(
                    callback.callback_hash
                    for callback in self.checkpoint_callback_receipts
                ),
                "eviction_callback_hashes": tuple(
                    callback.callback_hash
                    for callback in self.eviction_callback_receipts
                ),
                "selected_physical_fd_rank": self.selected_physical_fd_rank,
                "no_c_refusal_hash": (
                    None
                    if self.no_c_refusal is None
                    else self.no_c_refusal.refusal_hash
                ),
                "source_fqr_outcome_hash": self.source_fqr_outcome_hash,
                "resumed_from_checkpoint_hash": self.resumed_from_checkpoint_hash,
                "restart_checkpoint_verified": self.restart_checkpoint_verified,
                "authority_scope": self.authority_scope,
                "C_authority": self.C_authority,
                "calibration_authority": self.calibration_authority,
                "oracle_authority": self.oracle_authority,
                "mask_commit_authority": self.mask_commit_authority,
            },
            prefix=_ADAPTIVE_LOOP_PREFIX,
        )


def build_v1_sampled_composition_physical_fd_design(
    *,
    physical_fd_rank: int,
    maximum_identifiable_fd_rank: int,
    fd_design_hash: str,
    fd_design_semantics: str = "rank_explicit_physical_fd_design_v1",
) -> V1SampledCompositionPhysicalFDDesign:
    return V1SampledCompositionPhysicalFDDesign(
        physical_fd_rank=physical_fd_rank,
        maximum_identifiable_fd_rank=maximum_identifiable_fd_rank,
        fd_design_hash=fd_design_hash,
        fd_design_semantics=fd_design_semantics,
    )


def build_v1_sampled_composition_registered_d_transaction(
    *,
    design: V1SampledCompositionPhysicalFDDesign,
    rank_invariant_raw_prefix_coordinate: tuple[object, ...],
    registered_look_count: int,
    literal_j_source_receipt_hash: str,
    source_replay_axis_binding_hash: str | None = None,
) -> V1SampledCompositionRegisteredDTransaction:
    return V1SampledCompositionRegisteredDTransaction(
        design=design,
        rank_invariant_raw_prefix_coordinate=rank_invariant_raw_prefix_coordinate,
        registered_look_count=registered_look_count,
        literal_j_source_receipt_hash=literal_j_source_receipt_hash,
        source_replay_axis_binding_hash=source_replay_axis_binding_hash,
    )


def build_v1_sampled_composition_registered_d_prefix_binding(
    *,
    rank_invariant_raw_prefix_coordinate: tuple[object, ...],
    registered_look_count: int,
    literal_j_source_receipt_hash: str,
    current_mask_hash: str,
    route_axis_hash: str,
    route_count: int,
    proposal_generation: int,
    cumulative_stratum_counts: tuple[int, ...],
    D_context_axis_hash: str,
    D_draw_axis_hash: str,
    D_source_measurement_axis_hash: str,
    D_weight_hash: str,
) -> V1SampledCompositionRegisteredDPrefixBinding:
    return V1SampledCompositionRegisteredDPrefixBinding(
        rank_invariant_raw_prefix_coordinate=rank_invariant_raw_prefix_coordinate,
        registered_look_count=registered_look_count,
        literal_j_source_receipt_hash=literal_j_source_receipt_hash,
        current_mask_hash=current_mask_hash,
        route_axis_hash=route_axis_hash,
        route_count=route_count,
        proposal_generation=proposal_generation,
        cumulative_stratum_counts=cumulative_stratum_counts,
        D_context_axis_hash=D_context_axis_hash,
        D_draw_axis_hash=D_draw_axis_hash,
        D_source_measurement_axis_hash=D_source_measurement_axis_hash,
        D_weight_hash=D_weight_hash,
    )


def _read_field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value[name]
    return getattr(value, name)


def _coerce_d_replay_receipt(
    raw: object,
    *,
    transaction: V1SampledCompositionRegisteredDTransaction,
) -> V1SampledCompositionDReplayReceipt:
    if type(raw) is V1SampledCompositionDReplayReceipt:
        replay = raw
    else:
        rank_hash = _identity(
            _read_field(raw, "rank_invariant_raw_prefix_hash"),
            name="D replay rank-invariant raw-prefix hash",
        )
        if rank_hash != transaction.rank_invariant_raw_prefix_hash:
            raise ValueError("D replay used another rank-invariant raw prefix")
        uncertainty = V1SampledCompositionFDUncertaintyReceipt(
            physical_fd_rank=int(_read_field(raw, "physical_fd_rank")),
            rank_invariant_raw_prefix_hash=rank_hash,
            uncertainty_components=_immutable_payload_mapping(
                _read_field(raw, "fd_uncertainty_components"),  # type: ignore[arg-type]
                name="FD uncertainty components",
            ),
        )
        replay = V1SampledCompositionDReplayReceipt(
            transaction_hash=_identity(
                _read_field(raw, "transaction_hash"),
                name="D replay transaction hash",
            ),
            physical_fd_rank=int(_read_field(raw, "physical_fd_rank")),
            rank_invariant_raw_prefix_hash=rank_hash,
            replayed_look_indices=tuple(
                int(value)
                for value in _read_field(raw, "replayed_look_indices")  # type: ignore[union-attr]
            ),
            rank_local_cache_keys=tuple(
                str(value)
                for value in _read_field(raw, "rank_local_cache_keys")  # type: ignore[union-attr]
            ),
            rank_local_cache_content_hashes=tuple(
                str(value)
                for value in _read_field(  # type: ignore[union-attr]
                    raw,
                    "rank_local_cache_content_hashes",
                )
            ),
            fd_design_hash=_identity(
                _read_field(raw, "fd_design_hash"),
                name="FD design hash",
            ),
            fd_snapshot_hash=_identity(
                _read_field(raw, "fd_snapshot_hash"),
                name="FD snapshot hash",
            ),
            fd_uncertainty=uncertainty,
            literal_j_source_receipt_hash=_identity(
                _read_field(raw, "literal_j_source_receipt_hash"),
                name="literal-J source receipt hash",
            ),
        )
    if replay.transaction_hash != transaction.transaction_hash:
        raise ValueError("D replay receipt names another transaction")
    if replay.physical_fd_rank != transaction.physical_fd_rank:
        raise ValueError("D replay receipt names another physical FD rank")
    if (
        replay.rank_invariant_raw_prefix_hash
        != transaction.rank_invariant_raw_prefix_hash
    ):
        raise ValueError("D replay receipt names another raw-prefix coordinate")
    if replay.replayed_look_indices != transaction.replay_look_indices:
        raise ValueError("D replay did not replay the registered prefix from look 1")
    if replay.rank_local_cache_keys != transaction.rank_local_cache_keys:
        raise ValueError("D replay did not use rank-local cache keys")
    if replay.fd_design_hash != transaction.fd_design_hash:
        raise ValueError("D replay used another FD design")
    if (
        replay.literal_j_source_receipt_hash
        != transaction.literal_j_source_receipt_hash
    ):
        raise ValueError("D replay literal-J source receipt differs")
    return replay


def _coerce_branch_receipt(
    raw: object, *, branch_id: KKTBranchId
) -> V1SampledCompositionKKTBranchReceipt:
    if type(raw) is V1SampledCompositionKKTBranchReceipt:
        branch = raw
    else:
        rungs = tuple(
            (
                rung
                if type(rung) is V1SampledCompositionHardRung
                else V1SampledCompositionHardRung(
                    branch_id=_branch_id(_read_field(rung, "branch_id")),
                    rung_id=_identity(_read_field(rung, "rung_id"), name="rung ID"),
                    hard_mask_hash=_identity(
                        _read_field(rung, "hard_mask_hash"),
                        name="hard mask hash",
                    ),
                    registered_resource_count=int(
                        _read_field(rung, "registered_resource_count")
                    ),
                    packed_action_hash=_identity(
                        _read_field(rung, "packed_action_hash"),
                        name="packed action hash",
                    ),
                    added_physical_route_indices=tuple(
                        int(value)
                        for value in _read_field(  # type: ignore[union-attr]
                            rung,
                            "added_physical_route_indices",
                        )
                    ),
                    rounding_prefix_hash=_identity(
                        _read_field(rung, "rounding_prefix_hash"),
                        name="rounding prefix hash",
                    ),
                    parent_rung_hash=_optional_identity(
                        _read_optional_field(rung, "parent_rung_hash"),
                        name="parent rung hash",
                    ),
                )
            )
            for rung in _read_field(raw, "hard_rungs")  # type: ignore[union-attr]
        )
        branch = V1SampledCompositionKKTBranchReceipt(
            branch_id=_branch_id(_read_field(raw, "branch_id")),
            relaxed_solution_hash=_identity(
                _read_field(raw, "relaxed_solution_hash"),
                name="relaxed solution hash",
            ),
            kkt_optimality_receipt_hash=_identity(
                _read_field(raw, "kkt_optimality_receipt_hash"),
                name="KKT optimality receipt hash",
            ),
            current_mask_hash=_identity(
                _read_field(raw, "current_mask_hash"),
                name="current mask hash",
            ),
            route_axis_hash=_identity(
                _read_field(raw, "route_axis_hash"),
                name="route axis hash",
            ),
            route_count=int(_read_field(raw, "route_count")),
            deterministic_rounding_policy_hash=_identity(
                _read_field(raw, "deterministic_rounding_policy_hash"),
                name="rounding policy hash",
            ),
            rounding_key_axis_hash=_identity(
                _read_field(raw, "rounding_key_axis_hash"),
                name="rounding key axis hash",
            ),
            legality_receipt_hash=_identity(
                _read_field(raw, "legality_receipt_hash"),
                name="legality receipt hash",
            ),
            source_fqr_hard_ladder_hash=_identity(
                _read_field(raw, "source_fqr_hard_ladder_hash"),
                name="source FQR hard-ladder hash",
            ),
            source_fqr_branch_decision_receipt_hash=_identity(
                _read_field(raw, "source_fqr_branch_decision_receipt_hash"),
                name="source FQR branch decision receipt hash",
            ),
            hard_rungs=rungs,
        )
    if branch.branch_id != branch_id:
        raise ValueError("KKT branch executor returned another branch")
    return branch


def _coerce_fq_replay(
    raw: object,
    *,
    branch: V1SampledCompositionKKTBranchReceipt,
    rung: V1SampledCompositionHardRung,
    d_replay: V1SampledCompositionDReplayReceipt,
) -> V1SampledCompositionHardFQReplayReceipt:
    if type(raw) is V1SampledCompositionHardFQReplayReceipt:
        replay = raw
    else:
        replay = V1SampledCompositionHardFQReplayReceipt(
            branch_id=_branch_id(_read_field(raw, "branch_id")),
            rung_id=_identity(_read_field(raw, "rung_id"), name="rung ID"),
            hard_mask_hash=_identity(
                _read_field(raw, "hard_mask_hash"),
                name="hard mask hash",
            ),
            source_rung_hash=_identity(
                _read_field(raw, "source_rung_hash"),
                name="source rung hash",
            ),
            source_branch_receipt_hash=_identity(
                _read_field(raw, "source_branch_receipt_hash"),
                name="source branch receipt hash",
            ),
            source_d_replay_receipt_hash=_identity(
                _read_field(raw, "source_d_replay_receipt_hash"),
                name="source D replay receipt hash",
            ),
            registered_resource_count=int(
                _read_field(raw, "registered_resource_count")
            ),
            f_upper=float(_read_field(raw, "f_upper")),
            q_upper=float(_read_field(raw, "q_upper")),
        )
    if (
        replay.branch_id != branch.branch_id
        or replay.rung_id != rung.rung_id
        or replay.hard_mask_hash != rung.hard_mask_hash
        or replay.source_rung_hash != rung.rung_hash
        or replay.source_branch_receipt_hash != branch.receipt_hash
        or replay.source_d_replay_receipt_hash != d_replay.receipt_hash
        or replay.registered_resource_count != rung.registered_resource_count
    ):
        raise ValueError("hard F/Q replay receipt is not bound to its rung")
    return replay


def _read_optional_field(value: object, name: str) -> object | None:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _coerce_rank_stability_receipt(
    raw: object,
    *,
    one_rank_execution: V1SampledCompositionOneRankExecutionReceipt,
    executed_rank_prefix: tuple[int, ...],
    source_fqr_policy: V1FQRAdaptiveRankPolicy,
    prefix_binding: V1SampledCompositionRegisteredDPrefixBinding,
) -> V1SampledCompositionRankStabilityReceipt:
    if type(source_fqr_policy) is not V1FQRAdaptiveRankPolicy:
        raise TypeError("sampled rank adapter requires the exact FQR policy")
    if type(prefix_binding) is not V1SampledCompositionRegisteredDPrefixBinding:
        raise TypeError("sampled rank adapter requires the exact D-prefix binding")
    if type(raw) is not V1SampledCompositionSourceFQRRankEvaluation:
        raise TypeError(
            "rank evaluator must return an exact source FQR ledger/outcome transport"
        )
    evaluation = raw
    ledger = evaluation.replay_ledger
    policy_hash = source_fqr_policy.assert_integrity()
    if ledger.policy.policy_hash != policy_hash:
        raise ValueError("source FQR replay ledger belongs to another policy")
    records = ledger.records
    record_ranks = tuple(value.rank for value in records)
    if record_ranks != executed_rank_prefix:
        raise ValueError("source FQR ledger does not cover this exact execution prefix")
    for record in records:
        prefix_binding.assert_matches_record(record)
    frozen = records[-1]
    if (
        frozen.rank != one_rank_execution.physical_fd_rank
        or frozen.fd_design_hash != one_rank_execution.d_replay.fd_design_hash
        or frozen.fd_snapshot_hash != one_rank_execution.d_replay.fd_snapshot_hash
    ):
        raise ValueError(
            "source FQR frozen replay does not bind this physical execution"
        )
    source_branches = {
        value.branch: value for value in frozen.decision_certificate.branch_decisions
    }
    for branch in one_rank_execution.branch_receipts:
        source = source_branches[branch.branch_id]
        if (
            branch.source_fqr_hard_ladder_hash != source.hard_ladder_hash
            or branch.source_fqr_branch_decision_receipt_hash
            != source.decision_receipt_hash
            or tuple(rung.packed_action_hash for rung in branch.hard_rungs)
            != source.ordered_packed_action_hashes
        ):
            raise ValueError("sampled hard ladder drifted from its source FQR branch")

    # Re-run the sole canonical authority and require byte-identical outcome.
    try:
        canonical_outcome: (
            V1FQRAdaptiveRankStabilityCertificate | V1FQRAdaptiveRankNoCRefusal
        ) = certify_v1_fqr_adaptive_rank_stability(
            policy=source_fqr_policy,
            ledger=ledger,
        )
    except V1FQRAdaptiveRankRefusal as refusal:
        canonical_outcome = refusal.receipt
    if type(canonical_outcome) is not type(evaluation.outcome):
        raise ValueError("source FQR outcome differs from canonical recertification")
    canonical_hash = (
        canonical_outcome.certificate_hash
        if type(canonical_outcome) is V1FQRAdaptiveRankStabilityCertificate
        else canonical_outcome.refusal_hash
    )
    if canonical_hash != evaluation.outcome_hash:
        raise ValueError(
            "source FQR outcome hash differs from canonical recertification"
        )

    common = {
        "physical_fd_rank": frozen.rank,
        "rank_invariant_raw_prefix_hash": (
            one_rank_execution.rank_invariant_raw_prefix_hash
        ),
        "source_replay_axis_binding_hash": prefix_binding.binding_hash,
        "one_rank_execution_receipt_hash": one_rank_execution.receipt_hash,
        "source_fqr_policy_hash": policy_hash,
        "source_fqr_replay_ledger_hash": ledger.ledger_hash,
        "source_fqr_executed_rank_prefix": record_ranks,
        "source_fqr_replay_record_hashes": tuple(
            value.record_hash for value in records
        ),
        "source_fqr_frozen_replay_record_hash": frozen.record_hash,
        "current_mask_hash": frozen.current_mask_hash,
        "route_axis_hash": frozen.route_axis_hash,
        "route_count": frozen.route_count,
        "proposal_generation": frozen.proposal_generation,
        "source_fqr_rank_local_decision_certificate_hash": (
            frozen.decision_certificate.certificate_hash
        ),
    }
    if type(canonical_outcome) is V1FQRAdaptiveRankStabilityCertificate:
        return V1SampledCompositionRankStabilityReceipt(
            **common,
            decision="decision_stable",
            source_fqr_stability_certificate_hash=canonical_outcome.certificate_hash,
        )
    code = canonical_outcome.code
    if code not in (
        V1FQRAdaptiveRankRefusalCode.RANK_PROMOTION_REQUIRED,
        V1FQRAdaptiveRankRefusalCode.DECISION_FAMILY_AMBIGUOUS,
    ):
        raise V1FQRAdaptiveRankRefusal(canonical_outcome)
    return V1SampledCompositionRankStabilityReceipt(
        **common,
        decision=(
            "promote"
            if code is V1FQRAdaptiveRankRefusalCode.RANK_PROMOTION_REQUIRED
            else "ambiguous"
        ),
        source_fqr_refusal_hash=canonical_outcome.refusal_hash,
        source_fqr_refusal_code=code,
    )


def _coerce_rank_callback_receipt(
    raw: object,
    *,
    checkpoint: V1SampledCompositionRankCheckpointReceipt,
    callback_kind: RankCallbackKind,
) -> V1SampledCompositionRankCallbackReceipt:
    if type(raw) is not V1SampledCompositionRankCallbackReceipt:
        raise TypeError("rank callback must return its exact typed receipt")
    receipt = raw
    if (
        receipt.callback_kind != callback_kind
        or receipt.source_checkpoint_receipt_hash != checkpoint.checkpoint_hash
        or receipt.physical_fd_rank != checkpoint.physical_fd_rank
        or receipt.rank_invariant_raw_prefix_hash
        != checkpoint.rank_invariant_raw_prefix_hash
        or receipt.rank_local_cache_keys != checkpoint.rank_local_cache_keys
        or receipt.rank_local_cache_content_hashes
        != checkpoint.rank_local_cache_content_hashes
        or receipt.observed_live_geometry_hashes != checkpoint.live_geometry_hashes
        or receipt.restart_replay_hash != checkpoint.restart_replay_hash
    ):
        raise ValueError("rank callback receipt drifted from checkpoint")
    if callback_kind == "checkpoint":
        if (
            receipt.persisted_checkpoint_hash != checkpoint.checkpoint_hash
            or receipt.durable_payload_hash != checkpoint.restart_replay_hash
        ):
            raise ValueError(
                "checkpoint callback did not persist the exact restart state"
            )
    elif (
        receipt.evicted_geometry_hashes != checkpoint.live_geometry_hashes
        or receipt.remaining_rank_local_geometry_hashes
    ):
        raise ValueError("eviction callback left rank-local geometry live")
    return receipt


def execute_v1_sampled_composition_one_physical_rank(
    *,
    transaction: V1SampledCompositionRegisteredDTransaction,
    registered_d_replayer: Callable[
        [V1SampledCompositionRegisteredDTransaction],
        object,
    ],
    kkt_branch_executor: Callable[
        [KKTBranchId, V1SampledCompositionDReplayReceipt],
        object,
    ],
    hard_fq_replayer: Callable[
        [
            V1SampledCompositionKKTBranchReceipt,
            V1SampledCompositionHardRung,
            V1SampledCompositionDReplayReceipt,
        ],
        object,
    ],
) -> V1SampledCompositionOneRankExecutionReceipt:
    """Execute one explicit physical rank and replay every hard F/Q rung."""

    if type(transaction) is not V1SampledCompositionRegisteredDTransaction:
        raise TypeError("one-rank sampled composition needs a registered transaction")
    d_replay = _coerce_d_replay_receipt(
        registered_d_replayer(transaction),
        transaction=transaction,
    )
    branches = tuple(
        _coerce_branch_receipt(
            kkt_branch_executor(branch_id, d_replay),
            branch_id=branch_id,
        )
        for branch_id in REQUIRED_KKT_BRANCHES
    )
    hard_replays: list[V1SampledCompositionHardFQReplayReceipt] = []
    for branch in branches:
        for rung in branch.hard_rungs:
            hard_replays.append(
                _coerce_fq_replay(
                    hard_fq_replayer(branch, rung, d_replay),
                    branch=branch,
                    rung=rung,
                    d_replay=d_replay,
                )
            )
    return V1SampledCompositionOneRankExecutionReceipt(
        transaction_hash=transaction.transaction_hash,
        physical_fd_rank=transaction.physical_fd_rank,
        rank_invariant_raw_prefix_hash=transaction.rank_invariant_raw_prefix_hash,
        d_replay=d_replay,
        branch_receipts=branches,
        hard_fq_replays=tuple(hard_replays),
        literal_j_source_receipt_hash=transaction.literal_j_source_receipt_hash,
    )


def execute_v1_sampled_composition_adaptive_physical_rank_loop(
    *,
    rank_ladder: Sequence[int],
    maximum_identifiable_fd_rank: int,
    fd_design_hash_for_rank: Callable[[int], str],
    registered_d_prefix_binding: V1SampledCompositionRegisteredDPrefixBinding,
    source_fqr_adaptive_rank_policy: V1FQRAdaptiveRankPolicy,
    registered_d_replayer: Callable[
        [V1SampledCompositionRegisteredDTransaction],
        object,
    ],
    kkt_branch_executor: Callable[
        [KKTBranchId, V1SampledCompositionDReplayReceipt],
        object,
    ],
    hard_fq_replayer: Callable[
        [
            V1SampledCompositionKKTBranchReceipt,
            V1SampledCompositionHardRung,
            V1SampledCompositionDReplayReceipt,
        ],
        object,
    ],
    rank_stability_evaluator: Callable[
        [V1SampledCompositionOneRankExecutionReceipt],
        V1SampledCompositionSourceFQRRankEvaluation,
    ],
    checkpoint_callback: Callable[
        [V1SampledCompositionRankCheckpointReceipt],
        V1SampledCompositionRankCallbackReceipt,
    ],
    eviction_callback: Callable[
        [V1SampledCompositionRankCheckpointReceipt],
        V1SampledCompositionRankCallbackReceipt,
    ],
    resume_checkpoint: V1SampledCompositionRankCheckpointReceipt | None = None,
) -> V1SampledCompositionAdaptiveRankLoopReceipt:
    """Replay ranks contiguously until existing FQR authority is stable/refused."""

    if type(source_fqr_adaptive_rank_policy) is not V1FQRAdaptiveRankPolicy:
        raise TypeError("sampled adaptive loop requires the exact FQR rank policy")
    if type(registered_d_prefix_binding) is not (
        V1SampledCompositionRegisteredDPrefixBinding
    ):
        raise TypeError("sampled adaptive loop requires the exact D-prefix binding")
    ladder = _strict_rank_ladder(rank_ladder, name="adaptive physical-rank ladder")
    maximum = _positive_int(
        maximum_identifiable_fd_rank,
        name="maximum identifiable FD rank",
    )
    if ladder != source_fqr_adaptive_rank_policy.admitted_rank_ladder:
        raise ValueError("sampled adaptive ladder differs from source FQR policy")
    if maximum != source_fqr_adaptive_rank_policy.requested_rank_ladder[-1]:
        raise ValueError("identifiable rank differs from the frozen source FQR policy")
    if (
        registered_d_prefix_binding.route_count
        != source_fqr_adaptive_rank_policy.route_count
    ):
        raise ValueError("registered-D route count differs from source FQR policy")
    source_fqr_policy_hash = source_fqr_adaptive_rank_policy.assert_integrity()
    if resume_checkpoint is not None:
        if type(resume_checkpoint) is not V1SampledCompositionRankCheckpointReceipt:
            raise TypeError("sampled restart requires an exact checkpoint receipt")
        if (
            resume_checkpoint.physical_fd_rank not in ladder
            or resume_checkpoint.source_fqr_policy_hash != source_fqr_policy_hash
            or resume_checkpoint.rank_invariant_raw_prefix_hash
            != registered_d_prefix_binding.rank_invariant_raw_prefix_hash
            or resume_checkpoint.source_replay_axis_binding_hash
            != registered_d_prefix_binding.binding_hash
        ):
            raise ValueError("resume checkpoint belongs to another adaptive replay")

    executions: list[V1SampledCompositionOneRankExecutionReceipt] = []
    stability_receipts: list[V1SampledCompositionRankStabilityReceipt] = []
    checkpoints: list[V1SampledCompositionRankCheckpointReceipt] = []
    checkpoint_callbacks: list[V1SampledCompositionRankCallbackReceipt] = []
    eviction_callbacks: list[V1SampledCompositionRankCallbackReceipt] = []
    resume_verified = False

    rank_invariant_hash: str | None = None
    for index, rank in enumerate(ladder):
        design = build_v1_sampled_composition_physical_fd_design(
            physical_fd_rank=rank,
            maximum_identifiable_fd_rank=maximum,
            fd_design_hash=_identity(
                fd_design_hash_for_rank(rank),
                name="rank-explicit FD design hash",
            ),
        )
        transaction = build_v1_sampled_composition_registered_d_transaction(
            design=design,
            rank_invariant_raw_prefix_coordinate=(
                registered_d_prefix_binding.rank_invariant_raw_prefix_coordinate
            ),
            registered_look_count=registered_d_prefix_binding.registered_look_count,
            literal_j_source_receipt_hash=(
                registered_d_prefix_binding.literal_j_source_receipt_hash
            ),
            source_replay_axis_binding_hash=registered_d_prefix_binding.binding_hash,
        )
        if rank_invariant_hash is None:
            rank_invariant_hash = transaction.rank_invariant_raw_prefix_hash
        elif transaction.rank_invariant_raw_prefix_hash != rank_invariant_hash:
            raise ValueError("adaptive loop rebuilt a different raw-D prefix")

        execution = execute_v1_sampled_composition_one_physical_rank(
            transaction=transaction,
            registered_d_replayer=registered_d_replayer,
            kkt_branch_executor=kkt_branch_executor,
            hard_fq_replayer=hard_fq_replayer,
        )
        for branch in execution.branch_receipts:
            if (
                branch.current_mask_hash
                != registered_d_prefix_binding.current_mask_hash
                or branch.route_axis_hash != registered_d_prefix_binding.route_axis_hash
                or branch.route_count != registered_d_prefix_binding.route_count
            ):
                raise ValueError(
                    "sampled KKT branch drifted from registered mask/route axes"
                )
        raw_stability = rank_stability_evaluator(execution)
        stability = _coerce_rank_stability_receipt(
            raw_stability,
            one_rank_execution=execution,
            executed_rank_prefix=tuple(
                value.physical_fd_rank for value in (*executions, execution)
            ),
            source_fqr_policy=source_fqr_adaptive_rank_policy,
            prefix_binding=registered_d_prefix_binding,
        )
        final_or_stable = stability.is_stable or index == len(ladder) - 1
        checkpoint = V1SampledCompositionRankCheckpointReceipt(
            event="rank_finalized" if final_or_stable else "rank_promoted",
            physical_fd_rank=execution.physical_fd_rank,
            rank_invariant_raw_prefix_hash=execution.rank_invariant_raw_prefix_hash,
            source_replay_axis_binding_hash=registered_d_prefix_binding.binding_hash,
            transaction_hash=execution.transaction_hash,
            rank_local_cache_keys=execution.d_replay.rank_local_cache_keys,
            rank_local_cache_content_hashes=(
                execution.d_replay.rank_local_cache_content_hashes
            ),
            live_geometry_hashes=(
                execution.d_replay.receipt_hash,
                execution.d_replay.fd_uncertainty.receipt_hash,
                *(value.receipt_hash for value in execution.branch_receipts),
                *(value.replay_hash for value in execution.hard_fq_replays),
            ),
            one_rank_execution_receipt_hash=execution.receipt_hash,
            stability_receipt_hash=stability.receipt_hash,
            source_fqr_policy_hash=source_fqr_policy_hash,
            source_fqr_outcome_hash=stability.source_fqr_outcome_hash,
            previous_checkpoint_hash=(
                None if not checkpoints else checkpoints[-1].checkpoint_hash
            ),
        )
        if (
            resume_checkpoint is not None
            and checkpoint.physical_fd_rank == resume_checkpoint.physical_fd_rank
        ):
            if checkpoint.checkpoint_hash != resume_checkpoint.checkpoint_hash:
                raise ValueError(
                    "restart replay is not hash-identical to its checkpoint"
                )
            resume_verified = True
        checkpoint_receipt = _coerce_rank_callback_receipt(
            checkpoint_callback(checkpoint),
            checkpoint=checkpoint,
            callback_kind="checkpoint",
        )
        eviction_receipt = _coerce_rank_callback_receipt(
            eviction_callback(checkpoint),
            checkpoint=checkpoint,
            callback_kind="eviction",
        )

        executions.append(execution)
        stability_receipts.append(stability)
        checkpoints.append(checkpoint)
        checkpoint_callbacks.append(checkpoint_receipt)
        eviction_callbacks.append(eviction_receipt)

        if stability.is_stable:
            if resume_checkpoint is not None and not resume_verified:
                raise ValueError("adaptive replay stopped before its resume checkpoint")
            return V1SampledCompositionAdaptiveRankLoopReceipt(
                rank_ladder=ladder,
                rank_invariant_raw_prefix_hash=execution.rank_invariant_raw_prefix_hash,
                source_replay_axis_binding_hash=registered_d_prefix_binding.binding_hash,
                source_fqr_policy_hash=source_fqr_policy_hash,
                status="stable",
                executions=tuple(executions),
                stability_receipts=tuple(stability_receipts),
                rank_checkpoint_receipts=tuple(checkpoints),
                checkpoint_callback_receipts=tuple(checkpoint_callbacks),
                eviction_callback_receipts=tuple(eviction_callbacks),
                selected_physical_fd_rank=execution.physical_fd_rank,
                resumed_from_checkpoint_hash=(
                    None
                    if resume_checkpoint is None
                    else resume_checkpoint.checkpoint_hash
                ),
                restart_checkpoint_verified=resume_verified,
            )
        if index == len(ladder) - 1:
            if stability.source_fqr_refusal_code is not (
                V1FQRAdaptiveRankRefusalCode.DECISION_FAMILY_AMBIGUOUS
            ):
                raise ValueError("max rank ended without canonical decision ambiguity")
            if resume_checkpoint is not None and not resume_verified:
                raise ValueError("adaptive replay stopped before its resume checkpoint")
            source_outcome = raw_stability.outcome
            assert type(source_outcome) is V1FQRAdaptiveRankNoCRefusal
            refusal = V1SampledCompositionNoCRefusalReceipt(
                maximum_physical_fd_rank=execution.physical_fd_rank,
                rank_invariant_raw_prefix_hash=execution.rank_invariant_raw_prefix_hash,
                last_one_rank_execution_receipt_hash=execution.receipt_hash,
                last_stability_receipt_hash=stability.receipt_hash,
                source_fqr_policy_hash=source_fqr_policy_hash,
                source_fqr_no_c_refusal_hash=stability.source_fqr_outcome_hash,
                source_fqr_refusal_code=source_outcome.code,
                reason=source_outcome.reason,
            )
            return V1SampledCompositionAdaptiveRankLoopReceipt(
                rank_ladder=ladder,
                rank_invariant_raw_prefix_hash=execution.rank_invariant_raw_prefix_hash,
                source_replay_axis_binding_hash=registered_d_prefix_binding.binding_hash,
                source_fqr_policy_hash=source_fqr_policy_hash,
                status="no_c_refusal",
                executions=tuple(executions),
                stability_receipts=tuple(stability_receipts),
                rank_checkpoint_receipts=tuple(checkpoints),
                checkpoint_callback_receipts=tuple(checkpoint_callbacks),
                eviction_callback_receipts=tuple(eviction_callbacks),
                no_c_refusal=refusal,
                resumed_from_checkpoint_hash=(
                    None
                    if resume_checkpoint is None
                    else resume_checkpoint.checkpoint_hash
                ),
                restart_checkpoint_verified=resume_verified,
            )

    raise AssertionError("adaptive rank loop must return from a nonempty ladder")


__all__ = [
    "DIAGONAL_L1_KKT_BRANCH",
    "REQUIRED_KKT_BRANCHES",
    "SAMPLED_COMPOSITION_ADAPTIVE_LOOP_AUTHORITY_SCOPE",
    "SAMPLED_COMPOSITION_ONE_RANK_AUTHORITY_SCOPE",
    "SPECTRAL_KKT_BRANCH",
    "V1SampledCompositionAdaptiveRankLoopReceipt",
    "V1SampledCompositionDReplayReceipt",
    "V1SampledCompositionFDUncertaintyReceipt",
    "V1SampledCompositionHardFQReplayReceipt",
    "V1SampledCompositionHardRung",
    "V1SampledCompositionKKTBranchReceipt",
    "V1SampledCompositionNoCRefusalReceipt",
    "V1SampledCompositionOneRankExecutionReceipt",
    "V1SampledCompositionPhysicalFDDesign",
    "V1SampledCompositionRankCallbackReceipt",
    "V1SampledCompositionRankCheckpointReceipt",
    "V1SampledCompositionRankStabilityReceipt",
    "V1SampledCompositionRegisteredDPrefixBinding",
    "V1SampledCompositionRegisteredDTransaction",
    "V1SampledCompositionSourceFQRRankEvaluation",
    "build_v1_sampled_composition_physical_fd_design",
    "build_v1_sampled_composition_registered_d_prefix_binding",
    "build_v1_sampled_composition_registered_d_transaction",
    "execute_v1_sampled_composition_adaptive_physical_rank_loop",
    "execute_v1_sampled_composition_one_physical_rank",
    "sampled_composition_hard_mask_hash",
    "sampled_composition_rounding_prefix_hash",
]
