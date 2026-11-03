"""Action-complete causal custody for literal-J population geometry.

Each :class:`V1LiteralJJointPopulationPreoutcomeSource` owns one physical
action.  A tuple of those owners is not, by itself, evidence that the complete
pre-registered action axis was frozen before residual outcomes.  This module
provides that missing aggregate owner.

The preoutcome issuer accepts only the complete ordered axis implied by the
shared joint direction registration.  It replays every singular owner while
its logical source is still unbound, proves ordinals ``1..A`` exactly once,
and seals the common D1/residual measure and chart axes.  The postsource binder
then accepts only the exact singular POST owners descended from those PRE
objects.  Neither phase projects rows, spends alpha, certifies D, opens C/K/O,
or commits a mask.

Packed actions remain ordinal-addressable and are materialised one at a time;
the aggregate never constructs an ``A x route`` bank.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
from threading import RLock
from typing import Any, Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_literal_j_joint_population_source import (
    V1LiteralJJointPopulationPreoutcomeSource,
    V1LiteralJJointPopulationSource,
)


LITERAL_J_JOINT_POPULATION_ACTION_AXIS_PREOUTCOME_SCHEMA: Final = (
    "partition_free_v1_literal_J_joint_population_action_axis_preoutcome_v1"
)
LITERAL_J_JOINT_POPULATION_ACTION_AXIS_SOURCE_SCHEMA: Final = (
    "partition_free_v1_literal_J_joint_population_action_axis_source_v1"
)
LITERAL_J_JOINT_POPULATION_ACTION_AXIS_AUTHORITY: Final = (
    "causal_complete_ordered_physical_action_axis_D1_residual_measure_source_"
    "custody_only_no_projection_confidence_FD_solver_D_C_K_O_commit_terminal_"
    "or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-literal-j-joint-population-action-axis-"
_PRE_RECEIPT_TOKEN: Final = object()
_PRE_OWNER_TOKEN: Final = object()
_POST_RECEIPT_TOKEN: Final = object()
_POST_OWNER_TOKEN: Final = object()


class V1LiteralJJointPopulationActionAxisRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    INCOMPLETE_ACTION_AXIS = "incomplete_action_axis"
    ACTION_ORDER_MISMATCH = "action_order_mismatch"
    COMMON_AXIS_MISMATCH = "common_axis_mismatch"
    SOURCE_DESCENT_MISMATCH = "source_descent_mismatch"
    PREOUTCOME_REQUIRED = "preoutcome_required"
    BIND_REPLAY = "bind_replay"
    STALE_SOURCE = "stale_source"
    EVICTED = "evicted"
    AUTHORITY_VIOLATION = "authority_violation"


class V1LiteralJJointPopulationActionAxisRefusal(RuntimeError):
    def __init__(
        self,
        code: V1LiteralJJointPopulationActionAxisRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1LiteralJJointPopulationActionAxisRefusalCode:
            raise TypeError("action-axis refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1LiteralJJointPopulationActionAxisRefusalCode,
    message: str,
) -> NoReturn:
    raise V1LiteralJJointPopulationActionAxisRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1LiteralJJointPopulationActionAxisRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty identity",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _refuse(
            V1LiteralJJointPopulationActionAxisRefusalCode.INVALID_TYPE,
            f"{name} must be one positive integer",
        )
    return int(value)


def _identity_axis(
    value: object,
    *,
    name: str,
    unique: bool = True,
) -> tuple[str, ...]:
    if type(value) is not tuple or not value:
        _refuse(
            V1LiteralJJointPopulationActionAxisRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty immutable tuple",
        )
    result = tuple(_identity(item, name=f"{name} member") for item in value)
    if unique and len(set(result)) != len(result):
        _refuse(
            V1LiteralJJointPopulationActionAxisRefusalCode.ACTION_ORDER_MISMATCH,
            f"{name} is not unique",
        )
    return result


def _authority_payload() -> dict[str, bool | str]:
    return {
        "population_projection_authority": False,
        "confidence_authority": False,
        "fd_replacement_authority": False,
        "solver_authority": False,
        "d_stability_authority": False,
        "c_k_o_authority": False,
        "mask_commit_authority": False,
        "terminal_or_pareto_authority": False,
        "authority_scope": LITERAL_J_JOINT_POPULATION_ACTION_AXIS_AUTHORITY,
    }


def _closed_authority(value: object) -> bool:
    return all(
        getattr(value, name, None) is expected
        if type(expected) is bool
        else getattr(value, name, None) == expected
        for name, expected in _authority_payload().items()
    )


_COMMON_FIELDS: Final = (
    "fanout_receipt_hash",
    "fanout_binding_axis_hash",
    "physical_action_axis_hash",
    "joint_preoutcome_registration_hash",
    "joint_direction_axis_hash",
    "joint_direction_ids",
    "d1_replay_receipt_hash",
    "d1_compact_source_receipt_hash",
    "d1_source_registration_hash",
    "d1_source_axis_hash",
    "d1_literal_j_shard_receipt_hashes",
    "d1_literal_j_content_sha256s",
    "d1_corrected_evidence_stream_source_hash",
    "d1_literal_j_stream_source_hash",
    "d1_same_capture_vjp_source_hash",
    "measure_design_hash",
    "measure_weight_receipt_hash",
    "model_identity_hash",
    "chart_identity_hash",
    "current_mask_hash",
    "proposal_generation",
    "route_axis_hash",
    "route_id_axis_hash",
    "route_count",
    "registered_d_prefix_hash",
    "registered_d_w2_prefix_hash",
    "shell_freeze_receipt_hash",
    "shell_indices",
    "shell_route_ids",
    "native_fiber_axis_receipt_hash",
    "native_fiber_measure_hash",
    "corrected_evidence_stream_source_hash",
    "literal_j_stream_source_hash",
    "same_capture_vjp_source_hash",
    "literal_j_stream_binding_hash",
    "literal_j_cell_hash",
    "literal_j_cell_scope_hash",
    "alpha_lineage_hash",
    "d1_registered_prefix_hash",
    "d1_pair_prefix_axis_hash",
    "d1_context_count",
    "d1_stratum_counts",
    "residual_sample_stratum_counts",
    "population_stratum_counts",
    "d1_context_weight_fraction",
    "residual_context_weight_fractions",
    "residual_stratum_mass_fractions",
    "residual_population_mass_fraction",
    "common_measure_denominator",
    "d1_context_quantum",
    "residual_context_quanta",
    "total_measure_quanta",
)


def _validate_pre_sources(
    sources: tuple[V1LiteralJJointPopulationPreoutcomeSource, ...],
) -> tuple[object, ...]:
    if (
        type(sources) is not tuple
        or not sources
        or any(
            type(source) is not V1LiteralJJointPopulationPreoutcomeSource
            for source in sources
        )
    ):
        _refuse(
            V1LiteralJJointPopulationActionAxisRefusalCode.INVALID_TYPE,
            "the PRE action axis requires a nonempty exact source tuple",
        )
    receipts = tuple(source.receipt for source in sources)
    for source in sources:
        try:
            source.assert_preoutcome_live()
        except Exception as error:
            _refuse(
                V1LiteralJJointPopulationActionAxisRefusalCode.PREOUTCOME_REQUIRED,
                "one singular action source was not live before outcomes: "
                f"{type(error).__name__}: {error}",
            )
    first = receipts[0]
    first_source = sources[0]
    shell_size = len(first.shell_indices)
    action_count = len(first.joint_direction_ids) - shell_size
    if action_count <= 0 or len(receipts) != action_count:
        _refuse(
            V1LiteralJJointPopulationActionAxisRefusalCode.INCOMPLETE_ACTION_AXIS,
            "singular sources do not cover the action count implied by the joint axis",
        )
    ordinals = tuple(receipt.physical_action_ordinal for receipt in receipts)
    if ordinals != tuple(range(1, action_count + 1)):
        _refuse(
            V1LiteralJJointPopulationActionAxisRefusalCode.ACTION_ORDER_MISMATCH,
            "physical actions must appear exactly once in canonical ordinal order",
        )
    for receipt in receipts:
        if any(
            getattr(receipt, name) != getattr(first, name)
            for name in _COMMON_FIELDS
        ):
            _refuse(
                V1LiteralJJointPopulationActionAxisRefusalCode.COMMON_AXIS_MISMATCH,
                "singular sources do not share one chart, D1/residual measure, shell, and joint axis",
            )
        expected_outside = first.joint_direction_ids[
            receipt.physical_action_ordinal - 1
        ]
        if receipt.joint_outside_direction_id != expected_outside:
            _refuse(
                V1LiteralJJointPopulationActionAxisRefusalCode.ACTION_ORDER_MISMATCH,
                "one singular source occupies the wrong joint outside-direction coordinate",
            )
    if any(
        source._logical_source is not first_source._logical_source
        or source._conditioned_d1 is not first_source._conditioned_d1
        or source._measure_design is not first_source._measure_design
        or source._measure_weight_receipt is not first_source._measure_weight_receipt
        or source._fanout is not first_source._fanout
        or source._joint is not first_source._joint
        for source in sources
    ):
        _refuse(
            V1LiteralJJointPopulationActionAxisRefusalCode.COMMON_AXIS_MISMATCH,
            "action members do not retain one exact shared logical look, D1 source, measure, fanout, and joint registration",
        )
    for name in (
        "destination_binding_hash",
        "residual_preoutcome_registration_hash",
        "physical_action_identity_hash",
        "physical_action_hash",
        "packed_action_hash",
    ):
        axis = tuple(getattr(receipt, name) for receipt in receipts)
        if len(set(axis)) != action_count:
            _refuse(
                V1LiteralJJointPopulationActionAxisRefusalCode.ACTION_ORDER_MISMATCH,
                f"{name} does not define one complete unique action axis",
            )
    return receipts


@dataclass(frozen=True, slots=True)
class V1LiteralJJointPopulationActionAxisPreoutcomeReceipt:
    singular_preoutcome_receipt_hashes: tuple[str, ...]
    logical_source_receipt_hashes: tuple[str, ...]
    destination_binding_hashes: tuple[str, ...]
    residual_preoutcome_registration_hashes: tuple[str, ...]
    residual_source_registration_hashes: tuple[str, ...]
    action_custody_hashes: tuple[str, ...]
    action_source_registration_hashes: tuple[str, ...]
    physical_action_identity_hashes: tuple[str, ...]
    physical_action_hashes: tuple[str, ...]
    packed_action_hashes: tuple[str, ...]
    resulting_mask_hashes: tuple[str, ...]
    removed_route_index_hashes: tuple[str, ...]
    removed_counts: tuple[int, ...]
    fanout_receipt_hash: str
    fanout_binding_axis_hash: str
    physical_action_axis_hash: str
    joint_preoutcome_registration_hash: str
    joint_direction_axis_hash: str
    joint_direction_ids: tuple[str, ...]
    d1_source_axis_hash: str
    d1_source_registration_hash: str
    d1_corrected_evidence_stream_source_hash: str
    d1_literal_j_stream_source_hash: str
    d1_same_capture_vjp_source_hash: str
    measure_design_hash: str
    measure_weight_receipt_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    route_axis_hash: str
    route_id_axis_hash: str
    literal_j_stream_source_hash: str
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    shell_freeze_receipt_hash: str
    parent_mask_hash: str
    shell_indices: tuple[int, ...]
    shell_route_ids: tuple[str, ...]
    alpha_lineage_hash: str
    action_count: int
    shell_size: int
    direction_count: int
    route_count: int
    proposal_generation: int
    d1_context_count: int
    d1_stratum_counts: tuple[int, int]
    residual_sample_stratum_counts: tuple[int, int]
    population_stratum_counts: tuple[int, int]
    d1_context_weight_fraction: tuple[int, int]
    residual_context_weight_fractions: tuple[tuple[int, int], tuple[int, int]]
    residual_stratum_mass_fractions: tuple[tuple[int, int], tuple[int, int]]
    residual_population_mass_fraction: tuple[int, int]
    common_measure_denominator: int
    d1_context_quantum: int
    residual_context_quanta: tuple[int, int]
    total_measure_quanta: int
    provider_outcomes_seen: bool = False
    logical_sources_bound_at_registration: bool = False
    complete_ordered_action_axis: bool = True
    one_action_materialized_at_a_time: bool = True
    action_route_bank_retained: bool = False
    array_free_durable_receipt: bool = True
    population_projection_authority: bool = False
    confidence_authority: bool = False
    fd_replacement_authority: bool = False
    solver_authority: bool = False
    d_stability_authority: bool = False
    c_k_o_authority: bool = False
    mask_commit_authority: bool = False
    terminal_or_pareto_authority: bool = False
    authority_scope: str = LITERAL_J_JOINT_POPULATION_ACTION_AXIS_AUTHORITY
    schema: str = LITERAL_J_JOINT_POPULATION_ACTION_AXIS_PREOUTCOME_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _PRE_RECEIPT_TOKEN:
            raise PermissionError("action-axis PRE receipts are issuer-only")
        action_count = _positive_int(self.action_count, name="action_count")
        shell_size = _positive_int(self.shell_size, name="shell_size")
        direction_count = _positive_int(self.direction_count, name="direction_count")
        for name in (
            "fanout_receipt_hash",
            "fanout_binding_axis_hash",
            "physical_action_axis_hash",
            "joint_preoutcome_registration_hash",
            "joint_direction_axis_hash",
            "d1_source_axis_hash",
            "d1_source_registration_hash",
            "d1_corrected_evidence_stream_source_hash",
            "d1_literal_j_stream_source_hash",
            "d1_same_capture_vjp_source_hash",
            "measure_design_hash",
            "measure_weight_receipt_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "literal_j_stream_source_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "shell_freeze_receipt_hash",
            "parent_mask_hash",
            "alpha_lineage_hash",
        ):
            _identity(getattr(self, name), name=name)
        axes = (
            self.singular_preoutcome_receipt_hashes,
            self.logical_source_receipt_hashes,
            self.destination_binding_hashes,
            self.residual_preoutcome_registration_hashes,
            self.residual_source_registration_hashes,
            self.action_custody_hashes,
            self.action_source_registration_hashes,
            self.physical_action_identity_hashes,
            self.physical_action_hashes,
            self.packed_action_hashes,
            self.resulting_mask_hashes,
            self.removed_route_index_hashes,
        )
        if any(len(_identity_axis(axis, name="action identity axis", unique=False)) != action_count for axis in axes):
            _refuse(
                V1LiteralJJointPopulationActionAxisRefusalCode.INCOMPLETE_ACTION_AXIS,
                "one durable identity axis is not action-complete",
            )
        _identity_axis(self.joint_direction_ids, name="joint_direction_ids")
        if (
            len(self.removed_counts) != action_count
            or any(_positive_int(value, name="removed count") <= 0 for value in self.removed_counts)
            or direction_count != action_count + shell_size
            or len(self.joint_direction_ids) != direction_count
            or type(self.shell_indices) is not tuple
            or type(self.shell_route_ids) is not tuple
            or len(self.shell_indices) != shell_size
            or len(self.shell_route_ids) != shell_size
            or len(set(self.shell_indices)) != shell_size
            or len(set(self.shell_route_ids)) != shell_size
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value >= self.route_count
                for value in self.shell_indices
            )
            or self.provider_outcomes_seen is not False
            or self.logical_sources_bound_at_registration is not False
            or self.complete_ordered_action_axis is not True
            or self.one_action_materialized_at_a_time is not True
            or self.action_route_bank_retained is not False
            or self.array_free_durable_receipt is not True
            or not _closed_authority(self)
            or self.schema != LITERAL_J_JOINT_POPULATION_ACTION_AXIS_PREOUTCOME_SCHEMA
        ):
            _refuse(
                V1LiteralJJointPopulationActionAxisRefusalCode.AUTHORITY_VIOLATION,
                "PRE action-axis receipt is incomplete or widened",
            )
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "preoutcome-")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1LiteralJJointPopulationActionAxisRefusalCode.STALE_SOURCE,
                "PRE action-axis receipt changed after issuance",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class V1LiteralJJointPopulationActionAxisPreoutcomeSource:
    __slots__ = ("receipt", "_sources", "_state", "_lock")
    __hash__ = None

    def __init__(
        self,
        *,
        receipt: V1LiteralJJointPopulationActionAxisPreoutcomeReceipt,
        sources: tuple[V1LiteralJJointPopulationPreoutcomeSource, ...],
        _issuance_token: object | None = None,
    ) -> None:
        if _issuance_token is not _PRE_OWNER_TOKEN:
            raise PermissionError("action-axis PRE owners are issuer-only")
        self.receipt = receipt
        self._sources = sources
        self._state = "preoutcome"
        self._lock = RLock()
        self.assert_preoutcome_live()

    @property
    def action_count(self) -> int:
        return self.receipt.action_count

    @property
    def is_consumed(self) -> bool:
        return self._state == "consumed"

    @property
    def is_evicted(self) -> bool:
        return self._state == "evicted"

    def _replay_sources(self, *, require_preoutcome: bool) -> tuple[str, ...]:
        hashes: list[str] = []
        for source in self._sources:
            try:
                if require_preoutcome:
                    hashes.append(source.assert_preoutcome_live())
                else:
                    source.receipt.assert_integrity()
                    source._replay_common()
                    hashes.append(source.receipt.receipt_hash)
            except Exception as error:
                _refuse(
                    V1LiteralJJointPopulationActionAxisRefusalCode.STALE_SOURCE,
                    "one singular action source failed aggregate replay: "
                    f"{type(error).__name__}: {error}",
                )
        return tuple(hashes)

    def assert_preoutcome_live(self) -> str:
        with self._lock:
            if self._state == "evicted":
                _refuse(
                    V1LiteralJJointPopulationActionAxisRefusalCode.EVICTED,
                    "PRE action-axis owner was evicted",
                )
            if self._state != "preoutcome":
                _refuse(
                    V1LiteralJJointPopulationActionAxisRefusalCode.BIND_REPLAY,
                    "PRE action-axis owner was already consumed",
                )
            self.receipt.assert_integrity()
            hashes = self._replay_sources(require_preoutcome=True)
            if hashes != self.receipt.singular_preoutcome_receipt_hashes:
                _refuse(
                    V1LiteralJJointPopulationActionAxisRefusalCode.STALE_SOURCE,
                    "the singular PRE source axis changed after aggregate issuance",
                )
            return self.receipt.receipt_hash

    def source_for_ordinal(
        self,
        ordinal: int,
    ) -> V1LiteralJJointPopulationPreoutcomeSource:
        with self._lock:
            self.assert_preoutcome_live()
            index = _positive_int(ordinal, name="physical action ordinal") - 1
            if index >= self.receipt.action_count:
                _refuse(
                    V1LiteralJJointPopulationActionAxisRefusalCode.ACTION_ORDER_MISMATCH,
                    "physical action ordinal is outside the sealed axis",
                )
            return self._sources[index]

    def materialize_action_float64(
        self,
        *,
        ordinal: int,
        device: object,
        tile_bytes: int,
        production_cuda_required: bool,
    ) -> tuple[Any, int, int]:
        return self.source_for_ordinal(ordinal).materialize_action_float64(
            device=device,
            tile_bytes=tile_bytes,
            production_cuda_required=production_cuda_required,
        )

    def evict(self) -> None:
        with self._lock:
            if self._state == "consumed":
                _refuse(
                    V1LiteralJJointPopulationActionAxisRefusalCode.BIND_REPLAY,
                    "consumed PRE custody is retained by the POST aggregate",
                )
            self._state = "evicted"

    def __reduce__(self) -> NoReturn:
        raise TypeError("live PRE action-axis owners are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live PRE action-axis owners are nonserializable")


def issue_v1_literal_j_joint_population_action_axis_preoutcome_source(
    *,
    action_sources: tuple[V1LiteralJJointPopulationPreoutcomeSource, ...],
) -> V1LiteralJJointPopulationActionAxisPreoutcomeSource:
    receipts = _validate_pre_sources(action_sources)
    first = receipts[0]
    action_count = len(receipts)
    shell_size = len(first.shell_indices)
    receipt = V1LiteralJJointPopulationActionAxisPreoutcomeReceipt(
        singular_preoutcome_receipt_hashes=tuple(item.receipt_hash for item in receipts),
        logical_source_receipt_hashes=tuple(item.logical_source_receipt_hash for item in receipts),
        destination_binding_hashes=tuple(item.destination_binding_hash for item in receipts),
        residual_preoutcome_registration_hashes=tuple(item.residual_preoutcome_registration_hash for item in receipts),
        residual_source_registration_hashes=tuple(item.residual_source_registration_hash for item in receipts),
        action_custody_hashes=tuple(item.action_custody_hash for item in receipts),
        action_source_registration_hashes=tuple(item.action_source_registration_hash for item in receipts),
        physical_action_identity_hashes=tuple(item.physical_action_identity_hash for item in receipts),
        physical_action_hashes=tuple(item.physical_action_hash for item in receipts),
        packed_action_hashes=tuple(item.packed_action_hash for item in receipts),
        resulting_mask_hashes=tuple(item.resulting_mask_hash for item in receipts),
        removed_route_index_hashes=tuple(item.removed_route_index_hash for item in receipts),
        removed_counts=tuple(item.removed_count for item in receipts),
        fanout_receipt_hash=first.fanout_receipt_hash,
        fanout_binding_axis_hash=first.fanout_binding_axis_hash,
        physical_action_axis_hash=first.physical_action_axis_hash,
        joint_preoutcome_registration_hash=first.joint_preoutcome_registration_hash,
        joint_direction_axis_hash=first.joint_direction_axis_hash,
        joint_direction_ids=first.joint_direction_ids,
        d1_source_axis_hash=first.d1_source_axis_hash,
        d1_source_registration_hash=first.d1_source_registration_hash,
        d1_corrected_evidence_stream_source_hash=(
            first.d1_corrected_evidence_stream_source_hash
        ),
        d1_literal_j_stream_source_hash=first.d1_literal_j_stream_source_hash,
        d1_same_capture_vjp_source_hash=first.d1_same_capture_vjp_source_hash,
        measure_design_hash=first.measure_design_hash,
        measure_weight_receipt_hash=first.measure_weight_receipt_hash,
        model_identity_hash=first.model_identity_hash,
        chart_identity_hash=first.chart_identity_hash,
        current_mask_hash=first.current_mask_hash,
        route_axis_hash=first.route_axis_hash,
        route_id_axis_hash=first.route_id_axis_hash,
        literal_j_stream_source_hash=first.literal_j_stream_source_hash,
        registered_d_prefix_hash=first.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=first.registered_d_w2_prefix_hash,
        shell_freeze_receipt_hash=first.shell_freeze_receipt_hash,
        parent_mask_hash=first.parent_mask_hash,
        shell_indices=first.shell_indices,
        shell_route_ids=first.shell_route_ids,
        alpha_lineage_hash=first.alpha_lineage_hash,
        action_count=action_count,
        shell_size=shell_size,
        direction_count=len(first.joint_direction_ids),
        route_count=first.route_count,
        proposal_generation=first.proposal_generation,
        d1_context_count=first.d1_context_count,
        d1_stratum_counts=first.d1_stratum_counts,
        residual_sample_stratum_counts=first.residual_sample_stratum_counts,
        population_stratum_counts=first.population_stratum_counts,
        d1_context_weight_fraction=first.d1_context_weight_fraction,
        residual_context_weight_fractions=first.residual_context_weight_fractions,
        residual_stratum_mass_fractions=first.residual_stratum_mass_fractions,
        residual_population_mass_fraction=first.residual_population_mass_fraction,
        common_measure_denominator=first.common_measure_denominator,
        d1_context_quantum=first.d1_context_quantum,
        residual_context_quanta=first.residual_context_quanta,
        total_measure_quanta=first.total_measure_quanta,
        **_authority_payload(),
        _issuance_token=_PRE_RECEIPT_TOKEN,
    )
    return V1LiteralJJointPopulationActionAxisPreoutcomeSource(
        receipt=receipt,
        sources=action_sources,
        _issuance_token=_PRE_OWNER_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1LiteralJJointPopulationActionAxisSourceReceipt:
    preoutcome_action_axis_receipt_hash: str
    singular_preoutcome_receipt_hashes: tuple[str, ...]
    singular_postsource_receipt_hashes: tuple[str, ...]
    logical_result_hashes: tuple[str, ...]
    residual_postsource_receipt_hashes: tuple[str, ...]
    physical_action_axis_hash: str
    joint_preoutcome_registration_hash: str
    joint_direction_axis_hash: str
    physical_action_identity_hashes: tuple[str, ...]
    physical_action_hashes: tuple[str, ...]
    packed_action_hashes: tuple[str, ...]
    action_count: int
    provider_outcomes_seen: bool = True
    exact_singular_descents_bound: bool = True
    complete_ordered_action_axis: bool = True
    array_free_durable_receipt: bool = True
    population_projection_authority: bool = False
    confidence_authority: bool = False
    fd_replacement_authority: bool = False
    solver_authority: bool = False
    d_stability_authority: bool = False
    c_k_o_authority: bool = False
    mask_commit_authority: bool = False
    terminal_or_pareto_authority: bool = False
    authority_scope: str = LITERAL_J_JOINT_POPULATION_ACTION_AXIS_AUTHORITY
    schema: str = LITERAL_J_JOINT_POPULATION_ACTION_AXIS_SOURCE_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _POST_RECEIPT_TOKEN:
            raise PermissionError("action-axis POST receipts are binder-only")
        action_count = _positive_int(self.action_count, name="action_count")
        for name in (
            "preoutcome_action_axis_receipt_hash",
            "physical_action_axis_hash",
            "joint_preoutcome_registration_hash",
            "joint_direction_axis_hash",
        ):
            _identity(getattr(self, name), name=name)
        axes = (
            self.singular_preoutcome_receipt_hashes,
            self.singular_postsource_receipt_hashes,
            self.logical_result_hashes,
            self.residual_postsource_receipt_hashes,
            self.physical_action_identity_hashes,
            self.physical_action_hashes,
            self.packed_action_hashes,
        )
        if (
            any(len(_identity_axis(axis, name="POST action identity axis", unique=False)) != action_count for axis in axes)
            or self.provider_outcomes_seen is not True
            or self.exact_singular_descents_bound is not True
            or self.complete_ordered_action_axis is not True
            or self.array_free_durable_receipt is not True
            or not _closed_authority(self)
            or self.schema != LITERAL_J_JOINT_POPULATION_ACTION_AXIS_SOURCE_SCHEMA
        ):
            _refuse(
                V1LiteralJJointPopulationActionAxisRefusalCode.AUTHORITY_VIOLATION,
                "POST action-axis receipt is incomplete or widened",
            )
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "postsource-")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1LiteralJJointPopulationActionAxisRefusalCode.STALE_SOURCE,
                "POST action-axis receipt changed after binding",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class V1LiteralJJointPopulationActionAxisSource:
    __slots__ = ("receipt", "preoutcome_source", "_sources", "_state", "_lock")
    __hash__ = None

    def __init__(
        self,
        *,
        receipt: V1LiteralJJointPopulationActionAxisSourceReceipt,
        preoutcome_source: V1LiteralJJointPopulationActionAxisPreoutcomeSource,
        sources: tuple[V1LiteralJJointPopulationSource, ...],
        _issuance_token: object | None = None,
    ) -> None:
        if _issuance_token is not _POST_OWNER_TOKEN:
            raise PermissionError("action-axis POST owners are binder-only")
        self.receipt = receipt
        self.preoutcome_source = preoutcome_source
        self._sources = sources
        self._state = "live"
        self._lock = RLock()
        self.assert_live()

    @property
    def action_count(self) -> int:
        return self.receipt.action_count

    @property
    def is_evicted(self) -> bool:
        return self._state == "evicted"

    def assert_live(self) -> str:
        with self._lock:
            if self._state == "evicted":
                _refuse(
                    V1LiteralJJointPopulationActionAxisRefusalCode.EVICTED,
                    "POST action-axis owner was evicted",
                )
            self.receipt.assert_integrity()
            pre = self.preoutcome_source
            pre.receipt.assert_integrity()
            hashes: list[str] = []
            for index, source in enumerate(self._sources):
                try:
                    hashes.append(source.assert_live())
                except Exception as error:
                    _refuse(
                        V1LiteralJJointPopulationActionAxisRefusalCode.STALE_SOURCE,
                        "one singular POST source failed aggregate replay: "
                        f"{type(error).__name__}: {error}",
                    )
                if source.preoutcome_source is not pre._sources[index]:
                    _refuse(
                        V1LiteralJJointPopulationActionAxisRefusalCode.SOURCE_DESCENT_MISMATCH,
                        "one POST source does not descend from the exact sealed PRE member",
                    )
            if tuple(hashes) != self.receipt.singular_postsource_receipt_hashes:
                _refuse(
                    V1LiteralJJointPopulationActionAxisRefusalCode.STALE_SOURCE,
                    "the singular POST source axis changed after aggregate binding",
                )
            return self.receipt.receipt_hash

    def source_for_ordinal(self, ordinal: int) -> V1LiteralJJointPopulationSource:
        with self._lock:
            self.assert_live()
            index = _positive_int(ordinal, name="physical action ordinal") - 1
            if index >= self.receipt.action_count:
                _refuse(
                    V1LiteralJJointPopulationActionAxisRefusalCode.ACTION_ORDER_MISMATCH,
                    "physical action ordinal is outside the sealed POST axis",
                )
            return self._sources[index]

    def materialize_action_float64(
        self,
        *,
        ordinal: int,
        device: object,
        tile_bytes: int,
        production_cuda_required: bool,
    ) -> tuple[Any, int, int]:
        return self.source_for_ordinal(ordinal).materialize_action_float64(
            device=device,
            tile_bytes=tile_bytes,
            production_cuda_required=production_cuda_required,
        )

    def evict(self) -> None:
        with self._lock:
            self._state = "evicted"

    def __reduce__(self) -> NoReturn:
        raise TypeError("live POST action-axis owners are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live POST action-axis owners are nonserializable")


def bind_v1_literal_j_joint_population_action_axis_source(
    *,
    preoutcome_source: V1LiteralJJointPopulationActionAxisPreoutcomeSource,
    action_sources: tuple[V1LiteralJJointPopulationSource, ...],
) -> V1LiteralJJointPopulationActionAxisSource:
    if (
        type(preoutcome_source)
        is not V1LiteralJJointPopulationActionAxisPreoutcomeSource
        or type(action_sources) is not tuple
        or any(type(source) is not V1LiteralJJointPopulationSource for source in action_sources)
    ):
        _refuse(
            V1LiteralJJointPopulationActionAxisRefusalCode.INVALID_TYPE,
            "POST action-axis binding requires exact aggregate PRE and singular POST owners",
        )
    with preoutcome_source._lock:
        if preoutcome_source._state != "preoutcome":
            _refuse(
                V1LiteralJJointPopulationActionAxisRefusalCode.BIND_REPLAY,
                "the PRE action-axis aggregate may be bound exactly once",
            )
        pre = preoutcome_source.receipt
        pre.assert_integrity()
        if len(action_sources) != pre.action_count:
            _refuse(
                V1LiteralJJointPopulationActionAxisRefusalCode.INCOMPLETE_ACTION_AXIS,
                "the POST singular source tuple is not action-complete",
            )
        for index, source in enumerate(action_sources):
            try:
                source.assert_live()
            except Exception as error:
                _refuse(
                    V1LiteralJJointPopulationActionAxisRefusalCode.STALE_SOURCE,
                    "one POST source is not live: "
                    f"{type(error).__name__}: {error}",
                )
            if (
                source.preoutcome_source is not preoutcome_source._sources[index]
                or source.receipt.preoutcome_receipt_hash
                != pre.singular_preoutcome_receipt_hashes[index]
                or source.receipt.physical_action_ordinal != index + 1
                or source.receipt.physical_action_identity_hash
                != pre.physical_action_identity_hashes[index]
                or source.receipt.physical_action_hash != pre.physical_action_hashes[index]
                or source.receipt.packed_action_hash != pre.packed_action_hashes[index]
            ):
                _refuse(
                    V1LiteralJJointPopulationActionAxisRefusalCode.SOURCE_DESCENT_MISMATCH,
                    "one POST member is reordered or descends from another PRE source",
                )
        if any(
            source.logical_result is not action_sources[0].logical_result
            for source in action_sources
        ):
            _refuse(
                V1LiteralJJointPopulationActionAxisRefusalCode.SOURCE_DESCENT_MISMATCH,
                "POST action members do not retain one exact shared logical-look result",
            )
        receipt = V1LiteralJJointPopulationActionAxisSourceReceipt(
            preoutcome_action_axis_receipt_hash=pre.receipt_hash,
            singular_preoutcome_receipt_hashes=pre.singular_preoutcome_receipt_hashes,
            singular_postsource_receipt_hashes=tuple(source.receipt.receipt_hash for source in action_sources),
            logical_result_hashes=tuple(source.receipt.logical_result_hash for source in action_sources),
            residual_postsource_receipt_hashes=tuple(source.receipt.residual_postsource_receipt_hash for source in action_sources),
            physical_action_axis_hash=pre.physical_action_axis_hash,
            joint_preoutcome_registration_hash=pre.joint_preoutcome_registration_hash,
            joint_direction_axis_hash=pre.joint_direction_axis_hash,
            physical_action_identity_hashes=pre.physical_action_identity_hashes,
            physical_action_hashes=pre.physical_action_hashes,
            packed_action_hashes=pre.packed_action_hashes,
            action_count=pre.action_count,
            **_authority_payload(),
            _issuance_token=_POST_RECEIPT_TOKEN,
        )
        preoutcome_source._state = "consumed"
        return V1LiteralJJointPopulationActionAxisSource(
            receipt=receipt,
            preoutcome_source=preoutcome_source,
            sources=action_sources,
            _issuance_token=_POST_OWNER_TOKEN,
        )


__all__ = [
    "LITERAL_J_JOINT_POPULATION_ACTION_AXIS_PREOUTCOME_SCHEMA",
    "LITERAL_J_JOINT_POPULATION_ACTION_AXIS_SOURCE_SCHEMA",
    "LITERAL_J_JOINT_POPULATION_ACTION_AXIS_AUTHORITY",
    "V1LiteralJJointPopulationActionAxisRefusalCode",
    "V1LiteralJJointPopulationActionAxisRefusal",
    "V1LiteralJJointPopulationActionAxisPreoutcomeReceipt",
    "V1LiteralJJointPopulationActionAxisPreoutcomeSource",
    "V1LiteralJJointPopulationActionAxisSourceReceipt",
    "V1LiteralJJointPopulationActionAxisSource",
    "issue_v1_literal_j_joint_population_action_axis_preoutcome_source",
    "bind_v1_literal_j_joint_population_action_axis_source",
]
