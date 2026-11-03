"""Bounded relaxed directions for independent signed-work physical actions.

The signed-work pre-C coordinator freezes *physical* packed actions after
deduplicating coordinate/branch aliases.  Those actions need not form a
ladder: each child is an independent irreversible descendant of one parent.
This module turns that exact axis into the two canonical finite-dose lanes

``m_eta = m - eta * (m - m_child), eta in {1/2, 1}``.

Only one parent lane is evaluated for each logical context packet.  Every
unique physical action contributes exactly two direction lanes, irrespective
of how many pre-C aliases name it.  Complete route-by-action matrices are
never retained.  Production execution is CUDA-only and materializes a bounded
packet of full route-granularity gate vectors from bit-packed children; the
NumPy path is a deliberately small mathematical reference and cannot be used
as a production fallback.

The receipts in this module prove source identity, mask arithmetic, bounded
packetization, and evaluator-call accounting only.  They grant no F/Q/R, C,
calibration, oracle, commit, terminal, or Pareto authority.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Final, Literal, NoReturn

import numpy as np

from mechanistic_systems.backends.torch_masks import CompiledTorchCircuitMask
from mechanistic_systems.circuits.irg.modern_v1_directional_remainder import (
    DIRECTIONAL_SECANT_DOSES,
)
from mechanistic_systems.circuits.irg.modern_v1_signed_work_pre_c_packed_actions import (
    V1SignedWorkCanonicalPackedPhysicalAction,
    V1SignedWorkPreCPackedActionFamily,
    v1_signed_work_packed_action_dedup_identity,
)
from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask
from mechanistic_systems.core.storage import RouteBlockSpec

try:  # pragma: no cover - availability depends on the execution host.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


PHYSICAL_ACTION_DIRECTION_POLICY_SCHEMA: Final = (
    "partition_free_v1_signed_work_physical_action_direction_policy_v1"
)
PHYSICAL_ACTION_DIRECTION_DESIGN_SCHEMA: Final = (
    "partition_free_v1_signed_work_physical_action_direction_design_v1"
)
PHYSICAL_ACTION_DIRECTION_FORECAST_SCHEMA: Final = (
    "partition_free_v1_signed_work_physical_action_direction_forecast_v1"
)
PHYSICAL_ACTION_DIRECTION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_signed_work_physical_action_direction_receipt_v1"
)
PHYSICAL_ACTION_CONTEXT_EXECUTION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_signed_work_physical_action_context_execution_receipt_v1"
)
PHYSICAL_ACTION_DIRECTION_AUTHORITY: Final = (
    "independent_packed_direction_execution_only_no_FQR_C_K_O_calibration_"
    "oracle_commit_terminal_or_pareto_authority_v1"
)
PHYSICAL_ACTION_DIRECTION_ALGORITHM_ID: Final = (
    "packed_independent_physical_actions_parent_once_two_exact_secant_lanes_v1"
)

_POLICY_PREFIX: Final = "partition-free-v1-physical-action-direction-policy-"
_DESIGN_PREFIX: Final = "partition-free-v1-physical-action-direction-design-"
_FORECAST_PREFIX: Final = "partition-free-v1-physical-action-direction-forecast-"
_DIRECTION_PREFIX: Final = "partition-free-v1-physical-action-direction-"
_LANE_PREFIX: Final = "partition-free-v1-physical-action-direction-lane-"
_RECEIPT_PREFIX: Final = "partition-free-v1-physical-action-direction-receipt-"
_EXECUTION_PREFIX: Final = "partition-free-v1-physical-action-context-execution-"
_F32_BYTES: Final = np.dtype(np.float32).itemsize
_PACKED_SHA256_HEX_LENGTH: Final = 64
_CUDA_DECODE_WORKSPACE_BYTES_PER_ROUTE: Final = 32
_ABSOLUTE_CPU_REFERENCE_ROUTE_LIMIT: Final = 1 << 16

V1PhysicalActionDirectionBackend = Literal["numpy_reference", "torch_cuda"]


class V1PhysicalActionDirectionRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    SOURCE_MISMATCH = "source_mismatch"
    AMBIGUOUS_PRE_C_SOURCE = "ambiguous_pre_c_source"
    ACTION_IDENTITY_MISMATCH = "action_identity_mismatch"
    CUDA_REQUIRED = "cuda_required"
    CPU_REFERENCE_LIMIT_EXCEEDED = "cpu_reference_limit_exceeded"
    MEMORY_FORECAST_EXCEEDED = "memory_forecast_exceeded"
    EVALUATOR_CONTRACT_VIOLATION = "evaluator_contract_violation"
    PROVIDER_CLOSED = "provider_closed"
    REENTRANT_EXECUTION = "reentrant_execution"
    AUTHORITY_VIOLATION = "authority_violation"


class V1PhysicalActionDirectionRefusal(RuntimeError):
    def __init__(
        self,
        code: V1PhysicalActionDirectionRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1PhysicalActionDirectionRefusalCode:
            raise TypeError("physical-action refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1PhysicalActionDirectionRefusalCode,
    message: str,
) -> NoReturn:
    raise V1PhysicalActionDirectionRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1PhysicalActionDirectionRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1PhysicalActionDirectionRefusalCode.INVALID_TYPE,
            f"{name} must be one positive integer",
        )
    result = int(value)
    if result <= 0:
        _refuse(
            V1PhysicalActionDirectionRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1PhysicalActionDirectionRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1PhysicalActionDirectionRefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _fraction(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be one real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0 or result > 0.85:
        raise ValueError(f"{name} must lie in (0, 0.85]")
    return result


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_sha256(value: object, *, name: str) -> str:
    result = _identity(value, name=name)
    if len(result) != _PACKED_SHA256_HEX_LENGTH or result != result.lower():
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be one lowercase SHA-256 digest") from error
    return result


def physical_action_direction_provider_source_hash() -> str:
    """Return the exact implementation identity used by frozen designs."""

    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def _normalize_route_blocks(
    route_blocks: Sequence[RouteBlockSpec],
    *,
    route_count: int,
) -> tuple[RouteBlockSpec, ...]:
    blocks = tuple(route_blocks)
    if not blocks or any(type(value) is not RouteBlockSpec for value in blocks):
        raise TypeError("physical-action directions require exact route blocks")
    expected = 0
    layer_ids: set[str] = set()
    for block in blocks:
        block.validate()
        if block.offset != expected or block.stop > route_count:
            raise ValueError("route blocks must form one complete contiguous axis")
        if block.layer_id in layer_ids:
            raise ValueError("physical-action route-block layer IDs must be unique")
        layer_ids.add(block.layer_id)
        expected = block.stop
    if expected != route_count:
        raise ValueError("route blocks do not cover the complete physical axis")
    return blocks


def _direct_mask_hash(mask: HardMask) -> str:
    if type(mask) is not HardMask:
        raise TypeError("physical-action directions require exact HardMask values")
    return HardMask(
        route_axis_hash=mask.route_axis_hash,
        route_count=mask.route_count,
        packed_bits=bytes(mask.packed_bits),
    ).mask_hash


def _closed_authority(values: tuple[bool, ...], *, owner: str) -> None:
    if any(type(value) is not bool for value in values):
        raise TypeError(f"{owner} authority flags must be exact booleans")
    if any(values):
        _refuse(
            V1PhysicalActionDirectionRefusalCode.AUTHORITY_VIOLATION,
            f"{owner} crossed its execution-only authority boundary",
        )


@dataclass(frozen=True, slots=True)
class V1PhysicalActionDirectionPolicy:
    maximum_actions_per_packet: int = 4
    route_decode_tile_size: int = 1 << 20
    maximum_provider_workspace_bytes: int = 8 * 1024**3
    maximum_retained_source_packed_bytes: int = 512 * 1024**2
    cpu_reference_route_limit: int = 4096
    hbm_ceiling_fraction: float = 0.85
    schema: str = PHYSICAL_ACTION_DIRECTION_POLICY_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "maximum_actions_per_packet",
            "route_decode_tile_size",
            "maximum_provider_workspace_bytes",
            "maximum_retained_source_packed_bytes",
            "cpu_reference_route_limit",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "hbm_ceiling_fraction",
            _fraction(self.hbm_ceiling_fraction, name="HBM ceiling fraction"),
        )
        if self.cpu_reference_route_limit > _ABSOLUTE_CPU_REFERENCE_ROUTE_LIMIT:
            raise ValueError(
                "physical-action NumPy reference limit exceeds the absolute "
                "small-domain ceiling"
            )
        if self.schema != PHYSICAL_ACTION_DIRECTION_POLICY_SCHEMA:
            raise ValueError("unsupported physical-action direction policy")

    @property
    def policy_hash(self) -> str:
        return content_hash(self, prefix=_POLICY_PREFIX)


@dataclass(frozen=True, slots=True)
class V1PhysicalActionDirectionDesign:
    """Frozen unique-action axis with exact parent/child byte custody."""

    action_family: V1SignedWorkPreCPackedActionFamily = field(
        repr=False,
        compare=False,
    )
    parent_mask: HardMask = field(repr=False, compare=False)
    policy_hash: str
    route_block_axis_hash: str
    provider_source_hash: str
    source_family_hash: str = field(init=False)
    parent_mask_hash: str = field(init=False)
    parent_mask_packed_sha256: str = field(init=False)
    route_axis_hash: str = field(init=False)
    route_count: int = field(init=False)
    canonical_action_hashes: tuple[str, ...] = field(init=False)
    packed_action_hashes: tuple[str, ...] = field(init=False)
    resulting_mask_hashes: tuple[str, ...] = field(init=False)
    resulting_mask_packed_sha256: tuple[str, ...] = field(init=False)
    removed_route_index_hashes: tuple[str, ...] = field(init=False)
    removed_counts: tuple[int, ...] = field(init=False)
    alias_counts: tuple[int, ...] = field(init=False)
    direction_hashes: tuple[str, ...] = field(init=False)
    doses: tuple[float, float] = DIRECTIONAL_SECANT_DOSES
    aliases_affect_execution_axis: bool = False
    independent_actions_need_not_be_nested: bool = True
    full_route_action_host_matrix_materialized: bool = False
    authority_scope: str = PHYSICAL_ACTION_DIRECTION_AUTHORITY
    fqr_authority: bool = False
    C_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = PHYSICAL_ACTION_DIRECTION_DESIGN_SCHEMA
    _design_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        family = self.action_family
        parent = self.parent_mask
        if type(family) is not V1SignedWorkPreCPackedActionFamily:
            raise TypeError("direction design requires the exact packed-action family")
        if type(parent) is not HardMask:
            raise TypeError("direction design requires the exact parent HardMask")
        if family.q_ambiguous_crosswalk_hashes:
            _refuse(
                V1PhysicalActionDirectionRefusalCode.AMBIGUOUS_PRE_C_SOURCE,
                "an unresolved hard-F/Q alias cannot open direction execution",
            )
        actions = tuple(family.canonical_actions)
        if not actions or any(
            type(value) is not V1SignedWorkCanonicalPackedPhysicalAction
            for value in actions
        ):
            raise TypeError(
                "direction design requires a nonempty canonical action axis"
            )
        if (
            family.parent_mask_hash != _direct_mask_hash(parent)
            or family.route_axis_hash != parent.route_axis_hash
            or family.route_count != parent.route_count
        ):
            _refuse(
                V1PhysicalActionDirectionRefusalCode.SOURCE_MISMATCH,
                "packed-action family differs from the supplied parent axis",
            )

        canonical_hashes: list[str] = []
        packed_hashes: list[str] = []
        child_hashes: list[str] = []
        child_sha: list[str] = []
        removed_hashes: list[str] = []
        removed_counts: list[int] = []
        aliases: list[int] = []
        directions: list[str] = []
        for expected_ordinal, canonical in enumerate(actions, start=1):
            action = canonical.packed_action
            child = action.resulting_mask
            if canonical.canonical_ordinal != expected_ordinal:
                raise ValueError("canonical physical-action axis is not contiguous")
            if (
                child.route_axis_hash != parent.route_axis_hash
                or child.route_count != parent.route_count
            ):
                _refuse(
                    V1PhysicalActionDirectionRefusalCode.SOURCE_MISMATCH,
                    "a canonical child lies on another physical route axis",
                )
            observed_count, observed_hash = child.removed_index_count_and_hash_from(
                parent
            )
            if (
                observed_count != action.removed_count
                or observed_hash != action.removed_route_index_hash
                or canonical.dedup_identity
                != v1_signed_work_packed_action_dedup_identity(action)
            ):
                _refuse(
                    V1PhysicalActionDirectionRefusalCode.ACTION_IDENTITY_MISMATCH,
                    "a canonical child does not replay its exact parent delta",
                )
            packed_sha = _sha256(child.packed_bits)
            canonical_hash = canonical.canonical_action_hash
            direction = content_hash(
                {
                    "algorithm_id": PHYSICAL_ACTION_DIRECTION_ALGORITHM_ID,
                    "parent_mask_hash": parent.mask_hash,
                    "parent_mask_packed_sha256": _sha256(parent.packed_bits),
                    "packed_action_hash": action.action_hash,
                    "resulting_mask_hash": child.mask_hash,
                    "resulting_mask_packed_sha256": packed_sha,
                    "removed_route_index_hash": action.removed_route_index_hash,
                    "removed_count": action.removed_count,
                },
                prefix=_DIRECTION_PREFIX,
            )
            canonical_hashes.append(canonical_hash)
            packed_hashes.append(action.action_hash)
            child_hashes.append(child.mask_hash)
            child_sha.append(packed_sha)
            removed_hashes.append(action.removed_route_index_hash)
            removed_counts.append(action.removed_count)
            aliases.append(len(canonical.aliases))
            directions.append(direction)

        if len(set(canonical_hashes)) != len(actions):
            raise ValueError("canonical physical-action hashes repeat")
        if len(set(packed_hashes)) != len(actions):
            raise ValueError("canonical physical actions were not deduplicated")
        doses = tuple(float(value) for value in self.doses)
        if doses != tuple(float(value) for value in DIRECTIONAL_SECANT_DOSES):
            raise ValueError("physical-action directions require canonical doses")
        if (
            type(self.aliases_affect_execution_axis) is not bool
            or self.aliases_affect_execution_axis
            or type(self.independent_actions_need_not_be_nested) is not bool
            or not self.independent_actions_need_not_be_nested
            or type(self.full_route_action_host_matrix_materialized) is not bool
            or self.full_route_action_host_matrix_materialized
        ):
            raise ValueError("physical-action direction topology was weakened")
        _closed_authority(
            (
                self.fqr_authority,
                self.C_measurement_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="physical-action direction design",
        )
        if (
            self.authority_scope != PHYSICAL_ACTION_DIRECTION_AUTHORITY
            or self.schema != PHYSICAL_ACTION_DIRECTION_DESIGN_SCHEMA
        ):
            raise ValueError("unsupported physical-action direction semantics")

        derived = {
            "source_family_hash": family.family_hash,
            "parent_mask_hash": parent.mask_hash,
            "parent_mask_packed_sha256": _sha256(parent.packed_bits),
            "route_axis_hash": parent.route_axis_hash,
            "route_count": parent.route_count,
            "canonical_action_hashes": tuple(canonical_hashes),
            "packed_action_hashes": tuple(packed_hashes),
            "resulting_mask_hashes": tuple(child_hashes),
            "resulting_mask_packed_sha256": tuple(child_sha),
            "removed_route_index_hashes": tuple(removed_hashes),
            "removed_counts": tuple(removed_counts),
            "alias_counts": tuple(aliases),
            "direction_hashes": tuple(directions),
        }
        for name, value in derived.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self, "policy_hash", _identity(self.policy_hash, name="policy hash")
        )
        object.__setattr__(
            self,
            "route_block_axis_hash",
            _identity(self.route_block_axis_hash, name="route-block axis hash"),
        )
        object.__setattr__(
            self,
            "provider_source_hash",
            _validate_sha256(self.provider_source_hash, name="provider source hash"),
        )
        object.__setattr__(self, "doses", DIRECTIONAL_SECANT_DOSES)
        object.__setattr__(
            self, "_design_hash", content_hash(self._payload(), prefix=_DESIGN_PREFIX)
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "algorithm_id": PHYSICAL_ACTION_DIRECTION_ALGORITHM_ID,
            "policy_hash": self.policy_hash,
            "route_block_axis_hash": self.route_block_axis_hash,
            "provider_source_hash": self.provider_source_hash,
            "source_family_hash": self.source_family_hash,
            "parent_mask_hash": self.parent_mask_hash,
            "parent_mask_packed_sha256": self.parent_mask_packed_sha256,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "canonical_action_hashes": self.canonical_action_hashes,
            "packed_action_hashes": self.packed_action_hashes,
            "resulting_mask_hashes": self.resulting_mask_hashes,
            "resulting_mask_packed_sha256": self.resulting_mask_packed_sha256,
            "removed_route_index_hashes": self.removed_route_index_hashes,
            "removed_counts": self.removed_counts,
            "alias_counts": self.alias_counts,
            "direction_hashes": self.direction_hashes,
            "doses": self.doses,
            "aliases_affect_execution_axis": self.aliases_affect_execution_axis,
            "independent_actions_need_not_be_nested": (
                self.independent_actions_need_not_be_nested
            ),
            "full_route_action_host_matrix_materialized": (
                self.full_route_action_host_matrix_materialized
            ),
            "authority_scope": self.authority_scope,
        }

    @property
    def design_hash(self) -> str:
        return self.assert_integrity()

    @property
    def action_count(self) -> int:
        return len(self.canonical_action_hashes)

    @property
    def direction_lane_count(self) -> int:
        return self.action_count * len(self.doses)

    def assert_integrity(self, *, expected_hash: str | None = None) -> str:
        if self.action_family.family_hash != self.source_family_hash:
            raise ValueError("physical-action source family was tampered")
        if (
            _direct_mask_hash(self.parent_mask) != self.parent_mask_hash
            or _sha256(self.parent_mask.packed_bits) != self.parent_mask_packed_sha256
        ):
            raise ValueError("physical-action parent bytes were tampered")
        actions = self.action_family.canonical_actions
        if tuple(value.canonical_action_hash for value in actions) != (
            self.canonical_action_hashes
        ) or tuple(
            _sha256(value.packed_action.resulting_mask.packed_bits) for value in actions
        ) != (
            self.resulting_mask_packed_sha256
        ):
            raise ValueError("physical-action child bytes or hashes were tampered")
        observed = content_hash(self._payload(), prefix=_DESIGN_PREFIX)
        if observed != self._design_hash:
            raise ValueError("physical-action direction design was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("physical-action direction design hash differs")
        return observed


@dataclass(frozen=True, slots=True)
class V1PhysicalActionDirectionForecast:
    design_hash: str
    policy_hash: str
    backend_kind: V1PhysicalActionDirectionBackend
    production_execution: bool
    selected_actions_per_packet: int
    selected_lanes_per_packet: int
    action_packet_sizes: tuple[int, ...]
    lane_packet_sizes: tuple[int, ...]
    source_retained_host_packed_payload_bytes: int
    provider_parent_packed_bytes: int
    provider_parent_gate_bytes: int
    maximum_live_lane_gate_bytes: int
    maximum_transient_child_packed_bytes: int
    maximum_decode_workspace_bytes: int
    provider_peak_workspace_upper_bound_bytes: int
    current_reserved_bytes: int
    device_or_workspace_ceiling_bytes: int
    full_route_action_host_matrix_materialized: bool = False
    schema: str = PHYSICAL_ACTION_DIRECTION_FORECAST_SCHEMA

    def __post_init__(self) -> None:
        for name in ("design_hash", "policy_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if self.backend_kind not in {"numpy_reference", "torch_cuda"}:
            raise ValueError("unsupported physical-action direction backend")
        if type(self.production_execution) is not bool:
            raise TypeError("production execution flag must be exact")
        actions = _positive_int(
            self.selected_actions_per_packet,
            name="selected actions per packet",
        )
        lanes = _positive_int(
            self.selected_lanes_per_packet,
            name="selected lanes per packet",
        )
        if lanes != actions * len(DIRECTIONAL_SECANT_DOSES):
            raise ValueError("each selected physical action must own two lanes")
        action_packets = tuple(
            _positive_int(value, name="action packet size")
            for value in self.action_packet_sizes
        )
        lane_packets = tuple(
            _positive_int(value, name="lane packet size")
            for value in self.lane_packet_sizes
        )
        if (
            not action_packets
            or len(action_packets) != len(lane_packets)
            or any(value > actions for value in action_packets)
            or lane_packets
            != tuple(value * len(DIRECTIONAL_SECANT_DOSES) for value in action_packets)
        ):
            raise ValueError("physical-action packet accounting is inconsistent")
        for name in (
            "source_retained_host_packed_payload_bytes",
            "provider_parent_packed_bytes",
            "provider_parent_gate_bytes",
            "maximum_live_lane_gate_bytes",
            "maximum_transient_child_packed_bytes",
            "maximum_decode_workspace_bytes",
            "provider_peak_workspace_upper_bound_bytes",
            "current_reserved_bytes",
            "device_or_workspace_ceiling_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        expected_peak = (
            self.provider_parent_packed_bytes
            + self.provider_parent_gate_bytes
            + self.maximum_live_lane_gate_bytes
            + self.maximum_transient_child_packed_bytes
            + self.maximum_decode_workspace_bytes
        )
        if self.provider_peak_workspace_upper_bound_bytes != expected_peak:
            raise ValueError("direction forecast peak does not replay")
        if (
            self.current_reserved_bytes + expected_peak
            > self.device_or_workspace_ceiling_bytes
        ):
            _refuse(
                V1PhysicalActionDirectionRefusalCode.MEMORY_FORECAST_EXCEEDED,
                "direction forecast exceeds its pre-outcome workspace ceiling",
            )
        if (
            type(self.full_route_action_host_matrix_materialized) is not bool
            or self.full_route_action_host_matrix_materialized
        ):
            raise ValueError("direction forecast permits a full host action matrix")
        if self.schema != PHYSICAL_ACTION_DIRECTION_FORECAST_SCHEMA:
            raise ValueError("unsupported physical-action direction forecast")
        object.__setattr__(self, "action_packet_sizes", action_packets)
        object.__setattr__(self, "lane_packet_sizes", lane_packets)

    @property
    def forecast_hash(self) -> str:
        return content_hash(self, prefix=_FORECAST_PREFIX)


@dataclass(frozen=True, slots=True)
class V1PhysicalActionDirectionLaneIdentity:
    physical_action_ordinal: int
    canonical_action_hash: str
    packed_action_hash: str
    resulting_mask_hash: str
    resulting_mask_packed_sha256: str
    removed_count: int
    dose: float
    direction_hash: str
    lane_hash: str
    lane_key: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "physical_action_ordinal",
            _positive_int(
                self.physical_action_ordinal,
                name="physical action ordinal",
            ),
        )
        object.__setattr__(
            self,
            "removed_count",
            _positive_int(self.removed_count, name="removed count"),
        )
        for name in (
            "canonical_action_hash",
            "packed_action_hash",
            "resulting_mask_hash",
            "direction_hash",
            "lane_hash",
            "lane_key",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "resulting_mask_packed_sha256",
            _validate_sha256(
                self.resulting_mask_packed_sha256,
                name="resulting-mask packed SHA-256",
            ),
        )
        dose = float(self.dose)
        if dose not in DIRECTIONAL_SECANT_DOSES:
            raise ValueError("physical-action lane dose is not canonical")
        object.__setattr__(self, "dose", dose)
        expected_lane = content_hash(
            {
                "direction_hash": self.direction_hash,
                "dose": dose,
                "resulting_mask_packed_sha256": self.resulting_mask_packed_sha256,
            },
            prefix=_LANE_PREFIX,
        )
        if self.lane_hash != expected_lane:
            raise ValueError("physical-action lane hash does not replay")


@dataclass(frozen=True, slots=True)
class V1PhysicalActionDirectionReceipt:
    design: V1PhysicalActionDirectionDesign = field(repr=False, compare=False)
    forecast: V1PhysicalActionDirectionForecast
    design_hash: str
    policy_hash: str
    source_family_hash: str
    parent_mask_hash: str
    parent_mask_packed_sha256: str
    backend_kind: V1PhysicalActionDirectionBackend
    device_identity: str
    production_execution: bool
    parent_lane_key: str
    lane_identities: tuple[V1PhysicalActionDirectionLaneIdentity, ...]
    alias_count: int
    unique_physical_action_count: int
    parent_lanes_per_context_packet: int = 1
    direction_lanes_per_physical_action: int = 2
    aliases_affect_lane_count: bool = False
    full_route_action_host_matrix_materialized: bool = False
    fqr_authority: bool = False
    C_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = PHYSICAL_ACTION_DIRECTION_AUTHORITY
    schema: str = PHYSICAL_ACTION_DIRECTION_RECEIPT_SCHEMA
    _receipt_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.design) is not V1PhysicalActionDirectionDesign:
            raise TypeError("direction receipt requires the exact frozen design")
        if type(self.forecast) is not V1PhysicalActionDirectionForecast:
            raise TypeError("direction receipt requires the exact forecast")
        self.design.assert_integrity(expected_hash=self.design_hash)
        if (
            self.policy_hash != self.design.policy_hash
            or self.source_family_hash != self.design.source_family_hash
            or self.parent_mask_hash != self.design.parent_mask_hash
            or self.parent_mask_packed_sha256 != self.design.parent_mask_packed_sha256
            or self.forecast.design_hash != self.design_hash
            or self.forecast.policy_hash != self.policy_hash
            or self.forecast.backend_kind != self.backend_kind
            or self.forecast.production_execution != self.production_execution
        ):
            _refuse(
                V1PhysicalActionDirectionRefusalCode.SOURCE_MISMATCH,
                "direction receipt differs from its design or forecast",
            )
        lanes = tuple(self.lane_identities)
        expected_lanes = self.design.direction_lane_count
        if (
            len(lanes) != expected_lanes
            or any(
                type(value) is not V1PhysicalActionDirectionLaneIdentity
                for value in lanes
            )
            or len({value.lane_hash for value in lanes}) != expected_lanes
        ):
            raise ValueError("direction receipt lane axis is incomplete")
        expected_coordinates = tuple(
            (ordinal, dose)
            for ordinal in range(1, self.design.action_count + 1)
            for dose in self.design.doses
        )
        if tuple((value.physical_action_ordinal, value.dose) for value in lanes) != (
            expected_coordinates
        ):
            raise ValueError("direction receipt reordered its action/dose axis")
        alias_count = _positive_int(self.alias_count, name="alias count")
        unique = _positive_int(
            self.unique_physical_action_count,
            name="unique physical-action count",
        )
        if (
            alias_count != sum(self.design.alias_counts)
            or unique != self.design.action_count
        ):
            raise ValueError("direction receipt alias/action accounting is stale")
        if (
            self.parent_lanes_per_context_packet != 1
            or self.direction_lanes_per_physical_action != len(DIRECTIONAL_SECANT_DOSES)
            or type(self.aliases_affect_lane_count) is not bool
            or self.aliases_affect_lane_count
            or type(self.full_route_action_host_matrix_materialized) is not bool
            or self.full_route_action_host_matrix_materialized
        ):
            raise ValueError("direction receipt execution cardinality was weakened")
        _closed_authority(
            (
                self.fqr_authority,
                self.C_measurement_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="physical-action direction receipt",
        )
        if (
            self.authority_scope != PHYSICAL_ACTION_DIRECTION_AUTHORITY
            or self.schema != PHYSICAL_ACTION_DIRECTION_RECEIPT_SCHEMA
        ):
            raise ValueError("unsupported physical-action direction receipt")
        object.__setattr__(self, "lane_identities", lanes)
        object.__setattr__(self, "alias_count", alias_count)
        object.__setattr__(self, "unique_physical_action_count", unique)
        for name in (
            "design_hash",
            "policy_hash",
            "source_family_hash",
            "parent_mask_hash",
            "device_identity",
            "parent_lane_key",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "parent_mask_packed_sha256",
            _validate_sha256(
                self.parent_mask_packed_sha256,
                name="parent-mask packed SHA-256",
            ),
        )
        object.__setattr__(
            self, "_receipt_hash", content_hash(self._payload(), prefix=_RECEIPT_PREFIX)
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "design_hash": self.design_hash,
            "forecast_hash": self.forecast.forecast_hash,
            "policy_hash": self.policy_hash,
            "source_family_hash": self.source_family_hash,
            "parent_mask_hash": self.parent_mask_hash,
            "parent_mask_packed_sha256": self.parent_mask_packed_sha256,
            "backend_kind": self.backend_kind,
            "device_identity": self.device_identity,
            "production_execution": self.production_execution,
            "parent_lane_key": self.parent_lane_key,
            "lane_hashes": tuple(value.lane_hash for value in self.lane_identities),
            "lane_keys": tuple(value.lane_key for value in self.lane_identities),
            "alias_count": self.alias_count,
            "unique_physical_action_count": self.unique_physical_action_count,
            "parent_lanes_per_context_packet": self.parent_lanes_per_context_packet,
            "direction_lanes_per_physical_action": (
                self.direction_lanes_per_physical_action
            ),
            "aliases_affect_lane_count": self.aliases_affect_lane_count,
            "full_route_action_host_matrix_materialized": (
                self.full_route_action_host_matrix_materialized
            ),
            "authority_scope": self.authority_scope,
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    @property
    def direction_lane_count(self) -> int:
        return len(self.lane_identities)

    def assert_integrity(self, *, expected_hash: str | None = None) -> str:
        self.design.assert_integrity(expected_hash=self.design_hash)
        observed = content_hash(self._payload(), prefix=_RECEIPT_PREFIX)
        if observed != self._receipt_hash:
            raise ValueError("physical-action direction receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("physical-action direction receipt hash differs")
        return observed


@dataclass(frozen=True, slots=True)
class V1PhysicalActionContextExecutionReceipt:
    provider_receipt_hash: str
    context_packet_hash: str
    parent_lane_key: str
    direction_lane_hashes: tuple[str, ...]
    direction_lane_keys: tuple[str, ...]
    direction_packet_sizes: tuple[int, ...]
    parent_lane_count: int
    direction_lane_count: int
    parent_evaluator_call_count: int
    direction_evaluator_call_count: int
    total_evaluator_call_count: int
    unique_physical_action_count: int
    alias_count: int
    aliases_affect_lane_count: bool = False
    parent_reused_across_all_actions: bool = True
    fqr_authority: bool = False
    C_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = PHYSICAL_ACTION_DIRECTION_AUTHORITY
    schema: str = PHYSICAL_ACTION_CONTEXT_EXECUTION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "provider_receipt_hash",
            "context_packet_hash",
            "parent_lane_key",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        hashes = tuple(
            _identity(value, name="direction lane hash")
            for value in self.direction_lane_hashes
        )
        keys = tuple(
            _identity(value, name="direction lane key")
            for value in self.direction_lane_keys
        )
        packets = tuple(
            _positive_int(value, name="direction packet size")
            for value in self.direction_packet_sizes
        )
        parent_lanes = _positive_int(self.parent_lane_count, name="parent lane count")
        direction_lanes = _positive_int(
            self.direction_lane_count,
            name="direction lane count",
        )
        parent_calls = _positive_int(
            self.parent_evaluator_call_count,
            name="parent evaluator call count",
        )
        direction_calls = _positive_int(
            self.direction_evaluator_call_count,
            name="direction evaluator call count",
        )
        total_calls = _positive_int(
            self.total_evaluator_call_count,
            name="total evaluator call count",
        )
        unique = _positive_int(
            self.unique_physical_action_count,
            name="unique physical-action count",
        )
        aliases = _positive_int(self.alias_count, name="alias count")
        if (
            parent_lanes != 1
            or parent_calls != 1
            or len(hashes) != direction_lanes
            or len(keys) != direction_lanes
            or len(set(hashes)) != direction_lanes
            or sum(packets) != direction_lanes
            or direction_calls != len(packets)
            or total_calls != 1 + direction_calls
            or direction_lanes != unique * len(DIRECTIONAL_SECANT_DOSES)
            or aliases < unique
            or type(self.aliases_affect_lane_count) is not bool
            or self.aliases_affect_lane_count
            or type(self.parent_reused_across_all_actions) is not bool
            or not self.parent_reused_across_all_actions
        ):
            raise ValueError("physical-action execution accounting is inconsistent")
        _closed_authority(
            (
                self.fqr_authority,
                self.C_measurement_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="physical-action context execution receipt",
        )
        if (
            self.authority_scope != PHYSICAL_ACTION_DIRECTION_AUTHORITY
            or self.schema != PHYSICAL_ACTION_CONTEXT_EXECUTION_RECEIPT_SCHEMA
        ):
            raise ValueError("unsupported physical-action context execution receipt")
        object.__setattr__(self, "direction_lane_hashes", hashes)
        object.__setattr__(self, "direction_lane_keys", keys)
        object.__setattr__(self, "direction_packet_sizes", packets)
        object.__setattr__(self, "parent_lane_count", parent_lanes)
        object.__setattr__(self, "direction_lane_count", direction_lanes)
        object.__setattr__(self, "parent_evaluator_call_count", parent_calls)
        object.__setattr__(self, "direction_evaluator_call_count", direction_calls)
        object.__setattr__(self, "total_evaluator_call_count", total_calls)
        object.__setattr__(self, "unique_physical_action_count", unique)
        object.__setattr__(self, "alias_count", aliases)

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_EXECUTION_PREFIX)


@dataclass(frozen=True, slots=True)
class V1PhysicalActionContextExecution:
    receipt: V1PhysicalActionContextExecutionReceipt
    parent_response: object = field(repr=False, compare=False)
    direction_responses: tuple[object, ...] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.receipt) is not V1PhysicalActionContextExecutionReceipt:
            raise TypeError("context execution requires the exact accounting receipt")
        responses = tuple(self.direction_responses)
        if len(responses) != self.receipt.direction_lane_count:
            raise ValueError("direction responses do not align with the lane axis")
        object.__setattr__(self, "direction_responses", responses)


def register_v1_physical_action_direction_design(
    *,
    action_family: V1SignedWorkPreCPackedActionFamily,
    parent_mask: HardMask,
    policy: V1PhysicalActionDirectionPolicy,
    route_blocks: Sequence[RouteBlockSpec],
) -> V1PhysicalActionDirectionDesign:
    """Freeze the unique physical-action axis before any direction outcome."""

    if type(policy) is not V1PhysicalActionDirectionPolicy:
        raise TypeError("direction design registration requires the exact policy")
    if type(action_family) is not V1SignedWorkPreCPackedActionFamily:
        raise TypeError(
            "direction design registration requires the exact action family"
        )
    if type(parent_mask) is not HardMask:
        raise TypeError("direction design registration requires the exact parent mask")
    blocks = _normalize_route_blocks(route_blocks, route_count=parent_mask.route_count)
    route_block_axis_hash = content_hash(
        tuple(block.to_dict() for block in blocks),
        prefix="partition-free-v1-physical-action-route-block-axis-",
    )
    return V1PhysicalActionDirectionDesign(
        action_family=action_family,
        parent_mask=parent_mask,
        policy_hash=policy.policy_hash,
        route_block_axis_hash=route_block_axis_hash,
        provider_source_hash=physical_action_direction_provider_source_hash(),
    )


def _forecast(
    *,
    design: V1PhysicalActionDirectionDesign,
    policy: V1PhysicalActionDirectionPolicy,
    backend_kind: V1PhysicalActionDirectionBackend,
    production_execution: bool,
    current_reserved_bytes: int,
    device_total_memory_bytes: int,
) -> V1PhysicalActionDirectionForecast:
    routes = design.route_count
    packed = (routes + 7) // 8
    reserved = _nonnegative_int(current_reserved_bytes, name="current reserved bytes")
    total = _positive_int(device_total_memory_bytes, name="device/workspace bytes")
    workspace_cap = min(policy.maximum_provider_workspace_bytes, total)
    if backend_kind == "torch_cuda":
        workspace_cap = min(
            workspace_cap,
            int(math.floor(total * policy.hbm_ceiling_fraction)),
        )
    parent_gate = routes * _F32_BYTES
    decode = min(policy.route_decode_tile_size, routes) * (
        _CUDA_DECODE_WORKSPACE_BYTES_PER_ROUTE
    )
    transient_child = packed
    fixed = packed + parent_gate + transient_child + decode
    maximum = min(policy.maximum_actions_per_packet, design.action_count)
    selected = 0
    for actions in range(maximum, 0, -1):
        live_lanes = actions * len(DIRECTIONAL_SECANT_DOSES) * routes * _F32_BYTES
        if reserved + fixed + live_lanes <= workspace_cap:
            selected = actions
            break
    if selected <= 0:
        _refuse(
            V1PhysicalActionDirectionRefusalCode.MEMORY_FORECAST_EXCEEDED,
            "not even one two-dose physical action fits the registered ceiling",
        )
    action_packets = tuple(
        min(selected, design.action_count - start)
        for start in range(0, design.action_count, selected)
    )
    lanes = selected * len(DIRECTIONAL_SECANT_DOSES)
    live_lane_bytes = lanes * routes * _F32_BYTES
    peak = fixed + live_lane_bytes
    return V1PhysicalActionDirectionForecast(
        design_hash=design.design_hash,
        policy_hash=policy.policy_hash,
        backend_kind=backend_kind,
        production_execution=production_execution,
        selected_actions_per_packet=selected,
        selected_lanes_per_packet=lanes,
        action_packet_sizes=action_packets,
        lane_packet_sizes=tuple(
            value * len(DIRECTIONAL_SECANT_DOSES) for value in action_packets
        ),
        source_retained_host_packed_payload_bytes=(
            design.action_family.canonical_retained_host_packed_payload_bytes
        ),
        provider_parent_packed_bytes=packed,
        provider_parent_gate_bytes=parent_gate,
        maximum_live_lane_gate_bytes=live_lane_bytes,
        maximum_transient_child_packed_bytes=transient_child,
        maximum_decode_workspace_bytes=decode,
        provider_peak_workspace_upper_bound_bytes=peak,
        current_reserved_bytes=reserved,
        device_or_workspace_ceiling_bytes=workspace_cap,
    )


class V1PhysicalActionDirectionProvider:
    """Execute independent packed actions with one parent and two lanes each."""

    __slots__ = (
        "_active",
        "_closed",
        "_device",
        "_parent_gate",
        "_route_blocks",
        "backend_kind",
        "design",
        "evaluator",
        "forecast",
        "policy",
        "production_execution",
        "receipt",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        evaluator: object,
        design: V1PhysicalActionDirectionDesign,
        policy: V1PhysicalActionDirectionPolicy,
        route_blocks: Sequence[RouteBlockSpec],
        backend_kind: V1PhysicalActionDirectionBackend,
        production_execution: bool,
        current_reserved_bytes: int,
        device_total_memory_bytes: int,
        device: object | None = None,
    ) -> None:
        if type(design) is not V1PhysicalActionDirectionDesign:
            raise TypeError("direction provider requires the exact design")
        if type(policy) is not V1PhysicalActionDirectionPolicy:
            raise TypeError("direction provider requires the exact policy")
        design.assert_integrity()
        if design.policy_hash != policy.policy_hash:
            _refuse(
                V1PhysicalActionDirectionRefusalCode.SOURCE_MISMATCH,
                "direction policy differs from its frozen design",
            )
        blocks = _normalize_route_blocks(route_blocks, route_count=design.route_count)
        block_hash = content_hash(
            tuple(block.to_dict() for block in blocks),
            prefix="partition-free-v1-physical-action-route-block-axis-",
        )
        if block_hash != design.route_block_axis_hash:
            _refuse(
                V1PhysicalActionDirectionRefusalCode.SOURCE_MISMATCH,
                "route blocks differ from the frozen direction design",
            )
        if backend_kind not in {"numpy_reference", "torch_cuda"}:
            raise ValueError("unsupported physical-action direction backend")
        if type(production_execution) is not bool:
            raise TypeError("production execution flag must be exact")
        if production_execution and backend_kind != "torch_cuda":
            _refuse(
                V1PhysicalActionDirectionRefusalCode.CUDA_REQUIRED,
                "production physical-action directions have no CPU fallback",
            )
        if (
            design.action_family.canonical_retained_host_packed_payload_bytes
            > policy.maximum_retained_source_packed_bytes
        ):
            _refuse(
                V1PhysicalActionDirectionRefusalCode.MEMORY_FORECAST_EXCEEDED,
                "canonical packed-action source exceeds its custody ceiling",
            )
        resolved_device: object | None = None
        if backend_kind == "numpy_reference":
            if design.route_count > policy.cpu_reference_route_limit:
                _refuse(
                    V1PhysicalActionDirectionRefusalCode.CPU_REFERENCE_LIMIT_EXCEEDED,
                    "NumPy directions are restricted to the small reference domain",
                )
            device_identity = "numpy_reference:cpu"
        else:
            if torch is None or not torch.cuda.is_available():
                _refuse(
                    V1PhysicalActionDirectionRefusalCode.CUDA_REQUIRED,
                    "CUDA direction execution requires an available PyTorch CUDA runtime",
                )
            resolved_device = torch.device("cuda" if device is None else device)
            if resolved_device.type != "cuda":
                _refuse(
                    V1PhysicalActionDirectionRefusalCode.CUDA_REQUIRED,
                    "production direction tensors must reside on CUDA",
                )
            device_identity = (
                str(torch.cuda.get_device_properties(resolved_device).name)
                + f":{resolved_device}"
            )

        forecast = _forecast(
            design=design,
            policy=policy,
            backend_kind=backend_kind,
            production_execution=production_execution,
            current_reserved_bytes=current_reserved_bytes,
            device_total_memory_bytes=device_total_memory_bytes,
        )
        lanes = self._lane_axis(design)
        parent_lane_key = f"physical-action-parent:{design.design_hash}"
        self.evaluator = evaluator
        self.design = design
        self.policy = policy
        self._route_blocks = blocks
        self.backend_kind = backend_kind
        self.production_execution = production_execution
        self.forecast = forecast
        self._device = resolved_device
        self._active = False
        self._closed = False
        self._parent_gate = self._materialize_parent_gate()
        self.receipt = V1PhysicalActionDirectionReceipt(
            design=design,
            forecast=forecast,
            design_hash=design.design_hash,
            policy_hash=policy.policy_hash,
            source_family_hash=design.source_family_hash,
            parent_mask_hash=design.parent_mask_hash,
            parent_mask_packed_sha256=design.parent_mask_packed_sha256,
            backend_kind=backend_kind,
            device_identity=device_identity,
            production_execution=production_execution,
            parent_lane_key=parent_lane_key,
            lane_identities=lanes,
            alias_count=sum(design.alias_counts),
            unique_physical_action_count=design.action_count,
        )

    @staticmethod
    def _lane_axis(
        design: V1PhysicalActionDirectionDesign,
    ) -> tuple[V1PhysicalActionDirectionLaneIdentity, ...]:
        result: list[V1PhysicalActionDirectionLaneIdentity] = []
        for index in range(design.action_count):
            for dose in design.doses:
                lane_hash = content_hash(
                    {
                        "direction_hash": design.direction_hashes[index],
                        "dose": dose,
                        "resulting_mask_packed_sha256": (
                            design.resulting_mask_packed_sha256[index]
                        ),
                    },
                    prefix=_LANE_PREFIX,
                )
                result.append(
                    V1PhysicalActionDirectionLaneIdentity(
                        physical_action_ordinal=index + 1,
                        canonical_action_hash=design.canonical_action_hashes[index],
                        packed_action_hash=design.packed_action_hashes[index],
                        resulting_mask_hash=design.resulting_mask_hashes[index],
                        resulting_mask_packed_sha256=(
                            design.resulting_mask_packed_sha256[index]
                        ),
                        removed_count=design.removed_counts[index],
                        dose=dose,
                        direction_hash=design.direction_hashes[index],
                        lane_hash=lane_hash,
                        lane_key=(
                            f"physical-action:{design.direction_hashes[index]}:"
                            f"eta:{float(dose).hex()}"
                        ),
                    )
                )
        return tuple(result)

    def _require_live(self) -> None:
        if self._closed:
            _refuse(
                V1PhysicalActionDirectionRefusalCode.PROVIDER_CLOSED,
                "physical-action direction provider is closed",
            )

    def _decode_cuda_packed(self, packed_bits: bytes) -> object:
        assert torch is not None
        device = self._device
        packed = torch.tensor(
            np.frombuffer(packed_bits, dtype=np.uint8),
            dtype=torch.uint8,
            device=device,
        )
        output = torch.empty(
            self.design.route_count,
            dtype=torch.float32,
            device=device,
        )
        tile = min(self.policy.route_decode_tile_size, self.design.route_count)
        for start in range(0, self.design.route_count, tile):
            stop = min(self.design.route_count, start + tile)
            indices = torch.arange(start, stop, dtype=torch.int64, device=device)
            byte_indices = torch.bitwise_right_shift(indices, 3)
            shifts = 7 - torch.bitwise_and(indices, 7)
            bits = torch.bitwise_and(
                torch.bitwise_right_shift(
                    packed[byte_indices].to(dtype=torch.int64),
                    shifts,
                ),
                1,
            )
            output[start:stop].copy_(bits.to(dtype=torch.float32))
            del bits, byte_indices, indices, shifts
        del packed
        return output

    def _materialize_parent_gate(self) -> object:
        if self.backend_kind == "numpy_reference":
            values = self.design.parent_mask.to_bool_array().astype(
                np.float32,
                copy=True,
            )
            if values.shape != (self.design.route_count,):
                raise ValueError(
                    "parent mask did not decode on its physical route axis"
                )
            return values
        return self._decode_cuda_packed(self.design.parent_mask.packed_bits)

    def _action_gates(
        self,
        canonical: V1SignedWorkCanonicalPackedPhysicalAction,
    ) -> tuple[object, object]:
        child = canonical.packed_action.resulting_mask
        expected_index = canonical.canonical_ordinal - 1
        if (
            _sha256(child.packed_bits)
            != self.design.resulting_mask_packed_sha256[expected_index]
            or child.mask_hash != self.design.resulting_mask_hashes[expected_index]
        ):
            _refuse(
                V1PhysicalActionDirectionRefusalCode.ACTION_IDENTITY_MISMATCH,
                "physical-action child changed after direction registration",
            )
        if self.backend_kind == "numpy_reference":
            parent = np.asarray(self._parent_gate, dtype=np.float32)
            child_values = child.to_bool_array()
            parent_live = parent != 0.0
            if bool(np.any(child_values & ~parent_live)):
                _refuse(
                    V1PhysicalActionDirectionRefusalCode.ACTION_IDENTITY_MISMATCH,
                    "physical-action child reactivates a parent-dead route",
                )
            delta = parent_live & ~child_values
            gates = tuple(
                parent - np.float32(dose) * delta.astype(np.float32)
                for dose in self.design.doses
            )
            return gates  # type: ignore[return-value]

        assert torch is not None
        parent = self._parent_gate
        child_packed = torch.tensor(
            np.frombuffer(child.packed_bits, dtype=np.uint8),
            dtype=torch.uint8,
            device=self._device,
        )
        result = tuple(parent.clone() for _dose in self.design.doses)
        tile = min(self.policy.route_decode_tile_size, self.design.route_count)
        for start in range(0, self.design.route_count, tile):
            stop = min(self.design.route_count, start + tile)
            indices = torch.arange(
                start,
                stop,
                dtype=torch.int64,
                device=self._device,
            )
            byte_indices = torch.bitwise_right_shift(indices, 3)
            shifts = 7 - torch.bitwise_and(indices, 7)
            child_live = torch.bitwise_and(
                torch.bitwise_right_shift(
                    child_packed[byte_indices].to(dtype=torch.int64),
                    shifts,
                ),
                1,
            ).to(dtype=torch.bool)
            parent_live = parent[start:stop] != 0.0
            if bool(torch.any(child_live & ~parent_live).item()):
                _refuse(
                    V1PhysicalActionDirectionRefusalCode.ACTION_IDENTITY_MISMATCH,
                    "physical-action child reactivates a parent-dead route",
                )
            removed = parent_live & ~child_live
            for dose, gate in zip(self.design.doses, result, strict=True):
                gate[start:stop].masked_fill_(removed, 1.0 - float(dose))
            del byte_indices, child_live, indices, parent_live, removed, shifts
        del child_packed
        return result  # type: ignore[return-value]

    def _compiled_from_flat_gate(self, flat_gate: object) -> CompiledTorchCircuitMask:
        gates: dict[str, object] = {}
        residual: dict[str, object] = {}
        for block in self._route_blocks:
            value = flat_gate[block.offset : block.stop].reshape(block.shape)  # type: ignore[index,union-attr]
            key = str(block.layer_id)
            gates[key] = value
            if block.route_kind == "residual_stream_gate":
                residual[key] = value
        compiler = getattr(self.evaluator, "compile_gate_values", None)
        if callable(compiler):
            compiled = compiler(dict(gates))
            if type(compiled) is not CompiledTorchCircuitMask:
                _refuse(
                    V1PhysicalActionDirectionRefusalCode.EVALUATOR_CONTRACT_VIOLATION,
                    "relaxed-gate compiler returned a noncanonical mask",
                )
            return compiled
        return CompiledTorchCircuitMask(
            parameter_overrides={},
            residual_gates=residual,
            gate_values=gates,
            float_gate_values=gates,
        )

    def _evaluate(
        self,
        *,
        batch_factory: Callable[[], object],
        compiled_masks: tuple[CompiledTorchCircuitMask, ...],
        lane_keys: tuple[str, ...],
    ) -> tuple[object, ...]:
        if not callable(batch_factory):
            raise TypeError("physical-action execution requires a batch factory")
        method = getattr(self.evaluator, "readout_responses_from_batches", None)
        if not callable(method):
            _refuse(
                V1PhysicalActionDirectionRefusalCode.EVALUATOR_CONTRACT_VIOLATION,
                "direction evaluator lacks the canonical readout method",
            )
        responses = tuple(
            method(
                batch_factory(),
                compiled_masks=compiled_masks,
                lane_keys=lane_keys,
            )
        )
        if len(responses) != len(compiled_masks):
            _refuse(
                V1PhysicalActionDirectionRefusalCode.EVALUATOR_CONTRACT_VIOLATION,
                "direction evaluator changed the requested lane cardinality",
            )
        return responses

    def _iter_action_packets(
        self,
    ) -> Iterator[
        tuple[
            tuple[V1PhysicalActionDirectionLaneIdentity, ...],
            tuple[CompiledTorchCircuitMask, ...],
        ]
    ]:
        actions = self.design.action_family.canonical_actions
        width = self.forecast.selected_actions_per_packet
        for start in range(0, len(actions), width):
            stop = min(len(actions), start + width)
            lane_identities: list[V1PhysicalActionDirectionLaneIdentity] = []
            masks: list[CompiledTorchCircuitMask] = []
            for action_index in range(start, stop):
                action_gates = self._action_gates(actions[action_index])
                lane_offset = action_index * len(self.design.doses)
                for dose_offset, gate in enumerate(action_gates):
                    lane_identities.append(
                        self.receipt.lane_identities[lane_offset + dose_offset]
                    )
                    masks.append(self._compiled_from_flat_gate(gate))
            yield tuple(lane_identities), tuple(masks)

    def execute_context_packet(
        self,
        *,
        context_packet_hash: str,
        batch_factory: Callable[[], object],
    ) -> V1PhysicalActionContextExecution:
        """Evaluate one parent once, then exactly two lanes per unique action."""

        self._require_live()
        packet_hash = _identity(context_packet_hash, name="context packet hash")
        if self._active:
            _refuse(
                V1PhysicalActionDirectionRefusalCode.REENTRANT_EXECUTION,
                "physical-action context execution is already active",
            )
        self._active = True
        try:
            parent_compiled = self._compiled_from_flat_gate(self._parent_gate)
            parent_response = self._evaluate(
                batch_factory=batch_factory,
                compiled_masks=(parent_compiled,),
                lane_keys=(self.receipt.parent_lane_key,),
            )[0]
            direction_responses: list[object] = []
            observed_sizes: list[int] = []
            for identities, compiled in self._iter_action_packets():
                responses = self._evaluate(
                    batch_factory=batch_factory,
                    compiled_masks=compiled,
                    lane_keys=tuple(value.lane_key for value in identities),
                )
                direction_responses.extend(responses)
                observed_sizes.append(len(responses))
            if tuple(observed_sizes) != self.forecast.lane_packet_sizes:
                raise RuntimeError(
                    "direction packet execution drifted from its forecast"
                )
            receipt = V1PhysicalActionContextExecutionReceipt(
                provider_receipt_hash=self.receipt.receipt_hash,
                context_packet_hash=packet_hash,
                parent_lane_key=self.receipt.parent_lane_key,
                direction_lane_hashes=tuple(
                    value.lane_hash for value in self.receipt.lane_identities
                ),
                direction_lane_keys=tuple(
                    value.lane_key for value in self.receipt.lane_identities
                ),
                direction_packet_sizes=tuple(observed_sizes),
                parent_lane_count=1,
                direction_lane_count=len(direction_responses),
                parent_evaluator_call_count=1,
                direction_evaluator_call_count=len(observed_sizes),
                total_evaluator_call_count=1 + len(observed_sizes),
                unique_physical_action_count=self.design.action_count,
                alias_count=sum(self.design.alias_counts),
            )
            return V1PhysicalActionContextExecution(
                receipt=receipt,
                parent_response=parent_response,
                direction_responses=tuple(direction_responses),
            )
        finally:
            self._active = False

    def close(self) -> None:
        if self._closed:
            return
        if self._active:
            _refuse(
                V1PhysicalActionDirectionRefusalCode.REENTRANT_EXECUTION,
                "cannot close a provider during context execution",
            )
        self._parent_gate = None
        self._closed = True

    def __enter__(self) -> "V1PhysicalActionDirectionProvider":
        self._require_live()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> bool:
        del exc_type, exc_value, traceback
        self.close()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("physical-action direction providers are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("physical-action direction providers are nonserializable")


def open_v1_physical_action_direction_provider(
    *,
    evaluator: object,
    design: V1PhysicalActionDirectionDesign,
    policy: V1PhysicalActionDirectionPolicy,
    route_blocks: Sequence[RouteBlockSpec],
    backend_kind: V1PhysicalActionDirectionBackend,
    production_execution: bool,
    current_reserved_bytes: int,
    device_total_memory_bytes: int,
    device: object | None = None,
) -> V1PhysicalActionDirectionProvider:
    """Open one nonserializable bounded direction-execution custody scope."""

    return V1PhysicalActionDirectionProvider(
        evaluator=evaluator,
        design=design,
        policy=policy,
        route_blocks=route_blocks,
        backend_kind=backend_kind,
        production_execution=production_execution,
        current_reserved_bytes=current_reserved_bytes,
        device_total_memory_bytes=device_total_memory_bytes,
        device=device,
    )


__all__ = [
    "PHYSICAL_ACTION_CONTEXT_EXECUTION_RECEIPT_SCHEMA",
    "PHYSICAL_ACTION_DIRECTION_ALGORITHM_ID",
    "PHYSICAL_ACTION_DIRECTION_AUTHORITY",
    "PHYSICAL_ACTION_DIRECTION_DESIGN_SCHEMA",
    "PHYSICAL_ACTION_DIRECTION_FORECAST_SCHEMA",
    "PHYSICAL_ACTION_DIRECTION_POLICY_SCHEMA",
    "PHYSICAL_ACTION_DIRECTION_RECEIPT_SCHEMA",
    "V1PhysicalActionContextExecution",
    "V1PhysicalActionContextExecutionReceipt",
    "V1PhysicalActionDirectionBackend",
    "V1PhysicalActionDirectionDesign",
    "V1PhysicalActionDirectionForecast",
    "V1PhysicalActionDirectionLaneIdentity",
    "V1PhysicalActionDirectionPolicy",
    "V1PhysicalActionDirectionProvider",
    "V1PhysicalActionDirectionReceipt",
    "V1PhysicalActionDirectionRefusal",
    "V1PhysicalActionDirectionRefusalCode",
    "open_v1_physical_action_direction_provider",
    "physical_action_direction_provider_source_hash",
    "register_v1_physical_action_direction_design",
]
