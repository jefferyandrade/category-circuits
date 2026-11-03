"""Compact source-native replay after the sampled-D action family freezes.

The device-resident sampled-D session deliberately does not retain a
context-by-route A/J bank.  Once a finite physical-action family and its
``B <= 8`` variable shell are frozen, the exact registered D prefix can be
replayed once into compact consumers instead.  This module owns that narrow
custody transition.

The durable objects here contain identities, digests, and cursors only.  Raw
rows remain source-owned transient tensors.  Corrected-A rows are fanned into
the finite signed-F action-family consumer and the variable-shell consumer;
literal-J rows are fanned into the variable-shell consumer and an honest
two-pass literal-J consumer.  Literal-J pass 2 must replay every pass-1 tile
digest exactly.  A source that can only perform a one-pass ``Mv`` is therefore
not admissible.

Production execution is CUDA-only and permits neither route-scale D2H nor row
scratch.  The bounded CPU mode exists solely for deterministic protocol tests.
Canonical source issuance is available only through the exact-owner row-free
device-final bridge; hash-shaped source surrogates and historical row-backed
compatibility objects cannot mint canonical provenance.
No receipt in this module authorizes D stability, C/K/O, mask commit, runtime,
terminal validation, or a Pareto claim.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
import hashlib
import math
from numbers import Integral
from typing import Any, Final, NoReturn, Sequence

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_literal_j_two_pass_d_fixed_projection_cuda import (
    V1LiteralJTwoPassDFixedProjectionCUDAAccumulator,
    V1LiteralJTwoPassDFixedProjectionRegistration,
)
from .modern_v1_sampled_device_final_rank_source import (
    V1SampledDeviceFinalRankSourceReceipt,
)
from .modern_v1_sampled_device_resident_d_session import (
    V1SampledDeviceResidentDLookReceipt,
)
from .modern_v1_sampled_literal_j_fd_rank_replay import (
    V1SampledLiteralJOriginalCaptureBinding,
)
from .modern_v1_sampled_v7_affine_shell_preoutcome_fanout import (
    V1SampledV7AffineShellPreoutcomeFanoutReceipt,
)
from .modern_v1_sampled_v7_branch_action_pair import (
    V1SampledV7PhysicalActionCustody,
)
from .modern_v1_signed_work_variable_shell_source_cuda import (
    V1SignedWorkVariableShellSourceCUDAAccumulator,
)

try:  # pragma: no cover - CUDA availability is host specific.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


COMPACT_POST_D_SOURCE_LEDGER_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_compact_post_D_source_ledger_v1"
)
COMPACT_POST_D_DEVICE_FINAL_SOURCE_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_compact_device_resident_D_final_source_v1"
)
COMPACT_POST_D_DEVICE_FINAL_SOURCE_SCHEMA_V2: Final = (
    "partition_free_v1_sampled_v7_compact_device_resident_D_"
    "row_free_final_source_v2"
)
COMPACT_POST_D_SOURCE_LEDGER_SCHEMA_V2: Final = (
    "partition_free_v1_sampled_v7_compact_post_D_row_free_source_ledger_v2"
)
COMPACT_POST_D_ACTION_REGISTRY_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_compact_post_D_frozen_action_registry_v1"
)
COMPACT_POST_D_DESIGN_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_compact_post_D_replay_design_v1"
)
COMPACT_POST_D_PLAN_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_compact_post_D_replay_plan_v1"
)
COMPACT_POST_D_CHECKPOINT_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_compact_post_D_replay_checkpoint_v1"
)
COMPACT_POST_D_EXECUTION_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_compact_post_D_replay_execution_v1"
)
COMPACT_POST_D_AUTHORITY_SCOPE: Final = (
    "post_D_compact_source_replay_custody_only_no_D_stability_C_K_O_commit_"
    "runtime_terminal_validation_or_pareto_authority_v1"
)
COMPACT_POST_D_LITERAL_J_EXACTNESS: Final = (
    "literal_J_pass2_replays_digest_identical_pass1_packet_route_tiles_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-v7-compact-post-D-replay-"
_MAXIMUM_ACTIONS: Final = 8
_MAXIMUM_SHELL: Final = 8
_SOURCE_TOKEN: Final = object()
_ACTION_TOKEN: Final = object()
_DESIGN_TOKEN: Final = object()
_PLAN_TOKEN: Final = object()
_CHECKPOINT_TOKEN: Final = object()
_EXECUTION_TOKEN: Final = object()
_PACKET_TOKEN: Final = object()
_FROZEN_ACTION_TOKEN: Final = object()
_DEVICE_FINAL_SOURCE_TOKEN: Final = object()
_DEVICE_FINAL_SOURCE_V2_TOKEN: Final = object()
_SOURCE_V2_TOKEN: Final = object()
_EMPTY_ROLLING_HASH: Final = content_hash(
    {
        "schema": "partition_free_v1_sampled_v7_compact_post_D_empty_event_axis_v1",
        "events": (),
    },
    prefix=_PREFIX + "event-chain-",
)


def _stratum_weight_axis(
    names: object,
    weights: object,
    population_counts: object,
    registered_counts: object,
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[int, ...], tuple[int, ...]]:
    stratum_names = _identity_axis(names, name="stratum names", unique=True)
    try:
        stratum_weights = tuple(float(value) for value in weights)  # type: ignore[union-attr]
        population = tuple(
            _positive_int(value, name="population stratum count")
            for value in population_counts  # type: ignore[union-attr]
        )
        registered = tuple(
            _positive_int(value, name="registered stratum count")
            for value in registered_counts  # type: ignore[union-attr]
        )
    except (TypeError, ValueError) as error:
        raise V1SampledV7CompactPostDReplayRefusal(
            V1SampledV7CompactPostDReplayRefusalCode.INVALID_TYPE,
            "stratum weights/counts must be finite exact axes",
        ) from error
    if (
        len({
            len(stratum_names),
            len(stratum_weights),
            len(population),
            len(registered),
        })
        != 1
        or any(not math.isfinite(value) or value <= 0.0 for value in stratum_weights)
        or not math.isclose(
            math.fsum(stratum_weights),
            1.0,
            rel_tol=0.0,
            abs_tol=128.0 * math.ulp(1.0),
        )
        or any(observed > total for observed, total in zip(registered, population, strict=True))
    ):
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.SOURCE_DRIFT,
            "stratum names, weights, population counts, and registered counts do not align",
        )
    return stratum_names, stratum_weights, population, registered


def v1_sampled_v7_compact_sampling_measure_hash(
    *,
    fd_design_hash: str,
    stratum_names: tuple[str, ...],
    stratum_weights: tuple[float, ...],
    population_stratum_counts: tuple[int, ...],
    registered_stratum_counts: tuple[int, ...],
    population_order_axis_hash: str,
) -> str:
    """Bind the finite-population estimand used by signed-work replay."""

    names, weights, population, registered = _stratum_weight_axis(
        stratum_names,
        stratum_weights,
        population_stratum_counts,
        registered_stratum_counts,
    )
    return content_hash(
        {
            "schema": "partition_free_v1_sampled_v7_compact_sampling_measure_v1",
            "fd_design_hash": _identity(fd_design_hash, name="FD design hash"),
            "stratum_names": names,
            "stratum_weights": weights,
            "population_stratum_counts": population,
            "registered_stratum_counts": registered,
            "population_order_axis_hash": _identity(
                population_order_axis_hash,
                name="population order axis hash",
            ),
            "context_weight_rule": "stratum_weight_divided_by_registered_stratum_count_v1",
        },
        prefix=_PREFIX + "sampling-measure-",
    )


class V1SampledV7CompactPostDReplayExecutionMode(str, Enum):
    CUDA_PRODUCTION = "cuda_production"
    CPU_REFERENCE_TEST_ONLY = "cpu_reference_test_only"


class V1SampledV7CompactPostDSourceProvenance(str, Enum):
    SOURCE_NATIVE_COMPACT = "source_native_compact"
    LEGACY_ROW_BACKED_COMPATIBILITY = "legacy_row_backed_compatibility"
    REFERENCE_TEST_ONLY = "reference_test_only"


class V1SampledV7CompactPostDLiteralJReplayMode(str, Enum):
    GLOBAL_DIGEST_IDENTICAL_TWO_PASS = "global_digest_identical_two_pass"
    PACKET_INTERLEAVED_DIGEST_IDENTICAL_TWO_PASS = (
        "packet_interleaved_digest_identical_two_pass"
    )


class V1SampledV7CompactPostDReplayRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    SOURCE_NOT_FINAL = "source_not_final"
    SOURCE_DRIFT = "source_drift"
    ACTION_REGISTRY_NOT_FROZEN = "action_registry_not_frozen"
    ACTION_DRIFT = "action_drift"
    SHELL_TOO_LARGE = "shell_too_large"
    PACKET_DRIFT = "packet_drift"
    TILE_DRIFT = "tile_drift"
    PASS_DRIFT = "pass_drift"
    CUDA_REQUIRED = "cuda_required"
    CPU_REFERENCE_SCALE_EXCEEDED = "cpu_reference_scale_exceeded"
    SOURCE_DIGEST_REQUIRED = "source_digest_required"
    ROW_TRANSFER_FORBIDDEN = "row_transfer_forbidden"
    RESTART_MISMATCH = "restart_mismatch"
    PARTIAL_REPLAY = "partial_replay"
    CONSUMER_MISMATCH = "consumer_mismatch"
    CONSUMER_FAILURE = "consumer_failure"
    AUTHORITY_VIOLATION = "authority_violation"
    MUTATED_RECEIPT = "mutated_receipt"


class V1SampledV7CompactPostDReplayRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledV7CompactPostDReplayRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7CompactPostDReplayRefusalCode:
            raise TypeError("compact post-D refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


class V1SampledV7CompactPostDReplayInterrupted(RuntimeError):
    """Testable failure-atomic interruption carrying one compact cursor."""

    def __init__(
        self,
        message: str,
        *,
        checkpoint: "V1SampledV7CompactPostDReplayCheckpoint",
    ) -> None:
        self.checkpoint = checkpoint
        super().__init__(message)


def _refuse(
    code: V1SampledV7CompactPostDReplayRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7CompactPostDReplayRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty identity",
        )
    return value


def _identity_axis(
    values: object,
    *,
    name: str,
    unique: bool,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.INVALID_TYPE,
            f"{name} must be one ordered identity axis",
        )
    try:
        result = tuple(_identity(value, name=f"{name} member") for value in values)  # type: ignore[union-attr]
    except TypeError as error:
        raise V1SampledV7CompactPostDReplayRefusal(
            V1SampledV7CompactPostDReplayRefusalCode.INVALID_TYPE,
            f"{name} must be iterable",
        ) from error
    if (not result and not allow_empty) or (unique and len(set(result)) != len(result)):
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.SOURCE_DRIFT,
            f"{name} is empty or contains duplicate identities",
        )
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _payload(value: object) -> dict[str, object]:
    return {
        item.name: getattr(value, item.name)
        for item in fields(value)
        if not item.name.startswith("_")
    }


def _axis_hash(role: str, values: Sequence[object]) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_sampled_v7_compact_post_D_axis_v1",
            "role": role,
            "values": tuple(values),
        },
        prefix=_PREFIX + role + "-axis-",
    )


def _authority_closed(value: object) -> None:
    for name in (
        "d_stability_authority",
        "c_stream_authority",
        "calibration_authority",
        "oracle_authority",
        "action_authority",
        "mask_commit_authority",
        "runtime_authority",
        "terminal_authority",
        "validation_authority",
        "pareto_authority",
    ):
        if getattr(value, name, None) is not False:
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.AUTHORITY_VIOLATION,
                f"{name} must remain closed",
            )


@dataclass(frozen=True, slots=True)
class V1SampledV7CompactPostDReplayPacket:
    source_packet_ordinal: int
    logical_look_index: int
    packet_ordinal_within_logical_look: int
    packet_plan_hash: str
    source_capture_identity_hash: str
    provider_proof_hash: str
    raw_dual_proof_hash: str
    same_capture_vjp_source_hash: str
    context_ids: tuple[str, ...]
    draw_ids: tuple[str, ...]
    measurement_ids: tuple[str, ...]
    capture_ids: tuple[str, ...]
    source_packet_hashes: tuple[str, ...]
    stratum_labels: tuple[str, ...]
    source_native_compact: bool
    reference_test_only: bool
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str = field(init=False, default="", repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _PACKET_TOKEN:
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.AUTHORITY_VIOLATION,
                "replay packet descriptors are source-ledger issued",
            )
        object.__setattr__(
            self,
            "source_packet_ordinal",
            _nonnegative_int(
                self.source_packet_ordinal,
                name="global source packet ordinal",
            ),
        )
        object.__setattr__(
            self,
            "packet_ordinal_within_logical_look",
            _nonnegative_int(
                self.packet_ordinal_within_logical_look,
                name="packet ordinal within logical look",
            ),
        )
        object.__setattr__(
            self,
            "logical_look_index",
            _positive_int(self.logical_look_index, name="logical look index"),
        )
        for name in (
            "packet_plan_hash",
            "source_capture_identity_hash",
            "provider_proof_hash",
            "raw_dual_proof_hash",
            "same_capture_vjp_source_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        axes = {}
        for name, unique in (
            ("context_ids", True),
            ("draw_ids", True),
            ("measurement_ids", True),
            ("capture_ids", True),
            ("source_packet_hashes", False),
            ("stratum_labels", False),
        ):
            axis = _identity_axis(getattr(self, name), name=name, unique=unique)
            object.__setattr__(self, name, axis)
            axes[name] = axis
        if len({len(axis) for axis in axes.values()}) != 1:
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.PACKET_DRIFT,
                "packet measurement, context, draw, capture, source, and stratum axes must align",
            )
        if (
            type(self.source_native_compact) is not bool
            or type(self.reference_test_only) is not bool
            or (self.source_native_compact, self.reference_test_only)
            not in {(True, False), (False, True), (False, False)}
        ):
            # Legacy compatibility packets set both flags false.  A packet
            # may never claim source-native and reference provenance together.
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.AUTHORITY_VIOLATION,
                "packet mixed source-native and reference provenance",
            )
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    @property
    def context_count(self) -> int:
        return len(self.context_ids)

    def _compute_hash(self) -> str:
        return content_hash(_payload(self), prefix=_PREFIX + "packet-")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash or (
            expected_hash is not None and observed != expected_hash
        ):
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.MUTATED_RECEIPT,
                "compact replay packet changed after source registration",
            )
        return observed

    @property
    def packet_hash(self) -> str:
        return self.assert_integrity()


def _issue_v1_sampled_v7_compact_post_d_source_native_packet(
    *,
    source_packet_ordinal: int,
    look_receipt: V1SampledDeviceResidentDLookReceipt,
    context_ids: Sequence[str],
    draw_ids: Sequence[str],
    measurement_ids: Sequence[str],
    capture_ids: Sequence[str],
    source_packet_hashes: Sequence[str],
    stratum_labels: Sequence[str],
    same_capture_chunk_source_hashes: Sequence[str],
    packet_plan_hash: str,
    source_capture_identity_hash: str,
    provider_proof_hash: str,
    raw_dual_proof_hash: str,
    same_capture_vjp_source_hash: str,
) -> V1SampledV7CompactPostDReplayPacket:
    """Issue one row-free packet from exact request/provider/capture fields.

    The context, draw, measurement, and same-capture route-window axes must
    reproduce the already issued device-session look receipt.  The remaining
    identities are compact request/provider facts supplied by the exact live
    source parent; no row attachment or manifest is accepted by this API.
    """

    if type(look_receipt) is not V1SampledDeviceResidentDLookReceipt:
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.INVALID_TYPE,
            "source-native packet issuance requires an exact device-D look receipt",
        )
    look_receipt.assert_integrity()
    contexts = _identity_axis(context_ids, name="context IDs", unique=True)
    draws = _identity_axis(draw_ids, name="draw IDs", unique=True)
    measurements = _identity_axis(
        measurement_ids, name="measurement IDs", unique=True
    )
    captures = _identity_axis(capture_ids, name="capture IDs", unique=True)
    source_packets = _identity_axis(
        source_packet_hashes, name="source packet hashes", unique=False
    )
    strata = _identity_axis(stratum_labels, name="stratum labels", unique=False)
    chunk_sources = _identity_axis(
        same_capture_chunk_source_hashes,
        name="same-capture chunk source hashes",
        unique=True,
    )
    prefix = "partition-free-v1-sampled-device-resident-D-"
    if (
        len({len(axis) for axis in (contexts, draws, measurements, captures, source_packets, strata)})
        != 1
        or content_hash(contexts, prefix=prefix + "look-context-axis-")
        != look_receipt.context_axis_hash
        or content_hash(draws, prefix=prefix + "look-draw-axis-")
        != look_receipt.draw_axis_hash
        or content_hash(measurements, prefix=prefix + "look-source-measurement-axis-")
        != look_receipt.source_measurement_axis_hash
        or content_hash(chunk_sources, prefix=prefix + "same-capture-chunk-axis-")
        != look_receipt.same_capture_chunk_source_axis_hash
        or any(label != look_receipt.stratum_name for label in strata)
    ):
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.SOURCE_DRIFT,
            "source-native request/provider axes do not replay their device-D look",
        )
    return V1SampledV7CompactPostDReplayPacket(
        source_packet_ordinal=source_packet_ordinal,
        logical_look_index=look_receipt.registered_look_index,
        packet_ordinal_within_logical_look=look_receipt.packet_ordinal,
        packet_plan_hash=packet_plan_hash,
        source_capture_identity_hash=source_capture_identity_hash,
        provider_proof_hash=provider_proof_hash,
        raw_dual_proof_hash=raw_dual_proof_hash,
        same_capture_vjp_source_hash=same_capture_vjp_source_hash,
        context_ids=contexts,
        draw_ids=draws,
        measurement_ids=measurements,
        capture_ids=captures,
        source_packet_hashes=source_packets,
        stratum_labels=strata,
        source_native_compact=True,
        reference_test_only=False,
        _issuance_token=_PACKET_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7CompactDeviceResidentDFinalSourceReceipt:
    """Exact row-free bridge receipt required by canonical replay.

    This module intentionally exposes no hash-shaped constructor for this
    type.  It can be issued only after the exact bridge integration consumes
    the exact live device-session materialization, materialization-rank
    receipt, registered-D transaction and prefix binding, plus every ordered
    request/result/provider/capture proof.  Merely supplying their hashes is
    not evidence that the source-native, zero-row path executed.
    """

    source_parent_receipt_hash: str
    session_design_hash: str
    hbm_forecast_hash: str
    session_materialization_hash: str
    materialization_rank_receipt_hash: str
    registered_d_transaction_hash: str
    registered_d_prefix_binding_hash: str
    registered_request_prefix_hash: str
    literal_j_source_receipt_hash: str
    fd_design_hash: str
    fd_snapshot_hash: str
    corrected_moments_state_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    corrected_a_stream_identity_hash: str
    literal_j_stream_identity_hash: str
    registered_look_count: int
    context_axis_hash: str
    draw_axis_hash: str
    stratum_axis_hash: str
    measurement_axis_hash: str
    packet_axis_hash: str
    population_count: int
    population_context_axis_hash: str
    population_draw_axis_hash: str
    population_order_axis_hash: str
    packets: tuple[V1SampledV7CompactPostDReplayPacket, ...]
    source_native_compact: bool = True
    row_binding_present: bool = False
    row_manifest_present: bool = False
    row_attachment_present: bool = False
    raw_row_d2h_bytes: int = 0
    raw_row_scratch_bytes: int = 0
    downstream_authority: bool = False
    schema: str = COMPACT_POST_D_DEVICE_FINAL_SOURCE_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str = field(init=False, default="", repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _DEVICE_FINAL_SOURCE_TOKEN:
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.AUTHORITY_VIOLATION,
                "device-final source receipts require the exact-owner bridge issuer",
            )
        for item in fields(self):
            if not item.name.startswith("_") and item.name.endswith("_hash"):
                object.__setattr__(
                    self,
                    item.name,
                    _identity(getattr(self, item.name), name=item.name),
                )
        packets = tuple(self.packets)
        if (
            not packets
            or any(type(packet) is not V1SampledV7CompactPostDReplayPacket for packet in packets)
            or tuple(packet.source_packet_ordinal for packet in packets)
            != tuple(range(len(packets)))
            or max(packet.logical_look_index for packet in packets)
            != _positive_int(self.registered_look_count, name="registered look count")
            or _positive_int(self.population_count, name="population count")
            < sum(packet.context_count for packet in packets)
            or _positive_int(self.route_count, name="route count") <= 0
            or self.corrected_a_stream_identity_hash
            == self.literal_j_stream_identity_hash
            or any(
                packet.source_native_compact is not True
                or packet.reference_test_only is not False
                for packet in packets
            )
        ):
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.SOURCE_DRIFT,
                "source-native parent packet/look/population axes do not close",
            )
        if (
            self.source_native_compact is not True
            or self.row_binding_present is not False
            or self.row_manifest_present is not False
            or self.row_attachment_present is not False
            or self.raw_row_d2h_bytes != 0
            or self.raw_row_scratch_bytes != 0
            or self.downstream_authority is not False
            or self.schema != COMPACT_POST_D_DEVICE_FINAL_SOURCE_SCHEMA
        ):
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.AUTHORITY_VIOLATION,
                "device-final source receipt smuggled row custody or authority",
            )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        object.__setattr__(self, "packets", packets)
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        value = _payload(self)
        value["packets"] = tuple(packet.packet_hash for packet in self.packets)
        return content_hash(value, prefix=_PREFIX + "source-native-parent-")

    def assert_integrity(self) -> str:
        for packet in self.packets:
            packet.assert_integrity()
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.MUTATED_RECEIPT,
                "device-final source receipt changed after issuance",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1SampledV7CompactDeviceResidentDRowFreeFinalSourceReceiptV2:
    """Canonical source parent with scalar D custody and no portable arrays."""

    source_parent_receipt_hash: str
    session_design_hash: str
    hbm_forecast_hash: str
    row_free_finalization_receipt_hash: str
    materialization_rank_receipt_hash: str
    registered_d_transaction_hash: str
    registered_d_prefix_binding_hash: str
    registered_request_prefix_hash: str
    literal_j_source_receipt_hash: str
    fd_design_hash: str
    corrected_reacquisition_source_hash: str
    live_fd_source_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    corrected_a_stream_identity_hash: str
    literal_j_stream_identity_hash: str
    registered_look_count: int
    context_axis_hash: str
    draw_axis_hash: str
    stratum_axis_hash: str
    measurement_axis_hash: str
    packet_axis_hash: str
    population_count: int
    population_context_axis_hash: str
    population_draw_axis_hash: str
    population_order_axis_hash: str
    sampling_measure_hash: str
    stratum_names: tuple[str, ...]
    stratum_weights: tuple[float, ...]
    population_stratum_counts: tuple[int, ...]
    registered_stratum_counts: tuple[int, ...]
    packets: tuple[V1SampledV7CompactPostDReplayPacket, ...]
    source_native_compact: bool = True
    corrected_f_reacquisition_required: bool = True
    live_device_fd_owner_required: bool = True
    portable_corrected_moments_present: bool = False
    portable_fd_snapshot_present: bool = False
    row_binding_present: bool = False
    row_manifest_present: bool = False
    row_attachment_present: bool = False
    route_scale_d2h_bytes: int = 0
    raw_row_scratch_bytes: int = 0
    downstream_authority: bool = False
    schema: str = COMPACT_POST_D_DEVICE_FINAL_SOURCE_SCHEMA_V2
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str = field(init=False, default="", repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _DEVICE_FINAL_SOURCE_V2_TOKEN:
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.AUTHORITY_VIOLATION,
                "row-free v2 source receipts require the exact scalar bridge",
            )
        for item in fields(self):
            if not item.name.startswith("_") and item.name.endswith("_hash"):
                object.__setattr__(
                    self,
                    item.name,
                    _identity(getattr(self, item.name), name=item.name),
                )
        packets = tuple(self.packets)
        names, weights, population, registered = _stratum_weight_axis(
            self.stratum_names,
            self.stratum_weights,
            self.population_stratum_counts,
            self.registered_stratum_counts,
        )
        observed_registered = tuple(
            sum(label == name for packet in packets for label in packet.stratum_labels)
            for name in names
        )
        expected_measure_hash = v1_sampled_v7_compact_sampling_measure_hash(
            fd_design_hash=self.fd_design_hash,
            stratum_names=names,
            stratum_weights=weights,
            population_stratum_counts=population,
            registered_stratum_counts=registered,
            population_order_axis_hash=self.population_order_axis_hash,
        )
        if (
            not packets
            or any(type(packet) is not V1SampledV7CompactPostDReplayPacket for packet in packets)
            or tuple(packet.source_packet_ordinal for packet in packets)
            != tuple(range(len(packets)))
            or max(packet.logical_look_index for packet in packets)
            != _positive_int(self.registered_look_count, name="registered look count")
            or _positive_int(self.population_count, name="population count")
            < sum(packet.context_count for packet in packets)
            or sum(population) != self.population_count
            or sum(registered) != sum(packet.context_count for packet in packets)
            or registered != observed_registered
            or self.sampling_measure_hash != expected_measure_hash
            or _positive_int(self.route_count, name="route count") <= 0
            or self.route_axis_hash == self.route_id_axis_hash
            or self.corrected_a_stream_identity_hash
            == self.literal_j_stream_identity_hash
        ):
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.SOURCE_DRIFT,
                "row-free v2 source packet/look/route axes do not close",
            )
        flags = (
            self.source_native_compact,
            self.corrected_f_reacquisition_required,
            self.live_device_fd_owner_required,
            self.portable_corrected_moments_present,
            self.portable_fd_snapshot_present,
            self.row_binding_present,
            self.row_manifest_present,
            self.row_attachment_present,
            self.downstream_authority,
        )
        if (
            flags[:3] != (True, True, True)
            or any(type(flag) is not bool for flag in flags)
            or any(flags[3:])
            or self.route_scale_d2h_bytes != 0
            or self.raw_row_scratch_bytes != 0
            or self.schema != COMPACT_POST_D_DEVICE_FINAL_SOURCE_SCHEMA_V2
        ):
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.AUTHORITY_VIOLATION,
                "row-free v2 source retained portable arrays, rows, or authority",
            )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        for packet in packets:
            packet.assert_integrity()
        object.__setattr__(self, "stratum_names", names)
        object.__setattr__(self, "stratum_weights", weights)
        object.__setattr__(self, "population_stratum_counts", population)
        object.__setattr__(self, "registered_stratum_counts", registered)
        object.__setattr__(self, "packets", packets)
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        value = _payload(self)
        value["packets"] = tuple(packet.packet_hash for packet in self.packets)
        return content_hash(value, prefix=_PREFIX + "row-free-source-native-parent-v2-")

    def assert_integrity(self) -> str:
        for packet in self.packets:
            packet.assert_integrity()
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.MUTATED_RECEIPT,
                "row-free v2 source changed after issuance",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


def issue_v1_sampled_v7_compact_device_final_source_from_exact_bridge(
    *,
    exact_bridge: object,
) -> V1SampledV7CompactDeviceResidentDFinalSourceReceipt:
    """Issue the canonical parent solely from the exact row-free bridge.

    The lazy import is intentional: the bridge consumes this module's private
    packet/final-source issuers, while this function must exact-type-check the
    bridge before it can cross that boundary.  Hash-shaped lookalikes cannot
    mint canonical source-native provenance.
    """

    from .modern_v1_sampled_v7_compact_device_resident_d_bridge import (
        V1SampledV7CompactDeviceResidentDFinalSourceBridge,
    )

    if type(exact_bridge) is not V1SampledV7CompactDeviceResidentDFinalSourceBridge:
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.SOURCE_NOT_FINAL,
            "canonical device-final issuance requires the exact row-free bridge owner",
        )
    bridge = exact_bridge
    bridge.assert_integrity()
    packets = []
    device_chunk_prefix = (
        "partition-free-v1-sampled-device-resident-D-same-capture-chunk-"
    )
    for registration, boundary, look in zip(
        bridge.packet_registrations,
        bridge.packet_boundaries,
        bridge.look_receipts,
        strict=True,
    ):
        chunk_sources = tuple(
            content_hash(
                {
                    "schema": "partition_free_v1_device_resident_D_same_capture_chunk_v1",
                    "transaction_request_hash": registration.transaction_request_hash,
                    "packet_plan_hash": registration.packet_plan_hash,
                    "packet_ordinal": (
                        registration.packet_ordinal_within_logical_look
                    ),
                    "route_window": (int(start), int(stop)),
                    "corrected_evidence_stream_identity_hash": (
                        registration.corrected_a_stream_identity_hash
                    ),
                    "literal_jacobian_stream_identity_hash": (
                        registration.literal_j_stream_identity_hash
                    ),
                    "same_capture_and_vjp_required": True,
                    "route_scale_values_not_hashed": True,
                },
                prefix=device_chunk_prefix,
            )
            for start, stop in boundary.route_windows
        )
        capture_ids = tuple(
            content_hash(
                {
                    "schema": "partition_free_v1_sampled_v7_compact_capture_coordinate_v1",
                    "source_capture_identity_hash": (
                        registration.source_capture_identity_hash
                    ),
                    "context_id": context,
                    "draw_id": draw,
                },
                prefix=_PREFIX + "capture-coordinate-",
            )
            for context, draw in zip(
                registration.context_ids,
                registration.draw_ids,
                strict=True,
            )
        )
        packets.append(
            _issue_v1_sampled_v7_compact_post_d_source_native_packet(
                source_packet_ordinal=registration.source_packet_ordinal,
                look_receipt=look,
                context_ids=registration.context_ids,
                draw_ids=registration.draw_ids,
                measurement_ids=registration.source_measurement_hashes,
                capture_ids=capture_ids,
                source_packet_hashes=(
                    (registration.provider_plan_identity_hash,)
                    * len(registration.context_ids)
                ),
                stratum_labels=registration.stratum_labels,
                same_capture_chunk_source_hashes=chunk_sources,
                packet_plan_hash=registration.packet_plan_hash,
                source_capture_identity_hash=(
                    registration.source_capture_identity_hash
                ),
                provider_proof_hash=boundary.provider_proof_hash,
                raw_dual_proof_hash=boundary.raw_dual_proof_hash,
                same_capture_vjp_source_hash=(
                    boundary.same_capture_vjp_source_hash
                ),
            )
        )
    packet_axis = tuple(
        (
            packet.packet_plan_hash,
            packet.source_capture_identity_hash,
            packet.provider_proof_hash,
            packet.raw_dual_proof_hash,
            packet.same_capture_vjp_source_hash,
        )
        for packet in packets
    )
    if content_hash(packet_axis, prefix="partition-free-v1-literal-j-replay-packet-axis-") != bridge.packet_axis_hash:
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.SOURCE_DRIFT,
            "exact bridge packet/proof axis changed during final-source issuance",
        )
    return V1SampledV7CompactDeviceResidentDFinalSourceReceipt(
        source_parent_receipt_hash=bridge.bridge_hash,
        session_design_hash=bridge.session_design_hash,
        hbm_forecast_hash=bridge.hbm_forecast_hash,
        session_materialization_hash=bridge.session_materialization_hash,
        materialization_rank_receipt_hash=(
            bridge.materialization_rank_receipt_hash
        ),
        registered_d_transaction_hash=bridge.registered_d_transaction_hash,
        registered_d_prefix_binding_hash=(
            bridge.registered_d_prefix_binding_hash
        ),
        registered_request_prefix_hash=bridge.registered_request_prefix_hash,
        literal_j_source_receipt_hash=bridge.literal_j_source_receipt_hash,
        fd_design_hash=bridge.fd_design_hash,
        fd_snapshot_hash=bridge.fd_snapshot_hash,
        corrected_moments_state_hash=bridge.corrected_moments_state_hash,
        chart_identity_hash=bridge.chart_identity_hash,
        current_mask_hash=bridge.current_mask_hash,
        proposal_generation=bridge.proposal_generation,
        route_axis_hash=bridge.route_axis_hash,
        route_id_axis_hash=bridge.route_id_axis_hash,
        route_count=bridge.route_count,
        corrected_a_stream_identity_hash=(
            bridge.corrected_a_stream_identity_hash
        ),
        literal_j_stream_identity_hash=bridge.literal_j_stream_identity_hash,
        registered_look_count=bridge.registered_look_count,
        context_axis_hash=bridge.context_axis_hash,
        draw_axis_hash=bridge.draw_axis_hash,
        stratum_axis_hash=bridge.stratum_axis_hash,
        measurement_axis_hash=bridge.measurement_axis_hash,
        packet_axis_hash=bridge.packet_axis_hash,
        population_count=bridge.population_count,
        population_context_axis_hash=bridge.population_context_axis_hash,
        population_draw_axis_hash=bridge.population_draw_axis_hash,
        population_order_axis_hash=bridge.population_order_axis_hash,
        packets=tuple(packets),
        _issuance_token=_DEVICE_FINAL_SOURCE_TOKEN,
    )


def issue_v1_sampled_v7_compact_row_free_final_source_from_exact_bridge_v2(
    *,
    exact_bridge: object,
) -> V1SampledV7CompactDeviceResidentDRowFreeFinalSourceReceiptV2:
    """Issue v2 solely from the exact scalar/live-device bridge owner."""

    from .modern_v1_sampled_v7_compact_device_resident_d_bridge import (
        V1SampledV7CompactDeviceResidentDRowFreeFinalSourceBridgeV2,
    )

    if type(exact_bridge) is not V1SampledV7CompactDeviceResidentDRowFreeFinalSourceBridgeV2:
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.SOURCE_NOT_FINAL,
            "row-free v2 issuance requires the exact scalar bridge owner",
        )
    bridge = exact_bridge
    bridge.assert_integrity()
    packets = []
    device_chunk_prefix = (
        "partition-free-v1-sampled-device-resident-D-same-capture-chunk-"
    )
    for registration, boundary, look in zip(
        bridge.packet_registrations,
        bridge.packet_boundaries,
        bridge.look_receipts,
        strict=True,
    ):
        chunk_sources = tuple(
            content_hash(
                {
                    "schema": "partition_free_v1_device_resident_D_same_capture_chunk_v1",
                    "transaction_request_hash": registration.transaction_request_hash,
                    "packet_plan_hash": registration.packet_plan_hash,
                    "packet_ordinal": registration.packet_ordinal_within_logical_look,
                    "route_window": (int(start), int(stop)),
                    "corrected_evidence_stream_identity_hash": (
                        registration.corrected_a_stream_identity_hash
                    ),
                    "literal_jacobian_stream_identity_hash": (
                        registration.literal_j_stream_identity_hash
                    ),
                    "same_capture_and_vjp_required": True,
                    "route_scale_values_not_hashed": True,
                },
                prefix=device_chunk_prefix,
            )
            for start, stop in boundary.route_windows
        )
        capture_ids = tuple(
            content_hash(
                {
                    "schema": "partition_free_v1_sampled_v7_compact_capture_coordinate_v1",
                    "source_capture_identity_hash": registration.source_capture_identity_hash,
                    "context_id": context,
                    "draw_id": draw,
                },
                prefix=_PREFIX + "capture-coordinate-",
            )
            for context, draw in zip(
                registration.context_ids,
                registration.draw_ids,
                strict=True,
            )
        )
        packets.append(
            _issue_v1_sampled_v7_compact_post_d_source_native_packet(
                source_packet_ordinal=registration.source_packet_ordinal,
                look_receipt=look,
                context_ids=registration.context_ids,
                draw_ids=registration.draw_ids,
                measurement_ids=registration.source_measurement_hashes,
                capture_ids=capture_ids,
                source_packet_hashes=(
                    (registration.provider_plan_identity_hash,)
                    * len(registration.context_ids)
                ),
                stratum_labels=registration.stratum_labels,
                same_capture_chunk_source_hashes=chunk_sources,
                packet_plan_hash=registration.packet_plan_hash,
                source_capture_identity_hash=registration.source_capture_identity_hash,
                provider_proof_hash=boundary.provider_proof_hash,
                raw_dual_proof_hash=boundary.raw_dual_proof_hash,
                same_capture_vjp_source_hash=boundary.same_capture_vjp_source_hash,
            )
        )
    packet_axis = tuple(
        (
            packet.packet_plan_hash,
            packet.source_capture_identity_hash,
            packet.provider_proof_hash,
            packet.raw_dual_proof_hash,
            packet.same_capture_vjp_source_hash,
        )
        for packet in packets
    )
    if content_hash(
        packet_axis,
        prefix="partition-free-v1-literal-j-replay-packet-axis-",
    ) != bridge.packet_axis_hash:
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.SOURCE_DRIFT,
            "row-free v2 packet/proof axis changed during issuance",
        )
    return V1SampledV7CompactDeviceResidentDRowFreeFinalSourceReceiptV2(
        source_parent_receipt_hash=bridge.bridge_hash,
        session_design_hash=bridge.session_design_hash,
        hbm_forecast_hash=bridge.hbm_forecast_hash,
        row_free_finalization_receipt_hash=(
            bridge.row_free_finalization_receipt_hash
        ),
        materialization_rank_receipt_hash=bridge.materialization_rank_receipt_hash,
        registered_d_transaction_hash=bridge.registered_d_transaction_hash,
        registered_d_prefix_binding_hash=bridge.registered_d_prefix_binding_hash,
        registered_request_prefix_hash=bridge.registered_request_prefix_hash,
        literal_j_source_receipt_hash=bridge.literal_j_source_receipt_hash,
        fd_design_hash=bridge.fd_design_hash,
        corrected_reacquisition_source_hash=(
            bridge.corrected_reacquisition_source_hash
        ),
        live_fd_source_hash=bridge.live_fd_source_hash,
        chart_identity_hash=bridge.chart_identity_hash,
        current_mask_hash=bridge.current_mask_hash,
        proposal_generation=bridge.proposal_generation,
        route_axis_hash=bridge.route_axis_hash,
        route_id_axis_hash=bridge.route_id_axis_hash,
        route_count=bridge.route_count,
        corrected_a_stream_identity_hash=bridge.corrected_a_stream_identity_hash,
        literal_j_stream_identity_hash=bridge.literal_j_stream_identity_hash,
        registered_look_count=bridge.registered_look_count,
        context_axis_hash=bridge.context_axis_hash,
        draw_axis_hash=bridge.draw_axis_hash,
        stratum_axis_hash=bridge.stratum_axis_hash,
        measurement_axis_hash=bridge.measurement_axis_hash,
        packet_axis_hash=bridge.packet_axis_hash,
        population_count=bridge.population_count,
        population_context_axis_hash=bridge.population_context_axis_hash,
        population_draw_axis_hash=bridge.population_draw_axis_hash,
        population_order_axis_hash=bridge.population_order_axis_hash,
        sampling_measure_hash=bridge.sampling_measure_hash,
        stratum_names=bridge.stratum_names,
        stratum_weights=bridge.stratum_weights,
        population_stratum_counts=bridge.population_stratum_counts,
        registered_stratum_counts=bridge.registered_stratum_counts,
        packets=tuple(packets),
        _issuance_token=_DEVICE_FINAL_SOURCE_V2_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7CompactPostDReplaySourceLedger:
    source_parent_receipt_hash: str
    source_parent_integrity_hash: str
    session_design_hash: str
    hbm_forecast_hash: str
    session_materialization_hash: str
    materialization_rank_receipt_hash: str
    literal_j_compact_source_hash: str
    registered_d_transaction_hash: str
    registered_d_prefix_binding_hash: str
    registered_request_prefix_hash: str
    literal_j_source_receipt_hash: str
    fd_design_hash: str
    fd_snapshot_hash: str
    corrected_moments_state_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    corrected_a_stream_identity_hash: str
    literal_j_stream_identity_hash: str
    registered_look_count: int
    packet_count: int
    context_count: int
    context_axis_hash: str
    draw_axis_hash: str
    stratum_axis_hash: str
    measurement_axis_hash: str
    capture_axis_hash: str
    source_packet_axis_hash: str
    packet_axis_hash: str
    population_count: int
    population_context_axis_hash: str
    population_draw_axis_hash: str
    population_order_axis_hash: str
    packets: tuple[V1SampledV7CompactPostDReplayPacket, ...]
    source_provenance: V1SampledV7CompactPostDSourceProvenance
    canonical_zero_row_ordinary_path: bool
    reference_test_only: bool = False
    exact_registered_source_ledger: bool = True
    row_payload_present: bool = False
    raw_row_d2h_bytes: int = 0
    raw_row_scratch_bytes: int = 0
    d_stability_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    action_authority: bool = False
    mask_commit_authority: bool = False
    runtime_authority: bool = False
    terminal_authority: bool = False
    validation_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = COMPACT_POST_D_AUTHORITY_SCOPE
    schema: str = COMPACT_POST_D_SOURCE_LEDGER_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str = field(init=False, default="", repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _SOURCE_TOKEN:
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.AUTHORITY_VIOLATION,
                "source ledgers are issuer-only",
            )
        for item in fields(self):
            if not item.name.startswith("_") and item.name.endswith("_hash"):
                object.__setattr__(
                    self,
                    item.name,
                    _identity(getattr(self, item.name), name=item.name),
                )
        for name in (
            "registered_look_count",
            "packet_count",
            "context_count",
            "route_count",
            "population_count",
        ):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name=name)
            )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        packets = tuple(self.packets)
        if type(self.source_provenance) is not V1SampledV7CompactPostDSourceProvenance:
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.INVALID_TYPE,
                "source provenance tag must be exact",
            )
        expected_packet_provenance = {
            V1SampledV7CompactPostDSourceProvenance.SOURCE_NATIVE_COMPACT: (
                True,
                False,
            ),
            V1SampledV7CompactPostDSourceProvenance.LEGACY_ROW_BACKED_COMPATIBILITY: (
                False,
                False,
            ),
            V1SampledV7CompactPostDSourceProvenance.REFERENCE_TEST_ONLY: (
                False,
                True,
            ),
        }[self.source_provenance]
        if (
            not packets
            or any(type(packet) is not V1SampledV7CompactPostDReplayPacket for packet in packets)
            or tuple(packet.source_packet_ordinal for packet in packets)
            != tuple(range(len(packets)))
            or tuple(packet.logical_look_index for packet in packets)
            != tuple(sorted(packet.logical_look_index for packet in packets))
            or any(
                tuple(
                    packet.packet_ordinal_within_logical_look
                    for packet in packets
                    if packet.logical_look_index == look_index
                )
                != tuple(
                    range(
                        sum(
                            packet.logical_look_index == look_index
                            for packet in packets
                        )
                    )
                )
                for look_index in range(1, self.registered_look_count + 1)
            )
            or max(packet.logical_look_index for packet in packets)
            != self.registered_look_count
            or len(packets) != self.packet_count
            or sum(packet.context_count for packet in packets) != self.context_count
            or self.registered_look_count > self.packet_count
            or self.population_count < self.context_count
        ):
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.PACKET_DRIFT,
                "logical-look, packet, or population axes do not close",
            )
        for packet in packets:
            packet.assert_integrity()
        if (
            self.corrected_a_stream_identity_hash
            == self.literal_j_stream_identity_hash
            or self.canonical_zero_row_ordinary_path
            is not (
                self.source_provenance
                is V1SampledV7CompactPostDSourceProvenance.SOURCE_NATIVE_COMPACT
            )
            or self.reference_test_only
            is not (
                self.source_provenance
                is V1SampledV7CompactPostDSourceProvenance.REFERENCE_TEST_ONLY
            )
            or any(
                (packet.source_native_compact, packet.reference_test_only)
                != expected_packet_provenance
                for packet in packets
            )
            or self.exact_registered_source_ledger is not True
            or self.row_payload_present is not False
            or self.raw_row_d2h_bytes != 0
            or self.raw_row_scratch_bytes != 0
            or type(self.reference_test_only) is not bool
            or self.authority_scope != COMPACT_POST_D_AUTHORITY_SCOPE
            or self.schema != COMPACT_POST_D_SOURCE_LEDGER_SCHEMA
        ):
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.AUTHORITY_VIOLATION,
                "source ledger retained rows, mixed A/J, or crossed authority",
            )
        _authority_closed(self)
        object.__setattr__(self, "packets", packets)
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        value = _payload(self)
        value["packets"] = tuple(packet.packet_hash for packet in self.packets)
        return content_hash(value, prefix=_PREFIX + "source-ledger-")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for packet in self.packets:
            packet.assert_integrity()
        observed = self._compute_hash()
        if observed != self._sealed_hash or (
            expected_hash is not None and observed != expected_hash
        ):
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.MUTATED_RECEIPT,
                "source ledger changed after issuance",
            )
        return observed

    @property
    def ledger_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1SampledV7CompactPostDRowFreeSourceLedgerV2:
    """Compact replay ledger backed by scalar finalization and live FD state."""

    source_parent_receipt_hash: str
    source_parent_integrity_hash: str
    session_design_hash: str
    hbm_forecast_hash: str
    row_free_finalization_receipt_hash: str
    materialization_rank_receipt_hash: str
    registered_d_transaction_hash: str
    registered_d_prefix_binding_hash: str
    registered_request_prefix_hash: str
    literal_j_source_receipt_hash: str
    fd_design_hash: str
    corrected_reacquisition_source_hash: str
    live_fd_source_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    corrected_a_stream_identity_hash: str
    literal_j_stream_identity_hash: str
    registered_look_count: int
    packet_count: int
    context_count: int
    context_axis_hash: str
    draw_axis_hash: str
    stratum_axis_hash: str
    measurement_axis_hash: str
    capture_axis_hash: str
    source_packet_axis_hash: str
    packet_axis_hash: str
    population_count: int
    population_context_axis_hash: str
    population_draw_axis_hash: str
    population_order_axis_hash: str
    sampling_measure_hash: str
    stratum_names: tuple[str, ...]
    stratum_weights: tuple[float, ...]
    population_stratum_counts: tuple[int, ...]
    registered_stratum_counts: tuple[int, ...]
    packets: tuple[V1SampledV7CompactPostDReplayPacket, ...]
    corrected_f_reacquisition_required: bool = True
    live_device_fd_owner_required: bool = True
    portable_corrected_moments_present: bool = False
    portable_fd_snapshot_present: bool = False
    row_payload_present: bool = False
    route_scale_d2h_bytes: int = 0
    raw_row_scratch_bytes: int = 0
    d_stability_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    action_authority: bool = False
    mask_commit_authority: bool = False
    runtime_authority: bool = False
    terminal_authority: bool = False
    validation_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = COMPACT_POST_D_AUTHORITY_SCOPE
    schema: str = COMPACT_POST_D_SOURCE_LEDGER_SCHEMA_V2
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str = field(init=False, default="", repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _SOURCE_V2_TOKEN:
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.AUTHORITY_VIOLATION,
                "row-free v2 ledgers are exact-source issued",
            )
        for item in fields(self):
            if not item.name.startswith("_") and item.name.endswith("_hash"):
                object.__setattr__(
                    self,
                    item.name,
                    _identity(getattr(self, item.name), name=item.name),
                )
        packets = tuple(self.packets)
        names, weights, population, registered = _stratum_weight_axis(
            self.stratum_names,
            self.stratum_weights,
            self.population_stratum_counts,
            self.registered_stratum_counts,
        )
        observed_registered = tuple(
            sum(label == name for packet in packets for label in packet.stratum_labels)
            for name in names
        )
        expected_measure_hash = v1_sampled_v7_compact_sampling_measure_hash(
            fd_design_hash=self.fd_design_hash,
            stratum_names=names,
            stratum_weights=weights,
            population_stratum_counts=population,
            registered_stratum_counts=registered,
            population_order_axis_hash=self.population_order_axis_hash,
        )
        if (
            not packets
            or len(packets) != _positive_int(self.packet_count, name="packet count")
            or tuple(packet.source_packet_ordinal for packet in packets)
            != tuple(range(len(packets)))
            or max(packet.logical_look_index for packet in packets)
            != _positive_int(self.registered_look_count, name="registered look count")
            or sum(packet.context_count for packet in packets)
            != _positive_int(self.context_count, name="context count")
            or _positive_int(self.population_count, name="population count")
            < self.context_count
            or sum(population) != self.population_count
            or sum(registered) != self.context_count
            or registered != observed_registered
            or self.sampling_measure_hash != expected_measure_hash
            or self.route_axis_hash == self.route_id_axis_hash
            or self.corrected_a_stream_identity_hash
            == self.literal_j_stream_identity_hash
        ):
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.PACKET_DRIFT,
                "row-free v2 ledger packet/look/route axes do not close",
            )
        for packet in packets:
            packet.assert_integrity()
        flags = (
            self.corrected_f_reacquisition_required,
            self.live_device_fd_owner_required,
            self.portable_corrected_moments_present,
            self.portable_fd_snapshot_present,
            self.row_payload_present,
            self.d_stability_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.action_authority,
            self.mask_commit_authority,
            self.runtime_authority,
            self.terminal_authority,
            self.validation_authority,
            self.pareto_authority,
        )
        if (
            flags[:2] != (True, True)
            or any(type(flag) is not bool for flag in flags)
            or any(flags[2:])
            or self.route_scale_d2h_bytes != 0
            or self.raw_row_scratch_bytes != 0
            or self.authority_scope != COMPACT_POST_D_AUTHORITY_SCOPE
            or self.schema != COMPACT_POST_D_SOURCE_LEDGER_SCHEMA_V2
        ):
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.AUTHORITY_VIOLATION,
                "row-free v2 ledger retained arrays, rows, or authority",
            )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        object.__setattr__(self, "route_count", _positive_int(self.route_count, name="route count"))
        object.__setattr__(self, "stratum_names", names)
        object.__setattr__(self, "stratum_weights", weights)
        object.__setattr__(self, "population_stratum_counts", population)
        object.__setattr__(self, "registered_stratum_counts", registered)
        object.__setattr__(self, "packets", packets)
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        value = _payload(self)
        value["packets"] = tuple(packet.packet_hash for packet in self.packets)
        return content_hash(value, prefix=_PREFIX + "row-free-source-ledger-v2-")

    def assert_integrity(self) -> str:
        for packet in self.packets:
            packet.assert_integrity()
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.MUTATED_RECEIPT,
                "row-free v2 ledger changed after issuance",
            )
        return observed

    @property
    def ledger_hash(self) -> str:
        return self.assert_integrity()


def register_v1_sampled_v7_compact_post_d_row_free_source_ledger_v2(
    *,
    device_final_source: (
        V1SampledV7CompactDeviceResidentDRowFreeFinalSourceReceiptV2
    ),
) -> V1SampledV7CompactPostDRowFreeSourceLedgerV2:
    if type(device_final_source) is not (
        V1SampledV7CompactDeviceResidentDRowFreeFinalSourceReceiptV2
    ):
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.SOURCE_NOT_FINAL,
            "row-free v2 ledger requires its exact final source",
        )
    parent = device_final_source
    parent.assert_integrity()
    captures = tuple(value for packet in parent.packets for value in packet.capture_ids)
    source_packets = tuple(
        value for packet in parent.packets for value in packet.source_packet_hashes
    )
    return V1SampledV7CompactPostDRowFreeSourceLedgerV2(
        source_parent_receipt_hash=parent.source_parent_receipt_hash,
        source_parent_integrity_hash=parent.receipt_hash,
        session_design_hash=parent.session_design_hash,
        hbm_forecast_hash=parent.hbm_forecast_hash,
        row_free_finalization_receipt_hash=parent.row_free_finalization_receipt_hash,
        materialization_rank_receipt_hash=parent.materialization_rank_receipt_hash,
        registered_d_transaction_hash=parent.registered_d_transaction_hash,
        registered_d_prefix_binding_hash=parent.registered_d_prefix_binding_hash,
        registered_request_prefix_hash=parent.registered_request_prefix_hash,
        literal_j_source_receipt_hash=parent.literal_j_source_receipt_hash,
        fd_design_hash=parent.fd_design_hash,
        corrected_reacquisition_source_hash=parent.corrected_reacquisition_source_hash,
        live_fd_source_hash=parent.live_fd_source_hash,
        chart_identity_hash=parent.chart_identity_hash,
        current_mask_hash=parent.current_mask_hash,
        proposal_generation=parent.proposal_generation,
        route_axis_hash=parent.route_axis_hash,
        route_id_axis_hash=parent.route_id_axis_hash,
        route_count=parent.route_count,
        corrected_a_stream_identity_hash=parent.corrected_a_stream_identity_hash,
        literal_j_stream_identity_hash=parent.literal_j_stream_identity_hash,
        registered_look_count=parent.registered_look_count,
        packet_count=len(parent.packets),
        context_count=sum(packet.context_count for packet in parent.packets),
        context_axis_hash=parent.context_axis_hash,
        draw_axis_hash=parent.draw_axis_hash,
        stratum_axis_hash=parent.stratum_axis_hash,
        measurement_axis_hash=parent.measurement_axis_hash,
        capture_axis_hash=_axis_hash("capture", captures),
        source_packet_axis_hash=_axis_hash("source-packet", source_packets),
        packet_axis_hash=parent.packet_axis_hash,
        population_count=parent.population_count,
        population_context_axis_hash=parent.population_context_axis_hash,
        population_draw_axis_hash=parent.population_draw_axis_hash,
        population_order_axis_hash=parent.population_order_axis_hash,
        sampling_measure_hash=parent.sampling_measure_hash,
        stratum_names=parent.stratum_names,
        stratum_weights=parent.stratum_weights,
        population_stratum_counts=parent.population_stratum_counts,
        registered_stratum_counts=parent.registered_stratum_counts,
        packets=parent.packets,
        _issuance_token=_SOURCE_V2_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7CompactPostDFrozenAction:
    physical_action_ordinal: int
    physical_action_identity_hash: str
    packed_action_hash: str
    resulting_mask_hash: str
    source_registration_hash: str
    direction_family_hash: str
    shell_indices: tuple[int, ...]
    shell_route_ids: tuple[str, ...]
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str = field(init=False, default="", repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _FROZEN_ACTION_TOKEN:
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.AUTHORITY_VIOLATION,
                "frozen actions are registry-issued",
            )
        object.__setattr__(
            self,
            "physical_action_ordinal",
            _positive_int(self.physical_action_ordinal, name="physical action ordinal"),
        )
        for name in (
            "physical_action_identity_hash",
            "packed_action_hash",
            "resulting_mask_hash",
            "source_registration_hash",
            "direction_family_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        indices = tuple(
            _nonnegative_int(index, name="shell index") for index in self.shell_indices
        )
        route_ids = _identity_axis(
            self.shell_route_ids, name="shell route IDs", unique=True
        )
        if (
            not 1 <= len(indices) <= _MAXIMUM_SHELL
            or len(indices) != len(route_ids)
            or len(indices) != len(set(indices))
        ):
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.SHELL_TOO_LARGE,
                "every frozen variable shell must contain one through eight unique routes",
            )
        object.__setattr__(self, "shell_indices", indices)
        object.__setattr__(self, "shell_route_ids", route_ids)
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return content_hash(_payload(self), prefix=_PREFIX + "frozen-action-")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.MUTATED_RECEIPT,
                "frozen action changed after registry issuance",
            )
        return observed

    @property
    def action_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1SampledV7CompactPostDFrozenActionRegistry:
    source_ledger_hash: str
    source_chart_identity_hash: str
    source_current_mask_hash: str
    source_proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    action_axis_hash: str
    affine_shell_fanout_receipt_hash: str
    actions: tuple[V1SampledV7CompactPostDFrozenAction, ...]
    reference_test_only: bool = False
    finite_action_axis_frozen: bool = True
    variable_shells_source_bound: bool = True
    maximum_action_count: int = _MAXIMUM_ACTIONS
    maximum_shell_size: int = _MAXIMUM_SHELL
    d_stability_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    action_authority: bool = False
    mask_commit_authority: bool = False
    runtime_authority: bool = False
    terminal_authority: bool = False
    validation_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = COMPACT_POST_D_AUTHORITY_SCOPE
    schema: str = COMPACT_POST_D_ACTION_REGISTRY_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str = field(init=False, default="", repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _ACTION_TOKEN:
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.AUTHORITY_VIOLATION,
                "action registries are issuer-only",
            )
        for name in (
            "source_ledger_hash",
            "source_chart_identity_hash",
            "source_current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "action_axis_hash",
            "affine_shell_fanout_receipt_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        generation = _nonnegative_int(
            self.source_proposal_generation, name="source proposal generation"
        )
        route_count = _positive_int(self.route_count, name="route count")
        action_cap = _positive_int(self.maximum_action_count, name="maximum action count")
        shell_cap = _positive_int(self.maximum_shell_size, name="maximum shell size")
        actions = tuple(self.actions)
        if (
            not 1 <= len(actions) <= _MAXIMUM_ACTIONS
            or action_cap != _MAXIMUM_ACTIONS
            or shell_cap != _MAXIMUM_SHELL
            or any(type(action) is not V1SampledV7CompactPostDFrozenAction for action in actions)
            or tuple(action.physical_action_ordinal for action in actions)
            != tuple(range(1, len(actions) + 1))
            or len({action.physical_action_identity_hash for action in actions})
            != len(actions)
            or len({action.packed_action_hash for action in actions}) != len(actions)
            or any(max(action.shell_indices) >= route_count for action in actions)
            or self.action_axis_hash
            != _axis_hash("frozen-action", tuple(action.action_hash for action in actions))
        ):
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.ACTION_REGISTRY_NOT_FROZEN,
                "physical-action axis is empty, oversized, reordered, aliased, or unbound",
            )
        for action in actions:
            action.assert_integrity()
        if (
            self.finite_action_axis_frozen is not True
            or self.variable_shells_source_bound is not True
            or type(self.reference_test_only) is not bool
            or self.authority_scope != COMPACT_POST_D_AUTHORITY_SCOPE
            or self.schema != COMPACT_POST_D_ACTION_REGISTRY_SCHEMA
        ):
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.AUTHORITY_VIOLATION,
                "action registry crossed its frozen identity-only boundary",
            )
        _authority_closed(self)
        object.__setattr__(self, "source_proposal_generation", generation)
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        value = _payload(self)
        value["actions"] = tuple(action.action_hash for action in self.actions)
        return content_hash(value, prefix=_PREFIX + "action-registry-")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for action in self.actions:
            action.assert_integrity()
        observed = self._compute_hash()
        if observed != self._sealed_hash or (
            expected_hash is not None and observed != expected_hash
        ):
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.MUTATED_RECEIPT,
                "frozen action registry changed after issuance",
            )
        return observed

    @property
    def registry_hash(self) -> str:
        return self.assert_integrity()


def register_v1_sampled_v7_compact_post_d_source_ledger(
    *,
    device_final_source: V1SampledV7CompactDeviceResidentDFinalSourceReceipt,
) -> V1SampledV7CompactPostDReplaySourceLedger:
    """Issue the canonical ledger from the exact row-free bridge receipt.

    The only issuer for ``device_final_source`` consumes the exact-owner
    row-free bridge.  Hash-shaped surrogates remain fail-closed.
    """

    if type(device_final_source) is not V1SampledV7CompactDeviceResidentDFinalSourceReceipt:
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.SOURCE_NOT_FINAL,
            "canonical source registration awaits the exact row-free device-final bridge receipt",
        )
    parent = device_final_source
    parent.assert_integrity()
    captures = tuple(value for packet in parent.packets for value in packet.capture_ids)
    source_packets = tuple(
        value for packet in parent.packets for value in packet.source_packet_hashes
    )
    return V1SampledV7CompactPostDReplaySourceLedger(
        source_parent_receipt_hash=parent.source_parent_receipt_hash,
        source_parent_integrity_hash=parent.receipt_hash,
        session_design_hash=parent.session_design_hash,
        hbm_forecast_hash=parent.hbm_forecast_hash,
        session_materialization_hash=parent.session_materialization_hash,
        materialization_rank_receipt_hash=(
            parent.materialization_rank_receipt_hash
        ),
        literal_j_compact_source_hash=parent.literal_j_source_receipt_hash,
        registered_d_transaction_hash=parent.registered_d_transaction_hash,
        registered_d_prefix_binding_hash=parent.registered_d_prefix_binding_hash,
        registered_request_prefix_hash=parent.registered_request_prefix_hash,
        literal_j_source_receipt_hash=parent.literal_j_source_receipt_hash,
        fd_design_hash=parent.fd_design_hash,
        fd_snapshot_hash=parent.fd_snapshot_hash,
        corrected_moments_state_hash=parent.corrected_moments_state_hash,
        chart_identity_hash=parent.chart_identity_hash,
        current_mask_hash=parent.current_mask_hash,
        proposal_generation=parent.proposal_generation,
        route_axis_hash=parent.route_axis_hash,
        route_id_axis_hash=parent.route_id_axis_hash,
        route_count=parent.route_count,
        corrected_a_stream_identity_hash=parent.corrected_a_stream_identity_hash,
        literal_j_stream_identity_hash=parent.literal_j_stream_identity_hash,
        registered_look_count=parent.registered_look_count,
        packet_count=len(parent.packets),
        context_count=sum(packet.context_count for packet in parent.packets),
        context_axis_hash=parent.context_axis_hash,
        draw_axis_hash=parent.draw_axis_hash,
        stratum_axis_hash=parent.stratum_axis_hash,
        measurement_axis_hash=parent.measurement_axis_hash,
        capture_axis_hash=_axis_hash("capture", captures),
        source_packet_axis_hash=_axis_hash("source-packet", source_packets),
        packet_axis_hash=parent.packet_axis_hash,
        population_count=parent.population_count,
        population_context_axis_hash=parent.population_context_axis_hash,
        population_draw_axis_hash=parent.population_draw_axis_hash,
        population_order_axis_hash=parent.population_order_axis_hash,
        packets=parent.packets,
        source_provenance=(
            V1SampledV7CompactPostDSourceProvenance.SOURCE_NATIVE_COMPACT
        ),
        canonical_zero_row_ordinary_path=True,
        _issuance_token=_SOURCE_TOKEN,
    )


def register_v1_sampled_v7_compact_post_d_legacy_row_backed_source_ledger(
    *,
    final_rank_source: V1SampledDeviceFinalRankSourceReceipt,
    original_capture_binding: V1SampledLiteralJOriginalCaptureBinding,
    look_receipts: tuple[V1SampledDeviceResidentDLookReceipt, ...],
    literal_j_registration: V1LiteralJTwoPassDFixedProjectionRegistration,
    population_count: int | None = None,
    population_context_axis_hash: str | None = None,
    population_draw_axis_hash: str | None = None,
    population_order_axis_hash: str | None = None,
) -> V1SampledV7CompactPostDReplaySourceLedger:
    """Provenance-only adapter for historical row-manifest source objects.

    This tagged result is readable and replayable in compatibility tooling,
    but it can never claim the canonical zero-row ordinary production path.
    """

    exact = (
        (final_rank_source, V1SampledDeviceFinalRankSourceReceipt),
        (original_capture_binding, V1SampledLiteralJOriginalCaptureBinding),
        (literal_j_registration, V1LiteralJTwoPassDFixedProjectionRegistration),
    )
    if any(type(value) is not expected for value, expected in exact) or type(
        look_receipts
    ) is not tuple or any(
        type(value) is not V1SampledDeviceResidentDLookReceipt
        for value in look_receipts
    ):
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.INVALID_TYPE,
            "production source registration requires exact device-D, capture, look, and literal-J owners",
        )
    final_rank_source.assert_integrity()
    original_capture_binding.assert_integrity()
    literal_j_registration.assert_integrity()
    for receipt in look_receipts:
        receipt.assert_integrity()
    closure = (
        final_rank_source.original_capture_binding_hash
        == original_capture_binding.binding_hash,
        final_rank_source.packet_count == len(look_receipts),
        final_rank_source.packet_count == len(literal_j_registration.packets),
        final_rank_source.registered_d_prefix_hash
        == original_capture_binding.registered_d_prefix_hash
        == literal_j_registration.registered_d_prefix_hash,
        final_rank_source.chart_identity_hash
        == original_capture_binding.chart_identity_hash
        == literal_j_registration.chart_identity_hash,
        final_rank_source.current_mask_hash
        == original_capture_binding.current_mask_hash
        == literal_j_registration.current_mask_hash,
        final_rank_source.proposal_generation
        == original_capture_binding.proposal_generation
        == literal_j_registration.proposal_generation,
        final_rank_source.route_axis_hash
        == original_capture_binding.route_axis_hash
        == literal_j_registration.route_axis_hash,
        final_rank_source.route_id_axis_hash
        == original_capture_binding.route_id_axis_hash
        == literal_j_registration.route_id_axis_hash,
        final_rank_source.route_count
        == original_capture_binding.route_count
        == literal_j_registration.route_count,
        final_rank_source.corrected_a_stream_identity_hash
        == original_capture_binding.corrected_a_stream_identity_hash,
        final_rank_source.literal_j_stream_identity_hash
        == original_capture_binding.literal_j_stream_identity_hash
        == literal_j_registration.composite_literal_j_stream_source_hash,
        all(
            tuple(
                receipt.packet_ordinal
                for receipt in look_receipts
                if receipt.registered_look_index == look_index
            )
            == tuple(
                range(
                    sum(
                        receipt.registered_look_index == look_index
                        for receipt in look_receipts
                    )
                )
            )
            for look_index in range(
                1,
                final_rank_source.registered_look_count + 1,
            )
        ),
        max(receipt.registered_look_index for receipt in look_receipts)
        == final_rank_source.registered_look_count,
    )
    if not all(closure):
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.SOURCE_DRIFT,
            "device-D, original capture, logical looks, and literal-J registration do not share one exact source ledger",
        )
    packets = []
    for ordinal, (capture_packet, projection_packet, look) in enumerate(
        zip(
            range(final_rank_source.packet_count),
            literal_j_registration.packets,
            look_receipts,
            strict=True,
        )
    ):
        source_packet_hashes = projection_packet.source_packet_hashes
        packets.append(
            V1SampledV7CompactPostDReplayPacket(
                source_packet_ordinal=ordinal,
                logical_look_index=look.registered_look_index,
                packet_ordinal_within_logical_look=look.packet_ordinal,
                packet_plan_hash=original_capture_binding.packet_plan_hashes[capture_packet],
                source_capture_identity_hash=(
                    original_capture_binding.packet_source_capture_identity_hashes[
                        capture_packet
                    ]
                ),
                provider_proof_hash=(
                    original_capture_binding.packet_provider_proof_hashes[capture_packet]
                ),
                raw_dual_proof_hash=(
                    original_capture_binding.packet_raw_dual_proof_hashes[capture_packet]
                ),
                same_capture_vjp_source_hash=(
                    projection_packet.same_capture_vjp_source_hash
                ),
                context_ids=projection_packet.context_ids,
                draw_ids=projection_packet.draw_ids,
                measurement_ids=projection_packet.measurement_ids,
                capture_ids=projection_packet.capture_ids,
                source_packet_hashes=source_packet_hashes,
                stratum_labels=original_capture_binding.packet_stratum_labels[
                    capture_packet
                ],
                source_native_compact=False,
                reference_test_only=False,
                _issuance_token=_PACKET_TOKEN,
            )
        )
    context_ids = tuple(value for packet in packets for value in packet.context_ids)
    draw_ids = tuple(value for packet in packets for value in packet.draw_ids)
    captures = tuple(value for packet in packets for value in packet.capture_ids)
    source_packets = tuple(
        value for packet in packets for value in packet.source_packet_hashes
    )
    if (
        _axis_hash("registered-context", context_ids)
        == _axis_hash("registered-draw", draw_ids)
    ):
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.SOURCE_DRIFT,
            "context and draw axes collapsed unexpectedly",
        )
    population = final_rank_source.context_count if population_count is None else _positive_int(
        population_count, name="population count"
    )
    population_context = (
        final_rank_source.context_axis_hash
        if population_context_axis_hash is None
        else _identity(population_context_axis_hash, name="population context axis hash")
    )
    population_draw = (
        final_rank_source.draw_axis_hash
        if population_draw_axis_hash is None
        else _identity(population_draw_axis_hash, name="population draw axis hash")
    )
    population_order = (
        _axis_hash("registered-prefix-order", tuple(packet.packet_hash for packet in packets))
        if population_order_axis_hash is None
        else _identity(population_order_axis_hash, name="population order axis hash")
    )
    return V1SampledV7CompactPostDReplaySourceLedger(
        source_parent_receipt_hash=final_rank_source.receipt_hash,
        source_parent_integrity_hash=original_capture_binding.binding_hash,
        session_design_hash=final_rank_source.session_design_hash,
        hbm_forecast_hash=final_rank_source.hbm_forecast_hash,
        session_materialization_hash=final_rank_source.device_materialization_hash,
        materialization_rank_receipt_hash=(
            final_rank_source.materialization_rank_receipt_hash
        ),
        literal_j_compact_source_hash=literal_j_registration.registration_hash,
        registered_d_transaction_hash=final_rank_source.registered_d_transaction_hash,
        registered_d_prefix_binding_hash=final_rank_source.registered_d_prefix_binding_hash,
        registered_request_prefix_hash=final_rank_source.rank_invariant_raw_prefix_hash,
        literal_j_source_receipt_hash=final_rank_source.literal_j_source_receipt_hash,
        fd_design_hash=final_rank_source.fd_design_hash,
        fd_snapshot_hash=final_rank_source.fd_snapshot_hash,
        corrected_moments_state_hash=final_rank_source.corrected_moments_state_hash,
        chart_identity_hash=final_rank_source.chart_identity_hash,
        current_mask_hash=final_rank_source.current_mask_hash,
        proposal_generation=final_rank_source.proposal_generation,
        route_axis_hash=final_rank_source.route_axis_hash,
        route_id_axis_hash=final_rank_source.route_id_axis_hash,
        route_count=final_rank_source.route_count,
        corrected_a_stream_identity_hash=final_rank_source.corrected_a_stream_identity_hash,
        literal_j_stream_identity_hash=final_rank_source.literal_j_stream_identity_hash,
        registered_look_count=final_rank_source.registered_look_count,
        packet_count=final_rank_source.packet_count,
        context_count=final_rank_source.context_count,
        context_axis_hash=final_rank_source.context_axis_hash,
        draw_axis_hash=final_rank_source.draw_axis_hash,
        stratum_axis_hash=final_rank_source.stratum_axis_hash,
        measurement_axis_hash=final_rank_source.source_measurement_axis_hash,
        capture_axis_hash=_axis_hash("capture", captures),
        source_packet_axis_hash=_axis_hash("source-packet", source_packets),
        packet_axis_hash=final_rank_source.packet_axis_hash,
        population_count=population,
        population_context_axis_hash=population_context,
        population_draw_axis_hash=population_draw,
        population_order_axis_hash=population_order,
        packets=tuple(packets),
        source_provenance=(
            V1SampledV7CompactPostDSourceProvenance.LEGACY_ROW_BACKED_COMPATIBILITY
        ),
        canonical_zero_row_ordinary_path=False,
        _issuance_token=_SOURCE_TOKEN,
    )


def register_v1_sampled_v7_compact_post_d_frozen_actions(
    *,
    source_ledger: (
        V1SampledV7CompactPostDReplaySourceLedger
        | V1SampledV7CompactPostDRowFreeSourceLedgerV2
    ),
    physical_actions: tuple[V1SampledV7PhysicalActionCustody, ...],
    affine_shell_fanout: V1SampledV7AffineShellPreoutcomeFanoutReceipt,
) -> V1SampledV7CompactPostDFrozenActionRegistry:
    """Bind the exact finite hard-action axis to its frozen shell directions."""

    source_is_v1 = type(source_ledger) is V1SampledV7CompactPostDReplaySourceLedger
    source_is_v2 = type(source_ledger) is V1SampledV7CompactPostDRowFreeSourceLedgerV2
    source_is_canonical = source_is_v2 or (
        source_is_v1
        and source_ledger.reference_test_only is False
        and source_ledger.source_provenance
        is V1SampledV7CompactPostDSourceProvenance.SOURCE_NATIVE_COMPACT
        and source_ledger.canonical_zero_row_ordinary_path is True
    )
    if (
        not source_is_canonical
        or type(physical_actions) is not tuple
        or not physical_actions
        or any(type(value) is not V1SampledV7PhysicalActionCustody for value in physical_actions)
        or type(affine_shell_fanout)
        is not V1SampledV7AffineShellPreoutcomeFanoutReceipt
    ):
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.INVALID_TYPE,
            "production registry requires exact non-reference source, physical actions, and affine-shell receipt",
        )
    source_ledger.assert_integrity()
    affine_shell_fanout.assert_integrity()
    if len(physical_actions) > _MAXIMUM_ACTIONS:
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.ACTION_REGISTRY_NOT_FROZEN,
            "the robust physical-action family exceeds B <= 8",
        )
    registrations = affine_shell_fanout.preoutcome_registrations
    if len(registrations) != len(physical_actions):
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.ACTION_DRIFT,
            "affine-shell and physical-action axes have different lengths",
        )
    actions = []
    for ordinal, (custody, registration) in enumerate(
        zip(physical_actions, registrations, strict=True), start=1
    ):
        custody.assert_integrity()
        registration.assert_integrity()
        if (
            custody.physical_action_ordinal != ordinal
            or registration.physical_action_identity_hash
            != custody.physical_action_identity_hash
            or registration.packed_action_hash != custody.packed_action_hash
            or registration.resulting_mask_hash != custody.resulting_mask_hash
            or registration.chart_identity_hash != source_ledger.chart_identity_hash
            or registration.current_mask_hash != source_ledger.current_mask_hash
            or registration.proposal_generation != source_ledger.proposal_generation
            or registration.route_axis_hash != source_ledger.route_axis_hash
            or registration.route_id_axis_hash != source_ledger.route_id_axis_hash
            or registration.route_count != source_ledger.route_count
            or registration.registered_d_prefix_hash
            != source_ledger.registered_request_prefix_hash
            or registration.corrected_evidence_stream_source_hash
            != source_ledger.corrected_a_stream_identity_hash
            or registration.literal_j_stream_source_hash
            != source_ledger.literal_j_stream_identity_hash
        ):
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.ACTION_DRIFT,
                "one physical action or shell left the registered D chart",
            )
        actions.append(
            V1SampledV7CompactPostDFrozenAction(
                physical_action_ordinal=ordinal,
                physical_action_identity_hash=custody.physical_action_identity_hash,
                packed_action_hash=custody.packed_action_hash,
                resulting_mask_hash=custody.resulting_mask_hash,
                source_registration_hash=registration.source_registration_hash,
                direction_family_hash=registration.direction_family_hash,
                shell_indices=registration.shell_indices,
                shell_route_ids=registration.shell_route_ids,
                _issuance_token=_FROZEN_ACTION_TOKEN,
            )
        )
    axis_hash = _axis_hash("frozen-action", tuple(action.action_hash for action in actions))
    return V1SampledV7CompactPostDFrozenActionRegistry(
        source_ledger_hash=source_ledger.ledger_hash,
        source_chart_identity_hash=source_ledger.chart_identity_hash,
        source_current_mask_hash=source_ledger.current_mask_hash,
        source_proposal_generation=source_ledger.proposal_generation,
        route_axis_hash=source_ledger.route_axis_hash,
        route_id_axis_hash=source_ledger.route_id_axis_hash,
        route_count=source_ledger.route_count,
        action_axis_hash=axis_hash,
        affine_shell_fanout_receipt_hash=affine_shell_fanout.receipt_hash,
        actions=tuple(actions),
        _issuance_token=_ACTION_TOKEN,
    )


def make_v1_sampled_v7_compact_post_d_reference_source_for_tests(
    *,
    route_count: int,
    packet_context_counts: Sequence[int],
    logical_look_indices: Sequence[int] | None = None,
    identity_seed: str = "tiny-reference",
) -> V1SampledV7CompactPostDReplaySourceLedger:
    """Issue one explicitly nonproduction tiny source for protocol tests."""

    routes = _positive_int(route_count, name="route count")
    counts = tuple(
        _positive_int(value, name="packet context count")
        for value in packet_context_counts
    )
    if not counts or sum(counts) > 4096 or routes > 4096:
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.CPU_REFERENCE_SCALE_EXCEEDED,
            "reference source exceeds the bounded test domain",
        )
    looks = (
        tuple(range(1, len(counts) + 1))
        if logical_look_indices is None
        else tuple(_positive_int(value, name="logical look") for value in logical_look_indices)
    )
    if len(looks) != len(counts) or looks != tuple(sorted(looks)):
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.PACKET_DRIFT,
            "reference logical-look axis must align and be nondecreasing",
        )
    seed = _identity(identity_seed, name="identity seed")

    def identity(role: str) -> str:
        return content_hash((seed, role), prefix=_PREFIX + "reference-")

    packets = []
    context_cursor = 0
    for ordinal, (count, look) in enumerate(zip(counts, looks, strict=True)):
        contexts = tuple(f"context-{context_cursor + index}" for index in range(count))
        draws = tuple(f"draw-{context_cursor + index}" for index in range(count))
        measurements = tuple(
            f"measurement-{context_cursor + index}" for index in range(count)
        )
        captures = tuple(f"capture-{context_cursor + index}" for index in range(count))
        source_packets = (f"source-packet-{ordinal}",) * count
        strata = tuple(
            "target" if (context_cursor + index) % 2 == 0 else "complement"
            for index in range(count)
        )
        packets.append(
            V1SampledV7CompactPostDReplayPacket(
                source_packet_ordinal=ordinal,
                logical_look_index=look,
                packet_ordinal_within_logical_look=sum(
                    prior_look == look for prior_look in looks[:ordinal]
                ),
                packet_plan_hash=identity(f"packet-plan-{ordinal}"),
                source_capture_identity_hash=identity(f"capture-source-{ordinal}"),
                provider_proof_hash=identity(f"provider-proof-{ordinal}"),
                raw_dual_proof_hash=identity(f"raw-dual-{ordinal}"),
                same_capture_vjp_source_hash=identity(f"same-vjp-{ordinal}"),
                context_ids=contexts,
                draw_ids=draws,
                measurement_ids=measurements,
                capture_ids=captures,
                source_packet_hashes=source_packets,
                stratum_labels=strata,
                source_native_compact=False,
                reference_test_only=True,
                _issuance_token=_PACKET_TOKEN,
            )
        )
        context_cursor += count
    flat = lambda name: tuple(  # noqa: E731 - compact local axis helper.
        value for packet in packets for value in getattr(packet, name)
    )
    return V1SampledV7CompactPostDReplaySourceLedger(
        source_parent_receipt_hash=identity("reference-source-parent"),
        source_parent_integrity_hash=identity("reference-source-parent-integrity"),
        session_design_hash=identity("session-design"),
        hbm_forecast_hash=identity("hbm-forecast"),
        session_materialization_hash=identity("session-materialization"),
        materialization_rank_receipt_hash=identity("materialization-rank"),
        literal_j_compact_source_hash=identity("literal-j-compact-source"),
        registered_d_transaction_hash=identity("d-transaction"),
        registered_d_prefix_binding_hash=identity("d-prefix-binding"),
        registered_request_prefix_hash=identity("d-prefix"),
        literal_j_source_receipt_hash=identity("literal-j-source"),
        fd_design_hash=identity("fd-design"),
        fd_snapshot_hash=identity("fd-snapshot"),
        corrected_moments_state_hash=identity("corrected-moments"),
        chart_identity_hash=identity("chart"),
        current_mask_hash=identity("mask"),
        proposal_generation=0,
        route_axis_hash=identity("route-axis"),
        route_id_axis_hash=identity("route-id-axis"),
        route_count=routes,
        corrected_a_stream_identity_hash=identity("corrected-a"),
        literal_j_stream_identity_hash=identity("literal-j"),
        registered_look_count=max(looks),
        packet_count=len(packets),
        context_count=sum(counts),
        context_axis_hash=_axis_hash("context", flat("context_ids")),
        draw_axis_hash=_axis_hash("draw", flat("draw_ids")),
        stratum_axis_hash=_axis_hash("stratum", flat("stratum_labels")),
        measurement_axis_hash=_axis_hash("measurement", flat("measurement_ids")),
        capture_axis_hash=_axis_hash("capture", flat("capture_ids")),
        source_packet_axis_hash=_axis_hash("source-packet", flat("source_packet_hashes")),
        packet_axis_hash=_axis_hash("packet", tuple(packet.packet_hash for packet in packets)),
        population_count=sum(counts),
        population_context_axis_hash=_axis_hash("population-context", flat("context_ids")),
        population_draw_axis_hash=_axis_hash("population-draw", flat("draw_ids")),
        population_order_axis_hash=_axis_hash("population-order", flat("draw_ids")),
        packets=tuple(packets),
        source_provenance=(
            V1SampledV7CompactPostDSourceProvenance.REFERENCE_TEST_ONLY
        ),
        canonical_zero_row_ordinary_path=False,
        reference_test_only=True,
        _issuance_token=_SOURCE_TOKEN,
    )


def make_v1_sampled_v7_compact_post_d_reference_actions_for_tests(
    *,
    source_ledger: V1SampledV7CompactPostDReplaySourceLedger,
    shells: Sequence[Sequence[int]],
) -> V1SampledV7CompactPostDFrozenActionRegistry:
    if (
        type(source_ledger) is not V1SampledV7CompactPostDReplaySourceLedger
        or not source_ledger.reference_test_only
    ):
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.INVALID_TYPE,
            "reference action issuance requires the explicit tiny reference source",
        )
    action_shells = tuple(tuple(value) for value in shells)
    if not action_shells or len(action_shells) > _MAXIMUM_ACTIONS:
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.ACTION_REGISTRY_NOT_FROZEN,
            "reference action family must contain one through eight actions",
        )
    actions = []
    for ordinal, shell in enumerate(action_shells, start=1):
        indices = tuple(_nonnegative_int(value, name="shell index") for value in shell)
        action_identity = content_hash(
            (source_ledger.ledger_hash, ordinal), prefix=_PREFIX + "reference-action-id-"
        )
        actions.append(
            V1SampledV7CompactPostDFrozenAction(
                physical_action_ordinal=ordinal,
                physical_action_identity_hash=action_identity,
                packed_action_hash=content_hash(indices, prefix=_PREFIX + "reference-packed-"),
                resulting_mask_hash=content_hash(indices, prefix=_PREFIX + "reference-mask-"),
                source_registration_hash=content_hash(
                    (action_identity, "source"), prefix=_PREFIX + "reference-source-registration-"
                ),
                direction_family_hash=content_hash(
                    indices, prefix=_PREFIX + "reference-direction-family-"
                ),
                shell_indices=indices,
                shell_route_ids=tuple(f"route-{index}" for index in indices),
                _issuance_token=_FROZEN_ACTION_TOKEN,
            )
        )
    action_axis = _axis_hash("frozen-action", tuple(action.action_hash for action in actions))
    return V1SampledV7CompactPostDFrozenActionRegistry(
        source_ledger_hash=source_ledger.ledger_hash,
        source_chart_identity_hash=source_ledger.chart_identity_hash,
        source_current_mask_hash=source_ledger.current_mask_hash,
        source_proposal_generation=source_ledger.proposal_generation,
        route_axis_hash=source_ledger.route_axis_hash,
        route_id_axis_hash=source_ledger.route_id_axis_hash,
        route_count=source_ledger.route_count,
        action_axis_hash=action_axis,
        affine_shell_fanout_receipt_hash=content_hash(
            action_axis, prefix=_PREFIX + "reference-affine-shell-"
        ),
        actions=tuple(actions),
        reference_test_only=True,
        _issuance_token=_ACTION_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7CompactPostDReplayDesign:
    execution_mode: V1SampledV7CompactPostDReplayExecutionMode
    literal_j_replay_mode: V1SampledV7CompactPostDLiteralJReplayMode
    route_tile_size: int
    cpu_reference_route_cap: int = 4096
    cpu_reference_context_cap: int = 4096
    raw_row_d2h_budget_bytes: int = 0
    raw_row_scratch_budget_bytes: int = 0
    schema: str = COMPACT_POST_D_DESIGN_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str = field(init=False, default="", repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _DESIGN_TOKEN:
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.AUTHORITY_VIOLATION,
                "compact replay designs are issuer-only",
            )
        if (
            type(self.execution_mode) is not V1SampledV7CompactPostDReplayExecutionMode
            or type(self.literal_j_replay_mode)
            is not V1SampledV7CompactPostDLiteralJReplayMode
        ):
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.INVALID_TYPE,
                "execution and literal-J replay modes must be exact",
            )
        for name in (
            "route_tile_size",
            "cpu_reference_route_cap",
            "cpu_reference_context_cap",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name=name))
        if (
            self.raw_row_d2h_budget_bytes != 0
            or self.raw_row_scratch_budget_bytes != 0
            or self.schema != COMPACT_POST_D_DESIGN_SCHEMA
        ):
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.ROW_TRANSFER_FORBIDDEN,
                "compact post-D replay has zero raw-row D2H and scratch budget",
            )
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return content_hash(_payload(self), prefix=_PREFIX + "design-")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.MUTATED_RECEIPT,
                "compact replay design changed after issuance",
            )
        return observed

    @property
    def design_hash(self) -> str:
        return self.assert_integrity()


def make_v1_sampled_v7_compact_post_d_replay_design(
    *,
    execution_mode: V1SampledV7CompactPostDReplayExecutionMode,
    literal_j_replay_mode: V1SampledV7CompactPostDLiteralJReplayMode = (
        V1SampledV7CompactPostDLiteralJReplayMode.GLOBAL_DIGEST_IDENTICAL_TWO_PASS
    ),
    route_tile_size: int = 1 << 20,
    cpu_reference_route_cap: int = 4096,
    cpu_reference_context_cap: int = 4096,
) -> V1SampledV7CompactPostDReplayDesign:
    return V1SampledV7CompactPostDReplayDesign(
        execution_mode=execution_mode,
        literal_j_replay_mode=literal_j_replay_mode,
        route_tile_size=route_tile_size,
        cpu_reference_route_cap=cpu_reference_route_cap,
        cpu_reference_context_cap=cpu_reference_context_cap,
        _issuance_token=_DESIGN_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7CompactPostDReplayPlan:
    source_ledger_hash: str
    action_registry_hash: str
    design_hash: str
    route_count: int
    packet_count: int
    action_count: int
    route_windows: tuple[tuple[int, int], ...]
    literal_j_replay_mode: V1SampledV7CompactPostDLiteralJReplayMode
    reference_test_only: bool
    action_axis_frozen_before_replay: bool = True
    shell_axis_frozen_before_replay: bool = True
    raw_rows_persisted: bool = False
    raw_row_d2h_bytes: int = 0
    raw_row_scratch_bytes: int = 0
    d_stability_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    action_authority: bool = False
    mask_commit_authority: bool = False
    runtime_authority: bool = False
    terminal_authority: bool = False
    validation_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = COMPACT_POST_D_AUTHORITY_SCOPE
    schema: str = COMPACT_POST_D_PLAN_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str = field(init=False, default="", repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _PLAN_TOKEN:
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.AUTHORITY_VIOLATION,
                "compact replay plans are issuer-only",
            )
        for name in ("source_ledger_hash", "action_registry_hash", "design_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        routes = _positive_int(self.route_count, name="route count")
        packets = _positive_int(self.packet_count, name="packet count")
        actions = _positive_int(self.action_count, name="action count")
        windows = []
        cursor = 0
        for raw_start, raw_stop in self.route_windows:
            start = _nonnegative_int(raw_start, name="route-window start")
            stop = _positive_int(raw_stop, name="route-window stop")
            if start != cursor or not start < stop <= routes:
                _refuse(
                    V1SampledV7CompactPostDReplayRefusalCode.TILE_DRIFT,
                    "route windows must cover the route axis once in order",
                )
            windows.append((start, stop))
            cursor = stop
        if cursor != routes or actions > _MAXIMUM_ACTIONS:
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.TILE_DRIFT,
                "route cover is incomplete or action count exceeds B <= 8",
            )
        if (
            type(self.literal_j_replay_mode)
            is not V1SampledV7CompactPostDLiteralJReplayMode
            or type(self.reference_test_only) is not bool
            or self.action_axis_frozen_before_replay is not True
            or self.shell_axis_frozen_before_replay is not True
            or self.raw_rows_persisted is not False
            or self.raw_row_d2h_bytes != 0
            or self.raw_row_scratch_bytes != 0
            or self.authority_scope != COMPACT_POST_D_AUTHORITY_SCOPE
            or self.schema != COMPACT_POST_D_PLAN_SCHEMA
        ):
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.AUTHORITY_VIOLATION,
                "replay plan retained rows or crossed its authority boundary",
            )
        _authority_closed(self)
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "packet_count", packets)
        object.__setattr__(self, "action_count", actions)
        object.__setattr__(self, "route_windows", tuple(windows))
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return content_hash(_payload(self), prefix=_PREFIX + "plan-")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.MUTATED_RECEIPT,
                "compact replay plan changed after issuance",
            )
        return observed

    @property
    def plan_hash(self) -> str:
        return self.assert_integrity()


def open_v1_sampled_v7_compact_post_d_replay(
    *,
    source_ledger: (
        V1SampledV7CompactPostDReplaySourceLedger
        | V1SampledV7CompactPostDRowFreeSourceLedgerV2
    ),
    action_registry: V1SampledV7CompactPostDFrozenActionRegistry,
    design: V1SampledV7CompactPostDReplayDesign,
    source_native_route_windows: tuple[tuple[int, int], ...] | None = None,
) -> V1SampledV7CompactPostDReplayPlan:
    """Open one replay from frozen source/action/execution coordinates.

    ``source_native_route_windows`` is the production path: it binds the
    producer's already-frozen block-local CUDA emission geometry without
    rechunking it.  Omitting it retains the historical uniform reference
    tiling.  The window axis affects execution and restart identity only;
    neither plan can authorize a scientific decision.
    """

    source_is_v1 = type(source_ledger) is V1SampledV7CompactPostDReplaySourceLedger
    source_is_v2 = type(source_ledger) is V1SampledV7CompactPostDRowFreeSourceLedgerV2
    if (
        not (source_is_v1 or source_is_v2)
        or type(action_registry) is not V1SampledV7CompactPostDFrozenActionRegistry
        or type(design) is not V1SampledV7CompactPostDReplayDesign
    ):
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.INVALID_TYPE,
            "opening compact replay requires exact source, action, and design types",
        )
    source_ledger.assert_integrity()
    action_registry.assert_integrity()
    design.assert_integrity()
    if (
        action_registry.source_ledger_hash != source_ledger.ledger_hash
        or action_registry.source_chart_identity_hash != source_ledger.chart_identity_hash
        or action_registry.source_current_mask_hash != source_ledger.current_mask_hash
        or action_registry.source_proposal_generation != source_ledger.proposal_generation
        or action_registry.route_axis_hash != source_ledger.route_axis_hash
        or action_registry.route_id_axis_hash != source_ledger.route_id_axis_hash
        or action_registry.route_count != source_ledger.route_count
    ):
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.ACTION_DRIFT,
            "frozen action registry belongs to another D source",
        )
    source_reference = source_is_v1 and source_ledger.reference_test_only
    reference = source_reference or action_registry.reference_test_only
    if reference is not (source_reference and action_registry.reference_test_only):
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.SOURCE_DRIFT,
            "reference and production source/action provenance cannot mix",
        )
    if design.execution_mode is V1SampledV7CompactPostDReplayExecutionMode.CUDA_PRODUCTION:
        if (
            reference
            or (
                source_is_v1
                and (
                    source_ledger.source_provenance
                    is not V1SampledV7CompactPostDSourceProvenance.SOURCE_NATIVE_COMPACT
                    or source_ledger.canonical_zero_row_ordinary_path is not True
                )
            )
        ):
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.CUDA_REQUIRED,
                "only the canonical row-free source-native ledger can enter production CUDA replay",
            )
        if torch is None or not torch.cuda.is_available():
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.CUDA_REQUIRED,
                "production compact replay requires live CUDA",
            )
    elif (
        not reference
        or source_ledger.route_count > design.cpu_reference_route_cap
        or source_ledger.context_count > design.cpu_reference_context_cap
    ):
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.CPU_REFERENCE_SCALE_EXCEEDED,
            "CPU mode accepts only the bounded explicit reference source",
        )
    if source_native_route_windows is None:
        windows = tuple(
            (start, min(source_ledger.route_count, start + design.route_tile_size))
            for start in range(0, source_ledger.route_count, design.route_tile_size)
        )
    else:
        if type(source_native_route_windows) is not tuple or not source_native_route_windows:
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.TILE_DRIFT,
                "source-native route windows must be one nonempty frozen tuple",
            )
        windows = source_native_route_windows
    return V1SampledV7CompactPostDReplayPlan(
        source_ledger_hash=source_ledger.ledger_hash,
        action_registry_hash=action_registry.registry_hash,
        design_hash=design.design_hash,
        route_count=source_ledger.route_count,
        packet_count=source_ledger.packet_count,
        action_count=len(action_registry.actions),
        route_windows=windows,
        literal_j_replay_mode=design.literal_j_replay_mode,
        reference_test_only=reference,
        _issuance_token=_PLAN_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7CompactPostDReplayTile:
    source_ledger_hash: str
    packet_hash: str
    pass_index: int
    route_start: int
    route_stop: int
    corrected_a_rows: Any = field(repr=False, compare=False)
    literal_j_rows: Any = field(repr=False, compare=False)
    corrected_a_source_digest: str | None = None
    literal_j_source_digest: str | None = None

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live compact replay tiles are nonserializable")


@dataclass(frozen=True, slots=True)
class V1SampledV7CompactPostDReplayFailurePoint:
    boundary: str
    pass_index: int
    source_packet_ordinal: int | None = None
    tile_ordinal: int | None = None

    def __post_init__(self) -> None:
        if self.boundary not in {"pass", "packet", "tile"}:
            raise ValueError("failure boundary must be pass, packet, or tile")
        if self.pass_index not in {1, 2}:
            raise ValueError("failure pass must be one or two")
        if (
            self.boundary in {"packet", "tile"}
            and self.source_packet_ordinal is None
        ):
            raise ValueError("packet and tile faults require a packet ordinal")
        if self.boundary == "tile" and self.tile_ordinal is None:
            raise ValueError("tile faults require a tile ordinal")


@dataclass(frozen=True, slots=True)
class V1SampledV7CompactPostDReplayCheckpoint:
    plan_hash: str
    source_ledger_hash: str
    action_registry_hash: str
    design_hash: str
    completed_tile_event_count: int
    completed_event_chain_hash: str
    next_pass_index: int
    next_source_packet_ordinal: int
    next_tile_ordinal: int
    pass1_literal_j_digest_axis_hash: str
    raw_row_d2h_bytes: int = 0
    raw_row_scratch_bytes: int = 0
    authority_scope: str = COMPACT_POST_D_AUTHORITY_SCOPE
    schema: str = COMPACT_POST_D_CHECKPOINT_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str = field(init=False, default="", repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _CHECKPOINT_TOKEN:
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.AUTHORITY_VIOLATION,
                "compact replay checkpoints are issuer-only",
            )
        for name in (
            "plan_hash",
            "source_ledger_hash",
            "action_registry_hash",
            "design_hash",
            "completed_event_chain_hash",
            "pass1_literal_j_digest_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "completed_tile_event_count",
            "next_source_packet_ordinal",
            "next_tile_ordinal",
        ):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name=name))
        if (
            self.next_pass_index not in {1, 2, 3}
            or self.raw_row_d2h_bytes != 0
            or self.raw_row_scratch_bytes != 0
            or self.authority_scope != COMPACT_POST_D_AUTHORITY_SCOPE
            or self.schema != COMPACT_POST_D_CHECKPOINT_SCHEMA
        ):
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.AUTHORITY_VIOLATION,
                "checkpoint retained rows or crossed the replay boundary",
            )
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return content_hash(_payload(self), prefix=_PREFIX + "checkpoint-")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.MUTATED_RECEIPT,
                "checkpoint changed after issuance",
            )
        return observed

    @property
    def checkpoint_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1SampledV7CompactPostDReplayExecutionReceipt:
    plan_hash: str
    source_ledger_hash: str
    action_registry_hash: str
    design_hash: str
    corrected_f_consumer_receipt_hash: str
    variable_shell_consumer_receipt_hash: str
    literal_j_consumer_receipt_hash: str
    packet_count: int
    route_window_count: int
    pass1_tile_count: int
    pass2_tile_count: int
    pass1_literal_j_digest_axis_hash: str
    pass2_literal_j_digest_axis_hash: str
    completed_event_chain_hash: str
    resumed_from_checkpoint_hash: str | None
    maximum_live_row_tile_pair_count: int = 1
    raw_rows_retained: bool = False
    raw_row_d2h_bytes: int = 0
    raw_row_scratch_bytes: int = 0
    literal_j_two_pass_exact: bool = True
    literal_j_pass_digest_identity_verified: bool = True
    d_stability_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    action_authority: bool = False
    mask_commit_authority: bool = False
    runtime_authority: bool = False
    terminal_authority: bool = False
    validation_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = COMPACT_POST_D_AUTHORITY_SCOPE
    literal_j_exactness: str = COMPACT_POST_D_LITERAL_J_EXACTNESS
    schema: str = COMPACT_POST_D_EXECUTION_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str = field(init=False, default="", repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _EXECUTION_TOKEN:
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.AUTHORITY_VIOLATION,
                "compact replay execution receipts are issuer-only",
            )
        for name in (
            "plan_hash",
            "source_ledger_hash",
            "action_registry_hash",
            "design_hash",
            "corrected_f_consumer_receipt_hash",
            "variable_shell_consumer_receipt_hash",
            "literal_j_consumer_receipt_hash",
            "pass1_literal_j_digest_axis_hash",
            "pass2_literal_j_digest_axis_hash",
            "completed_event_chain_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if self.resumed_from_checkpoint_hash is not None:
            object.__setattr__(
                self,
                "resumed_from_checkpoint_hash",
                _identity(
                    self.resumed_from_checkpoint_hash,
                    name="resumed checkpoint hash",
                ),
            )
        for name in (
            "packet_count",
            "route_window_count",
            "pass1_tile_count",
            "pass2_tile_count",
            "maximum_live_row_tile_pair_count",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name=name))
        if (
            self.pass1_tile_count != self.pass2_tile_count
            or self.pass1_literal_j_digest_axis_hash
            != self.pass2_literal_j_digest_axis_hash
            or self.maximum_live_row_tile_pair_count != 1
            or self.raw_rows_retained is not False
            or self.raw_row_d2h_bytes != 0
            or self.raw_row_scratch_bytes != 0
            or self.literal_j_two_pass_exact is not True
            or self.literal_j_pass_digest_identity_verified is not True
            or self.literal_j_exactness != COMPACT_POST_D_LITERAL_J_EXACTNESS
            or self.authority_scope != COMPACT_POST_D_AUTHORITY_SCOPE
            or self.schema != COMPACT_POST_D_EXECUTION_SCHEMA
        ):
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.AUTHORITY_VIOLATION,
                "execution receipt lost two-pass identity, bounded custody, or authority closure",
            )
        _authority_closed(self)
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return content_hash(_payload(self), prefix=_PREFIX + "execution-")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.MUTATED_RECEIPT,
                "execution receipt changed after issuance",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


def _tensor_rows(
    value: object,
    *,
    shape: tuple[int, int],
    mode: V1SampledV7CompactPostDReplayExecutionMode,
    name: str,
) -> Any:
    if mode is V1SampledV7CompactPostDReplayExecutionMode.CUDA_PRODUCTION:
        if (
            torch is None
            or type(value) is not torch.Tensor
            or value.device.type != "cuda"
            or value.dtype is not torch.float64
            or tuple(int(item) for item in value.shape) != shape
            or not value.is_contiguous()
            or value.requires_grad
        ):
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.CUDA_REQUIRED,
                f"production {name} must be one contiguous detached CUDA float64 tile",
            )
        if not bool(torch.isfinite(value).all().item()):
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.TILE_DRIFT,
                f"{name} contains nonfinite values",
            )
        return value
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not bool(np.isfinite(result).all()):
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.TILE_DRIFT,
            f"reference {name} has the wrong shape or nonfinite values",
        )
    return np.ascontiguousarray(result)


def _reference_row_digest(value: np.ndarray, *, role: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"partition-free-v1-compact-post-D-reference-row-v1\x00")
    digest.update(role.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(repr(tuple(int(item) for item in value.shape)).encode("ascii"))
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _rolling_event_hash(previous: str, event: tuple[object, ...]) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_sampled_v7_compact_post_D_event_chain_v1",
            "previous": previous,
            "event": event,
        },
        prefix=_PREFIX + "event-chain-",
    )


def _result_hash(value: object, *, name: str) -> str:
    if type(value) is str:
        return _identity(value, name=name)
    receipt_hash = getattr(value, "receipt_hash", None)
    if type(receipt_hash) is str and receipt_hash:
        return receipt_hash
    receipt = getattr(value, "receipt", None)
    receipt_hash = getattr(receipt, "receipt_hash", None)
    if type(receipt_hash) is str and receipt_hash:
        return receipt_hash
    _refuse(
        V1SampledV7CompactPostDReplayRefusalCode.CONSUMER_FAILURE,
        f"{name} finalize result lacks one compact receipt hash",
    )


def _require_consumer(
    value: object,
    *,
    methods: Sequence[str],
    source_ledger_hash: str,
    action_registry_hash: str,
    name: str,
) -> None:
    if any(not callable(getattr(value, method, None)) for method in methods):
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.INVALID_TYPE,
            f"{name} lacks the compact replay consumer interface",
        )
    if (
        getattr(value, "source_ledger_hash", None) != source_ledger_hash
        or getattr(value, "action_registry_hash", None) != action_registry_hash
    ):
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.CONSUMER_MISMATCH,
            f"{name} belongs to another source or candidate axis",
        )


def _schedule(
    plan: V1SampledV7CompactPostDReplayPlan,
    packets: Sequence[V1SampledV7CompactPostDReplayPacket],
) -> tuple[tuple[int, V1SampledV7CompactPostDReplayPacket], ...]:
    if (
        plan.literal_j_replay_mode
        is V1SampledV7CompactPostDLiteralJReplayMode.GLOBAL_DIGEST_IDENTICAL_TWO_PASS
    ):
        return tuple((pass_index, packet) for pass_index in (1, 2) for packet in packets)
    return tuple((pass_index, packet) for packet in packets for pass_index in (1, 2))


def _next_coordinate(
    *,
    schedule: Sequence[tuple[int, V1SampledV7CompactPostDReplayPacket]],
    windows: Sequence[tuple[int, int]],
    completed_events: int,
) -> tuple[int, int, int]:
    total = len(schedule) * len(windows)
    if completed_events >= total:
        return 3, 0, 0
    schedule_index, tile_index = divmod(completed_events, len(windows))
    pass_index, packet = schedule[schedule_index]
    return pass_index, packet.source_packet_ordinal, tile_index


def _checkpoint(
    *,
    plan: V1SampledV7CompactPostDReplayPlan,
    schedule: Sequence[tuple[int, V1SampledV7CompactPostDReplayPacket]],
    completed_events: int,
    event_chain_hash: str,
    pass1_literal_digests: Sequence[str],
) -> V1SampledV7CompactPostDReplayCheckpoint:
    next_pass, next_packet, next_tile = _next_coordinate(
        schedule=schedule,
        windows=plan.route_windows,
        completed_events=completed_events,
    )
    return V1SampledV7CompactPostDReplayCheckpoint(
        plan_hash=plan.plan_hash,
        source_ledger_hash=plan.source_ledger_hash,
        action_registry_hash=plan.action_registry_hash,
        design_hash=plan.design_hash,
        completed_tile_event_count=completed_events,
        completed_event_chain_hash=event_chain_hash,
        next_pass_index=next_pass,
        next_source_packet_ordinal=next_packet,
        next_tile_ordinal=next_tile,
        pass1_literal_j_digest_axis_hash=_axis_hash(
            "pass1-literal-j", tuple(pass1_literal_digests)
        ),
        _issuance_token=_CHECKPOINT_TOKEN,
    )


def execute_v1_sampled_v7_compact_post_d_replay(
    *,
    plan: V1SampledV7CompactPostDReplayPlan,
    source_ledger: (
        V1SampledV7CompactPostDReplaySourceLedger
        | V1SampledV7CompactPostDRowFreeSourceLedgerV2
    ),
    action_registry: V1SampledV7CompactPostDFrozenActionRegistry,
    design: V1SampledV7CompactPostDReplayDesign,
    source_provider: object,
    corrected_f_consumer: object,
    variable_shell_consumer: object,
    literal_j_consumer: object,
    resume_checkpoint: V1SampledV7CompactPostDReplayCheckpoint | None = None,
    failure_point: V1SampledV7CompactPostDReplayFailurePoint | None = None,
) -> V1SampledV7CompactPostDReplayExecutionReceipt:
    """Replay the exact D ledger into three compact consumer families.

    Consumer interfaces are deliberately narrow and match the existing
    streaming numerical owners through small adapters:

    * corrected-F: ``begin_packet``, ``consume_corrected_route_chunk``,
      ``end_packet``, ``finalize``;
    * variable shell: ``begin_packet``, ``consume_dual_route_chunk``,
      ``end_packet``, ``finalize``;
    * literal-J: ``begin_pass_packet``, ``consume_literal_j_route_chunk``,
      ``end_pass_packet``, ``rewind_for_pass2`` (or
      ``rewind_packet_for_pass2``), ``finalize``.

    A resumed execution reconstructs all consumer state from the beginning,
    verifies the checkpoint's rolling digest at its exact cursor, and only
    then advances.  Thus no device accumulator or raw row is serialized.
    """

    source_is_exact = type(source_ledger) in {
        V1SampledV7CompactPostDReplaySourceLedger,
        V1SampledV7CompactPostDRowFreeSourceLedgerV2,
    }
    exact = (
        (plan, V1SampledV7CompactPostDReplayPlan),
        (action_registry, V1SampledV7CompactPostDFrozenActionRegistry),
        (design, V1SampledV7CompactPostDReplayDesign),
    )
    if not source_is_exact or any(
        type(value) is not expected for value, expected in exact
    ):
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.INVALID_TYPE,
            "execution requires exact compact replay owners",
        )
    plan.assert_integrity()
    observed_ledger_hash = source_ledger.assert_integrity()
    if observed_ledger_hash != plan.source_ledger_hash:
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.SOURCE_DRIFT,
            "execution source differs from the open replay plan",
        )
    action_registry.assert_integrity(plan.action_registry_hash)
    design.assert_integrity()
    if design.design_hash != plan.design_hash:
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.SOURCE_DRIFT,
            "execution design differs from the open plan",
        )
    provider_methods = ("tiles_for", "begin_packet", "end_packet", "finalize")
    if any(not callable(getattr(source_provider, method, None)) for method in provider_methods):
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.INVALID_TYPE,
            "source provider lacks begin/tiles/end/finalize replay hooks",
        )
    if (
        getattr(source_provider, "source_ledger_hash", None) != source_ledger.ledger_hash
        or getattr(source_provider, "execution_mode", None) is not design.execution_mode
    ):
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.SOURCE_DRIFT,
            "source provider is not bound to the exact ledger and execution mode",
        )
    _require_consumer(
        corrected_f_consumer,
        methods=("begin_packet", "consume_corrected_route_chunk", "end_packet", "finalize"),
        source_ledger_hash=source_ledger.ledger_hash,
        action_registry_hash=action_registry.registry_hash,
        name="corrected-F action-family consumer",
    )
    _require_consumer(
        variable_shell_consumer,
        methods=("begin_packet", "consume_dual_route_chunk", "end_packet", "finalize"),
        source_ledger_hash=source_ledger.ledger_hash,
        action_registry_hash=action_registry.registry_hash,
        name="variable-shell consumer",
    )
    literal_methods = (
        "begin_pass_packet",
        "consume_literal_j_route_chunk",
        "end_pass_packet",
        "finalize",
    )
    _require_consumer(
        literal_j_consumer,
        methods=literal_methods,
        source_ledger_hash=source_ledger.ledger_hash,
        action_registry_hash=action_registry.registry_hash,
        name="literal-J two-pass consumer",
    )
    if plan.literal_j_replay_mode is V1SampledV7CompactPostDLiteralJReplayMode.GLOBAL_DIGEST_IDENTICAL_TWO_PASS:
        if not callable(getattr(literal_j_consumer, "rewind_for_pass2", None)):
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.PASS_DRIFT,
                "global literal-J mode requires an explicit pass-2 rewind",
            )
    elif not callable(getattr(literal_j_consumer, "rewind_packet_for_pass2", None)):
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.PASS_DRIFT,
            "interleaved literal-J mode requires a bounded packet rewind adapter",
        )
    checkpoint = resume_checkpoint
    if checkpoint is not None:
        if type(checkpoint) is not V1SampledV7CompactPostDReplayCheckpoint:
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.INVALID_TYPE,
                "resume checkpoint must be exact",
            )
        checkpoint.assert_integrity()
        if (
            checkpoint.plan_hash != plan.plan_hash
            or checkpoint.source_ledger_hash != source_ledger.ledger_hash
            or checkpoint.action_registry_hash != action_registry.registry_hash
            or checkpoint.design_hash != design.design_hash
        ):
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.RESTART_MISMATCH,
                "checkpoint belongs to another source, candidate family, or design",
            )
    schedule = _schedule(plan, source_ledger.packets)
    event_chain = _EMPTY_ROLLING_HASH
    completed_events = 0
    pass1_digests: list[str] = []
    pass2_digests: list[str] = []
    pass1_by_coordinate: dict[tuple[int, int, int], str] = {}
    maximum_live_row_tile_pair_count = 0
    checkpoint_verified = checkpoint is None or checkpoint.completed_tile_event_count == 0
    previous_pass = 0
    previous_packet: int | None = None
    try:
        for pass_index, packet in schedule:
            if pass_index != previous_pass:
                if (
                    pass_index == 2
                    and plan.literal_j_replay_mode
                    is V1SampledV7CompactPostDLiteralJReplayMode.GLOBAL_DIGEST_IDENTICAL_TWO_PASS
                ):
                    literal_j_consumer.rewind_for_pass2()
                previous_pass = pass_index
                if failure_point == V1SampledV7CompactPostDReplayFailurePoint(
                    boundary="pass", pass_index=pass_index
                ):
                    raise V1SampledV7CompactPostDReplayInterrupted(
                        "injected compact replay pass-boundary interruption",
                        checkpoint=_checkpoint(
                            plan=plan,
                            schedule=schedule,
                            completed_events=completed_events,
                            event_chain_hash=event_chain,
                            pass1_literal_digests=pass1_digests,
                        ),
                    )
            if (
                pass_index == 2
                and plan.literal_j_replay_mode
                is V1SampledV7CompactPostDLiteralJReplayMode.PACKET_INTERLEAVED_DIGEST_IDENTICAL_TWO_PASS
            ):
                literal_j_consumer.rewind_packet_for_pass2(packet)
            source_provider.begin_packet(pass_index, packet)
            literal_j_consumer.begin_pass_packet(pass_index, packet)
            if pass_index == 1:
                corrected_f_consumer.begin_packet(packet)
                variable_shell_consumer.begin_packet(packet)
            raw_tiles = source_provider.tiles_for(pass_index, packet, plan.route_windows)
            try:
                tile_iterator = iter(raw_tiles)
            except TypeError as error:
                raise V1SampledV7CompactPostDReplayRefusal(
                    V1SampledV7CompactPostDReplayRefusalCode.INVALID_TYPE,
                    "source tiles must be one finite ordered iterable",
                ) from error
            for tile_ordinal, window in enumerate(plan.route_windows):
                try:
                    tile = next(tile_iterator)
                except StopIteration:
                    _refuse(
                        V1SampledV7CompactPostDReplayRefusalCode.TILE_DRIFT,
                        "source returned a partial route-window axis",
                    )
                maximum_live_row_tile_pair_count = max(
                    maximum_live_row_tile_pair_count,
                    1,
                )
                if type(tile) is not V1SampledV7CompactPostDReplayTile:
                    _refuse(
                        V1SampledV7CompactPostDReplayRefusalCode.INVALID_TYPE,
                        "source tiles must use the exact compact tile type",
                    )
                start, stop = window
                if (
                    tile.source_ledger_hash != source_ledger.ledger_hash
                    or tile.packet_hash != packet.packet_hash
                    or tile.pass_index != pass_index
                    or (tile.route_start, tile.route_stop) != window
                ):
                    _refuse(
                        V1SampledV7CompactPostDReplayRefusalCode.TILE_DRIFT,
                        "source tile left its registered source/pass/packet/window coordinate",
                    )
                width = stop - start
                literal = _tensor_rows(
                    tile.literal_j_rows,
                    shape=(packet.context_count, width),
                    mode=design.execution_mode,
                    name="literal-J rows",
                )
                corrected = None
                if pass_index == 1:
                    corrected = _tensor_rows(
                        tile.corrected_a_rows,
                        shape=(packet.context_count, width),
                        mode=design.execution_mode,
                        name="corrected-A rows",
                    )
                elif tile.corrected_a_rows is not None:
                    _refuse(
                        V1SampledV7CompactPostDReplayRefusalCode.PASS_DRIFT,
                        "literal-J pass 2 must not reacquire or fan out corrected-A rows",
                    )
                if design.execution_mode is V1SampledV7CompactPostDReplayExecutionMode.CUDA_PRODUCTION:
                    corrected_digest = tile.corrected_a_source_digest
                    literal_digest = tile.literal_j_source_digest
                    if (
                        literal_digest is None
                        or (pass_index == 1 and corrected_digest is None)
                    ):
                        _refuse(
                            V1SampledV7CompactPostDReplayRefusalCode.SOURCE_DIGEST_REQUIRED,
                            "production tiles require source-owned immutable A/J digests",
                        )
                    literal_digest = _identity(literal_digest, name="literal-J source digest")
                    if corrected_digest is not None:
                        corrected_digest = _identity(
                            corrected_digest, name="corrected-A source digest"
                        )
                else:
                    literal_digest = _reference_row_digest(literal, role="literal-J")
                    corrected_digest = (
                        None
                        if corrected is None
                        else _reference_row_digest(corrected, role="corrected-A")
                    )
                    if tile.literal_j_source_digest not in {None, literal_digest}:
                        _refuse(
                            V1SampledV7CompactPostDReplayRefusalCode.SOURCE_DRIFT,
                            "reference literal-J digest differs from exact row bytes",
                        )
                    if tile.corrected_a_source_digest not in {None, corrected_digest}:
                        _refuse(
                            V1SampledV7CompactPostDReplayRefusalCode.SOURCE_DRIFT,
                            "reference corrected-A digest differs from exact row bytes",
                        )
                coordinate = (packet.source_packet_ordinal, tile_ordinal, start)
                if pass_index == 1:
                    pass1_by_coordinate[coordinate] = literal_digest
                    pass1_digests.append(literal_digest)
                    corrected_f_consumer.consume_corrected_route_chunk(
                        packet, start, stop, corrected, corrected_digest
                    )
                    variable_shell_consumer.consume_dual_route_chunk(
                        packet,
                        start,
                        stop,
                        corrected,
                        literal,
                        corrected_digest,
                        literal_digest,
                    )
                else:
                    if pass1_by_coordinate.get(coordinate) != literal_digest:
                        _refuse(
                            V1SampledV7CompactPostDReplayRefusalCode.PASS_DRIFT,
                            "literal-J pass 2 did not replay the exact pass-1 tile digest",
                        )
                    pass2_digests.append(literal_digest)
                literal_j_consumer.consume_literal_j_route_chunk(
                    pass_index, packet, start, stop, literal, literal_digest
                )
                event = (
                    pass_index,
                    packet.packet_hash,
                    tile_ordinal,
                    start,
                    stop,
                    corrected_digest if pass_index == 1 else None,
                    literal_digest,
                )
                event_chain = _rolling_event_hash(event_chain, event)
                completed_events += 1
                if (
                    checkpoint is not None
                    and completed_events == checkpoint.completed_tile_event_count
                ):
                    observed_pass1_axis = _axis_hash(
                        "pass1-literal-j", tuple(pass1_digests)
                    )
                    if (
                        event_chain != checkpoint.completed_event_chain_hash
                        or observed_pass1_axis
                        != checkpoint.pass1_literal_j_digest_axis_hash
                        or _next_coordinate(
                            schedule=schedule,
                            windows=plan.route_windows,
                            completed_events=completed_events,
                        )
                        != (
                            checkpoint.next_pass_index,
                            checkpoint.next_source_packet_ordinal,
                            checkpoint.next_tile_ordinal,
                        )
                    ):
                        _refuse(
                            V1SampledV7CompactPostDReplayRefusalCode.RESTART_MISMATCH,
                            "replayed source prefix differs from the compact checkpoint",
                        )
                    checkpoint_verified = True
                if failure_point == V1SampledV7CompactPostDReplayFailurePoint(
                    boundary="tile",
                    pass_index=pass_index,
                    source_packet_ordinal=packet.source_packet_ordinal,
                    tile_ordinal=tile_ordinal,
                ):
                    raise V1SampledV7CompactPostDReplayInterrupted(
                        "injected compact replay tile-boundary interruption",
                        checkpoint=_checkpoint(
                            plan=plan,
                            schedule=schedule,
                            completed_events=completed_events,
                            event_chain_hash=event_chain,
                            pass1_literal_digests=pass1_digests,
                        ),
                    )
                # All consumers are synchronous.  Release every row view and
                # the source tile before requesting the next registered
                # window; the iterator is never prefetched or materialized.
                del corrected, literal, tile
            try:
                extra_tile = next(tile_iterator)
            except StopIteration:
                pass
            else:
                del extra_tile
                _refuse(
                    V1SampledV7CompactPostDReplayRefusalCode.TILE_DRIFT,
                    "source returned an extra route-window tile",
                )
            del tile_iterator, raw_tiles
            literal_j_consumer.end_pass_packet(pass_index, packet)
            if pass_index == 1:
                corrected_f_consumer.end_packet(packet)
                variable_shell_consumer.end_packet(packet)
            source_provider.end_packet(pass_index, packet)
            previous_packet = packet.source_packet_ordinal
            if failure_point == V1SampledV7CompactPostDReplayFailurePoint(
                boundary="packet",
                pass_index=pass_index,
                source_packet_ordinal=packet.source_packet_ordinal,
            ):
                raise V1SampledV7CompactPostDReplayInterrupted(
                    "injected compact replay packet-boundary interruption",
                    checkpoint=_checkpoint(
                        plan=plan,
                        schedule=schedule,
                        completed_events=completed_events,
                        event_chain_hash=event_chain,
                        pass1_literal_digests=pass1_digests,
                    ),
                )
        del previous_packet
        if not checkpoint_verified:
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.RESTART_MISMATCH,
                "checkpoint cursor lies beyond the completed replay schedule",
            )
        source_provider.finalize()
        corrected_result = corrected_f_consumer.finalize()
        variable_result = variable_shell_consumer.finalize()
        literal_result = literal_j_consumer.finalize()
    except V1SampledV7CompactPostDReplayInterrupted:
        for owner in (
            source_provider,
            corrected_f_consumer,
            variable_shell_consumer,
            literal_j_consumer,
        ):
            abort = getattr(owner, "abort_unsealed", None)
            if callable(abort):
                abort()
        raise
    except V1SampledV7CompactPostDReplayRefusal:
        for owner in (
            source_provider,
            corrected_f_consumer,
            variable_shell_consumer,
            literal_j_consumer,
        ):
            abort = getattr(owner, "abort_unsealed", None)
            if callable(abort):
                abort()
        raise
    except Exception as error:
        for owner in (
            source_provider,
            corrected_f_consumer,
            variable_shell_consumer,
            literal_j_consumer,
        ):
            abort = getattr(owner, "abort_unsealed", None)
            if callable(abort):
                abort()
        raise V1SampledV7CompactPostDReplayRefusal(
            V1SampledV7CompactPostDReplayRefusalCode.CONSUMER_FAILURE,
            f"compact replay consumer/provider failed: {type(error).__name__}: {error}",
        ) from error
    pass1_axis = _axis_hash("pass1-literal-j", tuple(pass1_digests))
    pass2_axis = _axis_hash("pass1-literal-j", tuple(pass2_digests))
    if pass1_axis != pass2_axis or len(pass1_digests) != len(pass2_digests):
        _refuse(
            V1SampledV7CompactPostDReplayRefusalCode.PASS_DRIFT,
            "literal-J two-pass digest axes do not close",
        )
    return V1SampledV7CompactPostDReplayExecutionReceipt(
        plan_hash=plan.plan_hash,
        source_ledger_hash=source_ledger.ledger_hash,
        action_registry_hash=action_registry.registry_hash,
        design_hash=design.design_hash,
        corrected_f_consumer_receipt_hash=_result_hash(
            corrected_result, name="corrected-F consumer receipt"
        ),
        variable_shell_consumer_receipt_hash=_result_hash(
            variable_result, name="variable-shell consumer receipt"
        ),
        literal_j_consumer_receipt_hash=_result_hash(
            literal_result, name="literal-J consumer receipt"
        ),
        packet_count=source_ledger.packet_count,
        route_window_count=len(plan.route_windows),
        pass1_tile_count=len(pass1_digests),
        pass2_tile_count=len(pass2_digests),
        pass1_literal_j_digest_axis_hash=pass1_axis,
        pass2_literal_j_digest_axis_hash=pass2_axis,
        completed_event_chain_hash=event_chain,
        resumed_from_checkpoint_hash=(
            None if checkpoint is None else checkpoint.checkpoint_hash
        ),
        maximum_live_row_tile_pair_count=maximum_live_row_tile_pair_count,
        _issuance_token=_EXECUTION_TOKEN,
    )


class V1SampledV7CompactPostDVariableShellAccumulatorAdapter:
    """Adapter from compact fanout to the existing variable-shell accumulator."""

    def __init__(
        self,
        *,
        accumulator: V1SignedWorkVariableShellSourceCUDAAccumulator,
        source_ledger_hash: str,
        action_registry_hash: str,
    ) -> None:
        if type(accumulator) is not V1SignedWorkVariableShellSourceCUDAAccumulator:
            raise TypeError("variable-shell adapter requires the exact accumulator")
        self.accumulator = accumulator
        self.source_ledger_hash = _identity(source_ledger_hash, name="source ledger hash")
        self.action_registry_hash = _identity(action_registry_hash, name="action registry hash")
        self._context_cursor = 0

    def begin_packet(self, packet: V1SampledV7CompactPostDReplayPacket) -> None:
        registration = self.accumulator.registration
        start = self._context_cursor
        stop = start + packet.context_count
        if (
            tuple(registration.registered_context_ids[start:stop]) != packet.context_ids
            or tuple(registration.registered_draw_ids[start:stop]) != packet.draw_ids
            or tuple(registration.registered_measurement_ids[start:stop])
            != packet.measurement_ids
        ):
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.PACKET_DRIFT,
                "variable-shell registered packet prefix differs from replay source",
            )
        self.accumulator.begin_packet(
            context_ids=registration.registered_context_ids[start:stop],
            draw_ids=registration.registered_draw_ids[start:stop],
            measurement_ids=registration.registered_measurement_ids[start:stop],
            capture_ids=registration.registered_capture_ids[start:stop],
            source_packet_hashes=registration.registered_source_packet_hashes[start:stop],
        )

    def consume_dual_route_chunk(
        self,
        packet: V1SampledV7CompactPostDReplayPacket,
        start: int,
        stop: int,
        corrected_rows: Any,
        literal_rows: Any,
        corrected_digest: str,
        literal_digest: str,
    ) -> None:
        del packet, stop, corrected_digest, literal_digest
        registration = self.accumulator.registration
        binding = self.accumulator.issue_route_chunk_binding(
            route_start=start,
            corrected_evidence_rows=corrected_rows,
            literal_j_rows=literal_rows,
            route_axis_hash=registration.route_axis_hash,
            route_id_axis_hash=registration.route_id_axis_hash,
            same_capture_vjp_source_hash=registration.same_capture_vjp_source_hash,
            corrected_evidence_stream_source_hash=(
                registration.corrected_evidence_stream_source_hash
            ),
            literal_j_stream_source_hash=registration.literal_j_stream_source_hash,
        )
        self.accumulator.consume_route_chunk(
            route_start=start,
            corrected_evidence_rows=corrected_rows,
            literal_j_rows=literal_rows,
            route_axis_hash=registration.route_axis_hash,
            route_id_axis_hash=registration.route_id_axis_hash,
            same_capture_vjp_source_hash=registration.same_capture_vjp_source_hash,
            corrected_evidence_stream_source_hash=(
                registration.corrected_evidence_stream_source_hash
            ),
            literal_j_stream_source_hash=registration.literal_j_stream_source_hash,
            chunk_binding=binding,
        )

    def end_packet(self, packet: V1SampledV7CompactPostDReplayPacket) -> None:
        self.accumulator.end_packet()
        self._context_cursor += packet.context_count

    def finalize(self) -> object:
        return self.accumulator.seal()

    def abort_unsealed(self) -> None:
        self.accumulator.abort_unsealed()


class V1SampledV7CompactPostDLiteralJAccumulatorAdapter:
    """Global two-pass adapter for the existing literal-J CUDA accumulator."""

    def __init__(
        self,
        *,
        accumulator: V1LiteralJTwoPassDFixedProjectionCUDAAccumulator,
        source_ledger_hash: str,
        action_registry_hash: str,
    ) -> None:
        if type(accumulator) is not V1LiteralJTwoPassDFixedProjectionCUDAAccumulator:
            raise TypeError("literal-J adapter requires the exact two-pass accumulator")
        self.accumulator = accumulator
        self.source_ledger_hash = _identity(source_ledger_hash, name="source ledger hash")
        self.action_registry_hash = _identity(action_registry_hash, name="action registry hash")

    def begin_pass_packet(
        self, pass_index: int, packet: V1SampledV7CompactPostDReplayPacket
    ) -> None:
        registration = self.accumulator.registration
        expected = registration.packets[packet.source_packet_ordinal]
        if (
            expected.context_ids != packet.context_ids
            or expected.draw_ids != packet.draw_ids
            or expected.measurement_ids != packet.measurement_ids
            or expected.capture_ids != packet.capture_ids
            or expected.source_packet_hashes != packet.source_packet_hashes
        ):
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.PACKET_DRIFT,
                "literal-J registration packet differs from replay source",
            )
        kwargs = dict(
            context_ids=expected.context_ids,
            draw_ids=expected.draw_ids,
            measurement_ids=expected.measurement_ids,
            capture_ids=expected.capture_ids,
            source_packet_hashes=expected.source_packet_hashes,
            model_identity_hash=registration.model_identity_hash,
            chart_identity_hash=registration.chart_identity_hash,
            current_mask_hash=registration.current_mask_hash,
            proposal_generation=registration.proposal_generation,
            route_axis_hash=registration.route_axis_hash,
            route_id_axis_hash=registration.route_id_axis_hash,
            same_capture_and_vjp_execution_proof_hash=(
                registration.same_capture_and_vjp_execution_proof_hash
            ),
            same_capture_vjp_source_hash=expected.same_capture_vjp_source_hash,
            literal_j_stream_source_hash=expected.literal_j_stream_source_hash,
        )
        if pass_index == 1:
            self.accumulator.begin_pass1_packet(**kwargs)
        elif pass_index == 2:
            self.accumulator.begin_pass2_packet(**kwargs)
        else:
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.PASS_DRIFT,
                "literal-J pass index must be one or two",
            )

    def consume_literal_j_route_chunk(
        self,
        pass_index: int,
        packet: V1SampledV7CompactPostDReplayPacket,
        start: int,
        stop: int,
        literal_rows: Any,
        literal_digest: str,
    ) -> None:
        del packet
        if pass_index == 1:
            self.accumulator.consume_pass1_tile(
                start,
                stop,
                literal_rows,
                literal_j_tile_source_hash=literal_digest,
            )
        elif pass_index == 2:
            self.accumulator.consume_pass2_tile(
                start,
                stop,
                literal_rows,
                literal_j_tile_source_hash=literal_digest,
            )
        else:
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.PASS_DRIFT,
                "literal-J pass index must be one or two",
            )

    def end_pass_packet(
        self, pass_index: int, packet: V1SampledV7CompactPostDReplayPacket
    ) -> None:
        del packet
        if pass_index == 1:
            self.accumulator.end_pass1_packet()
        elif pass_index == 2:
            self.accumulator.end_pass2_packet()
        else:
            _refuse(
                V1SampledV7CompactPostDReplayRefusalCode.PASS_DRIFT,
                "literal-J pass index must be one or two",
            )

    def rewind_for_pass2(self) -> str:
        return self.accumulator.rewind_for_pass2()

    def finalize(self) -> object:
        return self.accumulator.finalize()


__all__ = [
    "COMPACT_POST_D_ACTION_REGISTRY_SCHEMA",
    "COMPACT_POST_D_AUTHORITY_SCOPE",
    "COMPACT_POST_D_CHECKPOINT_SCHEMA",
    "COMPACT_POST_D_DESIGN_SCHEMA",
    "COMPACT_POST_D_DEVICE_FINAL_SOURCE_SCHEMA",
    "COMPACT_POST_D_DEVICE_FINAL_SOURCE_SCHEMA_V2",
    "COMPACT_POST_D_EXECUTION_SCHEMA",
    "COMPACT_POST_D_LITERAL_J_EXACTNESS",
    "COMPACT_POST_D_PLAN_SCHEMA",
    "COMPACT_POST_D_SOURCE_LEDGER_SCHEMA",
    "COMPACT_POST_D_SOURCE_LEDGER_SCHEMA_V2",
    "V1SampledV7CompactDeviceResidentDFinalSourceReceipt",
    "V1SampledV7CompactDeviceResidentDRowFreeFinalSourceReceiptV2",
    "V1SampledV7CompactPostDFrozenAction",
    "V1SampledV7CompactPostDFrozenActionRegistry",
    "V1SampledV7CompactPostDLiteralJAccumulatorAdapter",
    "V1SampledV7CompactPostDLiteralJReplayMode",
    "V1SampledV7CompactPostDReplayCheckpoint",
    "V1SampledV7CompactPostDReplayDesign",
    "V1SampledV7CompactPostDReplayExecutionMode",
    "V1SampledV7CompactPostDReplayExecutionReceipt",
    "V1SampledV7CompactPostDReplayFailurePoint",
    "V1SampledV7CompactPostDReplayInterrupted",
    "V1SampledV7CompactPostDReplayPacket",
    "V1SampledV7CompactPostDReplayPlan",
    "V1SampledV7CompactPostDReplayRefusal",
    "V1SampledV7CompactPostDReplayRefusalCode",
    "V1SampledV7CompactPostDReplaySourceLedger",
    "V1SampledV7CompactPostDRowFreeSourceLedgerV2",
    "V1SampledV7CompactPostDReplayTile",
    "V1SampledV7CompactPostDSourceProvenance",
    "V1SampledV7CompactPostDVariableShellAccumulatorAdapter",
    "execute_v1_sampled_v7_compact_post_d_replay",
    "issue_v1_sampled_v7_compact_device_final_source_from_exact_bridge",
    "issue_v1_sampled_v7_compact_row_free_final_source_from_exact_bridge_v2",
    "make_v1_sampled_v7_compact_post_d_reference_actions_for_tests",
    "make_v1_sampled_v7_compact_post_d_reference_source_for_tests",
    "make_v1_sampled_v7_compact_post_d_replay_design",
    "open_v1_sampled_v7_compact_post_d_replay",
    "register_v1_sampled_v7_compact_post_d_frozen_actions",
    "register_v1_sampled_v7_compact_post_d_legacy_row_backed_source_ledger",
    "register_v1_sampled_v7_compact_post_d_source_ledger",
    "register_v1_sampled_v7_compact_post_d_row_free_source_ledger_v2",
]
