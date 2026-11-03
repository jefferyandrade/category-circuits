"""Pre-outcome binding of one sampled ``D_W2`` packet to compact source custody.

The sampled variable-shell runtime has three independently useful owners:

* the causal suffix registration owns the complete target/complement order;
* a source-issued measurement schedule owns every predictable draw,
  measurement, capture, and source-packet coordinate; and
* a preregistered :class:`MeasurementRequest` records request provenance; and
* the live append owner owns the chart, shell, geometry, stream, resource,
  and alpha lineage into which the packet may be lowered.

This module joins those owners before any provider result exists.  It emits a
complete :class:`V1SignedWorkVariableShellSourceRegistration` plus the exact
append-packet reservation that will later admit its compact source handle.
Target contexts always precede complement contexts.  The source-issued
finite-population measure owns the exact ``D1`` plus cumulative residual
weights.  This binder only derives the packet-local normalized representation
required by the compact-source transport; it never presents that normalized
representation as the full-population estimand.  Neither raw population
proportions, empirical packet proportions, nor caller-normalized weights are
accepted.  The compact source has no full-population decision authority until
an exact ``D1`` fixed-action source and a stratum-resolved residual compositor
join the registered measure downstream.

``MeasurementRequest.request_id`` is deliberately not a sampling coordinate.
The complete append-order identity axis is frozen before a request exists, so
request replay, request metadata, or a different request producer cannot
change which physical population draw is being measured.

The registered ``D_W2`` prefix identity is derived from the sealed suffix and
the predictable packet, not from the outcome-dependent terminal prefix-freeze
receipt.  A later freeze may therefore verify that it covers this packet
without becoming an ancestor of the measurement request.

No provider receipt, measurement outcome, score, confidence result, action
decision, C/K/O authority, commit authority, or Pareto authority is accepted
or emitted here.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
from fractions import Fraction
import math
from numbers import Integral
from typing import Final, NoReturn, Sequence

from mechanistic_systems.contracts.measurement import MeasurementRequest

from .contracts import content_hash
from .modern_v1_alexnet_fq_registration import (
    V1AlexNetFQCausalGenerationFrameReceipt,
    V1AlexNetFQCausalRoleSamplingStreamReceipt,
)
from .modern_v1_native_fiber_axis import V1NativeDestinationFiberAxisReceipt
from .modern_v1_sampled_variable_shell_dw2_suffix import (
    V1SampledVariableShellDW2SuffixPacket,
    V1SampledVariableShellDW2SuffixPrefixFreezeReceipt,
    V1SampledVariableShellDW2SuffixRegistration,
    _axis_hash as _suffix_axis_hash,
    _pair_axis_hash as _suffix_pair_axis_hash,
)
from .modern_v1_sampled_residual_population_measure import (
    V1SampledResidualPopulationMeasureDesign,
    V1SampledResidualPopulationMeasureRefusal,
    V1SampledResidualPopulationWeightReceipt,
    issue_v1_sampled_residual_population_weights_for_packet,
    register_v1_sampled_residual_population_measure,
)
from .modern_v1_signed_work_fixed_primal_shell_union import (
    V1SignedWorkFixedPrimalShellUnionReceipt,
)
from .modern_v1_signed_work_variable_shell_source_append import (
    V1SignedWorkVariableShellSourceAppendLineage,
    V1SignedWorkVariableShellSourceAppendOwner,
    V1SignedWorkVariableShellSourceAppendPacketRegistration,
    V1SignedWorkVariableShellSourceAppendRefusal,
)
from .modern_v1_signed_work_variable_shell_source_cuda import (
    V1SignedWorkVariableShellSourceRefusal,
    V1SignedWorkVariableShellSourceRegistration,
    v1_variable_shell_population_axis_hash,
)


SAMPLED_VARIABLE_SHELL_PACKET_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_sampled_variable_shell_D_W2_packet_registration_v6"
)
SAMPLED_VARIABLE_SHELL_PACKET_REGISTRATION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_sampled_variable_shell_D_W2_packet_registration_receipt_v6"
)
SAMPLED_VARIABLE_SHELL_PACKET_REQUEST_PAYLOAD_SCHEMA: Final = (
    "partition_free_v1_sampled_variable_shell_D_W2_packet_measurement_request_v6"
)
SAMPLED_VARIABLE_SHELL_PACKET_REQUEST_VERSION: Final = (
    "modern-v1-sampled-variable-shell-D-W2-source-scheduled-v6"
)
SAMPLED_VARIABLE_SHELL_MEASUREMENT_SOURCE_LINEAGE_SCHEMA: Final = (
    "partition_free_v1_sampled_variable_shell_D_W2_measurement_source_lineage_v2"
)
SAMPLED_VARIABLE_SHELL_MEASUREMENT_SCHEDULE_SCHEMA: Final = (
    "partition_free_v1_sampled_variable_shell_D_W2_measurement_schedule_v1"
)
SAMPLED_VARIABLE_SHELL_MEASUREMENT_SCHEDULE_PACKET_SCHEMA: Final = (
    "partition_free_v1_sampled_variable_shell_D_W2_measurement_schedule_packet_v1"
)
SAMPLED_VARIABLE_SHELL_TWO_BRANCH_KKT_ID: Final = (
    "spectral_plus_diagonal_l1_provisional_shell_union_v1"
)
SAMPLED_VARIABLE_SHELL_PACKET_REGISTRATION_AUTHORITY: Final = (
    "pre_outcome_packet_and_compact_source_registration_only_no_provider_"
    "outcome_score_confidence_action_C_K_O_commit_terminal_or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-variable-shell-packet-registration-"
_RAW_TARGET_LOGIT_COORDINATE: Final = "raw_target_logit"
_STRATA: Final = ("target", "complement")
_ISSUER_TOKEN: Final = object()
_SCHEDULE_ISSUER_TOKEN: Final = object()
_SCHEDULE_PACKET_ISSUER_TOKEN: Final = object()
_MEASUREMENT_SOURCE_ISSUER_TOKEN: Final = object()


class V1SampledVariableShellPacketRegistrationRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_IDENTITY = "stale_identity"
    REQUEST_MISMATCH = "request_mismatch"
    ORDER_MISMATCH = "order_mismatch"
    OVERLAP = "overlap"
    WEIGHT_MISMATCH = "weight_mismatch"
    OUTCOME_LEAKAGE = "outcome_leakage"
    AUTHORITY_VIOLATION = "authority_violation"
    PREFIX_FREEZE_MISMATCH = "prefix_freeze_mismatch"
    SCHEDULE_MISMATCH = "schedule_mismatch"
    REPLAY = "replay"


class V1SampledVariableShellPacketRegistrationRefusal(RuntimeError):
    """Typed, fail-closed refusal for one predictable source packet."""

    def __init__(
        self,
        code: V1SampledVariableShellPacketRegistrationRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledVariableShellPacketRegistrationRefusalCode:
            raise TypeError("packet-registration refusal code has the wrong type")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledVariableShellPacketRegistrationRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledVariableShellPacketRegistrationRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _fraction(value: object, *, name: str) -> Fraction:
    if type(value) is not tuple or len(value) != 2:
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.INVALID_TYPE,
            f"{name} must be one exact rational pair",
        )
    numerator = _nonnegative_int(value[0], name=f"{name} numerator")
    denominator = _nonnegative_int(value[1], name=f"{name} denominator")
    if denominator <= 0:
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.WEIGHT_MISMATCH,
            f"{name} denominator must be positive",
        )
    return Fraction(numerator, denominator)


def _binary64_fraction_enclosure(value: Fraction) -> tuple[float, float, float]:
    """Return a nearest binary64 center with a strict outward enclosure."""

    center = float(value)
    if not math.isfinite(center):
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.WEIGHT_MISMATCH,
            "exact source-local context weight overflowed binary64",
        )
    center_fraction = Fraction.from_float(center)
    lower = center
    upper = center
    if center_fraction > value:
        lower = math.nextafter(lower, -math.inf)
    elif center_fraction < value:
        upper = math.nextafter(upper, math.inf)
    lower = math.nextafter(lower, -math.inf)
    upper = math.nextafter(upper, math.inf)
    if not (
        lower <= center <= upper
        and Fraction.from_float(lower) <= value <= Fraction.from_float(upper)
    ):
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.WEIGHT_MISMATCH,
            "source-local binary64 weight enclosure is not outward",
        )
    return lower, center, upper


def _validate_binary64_fraction_enclosure(
    *,
    exact: Fraction,
    enclosure: object,
    name: str,
) -> tuple[float, float, float]:
    if (
        type(enclosure) is not tuple
        or len(enclosure) != 3
        or any(type(item) is not float or not math.isfinite(item) for item in enclosure)
    ):
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.WEIGHT_MISMATCH,
            f"{name} must be one finite binary64 enclosure",
        )
    observed = enclosure
    lower, center, upper = observed
    if not (
        lower <= center <= upper
        and Fraction.from_float(lower) <= exact <= Fraction.from_float(upper)
        and observed == _binary64_fraction_enclosure(exact)
    ):
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.WEIGHT_MISMATCH,
            f"{name} is not the canonical strict enclosure of its exact rational",
        )
    return observed


def _identity_axis(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) not in {tuple, list}:
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.INVALID_TYPE,
            f"{name} must be one ordered string sequence",
        )
    result = tuple(_identity(item, name=f"{name} item") for item in value)
    if not result or len(result) != len(set(result)):
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.OVERLAP,
            f"{name} must be nonempty and unique",
        )
    return result


def _source_axis_hash(role: str, values: Sequence[str]) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_variable_shell_registered_axis_v1",
            "role": role,
            "values": tuple(values),
        },
        prefix="partition-free-v1-variable-shell-source-axis-",
    )


def _possibly_empty_identity_axis(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.INVALID_TYPE,
            f"{name} must be one immutable ordered string axis",
        )
    result = tuple(_identity(item, name=f"{name} item") for item in value)
    if len(result) != len(set(result)):
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.OVERLAP,
            f"{name} repeats an identity",
        )
    return result


def _packet_stratum_axis(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.INVALID_TYPE,
            f"{name} must be one immutable ordered stratum axis",
        )
    result = tuple(_identity(item, name=f"{name} item") for item in value)
    if any(item not in _STRATA for item in result):
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.SCHEDULE_MISMATCH,
            f"{name} may contain only target/complement strata",
        )
    return result


@dataclass(frozen=True, slots=True)
class V1SampledVariableShellMeasurementSchedulePacket:
    """Request-independent identities for one predictable suffix packet."""

    schedule_source_hash: str
    suffix_registration_hash: str
    generation_frame_hash: str
    discovery_stream_hash: str
    capture_policy_identity_hash: str
    packet_ordinal: int
    suffix_packet_hash: str
    before_suffix_stratum_counts: tuple[int, int]
    after_suffix_stratum_counts: tuple[int, int]
    registered_context_ids: tuple[str, ...]
    registered_stratum_ids: tuple[str, ...]
    registered_draw_ids: tuple[str, ...]
    registered_measurement_ids: tuple[str, ...]
    registered_capture_ids: tuple[str, ...]
    registered_source_packet_hashes: tuple[str, ...]
    source_capture_identity_hash: str
    request_id_is_not_a_sampling_coordinate: bool = True
    outcome_payload_present: bool = False
    scoring_or_decision_authority: bool = False
    schema: str = SAMPLED_VARIABLE_SHELL_MEASUREMENT_SCHEDULE_PACKET_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _packet_schedule_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _SCHEDULE_PACKET_ISSUER_TOKEN:
            raise PermissionError("measurement-schedule packets are source-issued only")
        for name in (
            "schedule_source_hash",
            "suffix_registration_hash",
            "generation_frame_hash",
            "discovery_stream_hash",
            "capture_policy_identity_hash",
            "suffix_packet_hash",
            "source_capture_identity_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        ordinal = _nonnegative_int(self.packet_ordinal, name="schedule packet ordinal")
        before = tuple(
            _nonnegative_int(value, name="schedule before count")
            for value in self.before_suffix_stratum_counts
        )
        after = tuple(
            _nonnegative_int(value, name="schedule after count")
            for value in self.after_suffix_stratum_counts
        )
        if len(before) != 2 or len(after) != 2:
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.SCHEDULE_MISMATCH,
                "schedule counts must contain target then complement",
            )
        axes = (
            _possibly_empty_identity_axis(
                self.registered_context_ids,
                name="registered_context_ids",
            ),
            _packet_stratum_axis(
                self.registered_stratum_ids,
                name="registered_stratum_ids",
            ),
            _possibly_empty_identity_axis(
                self.registered_draw_ids,
                name="registered_draw_ids",
            ),
            _possibly_empty_identity_axis(
                self.registered_measurement_ids,
                name="registered_measurement_ids",
            ),
            _possibly_empty_identity_axis(
                self.registered_capture_ids,
                name="registered_capture_ids",
            ),
            _possibly_empty_identity_axis(
                self.registered_source_packet_hashes,
                name="registered_source_packet_hashes",
            ),
        )
        count = len(axes[0])
        if (
            count <= 0
            or any(len(axis) != count for axis in axes)
            or axes[1]
            != ("target",) * (after[0] - before[0])
            + ("complement",) * (after[1] - before[1])
            or count != sum(after[index] - before[index] for index in (0, 1))
            or any(after[index] < before[index] for index in (0, 1))
        ):
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.SCHEDULE_MISMATCH,
                "schedule packet axes do not close one target-then-complement slice",
            )
        required = (
            self.request_id_is_not_a_sampling_coordinate,
            self.outcome_payload_present,
            self.scoring_or_decision_authority,
        )
        if (
            required != (True, False, False)
            or self.schema
            != SAMPLED_VARIABLE_SHELL_MEASUREMENT_SCHEDULE_PACKET_SCHEMA
        ):
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.AUTHORITY_VIOLATION,
                "measurement schedule packet crossed its source-only boundary",
            )
        object.__setattr__(self, "packet_ordinal", ordinal)
        object.__setattr__(self, "before_suffix_stratum_counts", before)
        object.__setattr__(self, "after_suffix_stratum_counts", after)
        for name, axis in zip(
            (
                "registered_context_ids",
                "registered_stratum_ids",
                "registered_draw_ids",
                "registered_measurement_ids",
                "registered_capture_ids",
                "registered_source_packet_hashes",
            ),
            axes,
            strict=True,
        ):
            object.__setattr__(self, name, axis)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(
            self._payload(),
            prefix=_PREFIX + "measurement-schedule-packet-",
        )
        if self._packet_schedule_hash is not None and observed != self._packet_schedule_hash:
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.SCHEDULE_MISMATCH,
                "measurement schedule packet mutated after issuance",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.SCHEDULE_MISMATCH,
                "measurement schedule packet hash differs",
            )
        object.__setattr__(self, "_packet_schedule_hash", observed)
        return observed

    @property
    def packet_schedule_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1SampledVariableShellMeasurementSchedule:
    """Immutable append-order census identities fixed before requests exist."""

    generation_frame_hash: str
    discovery_stream_hash: str
    suffix_registration_hash: str
    capture_policy_identity_hash: str
    schedule_source_hash: str
    transport_packet_size: int
    population_count: int
    packets: tuple[V1SampledVariableShellMeasurementSchedulePacket, ...]
    population_context_ids: tuple[str, ...]
    population_draw_ids: tuple[str, ...]
    population_measurement_ids: tuple[str, ...]
    population_capture_ids: tuple[str, ...]
    population_source_packet_hashes: tuple[str, ...]
    population_context_axis_hash: str
    population_draw_axis_hash: str
    population_measurement_axis_hash: str
    population_capture_axis_hash: str
    population_source_packet_axis_hash: str
    request_id_is_provenance_only: bool = True
    append_order_frozen_before_measurement_requests: bool = True
    full_suffix_census_axis_owned: bool = True
    outcome_payload_present: bool = False
    scoring_authority: bool = False
    statistical_stopping_authority: bool = False
    action_or_commit_authority: bool = False
    schema: str = SAMPLED_VARIABLE_SHELL_MEASUREMENT_SCHEDULE_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _schedule_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _SCHEDULE_ISSUER_TOKEN:
            raise PermissionError("measurement schedules are source-issued only")
        for name in (
            "generation_frame_hash",
            "discovery_stream_hash",
            "suffix_registration_hash",
            "capture_policy_identity_hash",
            "schedule_source_hash",
            "population_context_axis_hash",
            "population_draw_axis_hash",
            "population_measurement_axis_hash",
            "population_capture_axis_hash",
            "population_source_packet_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        packet_size = _nonnegative_int(
            self.transport_packet_size,
            name="measurement schedule transport packet size",
        )
        population_count = _nonnegative_int(
            self.population_count,
            name="measurement schedule population count",
        )
        packets = tuple(self.packets)
        if (
            packet_size <= 0
            or population_count <= 0
            or not packets
            or any(
                type(packet) is not V1SampledVariableShellMeasurementSchedulePacket
                for packet in packets
            )
            or tuple(packet.packet_ordinal for packet in packets)
            != tuple(range(len(packets)))
        ):
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.SCHEDULE_MISMATCH,
                "measurement schedule lost a complete ordered packet axis",
            )
        axes = tuple(
            _identity_axis(getattr(self, name), name=name)
            for name in (
                "population_context_ids",
                "population_draw_ids",
                "population_measurement_ids",
                "population_capture_ids",
                "population_source_packet_hashes",
            )
        )
        flattened = tuple(
            tuple(
                value
                for packet in packets
                for value in getattr(packet, packet_field)
            )
            for packet_field in (
                "registered_context_ids",
                "registered_draw_ids",
                "registered_measurement_ids",
                "registered_capture_ids",
                "registered_source_packet_hashes",
            )
        )
        expected_hashes = tuple(
            v1_variable_shell_population_axis_hash(role, axis)
            for role, axis in zip(
                ("context", "draw", "measurement", "capture", "source_packet"),
                axes,
                strict=True,
            )
        )
        observed_hashes = (
            self.population_context_axis_hash,
            self.population_draw_axis_hash,
            self.population_measurement_axis_hash,
            self.population_capture_axis_hash,
            self.population_source_packet_axis_hash,
        )
        if (
            any(packet.schedule_source_hash != self.schedule_source_hash for packet in packets)
            or any(
                packet.capture_policy_identity_hash
                != self.capture_policy_identity_hash
                for packet in packets
            )
            or any(packet.suffix_registration_hash != self.suffix_registration_hash for packet in packets)
            or any(packet.generation_frame_hash != self.generation_frame_hash for packet in packets)
            or any(packet.discovery_stream_hash != self.discovery_stream_hash for packet in packets)
            or axes != flattened
            or any(len(axis) != population_count for axis in axes)
            or observed_hashes != expected_hashes
        ):
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.SCHEDULE_MISMATCH,
                "measurement schedule does not own its exact append-order census axes",
            )
        flags = (
            self.request_id_is_provenance_only,
            self.append_order_frozen_before_measurement_requests,
            self.full_suffix_census_axis_owned,
            self.outcome_payload_present,
            self.scoring_authority,
            self.statistical_stopping_authority,
            self.action_or_commit_authority,
        )
        if (
            flags != (True, True, True, False, False, False, False)
            or self.schema != SAMPLED_VARIABLE_SHELL_MEASUREMENT_SCHEDULE_SCHEMA
        ):
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.AUTHORITY_VIOLATION,
                "measurement schedule crossed its pre-outcome identity boundary",
            )
        object.__setattr__(self, "transport_packet_size", packet_size)
        object.__setattr__(self, "population_count", population_count)
        object.__setattr__(self, "packets", packets)
        for name, axis in zip(
            (
                "population_context_ids",
                "population_draw_ids",
                "population_measurement_ids",
                "population_capture_ids",
                "population_source_packet_hashes",
            ),
            axes,
            strict=True,
        ):
            object.__setattr__(self, name, axis)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for packet in self.packets:
            packet.assert_integrity()
        observed = content_hash(
            self._payload(),
            prefix=_PREFIX + "measurement-schedule-",
        )
        if self._schedule_hash is not None and observed != self._schedule_hash:
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.SCHEDULE_MISMATCH,
                "measurement schedule mutated after issuance",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.SCHEDULE_MISMATCH,
                "measurement schedule hash differs",
            )
        object.__setattr__(self, "_schedule_hash", observed)
        return observed

    @property
    def schedule_hash(self) -> str:
        return self.assert_integrity()

    def packet_for(
        self,
        packet: V1SampledVariableShellDW2SuffixPacket,
    ) -> V1SampledVariableShellMeasurementSchedulePacket:
        self.assert_integrity()
        if type(packet) is not V1SampledVariableShellDW2SuffixPacket:
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.INVALID_TYPE,
                "measurement schedule requires the exact suffix packet",
            )
        packet.assert_integrity()
        if packet.packet_ordinal >= len(self.packets):
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.SCHEDULE_MISMATCH,
                "suffix packet ordinal is outside the frozen measurement schedule",
            )
        scheduled = self.packets[packet.packet_ordinal]
        if (
            scheduled.suffix_packet_hash != packet.receipt_hash
            or scheduled.before_suffix_stratum_counts
            != packet.before_suffix_stratum_counts
            or scheduled.after_suffix_stratum_counts
            != packet.after_suffix_stratum_counts
            or scheduled.registered_context_ids
            != packet.target_context_ids + packet.complement_context_ids
        ):
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.SCHEDULE_MISMATCH,
                "suffix packet differs from its source-issued measurement schedule",
            )
        return scheduled

    def assert_parent_graph(
        self,
        *,
        frame: V1AlexNetFQCausalGenerationFrameReceipt,
        discovery_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt,
        suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    ) -> str:
        if (
            type(frame) is not V1AlexNetFQCausalGenerationFrameReceipt
            or type(discovery_stream)
            is not V1AlexNetFQCausalRoleSamplingStreamReceipt
            or type(suffix_registration)
            is not V1SampledVariableShellDW2SuffixRegistration
        ):
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.INVALID_TYPE,
                "measurement schedule parent graph requires exact source receipts",
            )
        suffix_registration.assert_integrity()
        suffix_registration.assert_parent_graph(
            frame=frame,
            discovery_stream=discovery_stream,
        )
        if (
            self.generation_frame_hash != frame.receipt_hash
            or self.discovery_stream_hash != discovery_stream.receipt_hash
            or self.suffix_registration_hash != suffix_registration.receipt_hash
            or self.transport_packet_size
            != suffix_registration.transport_packet_size
        ):
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.SCHEDULE_MISMATCH,
                "measurement schedule belongs to another suffix parent graph",
            )
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1SampledVariableShellMeasurementSourceLineage:
    """Branch-independent custody for one full-suffix measurement campaign.

    The lineage deliberately excludes compact-support receipts, CUDA source
    designs, append owners, and preferred actions.  Those are branch-local
    lowering coordinates and may not influence the shared draw/capture axis or
    the logical measurement request.
    """

    source_graph_hash: str
    capture_policy_identity_hash: str
    measurement_schedule_hash: str
    suffix_registration_hash: str
    residual_population_measure_design_hash: str
    shell_union_receipt_hash: str
    campaign_binding_hash: str
    calibration_hash: str
    scorer_source_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    witness_stream_source_hash: str
    same_capture_vjp_source_hash: str
    corrected_evidence_stream_source_hash: str
    literal_j_stream_source_hash: str
    alpha_ledger_id: str
    alpha_coordinate_id: str
    alpha_lineage_hash: str
    branch_support_receipt_present: bool = False
    preferred_action_present: bool = False
    provider_outcomes_seen: bool = False
    acquisition_execution_authority: bool = False
    alpha_spend_authority: bool = False
    trajectory_nonrecycling_alpha_closed: bool = False
    action_or_commit_authority: bool = False
    schema: str = SAMPLED_VARIABLE_SHELL_MEASUREMENT_SOURCE_LINEAGE_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _lineage_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _MEASUREMENT_SOURCE_ISSUER_TOKEN:
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.INVALID_TYPE,
                "measurement-source lineages are source-issued only",
            )
        for name in (
            "source_graph_hash",
            "capture_policy_identity_hash",
            "measurement_schedule_hash",
            "suffix_registration_hash",
            "residual_population_measure_design_hash",
            "shell_union_receipt_hash",
            "campaign_binding_hash",
            "calibration_hash",
            "scorer_source_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "witness_stream_source_hash",
            "same_capture_vjp_source_hash",
            "corrected_evidence_stream_source_hash",
            "literal_j_stream_source_hash",
            "alpha_ledger_id",
            "alpha_coordinate_id",
            "alpha_lineage_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        closed = (
            self.branch_support_receipt_present,
            self.preferred_action_present,
            self.provider_outcomes_seen,
            self.acquisition_execution_authority,
            self.alpha_spend_authority,
            self.trajectory_nonrecycling_alpha_closed,
            self.action_or_commit_authority,
        )
        if (
            any(type(value) is not bool or value is not False for value in closed)
            or self.schema
            != SAMPLED_VARIABLE_SHELL_MEASUREMENT_SOURCE_LINEAGE_SCHEMA
        ):
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.AUTHORITY_VIOLATION,
                "measurement-source lineage crossed its shared pre-acquisition boundary",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(
            self._payload(), prefix=_PREFIX + "measurement-source-lineage-"
        )
        if (
            (self._lineage_hash is not None and observed != self._lineage_hash)
            or (expected_hash is not None and observed != expected_hash)
        ):
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.STALE_IDENTITY,
                "measurement-source lineage changed after issuance",
            )
        object.__setattr__(self, "_lineage_hash", observed)
        return observed

    @property
    def lineage_hash(self) -> str:
        return self.assert_integrity()


def _issue_v1_sampled_variable_shell_measurement_source_lineage(
    **values: object,
) -> V1SampledVariableShellMeasurementSourceLineage:
    """Internal issuer after an upstream live-source graph is validated."""

    return V1SampledVariableShellMeasurementSourceLineage(
        **values,
        _issuance_token=_MEASUREMENT_SOURCE_ISSUER_TOKEN,
    )


def _issue_v1_sampled_variable_shell_measurement_schedule(
    *,
    frame: V1AlexNetFQCausalGenerationFrameReceipt,
    discovery_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    capture_policy_identity_hash: str,
) -> V1SampledVariableShellMeasurementSchedule:
    """Internal issuer used only after an upstream live-source policy closes."""

    if (
        type(frame) is not V1AlexNetFQCausalGenerationFrameReceipt
        or type(discovery_stream) is not V1AlexNetFQCausalRoleSamplingStreamReceipt
        or type(suffix_registration)
        is not V1SampledVariableShellDW2SuffixRegistration
    ):
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.INVALID_TYPE,
            "measurement schedule requires exact causal suffix parents",
        )
    suffix_registration.assert_integrity()
    suffix_registration.assert_parent_graph(
        frame=frame,
        discovery_stream=discovery_stream,
    )
    capture_policy = _identity(
        capture_policy_identity_hash,
        name="upstream capture-policy identity hash",
    )
    d1 = suffix_registration.consumed_d1_stratum_counts
    target = discovery_stream.target_assignment_receipt.generated_order[d1[0] :]
    complement = discovery_stream.complement_assignment_receipt.generated_order[d1[1] :]
    capacities = (len(target), len(complement))
    packet_size = suffix_registration.transport_packet_size
    schedule_source = content_hash(
        {
            "schema": "sampled-variable-shell-measurement-schedule-source-v1",
            "generation_frame_hash": frame.receipt_hash,
            "discovery_stream_hash": discovery_stream.receipt_hash,
            "suffix_registration_hash": suffix_registration.receipt_hash,
            "target_suffix_axis_hash": suffix_registration.target_suffix_axis_hash,
            "complement_suffix_axis_hash": (
                suffix_registration.complement_suffix_axis_hash
            ),
            "suffix_pair_axis_hash": suffix_registration.suffix_pair_axis_hash,
            "transport_packet_size": packet_size,
            "capture_policy_identity_hash": capture_policy,
            "request_id_is_not_a_sampling_coordinate": True,
        },
        prefix=_PREFIX + "measurement-schedule-source-",
    )
    packets: list[V1SampledVariableShellMeasurementSchedulePacket] = []
    consumed = [0, 0]
    while tuple(consumed) != capacities:
        ordinal = len(packets)
        before = tuple(consumed)
        after = (
            min(capacities[0], consumed[0] + packet_size),
            min(capacities[1], consumed[1] + packet_size),
        )
        target_packet = target[consumed[0] : after[0]]
        complement_packet = complement[consumed[1] : after[1]]
        suffix_packet = V1SampledVariableShellDW2SuffixPacket(
            suffix_registration_hash=suffix_registration.receipt_hash,
            generation_frame_hash=frame.receipt_hash,
            discovery_stream_hash=discovery_stream.receipt_hash,
            packet_ordinal=ordinal,
            before_suffix_stratum_counts=before,  # type: ignore[arg-type]
            after_suffix_stratum_counts=after,
            target_context_ids=target_packet,
            complement_context_ids=complement_packet,
            target_packet_axis_hash=_suffix_axis_hash(
                f"D-W2-target-packet-{ordinal}", target_packet
            ),
            complement_packet_axis_hash=_suffix_axis_hash(
                f"D-W2-complement-packet-{ordinal}", complement_packet
            ),
            packet_pair_axis_hash=_suffix_pair_axis_hash(
                role=f"D-W2-packet-{ordinal}",
                target=target_packet,
                complement=complement_packet,
            ),
            transport_packet_size=packet_size,
        )
        contexts = target_packet + complement_packet
        strata = ("target",) * len(target_packet) + ("complement",) * len(
            complement_packet
        )
        draw_ids: list[str] = []
        measurement_ids: list[str] = []
        capture_ids: list[str] = []
        source_packet_hashes: list[str] = []
        for item_ordinal, (stratum, context_id) in enumerate(
            zip(strata, contexts, strict=True)
        ):
            common = {
                "schema": "sampled-variable-shell-scheduled-population-item-v1",
                "schedule_source_hash": schedule_source,
                "suffix_registration_hash": suffix_registration.receipt_hash,
                "suffix_packet_hash": suffix_packet.receipt_hash,
                "capture_policy_identity_hash": capture_policy,
                "packet_ordinal": ordinal,
                "item_ordinal": item_ordinal,
                "stratum_id": stratum,
                "context_id": context_id,
            }
            draw = content_hash(common | {"role": "draw"}, prefix=_PREFIX + "draw-")
            measurement = content_hash(
                common | {"role": "measurement", "draw_id": draw},
                prefix=_PREFIX + "measurement-",
            )
            capture = content_hash(
                common
                | {
                    "role": "capture",
                    "draw_id": draw,
                    "measurement_id": measurement,
                },
                prefix=_PREFIX + "capture-",
            )
            source_packet = content_hash(
                common
                | {
                    "role": "source-packet",
                    "draw_id": draw,
                    "measurement_id": measurement,
                    "capture_id": capture,
                },
                prefix=_PREFIX + "source-packet-",
            )
            draw_ids.append(draw)
            measurement_ids.append(measurement)
            capture_ids.append(capture)
            source_packet_hashes.append(source_packet)
        source_capture = content_hash(
            {
                "schema": "sampled-variable-shell-logical-source-capture-plan-v1",
                "schedule_source_hash": schedule_source,
                "suffix_packet_hash": suffix_packet.receipt_hash,
                "capture_policy_identity_hash": capture_policy,
                "packet_ordinal": ordinal,
                "registered_context_ids": contexts,
                "registered_capture_ids": tuple(capture_ids),
                "one_logical_request_provenance_record": True,
                "physical_target_and_complement_calls_remain_distinct": True,
            },
            prefix=_PREFIX + "logical-source-capture-",
        )
        packets.append(
            V1SampledVariableShellMeasurementSchedulePacket(
                schedule_source_hash=schedule_source,
                suffix_registration_hash=suffix_registration.receipt_hash,
                generation_frame_hash=frame.receipt_hash,
                discovery_stream_hash=discovery_stream.receipt_hash,
                capture_policy_identity_hash=capture_policy,
                packet_ordinal=ordinal,
                suffix_packet_hash=suffix_packet.receipt_hash,
                before_suffix_stratum_counts=before,  # type: ignore[arg-type]
                after_suffix_stratum_counts=after,
                registered_context_ids=contexts,
                registered_stratum_ids=strata,
                registered_draw_ids=tuple(draw_ids),
                registered_measurement_ids=tuple(measurement_ids),
                registered_capture_ids=tuple(capture_ids),
                registered_source_packet_hashes=tuple(source_packet_hashes),
                source_capture_identity_hash=source_capture,
                _issuance_token=_SCHEDULE_PACKET_ISSUER_TOKEN,
            )
        )
        consumed[:] = after
    population_axes = tuple(
        tuple(value for packet in packets for value in getattr(packet, field_name))
        for field_name in (
            "registered_context_ids",
            "registered_draw_ids",
            "registered_measurement_ids",
            "registered_capture_ids",
            "registered_source_packet_hashes",
        )
    )
    population_hashes = tuple(
        v1_variable_shell_population_axis_hash(role, axis)
        for role, axis in zip(
            ("context", "draw", "measurement", "capture", "source_packet"),
            population_axes,
            strict=True,
        )
    )
    return V1SampledVariableShellMeasurementSchedule(
        generation_frame_hash=frame.receipt_hash,
        discovery_stream_hash=discovery_stream.receipt_hash,
        suffix_registration_hash=suffix_registration.receipt_hash,
        capture_policy_identity_hash=capture_policy,
        schedule_source_hash=schedule_source,
        transport_packet_size=packet_size,
        population_count=sum(capacities),
        packets=tuple(packets),
        population_context_ids=population_axes[0],
        population_draw_ids=population_axes[1],
        population_measurement_ids=population_axes[2],
        population_capture_ids=population_axes[3],
        population_source_packet_hashes=population_axes[4],
        population_context_axis_hash=population_hashes[0],
        population_draw_axis_hash=population_hashes[1],
        population_measurement_axis_hash=population_hashes[2],
        population_capture_axis_hash=population_hashes[3],
        population_source_packet_axis_hash=population_hashes[4],
        _issuance_token=_SCHEDULE_ISSUER_TOKEN,
    )


def v1_sampled_variable_shell_stratum_measure_source_hash(
    registration: V1SampledVariableShellDW2SuffixRegistration,
) -> str:
    """Return the source-issued exact ``D1`` plus residual measure identity."""

    if type(registration) is not V1SampledVariableShellDW2SuffixRegistration:
        raise TypeError(
            "stratum-measure identity requires the exact suffix registration"
        )
    return register_v1_sampled_residual_population_measure(
        suffix_registration=registration
    ).design_hash


def _validate_measure_design(
    *,
    design: V1SampledResidualPopulationMeasureDesign,
    registration: V1SampledVariableShellDW2SuffixRegistration,
) -> str:
    if type(design) is not V1SampledResidualPopulationMeasureDesign:
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.INVALID_TYPE,
            "packet binder requires the source-issued residual-population measure design",
        )
    try:
        observed = design.assert_integrity()
    except V1SampledResidualPopulationMeasureRefusal as error:
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.STALE_IDENTITY,
            f"residual-population measure design is stale: {error}",
        )
    expected = v1_sampled_variable_shell_stratum_measure_source_hash(registration)
    if (
        observed != expected
        or design.suffix_registration_hash != registration.receipt_hash
    ):
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.STALE_IDENTITY,
            "residual-population measure design does not own this D_W2 suffix",
        )
    return observed


def _issue_measure_weights(
    *,
    design: V1SampledResidualPopulationMeasureDesign,
    registration: V1SampledVariableShellDW2SuffixRegistration,
    packet: V1SampledVariableShellDW2SuffixPacket,
) -> V1SampledResidualPopulationWeightReceipt:
    _validate_measure_design(design=design, registration=registration)
    try:
        return issue_v1_sampled_residual_population_weights_for_packet(
            design=design,
            suffix_registration=registration,
            packet=packet,
        )
    except V1SampledResidualPopulationMeasureRefusal as error:
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.WEIGHT_MISMATCH,
            f"source-issued full-population packet weights refused: {error}",
        )


def _residual_population_weight_enclosure_hash(
    receipt: V1SampledResidualPopulationWeightReceipt,
) -> str:
    """Bind the exact rational weights to their canonical binary64 enclosures."""

    receipt.assert_integrity()
    return content_hash(
        {
            "schema": "sampled-residual-population-weight-enclosures-v1",
            "weight_receipt_hash": receipt.receipt_hash,
            "cumulative_dw2_stratum_counts": receipt.cumulative_dw2_stratum_counts,
            "d1_context_weight_fraction": receipt.d1_context_weight_fraction,
            "dw2_context_weight_fractions": receipt.dw2_context_weight_fractions,
            "residual_conditional_context_weight_fractions": (
                receipt.residual_conditional_context_weight_fractions
            ),
            "d1_context_weight_binary64_enclosure": (
                receipt.d1_context_weight_binary64_enclosure
            ),
            "dw2_context_weight_binary64_enclosures": (
                receipt.dw2_context_weight_binary64_enclosures
            ),
            "residual_conditional_weight_binary64_enclosures": (
                receipt.residual_conditional_weight_binary64_enclosures
            ),
        },
        prefix=_PREFIX + "population-weight-enclosures-",
    )


def _append_logical_stratum_mapping_hash(
    *,
    append_stratum_ids: Sequence[str],
    logical_stratum_ids: Sequence[str],
) -> str:
    if len(append_stratum_ids) != len(logical_stratum_ids):
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.ORDER_MISMATCH,
            "append and logical stratum axes do not share one length",
        )
    return content_hash(
        {
            "schema": "sampled-variable-shell-append-to-logical-stratum-map-v1",
            "append_stratum_ids": tuple(append_stratum_ids),
            "logical_stratum_ids": tuple(logical_stratum_ids),
            "logical_stratum_domain": _STRATA,
        },
        prefix=_PREFIX + "append-logical-stratum-map-",
    )


def _packet_counts(
    packet: V1SampledVariableShellDW2SuffixPacket,
) -> tuple[int, int]:
    return (len(packet.target_context_ids), len(packet.complement_context_ids))


def v1_sampled_variable_shell_common_measure_unit_hash(
    *,
    registration: V1SampledVariableShellDW2SuffixRegistration,
) -> str:
    """Bind the packet-invariant exact residual-measure lowering rule.

    Each packet receives its own smallest common integer denominator after the
    source-issued residual stratum masses are divided over that packet's
    target and complement draws.  The rule, rather than any one packet's
    denominator, is invariant.  This permits an unequal terminal tail while
    preserving identical stratum masses across the typed source segments.
    """

    if type(registration) is not V1SampledVariableShellDW2SuffixRegistration:
        raise TypeError(
            "common measure identity requires the exact suffix registration"
        )
    design = register_v1_sampled_residual_population_measure(
        suffix_registration=registration
    )
    return content_hash(
        {
            "schema": "sampled-variable-shell-segmented-measure-unit-rule-v5",
            "residual_population_measure_design_hash": design.design_hash,
            "packet_local_integer_denominator_is_derived_exactly": True,
            "arbitrary_positive_packet_stratum_counts_supported": True,
            "zero_count_unequal_terminal_tail_supported": True,
            "packet_stratum_masses_equal_residual_population_masses": True,
            "packet_local_transport_normalization_only": True,
            "source_issued_full_population_measure_is_separate": True,
            "exact_D1_compact_source_join_not_present": True,
            "stratum_resolved_segment_composition_required": True,
        },
        prefix=_PREFIX + "common-measure-unit-",
    )


def v1_sampled_variable_shell_planned_prefix_hash(
    *,
    registration: V1SampledVariableShellDW2SuffixRegistration,
    packet: V1SampledVariableShellDW2SuffixPacket,
) -> str:
    """Identity of the predictable suffix prefix after ``packet``.

    No terminal freeze receipt appears in this hash.  The sealed suffix axes
    and post-packet stratum counts uniquely select the prefix before outcomes.
    """

    if type(registration) is not V1SampledVariableShellDW2SuffixRegistration:
        raise TypeError("planned prefix requires the exact suffix registration")
    if type(packet) is not V1SampledVariableShellDW2SuffixPacket:
        raise TypeError("planned prefix requires the exact suffix packet")
    registration.assert_integrity()
    packet.assert_integrity()
    if (
        packet.suffix_registration_hash != registration.receipt_hash
        or packet.generation_frame_hash != registration.generation_frame_hash
        or packet.discovery_stream_hash != registration.discovery_stream_hash
    ):
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.STALE_IDENTITY,
            "packet does not descend from the registered suffix",
        )
    return content_hash(
        {
            "schema": "sampled-variable-shell-predictable-D-W2-prefix-v1",
            "suffix_registration_hash": registration.receipt_hash,
            "generation_frame_hash": registration.generation_frame_hash,
            "discovery_stream_hash": registration.discovery_stream_hash,
            "target_suffix_axis_hash": registration.target_suffix_axis_hash,
            "complement_suffix_axis_hash": registration.complement_suffix_axis_hash,
            "packet_ordinal": packet.packet_ordinal,
            "packet_hash": packet.receipt_hash,
            "after_suffix_stratum_counts": packet.after_suffix_stratum_counts,
            "terminal_outcome_dependent_freeze_is_not_an_ancestor": True,
        },
        prefix=_PREFIX + "planned-prefix-",
    )


def v1_sampled_variable_shell_packet_request_state_hash(
    *,
    registration: V1SampledVariableShellDW2SuffixRegistration,
    packet: V1SampledVariableShellDW2SuffixPacket,
    measurement_schedule: V1SampledVariableShellMeasurementSchedule,
    measurement_source_lineage: V1SampledVariableShellMeasurementSourceLineage,
    residual_measure_design: V1SampledResidualPopulationMeasureDesign,
    shell_union: V1SignedWorkFixedPrimalShellUnionReceipt,
) -> str:
    """Return the complete pre-outcome state identity for the provider request."""

    weight_receipt = _issue_measure_weights(
        design=residual_measure_design,
        registration=registration,
        packet=packet,
    )
    scheduled = measurement_schedule.packet_for(packet)
    measurement_source_lineage.assert_integrity()
    return content_hash(
        {
            "schema": SAMPLED_VARIABLE_SHELL_PACKET_REQUEST_PAYLOAD_SCHEMA,
            "suffix_registration_hash": registration.receipt_hash,
            "packet_hash": packet.receipt_hash,
            "planned_prefix_hash": v1_sampled_variable_shell_planned_prefix_hash(
                registration=registration,
                packet=packet,
            ),
            "shell_union_receipt_hash": shell_union.receipt_hash,
            "measurement_source_lineage_hash": (
                measurement_source_lineage.lineage_hash
            ),
            "measurement_schedule_hash": measurement_schedule.schedule_hash,
            "measurement_schedule_packet_hash": scheduled.packet_schedule_hash,
            "capture_policy_identity_hash": (
                measurement_schedule.capture_policy_identity_hash
            ),
            "residual_population_measure_design_hash": (
                residual_measure_design.design_hash
            ),
            "residual_population_weight_receipt_hash": weight_receipt.receipt_hash,
            "current_mask_hash": measurement_source_lineage.current_mask_hash,
            "proposal_generation": measurement_source_lineage.proposal_generation,
            "raw_target_logit_coordinate": _RAW_TARGET_LOGIT_COORDINATE,
        },
        prefix=_PREFIX + "request-state-",
    )


def v1_sampled_variable_shell_packet_request_payload(
    *,
    registration: V1SampledVariableShellDW2SuffixRegistration,
    packet: V1SampledVariableShellDW2SuffixPacket,
    residual_measure_design: V1SampledResidualPopulationMeasureDesign,
    shell_union: V1SignedWorkFixedPrimalShellUnionReceipt,
    measurement_schedule: V1SampledVariableShellMeasurementSchedule,
    measurement_source_lineage: V1SampledVariableShellMeasurementSourceLineage,
) -> dict[str, object]:
    """Build the exact allow-listed request payload for one packet."""

    scheduled = measurement_schedule.packet_for(packet)
    weight_receipt = _issue_measure_weights(
        design=residual_measure_design,
        registration=registration,
        packet=packet,
    )
    measurement_source_lineage.assert_integrity()
    return {
        "schema": SAMPLED_VARIABLE_SHELL_PACKET_REQUEST_PAYLOAD_SCHEMA,
        "suffix_registration_hash": registration.receipt_hash,
        "packet_hash": packet.receipt_hash,
        "registered_d_w2_planned_prefix_hash": (
            v1_sampled_variable_shell_planned_prefix_hash(
                registration=registration,
                packet=packet,
            )
        ),
        "shell_union_receipt_hash": shell_union.receipt_hash,
        "measurement_source_lineage_hash": measurement_source_lineage.lineage_hash,
        "measurement_schedule_hash": measurement_schedule.schedule_hash,
        "measurement_schedule_source_hash": measurement_schedule.schedule_source_hash,
        "measurement_schedule_packet_hash": scheduled.packet_schedule_hash,
        "capture_policy_identity_hash": (
            measurement_schedule.capture_policy_identity_hash
        ),
        "residual_population_measure_design_hash": (
            residual_measure_design.design_hash
        ),
        "residual_population_weight_receipt_hash": weight_receipt.receipt_hash,
        "residual_population_weight_binary64_enclosure_hash": (
            _residual_population_weight_enclosure_hash(weight_receipt)
        ),
        "cumulative_dw2_stratum_counts": (weight_receipt.cumulative_dw2_stratum_counts),
        "d1_context_weight_fraction": weight_receipt.d1_context_weight_fraction,
        "cumulative_dw2_context_weight_fractions": (
            weight_receipt.dw2_context_weight_fractions
        ),
        "d1_context_weight_binary64_enclosure": (
            weight_receipt.d1_context_weight_binary64_enclosure
        ),
        "cumulative_dw2_context_weight_binary64_enclosures": (
            weight_receipt.dw2_context_weight_binary64_enclosures
        ),
        "residual_conditional_weight_binary64_enclosures": (
            weight_receipt.residual_conditional_weight_binary64_enclosures
        ),
        "population_context_axis_hash": (
            measurement_schedule.population_context_axis_hash
        ),
        "population_draw_axis_hash": measurement_schedule.population_draw_axis_hash,
        "population_measurement_axis_hash": (
            measurement_schedule.population_measurement_axis_hash
        ),
        "population_capture_axis_hash": (
            measurement_schedule.population_capture_axis_hash
        ),
        "population_source_packet_axis_hash": (
            measurement_schedule.population_source_packet_axis_hash
        ),
        "source_capture_identity_hash": scheduled.source_capture_identity_hash,
        "witness_stream_source_hash": (
            measurement_source_lineage.witness_stream_source_hash
        ),
        "same_capture_vjp_source_hash": (
            measurement_source_lineage.same_capture_vjp_source_hash
        ),
        "corrected_evidence_stream_source_hash": (
            measurement_source_lineage.corrected_evidence_stream_source_hash
        ),
        "literal_j_stream_source_hash": (
            measurement_source_lineage.literal_j_stream_source_hash
        ),
        "stratum_measure_source_hash": (
            v1_sampled_variable_shell_stratum_measure_source_hash(registration)
        ),
        "common_integer_measure_unit_hash": (
            v1_sampled_variable_shell_common_measure_unit_hash(
                registration=registration,
            )
        ),
        "raw_target_logit_coordinate": _RAW_TARGET_LOGIT_COORDINATE,
        "same_capture_corrected_and_literal_j_required": True,
        "provider_or_outcome_receipts_forbidden": True,
        "source_registration_is_conditional_residual_transport_only": True,
        "arbitrary_positive_packet_stratum_counts_supported": True,
        "stratum_resolved_segment_registration_required": True,
        "exact_d1_compact_source_contribution_joined": False,
        "stratum_resolved_full_population_compositor_joined": False,
        "full_population_decision_witness_authority": False,
        "measurement_request_id_is_sampling_coordinate": False,
    }


def _derive_packet_axes(
    *,
    packet: V1SampledVariableShellDW2SuffixPacket,
    measurement_schedule: V1SampledVariableShellMeasurementSchedule,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    scheduled = measurement_schedule.packet_for(packet)
    return (
        scheduled.registered_context_ids,
        scheduled.registered_draw_ids,
        scheduled.registered_measurement_ids,
        scheduled.registered_capture_ids,
        scheduled.registered_source_packet_hashes,
    )


def _source_local_weights_and_quanta(
    *,
    weight_receipt: V1SampledResidualPopulationWeightReceipt,
    packet: V1SampledVariableShellDW2SuffixPacket,
) -> tuple[tuple[Fraction, ...], tuple[int, ...], int]:
    """Lower the exact residual measure into a packet-local transport unit.

    ``V1SignedWorkVariableShellSourceRegistration`` requires its local weights
    to sum to one.  These weights are therefore a compact-source transport
    representation, not the decision estimand.  The source-issued cumulative
    full-population weights remain authoritative in ``weight_receipt``.
    """

    weight_receipt.assert_integrity()
    target_count, complement_count = _packet_counts(packet)
    if target_count + complement_count <= 0:
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.WEIGHT_MISMATCH,
            "a compact residual source packet cannot be empty",
        )
    target_residual, complement_residual = weight_receipt.residual_stratum_counts
    residual_total = target_residual + complement_residual
    residual_masses = tuple(
        Fraction(*pair) for pair in weight_receipt.residual_stratum_mass_fractions
    )
    full_population_residual_mass = sum(residual_masses, start=Fraction(0))
    if (
        residual_total <= 0
        or full_population_residual_mass <= 0
        or residual_masses
        != tuple(
            Fraction(
                count,
                weight_receipt.population_stratum_counts[0]
                + weight_receipt.population_stratum_counts[1],
            )
            for count in weight_receipt.residual_stratum_counts
        )
    ):
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.WEIGHT_MISMATCH,
            "source-issued residual masses do not replay their registered counts",
        )
    present_masses = tuple(
        residual_masses[index]
        for index, count in enumerate((target_count, complement_count))
        if count > 0
    )
    present_mass = sum(present_masses, start=Fraction(0))
    if present_mass <= 0:
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.WEIGHT_MISMATCH,
            "a nonempty packet cannot represent a zero-mass residual stratum",
        )
    fractions = tuple(
        weight
        for index, count in enumerate((target_count, complement_count))
        if count > 0
        for weight in (residual_masses[index] / present_mass / count,) * count
    )
    if sum(fractions, start=Fraction(0)) != Fraction(1):
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.WEIGHT_MISMATCH,
            "packet-local residual transport weights do not close unit mass",
        )
    denominator = math.lcm(*(value.denominator for value in fractions))
    quanta = tuple(int(value * denominator) for value in fractions)
    if sum(quanta) != denominator or any(
        Fraction(quantum, denominator) != weight
        for quantum, weight in zip(quanta, fractions, strict=True)
    ):
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.WEIGHT_MISMATCH,
            "integer quanta do not replay packet-local residual transport weights",
        )
    return fractions, quanta, denominator


@dataclass(frozen=True, slots=True)
class V1SampledVariableShellPacketRegistrationReceipt:
    suffix_registration_hash: str
    packet_hash: str
    packet_ordinal: int
    generation_frame_hash: str
    discovery_stream_hash: str
    measurement_schedule_hash: str
    measurement_schedule_source_hash: str
    measurement_schedule_packet_hash: str
    capture_policy_identity_hash: str
    population_context_axis_hash: str
    population_draw_axis_hash: str
    population_measurement_axis_hash: str
    population_capture_axis_hash: str
    population_source_packet_axis_hash: str
    measurement_request_id: str
    measurement_request_state_hash: str
    source_capture_identity_hash: str
    planned_d_w2_prefix_hash: str
    shell_union_receipt_hash: str
    provisional_shell_hash: str
    provisional_action_family_hash: str
    measurement_source_lineage_hash: str
    append_lineage_hash: str
    append_packet_registration_hash: str
    source_registration_hash: str
    registered_context_ids: tuple[str, ...]
    registered_draw_ids: tuple[str, ...]
    registered_measurement_ids: tuple[str, ...]
    registered_capture_ids: tuple[str, ...]
    registered_source_packet_hashes: tuple[str, ...]
    registered_stratum_ids: tuple[str, ...]
    append_registered_stratum_ids: tuple[str, ...]
    packet_stratum_counts: tuple[int, int]
    after_suffix_stratum_counts: tuple[int, int]
    residual_population_measure_design_hash: str
    residual_population_weight_receipt_hash: str
    residual_population_weight_binary64_enclosure_hash: str
    append_logical_stratum_mapping_hash: str
    full_population_stratum_counts: tuple[int, int]
    consumed_d1_stratum_counts: tuple[int, int]
    residual_stratum_capacity_counts: tuple[int, int]
    full_population_count: int
    residual_population_count: int
    d1_context_weight_fraction: tuple[int, int]
    cumulative_dw2_context_weight_fractions: tuple[tuple[int, int], tuple[int, int]]
    d1_context_weight_binary64_enclosure: tuple[float, float, float]
    cumulative_dw2_context_weight_binary64_enclosures: tuple[
        tuple[float, float, float], tuple[float, float, float]
    ]
    residual_conditional_weight_binary64_enclosures: tuple[
        tuple[float, float, float], tuple[float, float, float]
    ]
    residual_stratum_mass_fractions: tuple[tuple[int, int], tuple[int, int]]
    full_population_measure_common_denominator: int
    d1_context_measure_quantum: int
    cumulative_dw2_context_measure_quanta: tuple[int, int]
    full_population_total_measure_quanta: int
    source_local_conditional_residual_weight_fractions: tuple[tuple[int, int], ...]
    source_local_conditional_residual_weight_binary64_enclosures: tuple[
        tuple[float, float, float], ...
    ]
    source_local_context_measure_quanta: tuple[int, ...]
    source_local_common_measure_denominator: int
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    kkt_branch_id: str
    kkt_branch_source_receipt_hash: str
    resource_axis_hash: str
    resource_source_receipt_hash: str
    witness_stream_source_hash: str
    same_capture_vjp_source_hash: str
    corrected_evidence_stream_source_hash: str
    literal_j_stream_source_hash: str
    alpha_ledger_id: str
    alpha_coordinate_id: str
    alpha_lineage_hash: str
    request_id_is_provenance_only: bool = True
    full_suffix_schedule_axes_validated: bool = True
    zero_count_unequal_terminal_tail_supported: bool = True
    arbitrary_positive_packet_stratum_counts_supported: bool = True
    source_local_integer_transport_exact: bool = True
    stratum_resolved_segment_registration_required: bool = True
    append_stratum_ids_are_context_unique: bool = True
    logical_stratum_axis_bound_separately: bool = True
    unequal_count_census_tail_requires_stratum_resolved_composition: bool = True
    source_issued_full_population_measure_bound: bool = True
    source_registration_is_conditional_residual_transport_only: bool = True
    source_local_normalization_not_decision_measure: bool = True
    stratum_resolved_full_population_compositor_required: bool = True
    exact_d1_compact_source_contribution_joined: bool = False
    full_population_decision_witness_authority: bool = False
    target_then_complement_order: bool = True
    request_and_append_reserved_before_outcomes: bool = True
    terminal_freeze_not_an_identity_ancestor: bool = True
    source_issued: bool = True
    provider_or_outcome_receipt_custody: bool = False
    alpha_spend_authority: bool = False
    trajectory_nonrecycling_alpha_closed: bool = False
    scoring_authority: bool = False
    confidence_authority: bool = False
    action_decision_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SAMPLED_VARIABLE_SHELL_PACKET_REGISTRATION_AUTHORITY
    schema: str = SAMPLED_VARIABLE_SHELL_PACKET_REGISTRATION_RECEIPT_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)
    _receipt_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self._issuance_token is not _ISSUER_TOKEN:
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.INVALID_TYPE,
                "packet-registration receipts are source-issued only",
            )
        identity_names = (
            "suffix_registration_hash",
            "packet_hash",
            "generation_frame_hash",
            "discovery_stream_hash",
            "measurement_schedule_hash",
            "measurement_schedule_source_hash",
            "measurement_schedule_packet_hash",
            "capture_policy_identity_hash",
            "population_context_axis_hash",
            "population_draw_axis_hash",
            "population_measurement_axis_hash",
            "population_capture_axis_hash",
            "population_source_packet_axis_hash",
            "measurement_request_id",
            "measurement_request_state_hash",
            "source_capture_identity_hash",
            "planned_d_w2_prefix_hash",
            "shell_union_receipt_hash",
            "provisional_shell_hash",
            "provisional_action_family_hash",
            "measurement_source_lineage_hash",
            "append_lineage_hash",
            "append_packet_registration_hash",
            "source_registration_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "kkt_branch_id",
            "kkt_branch_source_receipt_hash",
            "resource_axis_hash",
            "resource_source_receipt_hash",
            "witness_stream_source_hash",
            "same_capture_vjp_source_hash",
            "corrected_evidence_stream_source_hash",
            "literal_j_stream_source_hash",
            "alpha_ledger_id",
            "alpha_coordinate_id",
            "alpha_lineage_hash",
            "residual_population_weight_binary64_enclosure_hash",
            "append_logical_stratum_mapping_hash",
        )
        for name in identity_names:
            _identity(getattr(self, name), name=name)
        for name in (
            "registered_context_ids",
            "registered_draw_ids",
            "registered_measurement_ids",
            "registered_capture_ids",
            "registered_source_packet_hashes",
        ):
            axis = _identity_axis(getattr(self, name), name=name)
            object.__setattr__(self, name, axis)
        context_count = len(self.registered_context_ids)
        if any(
            len(getattr(self, name)) != context_count
            for name in (
                "registered_draw_ids",
                "registered_measurement_ids",
                "registered_capture_ids",
                "registered_source_packet_hashes",
                "registered_stratum_ids",
                "append_registered_stratum_ids",
                "source_local_conditional_residual_weight_fractions",
                "source_local_conditional_residual_weight_binary64_enclosures",
                "source_local_context_measure_quanta",
            )
        ):
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.ORDER_MISMATCH,
                "packet-registration axes do not share one context length",
            )
        strata = tuple(self.registered_stratum_ids)
        append_strata = _identity_axis(
            self.append_registered_stratum_ids,
            name="append registered stratum IDs",
        )
        object.__setattr__(
            self,
            "append_registered_stratum_ids",
            append_strata,
        )
        packet_counts = tuple(
            _nonnegative_int(value, name="packet stratum count")
            for value in self.packet_stratum_counts
        )
        after_counts = tuple(
            _nonnegative_int(value, name="after-suffix stratum count")
            for value in self.after_suffix_stratum_counts
        )
        if len(packet_counts) != 2 or len(after_counts) != 2:
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.ORDER_MISMATCH,
                "packet count axes must contain target then complement",
            )
        object.__setattr__(self, "packet_stratum_counts", packet_counts)
        object.__setattr__(self, "after_suffix_stratum_counts", after_counts)
        if sum(packet_counts) <= 0:
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.WEIGHT_MISMATCH,
                "a packet-registration receipt cannot describe an empty packet",
            )
        if strata != (
            ("target",) * packet_counts[0] + ("complement",) * packet_counts[1]
        ):
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.ORDER_MISMATCH,
                "registered contexts are not flattened target then complement",
            )
        for name in (
            "residual_population_measure_design_hash",
            "residual_population_weight_receipt_hash",
        ):
            _identity(getattr(self, name), name=name)
        expected_mapping_hash = _append_logical_stratum_mapping_hash(
            append_stratum_ids=append_strata,
            logical_stratum_ids=strata,
        )
        if self.append_logical_stratum_mapping_hash != expected_mapping_hash:
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.STALE_IDENTITY,
                "append-to-logical stratum mapping hash does not replay its axes",
            )
        local_denominator = _nonnegative_int(
            self.source_local_common_measure_denominator,
            name="source-local common measure denominator",
        )
        full_denominator = _nonnegative_int(
            self.full_population_measure_common_denominator,
            name="full-population common measure denominator",
        )
        d1_quantum = _nonnegative_int(
            self.d1_context_measure_quantum,
            name="D1 context measure quantum",
        )
        total_quanta = _nonnegative_int(
            self.full_population_total_measure_quanta,
            name="full-population total measure quanta",
        )
        local_fractions = tuple(
            _fraction(pair, name="source-local residual weight")
            for pair in self.source_local_conditional_residual_weight_fractions
        )
        d1_weight = _fraction(
            self.d1_context_weight_fraction,
            name="D1 context weight",
        )
        dw2_weights = tuple(
            _fraction(pair, name="cumulative D_W2 context weight")
            for pair in self.cumulative_dw2_context_weight_fractions
        )
        residual_masses = tuple(
            _fraction(pair, name="residual stratum mass")
            for pair in self.residual_stratum_mass_fractions
        )
        _validate_binary64_fraction_enclosure(
            exact=d1_weight,
            enclosure=self.d1_context_weight_binary64_enclosure,
            name="D1 context weight binary64 enclosure",
        )
        dw2_enclosures = tuple(self.cumulative_dw2_context_weight_binary64_enclosures)
        residual_conditional_enclosures = tuple(
            self.residual_conditional_weight_binary64_enclosures
        )
        local_enclosures = tuple(
            self.source_local_conditional_residual_weight_binary64_enclosures
        )
        full_counts = tuple(
            _nonnegative_int(value, name="full-population stratum count")
            for value in self.full_population_stratum_counts
        )
        d1_counts = tuple(
            _nonnegative_int(value, name="consumed D1 stratum count")
            for value in self.consumed_d1_stratum_counts
        )
        residual_counts = tuple(
            _nonnegative_int(value, name="residual stratum capacity")
            for value in self.residual_stratum_capacity_counts
        )
        full_population = _nonnegative_int(
            self.full_population_count,
            name="full population count",
        )
        residual_population = _nonnegative_int(
            self.residual_population_count,
            name="residual population count",
        )
        if (
            len(full_counts) != 2
            or len(d1_counts) != 2
            or len(residual_counts) != 2
            or residual_counts
            != tuple(full_counts[index] - d1_counts[index] for index in (0, 1))
            or sum(full_counts) != full_population
            or sum(residual_counts) != residual_population
            or residual_population <= 0
            or after_counts != tuple(self.after_suffix_stratum_counts)
            or any(
                residual_counts[index] > 0 and after_counts[index] <= 0
                for index in (0, 1)
            )
        ):
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.WEIGHT_MISMATCH,
                "D1 and residual counts do not partition the full population",
            )
        residual_scale = sum(residual_masses, start=Fraction(0))
        if residual_scale <= 0:
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.WEIGHT_MISMATCH,
                "residual population scale must be positive",
            )
        residual_conditional_weights = tuple(
            value / residual_scale for value in dw2_weights
        )
        if len(dw2_enclosures) != 2 or len(residual_conditional_enclosures) != 2:
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.WEIGHT_MISMATCH,
                "source-issued binary64 enclosure axes must contain both strata",
            )
        for index in (0, 1):
            _validate_binary64_fraction_enclosure(
                exact=dw2_weights[index],
                enclosure=dw2_enclosures[index],
                name=f"cumulative D_W2 stratum {index} binary64 enclosure",
            )
            _validate_binary64_fraction_enclosure(
                exact=residual_conditional_weights[index],
                enclosure=residual_conditional_enclosures[index],
                name=f"residual-conditional stratum {index} binary64 enclosure",
            )
        for index, (exact, enclosure) in enumerate(
            zip(local_fractions, local_enclosures, strict=True)
        ):
            _validate_binary64_fraction_enclosure(
                exact=exact,
                enclosure=enclosure,
                name=f"source-local context {index} binary64 enclosure",
            )
        expected_population_enclosure_hash = content_hash(
            {
                "schema": "sampled-residual-population-weight-enclosures-v1",
                "weight_receipt_hash": self.residual_population_weight_receipt_hash,
                "cumulative_dw2_stratum_counts": self.after_suffix_stratum_counts,
                "d1_context_weight_fraction": self.d1_context_weight_fraction,
                "dw2_context_weight_fractions": (
                    self.cumulative_dw2_context_weight_fractions
                ),
                "residual_conditional_context_weight_fractions": tuple(
                    (value.numerator, value.denominator)
                    for value in residual_conditional_weights
                ),
                "d1_context_weight_binary64_enclosure": (
                    self.d1_context_weight_binary64_enclosure
                ),
                "dw2_context_weight_binary64_enclosures": (
                    self.cumulative_dw2_context_weight_binary64_enclosures
                ),
                "residual_conditional_weight_binary64_enclosures": (
                    self.residual_conditional_weight_binary64_enclosures
                ),
            },
            prefix=_PREFIX + "population-weight-enclosures-",
        )
        if (
            self.residual_population_weight_binary64_enclosure_hash
            != expected_population_enclosure_hash
        ):
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.STALE_IDENTITY,
                "population-weight enclosure hash does not replay its exact measure",
            )
        local_quanta = tuple(
            _nonnegative_int(value, name="source-local context measure quantum")
            for value in self.source_local_context_measure_quanta
        )
        cumulative_quanta = tuple(
            _nonnegative_int(value, name="cumulative D_W2 context quantum")
            for value in self.cumulative_dw2_context_measure_quanta
        )
        if (
            min(local_denominator, full_denominator, d1_quantum, total_quanta) <= 0
            or len(dw2_weights) != 2
            or len(residual_masses) != 2
            or len(cumulative_quanta) != 2
            or any(value <= 0 for value in local_fractions)
            or any(value <= 0 for value in local_quanta)
            or sum(local_fractions, start=Fraction(0)) != Fraction(1)
            or residual_masses
            != tuple(Fraction(value, full_population) for value in residual_counts)
            or d1_weight != Fraction(1, full_population)
            or dw2_weights
            != tuple(
                (
                    Fraction(
                        residual_counts[index],
                        full_population * after_counts[index],
                    )
                    if residual_counts[index] > 0
                    else Fraction(0)
                )
                for index in (0, 1)
            )
            or sum(d1_counts) * d1_weight
            + sum(after_counts[index] * dw2_weights[index] for index in (0, 1))
            != Fraction(1)
            or sum(local_quanta) != local_denominator
            or any(
                Fraction(quantum, local_denominator) != weight
                for quantum, weight in zip(local_quanta, local_fractions, strict=True)
            )
            or Fraction(d1_quantum, full_denominator) != d1_weight
            or any(
                Fraction(cumulative_quanta[index], full_denominator)
                != dw2_weights[index]
                for index in (0, 1)
            )
            or total_quanta != full_denominator
        ):
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.WEIGHT_MISMATCH,
                "receipt weights and exact integer measure do not agree",
            )
        required_true = (
            self.target_then_complement_order,
            self.request_and_append_reserved_before_outcomes,
            self.terminal_freeze_not_an_identity_ancestor,
            self.source_issued,
            self.arbitrary_positive_packet_stratum_counts_supported,
            self.source_local_integer_transport_exact,
            self.stratum_resolved_segment_registration_required,
            self.append_stratum_ids_are_context_unique,
            self.logical_stratum_axis_bound_separately,
            self.unequal_count_census_tail_requires_stratum_resolved_composition,
            self.source_issued_full_population_measure_bound,
            self.source_registration_is_conditional_residual_transport_only,
            self.source_local_normalization_not_decision_measure,
            self.stratum_resolved_full_population_compositor_required,
            self.request_id_is_provenance_only,
            self.full_suffix_schedule_axes_validated,
            self.zero_count_unequal_terminal_tail_supported,
        )
        closed = (
            self.provider_or_outcome_receipt_custody,
            self.alpha_spend_authority,
            self.trajectory_nonrecycling_alpha_closed,
            self.scoring_authority,
            self.confidence_authority,
            self.action_decision_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
            self.exact_d1_compact_source_contribution_joined,
            self.full_population_decision_witness_authority,
        )
        if (
            any(type(value) is not bool for value in required_true + closed)
            or any(value is not True for value in required_true)
            or any(value is not False for value in closed)
            or self.authority_scope
            != SAMPLED_VARIABLE_SHELL_PACKET_REGISTRATION_AUTHORITY
            or self.schema != SAMPLED_VARIABLE_SHELL_PACKET_REGISTRATION_RECEIPT_SCHEMA
        ):
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.AUTHORITY_VIOLATION,
                "packet-registration receipt crossed its pre-outcome authority boundary",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_PREFIX + "receipt-")
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.STALE_IDENTITY,
                "packet-registration receipt changed after issuance",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.STALE_IDENTITY,
                "packet-registration receipt hash differs",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_covered_by_final_freeze(
        self,
        freeze: V1SampledVariableShellDW2SuffixPrefixFreezeReceipt,
    ) -> str:
        """Verify a later outcome-dependent freeze includes this planned packet."""

        if type(freeze) is not V1SampledVariableShellDW2SuffixPrefixFreezeReceipt:
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.INVALID_TYPE,
                "final coverage check requires the exact D_W2 freeze receipt",
            )
        freeze.assert_integrity()
        if (
            freeze.suffix_registration_hash != self.suffix_registration_hash
            or freeze.generation_frame_hash != self.generation_frame_hash
            or freeze.discovery_stream_hash != self.discovery_stream_hash
            or freeze.provisional_shell_hash != self.provisional_shell_hash
            or freeze.provisional_action_family_hash
            != self.provisional_action_family_hash
            or self.packet_ordinal >= len(freeze.completed_packet_hashes)
            or freeze.completed_packet_hashes[self.packet_ordinal] != self.packet_hash
            or any(
                freeze.consumed_dw2_stratum_counts[index]
                < self.after_suffix_stratum_counts[index]
                for index in (0, 1)
            )
        ):
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.PREFIX_FREEZE_MISMATCH,
                "terminal D_W2 freeze does not cover the predictable source packet",
            )
        return self.planned_d_w2_prefix_hash


@dataclass(frozen=True, slots=True)
class V1SampledVariableShellPacketRegistrationResult:
    source_registration: V1SignedWorkVariableShellSourceRegistration
    append_packet_registration: V1SignedWorkVariableShellSourceAppendPacketRegistration
    receipt: V1SampledVariableShellPacketRegistrationReceipt

    def __post_init__(self) -> None:
        if (
            type(self.source_registration)
            is not V1SignedWorkVariableShellSourceRegistration
        ):
            raise TypeError(
                "packet result requires the exact compact-source registration"
            )
        if (
            type(self.append_packet_registration)
            is not V1SignedWorkVariableShellSourceAppendPacketRegistration
        ):
            raise TypeError("packet result requires the exact append reservation")
        if type(self.receipt) is not V1SampledVariableShellPacketRegistrationReceipt:
            raise TypeError("packet result requires the exact binder receipt")
        if (
            self.source_registration.registration_hash
            != self.receipt.source_registration_hash
            or self.append_packet_registration.registration_hash
            != self.receipt.append_packet_registration_hash
        ):
            raise ValueError("packet result registrations differ from their receipt")
        self.receipt.assert_integrity()


def _validate_parent_graph_and_order(
    *,
    frame: V1AlexNetFQCausalGenerationFrameReceipt,
    discovery_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt,
    registration: V1SampledVariableShellDW2SuffixRegistration,
    packet: V1SampledVariableShellDW2SuffixPacket,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if type(frame) is not V1AlexNetFQCausalGenerationFrameReceipt:
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.INVALID_TYPE,
            "packet binder requires the exact causal generation frame",
        )
    if type(discovery_stream) is not V1AlexNetFQCausalRoleSamplingStreamReceipt:
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.INVALID_TYPE,
            "packet binder requires the exact discovery stream",
        )
    try:
        registration.assert_integrity()
        registration.assert_parent_graph(frame=frame, discovery_stream=discovery_stream)
        packet.assert_integrity()
    except (TypeError, ValueError, PermissionError, RuntimeError) as error:
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.STALE_IDENTITY,
            f"D_W2 parent graph failed integrity: {error}",
        )
    if (
        packet.suffix_registration_hash != registration.receipt_hash
        or packet.generation_frame_hash != frame.receipt_hash
        or packet.discovery_stream_hash != discovery_stream.receipt_hash
    ):
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.STALE_IDENTITY,
            "packet does not descend from the exact registered D stream",
        )
    d1_target, d1_complement = registration.consumed_d1_stratum_counts
    before_target, before_complement = packet.before_suffix_stratum_counts
    target_order = discovery_stream.target_assignment_receipt.generated_order
    complement_order = discovery_stream.complement_assignment_receipt.generated_order
    expected_target = target_order[
        d1_target + before_target : d1_target + packet.after_suffix_stratum_counts[0]
    ]
    expected_complement = complement_order[
        d1_complement
        + before_complement : d1_complement
        + packet.after_suffix_stratum_counts[1]
    ]
    if (
        packet.target_context_ids != expected_target
        or packet.complement_context_ids != expected_complement
    ):
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.ORDER_MISMATCH,
            "packet contexts are not the exact sealed-permutation suffix slice",
        )
    full_target_suffix = target_order[d1_target:]
    full_complement_suffix = complement_order[d1_complement:]
    return full_target_suffix, full_complement_suffix


def _validate_request(
    *,
    request: MeasurementRequest,
    registration: V1SampledVariableShellDW2SuffixRegistration,
    packet: V1SampledVariableShellDW2SuffixPacket,
    residual_measure_design: V1SampledResidualPopulationMeasureDesign,
    shell_union: V1SignedWorkFixedPrimalShellUnionReceipt,
    measurement_schedule: V1SampledVariableShellMeasurementSchedule,
    measurement_source_lineage: V1SampledVariableShellMeasurementSourceLineage,
) -> None:
    if type(request) is not MeasurementRequest:
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.INVALID_TYPE,
            "packet binder requires one exact preregistered MeasurementRequest",
        )
    scheduled = measurement_schedule.packet_for(packet)
    contexts = scheduled.registered_context_ids
    selector = request.context_selector
    observed_request_id = content_hash(
        {
            "kind": request.kind,
            "state_hash": request.state_hash,
            "version": request.version,
            "context_selector": request.context_selector,
            "candidate_ids": tuple(request.candidate_ids),
            "expected_cost": request.expected_cost,
            "reuse_keys": tuple(request.reuse_keys),
            "provenance": request.provenance,
            "payload": dict(request.payload),
        },
        prefix="irg-request-",
    )
    if (
        observed_request_id != request.request_id
        or request.kind != "response_jacobian"
        or request.version != SAMPLED_VARIABLE_SHELL_PACKET_REQUEST_VERSION
        or request.state_hash
        != v1_sampled_variable_shell_packet_request_state_hash(
            registration=registration,
            packet=packet,
            residual_measure_design=residual_measure_design,
            shell_union=shell_union,
            measurement_schedule=measurement_schedule,
            measurement_source_lineage=measurement_source_lineage,
        )
        or selector.requested_count != len(contexts)
        or selector.context_ids != contexts
        or selector.strata != _STRATA
        or request.candidate_ids != shell_union.shell_route_ids
    ):
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.REQUEST_MISMATCH,
            "measurement request does not preregister this exact packet and shell",
        )
    payload = dict(request.payload)
    source_capture = payload.get("source_capture_identity_hash")
    if type(source_capture) is not str or not source_capture:
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.REQUEST_MISMATCH,
            "measurement request lacks its source-capture plan identity",
        )
    expected = v1_sampled_variable_shell_packet_request_payload(
        registration=registration,
        packet=packet,
        residual_measure_design=residual_measure_design,
        shell_union=shell_union,
        measurement_schedule=measurement_schedule,
        measurement_source_lineage=measurement_source_lineage,
    )
    if payload != expected:
        unexpected = set(payload).difference(expected)
        if any(
            "outcome" in key.lower()
            or "provider_receipt" in key.lower()
            or "measurement_receipt" in key.lower()
            or "commit_authority" in key.lower()
            for key in unexpected
        ):
            _refuse(
                V1SampledVariableShellPacketRegistrationRefusalCode.OUTCOME_LEAKAGE,
                "provider/outcome/downstream material cannot enter preregistration",
            )
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.REQUEST_MISMATCH,
            "measurement request payload differs from the exact pre-outcome allow-list",
        )


def bind_v1_sampled_variable_shell_packet_registration(
    *,
    frame: V1AlexNetFQCausalGenerationFrameReceipt,
    discovery_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    packet: V1SampledVariableShellDW2SuffixPacket,
    measurement_schedule: V1SampledVariableShellMeasurementSchedule,
    measurement_source_lineage: V1SampledVariableShellMeasurementSourceLineage,
    residual_measure_design: V1SampledResidualPopulationMeasureDesign,
    measurement_request: MeasurementRequest,
    shell_union: V1SignedWorkFixedPrimalShellUnionReceipt,
    native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt,
    append_owner: V1SignedWorkVariableShellSourceAppendOwner,
) -> V1SampledVariableShellPacketRegistrationResult:
    """Reserve and bind one complete target-then-complement ``D_W2`` packet."""

    if type(suffix_registration) is not V1SampledVariableShellDW2SuffixRegistration:
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.INVALID_TYPE,
            "packet binder requires the exact D_W2 suffix registration",
        )
    if type(packet) is not V1SampledVariableShellDW2SuffixPacket:
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.INVALID_TYPE,
            "packet binder requires the exact D_W2 suffix packet",
        )
    if type(measurement_schedule) is not V1SampledVariableShellMeasurementSchedule:
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.INVALID_TYPE,
            "packet binder requires the source-issued full-suffix measurement schedule",
        )
    if (
        type(measurement_source_lineage)
        is not V1SampledVariableShellMeasurementSourceLineage
    ):
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.INVALID_TYPE,
            "packet binder requires the source-issued branch-independent measurement lineage",
        )
    if type(shell_union) is not V1SignedWorkFixedPrimalShellUnionReceipt:
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.INVALID_TYPE,
            "packet binder requires the source-issued two-branch shell union",
        )
    if type(native_fiber_axis_receipt) is not V1NativeDestinationFiberAxisReceipt:
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.INVALID_TYPE,
            "packet binder requires the source-issued native-fiber axis",
        )
    if type(append_owner) is not V1SignedWorkVariableShellSourceAppendOwner:
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.INVALID_TYPE,
            "packet binder requires the exact live append owner",
        )
    _validate_parent_graph_and_order(
        frame=frame,
        discovery_stream=discovery_stream,
        registration=suffix_registration,
        packet=packet,
    )
    measurement_schedule.assert_parent_graph(
        frame=frame,
        discovery_stream=discovery_stream,
        suffix_registration=suffix_registration,
    )
    scheduled_packet = measurement_schedule.packet_for(packet)
    measurement_source_lineage.assert_integrity()
    _validate_measure_design(
        design=residual_measure_design,
        registration=suffix_registration,
    )
    weight_receipt = _issue_measure_weights(
        design=residual_measure_design,
        registration=suffix_registration,
        packet=packet,
    )
    native_fiber_axis_receipt.assert_integrity()
    lineage = append_owner.lineage
    shell_fibers = tuple(
        native_fiber_axis_receipt.route_to_fiber(index).global_fiber_index
        for index in shell_union.shell_indices
    )
    expected_population_context_hash = (
        measurement_schedule.population_context_axis_hash
    )
    expected_measure_source = residual_measure_design.design_hash
    expected_unit = v1_sampled_variable_shell_common_measure_unit_hash(
        registration=suffix_registration,
    )
    if (
        measurement_source_lineage.measurement_schedule_hash
        != measurement_schedule.schedule_hash
        or measurement_source_lineage.capture_policy_identity_hash
        != measurement_schedule.capture_policy_identity_hash
        or measurement_source_lineage.suffix_registration_hash
        != suffix_registration.receipt_hash
        or measurement_source_lineage.residual_population_measure_design_hash
        != residual_measure_design.design_hash
        or measurement_source_lineage.shell_union_receipt_hash
        != shell_union.receipt_hash
        or measurement_source_lineage.model_identity_hash
        != lineage.model_identity_hash
        or measurement_source_lineage.chart_identity_hash
        != lineage.chart_identity_hash
        or measurement_source_lineage.current_mask_hash
        != lineage.current_mask_hash
        or measurement_source_lineage.proposal_generation
        != lineage.proposal_generation
        or measurement_source_lineage.route_axis_hash != lineage.route_axis_hash
        or measurement_source_lineage.route_id_axis_hash
        != lineage.route_id_axis_hash
        or measurement_source_lineage.native_fiber_axis_receipt_hash
        != lineage.native_fiber_axis_receipt_hash
        or measurement_source_lineage.native_fiber_measure_hash
        != lineage.native_fiber_measure_hash
        or measurement_source_lineage.witness_stream_source_hash
        != lineage.witness_stream_source_hash
        or measurement_source_lineage.same_capture_vjp_source_hash
        != lineage.same_capture_vjp_source_hash
        or measurement_source_lineage.corrected_evidence_stream_source_hash
        != lineage.corrected_evidence_stream_source_hash
        or measurement_source_lineage.literal_j_stream_source_hash
        != lineage.literal_j_stream_source_hash
        or measurement_source_lineage.alpha_ledger_id != lineage.alpha_ledger_id
        or measurement_source_lineage.alpha_coordinate_id
        != lineage.alpha_coordinate_id
        or measurement_source_lineage.alpha_lineage_hash
        != lineage.alpha_lineage_hash
        or
        suffix_registration.provisional_shell_hash != shell_union.provisional_shell_hash
        or suffix_registration.provisional_action_family_hash
        != shell_union.provisional_action_family_hash
        or lineage.chart_identity_hash != shell_union.chart_identity_hash
        or lineage.current_mask_hash != shell_union.current_mask_hash
        or lineage.current_mask_hash != suffix_registration.current_mask_hash
        or lineage.proposal_generation != suffix_registration.generation_index
        or lineage.route_axis_hash != shell_union.route_axis_hash
        or lineage.route_id_axis_hash != shell_union.route_id_axis_hash
        or lineage.shell_freeze_receipt_hash != shell_union.receipt_hash
        or lineage.shell_indices != shell_union.shell_indices
        or lineage.shell_route_ids != shell_union.shell_route_ids
        or lineage.shell_native_fiber_indices != shell_fibers
        or lineage.native_fiber_axis_receipt_hash
        != native_fiber_axis_receipt.receipt_hash
        or lineage.native_fiber_measure_hash != native_fiber_axis_receipt.measure_hash
        or native_fiber_axis_receipt.route_axis_hash != shell_union.route_axis_hash
        or native_fiber_axis_receipt.route_id_axis_hash
        != shell_union.route_id_axis_hash
        or native_fiber_axis_receipt.route_count != shell_union.route_count
        or lineage.route_count != native_fiber_axis_receipt.route_count
        or lineage.fiber_count != native_fiber_axis_receipt.fiber_count
        or lineage.cut_count != native_fiber_axis_receipt.cut_count
        or lineage.registered_d_prefix_hash
        != suffix_registration.d1_pair_prefix_axis_hash
        or lineage.kkt_branch_id != SAMPLED_VARIABLE_SHELL_TWO_BRANCH_KKT_ID
        or lineage.kkt_branch_source_receipt_hash != shell_union.receipt_hash
        or lineage.predictable_packet_order_source_hash
        != measurement_schedule.schedule_hash
        or lineage.stratum_measure_source_hash != expected_measure_source
        or lineage.common_integer_measure_unit_hash != expected_unit
        or lineage.population_count
        != sum(suffix_registration.suffix_capacity_stratum_counts)
        or lineage.population_context_axis_hash != expected_population_context_hash
        or lineage.population_draw_axis_hash
        != measurement_schedule.population_draw_axis_hash
    ):
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.STALE_IDENTITY,
            "live source lineage does not bind the exact D parent, shell, geometry, or measure",
        )
    if (
        append_owner.has_pending_packet
        or append_owner.segment_count != packet.packet_ordinal
        or append_owner.context_count != sum(packet.before_suffix_stratum_counts)
    ):
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.ORDER_MISMATCH,
            "packet is reordered, overlapping, or another append reservation is active",
        )
    _validate_request(
        request=measurement_request,
        registration=suffix_registration,
        packet=packet,
        residual_measure_design=residual_measure_design,
        shell_union=shell_union,
        measurement_schedule=measurement_schedule,
        measurement_source_lineage=measurement_source_lineage,
    )
    (
        contexts,
        draw_ids,
        measurement_ids,
        capture_ids,
        source_packet_hashes,
    ) = _derive_packet_axes(
        packet=packet,
        measurement_schedule=measurement_schedule,
    )
    all_axes = (
        contexts,
        draw_ids,
        measurement_ids,
        capture_ids,
        source_packet_hashes,
    )
    if any(len(axis) != len(set(axis)) for axis in all_axes):
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.OVERLAP,
            "a predictable packet identity axis repeats an item",
        )
    if (
        contexts != scheduled_packet.registered_context_ids
        or draw_ids != scheduled_packet.registered_draw_ids
        or measurement_ids != scheduled_packet.registered_measurement_ids
        or capture_ids != scheduled_packet.registered_capture_ids
        or source_packet_hashes
        != scheduled_packet.registered_source_packet_hashes
        or measurement_schedule.population_draw_axis_hash
        != v1_variable_shell_population_axis_hash(
            "draw", measurement_schedule.population_draw_ids
        )
        or measurement_schedule.population_context_axis_hash
        != v1_variable_shell_population_axis_hash(
            "context", measurement_schedule.population_context_ids
        )
    ):
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.SCHEDULE_MISMATCH,
            "packet axes or population draw custody differ from the frozen schedule",
        )
    fractions, quanta, common_denominator = _source_local_weights_and_quanta(
        weight_receipt=weight_receipt,
        packet=packet,
    )
    source_local_binary64_enclosures = tuple(
        _binary64_fraction_enclosure(value) for value in fractions
    )
    planned_prefix = v1_sampled_variable_shell_planned_prefix_hash(
        registration=suffix_registration,
        packet=packet,
    )
    registration_id = content_hash(
        {
            "schema": SAMPLED_VARIABLE_SHELL_PACKET_REGISTRATION_SCHEMA,
            "measurement_request_id": measurement_request.request_id,
            "measurement_schedule_hash": measurement_schedule.schedule_hash,
            "measurement_schedule_packet_hash": (
                scheduled_packet.packet_schedule_hash
            ),
            "planned_d_w2_prefix_hash": planned_prefix,
            "shell_union_receipt_hash": shell_union.receipt_hash,
            "measurement_source_lineage_hash": (
                measurement_source_lineage.lineage_hash
            ),
            "append_lineage_hash": lineage.lineage_hash,
            "context_axis_hash": _source_axis_hash("context", contexts),
            "draw_axis_hash": _source_axis_hash("draw", draw_ids),
            "measurement_axis_hash": _source_axis_hash("measurement", measurement_ids),
            "capture_axis_hash": _source_axis_hash("capture", capture_ids),
            "source_packet_axis_hash": _source_axis_hash(
                "source_packet", source_packet_hashes
            ),
            "residual_population_measure_design_hash": (
                residual_measure_design.design_hash
            ),
            "residual_population_weight_receipt_hash": weight_receipt.receipt_hash,
            "source_local_conditional_residual_weight_fractions": tuple(
                (value.numerator, value.denominator) for value in fractions
            ),
            "source_local_conditional_residual_weight_binary64_enclosures": (
                source_local_binary64_enclosures
            ),
        },
        prefix=_PREFIX + "registration-id-",
    )
    try:
        source_registration = V1SignedWorkVariableShellSourceRegistration(
            registration_id=registration_id,
            model_identity_hash=lineage.model_identity_hash,
            chart_identity_hash=lineage.chart_identity_hash,
            current_mask_hash=lineage.current_mask_hash,
            proposal_generation=lineage.proposal_generation,
            route_axis_hash=lineage.route_axis_hash,
            route_id_axis_hash=lineage.route_id_axis_hash,
            registered_d_prefix_hash=lineage.registered_d_prefix_hash,
            registered_d_w2_prefix_hash=planned_prefix,
            parent_d_source_receipt_hash=suffix_registration.discovery_stream_hash,
            shell_freeze_receipt_hash=shell_union.receipt_hash,
            shell_indices=shell_union.shell_indices,
            shell_route_ids=shell_union.shell_route_ids,
            native_fiber_axis_receipt_hash=lineage.native_fiber_axis_receipt_hash,
            native_fiber_measure_hash=lineage.native_fiber_measure_hash,
            kkt_branch_id=lineage.kkt_branch_id,
            kkt_branch_source_receipt_hash=lineage.kkt_branch_source_receipt_hash,
            resource_axis_hash=lineage.resource_axis_hash,
            resource_source_receipt_hash=lineage.resource_source_receipt_hash,
            witness_stream_source_hash=lineage.witness_stream_source_hash,
            same_capture_vjp_source_hash=lineage.same_capture_vjp_source_hash,
            corrected_evidence_stream_source_hash=(
                lineage.corrected_evidence_stream_source_hash
            ),
            literal_j_stream_source_hash=lineage.literal_j_stream_source_hash,
            alpha_ledger_id=lineage.alpha_ledger_id,
            alpha_coordinate_id=lineage.alpha_coordinate_id,
            alpha_lineage_hash=lineage.alpha_lineage_hash,
            population_count=lineage.population_count,
            population_context_axis_hash=lineage.population_context_axis_hash,
            population_draw_axis_hash=lineage.population_draw_axis_hash,
            registered_context_ids=contexts,
            registered_draw_ids=draw_ids,
            registered_measurement_ids=measurement_ids,
            registered_capture_ids=capture_ids,
            registered_source_packet_hashes=source_packet_hashes,
            context_weights=tuple(
                enclosure[1] for enclosure in source_local_binary64_enclosures
            ),
        )
    except V1SignedWorkVariableShellSourceRefusal as error:
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.STALE_IDENTITY,
            f"compact source registration refused the bound packet: {error}",
        )
    stratum_ids = ("target",) * len(packet.target_context_ids) + ("complement",) * len(
        packet.complement_context_ids
    )
    append_stratum_ids = tuple(
        content_hash(
            {
                "schema": "sampled-variable-shell-append-stratum-item-v1",
                "stratum_measure_source_hash": expected_measure_source,
                "packet_hash": packet.receipt_hash,
                "item_ordinal": ordinal,
                "stratum_id": stratum_id,
                "context_id": context_id,
            },
            prefix=_PREFIX + f"append-stratum-{stratum_id}-",
        )
        for ordinal, (stratum_id, context_id) in enumerate(
            zip(stratum_ids, contexts, strict=True)
        )
    )
    append_logical_stratum_mapping_hash = _append_logical_stratum_mapping_hash(
        append_stratum_ids=append_stratum_ids,
        logical_stratum_ids=stratum_ids,
    )
    try:
        append_registration = append_owner.register_next_packet(
            registered_d_w2_prefix_hash=planned_prefix,
            context_ids=contexts,
            draw_ids=draw_ids,
            measurement_ids=measurement_ids,
            capture_ids=capture_ids,
            source_packet_hashes=source_packet_hashes,
            stratum_ids=append_stratum_ids,
            context_measure_quanta=quanta,
            logical_stratum_ids=stratum_ids,
        )
    except V1SignedWorkVariableShellSourceAppendRefusal as error:
        _refuse(
            V1SampledVariableShellPacketRegistrationRefusalCode.ORDER_MISMATCH,
            f"live append owner refused the predictable packet reservation: {error}",
        )
    receipt = V1SampledVariableShellPacketRegistrationReceipt(
        suffix_registration_hash=suffix_registration.receipt_hash,
        packet_hash=packet.receipt_hash,
        packet_ordinal=packet.packet_ordinal,
        generation_frame_hash=frame.receipt_hash,
        discovery_stream_hash=discovery_stream.receipt_hash,
        measurement_schedule_hash=measurement_schedule.schedule_hash,
        measurement_schedule_source_hash=(measurement_schedule.schedule_source_hash),
        measurement_schedule_packet_hash=scheduled_packet.packet_schedule_hash,
        capture_policy_identity_hash=(
            measurement_schedule.capture_policy_identity_hash
        ),
        population_context_axis_hash=(
            measurement_schedule.population_context_axis_hash
        ),
        population_draw_axis_hash=measurement_schedule.population_draw_axis_hash,
        population_measurement_axis_hash=(
            measurement_schedule.population_measurement_axis_hash
        ),
        population_capture_axis_hash=(
            measurement_schedule.population_capture_axis_hash
        ),
        population_source_packet_axis_hash=(
            measurement_schedule.population_source_packet_axis_hash
        ),
        measurement_request_id=measurement_request.request_id,
        measurement_request_state_hash=measurement_request.state_hash,
        source_capture_identity_hash=scheduled_packet.source_capture_identity_hash,
        planned_d_w2_prefix_hash=planned_prefix,
        shell_union_receipt_hash=shell_union.receipt_hash,
        provisional_shell_hash=shell_union.provisional_shell_hash,
        provisional_action_family_hash=shell_union.provisional_action_family_hash,
        measurement_source_lineage_hash=measurement_source_lineage.lineage_hash,
        append_lineage_hash=lineage.lineage_hash,
        append_packet_registration_hash=append_registration.registration_hash,
        source_registration_hash=source_registration.registration_hash,
        registered_context_ids=contexts,
        registered_draw_ids=draw_ids,
        registered_measurement_ids=measurement_ids,
        registered_capture_ids=capture_ids,
        registered_source_packet_hashes=source_packet_hashes,
        registered_stratum_ids=stratum_ids,
        append_registered_stratum_ids=append_stratum_ids,
        packet_stratum_counts=_packet_counts(packet),
        after_suffix_stratum_counts=packet.after_suffix_stratum_counts,
        residual_population_measure_design_hash=(residual_measure_design.design_hash),
        residual_population_weight_receipt_hash=weight_receipt.receipt_hash,
        residual_population_weight_binary64_enclosure_hash=(
            _residual_population_weight_enclosure_hash(weight_receipt)
        ),
        append_logical_stratum_mapping_hash=append_logical_stratum_mapping_hash,
        full_population_stratum_counts=weight_receipt.population_stratum_counts,
        consumed_d1_stratum_counts=weight_receipt.d1_stratum_counts,
        residual_stratum_capacity_counts=weight_receipt.residual_stratum_counts,
        full_population_count=residual_measure_design.population_count,
        residual_population_count=sum(weight_receipt.residual_stratum_counts),
        d1_context_weight_fraction=weight_receipt.d1_context_weight_fraction,
        cumulative_dw2_context_weight_fractions=(
            weight_receipt.dw2_context_weight_fractions
        ),
        d1_context_weight_binary64_enclosure=(
            weight_receipt.d1_context_weight_binary64_enclosure
        ),
        cumulative_dw2_context_weight_binary64_enclosures=(
            weight_receipt.dw2_context_weight_binary64_enclosures
        ),
        residual_conditional_weight_binary64_enclosures=(
            weight_receipt.residual_conditional_weight_binary64_enclosures
        ),
        residual_stratum_mass_fractions=(
            weight_receipt.residual_stratum_mass_fractions
        ),
        full_population_measure_common_denominator=(weight_receipt.common_denominator),
        d1_context_measure_quantum=weight_receipt.d1_context_quantum,
        cumulative_dw2_context_measure_quanta=weight_receipt.dw2_context_quanta,
        full_population_total_measure_quanta=weight_receipt.total_weight_quanta,
        source_local_conditional_residual_weight_fractions=tuple(
            (value.numerator, value.denominator) for value in fractions
        ),
        source_local_conditional_residual_weight_binary64_enclosures=(
            source_local_binary64_enclosures
        ),
        source_local_context_measure_quanta=quanta,
        source_local_common_measure_denominator=common_denominator,
        model_identity_hash=lineage.model_identity_hash,
        chart_identity_hash=lineage.chart_identity_hash,
        current_mask_hash=lineage.current_mask_hash,
        proposal_generation=lineage.proposal_generation,
        route_axis_hash=lineage.route_axis_hash,
        route_id_axis_hash=lineage.route_id_axis_hash,
        native_fiber_axis_receipt_hash=lineage.native_fiber_axis_receipt_hash,
        native_fiber_measure_hash=lineage.native_fiber_measure_hash,
        kkt_branch_id=lineage.kkt_branch_id,
        kkt_branch_source_receipt_hash=lineage.kkt_branch_source_receipt_hash,
        resource_axis_hash=lineage.resource_axis_hash,
        resource_source_receipt_hash=lineage.resource_source_receipt_hash,
        witness_stream_source_hash=lineage.witness_stream_source_hash,
        same_capture_vjp_source_hash=lineage.same_capture_vjp_source_hash,
        corrected_evidence_stream_source_hash=(
            lineage.corrected_evidence_stream_source_hash
        ),
        literal_j_stream_source_hash=lineage.literal_j_stream_source_hash,
        alpha_ledger_id=lineage.alpha_ledger_id,
        alpha_coordinate_id=lineage.alpha_coordinate_id,
        alpha_lineage_hash=lineage.alpha_lineage_hash,
        _issuance_token=_ISSUER_TOKEN,
    )
    return V1SampledVariableShellPacketRegistrationResult(
        source_registration=source_registration,
        append_packet_registration=append_registration,
        receipt=receipt,
    )


__all__ = [
    "SAMPLED_VARIABLE_SHELL_PACKET_REGISTRATION_AUTHORITY",
    "SAMPLED_VARIABLE_SHELL_PACKET_REGISTRATION_RECEIPT_SCHEMA",
    "SAMPLED_VARIABLE_SHELL_PACKET_REGISTRATION_SCHEMA",
    "SAMPLED_VARIABLE_SHELL_PACKET_REQUEST_PAYLOAD_SCHEMA",
    "SAMPLED_VARIABLE_SHELL_PACKET_REQUEST_VERSION",
    "SAMPLED_VARIABLE_SHELL_MEASUREMENT_SCHEDULE_PACKET_SCHEMA",
    "SAMPLED_VARIABLE_SHELL_MEASUREMENT_SCHEDULE_SCHEMA",
    "SAMPLED_VARIABLE_SHELL_MEASUREMENT_SOURCE_LINEAGE_SCHEMA",
    "SAMPLED_VARIABLE_SHELL_TWO_BRANCH_KKT_ID",
    "V1SampledVariableShellMeasurementSchedule",
    "V1SampledVariableShellMeasurementSchedulePacket",
    "V1SampledVariableShellMeasurementSourceLineage",
    "V1SampledVariableShellPacketRegistrationReceipt",
    "V1SampledVariableShellPacketRegistrationRefusal",
    "V1SampledVariableShellPacketRegistrationRefusalCode",
    "V1SampledVariableShellPacketRegistrationResult",
    "bind_v1_sampled_variable_shell_packet_registration",
    "v1_sampled_variable_shell_common_measure_unit_hash",
    "v1_sampled_variable_shell_packet_request_payload",
    "v1_sampled_variable_shell_packet_request_state_hash",
    "v1_sampled_variable_shell_planned_prefix_hash",
    "v1_sampled_variable_shell_stratum_measure_source_hash",
]
