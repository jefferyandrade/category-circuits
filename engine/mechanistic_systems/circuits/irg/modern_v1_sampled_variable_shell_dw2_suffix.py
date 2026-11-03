"""Causal custody for the post-shell ``D_W2`` discovery suffix.

Variable-shell inference first consumes a discovery prefix ``D_1`` and freezes
a provisional shell/action family.  It may then need more observations to
resolve that shell without opening C.  Those observations must be the next
contexts of the *same* still-sealed discovery permutations, never a new or
caller-selected order.

This module owns that narrow causal boundary.  It binds the exact AlexNet F/Q
generation frame and discovery stream, records the already-consumed ``D_1``
prefix, and exposes a nonserializable cursor whose packets are deterministic
prefix slices of the remaining target and complement orders.  Packet size is
only a transport atom: the number of packets consumed, and hence the final
statistical stopping count, is intentionally not registered in advance.

No scorer, outcome, confidence statement, proposal decision, mask, C/K/O
authority, commit authority, or Pareto authority is present here.  The final
freeze receipt proves only which predictable ``D_W2`` suffix prefix was
revealed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from numbers import Integral
import threading
from typing import Final, NoReturn, Sequence

from .contracts import content_hash
from .modern_v1_alexnet_fq_registration import (
    V1AlexNetFQCausalGenerationFrameReceipt,
    V1AlexNetFQCausalRoleSamplingStreamReceipt,
)


DW2_SUFFIX_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_sampled_variable_shell_D_W2_suffix_registration_v1"
)
DW2_SUFFIX_PACKET_SCHEMA: Final = (
    "partition_free_v1_sampled_variable_shell_D_W2_suffix_packet_v1"
)
DW2_SUFFIX_PREFIX_FREEZE_SCHEMA: Final = (
    "partition_free_v1_sampled_variable_shell_D_W2_suffix_prefix_freeze_v1"
)
DW2_SUFFIX_ALGORITHM: Final = (
    "same_discovery_permutation_post_provisional_shell_predictable_suffix_"
    "paired_stratum_transport_packets_v1"
)
DW2_SUFFIX_AUTHORITY_SCOPE: Final = (
    "sampling_identity_and_predictable_suffix_prefix_custody_only_no_score_"
    "confidence_decision_mask_C_K_O_commit_terminal_or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-variable-shell-D-W2-suffix-"
_STRATA: Final = ("target", "complement")


class V1SampledVariableShellDW2SuffixRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    INVALID_REGISTRATION = "invalid_registration"
    STALE_IDENTITY = "stale_identity"
    OVERLAP = "overlap"
    ORDER_MISMATCH = "order_mismatch"
    EXHAUSTED = "exhausted"
    ACTIVE_PACKET = "active_packet"
    NO_ACTIVE_PACKET = "no_active_packet"
    OUTCOME_LEAKAGE = "outcome_leakage"
    AUTHORITY_VIOLATION = "authority_violation"
    EMPTY_SUFFIX = "empty_suffix"
    CLOSED = "closed"
    TAMPERED = "tampered"


class V1SampledVariableShellDW2SuffixRefusal(RuntimeError):
    """Typed, fail-closed refusal for ``D_W2`` suffix custody."""

    def __init__(
        self,
        code: V1SampledVariableShellDW2SuffixRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledVariableShellDW2SuffixRefusalCode:
            raise TypeError("D_W2 suffix refusal code must have exact enum type")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledVariableShellDW2SuffixRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledVariableShellDW2SuffixRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledVariableShellDW2SuffixRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SampledVariableShellDW2SuffixRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SampledVariableShellDW2SuffixRefusalCode.INVALID_REGISTRATION,
            f"{name} must be nonnegative",
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        _refuse(
            V1SampledVariableShellDW2SuffixRefusalCode.INVALID_REGISTRATION,
            f"{name} must be positive",
        )
    return result


def _count_pair(value: object, *, name: str) -> tuple[int, int]:
    if type(value) not in {tuple, list} or len(value) != 2:
        _refuse(
            V1SampledVariableShellDW2SuffixRefusalCode.INVALID_TYPE,
            f"{name} must contain target then complement counts",
        )
    return (
        _nonnegative_int(value[0], name=f"{name} target"),
        _nonnegative_int(value[1], name=f"{name} complement"),
    )


def _identity_axis(
    value: Sequence[str],
    *,
    name: str,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if type(value) not in {tuple, list}:
        _refuse(
            V1SampledVariableShellDW2SuffixRefusalCode.INVALID_TYPE,
            f"{name} must be one ordered string sequence",
        )
    result = tuple(_identity(item, name=f"{name} item") for item in value)
    if not allow_empty and not result:
        _refuse(
            V1SampledVariableShellDW2SuffixRefusalCode.INVALID_REGISTRATION,
            f"{name} cannot be empty",
        )
    if len(result) != len(set(result)):
        _refuse(
            V1SampledVariableShellDW2SuffixRefusalCode.OVERLAP,
            f"{name} repeats a context",
        )
    return result


def _axis_hash(role: str, values: Sequence[str]) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_sampled_variable_shell_D_W2_axis_v1",
            "role": role,
            "ordered_context_ids": tuple(values),
        },
        prefix=_PREFIX + role + "-axis-",
    )


def _pair_axis_hash(
    *,
    role: str,
    target: Sequence[str],
    complement: Sequence[str],
) -> str:
    return content_hash(
        {
            "schema": ("partition_free_v1_sampled_variable_shell_D_W2_pair_axis_v1"),
            "role": role,
            "target": tuple(target),
            "complement": tuple(complement),
        },
        prefix=_PREFIX + role + "-pair-axis-",
    )


def _assert_exact_parent_graph(
    *,
    frame: object,
    discovery_stream: object,
) -> tuple[
    V1AlexNetFQCausalGenerationFrameReceipt,
    V1AlexNetFQCausalRoleSamplingStreamReceipt,
]:
    if type(frame) is not V1AlexNetFQCausalGenerationFrameReceipt:
        _refuse(
            V1SampledVariableShellDW2SuffixRefusalCode.INVALID_TYPE,
            "D_W2 suffix registration requires the exact causal generation frame",
        )
    if type(discovery_stream) is not V1AlexNetFQCausalRoleSamplingStreamReceipt:
        _refuse(
            V1SampledVariableShellDW2SuffixRefusalCode.INVALID_TYPE,
            "D_W2 suffix registration requires the exact discovery stream",
        )
    try:
        frame.assert_integrity()
        discovery_stream.assert_integrity()
    except (TypeError, ValueError, PermissionError) as error:
        _refuse(
            V1SampledVariableShellDW2SuffixRefusalCode.TAMPERED,
            f"D_W2 suffix parent graph failed integrity: {error}",
        )
    if (
        discovery_stream.role != "discovery"
        or discovery_stream.registration_hash != frame.registration_hash
        or discovery_stream.generation_frame_hash != frame.receipt_hash
        or discovery_stream.generation_index != frame.generation_index
        or discovery_stream.causal_parent_receipt_hash != frame.receipt_hash
        or discovery_stream.current_mask_hash != frame.current_mask_hash
        or discovery_stream.frozen_transform_hash != frame.frozen_transform_hash
        or discovery_stream.full_population_stratum_counts
        != frame.full_population_stratum_counts
        or discovery_stream.target_population_membership_hash
        != frame.target_population_membership_hash
        or discovery_stream.complement_population_membership_hash
        != frame.complement_population_membership_hash
    ):
        _refuse(
            V1SampledVariableShellDW2SuffixRefusalCode.STALE_IDENTITY,
            "discovery stream is not the exact current-frame D parent",
        )
    target = discovery_stream.target_assignment_receipt.generated_order
    complement = discovery_stream.complement_assignment_receipt.generated_order
    if len(target) != len(set(target)) or len(complement) != len(set(complement)):
        _refuse(
            V1SampledVariableShellDW2SuffixRefusalCode.OVERLAP,
            "a discovery stratum repeats a context",
        )
    if set(target).intersection(complement):
        _refuse(
            V1SampledVariableShellDW2SuffixRefusalCode.OVERLAP,
            "target and complement discovery permutations overlap",
        )
    return frame, discovery_stream


@dataclass(frozen=True, slots=True)
class V1SampledVariableShellDW2SuffixRegistration:
    """Outcome-independent registration of the remaining same-D permutation axes."""

    generation_frame_hash: str
    discovery_stream_hash: str
    registration_hash: str
    generation_index: int
    current_mask_hash: str
    frozen_transform_hash: str
    target_assignment_receipt_hash: str
    complement_assignment_receipt_hash: str
    target_population_membership_hash: str
    complement_population_membership_hash: str
    population_stratum_counts: tuple[int, int]
    stratum_mass_numerators: tuple[int, int]
    stratum_mass_denominator: int
    consumed_d1_stratum_counts: tuple[int, int]
    d1_target_prefix_axis_hash: str
    d1_complement_prefix_axis_hash: str
    d1_pair_prefix_axis_hash: str
    target_suffix_axis_hash: str
    complement_suffix_axis_hash: str
    suffix_pair_axis_hash: str
    suffix_capacity_stratum_counts: tuple[int, int]
    provisional_shell_hash: str
    provisional_action_family_hash: str
    transport_packet_size: int
    transport_packet_size_is_hardware_only: bool = True
    statistical_stopping_count_predeclared: bool = False
    predictable_same_discovery_permutation_suffix: bool = True
    cross_stratum_disjointness_verified: bool = True
    outcome_payload_present: bool = False
    scoring_authority: bool = False
    statistical_stopping_authority: bool = False
    proposal_decision_authority: bool = False
    mask_authority: bool = False
    certification_calibration_oracle_authority: bool = False
    commit_terminal_or_pareto_authority: bool = False
    validation_rows_seen: bool = False
    algorithm_id: str = DW2_SUFFIX_ALGORITHM
    authority_scope: str = DW2_SUFFIX_AUTHORITY_SCOPE
    schema: str = DW2_SUFFIX_REGISTRATION_SCHEMA
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "generation_frame_hash",
            "discovery_stream_hash",
            "registration_hash",
            "current_mask_hash",
            "frozen_transform_hash",
            "target_assignment_receipt_hash",
            "complement_assignment_receipt_hash",
            "target_population_membership_hash",
            "complement_population_membership_hash",
            "d1_target_prefix_axis_hash",
            "d1_complement_prefix_axis_hash",
            "d1_pair_prefix_axis_hash",
            "target_suffix_axis_hash",
            "complement_suffix_axis_hash",
            "suffix_pair_axis_hash",
            "provisional_shell_hash",
            "provisional_action_family_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        generation = _nonnegative_int(self.generation_index, name="generation index")
        populations = _count_pair(
            self.population_stratum_counts,
            name="population stratum counts",
        )
        if min(populations) <= 0:
            _refuse(
                V1SampledVariableShellDW2SuffixRefusalCode.INVALID_REGISTRATION,
                "both discovery strata must have positive populations",
            )
        masses = _count_pair(
            self.stratum_mass_numerators,
            name="stratum mass numerators",
        )
        denominator = _positive_int(
            self.stratum_mass_denominator,
            name="stratum mass denominator",
        )
        if masses != populations or denominator != sum(populations):
            _refuse(
                V1SampledVariableShellDW2SuffixRefusalCode.INVALID_REGISTRATION,
                "stratum masses must be the exact full-population masses",
            )
        consumed = _count_pair(
            self.consumed_d1_stratum_counts,
            name="consumed D1 stratum counts",
        )
        capacities = _count_pair(
            self.suffix_capacity_stratum_counts,
            name="suffix capacity stratum counts",
        )
        if any(consumed[i] > populations[i] for i in (0, 1)):
            _refuse(
                V1SampledVariableShellDW2SuffixRefusalCode.EXHAUSTED,
                "D1 counts exceed the parent discovery populations",
            )
        if capacities != tuple(populations[i] - consumed[i] for i in (0, 1)):
            _refuse(
                V1SampledVariableShellDW2SuffixRefusalCode.INVALID_REGISTRATION,
                "suffix capacities do not equal population minus D1 prefix",
            )
        if sum(capacities) <= 0:
            _refuse(
                V1SampledVariableShellDW2SuffixRefusalCode.EXHAUSTED,
                "D1 already exhausted both discovery permutations",
            )
        packet_size = _positive_int(
            self.transport_packet_size,
            name="transport packet size",
        )
        positive_flags = (
            self.transport_packet_size_is_hardware_only,
            self.predictable_same_discovery_permutation_suffix,
            self.cross_stratum_disjointness_verified,
        )
        closed_flags = (
            self.statistical_stopping_count_predeclared,
            self.outcome_payload_present,
            self.scoring_authority,
            self.statistical_stopping_authority,
            self.proposal_decision_authority,
            self.mask_authority,
            self.certification_calibration_oracle_authority,
            self.commit_terminal_or_pareto_authority,
            self.validation_rows_seen,
        )
        if (
            any(type(value) is not bool for value in positive_flags + closed_flags)
            or any(value is not True for value in positive_flags)
            or any(value is not False for value in closed_flags)
            or self.algorithm_id != DW2_SUFFIX_ALGORITHM
            or self.authority_scope != DW2_SUFFIX_AUTHORITY_SCOPE
            or self.schema != DW2_SUFFIX_REGISTRATION_SCHEMA
        ):
            _refuse(
                V1SampledVariableShellDW2SuffixRefusalCode.AUTHORITY_VIOLATION,
                "D_W2 registration weakened its outcome or authority firewall",
            )
        object.__setattr__(self, "generation_index", generation)
        object.__setattr__(self, "population_stratum_counts", populations)
        object.__setattr__(self, "stratum_mass_numerators", masses)
        object.__setattr__(self, "stratum_mass_denominator", denominator)
        object.__setattr__(self, "consumed_d1_stratum_counts", consumed)
        object.__setattr__(self, "suffix_capacity_stratum_counts", capacities)
        object.__setattr__(self, "transport_packet_size", packet_size)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "_receipt_hash"
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    @property
    def stratum_masses(self) -> tuple[Fraction, Fraction]:
        return (
            Fraction(self.stratum_mass_numerators[0], self.stratum_mass_denominator),
            Fraction(self.stratum_mass_numerators[1], self.stratum_mass_denominator),
        )

    @property
    def maximum_packet_count(self) -> int:
        return max(
            (count + self.transport_packet_size - 1) // self.transport_packet_size
            for count in self.suffix_capacity_stratum_counts
        )

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_PREFIX + "registration-")
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1SampledVariableShellDW2SuffixRefusalCode.TAMPERED,
                "D_W2 suffix registration was mutated after issuance",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledVariableShellDW2SuffixRefusalCode.STALE_IDENTITY,
                "D_W2 suffix registration hash differs from the expected parent",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    def assert_parent_graph(
        self,
        *,
        frame: V1AlexNetFQCausalGenerationFrameReceipt,
        discovery_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt,
    ) -> None:
        frame, discovery_stream = _assert_exact_parent_graph(
            frame=frame,
            discovery_stream=discovery_stream,
        )
        target = discovery_stream.target_assignment_receipt.generated_order
        complement = discovery_stream.complement_assignment_receipt.generated_order
        d1_counts = self.consumed_d1_stratum_counts
        d1_target = target[: d1_counts[0]]
        d1_complement = complement[: d1_counts[1]]
        suffix_target = target[d1_counts[0] :]
        suffix_complement = complement[d1_counts[1] :]
        if (
            self.generation_frame_hash != frame.receipt_hash
            or self.discovery_stream_hash != discovery_stream.receipt_hash
            or self.registration_hash != frame.registration_hash
            or self.generation_index != frame.generation_index
            or self.current_mask_hash != frame.current_mask_hash
            or self.frozen_transform_hash != frame.frozen_transform_hash
            or self.target_assignment_receipt_hash
            != discovery_stream.target_assignment_receipt.content_hash
            or self.complement_assignment_receipt_hash
            != discovery_stream.complement_assignment_receipt.content_hash
            or self.target_population_membership_hash
            != discovery_stream.target_population_membership_hash
            or self.complement_population_membership_hash
            != discovery_stream.complement_population_membership_hash
            or self.population_stratum_counts
            != discovery_stream.full_population_stratum_counts
            or self.d1_target_prefix_axis_hash != _axis_hash("D1-target", d1_target)
            or self.d1_complement_prefix_axis_hash
            != _axis_hash("D1-complement", d1_complement)
            or self.d1_pair_prefix_axis_hash
            != _pair_axis_hash(
                role="D1",
                target=d1_target,
                complement=d1_complement,
            )
            or self.target_suffix_axis_hash
            != _axis_hash("D-W2-target-suffix", suffix_target)
            or self.complement_suffix_axis_hash
            != _axis_hash("D-W2-complement-suffix", suffix_complement)
            or self.suffix_pair_axis_hash
            != _pair_axis_hash(
                role="D-W2-suffix",
                target=suffix_target,
                complement=suffix_complement,
            )
            or self.suffix_capacity_stratum_counts
            != (len(suffix_target), len(suffix_complement))
        ):
            _refuse(
                V1SampledVariableShellDW2SuffixRefusalCode.STALE_IDENTITY,
                "D_W2 suffix registration no longer matches its exact D parent",
            )


@dataclass(frozen=True, slots=True)
class V1SampledVariableShellDW2SuffixPacket:
    """One deterministic paired transport slice of the registered suffix."""

    suffix_registration_hash: str
    generation_frame_hash: str
    discovery_stream_hash: str
    packet_ordinal: int
    before_suffix_stratum_counts: tuple[int, int]
    after_suffix_stratum_counts: tuple[int, int]
    target_context_ids: tuple[str, ...]
    complement_context_ids: tuple[str, ...]
    target_packet_axis_hash: str
    complement_packet_axis_hash: str
    packet_pair_axis_hash: str
    transport_packet_size: int
    predictable_prefix_slice: bool = True
    packet_is_hardware_transport_atom_only: bool = True
    outcome_payload_present: bool = False
    scoring_authority: bool = False
    statistical_stopping_authority: bool = False
    decision_or_commit_authority: bool = False
    schema: str = DW2_SUFFIX_PACKET_SCHEMA
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "suffix_registration_hash",
            "generation_frame_hash",
            "discovery_stream_hash",
            "target_packet_axis_hash",
            "complement_packet_axis_hash",
            "packet_pair_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        ordinal = _nonnegative_int(self.packet_ordinal, name="packet ordinal")
        before = _count_pair(
            self.before_suffix_stratum_counts,
            name="packet before counts",
        )
        after = _count_pair(
            self.after_suffix_stratum_counts,
            name="packet after counts",
        )
        target = _identity_axis(self.target_context_ids, name="target packet axis")
        complement = _identity_axis(
            self.complement_context_ids,
            name="complement packet axis",
        )
        if set(target).intersection(complement):
            _refuse(
                V1SampledVariableShellDW2SuffixRefusalCode.OVERLAP,
                "one D_W2 packet repeats a context across strata",
            )
        packet_size = _positive_int(
            self.transport_packet_size,
            name="packet transport size",
        )
        if (
            after != (before[0] + len(target), before[1] + len(complement))
            or sum(len(axis) for axis in (target, complement)) <= 0
            or any(len(axis) > packet_size for axis in (target, complement))
            or self.target_packet_axis_hash
            != _axis_hash(f"D-W2-target-packet-{ordinal}", target)
            or self.complement_packet_axis_hash
            != _axis_hash(f"D-W2-complement-packet-{ordinal}", complement)
            or self.packet_pair_axis_hash
            != _pair_axis_hash(
                role=f"D-W2-packet-{ordinal}",
                target=target,
                complement=complement,
            )
        ):
            _refuse(
                V1SampledVariableShellDW2SuffixRefusalCode.ORDER_MISMATCH,
                "D_W2 packet is not a coherent bounded suffix extension",
            )
        positive_flags = (
            self.predictable_prefix_slice,
            self.packet_is_hardware_transport_atom_only,
        )
        closed_flags = (
            self.outcome_payload_present,
            self.scoring_authority,
            self.statistical_stopping_authority,
            self.decision_or_commit_authority,
        )
        if (
            any(type(value) is not bool for value in positive_flags + closed_flags)
            or any(value is not True for value in positive_flags)
            or any(value is not False for value in closed_flags)
            or self.schema != DW2_SUFFIX_PACKET_SCHEMA
        ):
            _refuse(
                V1SampledVariableShellDW2SuffixRefusalCode.AUTHORITY_VIOLATION,
                "D_W2 packet crossed its transport-only authority boundary",
            )
        object.__setattr__(self, "packet_ordinal", ordinal)
        object.__setattr__(self, "before_suffix_stratum_counts", before)
        object.__setattr__(self, "after_suffix_stratum_counts", after)
        object.__setattr__(self, "target_context_ids", target)
        object.__setattr__(self, "complement_context_ids", complement)
        object.__setattr__(self, "transport_packet_size", packet_size)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "_receipt_hash"
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    @property
    def context_ids_by_stratum(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return (
            ("target", self.target_context_ids),
            ("complement", self.complement_context_ids),
        )

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_PREFIX + "packet-")
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1SampledVariableShellDW2SuffixRefusalCode.TAMPERED,
                "D_W2 suffix packet was mutated after issuance",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledVariableShellDW2SuffixRefusalCode.ORDER_MISMATCH,
                "D_W2 suffix packet does not match the active predictable slice",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1SampledVariableShellDW2SuffixPrefixFreezeReceipt:
    """Authority-inert identity of the externally stopped ``D_W2`` prefix."""

    suffix_registration_hash: str
    generation_frame_hash: str
    discovery_stream_hash: str
    generation_index: int
    provisional_shell_hash: str
    provisional_action_family_hash: str
    consumed_d1_stratum_counts: tuple[int, int]
    consumed_dw2_stratum_counts: tuple[int, int]
    final_d_stratum_counts: tuple[int, int]
    population_stratum_counts: tuple[int, int]
    dw2_target_prefix_axis_hash: str
    dw2_complement_prefix_axis_hash: str
    dw2_pair_prefix_axis_hash: str
    final_d_target_prefix_axis_hash: str
    final_d_complement_prefix_axis_hash: str
    final_d_pair_prefix_axis_hash: str
    completed_packet_hashes: tuple[str, ...]
    census_complete_by_stratum: tuple[bool, bool]
    exact_full_discovery_census: bool
    statistical_stop_was_external: bool = True
    transport_packet_size_was_not_a_stopping_target: bool = True
    no_outcome_payload_custody: bool = True
    scoring_authority: bool = False
    statistical_stopping_authority: bool = False
    proposal_decision_authority: bool = False
    mask_or_commit_authority: bool = False
    certification_calibration_oracle_authority: bool = False
    terminal_or_pareto_authority: bool = False
    validation_rows_seen: bool = False
    authority_scope: str = DW2_SUFFIX_AUTHORITY_SCOPE
    schema: str = DW2_SUFFIX_PREFIX_FREEZE_SCHEMA
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "suffix_registration_hash",
            "generation_frame_hash",
            "discovery_stream_hash",
            "provisional_shell_hash",
            "provisional_action_family_hash",
            "dw2_target_prefix_axis_hash",
            "dw2_complement_prefix_axis_hash",
            "dw2_pair_prefix_axis_hash",
            "final_d_target_prefix_axis_hash",
            "final_d_complement_prefix_axis_hash",
            "final_d_pair_prefix_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        generation = _nonnegative_int(self.generation_index, name="generation index")
        d1 = _count_pair(self.consumed_d1_stratum_counts, name="freeze D1 counts")
        dw2 = _count_pair(self.consumed_dw2_stratum_counts, name="freeze D_W2 counts")
        final = _count_pair(self.final_d_stratum_counts, name="freeze final D counts")
        populations = _count_pair(
            self.population_stratum_counts,
            name="freeze population counts",
        )
        if (
            final != (d1[0] + dw2[0], d1[1] + dw2[1])
            or any(final[i] > populations[i] for i in (0, 1))
            or sum(dw2) <= 0
        ):
            _refuse(
                V1SampledVariableShellDW2SuffixRefusalCode.INVALID_REGISTRATION,
                "D_W2 freeze counts do not form a nonempty bounded D extension",
            )
        packet_hashes = tuple(
            _identity(value, name="completed packet hash")
            for value in self.completed_packet_hashes
        )
        if not packet_hashes or len(packet_hashes) != len(set(packet_hashes)):
            _refuse(
                V1SampledVariableShellDW2SuffixRefusalCode.INVALID_REGISTRATION,
                "D_W2 freeze requires unique completed packet identities",
            )
        census = tuple(self.census_complete_by_stratum)
        expected_census = tuple(final[i] == populations[i] for i in (0, 1))
        if (
            len(census) != 2
            or any(type(value) is not bool for value in census)
            or census != expected_census
            or type(self.exact_full_discovery_census) is not bool
            or self.exact_full_discovery_census is not all(census)
        ):
            _refuse(
                V1SampledVariableShellDW2SuffixRefusalCode.INVALID_REGISTRATION,
                "D_W2 freeze census flags do not match the exact population caps",
            )
        positive_flags = (
            self.statistical_stop_was_external,
            self.transport_packet_size_was_not_a_stopping_target,
            self.no_outcome_payload_custody,
        )
        closed_flags = (
            self.scoring_authority,
            self.statistical_stopping_authority,
            self.proposal_decision_authority,
            self.mask_or_commit_authority,
            self.certification_calibration_oracle_authority,
            self.terminal_or_pareto_authority,
            self.validation_rows_seen,
        )
        if (
            any(type(value) is not bool for value in positive_flags + closed_flags)
            or any(value is not True for value in positive_flags)
            or any(value is not False for value in closed_flags)
            or self.authority_scope != DW2_SUFFIX_AUTHORITY_SCOPE
            or self.schema != DW2_SUFFIX_PREFIX_FREEZE_SCHEMA
        ):
            _refuse(
                V1SampledVariableShellDW2SuffixRefusalCode.AUTHORITY_VIOLATION,
                "D_W2 prefix freeze crossed its identity-only authority boundary",
            )
        object.__setattr__(self, "generation_index", generation)
        object.__setattr__(self, "consumed_d1_stratum_counts", d1)
        object.__setattr__(self, "consumed_dw2_stratum_counts", dw2)
        object.__setattr__(self, "final_d_stratum_counts", final)
        object.__setattr__(self, "population_stratum_counts", populations)
        object.__setattr__(self, "completed_packet_hashes", packet_hashes)
        object.__setattr__(self, "census_complete_by_stratum", census)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "_receipt_hash"
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_PREFIX + "freeze-")
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1SampledVariableShellDW2SuffixRefusalCode.TAMPERED,
                "D_W2 suffix-prefix freeze receipt was mutated after issuance",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledVariableShellDW2SuffixRefusalCode.STALE_IDENTITY,
                "D_W2 suffix-prefix freeze receipt hash differs from expectation",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


def register_v1_sampled_variable_shell_dw2_suffix(
    *,
    frame: V1AlexNetFQCausalGenerationFrameReceipt,
    discovery_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt,
    consumed_d1_stratum_counts: tuple[int, int],
    provisional_shell_hash: str,
    provisional_action_family_hash: str,
    transport_packet_size: int,
) -> V1SampledVariableShellDW2SuffixRegistration:
    """Bind the remaining predictable suffix of the exact current D stream."""

    frame, discovery_stream = _assert_exact_parent_graph(
        frame=frame,
        discovery_stream=discovery_stream,
    )
    d1_counts = _count_pair(
        consumed_d1_stratum_counts,
        name="consumed D1 stratum counts",
    )
    target = discovery_stream.target_assignment_receipt.generated_order
    complement = discovery_stream.complement_assignment_receipt.generated_order
    populations = (len(target), len(complement))
    if any(d1_counts[index] > populations[index] for index in (0, 1)):
        _refuse(
            V1SampledVariableShellDW2SuffixRefusalCode.EXHAUSTED,
            "D1 prefix counts exceed the exact discovery population",
        )
    d1_target = target[: d1_counts[0]]
    d1_complement = complement[: d1_counts[1]]
    suffix_target = target[d1_counts[0] :]
    suffix_complement = complement[d1_counts[1] :]
    if sum(len(axis) for axis in (suffix_target, suffix_complement)) <= 0:
        _refuse(
            V1SampledVariableShellDW2SuffixRefusalCode.EXHAUSTED,
            "D1 already exhausted the complete discovery stream",
        )
    result = V1SampledVariableShellDW2SuffixRegistration(
        generation_frame_hash=frame.receipt_hash,
        discovery_stream_hash=discovery_stream.receipt_hash,
        registration_hash=frame.registration_hash,
        generation_index=frame.generation_index,
        current_mask_hash=frame.current_mask_hash,
        frozen_transform_hash=frame.frozen_transform_hash,
        target_assignment_receipt_hash=(
            discovery_stream.target_assignment_receipt.content_hash
        ),
        complement_assignment_receipt_hash=(
            discovery_stream.complement_assignment_receipt.content_hash
        ),
        target_population_membership_hash=(
            discovery_stream.target_population_membership_hash
        ),
        complement_population_membership_hash=(
            discovery_stream.complement_population_membership_hash
        ),
        population_stratum_counts=populations,
        stratum_mass_numerators=populations,
        stratum_mass_denominator=sum(populations),
        consumed_d1_stratum_counts=d1_counts,
        d1_target_prefix_axis_hash=_axis_hash("D1-target", d1_target),
        d1_complement_prefix_axis_hash=_axis_hash("D1-complement", d1_complement),
        d1_pair_prefix_axis_hash=_pair_axis_hash(
            role="D1",
            target=d1_target,
            complement=d1_complement,
        ),
        target_suffix_axis_hash=_axis_hash("D-W2-target-suffix", suffix_target),
        complement_suffix_axis_hash=_axis_hash(
            "D-W2-complement-suffix", suffix_complement
        ),
        suffix_pair_axis_hash=_pair_axis_hash(
            role="D-W2-suffix",
            target=suffix_target,
            complement=suffix_complement,
        ),
        suffix_capacity_stratum_counts=(len(suffix_target), len(suffix_complement)),
        provisional_shell_hash=_identity(
            provisional_shell_hash,
            name="provisional shell hash",
        ),
        provisional_action_family_hash=_identity(
            provisional_action_family_hash,
            name="provisional action family hash",
        ),
        transport_packet_size=transport_packet_size,
    )
    result.assert_parent_graph(frame=frame, discovery_stream=discovery_stream)
    return result


class V1SampledVariableShellDW2SuffixCursor:
    """Nonserializable append-only reveal cursor over one registered suffix."""

    __slots__ = (
        "_active_packet",
        "_completed_packet_hashes",
        "_consumed",
        "_discovery_stream",
        "_frame",
        "_freeze_receipt",
        "_lock",
        "_registration",
    )

    def __init__(
        self,
        *,
        registration: V1SampledVariableShellDW2SuffixRegistration,
        frame: V1AlexNetFQCausalGenerationFrameReceipt,
        discovery_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt,
    ) -> None:
        if type(registration) is not V1SampledVariableShellDW2SuffixRegistration:
            _refuse(
                V1SampledVariableShellDW2SuffixRefusalCode.INVALID_TYPE,
                "D_W2 cursor requires the exact suffix registration type",
            )
        registration.assert_integrity()
        registration.assert_parent_graph(
            frame=frame,
            discovery_stream=discovery_stream,
        )
        self._registration = registration
        self._frame = frame
        self._discovery_stream = discovery_stream
        self._consumed = [0, 0]
        self._completed_packet_hashes: list[str] = []
        self._active_packet: V1SampledVariableShellDW2SuffixPacket | None = None
        self._freeze_receipt: (
            V1SampledVariableShellDW2SuffixPrefixFreezeReceipt | None
        ) = None
        self._lock = threading.RLock()

    @property
    def registration(self) -> V1SampledVariableShellDW2SuffixRegistration:
        return self._registration

    @property
    def consumed_dw2_stratum_counts(self) -> tuple[int, int]:
        with self._lock:
            return self._consumed[0], self._consumed[1]

    @property
    def final_d_stratum_counts(self) -> tuple[int, int]:
        with self._lock:
            d1 = self._registration.consumed_d1_stratum_counts
            return d1[0] + self._consumed[0], d1[1] + self._consumed[1]

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return (
                tuple(self._consumed)
                == self._registration.suffix_capacity_stratum_counts
            )

    @property
    def frozen(self) -> bool:
        with self._lock:
            return self._freeze_receipt is not None

    def _assert_live_parent(self) -> None:
        self._registration.assert_integrity()
        self._registration.assert_parent_graph(
            frame=self._frame,
            discovery_stream=self._discovery_stream,
        )
        if self._freeze_receipt is not None:
            _refuse(
                V1SampledVariableShellDW2SuffixRefusalCode.CLOSED,
                "D_W2 cursor is closed by its final prefix freeze",
            )

    def begin_next_packet(self) -> V1SampledVariableShellDW2SuffixPacket:
        """Reveal the next paired hardware atom of both remaining strata."""

        with self._lock:
            self._assert_live_parent()
            if self._active_packet is not None:
                _refuse(
                    V1SampledVariableShellDW2SuffixRefusalCode.ACTIVE_PACKET,
                    "one D_W2 suffix packet is already active",
                )
            capacities = self._registration.suffix_capacity_stratum_counts
            if tuple(self._consumed) == capacities:
                _refuse(
                    V1SampledVariableShellDW2SuffixRefusalCode.EXHAUSTED,
                    "the complete registered D_W2 suffix was consumed",
                )
            d1 = self._registration.consumed_d1_stratum_counts
            packet_size = self._registration.transport_packet_size
            orders = (
                self._discovery_stream.target_assignment_receipt.generated_order,
                self._discovery_stream.complement_assignment_receipt.generated_order,
            )
            axes: list[tuple[str, ...]] = []
            after: list[int] = []
            for index in (0, 1):
                start = self._consumed[index]
                stop = min(capacities[index], start + packet_size)
                axes.append(orders[index][d1[index] + start : d1[index] + stop])
                after.append(stop)
            ordinal = len(self._completed_packet_hashes)
            packet = V1SampledVariableShellDW2SuffixPacket(
                suffix_registration_hash=self._registration.receipt_hash,
                generation_frame_hash=self._frame.receipt_hash,
                discovery_stream_hash=self._discovery_stream.receipt_hash,
                packet_ordinal=ordinal,
                before_suffix_stratum_counts=tuple(self._consumed),  # type: ignore[arg-type]
                after_suffix_stratum_counts=tuple(after),  # type: ignore[arg-type]
                target_context_ids=axes[0],
                complement_context_ids=axes[1],
                target_packet_axis_hash=_axis_hash(
                    f"D-W2-target-packet-{ordinal}", axes[0]
                ),
                complement_packet_axis_hash=_axis_hash(
                    f"D-W2-complement-packet-{ordinal}", axes[1]
                ),
                packet_pair_axis_hash=_pair_axis_hash(
                    role=f"D-W2-packet-{ordinal}",
                    target=axes[0],
                    complement=axes[1],
                ),
                transport_packet_size=packet_size,
            )
            self._active_packet = packet
            return packet

    def accept_active_packet(
        self,
        packet: V1SampledVariableShellDW2SuffixPacket,
        *,
        outcome_payload: object | None = None,
    ) -> None:
        """Advance custody without accepting or retaining any score outcome."""

        with self._lock:
            self._assert_live_parent()
            if outcome_payload is not None:
                _refuse(
                    V1SampledVariableShellDW2SuffixRefusalCode.OUTCOME_LEAKAGE,
                    "D_W2 suffix custody cannot ingest scorer outcomes",
                )
            if self._active_packet is None:
                _refuse(
                    V1SampledVariableShellDW2SuffixRefusalCode.NO_ACTIVE_PACKET,
                    "no D_W2 suffix packet is active",
                )
            if type(packet) is not V1SampledVariableShellDW2SuffixPacket:
                _refuse(
                    V1SampledVariableShellDW2SuffixRefusalCode.INVALID_TYPE,
                    "D_W2 packet acceptance requires the exact packet type",
                )
            packet.assert_integrity(self._active_packet.receipt_hash)
            if packet != self._active_packet:
                _refuse(
                    V1SampledVariableShellDW2SuffixRefusalCode.ORDER_MISMATCH,
                    "D_W2 packet differs from the active predictable slice",
                )
            self._consumed[:] = packet.after_suffix_stratum_counts
            self._completed_packet_hashes.append(packet.receipt_hash)
            self._active_packet = None

    def freeze_suffix_prefix(
        self,
        *,
        outcome_payload: object | None = None,
        requested_decision_authority: bool = False,
    ) -> V1SampledVariableShellDW2SuffixPrefixFreezeReceipt:
        """Close at the externally selected packet boundary, without judging it."""

        with self._lock:
            if outcome_payload is not None:
                _refuse(
                    V1SampledVariableShellDW2SuffixRefusalCode.OUTCOME_LEAKAGE,
                    "D_W2 prefix freeze cannot retain scorer outcomes",
                )
            if type(requested_decision_authority) is not bool:
                _refuse(
                    V1SampledVariableShellDW2SuffixRefusalCode.INVALID_TYPE,
                    "requested decision authority must have exact bool type",
                )
            if requested_decision_authority:
                _refuse(
                    V1SampledVariableShellDW2SuffixRefusalCode.AUTHORITY_VIOLATION,
                    "D_W2 suffix-prefix freeze cannot authorize a decision",
                )
            self._assert_live_parent()
            if self._active_packet is not None:
                _refuse(
                    V1SampledVariableShellDW2SuffixRefusalCode.ACTIVE_PACKET,
                    "finish the active D_W2 packet before freezing its prefix",
                )
            if not self._completed_packet_hashes:
                _refuse(
                    V1SampledVariableShellDW2SuffixRefusalCode.EMPTY_SUFFIX,
                    "D_W2 suffix-prefix freeze requires at least one complete packet",
                )
            d1 = self._registration.consumed_d1_stratum_counts
            dw2 = tuple(self._consumed)
            final = (d1[0] + dw2[0], d1[1] + dw2[1])
            target = self._discovery_stream.target_assignment_receipt.generated_order
            complement = (
                self._discovery_stream.complement_assignment_receipt.generated_order
            )
            dw2_target = target[d1[0] : final[0]]
            dw2_complement = complement[d1[1] : final[1]]
            final_target = target[: final[0]]
            final_complement = complement[: final[1]]
            census = tuple(
                final[index] == self._registration.population_stratum_counts[index]
                for index in (0, 1)
            )
            receipt = V1SampledVariableShellDW2SuffixPrefixFreezeReceipt(
                suffix_registration_hash=self._registration.receipt_hash,
                generation_frame_hash=self._frame.receipt_hash,
                discovery_stream_hash=self._discovery_stream.receipt_hash,
                generation_index=self._registration.generation_index,
                provisional_shell_hash=self._registration.provisional_shell_hash,
                provisional_action_family_hash=(
                    self._registration.provisional_action_family_hash
                ),
                consumed_d1_stratum_counts=d1,
                consumed_dw2_stratum_counts=dw2,  # type: ignore[arg-type]
                final_d_stratum_counts=final,
                population_stratum_counts=(
                    self._registration.population_stratum_counts
                ),
                dw2_target_prefix_axis_hash=_axis_hash(
                    "D-W2-target-prefix-freeze", dw2_target
                ),
                dw2_complement_prefix_axis_hash=_axis_hash(
                    "D-W2-complement-prefix-freeze", dw2_complement
                ),
                dw2_pair_prefix_axis_hash=_pair_axis_hash(
                    role="D-W2-prefix-freeze",
                    target=dw2_target,
                    complement=dw2_complement,
                ),
                final_d_target_prefix_axis_hash=_axis_hash(
                    "final-D-target-prefix", final_target
                ),
                final_d_complement_prefix_axis_hash=_axis_hash(
                    "final-D-complement-prefix", final_complement
                ),
                final_d_pair_prefix_axis_hash=_pair_axis_hash(
                    role="final-D-prefix",
                    target=final_target,
                    complement=final_complement,
                ),
                completed_packet_hashes=tuple(self._completed_packet_hashes),
                census_complete_by_stratum=census,  # type: ignore[arg-type]
                exact_full_discovery_census=all(census),
            )
            self._freeze_receipt = receipt
            return receipt

    def __reduce__(self) -> NoReturn:
        raise TypeError("live D_W2 suffix cursors are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live D_W2 suffix cursors are nonserializable")

    def __getstate__(self) -> NoReturn:
        raise TypeError("live D_W2 suffix cursors are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("live D_W2 suffix cursors cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("live D_W2 suffix cursors cannot be copied")


def open_v1_sampled_variable_shell_dw2_suffix_cursor(
    *,
    registration: V1SampledVariableShellDW2SuffixRegistration,
    frame: V1AlexNetFQCausalGenerationFrameReceipt,
    discovery_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt,
) -> V1SampledVariableShellDW2SuffixCursor:
    """Open the unique live reveal cursor for an exact registered parent graph."""

    return V1SampledVariableShellDW2SuffixCursor(
        registration=registration,
        frame=frame,
        discovery_stream=discovery_stream,
    )


__all__ = [
    "DW2_SUFFIX_ALGORITHM",
    "DW2_SUFFIX_AUTHORITY_SCOPE",
    "DW2_SUFFIX_PACKET_SCHEMA",
    "DW2_SUFFIX_PREFIX_FREEZE_SCHEMA",
    "DW2_SUFFIX_REGISTRATION_SCHEMA",
    "V1SampledVariableShellDW2SuffixCursor",
    "V1SampledVariableShellDW2SuffixPacket",
    "V1SampledVariableShellDW2SuffixPrefixFreezeReceipt",
    "V1SampledVariableShellDW2SuffixRefusal",
    "V1SampledVariableShellDW2SuffixRefusalCode",
    "V1SampledVariableShellDW2SuffixRegistration",
    "open_v1_sampled_variable_shell_dw2_suffix_cursor",
    "register_v1_sampled_variable_shell_dw2_suffix",
]
