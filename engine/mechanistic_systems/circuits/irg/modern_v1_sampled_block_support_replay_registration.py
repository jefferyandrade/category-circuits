"""Array-free registrations for replaying the sampled ``D_fixed`` anchor.

Block-support coefficients must be reconstructed while corrected-A and
literal-J rows are physically replayable, but the proposal may have been fit
from several independently normalized compact sources: conditioned ``D1``
first and then one source per registered residual-fit packet.  This module
closes that registration seam without pretending those physical sources share
one capture/VJP identity.

Each physical replay registration keeps its source-local provenance and uses
the exact conditional-mean coordinate ``1 / n_s``.  A separate structural
registration concatenates the exact anchor-influencing axes and uses ``1 / d``
over all ``D_fixed`` contexts.  It replaces the five source-local identities
with ordered composite identities, while all scientific chart, route, shell,
native-fiber, KKT, resource, alpha, and population identities remain common.
No row, tensor, coefficient, statistical statement, or downstream authority
is created here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from fractions import Fraction
from numbers import Integral
from typing import Final, NoReturn, Sequence

from .contracts import content_hash
from .modern_v1_sampled_full_population_compact_source import (
    V1SampledAnchorInfluencingAxisReceipt,
)
from .modern_v1_signed_work_variable_shell_source_cuda import (
    V1SignedWorkVariableShellSourceRegistration,
)


SAMPLED_BLOCK_SUPPORT_REPLAY_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_sampled_block_support_replay_registration_receipt_v1"
)
SAMPLED_BLOCK_SUPPORT_REPLAY_PLAN_SCHEMA: Final = (
    "partition_free_v1_sampled_block_support_replay_plan_v1"
)
SAMPLED_BLOCK_SUPPORT_REPLAY_REGISTRATION_AUTHORITY: Final = (
    "array_free_D_fixed_physical_and_structural_replay_registration_only_no_"
    "rows_coefficients_statistics_action_C_K_O_commit_terminal_or_pareto_"
    "authority_v1"
)
SAMPLED_BLOCK_SUPPORT_REPLAY_REGISTRATION_ALGORITHM: Final = (
    "conditioned_D1_then_ordered_residual_packet_exact_axis_replay_with_"
    "segment_conditional_means_and_one_canonical_structural_mean_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-block-support-replay-registration-"
_SOURCE_LOCAL_FIELDS: Final = (
    "parent_d_source_receipt_hash",
    "witness_stream_source_hash",
    "same_capture_vjp_source_hash",
    "corrected_evidence_stream_source_hash",
    "literal_j_stream_source_hash",
)
_AXIS_FIELDS: Final = (
    "registered_context_ids",
    "registered_draw_ids",
    "registered_measurement_ids",
    "registered_capture_ids",
    "registered_source_packet_hashes",
)
_ANCHOR_AXIS_FIELDS: Final = (
    "context_ids",
    "draw_ids",
    "measurement_ids",
    "capture_ids",
    "source_packet_hashes",
)
_SCIENTIFIC_FIELDS: Final = (
    "model_identity_hash",
    "chart_identity_hash",
    "current_mask_hash",
    "proposal_generation",
    "route_axis_hash",
    "route_id_axis_hash",
    "shell_freeze_receipt_hash",
    "shell_indices",
    "shell_route_ids",
    "native_fiber_axis_receipt_hash",
    "native_fiber_measure_hash",
    "kkt_branch_id",
    "kkt_branch_source_receipt_hash",
    "resource_axis_hash",
    "resource_source_receipt_hash",
    "alpha_ledger_id",
    "alpha_coordinate_id",
    "alpha_lineage_hash",
    "population_count",
    "population_context_axis_hash",
    "population_draw_axis_hash",
)
_ISSUER_TOKEN: Final = object()


class V1SampledBlockSupportReplayRegistrationRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_IDENTITY = "stale_identity"
    SEGMENT_MISMATCH = "segment_mismatch"
    AXIS_MISMATCH = "axis_mismatch"
    SCIENTIFIC_IDENTITY_MISMATCH = "scientific_identity_mismatch"
    PREFIX_MISMATCH = "prefix_mismatch"
    WEIGHT_MISMATCH = "weight_mismatch"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SampledBlockSupportReplayRegistrationRefusal(RuntimeError):
    """Typed fail-closed boundary for sampled block-support replay plans."""

    def __init__(
        self,
        code: V1SampledBlockSupportReplayRegistrationRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledBlockSupportReplayRegistrationRefusalCode:
            raise TypeError("block-support replay refusal code has the wrong type")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledBlockSupportReplayRegistrationRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledBlockSupportReplayRegistrationRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledBlockSupportReplayRegistrationRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SampledBlockSupportReplayRegistrationRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SampledBlockSupportReplayRegistrationRefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _identity_axis(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not tuple or not value:
        _refuse(
            V1SampledBlockSupportReplayRegistrationRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty immutable string axis",
        )
    result = tuple(_identity(item, name=f"{name} item") for item in value)
    if len(result) != len(set(result)):
        _refuse(
            V1SampledBlockSupportReplayRegistrationRefusalCode.AXIS_MISMATCH,
            f"{name} must be unique",
        )
    return result


def _string_axis(value: object, *, name: str) -> tuple[str, ...]:
    """Validate an ordered identity axis whose values may intentionally repeat."""

    if type(value) is not tuple or not value:
        _refuse(
            V1SampledBlockSupportReplayRegistrationRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty immutable string axis",
        )
    return tuple(_identity(item, name=f"{name} item") for item in value)


def _positive_count_axis(value: object, *, name: str) -> tuple[int, ...]:
    if type(value) is not tuple or not value:
        _refuse(
            V1SampledBlockSupportReplayRegistrationRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty immutable count axis",
        )
    result = tuple(_nonnegative_int(item, name=f"{name} item") for item in value)
    if any(item <= 0 for item in result):
        _refuse(
            V1SampledBlockSupportReplayRegistrationRefusalCode.SEGMENT_MISMATCH,
            f"{name} must be strictly positive",
        )
    return result


def _fraction_axis(value: object, *, name: str) -> tuple[tuple[int, int], ...]:
    if type(value) is not tuple or not value:
        _refuse(
            V1SampledBlockSupportReplayRegistrationRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty immutable rational axis",
        )
    result: list[tuple[int, int]] = []
    for item in value:
        if (
            type(item) is not tuple
            or len(item) != 2
            or any(
                isinstance(part, bool) or not isinstance(part, Integral)
                for part in item
            )
            or int(item[0]) <= 0
            or int(item[1]) <= 0
        ):
            _refuse(
                V1SampledBlockSupportReplayRegistrationRefusalCode.WEIGHT_MISMATCH,
                f"{name} must contain positive rational pairs",
            )
        fraction = Fraction(int(item[0]), int(item[1]))
        result.append((fraction.numerator, fraction.denominator))
    return tuple(result)


def _ordered_composite_identity(
    *,
    role: str,
    anchor_receipt_hash: str,
    template_registration_hashes: Sequence[str],
    source_identities: Sequence[str],
) -> str:
    return content_hash(
        {
            "schema": (
                "partition_free_v1_sampled_block_support_ordered_source_" "identity_v1"
            ),
            "role": role,
            "anchor_influencing_axis_receipt_hash": anchor_receipt_hash,
            "template_registration_hashes": tuple(template_registration_hashes),
            "source_identities": tuple(source_identities),
            "source_local_identity_may_differ": True,
            "ordered_D1_then_residual_packets": True,
        },
        prefix=_PREFIX + role + "-composite-",
    )


@dataclass(frozen=True, slots=True)
class V1SampledBlockSupportReplayRegistrationReceipt:
    anchor_influencing_axis_receipt_hash: str
    full_population_compact_source_receipt_hash: str
    packet_registration_receipt_hashes: tuple[str, ...]
    residual_measure_design_hash: str
    final_residual_weight_receipt_hash: str
    template_registration_hashes: tuple[str, ...]
    physical_registration_hashes: tuple[str, ...]
    structural_registration_hash: str
    segment_roles: tuple[str, ...]
    segment_context_counts: tuple[int, ...]
    segment_uniform_weight_fractions: tuple[tuple[int, int], ...]
    structural_context_count: int
    structural_uniform_weight_fraction: tuple[int, int]
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    shell_freeze_receipt_hash: str
    shell_indices: tuple[int, ...]
    shell_route_ids: tuple[str, ...]
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    kkt_branch_id: str
    kkt_branch_source_receipt_hash: str
    resource_axis_hash: str
    resource_source_receipt_hash: str
    alpha_ledger_id: str
    alpha_coordinate_id: str
    alpha_lineage_hash: str
    population_count: int
    population_context_axis_hash: str
    population_draw_axis_hash: str
    conditioned_d1_registered_prefix_hash: str
    d1_pair_prefix_axis_hash: str
    residual_registered_dw2_prefix_hash: str
    final_planned_dw2_prefix_hash: str
    template_registered_d_prefix_hashes: tuple[str, ...]
    template_registered_dw2_prefix_hashes: tuple[str, ...]
    parent_d_source_receipt_hashes: tuple[str, ...]
    witness_stream_source_hashes: tuple[str, ...]
    same_capture_vjp_source_hashes: tuple[str, ...]
    corrected_evidence_stream_source_hashes: tuple[str, ...]
    literal_j_stream_source_hashes: tuple[str, ...]
    composite_parent_d_source_receipt_hash: str
    composite_witness_stream_source_hash: str
    composite_same_capture_vjp_source_hash: str
    composite_corrected_evidence_stream_source_hash: str
    composite_literal_j_stream_source_hash: str
    context_axis_hash: str
    draw_axis_hash: str
    measurement_axis_hash: str
    capture_axis_hash: str
    source_packet_axis_hash: str
    conditioned_d1_first: bool = True
    residual_packets_in_v4_order: bool = True
    physical_axes_exactly_partition_anchor_axis: bool = True
    physical_registrations_are_uniform_conditional_means: bool = True
    structural_registration_is_uniform_D_fixed_mean: bool = True
    source_local_identity_difference_permitted_and_order_bound: bool = True
    common_scientific_identity_replayed: bool = True
    array_free: bool = True
    raw_rows_present: bool = False
    tensor_payload_present: bool = False
    coefficient_payload_present: bool = False
    outcome_payload_present: bool = False
    statistical_authority: bool = False
    confidence_authority: bool = False
    action_authority: bool = False
    solution_sensitivity_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = SAMPLED_BLOCK_SUPPORT_REPLAY_REGISTRATION_ALGORITHM
    authority_scope: str = SAMPLED_BLOCK_SUPPORT_REPLAY_REGISTRATION_AUTHORITY
    schema: str = SAMPLED_BLOCK_SUPPORT_REPLAY_REGISTRATION_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)
    _bound_receipt_hash: str | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._issuance_token is not _ISSUER_TOKEN:
            _refuse(
                V1SampledBlockSupportReplayRegistrationRefusalCode.INVALID_TYPE,
                "block-support replay receipts must be registrar-issued",
            )
        identity_names = (
            "anchor_influencing_axis_receipt_hash",
            "full_population_compact_source_receipt_hash",
            "residual_measure_design_hash",
            "final_residual_weight_receipt_hash",
            "structural_registration_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "shell_freeze_receipt_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "kkt_branch_id",
            "kkt_branch_source_receipt_hash",
            "resource_axis_hash",
            "resource_source_receipt_hash",
            "alpha_ledger_id",
            "alpha_coordinate_id",
            "alpha_lineage_hash",
            "population_context_axis_hash",
            "population_draw_axis_hash",
            "conditioned_d1_registered_prefix_hash",
            "d1_pair_prefix_axis_hash",
            "residual_registered_dw2_prefix_hash",
            "final_planned_dw2_prefix_hash",
            "composite_parent_d_source_receipt_hash",
            "composite_witness_stream_source_hash",
            "composite_same_capture_vjp_source_hash",
            "composite_corrected_evidence_stream_source_hash",
            "composite_literal_j_stream_source_hash",
            "context_axis_hash",
            "draw_axis_hash",
            "measurement_axis_hash",
            "capture_axis_hash",
            "source_packet_axis_hash",
        )
        for name in identity_names:
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        unique_axis_names = (
            "packet_registration_receipt_hashes",
            "template_registration_hashes",
            "physical_registration_hashes",
            "segment_roles",
        )
        repeatable_axis_names = (
            "template_registered_d_prefix_hashes",
            "template_registered_dw2_prefix_hashes",
            "parent_d_source_receipt_hashes",
            "witness_stream_source_hashes",
            "same_capture_vjp_source_hashes",
            "corrected_evidence_stream_source_hashes",
            "literal_j_stream_source_hashes",
        )
        axes = {}
        for name in unique_axis_names:
            axis = _identity_axis(getattr(self, name), name=name)
            axes[name] = axis
            object.__setattr__(self, name, axis)
        for name in repeatable_axis_names:
            axis = _string_axis(getattr(self, name), name=name)
            axes[name] = axis
            object.__setattr__(self, name, axis)
        counts = _positive_count_axis(
            self.segment_context_counts,
            name="segment context counts",
        )
        fractions = _fraction_axis(
            self.segment_uniform_weight_fractions,
            name="segment uniform weights",
        )
        object.__setattr__(self, "segment_context_counts", counts)
        object.__setattr__(self, "segment_uniform_weight_fractions", fractions)
        segment_count = len(counts)
        if (
            segment_count < 2
            or len(self.packet_registration_receipt_hashes) != segment_count - 1
            or any(
                len(axes[name]) != segment_count
                for name in unique_axis_names + repeatable_axis_names
                if name != "packet_registration_receipt_hashes"
            )
            or len(fractions) != segment_count
            or fractions != tuple((1, count) for count in counts)
            or self.segment_roles
            != ("conditioned_D1",)
            + tuple(f"residual_fit:{index}" for index in range(segment_count - 1))
        ):
            _refuse(
                V1SampledBlockSupportReplayRegistrationRefusalCode.SEGMENT_MISMATCH,
                "physical segment identities, roles, counts, and exact means do not align",
            )
        structural_count = _nonnegative_int(
            self.structural_context_count,
            name="structural context count",
        )
        structural_fraction = _fraction_axis(
            (self.structural_uniform_weight_fraction,),
            name="structural uniform weight",
        )[0]
        if (
            structural_count <= 0
            or structural_count != sum(counts)
            or structural_fraction != (1, structural_count)
        ):
            _refuse(
                V1SampledBlockSupportReplayRegistrationRefusalCode.WEIGHT_MISMATCH,
                "structural registration must carry the exact uniform D_fixed mean",
            )
        object.__setattr__(self, "structural_context_count", structural_count)
        object.__setattr__(
            self,
            "structural_uniform_weight_fraction",
            structural_fraction,
        )
        for name in ("proposal_generation", "population_count"):
            value = _nonnegative_int(getattr(self, name), name=name)
            if name == "population_count" and value <= 0:
                _refuse(
                    V1SampledBlockSupportReplayRegistrationRefusalCode.SCIENTIFIC_IDENTITY_MISMATCH,
                    "population count must be positive",
                )
            object.__setattr__(self, name, value)
        if (
            type(self.shell_indices) is not tuple
            or not self.shell_indices
            or any(
                isinstance(item, bool)
                or not isinstance(item, Integral)
                or int(item) < 0
                for item in self.shell_indices
            )
            or len(self.shell_indices) != len(set(self.shell_indices))
        ):
            _refuse(
                V1SampledBlockSupportReplayRegistrationRefusalCode.SCIENTIFIC_IDENTITY_MISMATCH,
                "shell indices must be one unique immutable nonnegative axis",
            )
        shell_ids = _identity_axis(self.shell_route_ids, name="shell route IDs")
        if len(shell_ids) != len(self.shell_indices):
            _refuse(
                V1SampledBlockSupportReplayRegistrationRefusalCode.SCIENTIFIC_IDENTITY_MISMATCH,
                "shell route IDs do not align with shell indices",
            )
        object.__setattr__(
            self, "shell_indices", tuple(int(x) for x in self.shell_indices)
        )
        object.__setattr__(self, "shell_route_ids", shell_ids)
        template_hashes = self.template_registration_hashes
        for role, source_axis, composite in (
            (
                "parent_d_source_receipt",
                self.parent_d_source_receipt_hashes,
                self.composite_parent_d_source_receipt_hash,
            ),
            (
                "witness_stream_source",
                self.witness_stream_source_hashes,
                self.composite_witness_stream_source_hash,
            ),
            (
                "same_capture_vjp_source",
                self.same_capture_vjp_source_hashes,
                self.composite_same_capture_vjp_source_hash,
            ),
            (
                "corrected_evidence_stream_source",
                self.corrected_evidence_stream_source_hashes,
                self.composite_corrected_evidence_stream_source_hash,
            ),
            (
                "literal_j_stream_source",
                self.literal_j_stream_source_hashes,
                self.composite_literal_j_stream_source_hash,
            ),
        ):
            if (
                _ordered_composite_identity(
                    role=role,
                    anchor_receipt_hash=self.anchor_influencing_axis_receipt_hash,
                    template_registration_hashes=template_hashes,
                    source_identities=source_axis,
                )
                != composite
            ):
                _refuse(
                    V1SampledBlockSupportReplayRegistrationRefusalCode.STALE_IDENTITY,
                    f"ordered {role} composite identity does not replay",
                )
        required_true = (
            self.conditioned_d1_first,
            self.residual_packets_in_v4_order,
            self.physical_axes_exactly_partition_anchor_axis,
            self.physical_registrations_are_uniform_conditional_means,
            self.structural_registration_is_uniform_D_fixed_mean,
            self.source_local_identity_difference_permitted_and_order_bound,
            self.common_scientific_identity_replayed,
            self.array_free,
        )
        required_false = (
            self.raw_rows_present,
            self.tensor_payload_present,
            self.coefficient_payload_present,
            self.outcome_payload_present,
            self.statistical_authority,
            self.confidence_authority,
            self.action_authority,
            self.solution_sensitivity_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            any(type(item) is not bool for item in required_true + required_false)
            or any(item is not True for item in required_true)
            or any(item is not False for item in required_false)
            or self.algorithm != SAMPLED_BLOCK_SUPPORT_REPLAY_REGISTRATION_ALGORITHM
            or self.authority_scope
            != SAMPLED_BLOCK_SUPPORT_REPLAY_REGISTRATION_AUTHORITY
            or self.schema != SAMPLED_BLOCK_SUPPORT_REPLAY_REGISTRATION_SCHEMA
        ):
            _refuse(
                V1SampledBlockSupportReplayRegistrationRefusalCode.AUTHORITY_VIOLATION,
                "block-support replay receipt crossed its registration-only boundary",
            )
        observed = self._compute_receipt_hash()
        if (
            self._bound_receipt_hash is not None
            and observed != self._bound_receipt_hash
        ):
            _refuse(
                V1SampledBlockSupportReplayRegistrationRefusalCode.STALE_IDENTITY,
                "block-support replay receipt changed after issuance",
            )
        object.__setattr__(self, "_bound_receipt_hash", observed)

    def _compute_receipt_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_PREFIX + "receipt-",
        )

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_receipt_hash()
        if observed != self._bound_receipt_hash or (
            expected_hash is not None and observed != expected_hash
        ):
            _refuse(
                V1SampledBlockSupportReplayRegistrationRefusalCode.STALE_IDENTITY,
                "block-support replay receipt failed its bound hash",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1SampledBlockSupportReplayRegistrationPlan:
    physical_segment_registrations: tuple[
        V1SignedWorkVariableShellSourceRegistration, ...
    ] = field(repr=False)
    structural_registration: V1SignedWorkVariableShellSourceRegistration = field(
        repr=False
    )
    receipt: V1SampledBlockSupportReplayRegistrationReceipt
    _issuance_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _ISSUER_TOKEN:
            _refuse(
                V1SampledBlockSupportReplayRegistrationRefusalCode.INVALID_TYPE,
                "block-support replay plans must be registrar-issued",
            )
        if (
            type(self.physical_segment_registrations) is not tuple
            or not self.physical_segment_registrations
            or any(
                type(item) is not V1SignedWorkVariableShellSourceRegistration
                for item in self.physical_segment_registrations
            )
            or type(self.structural_registration)
            is not V1SignedWorkVariableShellSourceRegistration
            or type(self.receipt) is not V1SampledBlockSupportReplayRegistrationReceipt
        ):
            _refuse(
                V1SampledBlockSupportReplayRegistrationRefusalCode.INVALID_TYPE,
                "replay plan contains a noncanonical registration or receipt",
            )
        receipt = self.receipt
        receipt.assert_integrity()
        if (
            tuple(
                item.registration_hash for item in self.physical_segment_registrations
            )
            != receipt.physical_registration_hashes
            or self.structural_registration.registration_hash
            != receipt.structural_registration_hash
        ):
            _refuse(
                V1SampledBlockSupportReplayRegistrationRefusalCode.STALE_IDENTITY,
                "replay plan registrations differ from its issued receipt",
            )


def register_v1_sampled_block_support_replay(
    *,
    anchor_influencing_axis: V1SampledAnchorInfluencingAxisReceipt,
    physical_registration_templates: Sequence[
        V1SignedWorkVariableShellSourceRegistration
    ],
) -> V1SampledBlockSupportReplayRegistrationPlan:
    """Issue physical-segment and canonical structural replay registrations."""

    if type(anchor_influencing_axis) is not V1SampledAnchorInfluencingAxisReceipt:
        raise TypeError("block-support replay requires the exact anchor axis receipt")
    anchor_influencing_axis.assert_integrity()
    if type(physical_registration_templates) not in {tuple, list}:
        raise TypeError("physical replay templates must be one ordered sequence")
    templates = tuple(physical_registration_templates)
    if not templates or any(
        type(item) is not V1SignedWorkVariableShellSourceRegistration
        for item in templates
    ):
        _refuse(
            V1SampledBlockSupportReplayRegistrationRefusalCode.INVALID_TYPE,
            "every physical replay template must be an exact source registration",
        )
    anchor = anchor_influencing_axis
    expected_counts = (anchor.d1_context_count,) + anchor.packet_context_counts
    if len(templates) != anchor.packet_count + 1 or len(templates) != len(
        expected_counts
    ):
        _refuse(
            V1SampledBlockSupportReplayRegistrationRefusalCode.SEGMENT_MISMATCH,
            "physical templates omit or add a D1/residual-fit segment",
        )
    template_hashes = tuple(item.registration_hash for item in templates)
    common = templates[0]
    for ordinal, template in enumerate(templates):
        if any(
            getattr(template, name) != getattr(common, name)
            for name in _SCIENTIFIC_FIELDS
        ):
            _refuse(
                V1SampledBlockSupportReplayRegistrationRefusalCode.SCIENTIFIC_IDENTITY_MISMATCH,
                f"physical segment {ordinal} changed a common scientific identity",
            )
    if (
        templates[0].registered_d_prefix_hash
        != anchor.conditioned_d1_registered_prefix_hash
        or templates[0].registered_d_w2_prefix_hash != anchor.d1_pair_prefix_axis_hash
        or any(
            item.registered_d_prefix_hash != anchor.d1_pair_prefix_axis_hash
            for item in templates[1:]
        )
        or templates[-1].registered_d_w2_prefix_hash
        != anchor.final_planned_dw2_prefix_hash
    ):
        _refuse(
            V1SampledBlockSupportReplayRegistrationRefusalCode.PREFIX_MISMATCH,
            "physical D1/residual templates do not replay the anchor prefix lineage",
        )
    offsets = [0]
    for count in expected_counts:
        offsets.append(offsets[-1] + count)
    if offsets[-1] != anchor.context_count:
        _refuse(
            V1SampledBlockSupportReplayRegistrationRefusalCode.SEGMENT_MISMATCH,
            "anchor packet counts do not close the structural context axis",
        )
    for axis_field, anchor_field in zip(_AXIS_FIELDS, _ANCHOR_AXIS_FIELDS, strict=True):
        anchor_axis = getattr(anchor, anchor_field)
        concatenated = tuple(
            value for template in templates for value in getattr(template, axis_field)
        )
        if concatenated != anchor_axis:
            _refuse(
                V1SampledBlockSupportReplayRegistrationRefusalCode.AXIS_MISMATCH,
                f"physical {axis_field} do not exactly partition the anchor axis",
            )
        for ordinal, template in enumerate(templates):
            if len(getattr(template, axis_field)) != expected_counts[ordinal]:
                _refuse(
                    V1SampledBlockSupportReplayRegistrationRefusalCode.AXIS_MISMATCH,
                    f"physical segment {ordinal} {axis_field} has the wrong extent",
                )

    physical: list[V1SignedWorkVariableShellSourceRegistration] = []
    roles = ("conditioned_D1",) + tuple(
        f"residual_fit:{index}" for index in range(anchor.packet_count)
    )
    for ordinal, (template, count, role) in enumerate(
        zip(templates, expected_counts, roles, strict=True)
    ):
        registration_id = content_hash(
            {
                "schema": "partition_free_v1_sampled_block_support_physical_replay_v1",
                "anchor_influencing_axis_receipt_hash": anchor.receipt_hash,
                "template_registration_hash": template.registration_hash,
                "segment_ordinal": ordinal,
                "segment_role": role,
                "conditional_mean_fraction": (1, count),
            },
            prefix=_PREFIX + "physical-",
        )
        center = float(Fraction(1, count))
        physical.append(
            replace(
                template,
                registration_id=registration_id,
                context_weights=(center,) * count,
            )
        )

    source_axes = {
        name: tuple(getattr(item, name) for item in templates)
        for name in _SOURCE_LOCAL_FIELDS
    }
    composite_source = {
        name: _ordered_composite_identity(
            role=name.removesuffix("_hash"),
            anchor_receipt_hash=anchor.receipt_hash,
            template_registration_hashes=template_hashes,
            source_identities=source_axes[name],
        )
        for name in _SOURCE_LOCAL_FIELDS
    }
    structural_count = anchor.context_count
    structural_id = content_hash(
        {
            "schema": "partition_free_v1_sampled_block_support_structural_replay_v1",
            "anchor_influencing_axis_receipt_hash": anchor.receipt_hash,
            "template_registration_hashes": template_hashes,
            "physical_registration_hashes": tuple(
                item.registration_hash for item in physical
            ),
            "structural_uniform_mean_fraction": (1, structural_count),
            "ordered_composite_source_identities": composite_source,
        },
        prefix=_PREFIX + "structural-",
    )
    structural = V1SignedWorkVariableShellSourceRegistration(
        registration_id=structural_id,
        model_identity_hash=common.model_identity_hash,
        chart_identity_hash=common.chart_identity_hash,
        current_mask_hash=common.current_mask_hash,
        proposal_generation=common.proposal_generation,
        route_axis_hash=common.route_axis_hash,
        route_id_axis_hash=common.route_id_axis_hash,
        registered_d_prefix_hash=anchor.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=anchor.residual_registered_dw2_prefix_hash,
        parent_d_source_receipt_hash=composite_source["parent_d_source_receipt_hash"],
        shell_freeze_receipt_hash=common.shell_freeze_receipt_hash,
        shell_indices=common.shell_indices,
        shell_route_ids=common.shell_route_ids,
        native_fiber_axis_receipt_hash=common.native_fiber_axis_receipt_hash,
        native_fiber_measure_hash=common.native_fiber_measure_hash,
        kkt_branch_id=common.kkt_branch_id,
        kkt_branch_source_receipt_hash=common.kkt_branch_source_receipt_hash,
        resource_axis_hash=common.resource_axis_hash,
        resource_source_receipt_hash=common.resource_source_receipt_hash,
        witness_stream_source_hash=composite_source["witness_stream_source_hash"],
        same_capture_vjp_source_hash=composite_source["same_capture_vjp_source_hash"],
        corrected_evidence_stream_source_hash=composite_source[
            "corrected_evidence_stream_source_hash"
        ],
        literal_j_stream_source_hash=composite_source["literal_j_stream_source_hash"],
        alpha_ledger_id=common.alpha_ledger_id,
        alpha_coordinate_id=common.alpha_coordinate_id,
        alpha_lineage_hash=common.alpha_lineage_hash,
        population_count=common.population_count,
        population_context_axis_hash=common.population_context_axis_hash,
        population_draw_axis_hash=common.population_draw_axis_hash,
        registered_context_ids=anchor.context_ids,
        registered_draw_ids=anchor.draw_ids,
        registered_measurement_ids=anchor.measurement_ids,
        registered_capture_ids=anchor.capture_ids,
        registered_source_packet_hashes=anchor.source_packet_hashes,
        context_weights=(float(Fraction(1, structural_count)),) * structural_count,
    )
    receipt = V1SampledBlockSupportReplayRegistrationReceipt(
        anchor_influencing_axis_receipt_hash=anchor.receipt_hash,
        full_population_compact_source_receipt_hash=(
            anchor.full_population_compact_source_receipt_hash
        ),
        packet_registration_receipt_hashes=(anchor.packet_registration_receipt_hashes),
        residual_measure_design_hash=anchor.residual_measure_design_hash,
        final_residual_weight_receipt_hash=(anchor.final_residual_weight_receipt_hash),
        template_registration_hashes=template_hashes,
        physical_registration_hashes=tuple(item.registration_hash for item in physical),
        structural_registration_hash=structural.registration_hash,
        segment_roles=roles,
        segment_context_counts=expected_counts,
        segment_uniform_weight_fractions=tuple((1, n) for n in expected_counts),
        structural_context_count=structural_count,
        structural_uniform_weight_fraction=(1, structural_count),
        model_identity_hash=common.model_identity_hash,
        chart_identity_hash=common.chart_identity_hash,
        current_mask_hash=common.current_mask_hash,
        proposal_generation=common.proposal_generation,
        route_axis_hash=common.route_axis_hash,
        route_id_axis_hash=common.route_id_axis_hash,
        shell_freeze_receipt_hash=common.shell_freeze_receipt_hash,
        shell_indices=common.shell_indices,
        shell_route_ids=common.shell_route_ids,
        native_fiber_axis_receipt_hash=common.native_fiber_axis_receipt_hash,
        native_fiber_measure_hash=common.native_fiber_measure_hash,
        kkt_branch_id=common.kkt_branch_id,
        kkt_branch_source_receipt_hash=common.kkt_branch_source_receipt_hash,
        resource_axis_hash=common.resource_axis_hash,
        resource_source_receipt_hash=common.resource_source_receipt_hash,
        alpha_ledger_id=common.alpha_ledger_id,
        alpha_coordinate_id=common.alpha_coordinate_id,
        alpha_lineage_hash=common.alpha_lineage_hash,
        population_count=common.population_count,
        population_context_axis_hash=common.population_context_axis_hash,
        population_draw_axis_hash=common.population_draw_axis_hash,
        conditioned_d1_registered_prefix_hash=(
            anchor.conditioned_d1_registered_prefix_hash
        ),
        d1_pair_prefix_axis_hash=anchor.d1_pair_prefix_axis_hash,
        residual_registered_dw2_prefix_hash=(
            anchor.residual_registered_dw2_prefix_hash
        ),
        final_planned_dw2_prefix_hash=anchor.final_planned_dw2_prefix_hash,
        template_registered_d_prefix_hashes=tuple(
            item.registered_d_prefix_hash for item in templates
        ),
        template_registered_dw2_prefix_hashes=tuple(
            item.registered_d_w2_prefix_hash for item in templates
        ),
        parent_d_source_receipt_hashes=source_axes["parent_d_source_receipt_hash"],
        witness_stream_source_hashes=source_axes["witness_stream_source_hash"],
        same_capture_vjp_source_hashes=source_axes["same_capture_vjp_source_hash"],
        corrected_evidence_stream_source_hashes=source_axes[
            "corrected_evidence_stream_source_hash"
        ],
        literal_j_stream_source_hashes=source_axes["literal_j_stream_source_hash"],
        composite_parent_d_source_receipt_hash=composite_source[
            "parent_d_source_receipt_hash"
        ],
        composite_witness_stream_source_hash=composite_source[
            "witness_stream_source_hash"
        ],
        composite_same_capture_vjp_source_hash=composite_source[
            "same_capture_vjp_source_hash"
        ],
        composite_corrected_evidence_stream_source_hash=composite_source[
            "corrected_evidence_stream_source_hash"
        ],
        composite_literal_j_stream_source_hash=composite_source[
            "literal_j_stream_source_hash"
        ],
        context_axis_hash=anchor.context_axis_hash,
        draw_axis_hash=anchor.draw_axis_hash,
        measurement_axis_hash=anchor.measurement_axis_hash,
        capture_axis_hash=anchor.capture_axis_hash,
        source_packet_axis_hash=anchor.source_packet_axis_hash,
        _issuance_token=_ISSUER_TOKEN,
    )
    return V1SampledBlockSupportReplayRegistrationPlan(
        physical_segment_registrations=tuple(physical),
        structural_registration=structural,
        receipt=receipt,
        _issuance_token=_ISSUER_TOKEN,
    )


__all__ = [
    "SAMPLED_BLOCK_SUPPORT_REPLAY_PLAN_SCHEMA",
    "SAMPLED_BLOCK_SUPPORT_REPLAY_REGISTRATION_ALGORITHM",
    "SAMPLED_BLOCK_SUPPORT_REPLAY_REGISTRATION_AUTHORITY",
    "SAMPLED_BLOCK_SUPPORT_REPLAY_REGISTRATION_SCHEMA",
    "V1SampledBlockSupportReplayRegistrationPlan",
    "V1SampledBlockSupportReplayRegistrationReceipt",
    "V1SampledBlockSupportReplayRegistrationRefusal",
    "V1SampledBlockSupportReplayRegistrationRefusalCode",
    "register_v1_sampled_block_support_replay",
]
