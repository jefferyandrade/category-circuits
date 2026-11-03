"""Exactly-one materialization of a receipt-only label-native runtime plan.

The pre-O runtime plan freezes an ordered fallback queue but deliberately
retains no provider handle or hard-mask payload.  This module is the narrow
issuer seam that selects one descriptor by its frozen index, replays it
against the matching live qualified provider and parent mask, and asks the
provider to materialize exactly that one packed child carrier.  No other
descriptor is materialized or even offered to the provider.

The returned proposal is identity-only input to later candidate handling.  It
does not open calibration K, paired oracle O, commit, terminal validation, or
Pareto authority.  Production requires the exact CUDA provider receipt named
by the plan.  A route-limited underscore factory exists only to exercise the
same compact-to-source and packed-action semantics on the CPU reference
backend in focused tests; its receipt explicitly denies exact production
provider replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
import hashlib
from numbers import Integral
from pathlib import Path
from typing import Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask

from .modern_v1_candidate_composition import V1PackedRemovalAction
from .modern_v1_signed_work_label_native_direction_provider import (
    V1LabelNativePackedDirectionProvider,
    V1LabelNativeQualifiedLaneRegistration,
)
from .modern_v1_signed_work_label_native_runtime_acquisition import (
    V1LabelNativeRuntimeFallbackDescriptor,
    V1LabelNativeRuntimeFallbackPlan,
)


LABEL_NATIVE_RUNTIME_SELECTED_MATERIALIZATION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_label_native_runtime_selected_materialization_receipt_v1"
)
LABEL_NATIVE_RUNTIME_SELECTED_PROPOSAL_SCHEMA: Final = (
    "partition_free_v1_label_native_runtime_selected_proposal_v1"
)
LABEL_NATIVE_RUNTIME_SELECTED_MATERIALIZATION_AUTHORITY: Final = (
    "plan_bound_exactly_one_label_native_packed_child_issuer_only_no_K_O_"
    "commit_terminal_or_pareto_authority_v1"
)

_RECEIPT_PREFIX: Final = (
    "modern-v1-label-native-runtime-selected-materialization-"
)
_QUEUE_AXIS_PREFIX: Final = "modern-v1-label-native-runtime-plan-queue-axis-"
_RECEIPT_ISSUER_TOKEN: Final = object()
_PROPOSAL_ISSUER_TOKEN: Final = object()


class V1LabelNativeRuntimeMaterializationRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    DESCRIPTOR_INDEX_MISMATCH = "descriptor_index_mismatch"
    STALE_LIVE_SOURCE = "stale_live_source"
    PARENT_MASK_MISMATCH = "parent_mask_mismatch"
    PROVIDER_IDENTITY_MISMATCH = "provider_identity_mismatch"
    QUALIFIED_MAPPING_MISMATCH = "qualified_mapping_mismatch"
    CUDA_REQUIRED = "cuda_required"
    REFERENCE_LIMIT_EXCEEDED = "reference_limit_exceeded"
    ACTION_IDENTITY_MISMATCH = "action_identity_mismatch"
    AUTHORITY_VIOLATION = "authority_violation"
    TAMPERED = "tampered"


class V1LabelNativeRuntimeMaterializationRefusal(RuntimeError):
    def __init__(
        self,
        code: V1LabelNativeRuntimeMaterializationRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1LabelNativeRuntimeMaterializationRefusalCode:
            raise TypeError("runtime-materialization refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1LabelNativeRuntimeMaterializationRefusalCode,
    message: str,
) -> NoReturn:
    raise V1LabelNativeRuntimeMaterializationRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1LabelNativeRuntimeMaterializationRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        _refuse(
            V1LabelNativeRuntimeMaterializationRefusalCode.INVALID_TYPE,
            f"{name} must be one exact positive integer",
        )
    return int(value)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        _refuse(
            V1LabelNativeRuntimeMaterializationRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    return int(value)


def _fresh_mask_hash(mask: HardMask, *, name: str) -> str:
    if type(mask) is not HardMask:
        _refuse(
            V1LabelNativeRuntimeMaterializationRefusalCode.INVALID_TYPE,
            f"{name} must be an exact HardMask",
        )
    return HardMask(
        route_axis_hash=mask.route_axis_hash,
        route_count=mask.route_count,
        packed_bits=mask.packed_bits,
    ).mask_hash


def label_native_runtime_materializer_source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class V1LabelNativeRuntimeSelectedMaterializationReceipt:
    runtime_plan_hash: str
    runtime_descriptor_hash: str
    descriptor_index: int
    queue_descriptor_count: int
    ordered_descriptor_axis_hash: str
    campaign_identity_hash: str
    campaign_contract_hash: str
    campaign_registration_hash: str
    live_chart_identity_hash: str
    live_current_mask_hash: str
    live_route_axis_hash: str
    live_route_count: int
    live_proposal_generation: int
    selected_coordinate_hash: str
    canonical_physical_identity_hash: str
    crosswalk_receipt_hash: str
    source_action_receipt_hash: str
    planned_provider_receipt_hash: str
    live_provider_receipt_hash: str
    provider_design_hash: str
    provider_policy_hash: str
    provider_forecast_hash: str
    qualified_lane_registration_hash: str
    compact_provider_endpoint_label: int
    source_endpoint_label: int
    selected_endpoint_count: int
    selected_endpoint_action_hash: str
    selected_carrier_receipt_hash: str
    resulting_mask_packed_sha256: str
    resulting_mask_hash: str
    removed_route_index_hash: str
    packed_action_hash: str
    backend_kind: str
    device_identity: str
    packed_payload_d2h_bytes: int
    raw_label_d2h_bytes: int
    route_scale_label_or_index_d2h_bytes: int
    materializer_source_hash: str
    production_cuda_execution: bool
    private_reference_execution: bool
    exact_planned_provider_receipt_replayed: bool
    qualified_compact_to_source_mapping_replayed: bool
    exact_parent_and_live_frame_replayed: bool
    exactly_one_endpoint_materialized: bool
    complete_queue_materialized: bool
    unselected_endpoint_materialization_count: int
    proposal_only: bool
    paired_bce_oracle_still_required: bool
    calibration_K_open_authority: bool
    paired_oracle_O_open_authority: bool
    irreversible_decision_authority: bool
    mask_commit_authority: bool
    terminal_authority: bool
    pareto_authority: bool
    execution_authority: bool
    authority_scope: str
    schema: str = LABEL_NATIVE_RUNTIME_SELECTED_MATERIALIZATION_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __init__(self, *, _issuer_token: object, **values: object) -> None:
        if _issuer_token is not _RECEIPT_ISSUER_TOKEN:
            raise TypeError("selected materialization receipts are issuer-only")
        expected = {
            item.name for item in fields(type(self)) if item.init and item.name != "schema"
        }
        if set(values) != expected:
            raise TypeError(
                "selected materialization receipt fields differ: "
                f"missing={sorted(expected.difference(values))}, "
                f"extra={sorted(set(values).difference(expected))}"
            )
        for name in expected:
            object.__setattr__(self, name, values[name])
        object.__setattr__(
            self,
            "schema",
            LABEL_NATIVE_RUNTIME_SELECTED_MATERIALIZATION_RECEIPT_SCHEMA,
        )
        object.__setattr__(self, "_receipt_hash", None)
        self.__post_init__()

    def __post_init__(self) -> None:
        for name in (
            "runtime_plan_hash",
            "runtime_descriptor_hash",
            "ordered_descriptor_axis_hash",
            "campaign_identity_hash",
            "campaign_contract_hash",
            "campaign_registration_hash",
            "live_chart_identity_hash",
            "live_current_mask_hash",
            "live_route_axis_hash",
            "selected_coordinate_hash",
            "canonical_physical_identity_hash",
            "crosswalk_receipt_hash",
            "source_action_receipt_hash",
            "planned_provider_receipt_hash",
            "live_provider_receipt_hash",
            "provider_design_hash",
            "provider_policy_hash",
            "provider_forecast_hash",
            "qualified_lane_registration_hash",
            "selected_endpoint_action_hash",
            "selected_carrier_receipt_hash",
            "resulting_mask_packed_sha256",
            "resulting_mask_hash",
            "removed_route_index_hash",
            "packed_action_hash",
            "backend_kind",
            "device_identity",
            "materializer_source_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        index = _nonnegative_int(self.descriptor_index, name="descriptor index")
        queue_count = _positive_int(
            self.queue_descriptor_count,
            name="queue descriptor count",
        )
        if index >= queue_count:
            _refuse(
                V1LabelNativeRuntimeMaterializationRefusalCode.DESCRIPTOR_INDEX_MISMATCH,
                "selected descriptor index is outside the frozen queue",
            )
        for name in (
            "live_route_count",
            "compact_provider_endpoint_label",
            "source_endpoint_label",
            "selected_endpoint_count",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "live_proposal_generation",
            _nonnegative_int(
                self.live_proposal_generation,
                name="live proposal generation",
            ),
        )
        object.__setattr__(
            self,
            "packed_payload_d2h_bytes",
            _nonnegative_int(
                self.packed_payload_d2h_bytes,
                name="packed payload D2H bytes",
            ),
        )
        for name in (
            "raw_label_d2h_bytes",
            "route_scale_label_or_index_d2h_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "unselected_endpoint_materialization_count",
            _nonnegative_int(
                self.unselected_endpoint_materialization_count,
                name="unselected endpoint materialization count",
            ),
        )
        digest = self.resulting_mask_packed_sha256
        if len(digest) != 64 or digest != digest.lower():
            _refuse(
                V1LabelNativeRuntimeMaterializationRefusalCode.ACTION_IDENTITY_MISMATCH,
                "selected materialization packed digest is not lowercase SHA-256",
            )
        try:
            int(digest, 16)
        except ValueError as error:
            raise V1LabelNativeRuntimeMaterializationRefusal(
                V1LabelNativeRuntimeMaterializationRefusalCode.ACTION_IDENTITY_MISMATCH,
                "selected materialization packed digest is not lowercase SHA-256",
            ) from error
        required_true = (
            self.qualified_compact_to_source_mapping_replayed,
            self.exact_parent_and_live_frame_replayed,
            self.exactly_one_endpoint_materialized,
            self.proposal_only,
            self.paired_bce_oracle_still_required,
        )
        required_false = (
            self.complete_queue_materialized,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.irreversible_decision_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
            self.execution_authority,
        )
        if (
            any(
                type(value) is not bool
                for value in (
                    self.production_cuda_execution,
                    self.private_reference_execution,
                    self.exact_planned_provider_receipt_replayed,
                    *required_true,
                    *required_false,
                )
            )
            or self.production_cuda_execution is self.private_reference_execution
            or self.exact_planned_provider_receipt_replayed
            is not self.production_cuda_execution
            or (
                self.exact_planned_provider_receipt_replayed
                and self.planned_provider_receipt_hash
                != self.live_provider_receipt_hash
            )
            or not all(required_true)
            or any(required_false)
            or self.unselected_endpoint_materialization_count != 0
            or self.raw_label_d2h_bytes != 0
            or self.route_scale_label_or_index_d2h_bytes != 0
            or self.packed_payload_d2h_bytes
            != (
                (self.live_route_count + 7) // 8
                if self.production_cuda_execution
                else 0
            )
            or (
                self.backend_kind == "torch_cuda"
                and not self.production_cuda_execution
            )
            or (
                self.backend_kind == "numpy_reference"
                and not self.private_reference_execution
            )
            or self.backend_kind not in {"torch_cuda", "numpy_reference"}
            or self.authority_scope
            != LABEL_NATIVE_RUNTIME_SELECTED_MATERIALIZATION_AUTHORITY
            or self.schema
            != LABEL_NATIVE_RUNTIME_SELECTED_MATERIALIZATION_RECEIPT_SCHEMA
        ):
            _refuse(
                V1LabelNativeRuntimeMaterializationRefusalCode.AUTHORITY_VIOLATION,
                "selected materialization crossed its exactly-one proposal authority",
            )
        object.__setattr__(self, "descriptor_index", index)
        object.__setattr__(self, "queue_descriptor_count", queue_count)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(type(self))
            if not item.name.startswith("_")
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_RECEIPT_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1LabelNativeRuntimeMaterializationRefusalCode.TAMPERED,
                "selected materialization receipt was mutated after issuance",
            )
        if expected_hash is not None and observed != _identity(
            expected_hash,
            name="expected selected materialization receipt hash",
        ):
            _refuse(
                V1LabelNativeRuntimeMaterializationRefusalCode.TAMPERED,
                "selected materialization receipt hash differs",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def __reduce__(self) -> NoReturn:
        raise TypeError("selected materialization receipts are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("selected materialization receipts are nonserializable")


@dataclass(frozen=True, slots=True, init=False)
class V1LabelNativeRuntimeSelectedEndpointProposal:
    receipt: V1LabelNativeRuntimeSelectedMaterializationReceipt
    parent_mask: HardMask = field(repr=False, compare=False)
    packed_action: V1PackedRemovalAction = field(repr=False, compare=False)
    schema: str = LABEL_NATIVE_RUNTIME_SELECTED_PROPOSAL_SCHEMA

    def __init__(self, *, _issuer_token: object, **values: object) -> None:
        if _issuer_token is not _PROPOSAL_ISSUER_TOKEN:
            raise TypeError("selected endpoint proposals are issuer-only")
        expected = {
            item.name for item in fields(type(self)) if item.init and item.name != "schema"
        }
        if set(values) != expected:
            raise TypeError("selected endpoint proposal fields differ")
        for name in expected:
            object.__setattr__(self, name, values[name])
        object.__setattr__(
            self,
            "schema",
            LABEL_NATIVE_RUNTIME_SELECTED_PROPOSAL_SCHEMA,
        )
        self.__post_init__()

    def __post_init__(self) -> None:
        if type(self.receipt) is not V1LabelNativeRuntimeSelectedMaterializationReceipt:
            raise TypeError("selected endpoint proposal requires its exact receipt")
        if type(self.parent_mask) is not HardMask:
            raise TypeError("selected endpoint proposal requires an exact parent mask")
        if type(self.packed_action) is not V1PackedRemovalAction:
            raise TypeError("selected endpoint proposal requires one exact packed action")
        receipt = self.receipt
        receipt.assert_integrity()
        parent_hash = _fresh_mask_hash(self.parent_mask, name="proposal parent mask")
        child = self.packed_action.resulting_mask
        child_hash = _fresh_mask_hash(child, name="proposal resulting mask")
        self.parent_mask.assert_same_axis(child)
        removed_count, removed_hash = child.removed_index_count_and_hash_from(
            self.parent_mask
        )
        if (
            parent_hash != receipt.live_current_mask_hash
            or self.parent_mask.route_axis_hash != receipt.live_route_axis_hash
            or self.parent_mask.route_count != receipt.live_route_count
            or child_hash != receipt.resulting_mask_hash
            or self.packed_action.action_hash != receipt.packed_action_hash
            or self.packed_action.removed_count != receipt.selected_endpoint_count
            or self.packed_action.removed_count != removed_count
            or self.packed_action.removed_route_index_hash != removed_hash
            or removed_hash != receipt.removed_route_index_hash
            or hashlib.sha256(child.packed_bits).hexdigest()
            != receipt.resulting_mask_packed_sha256
            or self.schema != LABEL_NATIVE_RUNTIME_SELECTED_PROPOSAL_SCHEMA
        ):
            _refuse(
                V1LabelNativeRuntimeMaterializationRefusalCode.ACTION_IDENTITY_MISMATCH,
                "selected endpoint proposal no longer replays its packed action",
            )

    @property
    def proposal_hash(self) -> str:
        return self.receipt.receipt_hash

    @property
    def resulting_mask(self) -> HardMask:
        return self.packed_action.resulting_mask

    @property
    def resulting_mask_hash(self) -> str:
        return self.receipt.resulting_mask_hash

    @property
    def packed_action_hash(self) -> str:
        return self.receipt.packed_action_hash

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.__post_init__()
        if expected_hash is not None:
            self.receipt.assert_integrity(expected_hash)
        return self.proposal_hash

    def assert_can_open_calibration_K(self) -> NoReturn:
        raise PermissionError("selected endpoint proposals cannot open calibration K")

    def assert_can_open_paired_oracle_O(self) -> NoReturn:
        raise PermissionError("selected endpoint proposals cannot open paired oracle O")

    def assert_can_commit_mask(self) -> NoReturn:
        raise PermissionError("selected endpoint proposals cannot commit a hard mask")

    def __reduce__(self) -> NoReturn:
        raise TypeError("selected endpoint proposals are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("selected endpoint proposals are nonserializable")


def _validate_live_frame(
    *,
    plan: V1LabelNativeRuntimeFallbackPlan,
    parent_mask: HardMask,
    live_chart_identity_hash: str,
    live_current_mask_hash: str,
    live_route_axis_hash: str,
    live_route_count: int,
    live_proposal_generation: int,
) -> str:
    current_hash = _fresh_mask_hash(parent_mask, name="live parent mask")
    chart = _identity(live_chart_identity_hash, name="live chart identity")
    claimed_mask = _identity(live_current_mask_hash, name="live current-mask hash")
    axis = _identity(live_route_axis_hash, name="live route-axis hash")
    route_count = _positive_int(live_route_count, name="live route count")
    generation = _nonnegative_int(
        live_proposal_generation,
        name="live proposal generation",
    )
    if (
        chart != plan.live_chart_identity_hash
        or claimed_mask != plan.live_current_mask_hash
        or current_hash != claimed_mask
        or axis != plan.live_route_axis_hash
        or parent_mask.route_axis_hash != axis
        or route_count != plan.live_route_count
        or parent_mask.route_count != route_count
        or generation != plan.live_proposal_generation
    ):
        _refuse(
            V1LabelNativeRuntimeMaterializationRefusalCode.STALE_LIVE_SOURCE,
            "live chart, parent mask, route axis, or generation left the plan frame",
        )
    return current_hash


def _validate_provider_mapping(
    *,
    descriptor: V1LabelNativeRuntimeFallbackDescriptor,
    provider: V1LabelNativePackedDirectionProvider,
    production: bool,
) -> V1LabelNativeQualifiedLaneRegistration:
    provider._require_live()
    receipt = provider.receipt
    qualified = receipt.qualified_lane_registration
    if type(qualified) is not V1LabelNativeQualifiedLaneRegistration:
        _refuse(
            V1LabelNativeRuntimeMaterializationRefusalCode.QUALIFIED_MAPPING_MISMATCH,
            "selected materialization requires a live qualified provider",
        )
    try:
        qualified_hash = qualified.assert_integrity()
    except (TypeError, ValueError) as error:
        raise V1LabelNativeRuntimeMaterializationRefusal(
            V1LabelNativeRuntimeMaterializationRefusalCode.QUALIFIED_MAPPING_MISMATCH,
            f"qualified provider registration no longer replays: {error}",
        ) from error
    compact = descriptor.compact_provider_endpoint_label
    offset = compact - 1
    if (
        offset < 0
        or offset >= len(qualified.compact_provider_endpoint_labels)
        or qualified.compact_provider_endpoint_labels[offset] != compact
        or qualified.source_endpoint_labels[offset] != descriptor.source_endpoint_label
        or qualified.endpoint_counts[offset] != descriptor.cumulative_removed_count
        or qualified.endpoint_action_hashes[offset]
        != descriptor.source_action_receipt.packed_action_hash
        or qualified.endpoint_action_receipt_hashes[offset]
        != descriptor.source_action_receipt_hash
        or qualified.action_crosswalk_receipt_hashes[offset]
        != descriptor.crosswalk.receipt_hash
        or qualified.canonical_physical_identity_hashes[offset]
        != descriptor.canonical_physical_identity_hash
        or qualified_hash != descriptor.qualified_lane_registration_hash
        or qualified.branch_value != descriptor.branch.value
        or qualified.compact_C_bridge_branch_receipt_hash
        != descriptor.bridge_branch_receipt_hash
        or qualified.action_axis_registration_hash
        != descriptor.action_axis_registration_hash
        or qualified.source_endpoint_labels
        != receipt.executed_source_endpoint_labels
        or qualified.endpoint_counts != receipt.executed_endpoint_counts
        or qualified.endpoint_action_hashes
        != receipt.executed_endpoint_action_hashes
        or qualified.route_axis_hash != descriptor.live_route_axis_hash
        or qualified.route_count != descriptor.live_route_count
        or qualified.parent_mask_hash != descriptor.live_current_mask_hash
    ):
        _refuse(
            V1LabelNativeRuntimeMaterializationRefusalCode.QUALIFIED_MAPPING_MISMATCH,
            "live compact-to-source lane differs from the selected descriptor",
        )
    if production:
        if (
            receipt.backend_kind != "torch_cuda"
            or receipt.receipt_hash != descriptor.provider_receipt_hash
            or receipt.design_hash != descriptor.provider_design_hash
            or receipt.policy_hash != descriptor.provider_policy_hash
            or receipt.forecast_hash != descriptor.provider_forecast_hash
            or receipt.device_identity != descriptor.provider_device_identity
            or receipt.prefix_rank_receipt_hash
            != descriptor.prefix_rank_receipt_hash
            or receipt.endpoint_query_hash != descriptor.endpoint_query_hash
            or receipt.resulting_label_commitment
            != descriptor.resulting_label_commitment
        ):
            _refuse(
                V1LabelNativeRuntimeMaterializationRefusalCode.PROVIDER_IDENTITY_MISMATCH,
                "live production provider differs from the plan-bound provider",
            )
    elif (
        receipt.backend_kind != "numpy_reference"
        or receipt.route_count > provider.policy.private_cpu_route_limit
    ):
        _refuse(
            V1LabelNativeRuntimeMaterializationRefusalCode.REFERENCE_LIMIT_EXCEEDED,
            "private reference materialization exceeds its route/backend boundary",
        )
    return qualified


def _materialize_selected(
    *,
    plan: V1LabelNativeRuntimeFallbackPlan,
    descriptor_index: int,
    provider: V1LabelNativePackedDirectionProvider,
    parent_mask: HardMask,
    live_chart_identity_hash: str,
    live_current_mask_hash: str,
    live_route_axis_hash: str,
    live_route_count: int,
    live_proposal_generation: int,
    production: bool,
) -> V1LabelNativeRuntimeSelectedEndpointProposal:
    if type(plan) is not V1LabelNativeRuntimeFallbackPlan:
        raise TypeError("selected materialization requires the exact runtime plan")
    if type(provider) is not V1LabelNativePackedDirectionProvider:
        raise TypeError("selected materialization requires the exact live provider")
    if type(production) is not bool:
        raise TypeError("selected materialization production flag must be exact")
    try:
        plan_hash = plan.assert_integrity()
    except (TypeError, ValueError) as error:
        raise V1LabelNativeRuntimeMaterializationRefusal(
            V1LabelNativeRuntimeMaterializationRefusalCode.TAMPERED,
            f"runtime plan no longer replays: {error}",
        ) from error
    index = _nonnegative_int(descriptor_index, name="descriptor index")
    if index >= len(plan.descriptors):
        _refuse(
            V1LabelNativeRuntimeMaterializationRefusalCode.DESCRIPTOR_INDEX_MISMATCH,
            "selected descriptor index is outside the frozen runtime plan",
        )
    descriptor = plan.descriptors[index]
    descriptor.assert_integrity(plan.ordered_descriptor_hashes[index])
    if production and provider.receipt.backend_kind != "torch_cuda":
        _refuse(
            V1LabelNativeRuntimeMaterializationRefusalCode.CUDA_REQUIRED,
            "production selected materialization requires the exact CUDA provider",
        )
    current_hash = _validate_live_frame(
        plan=plan,
        parent_mask=parent_mask,
        live_chart_identity_hash=live_chart_identity_hash,
        live_current_mask_hash=live_current_mask_hash,
        live_route_axis_hash=live_route_axis_hash,
        live_route_count=live_route_count,
        live_proposal_generation=live_proposal_generation,
    )
    qualified = _validate_provider_mapping(
        descriptor=descriptor,
        provider=provider,
        production=production,
    )
    receipt = provider.receipt
    if (
        receipt.parent_mask_hash != current_hash
        or receipt.route_axis_hash != plan.live_route_axis_hash
        or receipt.route_count != plan.live_route_count
    ):
        _refuse(
            V1LabelNativeRuntimeMaterializationRefusalCode.PARENT_MASK_MISMATCH,
            "live provider parent or route axis differs from the plan parent",
        )
    carrier = provider.materialize_selected_endpoint_packed_carrier(
        endpoint_label=descriptor.compact_provider_endpoint_label,
        endpoint_count=descriptor.cumulative_removed_count,
        expected_action_hash=descriptor.source_action_receipt.packed_action_hash,
    )
    carrier_receipt = carrier.receipt
    child = HardMask(
        route_axis_hash=plan.live_route_axis_hash,
        route_count=plan.live_route_count,
        packed_bits=carrier.packed_child_bits,
    )
    packed_action = V1PackedRemovalAction.from_resulting_mask(
        parent_mask=parent_mask,
        resulting_mask=child,
    )
    if (
        carrier_receipt.endpoint_label != descriptor.source_endpoint_label
        or carrier_receipt.provider_receipt_hash != receipt.receipt_hash
        or carrier_receipt.design_hash != receipt.design_hash
        or carrier_receipt.prefix_rank_receipt_hash
        != receipt.prefix_rank_receipt_hash
        or carrier_receipt.endpoint_query_hash != receipt.endpoint_query_hash
        or carrier_receipt.endpoint_count != descriptor.cumulative_removed_count
        or carrier_receipt.expected_action_hash
        != descriptor.source_action_receipt.packed_action_hash
        or carrier_receipt.source_action_receipt
        != descriptor.source_action_receipt
        or carrier_receipt.source_action_receipt.receipt_hash
        != descriptor.source_action_receipt_hash
        or carrier_receipt.action_axis_registration_hash
        != descriptor.action_axis_registration_hash
        or carrier_receipt.source_shell_ladder_receipt_hash
        != descriptor.source_shell_ladder_receipt_hash
        or carrier_receipt.parent_mask_hash != current_hash
        or carrier_receipt.raw_label_d2h_bytes != 0
        or carrier_receipt.resulting_mask_hash
        != descriptor.source_action_receipt.resulting_mask_hash
        or carrier_receipt.removed_route_index_hash
        != descriptor.source_action_receipt.removed_route_index_hash
        or packed_action.action_hash
        != descriptor.source_action_receipt.packed_action_hash
        or packed_action.action_hash != descriptor.coordinate.action_hash
        or packed_action.resulting_mask.mask_hash
        != descriptor.source_action_receipt.resulting_mask_hash
        or packed_action.removed_count != descriptor.cumulative_removed_count
        or packed_action.removed_route_index_hash
        != descriptor.source_action_receipt.removed_route_index_hash
    ):
        _refuse(
            V1LabelNativeRuntimeMaterializationRefusalCode.ACTION_IDENTITY_MISMATCH,
            "materialized packed child differs from the selected frozen action",
        )
    queue_axis_hash = content_hash(
        {
            "runtime_plan_hash": plan_hash,
            "ordered_descriptor_hashes": plan.ordered_descriptor_hashes,
            "ordered_coordinate_hashes": plan.ordered_coordinate_hashes,
            "ordered_physical_candidate_identity_hashes": (
                plan.ordered_physical_candidate_identity_hashes
            ),
        },
        prefix=_QUEUE_AXIS_PREFIX,
    )
    selected_receipt = V1LabelNativeRuntimeSelectedMaterializationReceipt(
        _issuer_token=_RECEIPT_ISSUER_TOKEN,
        runtime_plan_hash=plan_hash,
        runtime_descriptor_hash=descriptor.descriptor_hash,
        descriptor_index=index,
        queue_descriptor_count=len(plan.descriptors),
        ordered_descriptor_axis_hash=queue_axis_hash,
        campaign_identity_hash=plan.campaign_identity_hash,
        campaign_contract_hash=plan.campaign_contract_hash,
        campaign_registration_hash=plan.campaign_registration_hash,
        live_chart_identity_hash=plan.live_chart_identity_hash,
        live_current_mask_hash=current_hash,
        live_route_axis_hash=plan.live_route_axis_hash,
        live_route_count=plan.live_route_count,
        live_proposal_generation=plan.live_proposal_generation,
        selected_coordinate_hash=descriptor.coordinate_hash,
        canonical_physical_identity_hash=(
            descriptor.canonical_physical_identity_hash
        ),
        crosswalk_receipt_hash=descriptor.crosswalk.receipt_hash,
        source_action_receipt_hash=descriptor.source_action_receipt_hash,
        planned_provider_receipt_hash=descriptor.provider_receipt_hash,
        live_provider_receipt_hash=receipt.receipt_hash,
        provider_design_hash=receipt.design_hash,
        provider_policy_hash=receipt.policy_hash,
        provider_forecast_hash=receipt.forecast_hash,
        qualified_lane_registration_hash=qualified.registration_hash,
        compact_provider_endpoint_label=(
            descriptor.compact_provider_endpoint_label
        ),
        source_endpoint_label=descriptor.source_endpoint_label,
        selected_endpoint_count=descriptor.cumulative_removed_count,
        selected_endpoint_action_hash=packed_action.action_hash,
        selected_carrier_receipt_hash=carrier_receipt.receipt_hash,
        resulting_mask_packed_sha256=(
            carrier_receipt.resulting_mask_packed_sha256
        ),
        resulting_mask_hash=packed_action.resulting_mask.mask_hash,
        removed_route_index_hash=packed_action.removed_route_index_hash,
        packed_action_hash=packed_action.action_hash,
        backend_kind=receipt.backend_kind,
        device_identity=receipt.device_identity,
        packed_payload_d2h_bytes=carrier_receipt.packed_payload_d2h_bytes,
        raw_label_d2h_bytes=carrier_receipt.raw_label_d2h_bytes,
        route_scale_label_or_index_d2h_bytes=0,
        materializer_source_hash=label_native_runtime_materializer_source_hash(),
        production_cuda_execution=production,
        private_reference_execution=not production,
        exact_planned_provider_receipt_replayed=production,
        qualified_compact_to_source_mapping_replayed=True,
        exact_parent_and_live_frame_replayed=True,
        exactly_one_endpoint_materialized=True,
        complete_queue_materialized=False,
        unselected_endpoint_materialization_count=0,
        proposal_only=True,
        paired_bce_oracle_still_required=True,
        calibration_K_open_authority=False,
        paired_oracle_O_open_authority=False,
        irreversible_decision_authority=False,
        mask_commit_authority=False,
        terminal_authority=False,
        pareto_authority=False,
        execution_authority=False,
        authority_scope=LABEL_NATIVE_RUNTIME_SELECTED_MATERIALIZATION_AUTHORITY,
    )
    return V1LabelNativeRuntimeSelectedEndpointProposal(
        _issuer_token=_PROPOSAL_ISSUER_TOKEN,
        receipt=selected_receipt,
        parent_mask=parent_mask,
        packed_action=packed_action,
    )


def materialize_v1_label_native_runtime_selected_endpoint(
    *,
    plan: V1LabelNativeRuntimeFallbackPlan,
    descriptor_index: int,
    provider: V1LabelNativePackedDirectionProvider,
    parent_mask: HardMask,
    live_chart_identity_hash: str,
    live_current_mask_hash: str,
    live_route_axis_hash: str,
    live_route_count: int,
    live_proposal_generation: int,
) -> V1LabelNativeRuntimeSelectedEndpointProposal:
    """Materialize exactly one production CUDA endpoint from a frozen plan."""

    return _materialize_selected(
        plan=plan,
        descriptor_index=descriptor_index,
        provider=provider,
        parent_mask=parent_mask,
        live_chart_identity_hash=live_chart_identity_hash,
        live_current_mask_hash=live_current_mask_hash,
        live_route_axis_hash=live_route_axis_hash,
        live_route_count=live_route_count,
        live_proposal_generation=live_proposal_generation,
        production=True,
    )


def _materialize_v1_label_native_runtime_selected_endpoint_reference_for_tests(
    *,
    plan: V1LabelNativeRuntimeFallbackPlan,
    descriptor_index: int,
    provider: V1LabelNativePackedDirectionProvider,
    parent_mask: HardMask,
    live_chart_identity_hash: str,
    live_current_mask_hash: str,
    live_route_axis_hash: str,
    live_route_count: int,
    live_proposal_generation: int,
) -> V1LabelNativeRuntimeSelectedEndpointProposal:
    """Route-limited reference oracle; never a production fallback."""

    return _materialize_selected(
        plan=plan,
        descriptor_index=descriptor_index,
        provider=provider,
        parent_mask=parent_mask,
        live_chart_identity_hash=live_chart_identity_hash,
        live_current_mask_hash=live_current_mask_hash,
        live_route_axis_hash=live_route_axis_hash,
        live_route_count=live_route_count,
        live_proposal_generation=live_proposal_generation,
        production=False,
    )


__all__ = [
    "LABEL_NATIVE_RUNTIME_SELECTED_MATERIALIZATION_AUTHORITY",
    "LABEL_NATIVE_RUNTIME_SELECTED_MATERIALIZATION_RECEIPT_SCHEMA",
    "LABEL_NATIVE_RUNTIME_SELECTED_PROPOSAL_SCHEMA",
    "V1LabelNativeRuntimeMaterializationRefusal",
    "V1LabelNativeRuntimeMaterializationRefusalCode",
    "V1LabelNativeRuntimeSelectedEndpointProposal",
    "V1LabelNativeRuntimeSelectedMaterializationReceipt",
    "label_native_runtime_materializer_source_hash",
    "materialize_v1_label_native_runtime_selected_endpoint",
]
