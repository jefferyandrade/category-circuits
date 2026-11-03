"""Compact signed-work hard-F/Q custody for label-native C measurement.

Discovery stream D owns the two device-shell action axes and their complete
hard-F/Q replay.  Stream C may inspect only the canonically Q-safe actions
that were frozen before any C outcome.  The older C bridge materializes a
packed child mask for every safe action.  This bridge freezes the same source
decision using receipts only:

* every source endpoint is checked in source order (Q may be nonmonotone);
* ambiguous canonical Q status refuses the whole family;
* robustly infeasible endpoints remain in the complete decision audit;
* safe endpoints receive a compact source-to-C action crosswalk; and
* exact physical duplicates are identified across the two KKT branches.

The durable return value contains no ``HardMask``, packed mask payload,
device label handle, direction provider, ndarray, or route-scale host copy.
It authorizes only preparation of the registered C measurement.  It cannot
authorize a C outcome, calibration, the paired oracle, mask commit,
terminality, or a Pareto claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Integral, Real
from typing import Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_device_shell_ladder import V1DeviceShellFrozenActionReceipt
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_signed_work_device_shell_cuda import (
    V1SignedWorkDeviceShellExecutionBundle,
    V1SignedWorkDeviceShellFamilyBundle,
)
from .modern_v1_signed_work_hard_fq_replay_cuda import (
    V1SignedWorkHardFQBranchReceipt,
    V1SignedWorkHardFQEndpointReceipt,
    V1SignedWorkHardFQEndpointStatus,
    V1SignedWorkHardFQFamilyBundle,
)


LABEL_NATIVE_C_ENDPOINT_DECISION_SCHEMA: Final = (
    "partition_free_v1_signed_work_label_native_pre_C_endpoint_decision_v2"
)
LABEL_NATIVE_C_ACTION_CROSSWALK_SCHEMA: Final = (
    "partition_free_v1_signed_work_label_native_pre_C_action_crosswalk_v2"
)
LABEL_NATIVE_C_BRANCH_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_signed_work_label_native_pre_C_branch_receipt_v1"
)
LABEL_NATIVE_C_FAMILY_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_signed_work_label_native_pre_C_family_receipt_v2"
)
LABEL_NATIVE_C_FAMILY_BUNDLE_SCHEMA: Final = (
    "partition_free_v1_signed_work_label_native_pre_C_family_bundle_v1"
)
LABEL_NATIVE_C_PHYSICAL_IDENTITY_SCHEME: Final = (
    "route_axis_parent_and_exact_streamed_child_action_v1"
)
LABEL_NATIVE_C_HARD_F_ORDER_RULE: Final = (
    "descending_exact_hard_resource_then_strictly_disjoint_hard_signed_F_"
    "with_identical_replay_physical_duplicate_tie_v1"
)
LABEL_NATIVE_C_BRIDGE_AUTHORITY: Final = (
    "complete_D_shell_hard_FQ_to_filtered_label_native_pre_C_action_axis_only_"
    "no_mask_payload_labels_C_outcome_K_O_commit_terminal_or_pareto_v1"
)

_ENDPOINT_PREFIX: Final = "partition-free-v1-label-native-pre-C-endpoint-"
_CROSSWALK_PREFIX: Final = "partition-free-v1-label-native-pre-C-crosswalk-"
_BRANCH_PREFIX: Final = "partition-free-v1-label-native-pre-C-branch-"
_FAMILY_PREFIX: Final = "partition-free-v1-label-native-pre-C-family-"
_BUNDLE_PREFIX: Final = "partition-free-v1-label-native-pre-C-bundle-"
_PHYSICAL_PREFIX: Final = "partition-free-v1-label-native-physical-action-"


class V1SignedWorkLabelNativeCBridgeRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    SOURCE_IDENTITY_MISMATCH = "source_identity_mismatch"
    ENDPOINT_BIJECTION_MISMATCH = "endpoint_bijection_mismatch"
    ENDPOINT_STATUS_REPLAY_MISMATCH = "endpoint_status_replay_mismatch"
    CANONICAL_Q_AMBIGUOUS = "canonical_q_ambiguous"
    HARD_F_ORDER_AMBIGUOUS = "hard_f_order_ambiguous"
    NO_Q_SAFE_CANDIDATE = "no_q_safe_c_candidate"
    CAUSAL_ORDER_VIOLATION = "causal_order_violation"
    FILTER_AXIS_MISMATCH = "filter_axis_mismatch"
    DUPLICATE_IDENTITY_MISMATCH = "duplicate_identity_mismatch"
    RECEIPT_INTEGRITY_MISMATCH = "receipt_integrity_mismatch"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SignedWorkLabelNativeCBridgeRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SignedWorkLabelNativeCBridgeRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SignedWorkLabelNativeCBridgeRefusalCode:
            raise TypeError("label-native C-bridge refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SignedWorkLabelNativeCBridgeRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SignedWorkLabelNativeCBridgeRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SignedWorkLabelNativeCBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkLabelNativeCBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be one positive integer",
        )
    result = int(value)
    if result <= 0:
        _refuse(
            V1SignedWorkLabelNativeCBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkLabelNativeCBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SignedWorkLabelNativeCBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SignedWorkLabelNativeCBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative finite scalar",
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _refuse(
            V1SignedWorkLabelNativeCBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be finite and nonnegative",
        )
    return result


def _closed_flags(values: tuple[bool, ...], *, owner: str) -> None:
    if any(type(value) is not bool for value in values):
        raise TypeError(f"{owner} authority flags must be exact booleans")
    if any(values):
        _refuse(
            V1SignedWorkLabelNativeCBridgeRefusalCode.AUTHORITY_VIOLATION,
            f"{owner} crossed its pre-C receipt-only authority",
        )


def _check_expected_hash(
    observed: str,
    expected: str | None,
    *,
    owner: str,
) -> str:
    if expected is not None and observed != _identity(expected, name=f"{owner} hash"):
        _refuse(
            V1SignedWorkLabelNativeCBridgeRefusalCode.RECEIPT_INTEGRITY_MISMATCH,
            f"{owner} differs from its registered hash",
        )
    return observed


def _status_from_interval(
    *, lower: float, upper: float, budget: float
) -> V1SignedWorkHardFQEndpointStatus:
    if upper < budget:
        return V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE
    if lower > budget:
        return V1SignedWorkHardFQEndpointStatus.Q_ROBUST_INFEASIBLE
    return V1SignedWorkHardFQEndpointStatus.Q_ENVELOPE_AMBIGUOUS


def _physical_key(
    *,
    route_axis_hash: str,
    route_count: int,
    parent_mask_hash: str,
    action: V1DeviceShellFrozenActionReceipt,
) -> tuple[str, int, str, str, str, str, str, int]:
    return (
        route_axis_hash,
        route_count,
        parent_mask_hash,
        action.resulting_mask_packed_sha256,
        action.resulting_mask_hash,
        action.removed_route_index_hash,
        action.packed_action_hash,
        action.cumulative_removed_count,
    )


def _physical_identity_hash(
    *,
    route_axis_hash: str,
    route_count: int,
    parent_mask_hash: str,
    action: V1DeviceShellFrozenActionReceipt,
) -> str:
    key = _physical_key(
        route_axis_hash=route_axis_hash,
        route_count=route_count,
        parent_mask_hash=parent_mask_hash,
        action=action,
    )
    return content_hash(
        {
            "identity_scheme": LABEL_NATIVE_C_PHYSICAL_IDENTITY_SCHEME,
            "route_axis_hash": key[0],
            "route_count": key[1],
            "parent_mask_hash": key[2],
            "resulting_mask_packed_sha256": key[3],
            "resulting_mask_hash": key[4],
            "removed_route_index_hash": key[5],
            "packed_action_hash": key[6],
            "cumulative_removed_count": key[7],
        },
        prefix=_PHYSICAL_PREFIX,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkLabelNativeCEndpointDecisionReceipt:
    """Complete hard-F/Q status audit for one source shell action."""

    branch: V1RelaxedKKTBranch
    source_endpoint_label: int
    source_action_receipt: V1DeviceShellFrozenActionReceipt
    hard_fq_endpoint_receipt_hash: str
    resource_nominal: float
    resource_lower: float
    resource_upper: float
    signed_f_nominal: float
    signed_f_lower: float
    signed_f_upper: float
    origin_branch_status: V1SignedWorkHardFQEndpointStatus
    canonical_status: V1SignedWorkHardFQEndpointStatus
    eligible_for_C_measurement: bool
    C_outcomes_seen: bool = False
    D_FQ_values_reused_as_C_observations: bool = False
    schema: str = LABEL_NATIVE_C_ENDPOINT_DECISION_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("endpoint decision branch must be exact")
        label = _positive_int(
            self.source_endpoint_label,
            name="source endpoint label",
        )
        object.__setattr__(self, "source_endpoint_label", label)
        if type(self.source_action_receipt) is not V1DeviceShellFrozenActionReceipt:
            raise TypeError("endpoint decision requires an exact shell action receipt")
        if self.source_action_receipt.endpoint_label != label:
            _refuse(
                V1SignedWorkLabelNativeCBridgeRefusalCode.ENDPOINT_BIJECTION_MISMATCH,
                "endpoint decision action label differs from its source coordinate",
            )
        object.__setattr__(
            self,
            "hard_fq_endpoint_receipt_hash",
            _identity(
                self.hard_fq_endpoint_receipt_hash,
                name="hard-F/Q endpoint receipt hash",
            ),
        )
        for name in (
            "resource_nominal",
            "resource_lower",
            "resource_upper",
            "signed_f_nominal",
            "signed_f_lower",
            "signed_f_upper",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name),
            )
        if not (
            self.resource_lower <= self.resource_nominal <= self.resource_upper
            and self.signed_f_lower <= self.signed_f_nominal <= self.signed_f_upper
        ):
            _refuse(
                V1SignedWorkLabelNativeCBridgeRefusalCode.ENDPOINT_STATUS_REPLAY_MISMATCH,
                "hard resource or signed-F nominal escaped its outward enclosure",
            )
        if (
            type(self.origin_branch_status) is not V1SignedWorkHardFQEndpointStatus
            or type(self.canonical_status) is not V1SignedWorkHardFQEndpointStatus
        ):
            raise TypeError("endpoint decision statuses must use the exact enum")
        expected = (
            self.canonical_status
            is V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE
        )
        if (
            type(self.eligible_for_C_measurement) is not bool
            or self.eligible_for_C_measurement is not expected
        ):
            _refuse(
                V1SignedWorkLabelNativeCBridgeRefusalCode.FILTER_AXIS_MISMATCH,
                "C eligibility differs from canonical hard-F/Q status",
            )
        _closed_flags(
            (self.C_outcomes_seen, self.D_FQ_values_reused_as_C_observations),
            owner="label-native endpoint decision",
        )
        if self.schema != LABEL_NATIVE_C_ENDPOINT_DECISION_SCHEMA:
            raise ValueError("unsupported label-native endpoint decision schema")

    @property
    def cumulative_removed_count(self) -> int:
        return self.source_action_receipt.cumulative_removed_count

    @property
    def source_action_receipt_hash(self) -> str:
        return self.source_action_receipt.receipt_hash

    @property
    def packed_action_hash(self) -> str:
        return self.source_action_receipt.packed_action_hash

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_ENDPOINT_PREFIX)

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.__post_init__()
        return _check_expected_hash(
            self.receipt_hash,
            expected_hash,
            owner="label-native endpoint decision receipt",
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkLabelNativeCActionCrosswalkReceipt:
    """One compact C-axis coordinate bound to an exact safe source action."""

    branch: V1RelaxedKKTBranch
    compact_C_endpoint_index: int
    source_endpoint_label: int
    source_action_receipt: V1DeviceShellFrozenActionReceipt
    endpoint_decision_receipt_hash: str
    hard_fq_endpoint_receipt_hash: str
    resource_nominal: float
    resource_lower: float
    resource_upper: float
    signed_f_nominal: float
    signed_f_lower: float
    signed_f_upper: float
    route_axis_hash: str
    route_count: int
    parent_mask_hash: str
    canonical_physical_identity_hash: str
    canonical_physical_occurrence: int
    first_occurrence_branch: V1RelaxedKKTBranch
    first_occurrence_source_endpoint_label: int
    exact_duplicate_of_earlier_occurrence: bool
    hard_fq_status: V1SignedWorkHardFQEndpointStatus = (
        V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE
    )
    physical_identity_scheme: str = LABEL_NATIVE_C_PHYSICAL_IDENTITY_SCHEME
    endpoint_hard_mask_materialized: bool = False
    packed_mask_payload_materialized: bool = False
    route_scale_D2H_bytes: int = 0
    C_outcomes_seen: bool = False
    C_observation_count: int = 0
    authority_scope: str = LABEL_NATIVE_C_BRIDGE_AUTHORITY
    schema: str = LABEL_NATIVE_C_ACTION_CROSSWALK_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.branch) is not V1RelaxedKKTBranch
            or type(self.first_occurrence_branch) is not V1RelaxedKKTBranch
        ):
            raise TypeError("crosswalk branch coordinates must be exact")
        compact = _positive_int(
            self.compact_C_endpoint_index,
            name="compact C endpoint index",
        )
        source = _positive_int(
            self.source_endpoint_label,
            name="source endpoint label",
        )
        first = _positive_int(
            self.first_occurrence_source_endpoint_label,
            name="first occurrence source endpoint label",
        )
        occurrence = _positive_int(
            self.canonical_physical_occurrence,
            name="canonical physical occurrence",
        )
        object.__setattr__(self, "compact_C_endpoint_index", compact)
        object.__setattr__(self, "source_endpoint_label", source)
        object.__setattr__(
            self,
            "first_occurrence_source_endpoint_label",
            first,
        )
        object.__setattr__(self, "canonical_physical_occurrence", occurrence)
        if type(self.source_action_receipt) is not V1DeviceShellFrozenActionReceipt:
            raise TypeError("crosswalk requires an exact shell action receipt")
        if self.source_action_receipt.endpoint_label != source:
            _refuse(
                V1SignedWorkLabelNativeCBridgeRefusalCode.ENDPOINT_BIJECTION_MISMATCH,
                "crosswalk source action has another endpoint label",
            )
        for name in (
            "endpoint_decision_receipt_hash",
            "hard_fq_endpoint_receipt_hash",
            "route_axis_hash",
            "parent_mask_hash",
            "canonical_physical_identity_hash",
            "physical_identity_scheme",
            "authority_scope",
        ):
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), name=name),
            )
        route_count = _positive_int(self.route_count, name="route count")
        object.__setattr__(self, "route_count", route_count)
        for name in (
            "resource_nominal",
            "resource_lower",
            "resource_upper",
            "signed_f_nominal",
            "signed_f_lower",
            "signed_f_upper",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name),
            )
        if not (
            self.resource_lower <= self.resource_nominal <= self.resource_upper
            and self.signed_f_lower <= self.signed_f_nominal <= self.signed_f_upper
        ):
            _refuse(
                V1SignedWorkLabelNativeCBridgeRefusalCode.ENDPOINT_STATUS_REPLAY_MISMATCH,
                "crosswalk hard resource or signed-F interval is invalid",
            )
        if (
            type(self.hard_fq_status) is not V1SignedWorkHardFQEndpointStatus
            or self.hard_fq_status
            is not V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE
        ):
            _refuse(
                V1SignedWorkLabelNativeCBridgeRefusalCode.FILTER_AXIS_MISMATCH,
                "only canonically Q-safe actions may enter the compact C axis",
            )
        expected_identity = _physical_identity_hash(
            route_axis_hash=self.route_axis_hash,
            route_count=route_count,
            parent_mask_hash=self.parent_mask_hash,
            action=self.source_action_receipt,
        )
        if (
            self.canonical_physical_identity_hash != expected_identity
            or self.physical_identity_scheme
            != LABEL_NATIVE_C_PHYSICAL_IDENTITY_SCHEME
        ):
            _refuse(
                V1SignedWorkLabelNativeCBridgeRefusalCode.DUPLICATE_IDENTITY_MISMATCH,
                "crosswalk physical identity does not replay its exact action",
            )
        if type(self.exact_duplicate_of_earlier_occurrence) is not bool:
            raise TypeError("duplicate flag must be exact")
        expected_duplicate = occurrence > 1
        if self.exact_duplicate_of_earlier_occurrence is not expected_duplicate:
            _refuse(
                V1SignedWorkLabelNativeCBridgeRefusalCode.DUPLICATE_IDENTITY_MISMATCH,
                "duplicate flag differs from the registered occurrence",
            )
        if not expected_duplicate and (
            self.first_occurrence_branch is not self.branch
            or first != source
        ):
            _refuse(
                V1SignedWorkLabelNativeCBridgeRefusalCode.DUPLICATE_IDENTITY_MISMATCH,
                "a first physical occurrence must point to itself",
            )
        object.__setattr__(
            self,
            "route_scale_D2H_bytes",
            _nonnegative_int(self.route_scale_D2H_bytes, name="route-scale D2H bytes"),
        )
        object.__setattr__(
            self,
            "C_observation_count",
            _nonnegative_int(self.C_observation_count, name="C observation count"),
        )
        _closed_flags(
            (
                self.endpoint_hard_mask_materialized,
                self.packed_mask_payload_materialized,
                self.C_outcomes_seen,
            ),
            owner="label-native action crosswalk",
        )
        if self.route_scale_D2H_bytes != 0 or self.C_observation_count != 0:
            _refuse(
                V1SignedWorkLabelNativeCBridgeRefusalCode.AUTHORITY_VIOLATION,
                "compact crosswalk may not transfer route payloads or observe C",
            )
        if (
            self.authority_scope != LABEL_NATIVE_C_BRIDGE_AUTHORITY
            or self.schema != LABEL_NATIVE_C_ACTION_CROSSWALK_SCHEMA
        ):
            raise ValueError("unsupported label-native action crosswalk semantics")

    @property
    def cumulative_removed_count(self) -> int:
        return self.source_action_receipt.cumulative_removed_count

    @property
    def source_action_receipt_hash(self) -> str:
        return self.source_action_receipt.receipt_hash

    @property
    def packed_action_hash(self) -> str:
        return self.source_action_receipt.packed_action_hash

    @property
    def resulting_mask_hash(self) -> str:
        return self.source_action_receipt.resulting_mask_hash

    @property
    def removed_route_index_hash(self) -> str:
        return self.source_action_receipt.removed_route_index_hash

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_CROSSWALK_PREFIX)

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.__post_init__()
        return _check_expected_hash(
            self.receipt_hash,
            expected_hash,
            owner="label-native action crosswalk receipt",
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkLabelNativeCBranchReceipt:
    """Complete source audit plus the compact Q-safe C action axis."""

    branch: V1RelaxedKKTBranch
    shell_family_hash: str
    shell_execution_hash: str
    shell_execution_receipt_hash: str
    shell_branch_identity_receipt_hash: str
    shell_ladder_receipt_hash: str
    hard_fq_bundle_hash: str
    hard_fq_family_receipt_hash: str
    hard_fq_branch_receipt_hash: str
    route_axis_hash: str
    route_count: int
    parent_mask_hash: str
    complete_source_endpoint_counts: tuple[int, ...]
    complete_source_action_receipt_hashes: tuple[str, ...]
    complete_source_action_hashes: tuple[str, ...]
    endpoint_decisions: tuple[V1SignedWorkLabelNativeCEndpointDecisionReceipt, ...]
    eligible_action_crosswalks: tuple[
        V1SignedWorkLabelNativeCActionCrosswalkReceipt, ...
    ]
    robustly_infeasible_source_endpoint_labels: tuple[int, ...]
    all_source_endpoints_considered: bool = True
    q_nonmonotone_scan_early_stop_used: bool = False
    eligible_axis_frozen_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    endpoint_hard_masks_registered: bool = False
    packed_mask_payloads_registered: bool = False
    route_scale_D2H_bytes: int = 0
    candidate_composition_recomputed: bool = False
    C_measurement_outcome_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = LABEL_NATIVE_C_BRIDGE_AUTHORITY
    schema: str = LABEL_NATIVE_C_BRANCH_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("label-native C branch must use the exact KKT enum")
        for name in (
            "shell_family_hash",
            "shell_execution_hash",
            "shell_execution_receipt_hash",
            "shell_branch_identity_receipt_hash",
            "shell_ladder_receipt_hash",
            "hard_fq_bundle_hash",
            "hard_fq_family_receipt_hash",
            "hard_fq_branch_receipt_hash",
            "route_axis_hash",
            "parent_mask_hash",
            "authority_scope",
        ):
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), name=name),
            )
        route_count = _positive_int(self.route_count, name="route count")
        object.__setattr__(self, "route_count", route_count)
        counts = tuple(
            _positive_int(value, name="source endpoint count")
            for value in self.complete_source_endpoint_counts
        )
        action_receipts = tuple(
            _identity(value, name="source action receipt hash")
            for value in self.complete_source_action_receipt_hashes
        )
        action_hashes = tuple(
            _identity(value, name="source packed action hash")
            for value in self.complete_source_action_hashes
        )
        decisions = tuple(self.endpoint_decisions)
        crosswalks = tuple(self.eligible_action_crosswalks)
        infeasible = tuple(
            _positive_int(value, name="robustly infeasible source endpoint label")
            for value in self.robustly_infeasible_source_endpoint_labels
        )
        source_count = len(counts)
        if (
            source_count <= 0
            or tuple(sorted(set(counts))) != counts
            or len(action_receipts) != source_count
            or len(action_hashes) != source_count
            or len(decisions) != source_count
            or any(
                type(value) is not V1SignedWorkLabelNativeCEndpointDecisionReceipt
                or value.branch is not self.branch
                for value in decisions
            )
            or tuple(value.source_endpoint_label for value in decisions)
            != tuple(range(1, source_count + 1))
            or tuple(value.cumulative_removed_count for value in decisions) != counts
            or tuple(value.source_action_receipt_hash for value in decisions)
            != action_receipts
            or tuple(value.packed_action_hash for value in decisions) != action_hashes
        ):
            _refuse(
                V1SignedWorkLabelNativeCBridgeRefusalCode.FILTER_AXIS_MISMATCH,
                "branch complete source audit is incomplete or reordered",
            )
        if any(
            type(value) is not V1SignedWorkLabelNativeCActionCrosswalkReceipt
            or value.branch is not self.branch
            for value in crosswalks
        ):
            raise TypeError("eligible axis requires exact same-branch crosswalks")
        safe_labels = tuple(
            value.source_endpoint_label
            for value in decisions
            if value.eligible_for_C_measurement
        )
        if (
            tuple(value.compact_C_endpoint_index for value in crosswalks)
            != tuple(range(1, len(crosswalks) + 1))
            or tuple(value.source_endpoint_label for value in crosswalks)
            != safe_labels
            or tuple(value.endpoint_decision_receipt_hash for value in crosswalks)
            != tuple(decisions[label - 1].receipt_hash for label in safe_labels)
            or tuple(
                (
                    value.resource_lower,
                    value.resource_nominal,
                    value.resource_upper,
                    value.signed_f_lower,
                    value.signed_f_nominal,
                    value.signed_f_upper,
                )
                for value in crosswalks
            )
            != tuple(
                (
                    decisions[label - 1].resource_lower,
                    decisions[label - 1].resource_nominal,
                    decisions[label - 1].resource_upper,
                    decisions[label - 1].signed_f_lower,
                    decisions[label - 1].signed_f_nominal,
                    decisions[label - 1].signed_f_upper,
                )
                for label in safe_labels
            )
            or tuple(sorted(infeasible)) != infeasible
            or set(infeasible) != set(range(1, source_count + 1)) - set(safe_labels)
        ):
            _refuse(
                V1SignedWorkLabelNativeCBridgeRefusalCode.FILTER_AXIS_MISMATCH,
                "compact C axis is not the exact source-ordered Q-safe subset",
            )
        object.__setattr__(self, "complete_source_endpoint_counts", counts)
        object.__setattr__(
            self,
            "complete_source_action_receipt_hashes",
            action_receipts,
        )
        object.__setattr__(self, "complete_source_action_hashes", action_hashes)
        object.__setattr__(self, "endpoint_decisions", decisions)
        object.__setattr__(self, "eligible_action_crosswalks", crosswalks)
        object.__setattr__(
            self,
            "robustly_infeasible_source_endpoint_labels",
            infeasible,
        )
        object.__setattr__(
            self,
            "route_scale_D2H_bytes",
            _nonnegative_int(self.route_scale_D2H_bytes, name="route-scale D2H bytes"),
        )
        required_true = (
            self.all_source_endpoints_considered,
            self.eligible_axis_frozen_before_C_outcomes,
        )
        required_false = (
            self.q_nonmonotone_scan_early_stop_used,
            self.C_outcomes_seen,
            self.endpoint_hard_masks_registered,
            self.packed_mask_payloads_registered,
            self.candidate_composition_recomputed,
            self.C_measurement_outcome_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in required_true + required_false):
            raise TypeError("label-native C branch flags must be exact booleans")
        if not all(required_true) or any(required_false) or self.route_scale_D2H_bytes:
            _refuse(
                V1SignedWorkLabelNativeCBridgeRefusalCode.AUTHORITY_VIOLATION,
                "branch did not remain a complete receipt-only pre-C freeze",
            )
        if (
            self.authority_scope != LABEL_NATIVE_C_BRIDGE_AUTHORITY
            or self.schema != LABEL_NATIVE_C_BRANCH_RECEIPT_SCHEMA
        ):
            raise ValueError("unsupported label-native C branch semantics")

    @property
    def eligible_source_endpoint_labels(self) -> tuple[int, ...]:
        return tuple(
            value.source_endpoint_label for value in self.eligible_action_crosswalks
        )

    @property
    def eligible_endpoint_counts(self) -> tuple[int, ...]:
        return tuple(
            value.cumulative_removed_count for value in self.eligible_action_crosswalks
        )

    @property
    def eligible_action_hashes(self) -> tuple[str, ...]:
        return tuple(value.packed_action_hash for value in self.eligible_action_crosswalks)

    @property
    def eligible_action_receipt_hashes(self) -> tuple[str, ...]:
        return tuple(
            value.source_action_receipt_hash for value in self.eligible_action_crosswalks
        )

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_BRANCH_PREFIX)

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for value in self.endpoint_decisions:
            value.assert_integrity()
        for value in self.eligible_action_crosswalks:
            value.assert_integrity()
        self.__post_init__()
        return _check_expected_hash(
            self.receipt_hash,
            expected_hash,
            owner="label-native C branch receipt",
        )


def _branch_order_key(
    value: V1SignedWorkLabelNativeCActionCrosswalkReceipt,
) -> tuple[int, int]:
    return (
        0 if value.branch is V1RelaxedKKTBranch.SPECTRAL else 1,
        value.source_endpoint_label,
    )


def _hard_resource_signature(
    value: V1SignedWorkLabelNativeCActionCrosswalkReceipt,
) -> tuple[float, float, float]:
    return (value.resource_lower, value.resource_nominal, value.resource_upper)


def _hard_F_signature(
    value: V1SignedWorkLabelNativeCActionCrosswalkReceipt,
) -> tuple[float, float, float]:
    return (value.signed_f_lower, value.signed_f_nominal, value.signed_f_upper)


def _freeze_family_hard_F_order(
    crosswalks: tuple[V1SignedWorkLabelNativeCActionCrosswalkReceipt, ...],
) -> tuple[V1SignedWorkLabelNativeCActionCrosswalkReceipt, ...]:
    """Freeze least-hard-F order independently within each exact resource.

    The deterministic physical-duplicate tie is lawful only when the hard
    resource and signed-F replay are byte-identical.  Distinct physical
    actions require weakly ordered outward F intervals: ``F_A,U <= F_B,L``
    proves that A is never more expensive, including an exact boundary tie.
    A genuine overlap leaves the least-work action unidentified and therefore
    refuses before stream C.
    """

    by_count: dict[int, list[V1SignedWorkLabelNativeCActionCrosswalkReceipt]] = {}
    for value in crosswalks:
        by_count.setdefault(value.cumulative_removed_count, []).append(value)
    frozen: list[V1SignedWorkLabelNativeCActionCrosswalkReceipt] = []
    for count in sorted(by_count, reverse=True):
        group = by_count[count]
        resource_signatures = {_hard_resource_signature(value) for value in group}
        if len(resource_signatures) != 1:
            _refuse(
                V1SignedWorkLabelNativeCBridgeRefusalCode.HARD_F_ORDER_AMBIGUOUS,
                f"resource {count} has inconsistent exact hard-resource replay",
            )
        resource_lower, resource_nominal, resource_upper = next(
            iter(resource_signatures)
        )
        if not (
            resource_nominal == float(count)
            and resource_lower <= float(count) <= resource_upper
        ):
            _refuse(
                V1SignedWorkLabelNativeCBridgeRefusalCode.HARD_F_ORDER_AMBIGUOUS,
                f"resource {count} is not the registered unit-route hard resource",
            )

        by_physical: dict[
            str, list[V1SignedWorkLabelNativeCActionCrosswalkReceipt]
        ] = {}
        for value in group:
            by_physical.setdefault(
                value.canonical_physical_identity_hash,
                [],
            ).append(value)
        representatives: list[V1SignedWorkLabelNativeCActionCrosswalkReceipt] = []
        occurrences_by_physical: dict[
            str, tuple[V1SignedWorkLabelNativeCActionCrosswalkReceipt, ...]
        ] = {}
        for physical, occurrences in by_physical.items():
            ordered_occurrences = tuple(sorted(occurrences, key=_branch_order_key))
            replay = {
                (_hard_resource_signature(value), _hard_F_signature(value))
                for value in ordered_occurrences
            }
            if len(replay) != 1:
                _refuse(
                    V1SignedWorkLabelNativeCBridgeRefusalCode.DUPLICATE_IDENTITY_MISMATCH,
                    "one exact physical action has unequal hard resource or signed-F replay",
                )
            representatives.append(ordered_occurrences[0])
            occurrences_by_physical[physical] = ordered_occurrences

        representatives.sort(
            key=lambda value: (
                value.signed_f_lower,
                value.signed_f_upper,
                value.signed_f_nominal,
                value.canonical_physical_identity_hash,
            )
        )
        for lower_work, higher_work in zip(
            representatives,
            representatives[1:],
            strict=False,
        ):
            if not lower_work.signed_f_upper <= higher_work.signed_f_lower:
                _refuse(
                    V1SignedWorkLabelNativeCBridgeRefusalCode.HARD_F_ORDER_AMBIGUOUS,
                    "distinct equal-resource hard actions have overlapping signed-F intervals",
                )
        for representative in representatives:
            frozen.extend(
                occurrences_by_physical[
                    representative.canonical_physical_identity_hash
                ]
            )
    return tuple(frozen)


@dataclass(frozen=True, slots=True)
class V1SignedWorkLabelNativeCFamilyReceipt:
    shell_family_hash: str
    hard_fq_bundle_hash: str
    hard_fq_family_receipt_hash: str
    route_axis_hash: str
    route_count: int
    parent_mask_hash: str
    spectral_branch_receipt_hash: str
    diagonal_l1_branch_receipt_hash: str
    unique_physical_identity_hashes: tuple[str, ...]
    pre_C_hard_F_order_crosswalk_receipt_hashes: tuple[str, ...]
    eligible_action_occurrence_count: int
    hard_F_order_rule: str = LABEL_NATIVE_C_HARD_F_ORDER_RULE
    both_q_branches_preserved: bool = True
    every_source_endpoint_considered: bool = True
    eligible_axes_frozen_before_C_outcomes: bool = True
    hard_F_order_frozen_before_C_outcomes: bool = True
    ambiguous_endpoint_count: int = 0
    label_handles_borrowed: bool = False
    endpoint_hard_masks_materialized: bool = False
    packed_mask_payloads_materialized: bool = False
    route_scale_D2H_bytes: int = 0
    C_outcomes_seen: bool = False
    C_observation_count: int = 0
    C_measurement_outcome_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = LABEL_NATIVE_C_BRIDGE_AUTHORITY
    schema: str = LABEL_NATIVE_C_FAMILY_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "shell_family_hash",
            "hard_fq_bundle_hash",
            "hard_fq_family_receipt_hash",
            "route_axis_hash",
            "parent_mask_hash",
            "spectral_branch_receipt_hash",
            "diagonal_l1_branch_receipt_hash",
            "hard_F_order_rule",
            "authority_scope",
        ):
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="route count"),
        )
        identities = tuple(
            _identity(value, name="unique physical identity hash")
            for value in self.unique_physical_identity_hashes
        )
        if not identities or len(set(identities)) != len(identities):
            raise ValueError("unique physical identity axis is empty or repeated")
        object.__setattr__(self, "unique_physical_identity_hashes", identities)
        hard_F_order = tuple(
            _identity(value, name="pre-C hard-F order crosswalk receipt hash")
            for value in self.pre_C_hard_F_order_crosswalk_receipt_hashes
        )
        if not hard_F_order or len(set(hard_F_order)) != len(hard_F_order):
            raise ValueError("pre-C hard-F order is empty or repeated")
        object.__setattr__(
            self,
            "pre_C_hard_F_order_crosswalk_receipt_hashes",
            hard_F_order,
        )
        occurrences = _positive_int(
            self.eligible_action_occurrence_count,
            name="eligible action occurrence count",
        )
        object.__setattr__(self, "eligible_action_occurrence_count", occurrences)
        if len(identities) > occurrences:
            raise ValueError("physical identity count exceeds action occurrences")
        for name in (
            "ambiguous_endpoint_count",
            "route_scale_D2H_bytes",
            "C_observation_count",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        required_true = (
            self.both_q_branches_preserved,
            self.every_source_endpoint_considered,
            self.eligible_axes_frozen_before_C_outcomes,
            self.hard_F_order_frozen_before_C_outcomes,
        )
        required_false = (
            self.label_handles_borrowed,
            self.endpoint_hard_masks_materialized,
            self.packed_mask_payloads_materialized,
            self.C_outcomes_seen,
            self.C_measurement_outcome_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in required_true + required_false):
            raise TypeError("label-native C-family flags must be exact booleans")
        if (
            not all(required_true)
            or any(required_false)
            or self.ambiguous_endpoint_count
            or self.route_scale_D2H_bytes
            or self.C_observation_count
        ):
            _refuse(
                V1SignedWorkLabelNativeCBridgeRefusalCode.AUTHORITY_VIOLATION,
                "family did not remain a complete receipt-only pre-C freeze",
            )
        if (
            self.hard_F_order_rule != LABEL_NATIVE_C_HARD_F_ORDER_RULE
            or len(hard_F_order) != occurrences
            or
            self.authority_scope != LABEL_NATIVE_C_BRIDGE_AUTHORITY
            or self.schema != LABEL_NATIVE_C_FAMILY_RECEIPT_SCHEMA
        ):
            raise ValueError("unsupported label-native C-family semantics")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_FAMILY_PREFIX)

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.__post_init__()
        return _check_expected_hash(
            self.receipt_hash,
            expected_hash,
            owner="label-native C family receipt",
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkLabelNativeCBridgeFamilyBundle:
    spectral: V1SignedWorkLabelNativeCBranchReceipt
    diagonal_l1: V1SignedWorkLabelNativeCBranchReceipt
    receipt: V1SignedWorkLabelNativeCFamilyReceipt
    schema: str = LABEL_NATIVE_C_FAMILY_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.spectral) is not V1SignedWorkLabelNativeCBranchReceipt
            or type(self.diagonal_l1) is not V1SignedWorkLabelNativeCBranchReceipt
            or type(self.receipt) is not V1SignedWorkLabelNativeCFamilyReceipt
        ):
            raise TypeError("label-native C family requires exact receipt types")
        if (
            self.spectral.branch is not V1RelaxedKKTBranch.SPECTRAL
            or self.diagonal_l1.branch is not V1RelaxedKKTBranch.DIAGONAL_L1
            or self.spectral.receipt_hash
            != self.receipt.spectral_branch_receipt_hash
            or self.diagonal_l1.receipt_hash
            != self.receipt.diagonal_l1_branch_receipt_hash
            or self.spectral.shell_family_hash != self.receipt.shell_family_hash
            or self.diagonal_l1.shell_family_hash != self.receipt.shell_family_hash
            or self.spectral.hard_fq_bundle_hash != self.receipt.hard_fq_bundle_hash
            or self.diagonal_l1.hard_fq_bundle_hash != self.receipt.hard_fq_bundle_hash
            or self.spectral.route_axis_hash != self.receipt.route_axis_hash
            or self.diagonal_l1.route_axis_hash != self.receipt.route_axis_hash
            or self.spectral.route_count != self.receipt.route_count
            or self.diagonal_l1.route_count != self.receipt.route_count
            or self.spectral.parent_mask_hash != self.receipt.parent_mask_hash
            or self.diagonal_l1.parent_mask_hash != self.receipt.parent_mask_hash
            or self.schema != LABEL_NATIVE_C_FAMILY_BUNDLE_SCHEMA
        ):
            raise ValueError("label-native C family branch identity differs")
        crosswalks = (
            self.spectral.eligible_action_crosswalks
            + self.diagonal_l1.eligible_action_crosswalks
        )
        unique: list[str] = []
        first_by_identity: dict[
            str, tuple[V1RelaxedKKTBranch, int, tuple[str, int, str, str, str, str, str, int]]
        ] = {}
        occurrence_by_identity: dict[str, int] = {}
        for value in crosswalks:
            key = _physical_key(
                route_axis_hash=value.route_axis_hash,
                route_count=value.route_count,
                parent_mask_hash=value.parent_mask_hash,
                action=value.source_action_receipt,
            )
            identity = value.canonical_physical_identity_hash
            occurrence = occurrence_by_identity.get(identity, 0) + 1
            occurrence_by_identity[identity] = occurrence
            if identity not in first_by_identity:
                first_by_identity[identity] = (
                    value.branch,
                    value.source_endpoint_label,
                    key,
                )
                unique.append(identity)
            first_branch, first_label, first_key = first_by_identity[identity]
            if (
                key != first_key
                or value.canonical_physical_occurrence != occurrence
                or value.first_occurrence_branch is not first_branch
                or value.first_occurrence_source_endpoint_label != first_label
                or value.exact_duplicate_of_earlier_occurrence is not (occurrence > 1)
            ):
                _refuse(
                    V1SignedWorkLabelNativeCBridgeRefusalCode.DUPLICATE_IDENTITY_MISMATCH,
                    "family physical duplicate registration does not replay",
                )
        if (
            tuple(unique) != self.receipt.unique_physical_identity_hashes
            or len(crosswalks) != self.receipt.eligible_action_occurrence_count
        ):
            _refuse(
                V1SignedWorkLabelNativeCBridgeRefusalCode.DUPLICATE_IDENTITY_MISMATCH,
                "family unique/occurrence counts differ from its crosswalks",
            )
        frozen_hard_F_order = _freeze_family_hard_F_order(crosswalks)
        if tuple(
            value.receipt_hash for value in frozen_hard_F_order
        ) != self.receipt.pre_C_hard_F_order_crosswalk_receipt_hashes:
            _refuse(
                V1SignedWorkLabelNativeCBridgeRefusalCode.RECEIPT_INTEGRITY_MISMATCH,
                "family pre-C hard-F order does not replay from exact hard endpoints",
            )

    @property
    def bundle_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "family_receipt_hash": self.receipt.receipt_hash,
                "spectral_branch_receipt_hash": self.spectral.receipt_hash,
                "diagonal_l1_branch_receipt_hash": self.diagonal_l1.receipt_hash,
            },
            prefix=_BUNDLE_PREFIX,
        )

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.spectral.assert_integrity(
            self.receipt.spectral_branch_receipt_hash
        )
        self.diagonal_l1.assert_integrity(
            self.receipt.diagonal_l1_branch_receipt_hash
        )
        self.receipt.assert_integrity()
        self.__post_init__()
        return _check_expected_hash(
            self.bundle_hash,
            expected_hash,
            owner="label-native C bridge family bundle",
        )


@dataclass(frozen=True, slots=True)
class _ValidatedBranch:
    shell: V1SignedWorkDeviceShellExecutionBundle
    hard: V1SignedWorkHardFQBranchReceipt
    decisions: tuple[V1SignedWorkLabelNativeCEndpointDecisionReceipt, ...]


def _validate_branch(
    *,
    shell: V1SignedWorkDeviceShellExecutionBundle,
    hard: V1SignedWorkHardFQBranchReceipt,
) -> _ValidatedBranch:
    execution = shell.execution_receipt
    identity = shell.branch_identity
    ladder = shell.frozen_ladder
    if (
        execution.selected_branch is not hard.branch
        or shell.execution_hash != hard.shell_execution_hash
        or execution.receipt_hash != hard.shell_execution_receipt_hash
        or identity.receipt_hash != hard.branch_identity_receipt_hash
        or ladder.receipt_hash != hard.frozen_ladder_receipt_hash
        or ladder.label_commitment != hard.label_commitment
        or len(ladder.actions) != len(hard.endpoints)
    ):
        _refuse(
            V1SignedWorkLabelNativeCBridgeRefusalCode.SOURCE_IDENTITY_MISMATCH,
            "device shell and hard-F/Q branch are not one exact D action axis",
        )
    decisions: list[V1SignedWorkLabelNativeCEndpointDecisionReceipt] = []
    for action, endpoint in zip(ladder.actions, hard.endpoints, strict=True):
        if (
            endpoint.branch is not hard.branch
            or endpoint.endpoint_label != action.endpoint_label
            or endpoint.cumulative_removed_count != action.cumulative_removed_count
            or endpoint.frozen_action_receipt_hash != action.receipt_hash
            or endpoint.packed_action_hash != action.packed_action_hash
            or endpoint.resulting_mask_hash != action.resulting_mask_hash
            or endpoint.removed_route_index_hash != action.removed_route_index_hash
        ):
            _refuse(
                V1SignedWorkLabelNativeCBridgeRefusalCode.ENDPOINT_BIJECTION_MISMATCH,
                "shell and hard-F/Q endpoint identities are not bijective",
            )
        expected_origin = _status_from_interval(
            lower=endpoint.branch_q_lower,
            upper=endpoint.branch_q_upper,
            budget=endpoint.quadratic_budget,
        )
        expected_canonical = _status_from_interval(
            lower=endpoint.canonical_min_q_lower,
            upper=endpoint.canonical_min_q_upper,
            budget=endpoint.quadratic_budget,
        )
        if (
            endpoint.origin_branch_status is not expected_origin
            or endpoint.status is not expected_canonical
        ):
            _refuse(
                V1SignedWorkLabelNativeCBridgeRefusalCode.ENDPOINT_STATUS_REPLAY_MISMATCH,
                "hard-F/Q status does not replay its outward Q interval",
            )
        decisions.append(
            V1SignedWorkLabelNativeCEndpointDecisionReceipt(
                branch=hard.branch,
                source_endpoint_label=action.endpoint_label,
                source_action_receipt=action,
                hard_fq_endpoint_receipt_hash=endpoint.receipt_hash,
                resource_nominal=endpoint.resource_nominal,
                resource_lower=endpoint.resource_lower,
                resource_upper=endpoint.resource_upper,
                signed_f_nominal=endpoint.signed_f_nominal,
                signed_f_lower=endpoint.signed_f_lower,
                signed_f_upper=endpoint.signed_f_upper,
                origin_branch_status=endpoint.origin_branch_status,
                canonical_status=endpoint.status,
                eligible_for_C_measurement=(
                    endpoint.status
                    is V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE
                ),
            )
        )
    return _ValidatedBranch(shell=shell, hard=hard, decisions=tuple(decisions))


def _branch_pairs(
    *,
    shell_family: V1SignedWorkDeviceShellFamilyBundle,
    hard_fq: V1SignedWorkHardFQFamilyBundle,
) -> tuple[
    tuple[V1SignedWorkDeviceShellExecutionBundle, V1SignedWorkHardFQBranchReceipt],
    tuple[V1SignedWorkDeviceShellExecutionBundle, V1SignedWorkHardFQBranchReceipt],
]:
    return (
        (shell_family.spectral, hard_fq.receipt.spectral),
        (shell_family.diagonal_l1, hard_fq.receipt.diagonal_l1),
    )


def freeze_v1_signed_work_label_native_pre_C_bridge(
    *,
    shell_family: V1SignedWorkDeviceShellFamilyBundle,
    hard_fq: V1SignedWorkHardFQFamilyBundle,
    conditioned_on_c_outcomes: bool,
) -> V1SignedWorkLabelNativeCBridgeFamilyBundle:
    """Freeze the complete receipt-only hard-F/Q filter before stream C.

    The source scan never stops at the first Q violation.  Crosswalk order is
    the original shell order within spectral then diagonal-L1 branches.
    """

    if type(shell_family) is not V1SignedWorkDeviceShellFamilyBundle:
        raise TypeError("label-native C bridge requires an exact shell family")
    if type(hard_fq) is not V1SignedWorkHardFQFamilyBundle:
        raise TypeError("label-native C bridge requires an exact hard-F/Q family")
    if type(conditioned_on_c_outcomes) is not bool:
        raise TypeError("C-conditioning declaration must be an exact bool")
    if conditioned_on_c_outcomes:
        _refuse(
            V1SignedWorkLabelNativeCBridgeRefusalCode.CAUSAL_ORDER_VIOLATION,
            "hard-F/Q action filtering must freeze before any C outcome",
        )
    shell_receipt = shell_family.receipt
    hard_receipt = hard_fq.receipt
    source_authority_flags = (
        shell_receipt.c_stream_outcomes_seen,
        shell_receipt.fqr_authority,
        shell_receipt.calibration_authority,
        shell_receipt.oracle_authority,
        shell_receipt.mask_commit_authority,
        shell_receipt.terminal_authority,
        shell_receipt.pareto_authority,
        hard_receipt.validation_data_seen,
        hard_receipt.c_stream_authority,
        hard_receipt.nonlinear_remainder_authority,
        hard_receipt.calibration_authority,
        hard_receipt.oracle_authority,
        hard_receipt.mask_commit_authority,
        hard_receipt.terminal_authority,
        hard_receipt.pareto_authority,
    )
    if any(type(value) is not bool for value in source_authority_flags):
        raise TypeError("source authority flags must be exact booleans")
    if any(source_authority_flags):
        _refuse(
            V1SignedWorkLabelNativeCBridgeRefusalCode.CAUSAL_ORDER_VIOLATION,
            "source receipts are not a pre-C proposal-only state",
        )
    if (
        hard_receipt.shell_family_hash != shell_family.family_hash
        or hard_receipt.problem_hash != shell_receipt.problem_hash
        or hard_receipt.policy_hash != shell_receipt.policy_hash
        or hard_receipt.least_work_execution_hash
        != shell_receipt.least_work_execution_hash
        or hard_receipt.hbm_forecast_hash != hard_fq.hbm_forecast.forecast_hash
    ):
        _refuse(
            V1SignedWorkLabelNativeCBridgeRefusalCode.SOURCE_IDENTITY_MISMATCH,
            "shell family and complete hard-F/Q replay have different D sources",
        )
    validated = tuple(
        _validate_branch(
            shell=shell,
            hard=hard,
        )
        for shell, hard in _branch_pairs(shell_family=shell_family, hard_fq=hard_fq)
    )
    ambiguous = tuple(
        (item.hard.branch.value, decision.source_endpoint_label)
        for item in validated
        for decision in item.decisions
        if decision.canonical_status
        is V1SignedWorkHardFQEndpointStatus.Q_ENVELOPE_AMBIGUOUS
    )
    if ambiguous:
        _refuse(
            V1SignedWorkLabelNativeCBridgeRefusalCode.CANONICAL_Q_AMBIGUOUS,
            f"canonical Q remains ambiguous at source endpoints {ambiguous!r}",
        )
    if not any(
        decision.eligible_for_C_measurement
        for item in validated
        for decision in item.decisions
    ):
        _refuse(
            V1SignedWorkLabelNativeCBridgeRefusalCode.NO_Q_SAFE_CANDIDATE,
            "neither Q branch contains a canonically safe C action",
        )

    route_axis_hash = validated[0].shell.frozen_ladder.route_axis_hash
    route_count = validated[0].shell.frozen_ladder.route_count
    parent_mask_hash = shell_receipt.parent_mask_hash
    if any(
        item.shell.frozen_ladder.route_axis_hash != route_axis_hash
        or item.shell.frozen_ladder.route_count != route_count
        or item.shell.frozen_ladder.parent_mask_hash != parent_mask_hash
        for item in validated
    ):
        _refuse(
            V1SignedWorkLabelNativeCBridgeRefusalCode.SOURCE_IDENTITY_MISMATCH,
            "two shell branches do not share one physical parent action axis",
        )

    physical_first: dict[
        str,
        tuple[
            V1RelaxedKKTBranch,
            int,
            tuple[str, int, str, str, str, str, str, int],
        ],
    ] = {}
    physical_occurrence: dict[str, int] = {}
    unique_physical: list[str] = []
    branch_receipts: list[V1SignedWorkLabelNativeCBranchReceipt] = []
    for item in validated:
        crosswalks: list[V1SignedWorkLabelNativeCActionCrosswalkReceipt] = []
        for decision in item.decisions:
            if not decision.eligible_for_C_measurement:
                continue
            action = decision.source_action_receipt
            physical = _physical_identity_hash(
                route_axis_hash=route_axis_hash,
                route_count=route_count,
                parent_mask_hash=parent_mask_hash,
                action=action,
            )
            key = _physical_key(
                route_axis_hash=route_axis_hash,
                route_count=route_count,
                parent_mask_hash=parent_mask_hash,
                action=action,
            )
            occurrence = physical_occurrence.get(physical, 0) + 1
            physical_occurrence[physical] = occurrence
            if physical not in physical_first:
                physical_first[physical] = (
                    item.hard.branch,
                    decision.source_endpoint_label,
                    key,
                )
                unique_physical.append(physical)
            first_branch, first_label, first_key = physical_first[physical]
            if key != first_key:
                _refuse(
                    V1SignedWorkLabelNativeCBridgeRefusalCode.DUPLICATE_IDENTITY_MISMATCH,
                    "one physical identity hash names unequal exact actions",
                )
            crosswalks.append(
                V1SignedWorkLabelNativeCActionCrosswalkReceipt(
                    branch=item.hard.branch,
                    compact_C_endpoint_index=len(crosswalks) + 1,
                    source_endpoint_label=decision.source_endpoint_label,
                    source_action_receipt=action,
                    endpoint_decision_receipt_hash=decision.receipt_hash,
                    hard_fq_endpoint_receipt_hash=(
                        decision.hard_fq_endpoint_receipt_hash
                    ),
                    resource_nominal=decision.resource_nominal,
                    resource_lower=decision.resource_lower,
                    resource_upper=decision.resource_upper,
                    signed_f_nominal=decision.signed_f_nominal,
                    signed_f_lower=decision.signed_f_lower,
                    signed_f_upper=decision.signed_f_upper,
                    route_axis_hash=route_axis_hash,
                    route_count=route_count,
                    parent_mask_hash=parent_mask_hash,
                    canonical_physical_identity_hash=physical,
                    canonical_physical_occurrence=occurrence,
                    first_occurrence_branch=first_branch,
                    first_occurrence_source_endpoint_label=first_label,
                    exact_duplicate_of_earlier_occurrence=occurrence > 1,
                )
            )
        shell = item.shell
        hard = item.hard
        ladder = shell.frozen_ladder
        branch_receipts.append(
            V1SignedWorkLabelNativeCBranchReceipt(
                branch=hard.branch,
                shell_family_hash=shell_family.family_hash,
                shell_execution_hash=shell.execution_hash,
                shell_execution_receipt_hash=shell.execution_receipt.receipt_hash,
                shell_branch_identity_receipt_hash=shell.branch_identity.receipt_hash,
                shell_ladder_receipt_hash=ladder.receipt_hash,
                hard_fq_bundle_hash=hard_fq.bundle_hash,
                hard_fq_family_receipt_hash=hard_receipt.receipt_hash,
                hard_fq_branch_receipt_hash=hard.receipt_hash,
                route_axis_hash=route_axis_hash,
                route_count=route_count,
                parent_mask_hash=parent_mask_hash,
                complete_source_endpoint_counts=ladder.cumulative_counts,
                complete_source_action_receipt_hashes=tuple(
                    action.receipt_hash for action in ladder.actions
                ),
                complete_source_action_hashes=tuple(
                    action.packed_action_hash for action in ladder.actions
                ),
                endpoint_decisions=item.decisions,
                eligible_action_crosswalks=tuple(crosswalks),
                robustly_infeasible_source_endpoint_labels=tuple(
                    decision.source_endpoint_label
                    for decision in item.decisions
                    if decision.canonical_status
                    is V1SignedWorkHardFQEndpointStatus.Q_ROBUST_INFEASIBLE
                ),
            )
        )
    spectral, diagonal = branch_receipts
    frozen_hard_F_order = _freeze_family_hard_F_order(
        spectral.eligible_action_crosswalks
        + diagonal.eligible_action_crosswalks
    )
    family_receipt = V1SignedWorkLabelNativeCFamilyReceipt(
        shell_family_hash=shell_family.family_hash,
        hard_fq_bundle_hash=hard_fq.bundle_hash,
        hard_fq_family_receipt_hash=hard_receipt.receipt_hash,
        route_axis_hash=route_axis_hash,
        route_count=route_count,
        parent_mask_hash=parent_mask_hash,
        spectral_branch_receipt_hash=spectral.receipt_hash,
        diagonal_l1_branch_receipt_hash=diagonal.receipt_hash,
        unique_physical_identity_hashes=tuple(unique_physical),
        pre_C_hard_F_order_crosswalk_receipt_hashes=tuple(
            value.receipt_hash for value in frozen_hard_F_order
        ),
        eligible_action_occurrence_count=sum(
            len(value.eligible_action_crosswalks) for value in branch_receipts
        ),
    )
    return V1SignedWorkLabelNativeCBridgeFamilyBundle(
        spectral=spectral,
        diagonal_l1=diagonal,
        receipt=family_receipt,
    )


__all__ = [
    "LABEL_NATIVE_C_ACTION_CROSSWALK_SCHEMA",
    "LABEL_NATIVE_C_BRANCH_RECEIPT_SCHEMA",
    "LABEL_NATIVE_C_BRIDGE_AUTHORITY",
    "LABEL_NATIVE_C_ENDPOINT_DECISION_SCHEMA",
    "LABEL_NATIVE_C_FAMILY_BUNDLE_SCHEMA",
    "LABEL_NATIVE_C_FAMILY_RECEIPT_SCHEMA",
    "LABEL_NATIVE_C_HARD_F_ORDER_RULE",
    "LABEL_NATIVE_C_PHYSICAL_IDENTITY_SCHEME",
    "V1SignedWorkLabelNativeCActionCrosswalkReceipt",
    "V1SignedWorkLabelNativeCBranchReceipt",
    "V1SignedWorkLabelNativeCBridgeFamilyBundle",
    "V1SignedWorkLabelNativeCBridgeRefusal",
    "V1SignedWorkLabelNativeCBridgeRefusalCode",
    "V1SignedWorkLabelNativeCEndpointDecisionReceipt",
    "V1SignedWorkLabelNativeCFamilyReceipt",
    "freeze_v1_signed_work_label_native_pre_C_bridge",
]
