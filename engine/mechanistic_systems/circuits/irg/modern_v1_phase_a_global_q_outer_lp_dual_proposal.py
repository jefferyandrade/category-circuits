"""Compact pre-outcome dual proposal for the global true-Q outer LP.

The global true-Q scan needs one nonnegative multiplier for every registered
support plane and one for every compact block-capacity constraint.  Those
multipliers must be frozen before any routewise ``Mv`` coefficient is exposed;
otherwise the same scan could both choose and certify its dual witness.

This module owns only that causal registration and compact tensor custody.  It
derives the incumbent physical action from the exact action-pair source join
and the stable robust-shell identity, and derives the shell and block axes from
their exact owners.  It does *not* inspect a routewise coefficient, establish
dual feasibility, compute an outer-LP upper bound, prove strict outside signs,
or grant D/C/K/O/commit/runtime/Pareto authority.  Those claims belong to the
later bounded population scan.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
import hashlib
from types import MappingProxyType
from typing import Any, Final, Mapping, NoReturn

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask

from .modern_v1_phase_a_global_q_support_family import (
    V1PhaseAGlobalQD1SupportFamilyReceipt,
    V1PhaseAGlobalQD1SupportFamilyResult,
    V1PhaseAGlobalQSupportFamilyHandle,
    V1PhaseAGlobalQSupportFamilyReceipt,
    V1PhaseAGlobalQSupportFamilyResult,
)
from .modern_v1_relaxed_kkt import V1RelaxedKKTProblem
from .modern_v1_sampled_v7_action_pair_alias_source_join import (
    V1SampledV7ActionPairAliasSourceJoinReceipt,
)
from .modern_v1_sampled_v7_phase_a_robust_shell_decision import (
    V1SampledV7PhaseARobustDecisionStatus,
    V1SampledV7PhaseARobustShellDecisionReceipt,
)

try:  # pragma: no cover - runtime-image dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


PHASE_A_GLOBAL_Q_OUTER_LP_DUAL_PROPOSAL_SCHEMA: Final = (
    "partition_free_v1_phase_a_global_true_q_outer_LP_dual_proposal_v1"
)
PHASE_A_GLOBAL_Q_OUTER_LP_DUAL_PROPOSAL_ALGORITHM: Final = (
    "exact_source_joined_stable_physical_action_support_and_block_dual_"
    "multipliers_frozen_before_routewise_mjv_outcomes_v1"
)
PHASE_A_GLOBAL_Q_OUTER_LP_DUAL_PROPOSAL_AUTHORITY: Final = (
    "compact_preoutcome_dual_multiplier_proposal_custody_only_no_dual_"
    "feasibility_outer_bound_outside_sign_D_C_K_O_commit_runtime_terminal_"
    "or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-phase-a-global-q-outer-lp-dual-proposal-"
_RECEIPT_TOKEN: Final = object()
_HANDLE_TOKEN: Final = object()
_LEGACY_PROJECTION_SUPPORT_SOURCE_MODE: Final = "legacy_projection_v1"
_FULL_POPULATION_D1_SUPPORT_SOURCE_MODE: Final = "full_population_d1_v1"


class V1PhaseAGlobalQOuterLPDualProposalRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_SOURCE = "stale_source"
    SOURCE_MISMATCH = "source_mismatch"
    NO_STABLE_PHYSICAL_ACTION = "no_stable_physical_action"
    ACTION_AXIS_MISMATCH = "action_axis_mismatch"
    SHELL_AXIS_MISMATCH = "shell_axis_mismatch"
    NUMERICAL_REFUSAL = "numerical_refusal"
    DEVICE_REFUSAL = "device_refusal"
    MUTATED_CUSTODY = "mutated_custody"
    ACTIVE_LEASE = "active_lease"
    EVICTED = "evicted"
    AUTHORITY_VIOLATION = "authority_violation"


class V1PhaseAGlobalQOuterLPDualProposalRefusal(RuntimeError):
    def __init__(
        self,
        code: V1PhaseAGlobalQOuterLPDualProposalRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1PhaseAGlobalQOuterLPDualProposalRefusalCode:
            raise TypeError("global-Q dual-proposal refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1PhaseAGlobalQOuterLPDualProposalRefusalCode,
    message: str,
) -> NoReturn:
    raise V1PhaseAGlobalQOuterLPDualProposalRefusal(code, message)


def _support_family_source_mode(
    support_family: object,
) -> str:
    """Classify one exact support owner without widening either source graph."""

    if (
        type(support_family) is V1PhaseAGlobalQSupportFamilyResult
        and type(support_family.receipt) is V1PhaseAGlobalQSupportFamilyReceipt
        and type(support_family.device_handle) is V1PhaseAGlobalQSupportFamilyHandle
        and support_family.device_handle.receipt is support_family.receipt
    ):
        return _LEGACY_PROJECTION_SUPPORT_SOURCE_MODE
    if (
        type(support_family) is V1PhaseAGlobalQD1SupportFamilyResult
        and type(support_family.receipt) is V1PhaseAGlobalQD1SupportFamilyReceipt
        and type(support_family.device_handle) is V1PhaseAGlobalQSupportFamilyHandle
        and support_family.device_handle.receipt is support_family.receipt
    ):
        return _FULL_POPULATION_D1_SUPPORT_SOURCE_MODE
    _refuse(
        V1PhaseAGlobalQOuterLPDualProposalRefusalCode.INVALID_TYPE,
        "global-Q dual proposal requires one exact legacy-projection or "
        "full-population-D1 support owner",
    )


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1PhaseAGlobalQOuterLPDualProposalRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty exact string",
        )
    return value


def _identity_axis(
    value: object,
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if type(value) is not tuple:
        _refuse(
            V1PhaseAGlobalQOuterLPDualProposalRefusalCode.INVALID_TYPE,
            f"{name} must be one immutable tuple",
        )
    result = tuple(_identity(item, name=f"{name} entry") for item in value)
    if (not result and not allow_empty) or len(result) != len(set(result)):
        _refuse(
            V1PhaseAGlobalQOuterLPDualProposalRefusalCode.INVALID_TYPE,
            f"{name} must be {'possibly empty, ' if allow_empty else ''}unique",
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value <= 0:
        _refuse(
            V1PhaseAGlobalQOuterLPDualProposalRefusalCode.INVALID_TYPE,
            f"{name} must be one positive exact integer",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        _refuse(
            V1PhaseAGlobalQOuterLPDualProposalRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative exact integer",
        )
    return value


def _payload_hash(payload: Mapping[str, object], *, role: str) -> str:
    return content_hash(dict(payload), prefix=_PREFIX + role + "-")


def _tensor_invariant(value: Any) -> tuple[object, ...]:
    return (
        id(value),
        int(value.data_ptr()),
        tuple(int(item) for item in value.shape),
        tuple(int(item) for item in value.stride()),
        str(value.dtype),
        str(value.device),
        int(value.storage_offset()),
        bool(value.requires_grad),
    )


def _compact_tensor_digest(value: Any, *, role: str) -> str:
    array = value.detach().to(device="cpu").contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(role.encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(repr(tuple(int(item) for item in array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return _payload_hash(
        {
            "role": role,
            "shape": tuple(int(item) for item in array.shape),
            "payload_sha256": digest.hexdigest(),
        },
        role="compact-tensor",
    )


def _outside_direction_id(ordinal: int, action_identity_hash: str) -> str:
    return f"literal-j:outside:action:{ordinal}:{action_identity_hash}"


def _shell_direction_id(route_id: str) -> str:
    return f"literal-j:shared-shell-basis:{route_id}"


def _hard_mask_bit(mask: HardMask, index: int) -> bool:
    """Read one canonical big-endian packed bit without route-scale unpacking."""

    byte = mask.packed_bits[index // 8]
    return bool(byte & (1 << (7 - (index % 8))))


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQOuterLPDualProposalReceipt:
    support_family_receipt_hash: str
    relaxed_kkt_problem_hash: str
    action_pair_alias_source_join_receipt_hash: str
    robust_shell_decision_receipt_hash: str
    action_pair_hash: str
    parent_mask_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    physical_action_identity_hashes: tuple[str, ...]
    action_count: int
    selected_physical_action_ordinal: int
    selected_physical_action_identity_hash: str
    selected_packed_action_hash: str
    selected_resulting_mask_hash: str
    selected_removed_route_index_hash: str
    selected_removed_count: int
    shell_indices: tuple[int, ...]
    shell_route_ids: tuple[str, ...]
    shell_axis_hash: str
    anchor_direction_ids: tuple[str, ...]
    support_plane_count: int
    block_ids: tuple[str, ...]
    block_hashes: tuple[str, ...]
    block_axis_hash: str
    block_count: int
    support_multiplier_digest: str
    block_multiplier_digest: str
    multiplier_scalar_count: int
    guard_scalar_count: int
    compact_live_scalar_count: int
    compact_digest_d2h_scalar_count: int
    measurement_device_identity: str
    production_source_authority: bool
    private_cpu_reference: bool
    multipliers_frozen_before_routewise_mjv_outcomes: bool = True
    routewise_mjv_outcome_consumed: bool = False
    selected_action_derived_from_exact_source_join: bool = True
    shell_axis_derived_from_robust_receipt: bool = True
    support_family_and_problem_bound: bool = True
    immutable_compact_tensor_custody: bool = True
    route_scale_parent_mask_unpack_used: bool = False
    routewise_coefficient_payload_present: bool = False
    dual_feasibility_authority: bool = False
    outer_lp_upper_bound_authority: bool = False
    outside_strict_sign_authority: bool = False
    d_stability_authority: bool = False
    c_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    runtime_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = PHASE_A_GLOBAL_Q_OUTER_LP_DUAL_PROPOSAL_ALGORITHM
    authority_scope: str = PHASE_A_GLOBAL_Q_OUTER_LP_DUAL_PROPOSAL_AUTHORITY
    schema: str = PHASE_A_GLOBAL_Q_OUTER_LP_DUAL_PROPOSAL_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _RECEIPT_TOKEN:
            raise PermissionError("global-Q dual-proposal receipts require their issuer")
        self._validate()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _validate(self) -> None:
        for name in (
            "support_family_receipt_hash",
            "relaxed_kkt_problem_hash",
            "action_pair_alias_source_join_receipt_hash",
            "robust_shell_decision_receipt_hash",
            "action_pair_hash",
            "parent_mask_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "selected_physical_action_identity_hash",
            "selected_packed_action_hash",
            "selected_resulting_mask_hash",
            "selected_removed_route_index_hash",
            "shell_axis_hash",
            "block_axis_hash",
            "support_multiplier_digest",
            "block_multiplier_digest",
            "measurement_device_identity",
        ):
            _identity(getattr(self, name), name=name)
        routes = _positive_int(self.route_count, name="route_count")
        actions = _positive_int(self.action_count, name="action_count")
        ordinal = _positive_int(
            self.selected_physical_action_ordinal,
            name="selected_physical_action_ordinal",
        )
        removed = _positive_int(self.selected_removed_count, name="selected_removed_count")
        supports = _positive_int(self.support_plane_count, name="support_plane_count")
        blocks = _nonnegative_int(self.block_count, name="block_count")
        physical = _identity_axis(
            self.physical_action_identity_hashes,
            name="physical_action_identity_hashes",
        )
        anchors = _identity_axis(
            self.anchor_direction_ids,
            name="anchor_direction_ids",
        )
        block_ids = _identity_axis(self.block_ids, name="block_ids", allow_empty=True)
        block_hashes = _identity_axis(
            self.block_hashes,
            name="block_hashes",
            allow_empty=True,
        )
        if (
            type(self.shell_indices) is not tuple
            or not self.shell_indices
            or any(type(item) is not int for item in self.shell_indices)
            or len(set(self.shell_indices)) != len(self.shell_indices)
            or any(item < 0 or item >= routes for item in self.shell_indices)
        ):
            _refuse(
                V1PhaseAGlobalQOuterLPDualProposalRefusalCode.SHELL_AXIS_MISMATCH,
                "dual-proposal shell indices are not one valid compact route axis",
            )
        shell_ids = _identity_axis(self.shell_route_ids, name="shell_route_ids")
        if len(shell_ids) != len(self.shell_indices):
            _refuse(
                V1PhaseAGlobalQOuterLPDualProposalRefusalCode.SHELL_AXIS_MISMATCH,
                "dual-proposal shell route identities do not align with indices",
            )
        expected_anchors = tuple(
            _outside_direction_id(index, identity)
            for index, identity in enumerate(physical, start=1)
        ) + tuple(_shell_direction_id(item) for item in shell_ids)
        scalar_count = supports + blocks
        if (
            len(physical) != actions
            or not 1 <= ordinal <= actions
            or physical[ordinal - 1] != self.selected_physical_action_identity_hash
            or removed > routes
            or anchors != expected_anchors
            or supports != actions + len(shell_ids)
            or len(block_ids) != blocks
            or len(block_hashes) != blocks
            or self.multiplier_scalar_count != scalar_count
            or self.guard_scalar_count != scalar_count
            or self.compact_live_scalar_count != 2 * scalar_count
            or self.compact_digest_d2h_scalar_count != scalar_count
        ):
            _refuse(
                V1PhaseAGlobalQOuterLPDualProposalRefusalCode.SOURCE_MISMATCH,
                "dual-proposal action, support, block, or compact-scalar axes diverged",
            )
        expected_shell_hash = _payload_hash(
            {
                "shell_indices": self.shell_indices,
                "shell_route_ids": shell_ids,
            },
            role="shell-axis",
        )
        expected_block_hash = _payload_hash(
            {"block_ids": block_ids, "block_hashes": block_hashes},
            role="block-axis",
        )
        if (
            self.shell_axis_hash != expected_shell_hash
            or self.block_axis_hash != expected_block_hash
        ):
            _refuse(
                V1PhaseAGlobalQOuterLPDualProposalRefusalCode.STALE_SOURCE,
                "dual-proposal compact structural axes no longer replay",
            )
        true_flags = (
            self.multipliers_frozen_before_routewise_mjv_outcomes,
            self.selected_action_derived_from_exact_source_join,
            self.shell_axis_derived_from_robust_receipt,
            self.support_family_and_problem_bound,
            self.immutable_compact_tensor_custody,
        )
        false_flags = (
            self.routewise_mjv_outcome_consumed,
            self.routewise_coefficient_payload_present,
            self.route_scale_parent_mask_unpack_used,
            self.dual_feasibility_authority,
            self.outer_lp_upper_bound_authority,
            self.outside_strict_sign_authority,
            self.d_stability_authority,
            self.c_measurement_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.runtime_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            any(type(item) is not bool or item is not True for item in true_flags)
            or any(type(item) is not bool or item is not False for item in false_flags)
            or type(self.production_source_authority) is not bool
            or type(self.private_cpu_reference) is not bool
            or self.production_source_authority == self.private_cpu_reference
            or (
                self.production_source_authority
                and not self.measurement_device_identity.startswith("cuda:")
            )
            or (self.private_cpu_reference and self.measurement_device_identity != "cpu")
            or self.algorithm != PHASE_A_GLOBAL_Q_OUTER_LP_DUAL_PROPOSAL_ALGORITHM
            or self.authority_scope != PHASE_A_GLOBAL_Q_OUTER_LP_DUAL_PROPOSAL_AUTHORITY
            or self.schema != PHASE_A_GLOBAL_Q_OUTER_LP_DUAL_PROPOSAL_SCHEMA
        ):
            _refuse(
                V1PhaseAGlobalQOuterLPDualProposalRefusalCode.AUTHORITY_VIOLATION,
                "dual proposal crossed its pre-outcome compact-custody authority",
            )

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                item.name: getattr(self, item.name)
                for item in fields(self)
                if not item.name.startswith("_")
            }
        )

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="receipt")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self._validate()
        observed = self._compute_hash()
        if observed != self._sealed_hash or (
            expected_hash is not None and observed != expected_hash
        ):
            _refuse(
                V1PhaseAGlobalQOuterLPDualProposalRefusalCode.MUTATED_CUSTODY,
                "global-Q dual-proposal receipt no longer replays",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class V1PhaseAGlobalQOuterLPDualProposalLease:
    __slots__ = (
        "_owner",
        "_token",
        "_closed",
        "support_multipliers",
        "block_multipliers",
    )

    def __init__(
        self,
        owner: "V1PhaseAGlobalQOuterLPDualProposalHandle",
        token: object,
        support_multipliers: Any,
        block_multipliers: Any,
    ) -> None:
        self._owner = owner
        self._token = token
        self._closed = False
        self.support_multipliers = support_multipliers
        self.block_multipliers = block_multipliers

    def __enter__(self) -> "V1PhaseAGlobalQOuterLPDualProposalLease":
        if self._closed:
            _refuse(
                V1PhaseAGlobalQOuterLPDualProposalRefusalCode.EVICTED,
                "global-Q dual-proposal lease is already closed",
            )
        self._owner.assert_live()
        return self

    def close(self) -> None:
        if not self._closed:
            self._owner._close(self._token)
            self._closed = True

    def __exit__(self, *_: object) -> bool:
        self.close()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("global-Q dual-proposal leases are nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("global-Q dual-proposal leases are nonserializable")


class V1PhaseAGlobalQOuterLPDualProposalHandle:
    __slots__ = (
        "receipt",
        "_support_multipliers",
        "_block_multipliers",
        "_guards",
        "_invariants",
        "_active_token",
        "_evicted",
    )

    def __init__(
        self,
        *,
        receipt: V1PhaseAGlobalQOuterLPDualProposalReceipt,
        support_multipliers: Any,
        block_multipliers: Any,
        _issuance_token: object | None = None,
    ) -> None:
        if _issuance_token is not _HANDLE_TOKEN:
            raise PermissionError("global-Q dual-proposal handles require their issuer")
        if type(receipt) is not V1PhaseAGlobalQOuterLPDualProposalReceipt:
            raise TypeError("dual-proposal handle requires its exact receipt")
        if (
            torch is None
            or type(support_multipliers) is not torch.Tensor
            or type(block_multipliers) is not torch.Tensor
        ):
            raise TypeError("dual-proposal numerics require exact torch tensors")
        expected_support = (receipt.support_plane_count,)
        expected_blocks = (receipt.block_count,)
        values = (support_multipliers, block_multipliers)
        if (
            tuple(support_multipliers.shape) != expected_support
            or tuple(block_multipliers.shape) != expected_blocks
            or any(item.dtype is not torch.float64 for item in values)
            or any(not item.is_contiguous() for item in values)
            or support_multipliers.device != block_multipliers.device
            or any(item.requires_grad for item in values)
            or any(not bool(torch.isfinite(item).all().item()) for item in values)
            or any(bool((item < 0.0).any().item()) for item in values)
            or str(support_multipliers.device) != receipt.measurement_device_identity
            or _compact_tensor_digest(
                support_multipliers, role="support-multipliers"
            )
            != receipt.support_multiplier_digest
            or _compact_tensor_digest(block_multipliers, role="block-multipliers")
            != receipt.block_multiplier_digest
        ):
            _refuse(
                V1PhaseAGlobalQOuterLPDualProposalRefusalCode.MUTATED_CUSTODY,
                "dual-proposal compact numerics do not match their receipt",
            )
        self.receipt = receipt
        self._support_multipliers = support_multipliers
        self._block_multipliers = block_multipliers
        self._guards = tuple(item.detach().clone() for item in values)
        self._invariants = tuple(_tensor_invariant(item) for item in values)
        self._active_token: object | None = None
        self._evicted = False
        self.assert_live()

    @property
    def is_evicted(self) -> bool:
        return self._evicted

    def assert_live(self) -> None:
        if self._evicted:
            _refuse(
                V1PhaseAGlobalQOuterLPDualProposalRefusalCode.EVICTED,
                "global-Q dual-proposal handle was evicted",
            )
        self.receipt.assert_integrity()
        values = (self._support_multipliers, self._block_multipliers)
        if any(
            _tensor_invariant(value) != invariant
            or not bool(torch.equal(value, guard))
            for value, invariant, guard in zip(
                values, self._invariants, self._guards, strict=True
            )
        ):
            _refuse(
                V1PhaseAGlobalQOuterLPDualProposalRefusalCode.MUTATED_CUSTODY,
                "global-Q dual-proposal compact numerics mutated",
            )

    def borrow(self) -> V1PhaseAGlobalQOuterLPDualProposalLease:
        self.assert_live()
        if self._active_token is not None:
            _refuse(
                V1PhaseAGlobalQOuterLPDualProposalRefusalCode.ACTIVE_LEASE,
                "only one global-Q dual-proposal lease may be live",
            )
        token = object()
        self._active_token = token
        return V1PhaseAGlobalQOuterLPDualProposalLease(
            self,
            token,
            self._support_multipliers,
            self._block_multipliers,
        )

    def _close(self, token: object) -> None:
        if token is not self._active_token:
            _refuse(
                V1PhaseAGlobalQOuterLPDualProposalRefusalCode.ACTIVE_LEASE,
                "global-Q dual-proposal lease token is stale",
            )
        try:
            self.assert_live()
        finally:
            self._active_token = None

    def evict(self) -> None:
        if self._evicted:
            _refuse(
                V1PhaseAGlobalQOuterLPDualProposalRefusalCode.EVICTED,
                "global-Q dual-proposal handle was already evicted",
            )
        if self._active_token is not None:
            _refuse(
                V1PhaseAGlobalQOuterLPDualProposalRefusalCode.ACTIVE_LEASE,
                "cannot evict global-Q dual proposal under an active lease",
            )
        self.assert_live()
        self._support_multipliers = None
        self._block_multipliers = None
        self._guards = ()
        self._invariants = ()
        self._evicted = True

    def __reduce__(self) -> NoReturn:
        raise TypeError("global-Q dual-proposal handles are nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("global-Q dual-proposal handles are nonserializable")


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQOuterLPDualProposalResult:
    receipt: V1PhaseAGlobalQOuterLPDualProposalReceipt
    device_handle: V1PhaseAGlobalQOuterLPDualProposalHandle = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            type(self.receipt) is not V1PhaseAGlobalQOuterLPDualProposalReceipt
            or type(self.device_handle) is not V1PhaseAGlobalQOuterLPDualProposalHandle
        ):
            raise TypeError("dual-proposal result requires exact components")
        self.device_handle.assert_live()
        if self.device_handle.receipt.receipt_hash != self.receipt.receipt_hash:
            _refuse(
                V1PhaseAGlobalQOuterLPDualProposalRefusalCode.SOURCE_MISMATCH,
                "dual-proposal receipt and live compact numerics diverged",
            )


def _validate_multiplier(
    value: object,
    *,
    size: int,
    device: object,
    name: str,
) -> Any:
    if torch is None or type(value) is not torch.Tensor:
        _refuse(
            V1PhaseAGlobalQOuterLPDualProposalRefusalCode.INVALID_TYPE,
            f"{name} must be one exact torch tensor",
        )
    assert torch is not None
    tensor = value
    if (
        tensor.ndim != 1
        or int(tensor.numel()) != size
        or tensor.dtype is not torch.float64
        or not tensor.is_contiguous()
        or tensor.device != device
        or tensor.requires_grad
    ):
        _refuse(
            V1PhaseAGlobalQOuterLPDualProposalRefusalCode.NUMERICAL_REFUSAL,
            f"{name} must be one contiguous detached float64 vector of length {size} "
            "on the registered support-family device",
        )
    if not bool(torch.isfinite(tensor).all().item()) or bool((tensor < 0.0).any().item()):
        _refuse(
            V1PhaseAGlobalQOuterLPDualProposalRefusalCode.NUMERICAL_REFUSAL,
            f"{name} must be finite and nonnegative",
        )
    return tensor.detach().clone().contiguous()


def issue_v1_phase_a_global_q_outer_lp_dual_proposal(
    *,
    support_family: (
        V1PhaseAGlobalQSupportFamilyResult
        | V1PhaseAGlobalQD1SupportFamilyResult
    ),
    relaxed_kkt_problem: V1RelaxedKKTProblem,
    action_pair_alias_source_join: V1SampledV7ActionPairAliasSourceJoinReceipt,
    robust_shell_decision: V1SampledV7PhaseARobustShellDecisionReceipt,
    parent_mask: HardMask,
    support_multipliers: Any,
    block_multipliers: Any,
    allow_private_cpu_reference_for_tests: bool = False,
) -> V1PhaseAGlobalQOuterLPDualProposalResult:
    """Freeze one compact dual proposal before the routewise true-Q scan."""

    if torch is None:
        raise RuntimeError("global-Q dual-proposal issuance requires PyTorch")
    if (
        type(relaxed_kkt_problem) is not V1RelaxedKKTProblem
        or type(action_pair_alias_source_join)
        is not V1SampledV7ActionPairAliasSourceJoinReceipt
        or type(robust_shell_decision)
        is not V1SampledV7PhaseARobustShellDecisionReceipt
        or type(parent_mask) is not HardMask
        or type(allow_private_cpu_reference_for_tests) is not bool
    ):
        _refuse(
            V1PhaseAGlobalQOuterLPDualProposalRefusalCode.INVALID_TYPE,
            "global-Q dual proposal requires exact source owners",
        )
    support_source_mode = _support_family_source_mode(support_family)
    try:
        support_family.device_handle.assert_live()
        support_hash = support_family.receipt.assert_integrity()
        source_join_hash = action_pair_alias_source_join.assert_integrity()
        robust_hash = robust_shell_decision.assert_integrity()
        pair = action_pair_alias_source_join.action_pair
        pair_hash = pair.assert_integrity()
    except V1PhaseAGlobalQOuterLPDualProposalRefusal:
        raise
    except Exception as error:
        _refuse(
            V1PhaseAGlobalQOuterLPDualProposalRefusalCode.STALE_SOURCE,
            "one dual-proposal source failed replay: "
            f"{type(error).__name__}: {error}",
        )
    support = support_family.receipt
    if support_source_mode == _FULL_POPULATION_D1_SUPPORT_SOURCE_MODE and (
        support.measure_is_source_custody_not_projection is not True
        or support.independent_d1_residual_projection_bound is not True
        or support.action_complete_source_binding_present
        != support.production_source_authority
        or support.private_atomic_reference_nonauthoritative
        != support.private_cpu_reference
    ):
        _refuse(
            V1PhaseAGlobalQOuterLPDualProposalRefusalCode.AUTHORITY_VIOLATION,
            "full-population D1 support crossed its measure/projection or "
            "action-complete source boundary",
        )
    robust = robust_shell_decision
    if robust.status is not V1SampledV7PhaseARobustDecisionStatus.STABLE_PHYSICAL_ACTION_PREMISE:
        _refuse(
            V1PhaseAGlobalQOuterLPDualProposalRefusalCode.NO_STABLE_PHYSICAL_ACTION,
            "dual multipliers require one robust stable physical action premise",
        )
    if (
        robust.private_reference_only is not False
        or robust.source_owned_resource_and_capacity_derivation is not True
        or robust.source_owned_registered_cell_action_quotient is not True
        or robust.exact_action_pair_alias_source_graph_joined is not True
        or robust.source_bound_population_loewner_geometry is not True
        or robust.local_endpoint_action_image_premise_closed is not True
        or robust.action_pair_alias_source_join_receipt_hash != source_join_hash
        or robust.action_pair_hash != pair_hash
        or robust.alias_map_receipt_hash
        != action_pair_alias_source_join.alias_map_receipt_hash
    ):
        _refuse(
            V1PhaseAGlobalQOuterLPDualProposalRefusalCode.SOURCE_MISMATCH,
            "robust stable action does not descend from the supplied exact source join",
        )
    physical = tuple(pair.physical_actions)
    physical_ids = tuple(item.physical_action_identity_hash for item in physical)
    if (
        not physical
        or robust.physical_action_identity_hashes != physical_ids
        or support.action_count != len(physical)
    ):
        _refuse(
            V1PhaseAGlobalQOuterLPDualProposalRefusalCode.ACTION_AXIS_MISMATCH,
            "support, robust, and source-joined physical-action axes diverged",
        )
    selected_identity = robust.selected_physical_action_identity_hash
    matches = tuple(
        item
        for item in physical
        if item.physical_action_identity_hash == selected_identity
    )
    if len(matches) != 1:
        _refuse(
            V1PhaseAGlobalQOuterLPDualProposalRefusalCode.ACTION_AXIS_MISMATCH,
            "robust selected identity has no unique source-joined physical action",
        )
    selected = matches[0]
    selected.assert_integrity()
    child = selected.packed_action.resulting_mask
    try:
        removed_count, removed_hash = child.removed_index_count_and_hash_from(parent_mask)
    except Exception as error:
        _refuse(
            V1PhaseAGlobalQOuterLPDualProposalRefusalCode.ACTION_AXIS_MISMATCH,
            "selected source action is not an irreversible child of the exact parent: "
            f"{type(error).__name__}: {error}",
        )
    if (
        removed_count != selected.removed_count
        or removed_hash != selected.removed_route_index_hash
        or child.mask_hash != selected.resulting_mask_hash
    ):
        _refuse(
            V1PhaseAGlobalQOuterLPDualProposalRefusalCode.ACTION_AXIS_MISMATCH,
            "selected source action delta no longer replays",
        )
    shell_indices = tuple(robust.shell_indices)
    shell_ids = tuple(robust.shell_route_ids)
    expected_anchors = tuple(
        _outside_direction_id(item.physical_action_ordinal, item.physical_action_identity_hash)
        for item in physical
    ) + tuple(_shell_direction_id(item) for item in shell_ids)
    if (
        len(shell_indices) != support.shell_size
        or support.anchor_direction_ids != expected_anchors
        or support.anchor_count != len(expected_anchors)
        or any(index < 0 or index >= relaxed_kkt_problem.route_count for index in shell_indices)
        or any(not _hard_mask_bit(parent_mask, index) for index in shell_indices)
        or any(not bool(relaxed_kkt_problem.eligible_mask[index]) for index in shell_indices)
    ):
        _refuse(
            V1PhaseAGlobalQOuterLPDualProposalRefusalCode.SHELL_AXIS_MISMATCH,
            "support-family anchors do not replay the robust action/shell family",
        )
    if (
        support.relaxed_kkt_problem_hash != relaxed_kkt_problem.problem_hash
        or robust.relaxed_kkt_problem_hash != relaxed_kkt_problem.problem_hash
        or pair.response_geometry_problem_hash != relaxed_kkt_problem.problem_hash
        or pair.response_geometry_source_hash != relaxed_kkt_problem.source_geometry_hash
        or support.pass1_loewner_receipt_hash != robust.loewner_stream_receipt_hash
        or support.route_axis_hash != relaxed_kkt_problem.route_axis_hash
        or robust.route_axis_hash != relaxed_kkt_problem.route_axis_hash
        or pair.route_axis_hash != relaxed_kkt_problem.route_axis_hash
        or parent_mask.route_axis_hash != relaxed_kkt_problem.route_axis_hash
        or support.route_count != relaxed_kkt_problem.route_count
        or robust.route_count != relaxed_kkt_problem.route_count
        or pair.route_count != relaxed_kkt_problem.route_count
        or parent_mask.route_count != relaxed_kkt_problem.route_count
        or support.current_mask_hash != parent_mask.mask_hash
        or robust.current_mask_hash != parent_mask.mask_hash
        or robust.parent_mask_hash != parent_mask.mask_hash
        or pair.parent_mask_hash != parent_mask.mask_hash
        or pair.parent_retained_count != parent_mask.retained_count
        or support.q_budget != relaxed_kkt_problem.quadratic_budget
        or robust.q_budget != relaxed_kkt_problem.quadratic_budget
    ):
        _refuse(
            V1PhaseAGlobalQOuterLPDualProposalRefusalCode.SOURCE_MISMATCH,
            "support family, robust decision, problem, action pair, and parent are not one chart",
        )
    blocks = tuple(relaxed_kkt_problem.block_capacities)
    production = bool(support.production_source_authority)
    if production:
        if support.private_cpu_reference is not False:
            _refuse(
                V1PhaseAGlobalQOuterLPDualProposalRefusalCode.AUTHORITY_VIOLATION,
                "production support family retained private-reference authority",
            )
    elif support.private_cpu_reference is not True or not allow_private_cpu_reference_for_tests:
        _refuse(
            V1PhaseAGlobalQOuterLPDualProposalRefusalCode.DEVICE_REFUSAL,
            "private CPU support geometry requires explicit test-only opt-in",
        )
    with support_family.device_handle.borrow() as source_lease:
        source_device = source_lease.q_upper.device
        if source_lease.rhs_upper.device != source_device:
            _refuse(
                V1PhaseAGlobalQOuterLPDualProposalRefusalCode.DEVICE_REFUSAL,
                "support-family compact tensors do not share one device",
            )
        device_identity = str(source_device)
    if (
        device_identity != support.measurement_device_identity
        or (production and not device_identity.startswith("cuda:"))
        or (not production and device_identity != "cpu")
    ):
        _refuse(
            V1PhaseAGlobalQOuterLPDualProposalRefusalCode.DEVICE_REFUSAL,
            "dual-proposal source device does not match its production/private authority",
        )
    frozen_support = _validate_multiplier(
        support_multipliers,
        size=support.anchor_count,
        device=source_device,
        name="support_multipliers",
    )
    frozen_blocks = _validate_multiplier(
        block_multipliers,
        size=len(blocks),
        device=source_device,
        name="block_multipliers",
    )
    shell_axis_hash = _payload_hash(
        {"shell_indices": shell_indices, "shell_route_ids": shell_ids},
        role="shell-axis",
    )
    block_ids = tuple(item.block_id for item in blocks)
    block_hashes = tuple(item.block_hash for item in blocks)
    block_axis_hash = _payload_hash(
        {"block_ids": block_ids, "block_hashes": block_hashes},
        role="block-axis",
    )
    scalar_count = support.anchor_count + len(blocks)
    receipt = V1PhaseAGlobalQOuterLPDualProposalReceipt(
        support_family_receipt_hash=support_hash,
        relaxed_kkt_problem_hash=relaxed_kkt_problem.problem_hash,
        action_pair_alias_source_join_receipt_hash=source_join_hash,
        robust_shell_decision_receipt_hash=robust_hash,
        action_pair_hash=pair_hash,
        parent_mask_hash=parent_mask.mask_hash,
        model_identity_hash=support.model_identity_hash,
        chart_identity_hash=support.chart_identity_hash,
        current_mask_hash=support.current_mask_hash,
        route_axis_hash=support.route_axis_hash,
        route_id_axis_hash=support.route_id_axis_hash,
        route_count=support.route_count,
        physical_action_identity_hashes=physical_ids,
        action_count=len(physical),
        selected_physical_action_ordinal=selected.physical_action_ordinal,
        selected_physical_action_identity_hash=selected.physical_action_identity_hash,
        selected_packed_action_hash=selected.packed_action_hash,
        selected_resulting_mask_hash=selected.resulting_mask_hash,
        selected_removed_route_index_hash=selected.removed_route_index_hash,
        selected_removed_count=selected.removed_count,
        shell_indices=shell_indices,
        shell_route_ids=shell_ids,
        shell_axis_hash=shell_axis_hash,
        anchor_direction_ids=support.anchor_direction_ids,
        support_plane_count=support.anchor_count,
        block_ids=block_ids,
        block_hashes=block_hashes,
        block_axis_hash=block_axis_hash,
        block_count=len(blocks),
        support_multiplier_digest=_compact_tensor_digest(
            frozen_support,
            role="support-multipliers",
        ),
        block_multiplier_digest=_compact_tensor_digest(
            frozen_blocks,
            role="block-multipliers",
        ),
        multiplier_scalar_count=scalar_count,
        guard_scalar_count=scalar_count,
        compact_live_scalar_count=2 * scalar_count,
        compact_digest_d2h_scalar_count=scalar_count,
        measurement_device_identity=device_identity,
        production_source_authority=production,
        private_cpu_reference=not production,
        _issuance_token=_RECEIPT_TOKEN,
    )
    handle = V1PhaseAGlobalQOuterLPDualProposalHandle(
        receipt=receipt,
        support_multipliers=frozen_support,
        block_multipliers=frozen_blocks,
        _issuance_token=_HANDLE_TOKEN,
    )
    return V1PhaseAGlobalQOuterLPDualProposalResult(
        receipt=receipt,
        device_handle=handle,
    )


__all__ = (
    "PHASE_A_GLOBAL_Q_OUTER_LP_DUAL_PROPOSAL_SCHEMA",
    "PHASE_A_GLOBAL_Q_OUTER_LP_DUAL_PROPOSAL_ALGORITHM",
    "PHASE_A_GLOBAL_Q_OUTER_LP_DUAL_PROPOSAL_AUTHORITY",
    "V1PhaseAGlobalQOuterLPDualProposalRefusalCode",
    "V1PhaseAGlobalQOuterLPDualProposalRefusal",
    "V1PhaseAGlobalQOuterLPDualProposalReceipt",
    "V1PhaseAGlobalQOuterLPDualProposalLease",
    "V1PhaseAGlobalQOuterLPDualProposalHandle",
    "V1PhaseAGlobalQOuterLPDualProposalResult",
    "issue_v1_phase_a_global_q_outer_lp_dual_proposal",
)
