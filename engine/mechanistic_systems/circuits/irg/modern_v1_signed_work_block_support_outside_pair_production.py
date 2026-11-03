"""Source-native production parents for the outside-pair theorem.

The historical outside-pair dual binder accepts caller-labelled provenance and
raw numerical operands.  Those inputs are useful for bounded reference replay,
but cannot establish that a production lower bound came from the exact
least-work problem and its retained final dual state.  This module closes that
joint without changing the replay surfaces:

* a V2 live-cover authority receipt joins the exact bound witness, canonical
  context, source-native V2 alias, physical CUDA binding, and current live
  cover while excluding every legacy diagnostic coordinate from its authority
  payload; and
* a V2 production dual owner derives the resource/floor/block operator and
  limits from ``V1SignedWorkLeastWorkCUDAProblem`` and derives correctly
  unscaled resource/block multipliers from the selected final KKT dual state.

The resulting objects are still narrow.  They authorize one alias-local raw
outside-family calculation only.  They grant no complete-union, statistical,
F/Q/R, C/K/O, commit, terminal, or Pareto authority.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
import math
from numbers import Real
from threading import RLock
from typing import Any, Final, NoReturn, Sequence

try:  # pragma: no cover - runtime image dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .contracts import content_hash
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_signed_work_block_support_domain_binding import (
    V1BlockSupportAliasCoverBindingV2,
    V1BlockSupportBoundWitnessSource,
    V1BlockSupportCanonicalBlockContextIdentity,
    require_v1_block_support_bound_witness_source,
    require_v1_block_support_canonical_context_identity,
)
from .modern_v1_signed_work_block_support_outside_pair_cuda import (
    V1BlockSupportOutsidePairCUDAExecutionDesign,
    V1BlockSupportOutsidePairExecutionMode,
    V1BlockSupportOutsidePairHBMForecast,
    _gamma_n,
    _nonnegative_arithmetic_upper,
)
from .modern_v1_signed_work_block_support_outside_pair_live_cover import (
    V1BlockSupportOutsidePairLiveCoverProvider,
    V1BlockSupportOutsidePairLiveCoverReceipt,
)
from .modern_v1_signed_work_kkt_cuda import (
    V1SignedWorkLeastWorkCUDABranchReceipt,
    V1SignedWorkLeastWorkCUDAProblem,
    V1SignedWorkLeastWorkCUDAStateHandle,
    _canonical_fused_scales,
)
from .modern_v1_signed_work_physical_alternate_domain_cuda import (
    V1SignedWorkPhysicalAlternateBlockSupportBinding,
)


OUTSIDE_PAIR_LIVE_COVER_AUTHORITY_V2_SCHEMA: Final = (
    "partition_free_v1_block_support_outside_pair_live_cover_authority_v2"
)
OUTSIDE_PAIR_PRODUCTION_DUAL_SOURCE_V2_SCHEMA: Final = (
    "partition_free_v1_block_support_outside_pair_production_dual_source_v2"
)
OUTSIDE_PAIR_PRODUCTION_DUAL_LEASE_V2_SCHEMA: Final = (
    "partition_free_v1_block_support_outside_pair_production_dual_lease_v2"
)
OUTSIDE_PAIR_LIVE_COVER_AUTHORITY_V2_SCOPE: Final = (
    "one_alias_source_native_live_rectangle_cover_only_no_complete_union_"
    "statistical_FQR_C_K_O_commit_terminal_or_pareto_authority_v2"
)
OUTSIDE_PAIR_PRODUCTION_DUAL_SOURCE_V2_SCOPE: Final = (
    "one_alias_exact_problem_final_KKT_resource_block_dual_source_only_no_"
    "solver_optimality_complete_union_statistical_FQR_C_K_O_commit_terminal_"
    "or_pareto_authority_v2"
)
OUTSIDE_PAIR_PRODUCTION_DUAL_UNSCALING_RULE: Final = (
    "lambda_native_equals_work_scale_times_scaled_lambda_over_resource_scale_"
    "and_mu_native_i_equals_work_scale_times_scaled_mu_i_over_sqrt_block_span_v1"
)

_PREFIX: Final = "partition-free-v1-block-support-outside-pair-production-"
_COVER_AUTHORITY_TOKEN: Final = object()
_DUAL_RECEIPT_TOKEN: Final = object()
_DUAL_HANDLE_TOKEN: Final = object()
_DUAL_LEASE_TOKEN: Final = object()
_F64_BYTES: Final = 8
_F64_MIN_NORMAL: Final = float.fromhex("0x1.0p-1022")

_LEGACY_LIVE_COVER_DIAGNOSTIC_FIELDS: Final = frozenset(
    {
        "registered_legacy_pair_cover_hash_diagnostic",
        "legacy_pair_cover_receipt_hash_diagnostic",
        "legacy_pair_axis_commitment_hash_diagnostic",
    }
)


class V1BlockSupportOutsidePairProductionRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    CUDA_REQUIRED = "cuda_required"
    DETERMINISM_REQUIRED = "determinism_required"
    STALE_SOURCE = "stale_source"
    SOURCE_MISMATCH = "source_mismatch"
    CONTEXT_MISMATCH = "context_mismatch"
    ALIAS_MISMATCH = "alias_mismatch"
    KKT_MISMATCH = "KKT_mismatch"
    NONCONVERGED_KKT = "nonconverged_KKT"
    HBM_FORECAST_REFUSED = "HBM_forecast_refused"
    NUMERICAL_REFUSAL = "numerical_refusal"
    MUTATED_CUSTODY = "mutated_custody"
    ACTIVE_LEASE = "active_lease"
    EVICTED = "evicted"
    AUTHORITY_VIOLATION = "authority_violation"


class V1BlockSupportOutsidePairProductionRefusal(RuntimeError):
    def __init__(
        self,
        code: V1BlockSupportOutsidePairProductionRefusalCode,
        detail: str,
    ) -> None:
        if type(code) is not V1BlockSupportOutsidePairProductionRefusalCode:
            raise TypeError("outside-pair production refusal code must be exact")
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")


def _refuse(
    code: V1BlockSupportOutsidePairProductionRefusalCode,
    detail: str,
) -> NoReturn:
    raise V1BlockSupportOutsidePairProductionRefusal(code, detail)


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1BlockSupportOutsidePairProductionRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _integer(value: object, *, name: str, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        _refuse(
            V1BlockSupportOutsidePairProductionRefusalCode.INVALID_TYPE,
            f"{name} must be an exact {'positive' if positive else 'nonnegative'} integer",
        )
    return value


def _finite(value: object, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1BlockSupportOutsidePairProductionRefusalCode.INVALID_TYPE,
            f"{name} must be a real scalar",
        )
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        _refuse(
            V1BlockSupportOutsidePairProductionRefusalCode.NUMERICAL_REFUSAL,
            f"{name} must be finite{' and nonnegative' if nonnegative else ''}",
        )
    return result


def _normalize(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return tuple(_normalize(item) for item in value)
    return value


def _payload(value: object) -> dict[str, object]:
    return {
        item.name: _normalize(getattr(value, item.name))
        for item in fields(value)  # type: ignore[arg-type]
        if not item.name.startswith("_") and item.name != "receipt_hash"
    }


def _closed(values: Sequence[object], *, owner: str) -> None:
    if any(type(value) is not bool or value for value in values):
        _refuse(
            V1BlockSupportOutsidePairProductionRefusalCode.AUTHORITY_VIOLATION,
            f"{owner} crossed its authority boundary",
        )


def _tensor_invariant(value: object) -> tuple[object, ...]:
    if torch is None or type(value) is not torch.Tensor:
        raise TypeError("production dual custody requires exact torch tensors")
    storage = value.untyped_storage()
    return (
        str(value.device),
        str(value.dtype),
        tuple(int(item) for item in value.shape),
        tuple(int(item) for item in value.stride()),
        int(value.storage_offset()),
        int(value.data_ptr()),
        int(storage.data_ptr()),
        int(value._version),
        bool(value.requires_grad),
        bool(value.is_contiguous()),
    )


def _live_cover_source_native_payload(
    receipt: V1BlockSupportOutsidePairLiveCoverReceipt,
) -> dict[str, object]:
    """Return the cover authority payload with legacy diagnostics removed.

    The current live-cover receipt hash cannot be used as a parent here: that
    hash intentionally retains historical replay coordinates.  This explicit
    projection consumes the source-native execution/geometry coordinates,
    including all four rectangle axes, their counts, and their source-native
    commitment, while the V2 alias and bound source are committed separately.
    """

    if type(receipt) is not V1BlockSupportOutsidePairLiveCoverReceipt:
        raise TypeError("source-native cover projection requires the exact receipt")
    # Keep this as an explicit allow-list.  In particular, do not transitively
    # import the V1 ``alias_hash``, registry receipt, legacy source/replay
    # hashes, complete-alternate-domain hashes, or the live-cover receipt hash.
    # Those values remain available on the authority-inert replay receipt, but
    # the source-native authority is reconstructed independently from the
    # bound witness, V2 alias, physical provider, and exact KKT identities.
    names = (
        "design_hash",
        "hbm_forecast_hash",
        "block_index",
        "block_id",
        "block_context_axis_hash",
        "residual_source_context_axis_hash",
        "physical_block_support_binding_receipt_hash",
        "structural_domain_receipt_hash",
        "source_registration_hash",
        "model_identity_hash",
        "chart_identity_hash",
        "current_mask_hash",
        "proposal_generation",
        "route_axis_hash",
        "route_id_axis_hash",
        "resident_device_identity",
        "native_fiber_axis_receipt_hash",
        "native_fiber_measure_hash",
        "registered_d_prefix_hash",
        "registered_d_w2_prefix_hash",
        "preferred_action_binding_hash",
        "comparison_preferred_action_hash",
        "comparison_preferred_removed_route_index_hash",
        "comparison_preferred_selected_count",
        "branch_id",
        "fd_rank",
        "resource_alias_id",
        "cardinality_alias_id",
        "alias_fixed_outside_removed_count",
        "alias_shell_prefix_count",
        "alias_total_removed_count",
        "fixed_membership_binding_hash",
        "fixed_outside_removed_route_index_hash",
        "parent_mask_hash",
        "shell_freeze_receipt_hash",
        "shell_indices",
        "shell_membership_identity_hash",
        "branch_execution_hash",
        "source_rounding_hash",
        "frozen_ladder_receipt_hash",
        "frozen_label_commitment",
        "frozen_eligibility_commitment",
        "frozen_endpoint_action_receipt_hash",
        "frozen_endpoint_packed_action_hash",
        "rounding_anchor_action_coordinate_hash",
        "rounding_anchor_removed_route_index_hash",
        "rounding_anchor_subset_identity_hash",
        "rounding_anchor_selected_shell_offsets",
        "rounding_anchor_shell_route_indices",
        "rounding_anchor_membership_identity_hash",
        "kkt_problem_hash",
        "kkt_policy_hash",
        "kkt_state_identity_hash",
        "kkt_branch_receipt_hash",
        "kkt_eligible_mask_identity_hash",
        "kkt_required_mask_identity_hash",
        "parent_live_membership_identity_hash",
        "selectable_membership_identity_hash",
        "add_outside_drop_axis_identity_hash",
        "add_outside_add_axis_identity_hash",
        "drop_outside_add_inside_drop_axis_identity_hash",
        "drop_outside_add_inside_add_axis_identity_hash",
        "source_native_pair_axis_commitment_hash",
        "route_count",
        "eligible_count",
        "required_count",
        "parent_live_count",
        "selectable_count",
        "fixed_outside_count",
        "shell_count",
        "rounding_anchor_count",
        "add_outside_droppable_count",
        "add_outside_addable_count",
        "drop_outside_add_inside_droppable_count",
        "drop_outside_add_inside_addable_count",
        "add_outside_rectangle_pair_count",
        "drop_outside_add_inside_rectangle_pair_count",
        "complete_pair_count",
        "route_tile_size",
        "route_tile_scan_count",
        "complete_route_scan_count",
        "scalar_device_to_host_transfer_bytes",
        "device_to_host_synchronization_count",
        "route_scale_device_to_host_transfer_bytes",
        "route_scale_content_digest_bytes",
        "owned_route_scale_device_bytes",
        "owned_pair_matrix_bytes",
        "source_native_pair_axis_complete",
        "exact_registered_device_source_join",
        "exact_frozen_rounding_anchor_join",
        "deterministic_physical_index_ties",
        "array_free_receipt",
        "production_cuda_execution",
        "borrowed_membership_not_copied",
        "legacy_hashes_diagnostic_only",
        "legacy_cover_equivalence_proved",
        "downstream_authority",
        "full_u_authority",
        "fqr_authority",
        "C_stream_open_authority",
        "calibration_authority",
        "oracle_authority",
        "commit_authority",
        "pareto_authority",
        "theorem_rule",
        "authority_scope",
        "schema",
    )
    if any(name in _LEGACY_LIVE_COVER_DIAGNOSTIC_FIELDS for name in names):
        raise AssertionError("legacy diagnostics entered the source-native allow-list")
    return {name: _normalize(getattr(receipt, name)) for name in names}


def _live_cover_source_native_payload_hash(
    receipt: V1BlockSupportOutsidePairLiveCoverReceipt,
) -> str:
    return content_hash(
        _live_cover_source_native_payload(receipt),
        prefix=_PREFIX + "live-cover-source-native-payload-",
    )


def _source_native_alias_commitment_hash(
    alias: V1BlockSupportAliasCoverBindingV2,
) -> str:
    """Hash only the V2 live-resource coordinates, never its legacy sibling."""

    if type(alias) is not V1BlockSupportAliasCoverBindingV2:
        raise TypeError("source-native alias commitment requires the exact V2 alias")
    live = alias.live_resource_alias
    return content_hash(
        {
            "schema": "partition_free_v1_block_support_source_native_alias_v2",
            "registry_receipt_hash": alias.registry_receipt_hash,
            "branch_registration_hash": alias.branch_registration_hash,
            "resource_alias_hash": alias.resource_alias_hash,
            "branch": alias.branch.value,
            "resource_alias_id": alias.resource_alias_id,
            "phase_a_source_receipt_hash": alias.source_receipt_hash,
            "phase_a_source_replay_hash": alias.source_replay_hash,
            "source_resource_receipt_hash": alias.source_resource_receipt_hash,
            "resource_coordinate_hash": alias.resource_coordinate_hash,
            "phase_a_resource_cell_id": alias.phase_a_resource_cell_id,
            "phase_a_invariance_receipt_hash": alias.phase_a_invariance_receipt_hash,
            "registered_fd_rank": alias.registered_fd_rank,
            "cardinality_alias_id": alias.cardinality_alias_id,
            "fixed_outside_removed_count": alias.fixed_outside_removed_count,
            "shell_top_k_count": alias.shell_top_k_count,
            "hard_total_removed_count": alias.hard_total_removed_count,
            "registered_shell_resource_floor": (live.registered_shell_resource_floor),
            "hard_endpoint_top_k_rounding_cell": (
                alias.hard_endpoint_top_k_rounding_cell
            ),
            "relaxed_cardinality_equality_used": (
                alias.relaxed_cardinality_equality_used
            ),
            "free_gamma_used": alias.free_gamma_used,
        },
        prefix=_PREFIX + "source-native-alias-",
    )


@dataclass(frozen=True, slots=True)
class V1BlockSupportOutsidePairLiveCoverAuthorityReceiptV2:
    """Authority-safe projection of one exact source-native live cover."""

    bound_witness_registration_receipt_hash: str
    preoutcome_domain_receipt_hash: str
    canonical_block_context_receipt_hash: str
    canonical_context_tuple_hash: str
    block_support_context_axis_hash: str
    live_cover_source_native_payload_hash: str
    source_native_alias_commitment_hash: str
    physical_block_support_binding_receipt_hash: str
    structural_domain_receipt_hash: str
    d_fixed_support_receipt_hash: str
    residual_support_receipt_hash: str
    block_index: int
    block_id: str
    source_registration_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    resident_device_identity: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    preferred_action_hash: str
    preferred_action_binding_hash: str
    preferred_removed_route_index_hash: str
    preferred_selected_count: int
    resource_alias_hash: str
    branch_id: str
    resource_alias_id: str
    phase_a_source_receipt_hash: str
    phase_a_source_replay_hash: str
    source_resource_receipt_hash: str
    resource_coordinate_hash: str
    phase_a_resource_cell_id: str
    phase_a_invariance_receipt_hash: str
    registered_fd_rank: int
    cardinality_alias_id: str
    fixed_outside_removed_count: int
    shell_top_k_count: int
    hard_total_removed_count: int
    registered_shell_resource_floor: float
    rounding_anchor_action_coordinate_hash: str
    rounding_anchor_removed_route_index_hash: str
    rounding_anchor_subset_identity_hash: str
    kkt_problem_hash: str
    kkt_policy_hash: str
    kkt_state_identity_hash: str
    kkt_branch_receipt_hash: str
    kkt_eligible_mask_identity_hash: str
    kkt_required_mask_identity_hash: str
    fixed_membership_binding_hash: str
    parent_live_membership_identity_hash: str
    selectable_membership_identity_hash: str
    add_outside_drop_axis_identity_hash: str
    add_outside_add_axis_identity_hash: str
    drop_outside_add_inside_drop_axis_identity_hash: str
    drop_outside_add_inside_add_axis_identity_hash: str
    source_native_pair_axis_commitment_hash: str
    add_outside_droppable_count: int
    add_outside_addable_count: int
    drop_outside_add_inside_droppable_count: int
    drop_outside_add_inside_addable_count: int
    add_outside_rectangle_pair_count: int
    drop_outside_add_inside_rectangle_pair_count: int
    complete_pair_count: int
    complete_route_scan_count: int
    d_weight_numerator: int
    residual_weight_numerator: int
    population_weight_denominator: int
    receipt_hash: str = field(init=False)
    exact_bound_witness_source_join: bool = True
    exact_canonical_context_join: bool = True
    exact_v2_live_resource_alias_join: bool = True
    exact_physical_live_provider_join: bool = True
    exact_current_live_cover_join: bool = True
    all_four_rectangle_axes_bound: bool = True
    legacy_diagnostic_values_excluded: bool = True
    one_alias_raw_outside_family_authority: bool = True
    complete_union_authority: bool = False
    statistical_authority: bool = False
    canonical_block_observation_authority: bool = False
    full_u_authority: bool = False
    fqr_authority: bool = False
    c_authority: bool = False
    k_authority: bool = False
    o_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = OUTSIDE_PAIR_LIVE_COVER_AUTHORITY_V2_SCOPE
    schema: str = OUTSIDE_PAIR_LIVE_COVER_AUTHORITY_V2_SCHEMA
    _issuer_token: InitVar[object] = None
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _COVER_AUTHORITY_TOKEN:
            raise TypeError("live-cover authority V2 receipts are issuer-only")
        for item in fields(self):
            if (
                not item.name.startswith("_")
                and item.name != "receipt_hash"
                and item.name.endswith(("_hash", "_identity", "_commitment"))
            ):
                object.__setattr__(
                    self,
                    item.name,
                    _text(getattr(self, item.name), name=item.name),
                )
        for name in (
            "block_id",
            "resident_device_identity",
            "branch_id",
            "resource_alias_id",
            "phase_a_resource_cell_id",
            "cardinality_alias_id",
            "authority_scope",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name=name),
            )
        positive = {
            "route_count",
            "registered_fd_rank",
            "preferred_selected_count",
            "population_weight_denominator",
        }
        for name in (
            "block_index",
            "proposal_generation",
            "route_count",
            "preferred_selected_count",
            "registered_fd_rank",
            "fixed_outside_removed_count",
            "shell_top_k_count",
            "hard_total_removed_count",
            "add_outside_droppable_count",
            "add_outside_addable_count",
            "drop_outside_add_inside_droppable_count",
            "drop_outside_add_inside_addable_count",
            "add_outside_rectangle_pair_count",
            "drop_outside_add_inside_rectangle_pair_count",
            "complete_pair_count",
            "complete_route_scan_count",
            "d_weight_numerator",
            "residual_weight_numerator",
            "population_weight_denominator",
        ):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    name=name,
                    positive=name in positive,
                ),
            )
        object.__setattr__(
            self,
            "registered_shell_resource_floor",
            _finite(
                self.registered_shell_resource_floor,
                name="registered shell resource floor",
                nonnegative=True,
            ),
        )
        required = (
            self.exact_bound_witness_source_join,
            self.exact_canonical_context_join,
            self.exact_v2_live_resource_alias_join,
            self.exact_physical_live_provider_join,
            self.exact_current_live_cover_join,
            self.all_four_rectangle_axes_bound,
            self.legacy_diagnostic_values_excluded,
            self.one_alias_raw_outside_family_authority,
        )
        if any(type(value) is not bool or value is not True for value in required):
            _refuse(
                V1BlockSupportOutsidePairProductionRefusalCode.AUTHORITY_VIOLATION,
                "live-cover authority lost a mandatory source-native invariant",
            )
        _closed(
            (
                self.complete_union_authority,
                self.statistical_authority,
                self.canonical_block_observation_authority,
                self.full_u_authority,
                self.fqr_authority,
                self.c_authority,
                self.k_authority,
                self.o_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="live-cover authority V2",
        )
        if (
            self.hard_total_removed_count
            != self.fixed_outside_removed_count + self.shell_top_k_count
            or self.add_outside_rectangle_pair_count
            != self.add_outside_droppable_count * self.add_outside_addable_count
            or self.drop_outside_add_inside_rectangle_pair_count
            != self.drop_outside_add_inside_droppable_count
            * self.drop_outside_add_inside_addable_count
            or self.complete_pair_count
            != self.add_outside_rectangle_pair_count
            + self.drop_outside_add_inside_rectangle_pair_count
            or self.complete_route_scan_count != self.route_count
            or not self.resident_device_identity.startswith("cuda")
            or self.authority_scope != OUTSIDE_PAIR_LIVE_COVER_AUTHORITY_V2_SCOPE
            or self.schema != OUTSIDE_PAIR_LIVE_COVER_AUTHORITY_V2_SCHEMA
        ):
            _refuse(
                V1BlockSupportOutsidePairProductionRefusalCode.AUTHORITY_VIOLATION,
                "live-cover authority counts, device, or semantics do not close",
            )
        observed = content_hash(_payload(self), prefix=_PREFIX + "cover-integrity-")
        if self._integrity_seal and self._integrity_seal != observed:
            _refuse(
                V1BlockSupportOutsidePairProductionRefusalCode.MUTATED_CUSTODY,
                "live-cover authority receipt was mutated",
            )
        object.__setattr__(self, "_integrity_seal", observed)
        object.__setattr__(
            self,
            "receipt_hash",
            content_hash(_payload(self), prefix=_PREFIX + "cover-authority-"),
        )

    def assert_integrity(self) -> str:
        observed = content_hash(_payload(self), prefix=_PREFIX + "cover-integrity-")
        expected = content_hash(_payload(self), prefix=_PREFIX + "cover-authority-")
        if observed != self._integrity_seal or expected != self.receipt_hash:
            _refuse(
                V1BlockSupportOutsidePairProductionRefusalCode.MUTATED_CUSTODY,
                "live-cover authority receipt no longer replays",
            )
        return expected


def bind_v1_block_support_outside_pair_live_cover_authority_v2(
    *,
    bound_witness_source: V1BlockSupportBoundWitnessSource,
    canonical_context_identity: V1BlockSupportCanonicalBlockContextIdentity,
    alias_binding: V1BlockSupportAliasCoverBindingV2,
    physical_block_support_binding: V1SignedWorkPhysicalAlternateBlockSupportBinding,
    live_cover_provider: V1BlockSupportOutsidePairLiveCoverProvider,
    live_cover_receipt: V1BlockSupportOutsidePairLiveCoverReceipt,
) -> V1BlockSupportOutsidePairLiveCoverAuthorityReceiptV2:
    """Promote only the source-native projection of one current live cover."""

    if (
        type(bound_witness_source) is not V1BlockSupportBoundWitnessSource
        or type(canonical_context_identity)
        is not V1BlockSupportCanonicalBlockContextIdentity
        or type(alias_binding) is not V1BlockSupportAliasCoverBindingV2
        or type(physical_block_support_binding)
        is not V1SignedWorkPhysicalAlternateBlockSupportBinding
        or type(live_cover_provider) is not V1BlockSupportOutsidePairLiveCoverProvider
        or type(live_cover_receipt) is not V1BlockSupportOutsidePairLiveCoverReceipt
    ):
        _refuse(
            V1BlockSupportOutsidePairProductionRefusalCode.INVALID_TYPE,
            "cover authority requires exact bound/context/V2-alias/physical/live-cover sources",
        )
    try:
        source = require_v1_block_support_bound_witness_source(bound_witness_source)
        context = require_v1_block_support_canonical_context_identity(
            source=source,
            identity=canonical_context_identity,
        )
        alias_binding.assert_integrity()
        physical_block_support_binding.assert_live()
        live_cover_provider.assert_live()
        live_cover_receipt.assert_integrity()
    except V1BlockSupportOutsidePairProductionRefusal:
        raise
    except (RuntimeError, PermissionError, TypeError, ValueError) as error:
        _refuse(
            V1BlockSupportOutsidePairProductionRefusalCode.STALE_SOURCE,
            f"one live-cover authority source is unavailable: {error}",
        )
    if (
        source.receipt.production_cuda_source_bound is not True
        or source.receipt.source_derived_registration_path is not True
        or source.preoutcome_domain.receipt.production_cuda_source_bound is not True
        or source.preoutcome_domain.structural_provider is None
    ):
        _refuse(
            V1BlockSupportOutsidePairProductionRefusalCode.CUDA_REQUIRED,
            "cover authority requires source-derived production CUDA custody",
        )
    aliases = source.preoutcome_domain.alias_bindings
    if not any(value is alias_binding for value in aliases):
        _refuse(
            V1BlockSupportOutsidePairProductionRefusalCode.ALIAS_MISMATCH,
            "V2 alias is not the exact registered object of the bound source",
        )
    if (
        live_cover_provider.receipt is not live_cover_receipt
        or live_cover_provider.block_witness_source is not source.legacy_source
        or live_cover_provider.alias_binding is not alias_binding.legacy_alias
        or live_cover_provider.physical_block_support_binding
        is not physical_block_support_binding
        or physical_block_support_binding.structural_provider
        is not source.preoutcome_domain.structural_provider
    ):
        _refuse(
            V1BlockSupportOutsidePairProductionRefusalCode.SOURCE_MISMATCH,
            "live owners are not the exact parents joined by the V2 source graph",
        )
    cover = live_cover_receipt
    physical = physical_block_support_binding.receipt
    structural = physical_block_support_binding.structural_provider.receipt
    d_receipt = physical_block_support_binding.d_fixed_support.receipt
    residual_receipt = physical_block_support_binding.residual_support.receipt
    pre = source.preoutcome_domain.receipt
    registration = source.registration
    live_alias = alias_binding.live_resource_alias
    if (
        context.bound_registration_receipt_hash != source.receipt.receipt_hash
        or context.preoutcome_domain_receipt_hash != pre.receipt_hash
        or context.block_index != cover.block_index
        or context.sealed_source_block_id != cover.block_id
        or cover.block_context_axis_hash != context.canonical_context_tuple_hash
        or cover.residual_source_context_axis_hash
        != context.block_support_context_axis_hash
        or physical.residual_block_ordinal != context.block_index
        or residual_receipt.block_ordinal != context.block_index
    ):
        _refuse(
            V1BlockSupportOutsidePairProductionRefusalCode.CONTEXT_MISMATCH,
            "live cover does not replay the exact sealed block context",
        )
    if (
        cover.alias_hash != alias_binding.legacy_alias.alias_hash
        or cover.branch_id != alias_binding.branch.value
        or cover.branch_id != live_alias.branch.value
        or cover.fd_rank != alias_binding.registered_fd_rank
        or cover.resource_alias_id != alias_binding.resource_alias_id
        or cover.source_receipt_hash != alias_binding.source_receipt_hash
        or cover.source_replay_hash != alias_binding.source_replay_hash
        or cover.cardinality_alias_id != alias_binding.cardinality_alias_id
        or cover.alias_fixed_outside_removed_count
        != alias_binding.fixed_outside_removed_count
        or cover.alias_shell_prefix_count != alias_binding.shell_top_k_count
        or cover.alias_total_removed_count != alias_binding.hard_total_removed_count
        or cover.rounding_anchor_count != alias_binding.hard_total_removed_count
        or cover.registry_receipt_hash != alias_binding.registry_receipt_hash
    ):
        _refuse(
            V1BlockSupportOutsidePairProductionRefusalCode.ALIAS_MISMATCH,
            "live cover does not replay the exact V2 live-resource alias",
        )
    common = (
        (cover.block_support_registration_hash, registration.registration_hash),
        (cover.physical_block_support_binding_receipt_hash, physical.receipt_hash),
        (cover.structural_domain_receipt_hash, structural.receipt_hash),
        (cover.structural_domain_receipt_hash, pre.structural_domain_receipt_hash),
        (cover.complete_alternate_domain_hash, pre.complete_alternate_domain_hash),
        (
            cover.complete_alternate_domain_receipt_hash,
            pre.complete_alternate_domain_receipt_hash,
        ),
        (cover.source_registration_hash, pre.source_registration_hash),
        (cover.model_identity_hash, pre.model_identity_hash),
        (cover.chart_identity_hash, pre.chart_identity_hash),
        (cover.current_mask_hash, pre.current_mask_hash),
        (cover.proposal_generation, pre.proposal_generation),
        (cover.route_axis_hash, pre.route_axis_hash),
        (cover.route_id_axis_hash, pre.route_id_axis_hash),
        (cover.route_count, pre.route_count),
        (cover.native_fiber_axis_receipt_hash, pre.native_fiber_axis_receipt_hash),
        (cover.native_fiber_measure_hash, pre.native_fiber_measure_hash),
        (cover.registered_d_prefix_hash, pre.registered_d_prefix_hash),
        (cover.registered_d_w2_prefix_hash, pre.registered_d_w2_prefix_hash),
        (cover.comparison_preferred_action_hash, pre.preferred_action_hash),
        (cover.preferred_action_binding_hash, pre.preferred_action_binding_hash),
        (
            cover.comparison_preferred_removed_route_index_hash,
            pre.preferred_removed_route_index_hash,
        ),
        (cover.comparison_preferred_selected_count, pre.preferred_selected_count),
        (physical.d_fixed_support_receipt_hash, d_receipt.receipt_hash),
        (physical.residual_support_receipt_hash, residual_receipt.receipt_hash),
        (cover.kkt_state_identity_hash, live_cover_provider.kkt_state.identity_hash),
        (cover.kkt_problem_hash, live_cover_provider.kkt_state.problem_hash),
        (cover.kkt_policy_hash, live_cover_provider.kkt_state.policy_hash),
    )
    if any(left != right for left, right in common):
        _refuse(
            V1BlockSupportOutsidePairProductionRefusalCode.SOURCE_MISMATCH,
            "bound/domain/physical/live-cover source identities differ",
        )
    if (
        cover.source_native_pair_axis_complete is not True
        or cover.exact_registered_device_source_join is not True
        or cover.production_cuda_execution is not True
        or cover.complete_route_scan_count != cover.route_count
        or cover.route_scale_device_to_host_transfer_bytes != 0
        or cover.route_scale_content_digest_bytes != 0
        or cover.owned_route_scale_device_bytes != 0
        or cover.owned_pair_matrix_bytes != 0
    ):
        _refuse(
            V1BlockSupportOutsidePairProductionRefusalCode.AUTHORITY_VIOLATION,
            "current live cover lacks complete bounded production custody",
        )
    alias_commitment = _source_native_alias_commitment_hash(alias_binding)
    return V1BlockSupportOutsidePairLiveCoverAuthorityReceiptV2(
        bound_witness_registration_receipt_hash=source.receipt.receipt_hash,
        preoutcome_domain_receipt_hash=pre.receipt_hash,
        canonical_block_context_receipt_hash=context.receipt_hash,
        canonical_context_tuple_hash=context.canonical_context_tuple_hash,
        block_support_context_axis_hash=context.block_support_context_axis_hash,
        live_cover_source_native_payload_hash=(
            _live_cover_source_native_payload_hash(cover)
        ),
        source_native_alias_commitment_hash=alias_commitment,
        physical_block_support_binding_receipt_hash=physical.receipt_hash,
        structural_domain_receipt_hash=structural.receipt_hash,
        d_fixed_support_receipt_hash=d_receipt.receipt_hash,
        residual_support_receipt_hash=residual_receipt.receipt_hash,
        block_index=context.block_index,
        block_id=context.sealed_source_block_id,
        source_registration_hash=pre.source_registration_hash,
        model_identity_hash=pre.model_identity_hash,
        chart_identity_hash=pre.chart_identity_hash,
        current_mask_hash=pre.current_mask_hash,
        proposal_generation=pre.proposal_generation,
        route_axis_hash=pre.route_axis_hash,
        route_id_axis_hash=pre.route_id_axis_hash,
        route_count=pre.route_count,
        resident_device_identity=cover.resident_device_identity,
        native_fiber_axis_receipt_hash=pre.native_fiber_axis_receipt_hash,
        native_fiber_measure_hash=pre.native_fiber_measure_hash,
        registered_d_prefix_hash=pre.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=pre.registered_d_w2_prefix_hash,
        preferred_action_hash=pre.preferred_action_hash,
        preferred_action_binding_hash=pre.preferred_action_binding_hash,
        preferred_removed_route_index_hash=pre.preferred_removed_route_index_hash,
        preferred_selected_count=pre.preferred_selected_count,
        resource_alias_hash=alias_binding.resource_alias_hash,
        branch_id=alias_binding.branch.value,
        resource_alias_id=alias_binding.resource_alias_id,
        phase_a_source_receipt_hash=alias_binding.source_receipt_hash,
        phase_a_source_replay_hash=alias_binding.source_replay_hash,
        source_resource_receipt_hash=alias_binding.source_resource_receipt_hash,
        resource_coordinate_hash=alias_binding.resource_coordinate_hash,
        phase_a_resource_cell_id=alias_binding.phase_a_resource_cell_id,
        phase_a_invariance_receipt_hash=alias_binding.phase_a_invariance_receipt_hash,
        registered_fd_rank=alias_binding.registered_fd_rank,
        cardinality_alias_id=alias_binding.cardinality_alias_id,
        fixed_outside_removed_count=alias_binding.fixed_outside_removed_count,
        shell_top_k_count=alias_binding.shell_top_k_count,
        hard_total_removed_count=alias_binding.hard_total_removed_count,
        registered_shell_resource_floor=live_alias.registered_shell_resource_floor,
        rounding_anchor_action_coordinate_hash=(
            cover.rounding_anchor_action_coordinate_hash
        ),
        rounding_anchor_removed_route_index_hash=(
            cover.rounding_anchor_removed_route_index_hash
        ),
        rounding_anchor_subset_identity_hash=(
            cover.rounding_anchor_subset_identity_hash
        ),
        kkt_problem_hash=cover.kkt_problem_hash,
        kkt_policy_hash=cover.kkt_policy_hash,
        kkt_state_identity_hash=cover.kkt_state_identity_hash,
        kkt_branch_receipt_hash=cover.kkt_branch_receipt_hash,
        kkt_eligible_mask_identity_hash=cover.kkt_eligible_mask_identity_hash,
        kkt_required_mask_identity_hash=cover.kkt_required_mask_identity_hash,
        fixed_membership_binding_hash=cover.fixed_membership_binding_hash,
        parent_live_membership_identity_hash=(
            cover.parent_live_membership_identity_hash
        ),
        selectable_membership_identity_hash=cover.selectable_membership_identity_hash,
        add_outside_drop_axis_identity_hash=cover.add_outside_drop_axis_identity_hash,
        add_outside_add_axis_identity_hash=cover.add_outside_add_axis_identity_hash,
        drop_outside_add_inside_drop_axis_identity_hash=(
            cover.drop_outside_add_inside_drop_axis_identity_hash
        ),
        drop_outside_add_inside_add_axis_identity_hash=(
            cover.drop_outside_add_inside_add_axis_identity_hash
        ),
        source_native_pair_axis_commitment_hash=(
            cover.source_native_pair_axis_commitment_hash
        ),
        add_outside_droppable_count=cover.add_outside_droppable_count,
        add_outside_addable_count=cover.add_outside_addable_count,
        drop_outside_add_inside_droppable_count=(
            cover.drop_outside_add_inside_droppable_count
        ),
        drop_outside_add_inside_addable_count=(
            cover.drop_outside_add_inside_addable_count
        ),
        add_outside_rectangle_pair_count=cover.add_outside_rectangle_pair_count,
        drop_outside_add_inside_rectangle_pair_count=(
            cover.drop_outside_add_inside_rectangle_pair_count
        ),
        complete_pair_count=cover.complete_pair_count,
        complete_route_scan_count=cover.complete_route_scan_count,
        d_weight_numerator=registration.d_weight_numerator,
        residual_weight_numerator=registration.residual_weight_numerator,
        population_weight_denominator=registration.population_weight_denominator,
        _issuer_token=_COVER_AUTHORITY_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class _NativeDualScalars:
    resource_multiplier: float
    block_multipliers: tuple[float, ...]
    capacity_limits: tuple[float, ...]
    resource_multiplier_radius: float
    block_multiplier_radii: tuple[float, ...]
    capacity_adjoint_l1_radius: float
    capacity_adjoint_linf_radius: float
    capacity_multiplier_dot_limit: float
    dual_constant_radius: float
    work_scale: float
    resource_scale: float
    block_scales: tuple[float, ...]


def _scaled_product_quotient_radius(value: float) -> float:
    """Enclose two correctly rounded operations around one observed scalar."""

    observed = _finite(value, name="unscaled dual coordinate", nonnegative=True)
    gamma = _gamma_n(2)
    return math.nextafter(
        (gamma * abs(observed) + 3.0 * _F64_MIN_NORMAL) / (1.0 - gamma),
        math.inf,
    )


def _native_dual_scalars_from_nominals(
    *,
    least_work_problem: V1SignedWorkLeastWorkCUDAProblem,
    resource_multiplier: Real,
    block_multipliers: Sequence[Real],
    work_scale: Real,
    resource_scale: Real,
    block_scales: Sequence[Real],
) -> _NativeDualScalars:
    """Attach outward error accounting to exact observed native coordinates."""

    source = least_work_problem.source_problem
    blocks = source.block_capacities
    resource = _finite(
        resource_multiplier,
        name="native resource multiplier",
        nonnegative=True,
    )
    native_blocks = tuple(
        _finite(value, name="native block multiplier", nonnegative=True)
        for value in block_multipliers
    )
    scales = tuple(
        _finite(value, name="block operator scale", nonnegative=True)
        for value in block_scales
    )
    if len(native_blocks) != len(blocks) or len(scales) != len(blocks):
        _refuse(
            V1BlockSupportOutsidePairProductionRefusalCode.KKT_MISMATCH,
            "native block-dual/scaling axes differ from the exact problem blocks",
        )
    resource_radius = _scaled_product_quotient_radius(resource)
    block_radii = tuple(
        _scaled_product_quotient_radius(value) for value in native_blocks
    )
    capacity_limits = tuple(float(block.removal_capacity) for block in blocks)
    spans = tuple(block.stop - block.start for block in blocks)
    l1_nominal = math.fsum(
        float(span) * radius for span, radius in zip(spans, block_radii, strict=True)
    )
    capacity_l1_radius = (
        0.0
        if not blocks
        else _nonnegative_arithmetic_upper(
            l1_nominal,
            operation_count=max(1, 3 * len(blocks)),
        )
    )
    capacity_linf_radius = max(block_radii, default=0.0)
    capacity_dot = math.fsum(
        multiplier * limit
        for multiplier, limit in zip(
            native_blocks,
            capacity_limits,
            strict=True,
        )
    )
    dot_mass = math.fsum(
        abs(multiplier * limit)
        for multiplier, limit in zip(
            native_blocks,
            capacity_limits,
            strict=True,
        )
    )
    dot_source_radius = math.fsum(
        abs(limit) * radius
        for limit, radius in zip(capacity_limits, block_radii, strict=True)
    )
    dot_radius = (
        0.0
        if not blocks
        else _nonnegative_arithmetic_upper(
            _gamma_n(len(blocks)) * dot_mass
            + dot_source_radius
            + float(len(blocks)) * _F64_MIN_NORMAL,
            operation_count=max(1, 4 * len(blocks)),
        )
    )
    floor = least_work_problem.resource_floor
    lambda_rho_mass = abs(resource * floor)
    lambda_rho_radius = _nonnegative_arithmetic_upper(
        _gamma_n(1) * lambda_rho_mass + abs(floor) * resource_radius + _F64_MIN_NORMAL,
        operation_count=8,
    )
    constant_radius = _nonnegative_arithmetic_upper(
        lambda_rho_radius
        + dot_radius
        + _gamma_n(1) * (lambda_rho_mass + dot_mass)
        + _F64_MIN_NORMAL,
        operation_count=8,
    )
    return _NativeDualScalars(
        resource_multiplier=resource,
        block_multipliers=native_blocks,
        capacity_limits=capacity_limits,
        resource_multiplier_radius=resource_radius,
        block_multiplier_radii=block_radii,
        capacity_adjoint_l1_radius=capacity_l1_radius,
        capacity_adjoint_linf_radius=capacity_linf_radius,
        capacity_multiplier_dot_limit=capacity_dot,
        dual_constant_radius=constant_radius,
        work_scale=_finite(work_scale, name="work operator scale", nonnegative=True),
        resource_scale=_finite(
            resource_scale,
            name="resource operator scale",
            nonnegative=True,
        ),
        block_scales=scales,
    )


def _native_dual_scalars_from_scaled_coordinates(
    *,
    least_work_problem: V1SignedWorkLeastWorkCUDAProblem,
    branch: V1RelaxedKKTBranch,
    scaled_resource_dual: Real,
    scaled_block_duals: Sequence[Real],
) -> _NativeDualScalars:
    """Private reference for the solver's exact physical-dual unscaling.

    This function is intentionally not a production binder.  It accepts raw
    scalars only so focused CPU tests can compare the exported formula against
    the retained solver coordinates.  The public production API below never
    accepts them from a caller.
    """

    if type(least_work_problem) is not V1SignedWorkLeastWorkCUDAProblem:
        raise TypeError("dual unscaling requires the exact least-work problem")
    if type(branch) is not V1RelaxedKKTBranch:
        raise TypeError("dual unscaling requires one exact KKT branch")
    least_work_problem.assert_integrity()
    source = least_work_problem.source_problem
    blocks = source.block_capacities
    raw_resource = _finite(
        scaled_resource_dual,
        name="scaled resource dual",
        nonnegative=True,
    )
    raw_blocks = tuple(
        _finite(value, name="scaled block dual", nonnegative=True)
        for value in scaled_block_duals
    )
    if len(raw_blocks) != len(blocks):
        _refuse(
            V1BlockSupportOutsidePairProductionRefusalCode.KKT_MISMATCH,
            "scaled block-dual axis differs from the exact problem blocks",
        )
    scales = _canonical_fused_scales(
        least_work_problem,
        active_branch=branch,
    )
    block_scales = tuple(float(value) for value in scales.block)
    if len(block_scales) != len(blocks):  # pragma: no cover - solver invariant.
        _refuse(
            V1BlockSupportOutsidePairProductionRefusalCode.KKT_MISMATCH,
            "solver block-scale axis differs from the exact problem",
        )
    resource = _finite(
        raw_resource * scales.work / scales.resource,
        name="native resource multiplier",
        nonnegative=True,
    )
    block_multipliers = tuple(
        _finite(
            raw * scales.work / scale,
            name="native block multiplier",
            nonnegative=True,
        )
        for raw, scale in zip(raw_blocks, block_scales, strict=True)
    )
    return _native_dual_scalars_from_nominals(
        least_work_problem=least_work_problem,
        resource_multiplier=resource,
        block_multipliers=block_multipliers,
        work_scale=float(scales.work),
        resource_scale=float(scales.resource),
        block_scales=block_scales,
    )


@dataclass(frozen=True, slots=True)
class V1BlockSupportOutsidePairProductionDualSourceReceiptV2:
    """Array-free receipt for exact problem/state-derived physical duals."""

    live_cover_authority_receipt_hash: str
    live_cover_source_native_payload_hash: str
    source_native_alias_commitment_hash: str
    least_work_problem_hash: str
    source_problem_hash: str
    kkt_policy_hash: str
    kkt_state_identity_hash: str
    kkt_branch_receipt_hash: str
    signed_work_operator_receipt_hash: str
    source_registration_hash: str
    registered_d_coordinate_hash: str
    block_index: int
    block_id: str
    canonical_block_context_receipt_hash: str
    physical_block_support_binding_receipt_hash: str
    resource_alias_hash: str
    branch_id: str
    resource_alias_id: str
    cardinality_alias_id: str
    registered_fd_rank: int
    route_axis_hash: str
    route_count: int
    kkt_eligible_mask_identity_hash: str
    kkt_required_mask_identity_hash: str
    source_native_pair_axis_commitment_hash: str
    resource_axis_hash: str
    resource_coefficient_source_hash: str
    capacity_operator_source_hash: str
    resource_multiplier_source_hash: str
    capacity_multiplier_source_hash: str
    operator_scales_hash: str
    capacity_multiplier_axis_hash: str
    capacity_limit_axis_hash: str
    route_count_from_problem: int
    capacity_count: int
    resource_multiplier_nominal: float
    resource_floor_nominal: float
    capacity_multiplier_dot_limit_nominal: float
    resource_multiplier_conversion_radius: float
    capacity_adjoint_roundoff_l1_radius: float
    capacity_adjoint_roundoff_linf_radius: float
    dual_constant_roundoff_radius: float
    work_operator_scale: float
    resource_operator_scale: float
    compact_dual_device_to_host_transfer_bytes: int
    compact_dual_device_to_host_synchronization_count: int
    design_hash: str
    hbm_forecast_hash: str
    resident_device_identity: str
    receipt_hash: str = field(init=False)
    exact_problem_source_operators_consumed: bool = True
    exact_final_decision_dual_custody_consumed: bool = True
    selected_branch_converged_production_cuda: bool = True
    resource_and_block_duals_correctly_unscaled: bool = True
    capacity_adjoint_derived_without_matrix: bool = True
    exact_cover_authority_join: bool = True
    alternate_Q_dropped: bool = True
    relaxed_cardinality_equality_enforced: bool = False
    legacy_pair_cover_consumed: bool = False
    caller_source_hash_consumed: bool = False
    caller_raw_operator_consumed: bool = False
    caller_raw_multiplier_consumed: bool = False
    route_scale_device_to_host_transfer_bytes: int = 0
    route_scale_content_digest_bytes: int = 0
    owned_resource_device_bytes: int = 0
    owned_capacity_adjoint_device_bytes: int = field(init=False)
    array_free_receipt: bool = True
    one_alias_raw_outside_family_authority: bool = True
    solver_optimality_authority: bool = False
    complete_union_authority: bool = False
    statistical_authority: bool = False
    canonical_block_observation_authority: bool = False
    full_u_authority: bool = False
    fqr_authority: bool = False
    c_authority: bool = False
    k_authority: bool = False
    o_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    unscaling_rule: str = OUTSIDE_PAIR_PRODUCTION_DUAL_UNSCALING_RULE
    authority_scope: str = OUTSIDE_PAIR_PRODUCTION_DUAL_SOURCE_V2_SCOPE
    schema: str = OUTSIDE_PAIR_PRODUCTION_DUAL_SOURCE_V2_SCHEMA
    _issuer_token: InitVar[object] = None
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _DUAL_RECEIPT_TOKEN:
            raise TypeError("production dual-source V2 receipts are issuer-only")
        for item in fields(self):
            if (
                not item.name.startswith("_")
                and item.name != "receipt_hash"
                and item.name.endswith(("_hash", "_identity", "_commitment"))
            ):
                object.__setattr__(
                    self,
                    item.name,
                    _text(getattr(self, item.name), name=item.name),
                )
        for name in (
            "block_id",
            "branch_id",
            "resource_alias_id",
            "cardinality_alias_id",
            "resident_device_identity",
            "unscaling_rule",
            "authority_scope",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name=name),
            )
        positive = {"registered_fd_rank", "route_count", "route_count_from_problem"}
        for name in (
            "block_index",
            "registered_fd_rank",
            "route_count",
            "route_count_from_problem",
            "capacity_count",
            "compact_dual_device_to_host_transfer_bytes",
            "compact_dual_device_to_host_synchronization_count",
            "route_scale_device_to_host_transfer_bytes",
            "route_scale_content_digest_bytes",
            "owned_resource_device_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    name=name,
                    positive=name in positive,
                ),
            )
        object.__setattr__(
            self,
            "owned_capacity_adjoint_device_bytes",
            self.route_count * _F64_BYTES,
        )
        for name in (
            "resource_multiplier_nominal",
            "resource_floor_nominal",
            "capacity_multiplier_dot_limit_nominal",
            "resource_multiplier_conversion_radius",
            "capacity_adjoint_roundoff_l1_radius",
            "capacity_adjoint_roundoff_linf_radius",
            "dual_constant_roundoff_radius",
            "work_operator_scale",
            "resource_operator_scale",
        ):
            object.__setattr__(
                self,
                name,
                _finite(
                    getattr(self, name),
                    name=name,
                    nonnegative=True,
                ),
            )
        required = (
            self.exact_problem_source_operators_consumed,
            self.exact_final_decision_dual_custody_consumed,
            self.selected_branch_converged_production_cuda,
            self.resource_and_block_duals_correctly_unscaled,
            self.capacity_adjoint_derived_without_matrix,
            self.exact_cover_authority_join,
            self.alternate_Q_dropped,
            self.array_free_receipt,
            self.one_alias_raw_outside_family_authority,
        )
        if any(type(value) is not bool or value is not True for value in required):
            _refuse(
                V1BlockSupportOutsidePairProductionRefusalCode.AUTHORITY_VIOLATION,
                "production dual receipt lost a mandatory source invariant",
            )
        _closed(
            (
                self.relaxed_cardinality_equality_enforced,
                self.legacy_pair_cover_consumed,
                self.caller_source_hash_consumed,
                self.caller_raw_operator_consumed,
                self.caller_raw_multiplier_consumed,
                self.solver_optimality_authority,
                self.complete_union_authority,
                self.statistical_authority,
                self.canonical_block_observation_authority,
                self.full_u_authority,
                self.fqr_authority,
                self.c_authority,
                self.k_authority,
                self.o_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="production outside-pair dual source V2",
        )
        if (
            self.route_count != self.route_count_from_problem
            or self.compact_dual_device_to_host_transfer_bytes
            != (self.capacity_count + 1) * _F64_BYTES
            or self.compact_dual_device_to_host_synchronization_count != 1
            or self.route_scale_device_to_host_transfer_bytes != 0
            or self.route_scale_content_digest_bytes != 0
            or self.owned_resource_device_bytes != 0
            or not self.resident_device_identity.startswith("cuda")
            or self.unscaling_rule != OUTSIDE_PAIR_PRODUCTION_DUAL_UNSCALING_RULE
            or self.authority_scope != OUTSIDE_PAIR_PRODUCTION_DUAL_SOURCE_V2_SCOPE
            or self.schema != OUTSIDE_PAIR_PRODUCTION_DUAL_SOURCE_V2_SCHEMA
        ):
            _refuse(
                V1BlockSupportOutsidePairProductionRefusalCode.AUTHORITY_VIOLATION,
                "production dual transfer, scaling, or authority semantics changed",
            )
        observed = content_hash(_payload(self), prefix=_PREFIX + "dual-integrity-")
        if self._integrity_seal and self._integrity_seal != observed:
            _refuse(
                V1BlockSupportOutsidePairProductionRefusalCode.MUTATED_CUSTODY,
                "production dual receipt was mutated",
            )
        object.__setattr__(self, "_integrity_seal", observed)
        object.__setattr__(
            self,
            "receipt_hash",
            content_hash(_payload(self), prefix=_PREFIX + "dual-receipt-"),
        )

    def assert_integrity(self) -> str:
        observed = content_hash(_payload(self), prefix=_PREFIX + "dual-integrity-")
        expected = content_hash(_payload(self), prefix=_PREFIX + "dual-receipt-")
        if observed != self._integrity_seal or expected != self.receipt_hash:
            _refuse(
                V1BlockSupportOutsidePairProductionRefusalCode.MUTATED_CUSTODY,
                "production dual receipt no longer replays",
            )
        return expected


class V1BlockSupportOutsidePairProductionDualSourceLeaseV2(AbstractContextManager):
    """Short-lived access to one sealed source-native capacity adjoint."""

    __slots__ = (
        "_owner",
        "_token",
        "_closed",
        "dual_source_receipt_hash",
        "mutation_identity_hash",
        "route_scale_device_to_host_transfer_bytes",
        "route_scale_content_digest_bytes",
        "schema",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        owner: "V1BlockSupportOutsidePairProductionDualSourceHandleV2",
        token: object,
        _issuer_token: object,
    ) -> None:
        if (
            _issuer_token is not _DUAL_LEASE_TOKEN
            or type(owner) is not V1BlockSupportOutsidePairProductionDualSourceHandleV2
        ):
            raise TypeError("production dual leases are exact handle-issued objects")
        self._owner = owner
        self._token = token
        self._closed = False
        self.dual_source_receipt_hash = owner.receipt.receipt_hash
        self.mutation_identity_hash = owner.mutation_identity_hash
        self.route_scale_device_to_host_transfer_bytes = 0
        self.route_scale_content_digest_bytes = 0
        self.schema = OUTSIDE_PAIR_PRODUCTION_DUAL_LEASE_V2_SCHEMA

    def _require_open(self) -> None:
        if self._closed:
            _refuse(
                V1BlockSupportOutsidePairProductionRefusalCode.EVICTED,
                "production dual lease is closed",
            )

    def live_capacity_adjoint(self) -> Any:
        self._require_open()
        if self._owner.mutation_identity_hash != self.mutation_identity_hash:
            _refuse(
                V1BlockSupportOutsidePairProductionRefusalCode.MUTATED_CUSTODY,
                "production dual source changed during its lease",
            )
        return self._owner._require_live()

    @property
    def compact_duals(
        self,
    ) -> tuple[float, float, tuple[float, ...], tuple[float, ...]]:
        self.live_capacity_adjoint()
        return self._owner._compact_duals

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._owner._close_lease(self._token)
        finally:
            self._closed = True

    def __enter__(self) -> "V1BlockSupportOutsidePairProductionDualSourceLeaseV2":
        self.live_capacity_adjoint()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        del exc_type, exc, traceback
        self.close()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("production dual leases are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("production dual leases are nonserializable")


class V1BlockSupportOutsidePairProductionDualSourceHandleV2:
    """Live owner of the derived capacity adjoint and exact source graph."""

    __slots__ = (
        "receipt",
        "cover_authority",
        "live_cover_provider",
        "least_work_problem",
        "kkt_state",
        "kkt_branch_receipt",
        "design",
        "forecast",
        "_capacity_adjoint",
        "_capacity_adjoint_guard",
        "_invariant",
        "_guard_invariant",
        "_compact_duals",
        "_active_leases",
        "_lock",
        "_evicted",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        receipt: V1BlockSupportOutsidePairProductionDualSourceReceiptV2,
        cover_authority: V1BlockSupportOutsidePairLiveCoverAuthorityReceiptV2,
        live_cover_provider: V1BlockSupportOutsidePairLiveCoverProvider,
        least_work_problem: V1SignedWorkLeastWorkCUDAProblem,
        kkt_state: V1SignedWorkLeastWorkCUDAStateHandle,
        kkt_branch_receipt: V1SignedWorkLeastWorkCUDABranchReceipt,
        design: V1BlockSupportOutsidePairCUDAExecutionDesign,
        forecast: V1BlockSupportOutsidePairHBMForecast,
        capacity_adjoint: Any,
        compact_duals: tuple[float, float, tuple[float, ...], tuple[float, ...]],
        _issuer_token: object,
    ) -> None:
        if _issuer_token is not _DUAL_HANDLE_TOKEN:
            raise TypeError("production dual-source V2 handles are issuer-only")
        if (
            type(receipt) is not V1BlockSupportOutsidePairProductionDualSourceReceiptV2
            or type(cover_authority)
            is not V1BlockSupportOutsidePairLiveCoverAuthorityReceiptV2
            or type(live_cover_provider)
            is not V1BlockSupportOutsidePairLiveCoverProvider
            or type(least_work_problem) is not V1SignedWorkLeastWorkCUDAProblem
            or type(kkt_state) is not V1SignedWorkLeastWorkCUDAStateHandle
            or type(kkt_branch_receipt) is not V1SignedWorkLeastWorkCUDABranchReceipt
            or type(design) is not V1BlockSupportOutsidePairCUDAExecutionDesign
            or type(forecast) is not V1BlockSupportOutsidePairHBMForecast
            or torch is None
            or type(capacity_adjoint) is not torch.Tensor
        ):
            raise TypeError("production dual-source handle inputs must be exact")
        if (
            capacity_adjoint.dtype is not torch.float64
            or tuple(capacity_adjoint.shape) != (receipt.route_count,)
            or not capacity_adjoint.is_contiguous()
            or str(capacity_adjoint.device) != receipt.resident_device_identity
            or capacity_adjoint.device.type != "cuda"
            or receipt.live_cover_authority_receipt_hash != cover_authority.receipt_hash
            or receipt.least_work_problem_hash != least_work_problem.problem_hash
            or receipt.kkt_state_identity_hash != kkt_state.identity_hash
            or receipt.kkt_branch_receipt_hash != kkt_branch_receipt.receipt_hash
            or receipt.design_hash != design.design_hash
            or receipt.hbm_forecast_hash != forecast.forecast_hash
        ):
            _refuse(
                V1BlockSupportOutsidePairProductionRefusalCode.SOURCE_MISMATCH,
                "production dual handle sources do not replay its receipt",
            )
        self.receipt = receipt
        self.cover_authority = cover_authority
        self.live_cover_provider = live_cover_provider
        self.least_work_problem = least_work_problem
        self.kkt_state = kkt_state
        self.kkt_branch_receipt = kkt_branch_receipt
        self.design = design
        self.forecast = forecast
        self._capacity_adjoint = capacity_adjoint
        self._capacity_adjoint_guard = capacity_adjoint.detach().clone()
        self._invariant = _tensor_invariant(self._capacity_adjoint)
        self._guard_invariant = _tensor_invariant(self._capacity_adjoint_guard)
        self._compact_duals = compact_duals
        self._active_leases: set[object] = set()
        self._lock = RLock()
        self._evicted = False
        self.assert_live()

    def _require_live(self) -> Any:
        with self._lock:
            if self._evicted:
                _refuse(
                    V1BlockSupportOutsidePairProductionRefusalCode.EVICTED,
                    "production dual source was evicted",
                )
            try:
                self.receipt.assert_integrity()
                self.cover_authority.assert_integrity()
                self.live_cover_provider.assert_live()
                self.least_work_problem.assert_integrity()
                state_identity = self.kkt_state.identity_hash
                state_problem = self.kkt_state.problem_hash
                self.kkt_branch_receipt.receipt_hash
            except (RuntimeError, PermissionError, TypeError, ValueError) as error:
                _refuse(
                    V1BlockSupportOutsidePairProductionRefusalCode.STALE_SOURCE,
                    f"one production dual source is stale: {error}",
                )
            if (
                state_identity != self.receipt.kkt_state_identity_hash
                or state_problem != self.receipt.least_work_problem_hash
                or self.live_cover_provider.kkt_state is not self.kkt_state
                or _live_cover_source_native_payload_hash(
                    self.live_cover_provider.receipt
                )
                != self.receipt.live_cover_source_native_payload_hash
                or _tensor_invariant(self._capacity_adjoint) != self._invariant
                or _tensor_invariant(self._capacity_adjoint_guard)
                != self._guard_invariant
                or not bool(
                    torch.equal(
                        self._capacity_adjoint,
                        self._capacity_adjoint_guard,
                    )
                )
            ):
                _refuse(
                    V1BlockSupportOutsidePairProductionRefusalCode.MUTATED_CUSTODY,
                    "production dual source or its exact parents changed",
                )
            return self._capacity_adjoint

    def assert_live(self) -> "V1BlockSupportOutsidePairProductionDualSourceHandleV2":
        self._require_live()
        return self

    @property
    def mutation_identity_hash(self) -> str:
        self._require_live()
        return content_hash(
            (
                self.receipt.receipt_hash,
                self._invariant,
                self._guard_invariant,
            ),
            prefix=_PREFIX + "dual-live-mutation-identity-",
        )

    @property
    def active_lease_count(self) -> int:
        with self._lock:
            return len(self._active_leases)

    def open_dual_lease(
        self,
    ) -> V1BlockSupportOutsidePairProductionDualSourceLeaseV2:
        with self._lock:
            self.assert_live()
            token = object()
            self._active_leases.add(token)
            return V1BlockSupportOutsidePairProductionDualSourceLeaseV2(
                owner=self,
                token=token,
                _issuer_token=_DUAL_LEASE_TOKEN,
            )

    def _close_lease(self, token: object) -> None:
        with self._lock:
            if token not in self._active_leases:
                _refuse(
                    V1BlockSupportOutsidePairProductionRefusalCode.EVICTED,
                    "production dual lease token is not active",
                )
            try:
                self.assert_live()
            finally:
                self._active_leases.remove(token)

    def evict(self) -> None:
        with self._lock:
            self.assert_live()
            if self._active_leases:
                _refuse(
                    V1BlockSupportOutsidePairProductionRefusalCode.ACTIVE_LEASE,
                    "cannot evict a production dual source with active leases",
                )
            self._capacity_adjoint = None
            self._capacity_adjoint_guard = None
            self._invariant = ()
            self._guard_invariant = ()
            self._compact_duals = (0.0, 0.0, (), ())
            self.cover_authority = None
            self.live_cover_provider = None
            self.least_work_problem = None
            self.kkt_state = None
            self.kkt_branch_receipt = None
            self._evicted = True

    def __reduce__(self) -> NoReturn:
        raise TypeError("production dual-source handles are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("production dual-source handles are nonserializable")


def _derived_dual_source_hashes(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    state: V1SignedWorkLeastWorkCUDAStateHandle,
    branch_receipt: V1SignedWorkLeastWorkCUDABranchReceipt,
    native: _NativeDualScalars,
) -> dict[str, str]:
    source = problem.source_problem
    resource_axis = content_hash(
        {
            "schema": "partition_free_v1_outside_pair_exact_problem_resource_axis_v2",
            "source_problem_hash": source.problem_hash,
            "route_axis_hash": source.route_axis_hash,
            "route_count": source.route_count,
        },
        prefix=_PREFIX + "resource-axis-",
    )
    resource_coefficient = content_hash(
        {
            "schema": (
                "partition_free_v1_outside_pair_exact_problem_resource_"
                "coefficient_source_v2"
            ),
            "source_problem_hash": source.problem_hash,
            "signed_work_operator_receipt_hash": (
                problem.signed_work_operator.receipt_hash
            ),
            "registered_d_coordinate_hash": problem.registered_d_coordinate_hash,
            "route_axis_hash": source.route_axis_hash,
            "route_count": source.route_count,
            "coordinate": "least_work_problem_resource_tensor",
        },
        prefix=_PREFIX + "resource-coefficient-source-",
    )
    capacity_operator = content_hash(
        {
            "schema": "partition_free_v1_outside_pair_exact_block_operator_v2",
            "source_problem_hash": source.problem_hash,
            "route_count": source.route_count,
            "blocks": tuple(
                (
                    block.block_hash,
                    block.block_id,
                    block.start,
                    block.stop,
                    block.removal_capacity,
                )
                for block in source.block_capacities
            ),
        },
        prefix=_PREFIX + "capacity-operator-",
    )
    scales_hash = content_hash(
        {
            "unscaling_rule": OUTSIDE_PAIR_PRODUCTION_DUAL_UNSCALING_RULE,
            "work": native.work_scale,
            "resource": native.resource_scale,
            "block": native.block_scales,
        },
        prefix=_PREFIX + "operator-scales-",
    )
    resource_multiplier = content_hash(
        {
            "state_identity_hash": state.identity_hash,
            "branch_receipt_hash": branch_receipt.receipt_hash,
            "operator_scales_hash": scales_hash,
            "coordinate": "final_scaled_resource_dual",
        },
        prefix=_PREFIX + "resource-multiplier-source-",
    )
    capacity_multiplier = content_hash(
        {
            "state_identity_hash": state.identity_hash,
            "branch_receipt_hash": branch_receipt.receipt_hash,
            "capacity_operator_source_hash": capacity_operator,
            "operator_scales_hash": scales_hash,
            "coordinate": "final_scaled_block_dual_axis",
        },
        prefix=_PREFIX + "capacity-multiplier-source-",
    )
    return {
        "resource_axis": resource_axis,
        "resource_coefficient": resource_coefficient,
        "capacity_operator": capacity_operator,
        "scales": scales_hash,
        "resource_multiplier": resource_multiplier,
        "capacity_multiplier": capacity_multiplier,
    }


def bind_v1_block_support_outside_pair_production_dual_source_v2(
    *,
    live_cover_authority: V1BlockSupportOutsidePairLiveCoverAuthorityReceiptV2,
    live_cover_provider: V1BlockSupportOutsidePairLiveCoverProvider,
    least_work_problem: V1SignedWorkLeastWorkCUDAProblem,
    kkt_state: V1SignedWorkLeastWorkCUDAStateHandle,
    kkt_branch_receipt: V1SignedWorkLeastWorkCUDABranchReceipt,
    design: V1BlockSupportOutsidePairCUDAExecutionDesign,
    hbm_forecast: V1BlockSupportOutsidePairHBMForecast,
) -> V1BlockSupportOutsidePairProductionDualSourceHandleV2:
    """Derive one physical outside-pair dual from exact final KKT custody."""

    if torch is None or not torch.cuda.is_available():
        _refuse(
            V1BlockSupportOutsidePairProductionRefusalCode.CUDA_REQUIRED,
            "production dual binding fails closed without CUDA",
        )
    if not torch.are_deterministic_algorithms_enabled():
        _refuse(
            V1BlockSupportOutsidePairProductionRefusalCode.DETERMINISM_REQUIRED,
            "production dual binding requires deterministic algorithms",
        )
    if (
        type(live_cover_authority)
        is not V1BlockSupportOutsidePairLiveCoverAuthorityReceiptV2
        or type(live_cover_provider) is not V1BlockSupportOutsidePairLiveCoverProvider
        or type(least_work_problem) is not V1SignedWorkLeastWorkCUDAProblem
        or type(kkt_state) is not V1SignedWorkLeastWorkCUDAStateHandle
        or type(kkt_branch_receipt) is not V1SignedWorkLeastWorkCUDABranchReceipt
        or type(design) is not V1BlockSupportOutsidePairCUDAExecutionDesign
        or type(hbm_forecast) is not V1BlockSupportOutsidePairHBMForecast
    ):
        _refuse(
            V1BlockSupportOutsidePairProductionRefusalCode.INVALID_TYPE,
            "production dual binding requires exact cover/problem/state/branch/design sources",
        )
    try:
        live_cover_authority.assert_integrity()
        live_cover_provider.assert_live()
        least_work_problem.assert_integrity()
        state_identity = kkt_state.identity_hash
        state_problem = kkt_state.problem_hash
        state_policy = kkt_state.policy_hash
        state_receipts = kkt_state.branch_receipt_hashes
        state_branches = kkt_state.branch_order
        state_route_count = kkt_state.route_count
        state_device = kkt_state.device
        state_eligible = kkt_state.eligible_mask_identity_hash
        state_required = kkt_state.required_mask_identity_hash
        dual_available = kkt_state.signed_work_decision_dual_custody_available
    except (RuntimeError, PermissionError, TypeError, ValueError) as error:
        _refuse(
            V1BlockSupportOutsidePairProductionRefusalCode.STALE_SOURCE,
            f"one production dual source is unavailable: {error}",
        )
    cover = live_cover_provider.receipt
    source = least_work_problem.source_problem
    operator = least_work_problem.signed_work_operator
    try:
        branch = V1RelaxedKKTBranch(live_cover_authority.branch_id)
        branch_index = state_branches.index(branch)
    except (ValueError, TypeError):
        _refuse(
            V1BlockSupportOutsidePairProductionRefusalCode.KKT_MISMATCH,
            "cover alias branch is absent from exact KKT state custody",
        )
    if (
        live_cover_provider.kkt_state is not kkt_state
        or _live_cover_source_native_payload_hash(cover)
        != live_cover_authority.live_cover_source_native_payload_hash
        or live_cover_authority.kkt_problem_hash != least_work_problem.problem_hash
        or least_work_problem.problem_hash != state_problem
        or cover.kkt_problem_hash != state_problem
        or live_cover_authority.kkt_policy_hash != state_policy
        or cover.kkt_policy_hash != state_policy
        or live_cover_authority.kkt_state_identity_hash != state_identity
        or cover.kkt_state_identity_hash != state_identity
        or state_receipts[branch_index] != kkt_branch_receipt.receipt_hash
        or live_cover_authority.kkt_branch_receipt_hash
        != kkt_branch_receipt.receipt_hash
        or cover.kkt_branch_receipt_hash != kkt_branch_receipt.receipt_hash
        or kkt_branch_receipt.branch is not branch
        or kkt_branch_receipt.problem_hash != least_work_problem.problem_hash
        or kkt_branch_receipt.policy_hash != state_policy
        or kkt_branch_receipt.operator_receipt_hash != operator.receipt_hash
        or kkt_branch_receipt.converged is not True
        or kkt_branch_receipt.finite_iterates is not True
        or kkt_branch_receipt.production_cuda_execution is not True
        or kkt_branch_receipt.private_cpu_reference_execution is not False
        or dual_available is not True
    ):
        code = (
            V1BlockSupportOutsidePairProductionRefusalCode.NONCONVERGED_KKT
            if kkt_branch_receipt.converged is not True
            else V1BlockSupportOutsidePairProductionRefusalCode.KKT_MISMATCH
        )
        _refuse(code, "selected final KKT branch/state does not replay the live cover")
    physical = live_cover_provider.physical_block_support_binding.receipt
    if (
        design.execution_mode
        is not V1BlockSupportOutsidePairExecutionMode.CUDA_PRODUCTION
        or hbm_forecast.execution_mode
        is not V1BlockSupportOutsidePairExecutionMode.CUDA_PRODUCTION
        or hbm_forecast.design_hash != design.design_hash
        or hbm_forecast.route_count != source.route_count
        or hbm_forecast.capacity_count != len(source.block_capacities)
        or hbm_forecast.admitted is not True
        or state_device != live_cover_authority.resident_device_identity
        or not state_device.startswith("cuda")
    ):
        _refuse(
            V1BlockSupportOutsidePairProductionRefusalCode.HBM_FORECAST_REFUSED,
            "production dual source lacks one exact admitted CUDA forecast/device",
        )
    if (
        least_work_problem.source_bundle_authoritative is not True
        or source.route_count != live_cover_authority.route_count
        or source.route_count != cover.route_count
        or source.route_count != physical.route_count
        or state_route_count != source.route_count
        or source.route_axis_hash != live_cover_authority.route_axis_hash
        or source.route_axis_hash != cover.route_axis_hash
        or least_work_problem.source_registration_hash
        != live_cover_authority.source_registration_hash
        or least_work_problem.source_registration_hash != cover.source_registration_hash
        or least_work_problem.eligible_mask_identity_hash != state_eligible
        or least_work_problem.required_mask_identity_hash != state_required
        or state_eligible != live_cover_authority.kkt_eligible_mask_identity_hash
        or state_required != live_cover_authority.kkt_required_mask_identity_hash
        or state_eligible != cover.kkt_eligible_mask_identity_hash
        or state_required != cover.kkt_required_mask_identity_hash
        or least_work_problem.resource_floor
        != live_cover_authority.registered_shell_resource_floor
    ):
        _refuse(
            V1BlockSupportOutsidePairProductionRefusalCode.SOURCE_MISMATCH,
            "problem resource/legality/block source does not align with cover/physical custody",
        )
    blocks = source.block_capacities
    if blocks and (
        blocks[0].start != 0
        or blocks[-1].stop != source.route_count
        or any(left.stop != right.start for left, right in zip(blocks, blocks[1:]))
    ):
        _refuse(
            V1BlockSupportOutsidePairProductionRefusalCode.SOURCE_MISMATCH,
            "exact problem block capacity axis is not contiguous and complete",
        )

    with kkt_state.borrow_for_signed_work_decision_witness(
        expected_problem_hash=least_work_problem.problem_hash,
        expected_policy_hash=state_policy,
        expected_branch_receipt_hashes=state_receipts,
    ) as lease:
        resource = lease.source_tensor("resource")
        scaled_resource = lease.decision_dual_tensor("resource_dual", branch)
        scaled_blocks = lease.decision_dual_tensor("block_dual", branch)
        scales = _canonical_fused_scales(
            least_work_problem,
            active_branch=branch,
        )
        if (
            resource.dtype is not torch.float64
            or tuple(resource.shape) != (source.route_count,)
            or str(resource.device) != state_device
            or scaled_resource.dtype is not torch.float64
            or scaled_resource.ndim != 0
            or scaled_resource.device != resource.device
            or scaled_blocks.dtype is not torch.float64
            or tuple(scaled_blocks.shape) != (len(blocks),)
            or scaled_blocks.device != resource.device
        ):
            _refuse(
                V1BlockSupportOutsidePairProductionRefusalCode.MUTATED_CUSTODY,
                "retained problem resource or final decision dual layout changed",
            )
        native_resource_tensor = (
            scaled_resource * float(scales.work) / float(scales.resource)
        )
        if blocks:
            block_scale_tensor = torch.tensor(
                tuple(float(value) for value in scales.block),
                dtype=torch.float64,
                device=resource.device,
            )
            native_block_tensor = (
                scaled_blocks * float(scales.work) / block_scale_tensor
            )
        else:
            native_block_tensor = scaled_blocks.clone()
        capacity_adjoint = torch.zeros_like(resource)
        for block_index, block in enumerate(blocks):
            capacity_adjoint[block.start : block.stop].fill_(
                native_block_tensor[block_index]
            )
        compact_native = torch.cat(
            (native_resource_tensor.reshape(1), native_block_tensor.reshape(-1))
        )
        compact_values = tuple(
            float(value)
            for value in compact_native.to(device="cpu", dtype=torch.float64).tolist()
        )
    if len(compact_values) != len(blocks) + 1:
        _refuse(
            V1BlockSupportOutsidePairProductionRefusalCode.MUTATED_CUSTODY,
            "compact native dual transfer lost its exact block axis",
        )
    resource_multiplier = _finite(
        compact_values[0],
        name="native resource multiplier",
        nonnegative=True,
    )
    block_multipliers = tuple(
        _finite(value, name="native block multiplier", nonnegative=True)
        for value in compact_values[1:]
    )
    # The observed device values are the nominal coordinates used by the scan.
    # Attach the same outward conversion accounting directly to those values;
    # do not reconstruct scaled coordinates through a second CPU round trip.
    native = _native_dual_scalars_from_nominals(
        least_work_problem=least_work_problem,
        resource_multiplier=resource_multiplier,
        block_multipliers=block_multipliers,
        work_scale=float(scales.work),
        resource_scale=float(scales.resource),
        block_scales=tuple(float(item) for item in scales.block),
    )
    hashes = _derived_dual_source_hashes(
        problem=least_work_problem,
        state=kkt_state,
        branch_receipt=kkt_branch_receipt,
        native=native,
    )
    capacity_multiplier_axis_hash = content_hash(
        native.block_multipliers,
        prefix=_PREFIX + "capacity-multiplier-axis-",
    )
    capacity_limit_axis_hash = content_hash(
        native.capacity_limits,
        prefix=_PREFIX + "capacity-limit-axis-",
    )
    receipt = V1BlockSupportOutsidePairProductionDualSourceReceiptV2(
        live_cover_authority_receipt_hash=live_cover_authority.receipt_hash,
        live_cover_source_native_payload_hash=(
            live_cover_authority.live_cover_source_native_payload_hash
        ),
        source_native_alias_commitment_hash=(
            live_cover_authority.source_native_alias_commitment_hash
        ),
        least_work_problem_hash=least_work_problem.problem_hash,
        source_problem_hash=source.problem_hash,
        kkt_policy_hash=state_policy,
        kkt_state_identity_hash=state_identity,
        kkt_branch_receipt_hash=kkt_branch_receipt.receipt_hash,
        signed_work_operator_receipt_hash=operator.receipt_hash,
        source_registration_hash=least_work_problem.source_registration_hash,
        registered_d_coordinate_hash=least_work_problem.registered_d_coordinate_hash,
        block_index=live_cover_authority.block_index,
        block_id=live_cover_authority.block_id,
        canonical_block_context_receipt_hash=(
            live_cover_authority.canonical_block_context_receipt_hash
        ),
        physical_block_support_binding_receipt_hash=(
            live_cover_authority.physical_block_support_binding_receipt_hash
        ),
        resource_alias_hash=live_cover_authority.resource_alias_hash,
        branch_id=live_cover_authority.branch_id,
        resource_alias_id=live_cover_authority.resource_alias_id,
        cardinality_alias_id=live_cover_authority.cardinality_alias_id,
        registered_fd_rank=live_cover_authority.registered_fd_rank,
        route_axis_hash=source.route_axis_hash,
        route_count=source.route_count,
        kkt_eligible_mask_identity_hash=state_eligible,
        kkt_required_mask_identity_hash=state_required,
        source_native_pair_axis_commitment_hash=(
            live_cover_authority.source_native_pair_axis_commitment_hash
        ),
        resource_axis_hash=hashes["resource_axis"],
        resource_coefficient_source_hash=hashes["resource_coefficient"],
        capacity_operator_source_hash=hashes["capacity_operator"],
        resource_multiplier_source_hash=hashes["resource_multiplier"],
        capacity_multiplier_source_hash=hashes["capacity_multiplier"],
        operator_scales_hash=hashes["scales"],
        capacity_multiplier_axis_hash=capacity_multiplier_axis_hash,
        capacity_limit_axis_hash=capacity_limit_axis_hash,
        route_count_from_problem=source.route_count,
        capacity_count=len(blocks),
        resource_multiplier_nominal=native.resource_multiplier,
        resource_floor_nominal=least_work_problem.resource_floor,
        capacity_multiplier_dot_limit_nominal=(native.capacity_multiplier_dot_limit),
        resource_multiplier_conversion_radius=(native.resource_multiplier_radius),
        capacity_adjoint_roundoff_l1_radius=(native.capacity_adjoint_l1_radius),
        capacity_adjoint_roundoff_linf_radius=(native.capacity_adjoint_linf_radius),
        dual_constant_roundoff_radius=native.dual_constant_radius,
        work_operator_scale=native.work_scale,
        resource_operator_scale=native.resource_scale,
        compact_dual_device_to_host_transfer_bytes=(len(blocks) + 1) * _F64_BYTES,
        compact_dual_device_to_host_synchronization_count=1,
        design_hash=design.design_hash,
        hbm_forecast_hash=hbm_forecast.forecast_hash,
        resident_device_identity=state_device,
        _issuer_token=_DUAL_RECEIPT_TOKEN,
    )
    return V1BlockSupportOutsidePairProductionDualSourceHandleV2(
        receipt=receipt,
        cover_authority=live_cover_authority,
        live_cover_provider=live_cover_provider,
        least_work_problem=least_work_problem,
        kkt_state=kkt_state,
        kkt_branch_receipt=kkt_branch_receipt,
        design=design,
        forecast=hbm_forecast,
        capacity_adjoint=capacity_adjoint.detach(),
        compact_duals=(
            native.resource_multiplier,
            least_work_problem.resource_floor,
            native.block_multipliers,
            native.capacity_limits,
        ),
        _issuer_token=_DUAL_HANDLE_TOKEN,
    )


__all__ = (
    "OUTSIDE_PAIR_LIVE_COVER_AUTHORITY_V2_SCHEMA",
    "OUTSIDE_PAIR_LIVE_COVER_AUTHORITY_V2_SCOPE",
    "OUTSIDE_PAIR_PRODUCTION_DUAL_LEASE_V2_SCHEMA",
    "OUTSIDE_PAIR_PRODUCTION_DUAL_SOURCE_V2_SCHEMA",
    "OUTSIDE_PAIR_PRODUCTION_DUAL_SOURCE_V2_SCOPE",
    "OUTSIDE_PAIR_PRODUCTION_DUAL_UNSCALING_RULE",
    "V1BlockSupportOutsidePairLiveCoverAuthorityReceiptV2",
    "V1BlockSupportOutsidePairProductionDualSourceHandleV2",
    "V1BlockSupportOutsidePairProductionDualSourceLeaseV2",
    "V1BlockSupportOutsidePairProductionDualSourceReceiptV2",
    "V1BlockSupportOutsidePairProductionRefusal",
    "V1BlockSupportOutsidePairProductionRefusalCode",
    "bind_v1_block_support_outside_pair_live_cover_authority_v2",
    "bind_v1_block_support_outside_pair_production_dual_source_v2",
)
