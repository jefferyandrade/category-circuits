"""Acyclic source binding for the canonical signed-work block witness.

The historical block-witness registration accepts several content hashes and
an alias axis directly.  That remains useful for replay, but those strings do
not prove that the registered action domain is the live, source-derived
physical domain.  This module closes that representation joint before any
residual-block outcome exists:

* every historical alias is paired with its exact live Phase-A resource alias;
* hard endpoint cardinality is recorded as ``fixed outside + shell top-k``;
* the structural cell partition is replayed exactly as preferred aliases plus
  physical alternates;
* a typed equivalence names the removed-set/physical-coordinate domain without
  upgrading it to a canonical packed-action hash; and
* a bound witness source proves that the legacy registration consumed this
  exact pre-outcome parent.

The pre-outcome receipt is intentionally block-free.  A bound registration is
its child, never its parent.  Cross-cardinality aliases are retained, but they
remain closed to the current global-P outside-pair argument until an
alias-specific rounding anchor is joined.  No object in this module grants
F/Q/R, C, K, O, commit, terminal, or Pareto authority.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
from numbers import Integral
from typing import Final, NoReturn, Sequence

from .contracts import content_hash
from .modern_v1_signed_work_block_support_source_cuda import (
    V1SignedWorkBlockSupportPreferredActionBinding,
)
from .modern_v1_signed_work_block_support_witness import (
    V1BlockSupportAliasBinding,
    V1BlockSupportDFixedTermReceipt,
    V1BlockSupportWitnessRegistration,
    V1BlockSupportWitnessSource,
    register_v1_signed_work_block_support_witness,
)
from .modern_v1_signed_work_physical_alternate_domain_cuda import (
    V1SignedWorkPhysicalAlternateStructuralCUDADomainProvider,
    V1SignedWorkPhysicalAlternateStructuralCUDADomainReceipt,
)
from .modern_v1_signed_work_variable_shell_action_separation_cuda import (
    V1SignedWorkVariableShellActionCellRegistry,
    V1SignedWorkVariableShellQBranch,
    V1SignedWorkVariableShellResourceAlias,
)
from .response_acquisition import SealedUniformPermutationReceipt


BLOCK_SUPPORT_ALIAS_COVER_BINDING_V2_SCHEMA: Final = (
    "partition_free_v1_block_support_alias_cover_binding_v2"
)
BLOCK_SUPPORT_STRUCTURAL_DOMAIN_EQUIVALENCE_SCHEMA: Final = (
    "partition_free_v1_block_support_structural_coordinate_domain_equivalence_v1"
)
BLOCK_SUPPORT_PREOUTCOME_DOMAIN_BINDING_SCHEMA: Final = (
    "partition_free_v1_block_support_preoutcome_domain_binding_v1"
)
BLOCK_SUPPORT_BOUND_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_block_support_bound_witness_registration_v1"
)
BLOCK_SUPPORT_CANONICAL_CONTEXT_SCHEMA: Final = (
    "partition_free_v1_block_support_canonical_block_context_identity_v1"
)

BLOCK_SUPPORT_CARDINALITY_RULE: Final = (
    "hard_endpoint_top_k_is_a_rounding_cell_label_equal_to_fixed_outside_plus_"
    "shell_prefix_without_relaxed_sum_equality_or_free_gamma_v1"
)
BLOCK_SUPPORT_COORDINATE_EQUIVALENCE_RULE: Final = (
    "source_derived_structural_cells_partition_into_all_preferred_aliases_and_"
    "deduplicated_removed_set_physical_coordinate_alternates_without_packed_"
    "action_hash_upgrade_v1"
)
BLOCK_SUPPORT_PREOUTCOME_DAG_RULE: Final = (
    "structural_domain_then_preoutcome_binding_then_witness_registration_then_"
    "sealed_context_blocks_with_no_reverse_outcome_hash_edge_v1"
)
BLOCK_SUPPORT_DOMAIN_BINDING_AUTHORITY: Final = (
    "structural_preoutcome_and_bound_registration_identity_only_no_complete_"
    "outside_pair_full_U_FQR_C_K_O_commit_terminal_or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-block-support-domain-binding-"
_ALIAS_TOKEN: Final = object()
_EQUIVALENCE_TOKEN: Final = object()
_PREOUTCOME_TOKEN: Final = object()
_BOUND_REGISTRATION_TOKEN: Final = object()
_CONTEXT_TOKEN: Final = object()
_LIVE_PREOUTCOME_TOKEN: Final = object()
_LIVE_BOUND_TOKEN: Final = object()


class V1BlockSupportDomainBindingRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    INVALID_IDENTITY = "invalid_identity"
    STALE_SOURCE = "stale_source"
    SOURCE_MISMATCH = "source_mismatch"
    ALIAS_AXIS_MISMATCH = "alias_axis_mismatch"
    CARDINALITY_MISMATCH = "cardinality_mismatch"
    INCOMPLETE_CELL_PARTITION = "incomplete_structural_cell_partition"
    DOMAIN_EQUIVALENCE_MISMATCH = "domain_equivalence_mismatch"
    REGISTRATION_MISMATCH = "bound_registration_mismatch"
    CROSS_CARDINALITY_OUTSIDE_PAIR_OPEN = "cross_cardinality_outside_pair_open"
    AUTHORITY_LEAKAGE = "authority_leakage"
    RECEIPT_TAMPERING = "receipt_tampering"


class V1BlockSupportDomainBindingRefusal(RuntimeError):
    def __init__(
        self,
        code: V1BlockSupportDomainBindingRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1BlockSupportDomainBindingRefusalCode:
            raise TypeError("domain-binding refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1BlockSupportDomainBindingRefusalCode,
    message: str,
) -> NoReturn:
    raise V1BlockSupportDomainBindingRefusal(code, message)


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1BlockSupportDomainBindingRefusalCode.INVALID_IDENTITY,
            f"{name} must be one nonempty string",
        )
    return value


def _integer(value: object, *, name: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1BlockSupportDomainBindingRefusalCode.INVALID_TYPE,
            f"{name} must be one integer",
        )
    answer = int(value)
    if answer < (1 if positive else 0):
        _refuse(
            V1BlockSupportDomainBindingRefusalCode.INVALID_IDENTITY,
            f"{name} is out of range",
        )
    return answer


def _closed(values: Sequence[object], *, owner: str) -> None:
    if any(type(value) is not bool or value for value in values):
        _refuse(
            V1BlockSupportDomainBindingRefusalCode.AUTHORITY_LEAKAGE,
            f"{owner} attempted to open downstream authority",
        )


def _payload(value: object) -> dict[str, object]:
    return {
        name: (
            getattr(value, name).value
            if isinstance(getattr(value, name), Enum)
            else getattr(value, name)
        )
        for name in value.__dataclass_fields__
        if not name.startswith("_") and name not in {"receipt_hash", "binding_hash"}
    }


def _block_support_context_axis_hash(values: Sequence[str]) -> str:
    """Replay the public wire identity used by block-support source receipts."""

    return content_hash(
        {
            "schema": "partition_free_v1_variable_shell_registered_axis_v1",
            "role": "block_support_context",
            "values": tuple(values),
        },
        prefix="partition-free-v1-variable-shell-source-axis-",
    )


@dataclass(frozen=True, slots=True)
class V1BlockSupportAliasCoverBindingV2:
    """Exact live-resource interpretation of one legacy cover alias."""

    registry_receipt_hash: str
    branch_registration_hash: str
    resource_alias_hash: str
    branch: V1SignedWorkVariableShellQBranch
    resource_alias_id: str
    source_receipt_hash: str
    source_replay_hash: str
    source_resource_receipt_hash: str
    resource_coordinate_hash: str
    phase_a_resource_cell_id: str
    phase_a_invariance_receipt_hash: str
    registered_fd_rank: int
    cardinality_alias_id: str
    fixed_outside_removed_count: int
    shell_top_k_count: int
    hard_total_removed_count: int
    legacy_alias: V1BlockSupportAliasBinding
    live_resource_alias: V1SignedWorkVariableShellResourceAlias
    binding_hash: str = field(init=False)
    _issuer_token: InitVar[object] = None
    hard_endpoint_top_k_rounding_cell: bool = True
    relaxed_cardinality_equality_used: bool = False
    free_gamma_used: bool = False
    alias_specific_rounding_anchor_required: bool = True
    complete_outside_pair_union_authority: bool = False
    full_u_action_domain_authority: bool = False
    fqr_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    cardinality_rule: str = BLOCK_SUPPORT_CARDINALITY_RULE
    schema: str = BLOCK_SUPPORT_ALIAS_COVER_BINDING_V2_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _ALIAS_TOKEN:
            raise TypeError("alias-cover v2 bindings are issuer-only")
        for name in (
            "registry_receipt_hash",
            "branch_registration_hash",
            "resource_alias_hash",
            "resource_alias_id",
            "source_receipt_hash",
            "source_replay_hash",
            "source_resource_receipt_hash",
            "resource_coordinate_hash",
            "phase_a_resource_cell_id",
            "phase_a_invariance_receipt_hash",
            "cardinality_alias_id",
            "cardinality_rule",
            "schema",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        if type(self.branch) is not V1SignedWorkVariableShellQBranch:
            raise TypeError("alias-cover branch must be exact")
        if type(self.legacy_alias) is not V1BlockSupportAliasBinding:
            raise TypeError("alias-cover binding requires the exact legacy alias")
        if type(self.live_resource_alias) is not V1SignedWorkVariableShellResourceAlias:
            raise TypeError("alias-cover binding requires the exact live resource alias")
        rank = _integer(self.registered_fd_rank, name="registered FD rank", positive=True)
        fixed = _integer(
            self.fixed_outside_removed_count,
            name="fixed outside removed count",
        )
        shell = _integer(self.shell_top_k_count, name="shell top-k count")
        total = _integer(self.hard_total_removed_count, name="hard total removed count")
        object.__setattr__(self, "registered_fd_rank", rank)
        object.__setattr__(self, "fixed_outside_removed_count", fixed)
        object.__setattr__(self, "shell_top_k_count", shell)
        object.__setattr__(self, "hard_total_removed_count", total)
        legacy = self.legacy_alias
        live = self.live_resource_alias
        if total != fixed + shell:
            _refuse(
                V1BlockSupportDomainBindingRefusalCode.CARDINALITY_MISMATCH,
                "hard total must equal fixed outside plus the shell top-k endpoint",
            )
        if (
            legacy.branch_id != self.branch.value
            or legacy.branch_id != live.branch.value
            or legacy.resource_alias_id != self.resource_alias_id
            or legacy.resource_alias_id != live.resource_alias_id
            or legacy.source_receipt_hash != self.source_receipt_hash
            or legacy.source_replay_hash != self.source_replay_hash
            or legacy.fd_rank != rank
            or legacy.cardinality_alias_id != self.cardinality_alias_id
            or legacy.cardinality != total
            or live.alias_hash != self.resource_alias_hash
            or live.fixed_outside_removed_count != fixed
            or live.shell_prefix_count != shell
            or live.total_removed_count != total
            or live.source_resource_receipt_hash != self.source_resource_receipt_hash
            or live.resource_coordinate_hash != self.resource_coordinate_hash
            or live.phase_a_resource_cell_id != self.phase_a_resource_cell_id
            or live.phase_a_invariance_receipt_hash
            != self.phase_a_invariance_receipt_hash
        ):
            _refuse(
                V1BlockSupportDomainBindingRefusalCode.ALIAS_AXIS_MISMATCH,
                "legacy alias does not replay its exact live resource coordinate",
            )
        if (
            self.hard_endpoint_top_k_rounding_cell is not True
            or self.alias_specific_rounding_anchor_required is not True
            or self.relaxed_cardinality_equality_used is not False
            or self.free_gamma_used is not False
        ):
            _refuse(
                V1BlockSupportDomainBindingRefusalCode.CARDINALITY_MISMATCH,
                "top-k is only a hard rounding cell; relaxed equality and free gamma are forbidden",
            )
        _closed(
            (
                self.complete_outside_pair_union_authority,
                self.full_u_action_domain_authority,
                self.fqr_authority,
                self.C_stream_open_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="alias-cover binding",
        )
        if (
            self.cardinality_rule != BLOCK_SUPPORT_CARDINALITY_RULE
            or self.schema != BLOCK_SUPPORT_ALIAS_COVER_BINDING_V2_SCHEMA
        ):
            raise ValueError("unsupported alias-cover binding semantics")
        object.__setattr__(
            self,
            "binding_hash",
            content_hash(_payload(self), prefix=_PREFIX + "alias-"),
        )

    def assert_integrity(self) -> str:
        expected = content_hash(_payload(self), prefix=_PREFIX + "alias-")
        if expected != self.binding_hash:
            _refuse(
                V1BlockSupportDomainBindingRefusalCode.RECEIPT_TAMPERING,
                "alias-cover binding no longer replays",
            )
        return expected


@dataclass(frozen=True, slots=True)
class V1BlockSupportStructuralAlternateDomainEquivalenceReceipt:
    """Source-derived within-shell equivalence, never a global-cover claim."""

    structural_domain_receipt_hash: str
    registry_receipt_hash: str
    preferred_action_binding_hash: str
    preferred_action_hash: str
    preferred_removed_route_index_hash: str
    preferred_selected_count: int
    registered_cell_count: int
    preferred_cell_count_excluded: int
    alternate_cell_count: int
    hard_alternate_action_count: int
    preferred_exclusion_axis_hash: str
    alternate_cell_axis_hash: str
    physical_coordinate_axis_hash: str
    complete_alternate_domain_hash: str
    receipt_hash: str = field(init=False)
    _issuer_token: InitVar[object] = None
    complete_structural_cell_partition: bool = True
    preferred_physical_coordinate_excluded: bool = True
    exact_removed_set_physical_coordinate_equivalence: bool = True
    structural_within_shell_domain_only: bool = True
    complete_constraint_cover_source_bound: bool = False
    canonical_packed_action_hash_authority: bool = False
    complete_outside_pair_union_authority: bool = False
    full_u_action_domain_authority: bool = False
    fqr_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    equivalence_rule: str = BLOCK_SUPPORT_COORDINATE_EQUIVALENCE_RULE
    schema: str = BLOCK_SUPPORT_STRUCTURAL_DOMAIN_EQUIVALENCE_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _EQUIVALENCE_TOKEN:
            raise TypeError("structural-domain equivalence receipts are issuer-only")
        for name in (
            "structural_domain_receipt_hash",
            "registry_receipt_hash",
            "preferred_action_binding_hash",
            "preferred_action_hash",
            "preferred_removed_route_index_hash",
            "preferred_exclusion_axis_hash",
            "alternate_cell_axis_hash",
            "physical_coordinate_axis_hash",
            "complete_alternate_domain_hash",
            "equivalence_rule",
            "schema",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        for name in (
            "preferred_selected_count",
            "registered_cell_count",
            "preferred_cell_count_excluded",
            "alternate_cell_count",
            "hard_alternate_action_count",
        ):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name=name, positive=True),
            )
        if (
            self.registered_cell_count
            != self.preferred_cell_count_excluded + self.alternate_cell_count
            or self.complete_structural_cell_partition is not True
            or self.preferred_physical_coordinate_excluded is not True
            or self.exact_removed_set_physical_coordinate_equivalence is not True
            or self.structural_within_shell_domain_only is not True
            or self.complete_constraint_cover_source_bound is not False
        ):
            _refuse(
                V1BlockSupportDomainBindingRefusalCode.INCOMPLETE_CELL_PARTITION,
                "structural preferred/alternate cell partition does not close",
            )
        _closed(
            (
                self.canonical_packed_action_hash_authority,
                self.complete_outside_pair_union_authority,
                self.full_u_action_domain_authority,
                self.fqr_authority,
                self.C_stream_open_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="structural coordinate-domain equivalence",
        )
        if (
            self.equivalence_rule != BLOCK_SUPPORT_COORDINATE_EQUIVALENCE_RULE
            or self.schema != BLOCK_SUPPORT_STRUCTURAL_DOMAIN_EQUIVALENCE_SCHEMA
        ):
            raise ValueError("unsupported structural-domain equivalence")
        object.__setattr__(
            self,
            "receipt_hash",
            content_hash(_payload(self), prefix=_PREFIX + "domain-equivalence-"),
        )

    def assert_integrity(self) -> str:
        expected = content_hash(_payload(self), prefix=_PREFIX + "domain-equivalence-")
        if expected != self.receipt_hash:
            _refuse(
                V1BlockSupportDomainBindingRefusalCode.RECEIPT_TAMPERING,
                "structural-domain equivalence no longer replays",
            )
        return expected


@dataclass(frozen=True, slots=True)
class V1BlockSupportPreOutcomeDomainBindingReceipt:
    """Block-free parent of every later witness registration and outcome."""

    structural_domain_receipt_hash: str
    domain_equivalence_receipt_hash: str
    registry_receipt_hash: str
    source_registration_hash: str
    source_receipt_hashes: tuple[str, ...]
    source_replay_hashes: tuple[str, ...]
    preferred_action_binding_hash: str
    preferred_action_hash: str
    preferred_resulting_mask_hash: str
    preferred_removed_route_index_hash: str
    preferred_selected_count: int
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    complete_alternate_domain_hash: str
    complete_alternate_domain_receipt_hash: str
    hard_alternate_action_count: int
    alias_binding_hashes: tuple[str, ...]
    alias_axis_hash: str
    constraint_union_registration_hash: str
    shared_action_minimum_domain_hash: str
    distinct_hard_total_cardinality_count: int
    production_cuda_source_bound: bool
    private_reference_test_only: bool
    receipt_hash: str = field(init=False)
    _issuer_token: InitVar[object] = None
    complete_alias_axis_bound: bool = True
    complete_structural_cell_partition_bound: bool = True
    cross_cardinality_aliases_structurally_registrable: bool = True
    alias_specific_rounding_anchor_required: bool = True
    structural_within_shell_domain_only: bool = True
    complete_constraint_cover_source_bound: bool = False
    relaxed_cardinality_equality_used: bool = False
    free_gamma_used: bool = False
    canonical_packed_action_hash_authority: bool = False
    complete_outside_pair_union_authority: bool = False
    full_u_action_domain_authority: bool = False
    fqr_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    dag_rule: str = BLOCK_SUPPORT_PREOUTCOME_DAG_RULE
    authority_scope: str = BLOCK_SUPPORT_DOMAIN_BINDING_AUTHORITY
    schema: str = BLOCK_SUPPORT_PREOUTCOME_DOMAIN_BINDING_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _PREOUTCOME_TOKEN:
            raise TypeError("pre-outcome domain-binding receipts are issuer-only")
        for name in (
            "structural_domain_receipt_hash",
            "domain_equivalence_receipt_hash",
            "registry_receipt_hash",
            "source_registration_hash",
            "preferred_action_binding_hash",
            "preferred_action_hash",
            "preferred_resulting_mask_hash",
            "preferred_removed_route_index_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "complete_alternate_domain_hash",
            "complete_alternate_domain_receipt_hash",
            "alias_axis_hash",
            "constraint_union_registration_hash",
            "shared_action_minimum_domain_hash",
            "dag_rule",
            "authority_scope",
            "schema",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        for name in ("proposal_generation",):
            object.__setattr__(self, name, _integer(getattr(self, name), name=name))
        for name in (
            "preferred_selected_count",
            "route_count",
            "hard_alternate_action_count",
            "distinct_hard_total_cardinality_count",
        ):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name=name, positive=True),
            )
        aliases = tuple(
            _text(value, name="alias binding hash")
            for value in self.alias_binding_hashes
        )
        source_receipts = tuple(
            _text(value, name="source receipt hash")
            for value in self.source_receipt_hashes
        )
        source_replays = tuple(
            _text(value, name="source replay hash")
            for value in self.source_replay_hashes
        )
        if (
            not aliases
            or len(set(aliases)) != len(aliases)
            or not source_receipts
            or len(source_receipts) != len(source_replays)
            or len(set(source_receipts)) != len(source_receipts)
            or len(set(source_replays)) != len(source_replays)
        ):
            _refuse(
                V1BlockSupportDomainBindingRefusalCode.ALIAS_AXIS_MISMATCH,
                "pre-outcome alias/source axes are empty, duplicated, or misaligned",
            )
        object.__setattr__(self, "alias_binding_hashes", aliases)
        object.__setattr__(self, "source_receipt_hashes", source_receipts)
        object.__setattr__(self, "source_replay_hashes", source_replays)
        if (
            type(self.production_cuda_source_bound) is not bool
            or type(self.private_reference_test_only) is not bool
            or self.production_cuda_source_bound == self.private_reference_test_only
            or self.complete_alias_axis_bound is not True
            or self.complete_structural_cell_partition_bound is not True
            or self.cross_cardinality_aliases_structurally_registrable is not True
            or self.alias_specific_rounding_anchor_required is not True
            or self.structural_within_shell_domain_only is not True
            or self.complete_constraint_cover_source_bound is not False
            or self.relaxed_cardinality_equality_used is not False
            or self.free_gamma_used is not False
        ):
            _refuse(
                V1BlockSupportDomainBindingRefusalCode.AUTHORITY_LEAKAGE,
                "pre-outcome source mode, alias coverage, or cardinality semantics changed",
            )
        _closed(
            (
                self.canonical_packed_action_hash_authority,
                self.complete_outside_pair_union_authority,
                self.full_u_action_domain_authority,
                self.fqr_authority,
                self.C_stream_open_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="pre-outcome domain binding",
        )
        if (
            self.complete_alternate_domain_receipt_hash
            != self.domain_equivalence_receipt_hash
            or self.dag_rule != BLOCK_SUPPORT_PREOUTCOME_DAG_RULE
            or self.authority_scope != BLOCK_SUPPORT_DOMAIN_BINDING_AUTHORITY
            or self.schema != BLOCK_SUPPORT_PREOUTCOME_DOMAIN_BINDING_SCHEMA
        ):
            _refuse(
                V1BlockSupportDomainBindingRefusalCode.DOMAIN_EQUIVALENCE_MISMATCH,
                "pre-outcome receipt does not name its exact structural equivalence",
            )
        object.__setattr__(
            self,
            "receipt_hash",
            content_hash(_payload(self), prefix=_PREFIX + "preoutcome-"),
        )

    @property
    def cross_cardinality_alias_present(self) -> bool:
        return self.distinct_hard_total_cardinality_count > 1

    def assert_integrity(self) -> str:
        expected = content_hash(_payload(self), prefix=_PREFIX + "preoutcome-")
        if expected != self.receipt_hash:
            _refuse(
                V1BlockSupportDomainBindingRefusalCode.RECEIPT_TAMPERING,
                "pre-outcome receipt no longer replays",
            )
        return expected


def _validate_structural_sources(
    *,
    structural_receipt: V1SignedWorkPhysicalAlternateStructuralCUDADomainReceipt,
    registry: V1SignedWorkVariableShellActionCellRegistry,
    preferred_action_binding: V1SignedWorkBlockSupportPreferredActionBinding,
) -> tuple[object, ...]:
    if (
        type(structural_receipt)
        is not V1SignedWorkPhysicalAlternateStructuralCUDADomainReceipt
        or type(registry) is not V1SignedWorkVariableShellActionCellRegistry
        or type(preferred_action_binding)
        is not V1SignedWorkBlockSupportPreferredActionBinding
    ):
        raise TypeError("pre-outcome binding requires exact structural sources")
    try:
        structural_receipt.assert_integrity()
        handles = registry.assert_live_sources()
        preferred_action_binding.assert_live()
    except (RuntimeError, PermissionError, TypeError, ValueError) as error:
        _refuse(
            V1BlockSupportDomainBindingRefusalCode.STALE_SOURCE,
            f"one structural source is stale: {error}",
        )
    registry_receipt = registry.receipt
    if (
        structural_receipt.registry_receipt_hash != registry_receipt.receipt_hash
        or structural_receipt.preferred_action_binding_hash
        != preferred_action_binding.binding_hash
        or structural_receipt.preferred_action_hash
        != preferred_action_binding.preferred_action_hash
        or structural_receipt.preferred_resulting_mask_hash
        != preferred_action_binding.preferred_resulting_mask_hash
        or structural_receipt.preferred_removed_route_index_hash
        != preferred_action_binding.preferred_removed_route_index_hash
        or structural_receipt.preferred_selected_count
        != preferred_action_binding.preferred_selected_count
        or structural_receipt.source_registration_hash
        != preferred_action_binding.source_registration_hash
        or structural_receipt.model_identity_hash
        != preferred_action_binding.model_identity_hash
        or structural_receipt.chart_identity_hash
        != preferred_action_binding.chart_identity_hash
        or structural_receipt.current_mask_hash
        != preferred_action_binding.current_mask_hash
        or structural_receipt.proposal_generation
        != preferred_action_binding.proposal_generation
        or structural_receipt.route_axis_hash
        != preferred_action_binding.route_axis_hash
        or structural_receipt.route_count != preferred_action_binding.route_count
    ):
        _refuse(
            V1BlockSupportDomainBindingRefusalCode.SOURCE_MISMATCH,
            "structural receipt and preferred packed-action custody differ",
        )
    common = (
        "model_identity_hash",
        "chart_identity_hash",
        "current_mask_hash",
        "proposal_generation",
        "route_axis_hash",
        "route_id_axis_hash",
        "registered_d_prefix_hash",
        "registered_d_w2_prefix_hash",
        "native_fiber_axis_receipt_hash",
        "native_fiber_measure_hash",
    )
    if any(
        getattr(structural_receipt, name) != getattr(registry_receipt, name)
        for name in common
    ) or any(
        any(getattr(handle.receipt, name) != getattr(registry_receipt, name) for name in common)
        for handle in handles
    ):
        _refuse(
            V1BlockSupportDomainBindingRefusalCode.SOURCE_MISMATCH,
            "registry branches do not share the structural chart/mask/route/prefix identity",
        )
    if structural_receipt.canonical_packed_action_hash_authority is not False:
        _refuse(
            V1BlockSupportDomainBindingRefusalCode.AUTHORITY_LEAKAGE,
            "structural coordinate custody may not impersonate packed-action hashes",
        )
    return handles


def _issue_alias_binding(
    *,
    registry: V1SignedWorkVariableShellActionCellRegistry,
    legacy_alias: V1BlockSupportAliasBinding,
) -> V1BlockSupportAliasCoverBindingV2:
    if type(legacy_alias) is not V1BlockSupportAliasBinding:
        raise TypeError("alias binding requires the exact legacy alias")
    try:
        branch = V1SignedWorkVariableShellQBranch(legacy_alias.branch_id)
    except ValueError:
        _refuse(
            V1BlockSupportDomainBindingRefusalCode.ALIAS_AXIS_MISMATCH,
            "legacy alias branch is not a registered Q branch",
        )
    source = registry.branch_source(branch)
    live = source.resource_alias(legacy_alias.resource_alias_id)
    receipt = source.assert_live_source().receipt
    if legacy_alias.source_receipt_hash != receipt.receipt_hash:
        _refuse(
            V1BlockSupportDomainBindingRefusalCode.SOURCE_MISMATCH,
            "legacy alias names another live source receipt",
        )
    if legacy_alias.source_replay_hash != receipt.source_replay_hash:
        _refuse(
            V1BlockSupportDomainBindingRefusalCode.SOURCE_MISMATCH,
            "legacy alias names another live source replay",
        )
    return V1BlockSupportAliasCoverBindingV2(
        registry_receipt_hash=registry.receipt.receipt_hash,
        branch_registration_hash=source.registration.registration_hash,
        resource_alias_hash=live.alias_hash,
        branch=branch,
        resource_alias_id=live.resource_alias_id,
        source_receipt_hash=receipt.receipt_hash,
        source_replay_hash=receipt.source_replay_hash,
        source_resource_receipt_hash=live.source_resource_receipt_hash,
        resource_coordinate_hash=live.resource_coordinate_hash,
        phase_a_resource_cell_id=live.phase_a_resource_cell_id,
        phase_a_invariance_receipt_hash=live.phase_a_invariance_receipt_hash,
        registered_fd_rank=legacy_alias.fd_rank,
        cardinality_alias_id=legacy_alias.cardinality_alias_id,
        fixed_outside_removed_count=live.fixed_outside_removed_count,
        shell_top_k_count=live.shell_prefix_count,
        hard_total_removed_count=live.total_removed_count,
        legacy_alias=legacy_alias,
        live_resource_alias=live,
        _issuer_token=_ALIAS_TOKEN,
    )


def _issue_domain_equivalence(
    *,
    structural_receipt: V1SignedWorkPhysicalAlternateStructuralCUDADomainReceipt,
    registry: V1SignedWorkVariableShellActionCellRegistry,
    preferred_action_binding: V1SignedWorkBlockSupportPreferredActionBinding,
) -> V1BlockSupportStructuralAlternateDomainEquivalenceReceipt:
    _validate_structural_sources(
        structural_receipt=structural_receipt,
        registry=registry,
        preferred_action_binding=preferred_action_binding,
    )
    cells = tuple(registry.iter_action_cells())
    cell_by_hash = {cell.action_cell_hash: cell for cell in cells}
    excluded = tuple(structural_receipt.excluded_preferred_action_cell_hashes)
    crosswalk = tuple(structural_receipt.cell_crosswalk_axis)
    alternate_hashes = tuple(value[0] for value in crosswalk)
    if (
        len(cells) != registry.receipt.total_action_cell_count
        or len(cell_by_hash) != len(cells)
        or structural_receipt.registered_cell_count != len(cells)
        or set(excluded) & set(alternate_hashes)
        or set(excluded) | set(alternate_hashes) != set(cell_by_hash)
        or len(excluded) != structural_receipt.preferred_cell_count_excluded
        or len(alternate_hashes) != structural_receipt.alternate_cell_count
    ):
        _refuse(
            V1BlockSupportDomainBindingRefusalCode.INCOMPLETE_CELL_PARTITION,
            "structural receipt does not partition every registered cell exactly once",
        )
    action_by_coordinate = {
        value[0]: value for value in structural_receipt.physical_action_axis
    }
    if (
        len(action_by_coordinate)
        != structural_receipt.deduplicated_physical_action_count
        or {value[4] for value in crosswalk} != set(action_by_coordinate)
        or any(
            value[1] == preferred_action_binding.preferred_removed_route_index_hash
            and value[2] == preferred_action_binding.preferred_selected_count
            for value in action_by_coordinate.values()
        )
    ):
        _refuse(
            V1BlockSupportDomainBindingRefusalCode.DOMAIN_EQUIVALENCE_MISMATCH,
            "structural alternate cells do not replay the physical-coordinate axis",
        )
    for cell_hash, branch, alias_id, ordinal, coordinate in crosswalk:
        cell = cell_by_hash.get(cell_hash)
        action = action_by_coordinate.get(coordinate)
        if (
            cell is None
            or action is None
            or cell.branch.value != branch
            or cell.resource_alias_id != alias_id
            or cell.action_cell_ordinal != ordinal
            or cell.physical_action_coordinate_hash != coordinate
            or cell.removed_route_index_hash != action[1]
            or structural_receipt.fixed_outside_removed_count
            + len(cell.selected_shell_offsets)
            != action[2]
            or cell.selected_shell_offsets != action[3]
        ):
            _refuse(
                V1BlockSupportDomainBindingRefusalCode.DOMAIN_EQUIVALENCE_MISMATCH,
                "one structural crosswalk entry differs from its live cell/action coordinate",
            )
    for cell_hash in excluded:
        cell = cell_by_hash[cell_hash]
        if (
            cell.removed_route_index_hash
            != preferred_action_binding.preferred_removed_route_index_hash
            or structural_receipt.fixed_outside_removed_count
            + len(cell.selected_shell_offsets)
            != preferred_action_binding.preferred_selected_count
        ):
            _refuse(
                V1BlockSupportDomainBindingRefusalCode.INCOMPLETE_CELL_PARTITION,
                "one excluded cell is not an alias of preferred P",
            )
    preferred_axis_hash = content_hash(
        {"ordered_preferred_cell_hashes": excluded},
        prefix=_PREFIX + "preferred-cell-axis-",
    )
    alternate_axis_hash = content_hash(
        {"ordered_alternate_cell_crosswalk": crosswalk},
        prefix=_PREFIX + "alternate-cell-axis-",
    )
    coordinate_axis_hash = content_hash(
        {"ordered_physical_coordinate_axis": structural_receipt.physical_action_axis},
        prefix=_PREFIX + "physical-coordinate-axis-",
    )
    domain_hash = content_hash(
        {
            "structural_domain_receipt_hash": structural_receipt.receipt_hash,
            "registry_receipt_hash": registry.receipt.receipt_hash,
            "preferred_action_binding_hash": preferred_action_binding.binding_hash,
            "preferred_cell_axis_hash": preferred_axis_hash,
            "alternate_cell_axis_hash": alternate_axis_hash,
            "physical_coordinate_axis_hash": coordinate_axis_hash,
            "hard_alternate_action_count": (
                structural_receipt.deduplicated_physical_action_count
            ),
            "coordinate_equivalence_only": True,
            "canonical_packed_action_hash_authority": False,
        },
        prefix=_PREFIX + "complete-alternate-coordinate-domain-",
    )
    return V1BlockSupportStructuralAlternateDomainEquivalenceReceipt(
        structural_domain_receipt_hash=structural_receipt.receipt_hash,
        registry_receipt_hash=registry.receipt.receipt_hash,
        preferred_action_binding_hash=preferred_action_binding.binding_hash,
        preferred_action_hash=preferred_action_binding.preferred_action_hash,
        preferred_removed_route_index_hash=(
            preferred_action_binding.preferred_removed_route_index_hash
        ),
        preferred_selected_count=preferred_action_binding.preferred_selected_count,
        registered_cell_count=structural_receipt.registered_cell_count,
        preferred_cell_count_excluded=(
            structural_receipt.preferred_cell_count_excluded
        ),
        alternate_cell_count=structural_receipt.alternate_cell_count,
        hard_alternate_action_count=(
            structural_receipt.deduplicated_physical_action_count
        ),
        preferred_exclusion_axis_hash=preferred_axis_hash,
        alternate_cell_axis_hash=alternate_axis_hash,
        physical_coordinate_axis_hash=coordinate_axis_hash,
        complete_alternate_domain_hash=domain_hash,
        _issuer_token=_EQUIVALENCE_TOKEN,
    )


def _issue_preoutcome_receipt(
    *,
    structural_receipt: V1SignedWorkPhysicalAlternateStructuralCUDADomainReceipt,
    registry: V1SignedWorkVariableShellActionCellRegistry,
    preferred_action_binding: V1SignedWorkBlockSupportPreferredActionBinding,
    legacy_alias_bindings: Sequence[V1BlockSupportAliasBinding],
) -> tuple[
    V1BlockSupportPreOutcomeDomainBindingReceipt,
    V1BlockSupportStructuralAlternateDomainEquivalenceReceipt,
    tuple[V1BlockSupportAliasCoverBindingV2, ...],
]:
    handles = _validate_structural_sources(
        structural_receipt=structural_receipt,
        registry=registry,
        preferred_action_binding=preferred_action_binding,
    )
    legacy = tuple(legacy_alias_bindings)
    if not legacy or any(type(value) is not V1BlockSupportAliasBinding for value in legacy):
        raise TypeError("complete legacy alias axis must contain exact aliases")
    expected_keys = tuple(
        (source.branch.value, alias.resource_alias_id)
        for source in registry.branch_sources
        for alias in source.resource_aliases
    )
    observed_keys = tuple((value.branch_id, value.resource_alias_id) for value in legacy)
    if observed_keys != expected_keys or len(set(observed_keys)) != len(observed_keys):
        _refuse(
            V1BlockSupportDomainBindingRefusalCode.ALIAS_AXIS_MISMATCH,
            "legacy aliases must cover the complete live branch/resource axis in order",
        )
    aliases = tuple(
        _issue_alias_binding(registry=registry, legacy_alias=value) for value in legacy
    )
    equivalence = _issue_domain_equivalence(
        structural_receipt=structural_receipt,
        registry=registry,
        preferred_action_binding=preferred_action_binding,
    )
    alias_hashes = tuple(value.binding_hash for value in aliases)
    alias_axis_hash = content_hash(
        {"ordered_alias_cover_binding_hashes": alias_hashes},
        prefix=_PREFIX + "alias-axis-",
    )
    constraint_union = content_hash(
        {
            "domain_equivalence_receipt_hash": equivalence.receipt_hash,
            "alias_axis_hash": alias_axis_hash,
            "legacy_cover_axis": tuple(
                (
                    value.legacy_alias.constraint_cover_hash,
                    value.legacy_alias.within_shell_cover_hash,
                    value.legacy_alias.outside_pair_cover_hash,
                )
                for value in aliases
            ),
            "registered_only_no_complete_outside_pair_authority": True,
        },
        prefix=_PREFIX + "constraint-union-registration-",
    )
    shared_minimum_domain = content_hash(
        {
            "complete_alternate_domain_hash": (
                equivalence.complete_alternate_domain_hash
            ),
            "constraint_union_registration_hash": constraint_union,
            "alias_axis_hash": alias_axis_hash,
            "relaxed_cardinality_equality_used": False,
            "free_gamma_used": False,
            "alias_specific_rounding_anchor_required": True,
        },
        prefix=_PREFIX + "shared-minimum-domain-",
    )
    first = handles[0].receipt
    receipt = V1BlockSupportPreOutcomeDomainBindingReceipt(
        structural_domain_receipt_hash=structural_receipt.receipt_hash,
        domain_equivalence_receipt_hash=equivalence.receipt_hash,
        registry_receipt_hash=registry.receipt.receipt_hash,
        source_registration_hash=structural_receipt.source_registration_hash,
        source_receipt_hashes=tuple(value.receipt.receipt_hash for value in handles),
        source_replay_hashes=tuple(value.receipt.source_replay_hash for value in handles),
        preferred_action_binding_hash=preferred_action_binding.binding_hash,
        preferred_action_hash=preferred_action_binding.preferred_action_hash,
        preferred_resulting_mask_hash=(
            preferred_action_binding.preferred_resulting_mask_hash
        ),
        preferred_removed_route_index_hash=(
            preferred_action_binding.preferred_removed_route_index_hash
        ),
        preferred_selected_count=preferred_action_binding.preferred_selected_count,
        model_identity_hash=registry.receipt.model_identity_hash,
        chart_identity_hash=registry.receipt.chart_identity_hash,
        current_mask_hash=registry.receipt.current_mask_hash,
        proposal_generation=registry.receipt.proposal_generation,
        route_axis_hash=registry.receipt.route_axis_hash,
        route_id_axis_hash=registry.receipt.route_id_axis_hash,
        route_count=first.route_count,
        native_fiber_axis_receipt_hash=(
            registry.receipt.native_fiber_axis_receipt_hash
        ),
        native_fiber_measure_hash=registry.receipt.native_fiber_measure_hash,
        registered_d_prefix_hash=registry.receipt.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=registry.receipt.registered_d_w2_prefix_hash,
        complete_alternate_domain_hash=equivalence.complete_alternate_domain_hash,
        complete_alternate_domain_receipt_hash=equivalence.receipt_hash,
        hard_alternate_action_count=equivalence.hard_alternate_action_count,
        alias_binding_hashes=alias_hashes,
        alias_axis_hash=alias_axis_hash,
        constraint_union_registration_hash=constraint_union,
        shared_action_minimum_domain_hash=shared_minimum_domain,
        distinct_hard_total_cardinality_count=len(
            {value.hard_total_removed_count for value in aliases}
        ),
        production_cuda_source_bound=structural_receipt.production_cuda_execution,
        private_reference_test_only=structural_receipt.private_cpu_reference_execution,
        _issuer_token=_PREOUTCOME_TOKEN,
    )
    return receipt, equivalence, aliases


class V1BlockSupportPreOutcomeDomainBinding:
    """Nonserializable live owner of the acyclic pre-outcome source joint."""

    __slots__ = (
        "receipt",
        "domain_equivalence",
        "alias_bindings",
        "registry",
        "preferred_action_binding",
        "structural_receipt",
        "structural_provider",
        "_live_identity_hash",
        "_evicted",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        receipt: V1BlockSupportPreOutcomeDomainBindingReceipt,
        domain_equivalence: V1BlockSupportStructuralAlternateDomainEquivalenceReceipt,
        alias_bindings: tuple[V1BlockSupportAliasCoverBindingV2, ...],
        registry: V1SignedWorkVariableShellActionCellRegistry,
        preferred_action_binding: V1SignedWorkBlockSupportPreferredActionBinding,
        structural_receipt: V1SignedWorkPhysicalAlternateStructuralCUDADomainReceipt,
        structural_provider: (
            V1SignedWorkPhysicalAlternateStructuralCUDADomainProvider | None
        ),
        _issuer_token: object,
    ) -> None:
        if _issuer_token is not _LIVE_PREOUTCOME_TOKEN:
            raise TypeError("live pre-outcome domain bindings are factory-issued")
        if (
            type(receipt) is not V1BlockSupportPreOutcomeDomainBindingReceipt
            or type(domain_equivalence)
            is not V1BlockSupportStructuralAlternateDomainEquivalenceReceipt
            or type(alias_bindings) is not tuple
            or not alias_bindings
            or any(type(value) is not V1BlockSupportAliasCoverBindingV2 for value in alias_bindings)
            or type(registry) is not V1SignedWorkVariableShellActionCellRegistry
            or type(preferred_action_binding)
            is not V1SignedWorkBlockSupportPreferredActionBinding
            or type(structural_receipt)
            is not V1SignedWorkPhysicalAlternateStructuralCUDADomainReceipt
            or (
                structural_provider is not None
                and type(structural_provider)
                is not V1SignedWorkPhysicalAlternateStructuralCUDADomainProvider
            )
        ):
            raise TypeError("live pre-outcome source types changed")
        if (
            receipt.structural_domain_receipt_hash != structural_receipt.receipt_hash
            or receipt.domain_equivalence_receipt_hash != domain_equivalence.receipt_hash
            or receipt.alias_binding_hashes
            != tuple(value.binding_hash for value in alias_bindings)
            or receipt.registry_receipt_hash != registry.receipt.receipt_hash
            or receipt.preferred_action_binding_hash
            != preferred_action_binding.binding_hash
            or receipt.production_cuda_source_bound is not (structural_provider is not None)
        ):
            _refuse(
                V1BlockSupportDomainBindingRefusalCode.SOURCE_MISMATCH,
                "live sources do not replay the pre-outcome receipt",
            )
        self.receipt = receipt
        self.domain_equivalence = domain_equivalence
        self.alias_bindings = alias_bindings
        self.registry = registry
        self.preferred_action_binding = preferred_action_binding
        self.structural_receipt = structural_receipt
        self.structural_provider = structural_provider
        self._evicted = False
        self._live_identity_hash = self._current_live_identity_hash()
        self.assert_live()

    def _current_live_identity_hash(self) -> str:
        return content_hash(
            {
                "preoutcome_receipt_hash": self.receipt.receipt_hash,
                "domain_equivalence_receipt_hash": self.domain_equivalence.receipt_hash,
                "alias_binding_hashes": tuple(
                    value.binding_hash for value in self.alias_bindings
                ),
                "registry_receipt_hash": self.registry.receipt.receipt_hash,
                "preferred_action_binding_hash": self.preferred_action_binding.binding_hash,
                "structural_domain_receipt_hash": self.structural_receipt.receipt_hash,
                "production_provider_receipt_hash": (
                    None
                    if self.structural_provider is None
                    else self.structural_provider.receipt.receipt_hash
                ),
            },
            prefix=_PREFIX + "live-preoutcome-",
        )

    def assert_live(self) -> "V1BlockSupportPreOutcomeDomainBinding":
        if self._evicted:
            _refuse(
                V1BlockSupportDomainBindingRefusalCode.STALE_SOURCE,
                "pre-outcome domain binding was evicted",
            )
        try:
            self.receipt.assert_integrity()
            self.domain_equivalence.assert_integrity()
            for alias in self.alias_bindings:
                alias.assert_integrity()
            self.structural_receipt.assert_integrity()
            self.registry.assert_live_sources()
            self.preferred_action_binding.assert_live()
            if self.structural_provider is not None:
                self.structural_provider.assert_live()
        except (RuntimeError, PermissionError, TypeError, ValueError) as error:
            _refuse(
                V1BlockSupportDomainBindingRefusalCode.STALE_SOURCE,
                f"one pre-outcome source is stale: {error}",
            )
        if self._current_live_identity_hash() != self._live_identity_hash:
            _refuse(
                V1BlockSupportDomainBindingRefusalCode.STALE_SOURCE,
                "pre-outcome live identity changed",
            )
        return self

    def evict(self) -> None:
        self.assert_live()
        self.registry = None
        self.preferred_action_binding = None
        self.structural_receipt = None
        self.structural_provider = None
        self._evicted = True

    def __reduce__(self) -> NoReturn:
        raise TypeError("live pre-outcome domain bindings are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live pre-outcome domain bindings are nonserializable")


def bind_v1_block_support_preoutcome_domain(
    *,
    structural_provider: V1SignedWorkPhysicalAlternateStructuralCUDADomainProvider,
    legacy_alias_bindings: Sequence[V1BlockSupportAliasBinding],
) -> V1BlockSupportPreOutcomeDomainBinding:
    """Bind the exact production CUDA structural provider before outcomes."""

    if (
        type(structural_provider)
        is not V1SignedWorkPhysicalAlternateStructuralCUDADomainProvider
    ):
        raise TypeError("production pre-outcome binding requires the exact provider")
    structural_provider.assert_live()
    receipt, equivalence, aliases = _issue_preoutcome_receipt(
        structural_receipt=structural_provider.receipt,
        registry=structural_provider.registry,
        preferred_action_binding=structural_provider.preferred_action_binding,
        legacy_alias_bindings=legacy_alias_bindings,
    )
    if receipt.production_cuda_source_bound is not True:
        _refuse(
            V1BlockSupportDomainBindingRefusalCode.SOURCE_MISMATCH,
            "production binding received a reference structural receipt",
        )
    return V1BlockSupportPreOutcomeDomainBinding(
        receipt=receipt,
        domain_equivalence=equivalence,
        alias_bindings=aliases,
        registry=structural_provider.registry,
        preferred_action_binding=structural_provider.preferred_action_binding,
        structural_receipt=structural_provider.receipt,
        structural_provider=structural_provider,
        _issuer_token=_LIVE_PREOUTCOME_TOKEN,
    )


def _bind_v1_block_support_preoutcome_domain_reference_test_only(
    *,
    structural_receipt: V1SignedWorkPhysicalAlternateStructuralCUDADomainReceipt,
    registry: V1SignedWorkVariableShellActionCellRegistry,
    preferred_action_binding: V1SignedWorkBlockSupportPreferredActionBinding,
    legacy_alias_bindings: Sequence[V1BlockSupportAliasBinding],
) -> V1BlockSupportPreOutcomeDomainBinding:
    """Private CPU-reference adapter; it grants no production authority."""

    receipt, equivalence, aliases = _issue_preoutcome_receipt(
        structural_receipt=structural_receipt,
        registry=registry,
        preferred_action_binding=preferred_action_binding,
        legacy_alias_bindings=legacy_alias_bindings,
    )
    if receipt.private_reference_test_only is not True:
        _refuse(
            V1BlockSupportDomainBindingRefusalCode.AUTHORITY_LEAKAGE,
            "reference adapter received a production structural receipt",
        )
    return V1BlockSupportPreOutcomeDomainBinding(
        receipt=receipt,
        domain_equivalence=equivalence,
        alias_bindings=aliases,
        registry=registry,
        preferred_action_binding=preferred_action_binding,
        structural_receipt=structural_receipt,
        structural_provider=None,
        _issuer_token=_LIVE_PREOUTCOME_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1BlockSupportBoundWitnessRegistrationReceipt:
    preoutcome_domain_receipt_hash: str
    domain_equivalence_receipt_hash: str
    legacy_registration_hash: str
    complete_alternate_domain_hash: str
    complete_alternate_domain_receipt_hash: str
    hard_alternate_action_count: int
    preferred_action_hash: str
    preferred_action_freeze_receipt_hash: str
    preferred_selected_count: int
    alias_axis_hash: str
    constraint_union_registration_hash: str
    shared_action_minimum_domain_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    route_axis_hash: str
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    production_cuda_source_bound: bool
    source_derived_registration_path: bool
    receipt_hash: str = field(init=False)
    _issuer_token: InitVar[object] = None
    exact_preoutcome_parent_consumed: bool = True
    raw_legacy_registration_is_not_bound_evidence: bool = True
    legacy_existing_source_compatibility_only: bool = field(init=False)
    structural_within_shell_domain_only: bool = True
    complete_constraint_cover_source_bound: bool = False
    canonical_packed_action_hash_authority: bool = False
    complete_outside_pair_union_authority: bool = False
    full_u_action_domain_authority: bool = False
    fqr_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    dag_rule: str = BLOCK_SUPPORT_PREOUTCOME_DAG_RULE
    authority_scope: str = BLOCK_SUPPORT_DOMAIN_BINDING_AUTHORITY
    schema: str = BLOCK_SUPPORT_BOUND_REGISTRATION_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _BOUND_REGISTRATION_TOKEN:
            raise TypeError("bound witness-registration receipts are issuer-only")
        for name in (
            "preoutcome_domain_receipt_hash",
            "domain_equivalence_receipt_hash",
            "legacy_registration_hash",
            "complete_alternate_domain_hash",
            "complete_alternate_domain_receipt_hash",
            "preferred_action_hash",
            "preferred_action_freeze_receipt_hash",
            "alias_axis_hash",
            "constraint_union_registration_hash",
            "shared_action_minimum_domain_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "dag_rule",
            "authority_scope",
            "schema",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "hard_alternate_action_count",
            _integer(
                self.hard_alternate_action_count,
                name="hard alternate action count",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "preferred_selected_count",
            _integer(
                self.preferred_selected_count,
                name="preferred selected count",
                positive=True,
            ),
        )
        if (
            type(self.production_cuda_source_bound) is not bool
            or type(self.source_derived_registration_path) is not bool
            or self.exact_preoutcome_parent_consumed is not True
            or self.raw_legacy_registration_is_not_bound_evidence is not True
            or self.structural_within_shell_domain_only is not True
            or self.complete_constraint_cover_source_bound is not False
        ):
            _refuse(
                V1BlockSupportDomainBindingRefusalCode.REGISTRATION_MISMATCH,
                "bound registration lost its exact pre-outcome parent",
            )
        object.__setattr__(
            self,
            "legacy_existing_source_compatibility_only",
            not self.source_derived_registration_path,
        )
        _closed(
            (
                self.canonical_packed_action_hash_authority,
                self.complete_outside_pair_union_authority,
                self.full_u_action_domain_authority,
                self.fqr_authority,
                self.C_stream_open_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="bound witness registration",
        )
        if (
            self.complete_alternate_domain_receipt_hash
            != self.domain_equivalence_receipt_hash
            or self.dag_rule != BLOCK_SUPPORT_PREOUTCOME_DAG_RULE
            or self.authority_scope != BLOCK_SUPPORT_DOMAIN_BINDING_AUTHORITY
            or self.schema != BLOCK_SUPPORT_BOUND_REGISTRATION_SCHEMA
        ):
            _refuse(
                V1BlockSupportDomainBindingRefusalCode.REGISTRATION_MISMATCH,
                "bound registration semantics changed",
            )
        object.__setattr__(
            self,
            "receipt_hash",
            content_hash(_payload(self), prefix=_PREFIX + "bound-registration-"),
        )

    def assert_integrity(self) -> str:
        expected = content_hash(_payload(self), prefix=_PREFIX + "bound-registration-")
        if expected != self.receipt_hash:
            _refuse(
                V1BlockSupportDomainBindingRefusalCode.RECEIPT_TAMPERING,
                "bound registration no longer replays",
            )
        return expected


@dataclass(frozen=True, slots=True)
class V1BlockSupportCanonicalBlockContextIdentity:
    """One typed exact sealed context tuple for downstream block execution."""

    bound_registration_receipt_hash: str
    preoutcome_domain_receipt_hash: str
    legacy_registration_hash: str
    sealed_source_block_id: str
    block_index: int
    context_ids: tuple[str, ...]
    canonical_context_tuple_hash: str
    block_support_context_axis_hash: str
    receipt_hash: str = field(init=False)
    _issuer_token: InitVar[object] = None
    exact_sealed_tuple_bound: bool = True
    prefix_local_hash_substitution_allowed: bool = False
    outcome_authority: bool = False
    schema: str = BLOCK_SUPPORT_CANONICAL_CONTEXT_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _CONTEXT_TOKEN:
            raise TypeError("canonical block-context identities are issuer-only")
        for name in (
            "bound_registration_receipt_hash",
            "preoutcome_domain_receipt_hash",
            "legacy_registration_hash",
            "sealed_source_block_id",
            "canonical_context_tuple_hash",
            "block_support_context_axis_hash",
            "schema",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        object.__setattr__(self, "block_index", _integer(self.block_index, name="block index"))
        contexts = tuple(_text(value, name="block context id") for value in self.context_ids)
        if not contexts or len(set(contexts)) != len(contexts):
            _refuse(
                V1BlockSupportDomainBindingRefusalCode.REGISTRATION_MISMATCH,
                "canonical block-context tuple is empty or repeated",
            )
        object.__setattr__(self, "context_ids", contexts)
        expected_tuple_hash = content_hash(
            {
                "schema": "partition_free_v1_canonical_sealed_context_tuple_v1",
                "context_ids": contexts,
            },
            prefix=_PREFIX + "canonical-context-tuple-",
        )
        expected_support_hash = _block_support_context_axis_hash(contexts)
        if (
            self.canonical_context_tuple_hash != expected_tuple_hash
            or self.block_support_context_axis_hash != expected_support_hash
            or self.exact_sealed_tuple_bound is not True
            or self.prefix_local_hash_substitution_allowed is not False
            or self.outcome_authority is not False
            or self.schema != BLOCK_SUPPORT_CANONICAL_CONTEXT_SCHEMA
        ):
            _refuse(
                V1BlockSupportDomainBindingRefusalCode.REGISTRATION_MISMATCH,
                "canonical context identity does not bind the exact sealed tuple",
            )
        object.__setattr__(
            self,
            "receipt_hash",
            content_hash(_payload(self), prefix=_PREFIX + "canonical-context-"),
        )

    def assert_integrity(self) -> str:
        expected = content_hash(_payload(self), prefix=_PREFIX + "canonical-context-")
        if expected != self.receipt_hash:
            _refuse(
                V1BlockSupportDomainBindingRefusalCode.RECEIPT_TAMPERING,
                "canonical block-context identity no longer replays",
            )
        return expected


class V1BlockSupportBoundWitnessSource:
    """Live legacy witness source with a typed pre-outcome parent proof."""

    __slots__ = (
        "receipt",
        "preoutcome_domain",
        "legacy_source",
        "_live_identity_hash",
        "_closed",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        receipt: V1BlockSupportBoundWitnessRegistrationReceipt,
        preoutcome_domain: V1BlockSupportPreOutcomeDomainBinding,
        legacy_source: V1BlockSupportWitnessSource,
        _issuer_token: object,
    ) -> None:
        if _issuer_token is not _LIVE_BOUND_TOKEN:
            raise TypeError("bound witness sources are factory-issued")
        if (
            type(receipt) is not V1BlockSupportBoundWitnessRegistrationReceipt
            or type(preoutcome_domain) is not V1BlockSupportPreOutcomeDomainBinding
            or type(legacy_source) is not V1BlockSupportWitnessSource
        ):
            raise TypeError("bound witness source types changed")
        if (
            receipt.preoutcome_domain_receipt_hash
            != preoutcome_domain.receipt.receipt_hash
            or receipt.legacy_registration_hash
            != legacy_source.registration.registration_hash
        ):
            _refuse(
                V1BlockSupportDomainBindingRefusalCode.REGISTRATION_MISMATCH,
                "live bound source does not replay its receipt parents",
            )
        self.receipt = receipt
        self.preoutcome_domain = preoutcome_domain
        self.legacy_source = legacy_source
        self._closed = False
        self._live_identity_hash = self._current_live_identity_hash()
        self.assert_live()

    @property
    def registration(self) -> V1BlockSupportWitnessRegistration:
        self.assert_live()
        return self.legacy_source.registration

    def _current_live_identity_hash(self) -> str:
        return content_hash(
            {
                "bound_registration_receipt_hash": self.receipt.receipt_hash,
                "preoutcome_domain_receipt_hash": self.preoutcome_domain.receipt.receipt_hash,
                "legacy_registration_hash": self.legacy_source.registration.registration_hash,
            },
            prefix=_PREFIX + "live-bound-source-",
        )

    def assert_live(self) -> "V1BlockSupportBoundWitnessSource":
        if self._closed:
            _refuse(
                V1BlockSupportDomainBindingRefusalCode.STALE_SOURCE,
                "bound witness source was closed",
            )
        try:
            self.receipt.assert_integrity()
            self.preoutcome_domain.assert_live()
            self.legacy_source.registration.assert_integrity()
            self.legacy_source.block_context_ids(0)
        except (RuntimeError, PermissionError, TypeError, ValueError) as error:
            _refuse(
                V1BlockSupportDomainBindingRefusalCode.STALE_SOURCE,
                f"one bound witness source is stale: {error}",
            )
        if self._current_live_identity_hash() != self._live_identity_hash:
            _refuse(
                V1BlockSupportDomainBindingRefusalCode.STALE_SOURCE,
                "bound witness live identity changed",
            )
        return self

    def block_context_ids(self, block_index: int) -> tuple[str, ...]:
        self.assert_live()
        return self.legacy_source.block_context_ids(block_index)

    def canonical_block_context_identity(
        self, block_index: int
    ) -> V1BlockSupportCanonicalBlockContextIdentity:
        self.assert_live()
        index = _integer(block_index, name="block index")
        contexts = self.legacy_source.block_context_ids(index)
        sealed_id = self.legacy_source.block_ids[index]
        tuple_hash = content_hash(
            {
                "schema": "partition_free_v1_canonical_sealed_context_tuple_v1",
                "context_ids": contexts,
            },
            prefix=_PREFIX + "canonical-context-tuple-",
        )
        return V1BlockSupportCanonicalBlockContextIdentity(
            bound_registration_receipt_hash=self.receipt.receipt_hash,
            preoutcome_domain_receipt_hash=self.receipt.preoutcome_domain_receipt_hash,
            legacy_registration_hash=self.receipt.legacy_registration_hash,
            sealed_source_block_id=sealed_id,
            block_index=index,
            context_ids=contexts,
            canonical_context_tuple_hash=tuple_hash,
            block_support_context_axis_hash=(
                _block_support_context_axis_hash(contexts)
            ),
            _issuer_token=_CONTEXT_TOKEN,
        )

    def close(self) -> None:
        self.assert_live()
        self.legacy_source.close()
        self._closed = True

    def __reduce__(self) -> NoReturn:
        raise TypeError("live bound witness sources are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live bound witness sources are nonserializable")


def _issue_bound_registration_receipt(
    *,
    preoutcome_domain: V1BlockSupportPreOutcomeDomainBinding,
    legacy_source: V1BlockSupportWitnessSource,
    source_derived_registration_path: bool,
) -> V1BlockSupportBoundWitnessRegistrationReceipt:
    if type(preoutcome_domain) is not V1BlockSupportPreOutcomeDomainBinding:
        raise TypeError("binding requires the exact live pre-outcome domain")
    if type(legacy_source) is not V1BlockSupportWitnessSource:
        raise TypeError("binding requires the exact legacy witness source")
    if type(source_derived_registration_path) is not bool:
        raise TypeError("source-derived registration marker must be exact")
    preoutcome_domain.assert_live()
    registration = legacy_source.registration
    registration.assert_integrity()
    pre = preoutcome_domain.receipt
    expected_aliases = tuple(
        value.legacy_alias for value in preoutcome_domain.alias_bindings
    )
    first_source = preoutcome_domain.registry.branch_sources[0].assert_live_source().receipt
    if (
        registration.complete_alternate_domain_hash
        != pre.complete_alternate_domain_hash
        or registration.complete_alternate_domain_receipt_hash
        != pre.complete_alternate_domain_receipt_hash
        or registration.alternate_action_count != pre.hard_alternate_action_count
        or registration.preferred_action_hash != pre.preferred_action_hash
        or registration.preferred_action_cardinality != pre.preferred_selected_count
        or registration.alias_bindings != expected_aliases
        or registration.complete_constraint_union_hash
        != pre.constraint_union_registration_hash
        or registration.shared_action_minimum_domain_hash
        != pre.shared_action_minimum_domain_hash
        or registration.corrected_evidence_source_hash
        != first_source.corrected_evidence_stream_source_hash
        or registration.registered_d_prefix_hash != pre.registered_d_prefix_hash
        or registration.registered_d_w2_source_hash
        != pre.registered_d_w2_prefix_hash
        or registration.chart_hash != pre.chart_identity_hash
        or registration.current_mask_hash != pre.current_mask_hash
        or registration.route_axis_hash != pre.route_axis_hash
        or registration.native_fiber_axis_hash
        != pre.native_fiber_axis_receipt_hash
        or registration.generation_index != pre.proposal_generation
    ):
        _refuse(
            V1BlockSupportDomainBindingRefusalCode.REGISTRATION_MISMATCH,
            "legacy registration did not consume the exact pre-outcome domain/source",
        )
    return V1BlockSupportBoundWitnessRegistrationReceipt(
        preoutcome_domain_receipt_hash=pre.receipt_hash,
        domain_equivalence_receipt_hash=pre.domain_equivalence_receipt_hash,
        legacy_registration_hash=registration.registration_hash,
        complete_alternate_domain_hash=registration.complete_alternate_domain_hash,
        complete_alternate_domain_receipt_hash=(
            registration.complete_alternate_domain_receipt_hash
        ),
        hard_alternate_action_count=registration.alternate_action_count,
        preferred_action_hash=registration.preferred_action_hash,
        preferred_action_freeze_receipt_hash=(
            registration.preferred_action_freeze_receipt_hash
        ),
        preferred_selected_count=registration.preferred_action_cardinality,
        alias_axis_hash=pre.alias_axis_hash,
        constraint_union_registration_hash=pre.constraint_union_registration_hash,
        shared_action_minimum_domain_hash=pre.shared_action_minimum_domain_hash,
        chart_identity_hash=pre.chart_identity_hash,
        current_mask_hash=pre.current_mask_hash,
        route_axis_hash=pre.route_axis_hash,
        registered_d_prefix_hash=pre.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=pre.registered_d_w2_prefix_hash,
        production_cuda_source_bound=pre.production_cuda_source_bound,
        source_derived_registration_path=source_derived_registration_path,
        _issuer_token=_BOUND_REGISTRATION_TOKEN,
    )


def _bind_witness_source(
    *,
    preoutcome_domain: V1BlockSupportPreOutcomeDomainBinding,
    legacy_source: V1BlockSupportWitnessSource,
    source_derived_registration_path: bool,
) -> V1BlockSupportBoundWitnessSource:
    receipt = _issue_bound_registration_receipt(
        preoutcome_domain=preoutcome_domain,
        legacy_source=legacy_source,
        source_derived_registration_path=source_derived_registration_path,
    )
    return V1BlockSupportBoundWitnessSource(
        receipt=receipt,
        preoutcome_domain=preoutcome_domain,
        legacy_source=legacy_source,
        _issuer_token=_LIVE_BOUND_TOKEN,
    )


def bind_v1_block_support_existing_witness_source_to_preoutcome_domain(
    *,
    preoutcome_domain: V1BlockSupportPreOutcomeDomainBinding,
    legacy_source: V1BlockSupportWitnessSource,
) -> V1BlockSupportBoundWitnessSource:
    """Compatibility-only join for replayed historical witness sources.

    Exact identity equality is necessary for replay, but it cannot prove that
    the historical registration was *constructed* from the live typed parent.
    The returned receipt therefore cannot pass the production source-derived
    registration gate.
    """

    return _bind_witness_source(
        preoutcome_domain=preoutcome_domain,
        legacy_source=legacy_source,
        source_derived_registration_path=False,
    )


def register_v1_signed_work_bound_block_support_witness(
    *,
    preoutcome_domain: V1BlockSupportPreOutcomeDomainBinding,
    preferred_action_freeze_receipt_hash: str,
    full_axis_support_plane_source_hash: str,
    full_population_context_ids: Sequence[str],
    d_fit_context_ids: Sequence[str],
    d_cert_context_ids: Sequence[str],
    d_fixed_term_receipt: V1BlockSupportDFixedTermReceipt,
    residual_permutation_receipt: SealedUniformPermutationReceipt,
    block_size: int,
    causal_partition_freeze_receipt_hash: str,
    action_freeze_sequence_index: int,
    partition_freeze_sequence_index: int,
    analytic_margin_lower_bound: float,
    analytic_margin_upper_bound: float,
    solver_lower_bound_policy_hash: str,
    numerical_lower_bound_policy_hash: str,
    alpha_ledger_hash: str,
    alpha_allocation_index: int,
    family_alpha_budget: float,
    family_alpha_spent_before: float,
    attempt_alpha_allocation: float,
) -> V1BlockSupportBoundWitnessSource:
    """Register the legacy witness with every domain/source value derived."""

    if type(preoutcome_domain) is not V1BlockSupportPreOutcomeDomainBinding:
        raise TypeError("bound registration requires the exact pre-outcome binding")
    preoutcome_domain.assert_live()
    pre = preoutcome_domain.receipt
    source_receipt = (
        preoutcome_domain.registry.branch_sources[0].assert_live_source().receipt
    )
    legacy = register_v1_signed_work_block_support_witness(
        preferred_action_hash=pre.preferred_action_hash,
        preferred_action_freeze_receipt_hash=preferred_action_freeze_receipt_hash,
        preferred_action_cardinality=pre.preferred_selected_count,
        complete_alternate_domain_hash=pre.complete_alternate_domain_hash,
        complete_alternate_domain_receipt_hash=(
            pre.complete_alternate_domain_receipt_hash
        ),
        alternate_action_count=pre.hard_alternate_action_count,
        complete_constraint_union_hash=pre.constraint_union_registration_hash,
        shared_action_minimum_domain_hash=pre.shared_action_minimum_domain_hash,
        alias_bindings=tuple(
            value.legacy_alias for value in preoutcome_domain.alias_bindings
        ),
        corrected_evidence_source_hash=(
            source_receipt.corrected_evidence_stream_source_hash
        ),
        full_axis_support_plane_source_hash=full_axis_support_plane_source_hash,
        registered_d_prefix_hash=pre.registered_d_prefix_hash,
        registered_d_w2_source_hash=pre.registered_d_w2_prefix_hash,
        chart_hash=pre.chart_identity_hash,
        current_mask_hash=pre.current_mask_hash,
        route_axis_hash=pre.route_axis_hash,
        native_fiber_axis_hash=pre.native_fiber_axis_receipt_hash,
        generation_index=pre.proposal_generation,
        full_population_context_ids=full_population_context_ids,
        d_fit_context_ids=d_fit_context_ids,
        d_cert_context_ids=d_cert_context_ids,
        d_fixed_term_receipt=d_fixed_term_receipt,
        residual_permutation_receipt=residual_permutation_receipt,
        block_size=block_size,
        causal_partition_freeze_receipt_hash=causal_partition_freeze_receipt_hash,
        action_freeze_sequence_index=action_freeze_sequence_index,
        partition_freeze_sequence_index=partition_freeze_sequence_index,
        analytic_margin_lower_bound=analytic_margin_lower_bound,
        analytic_margin_upper_bound=analytic_margin_upper_bound,
        solver_lower_bound_policy_hash=solver_lower_bound_policy_hash,
        numerical_lower_bound_policy_hash=numerical_lower_bound_policy_hash,
        alpha_ledger_hash=alpha_ledger_hash,
        alpha_allocation_index=alpha_allocation_index,
        family_alpha_budget=family_alpha_budget,
        family_alpha_spent_before=family_alpha_spent_before,
        attempt_alpha_allocation=attempt_alpha_allocation,
    )
    return _bind_witness_source(
        preoutcome_domain=preoutcome_domain,
        legacy_source=legacy,
        source_derived_registration_path=True,
    )


def require_v1_block_support_bound_witness_source(
    source: V1BlockSupportBoundWitnessSource,
) -> V1BlockSupportBoundWitnessSource:
    """Exact downstream type gate; raw legacy sources cannot impersonate it."""

    if type(source) is not V1BlockSupportBoundWitnessSource:
        raise TypeError("downstream execution requires a bound witness source")
    return source.assert_live()


def require_v1_block_support_production_bound_witness_source(
    source: V1BlockSupportBoundWitnessSource,
) -> V1BlockSupportBoundWitnessSource:
    """Complete-cover production gate; structural-only sources always refuse."""

    result = require_v1_block_support_bound_witness_source(source)
    if (
        result.receipt.production_cuda_source_bound is not True
        or result.receipt.source_derived_registration_path is not True
        or result.receipt.complete_constraint_cover_source_bound is not True
    ):
        _refuse(
            V1BlockSupportDomainBindingRefusalCode.AUTHORITY_LEAKAGE,
            "production descendants require CUDA custody, source-derived registration, "
            "and a complete within/outside constraint-cover binding",
        )
    return result


def require_v1_block_support_canonical_context_identity(
    *,
    source: V1BlockSupportBoundWitnessSource,
    identity: V1BlockSupportCanonicalBlockContextIdentity,
) -> V1BlockSupportCanonicalBlockContextIdentity:
    """Validate a downstream context by exact sealed tuple, not local hashes."""

    source = require_v1_block_support_bound_witness_source(source)
    if type(identity) is not V1BlockSupportCanonicalBlockContextIdentity:
        raise TypeError("downstream context requires the exact typed identity")
    identity.assert_integrity()
    expected = source.canonical_block_context_identity(identity.block_index)
    if identity != expected or identity.receipt_hash != expected.receipt_hash:
        _refuse(
            V1BlockSupportDomainBindingRefusalCode.REGISTRATION_MISMATCH,
            "context identity is not the exact sealed tuple of this bound source",
        )
    return identity


__all__ = [
    "BLOCK_SUPPORT_ALIAS_COVER_BINDING_V2_SCHEMA",
    "BLOCK_SUPPORT_BOUND_REGISTRATION_SCHEMA",
    "BLOCK_SUPPORT_CANONICAL_CONTEXT_SCHEMA",
    "BLOCK_SUPPORT_CARDINALITY_RULE",
    "BLOCK_SUPPORT_COORDINATE_EQUIVALENCE_RULE",
    "BLOCK_SUPPORT_DOMAIN_BINDING_AUTHORITY",
    "BLOCK_SUPPORT_PREOUTCOME_DAG_RULE",
    "BLOCK_SUPPORT_PREOUTCOME_DOMAIN_BINDING_SCHEMA",
    "BLOCK_SUPPORT_STRUCTURAL_DOMAIN_EQUIVALENCE_SCHEMA",
    "V1BlockSupportAliasCoverBindingV2",
    "V1BlockSupportBoundWitnessRegistrationReceipt",
    "V1BlockSupportBoundWitnessSource",
    "V1BlockSupportCanonicalBlockContextIdentity",
    "V1BlockSupportDomainBindingRefusal",
    "V1BlockSupportDomainBindingRefusalCode",
    "V1BlockSupportPreOutcomeDomainBinding",
    "V1BlockSupportPreOutcomeDomainBindingReceipt",
    "V1BlockSupportStructuralAlternateDomainEquivalenceReceipt",
    "bind_v1_block_support_existing_witness_source_to_preoutcome_domain",
    "bind_v1_block_support_preoutcome_domain",
    "register_v1_signed_work_bound_block_support_witness",
    "require_v1_block_support_bound_witness_source",
    "require_v1_block_support_canonical_context_identity",
    "require_v1_block_support_production_bound_witness_source",
]
