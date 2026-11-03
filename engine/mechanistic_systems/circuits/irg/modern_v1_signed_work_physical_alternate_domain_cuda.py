"""Source-bound physical alternate-action custody for the D-side witness.

The variable-shell action registry owns a complete, bounded cell axis and an
exact removed-set/action-coordinate identity for every cell.  It deliberately
does *not* own packed child masks.  This module closes the next representation
joint without silently upgrading that coordinate custody:

* every registered cell is mapped to exactly one physical removed-set;
* aliases across residual branches are deduplicated by an exact packed action
  in the small reference path;
* the preferred action is removed from the alternate domain by physical
  identity, including every branch/alias occurrence; and
* one block-invariant structural owner retains the source/chart/mask/route/
  D-prefix identity and resident membership across residual blocks; and
* a separate zero-copy per-block binding joins D-fixed/residual support
  handles without feeding their outcome hashes back into structural custody.

Production consumes the exact retained device-shell bundle through its
block-support-only label lease.  The frozen ladder binds that handle's content;
every frozen endpoint must replay to one registered subset identity before
label ``1`` outside the bounded shell can become fixed-prefix membership.  The
two branches must derive bit-identical fixed membership.  No caller-attested
tensor, route-scale host copy, or new route-scale digest is accepted.  The
exact CPU path remains limited to small focused tests and materializes every
action from an explicit fixed-prefix route set.

The CPU exact reference keeps its historical combined API for replay.  The
production CUDA surface is the acyclic structural-provider/per-block-binding
pair.  This is still a within-shell representation joint.  Full-U/outside-pair,
F/Q/R, C, K, O, commit, terminal, and Pareto authority remain closed.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
import hashlib
import math
from numbers import Integral, Real
from typing import Any, Final, NoReturn, Sequence

import numpy as np

from mechanistic_systems.contracts.hard_mask_actions import (
    HardMask,
    RemovedRouteIndexSet,
)

from .contracts import content_hash
from .modern_v1_candidate_composition import V1PackedRemovalAction
from .modern_v1_device_shell_ladder import V1DeviceShellBorrowOperation
from .modern_v1_signed_work_block_support_source_cuda import (
    V1SignedWorkBlockSupportExecutionMode,
    V1SignedWorkBlockSupportPreferredActionBinding,
    V1SignedWorkBlockSupportRefusal,
    V1SignedWorkBlockSupportSourceDeviceHandle,
    V1SignedWorkBlockSupportSourceRole,
)
from .modern_v1_signed_work_variable_shell_action_separation_cuda import (
    V1SignedWorkVariableShellActionCellIdentity,
    V1SignedWorkVariableShellActionCellRegistry,
    V1SignedWorkVariableShellActionSeparationMode,
    V1SignedWorkVariableShellActionSeparationRefusal,
    V1SignedWorkVariableShellQBranch,
)
from .modern_v1_signed_work_device_shell_cuda import (
    V1SignedWorkDeviceShellExecutionBundle,
)

try:  # pragma: no cover - runtime image dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


PHYSICAL_ALTERNATE_DOMAIN_DESIGN_SCHEMA: Final = (
    "partition_free_v1_signed_work_physical_alternate_domain_design_v1"
)
PHYSICAL_ALTERNATE_DOMAIN_HBM_SCHEMA: Final = (
    "partition_free_v1_signed_work_physical_alternate_domain_hbm_v1"
)
PHYSICAL_ALTERNATE_ACTION_SCHEMA: Final = (
    "partition_free_v1_signed_work_physical_alternate_action_v1"
)
PHYSICAL_ALTERNATE_CELL_CROSSWALK_SCHEMA: Final = (
    "partition_free_v1_signed_work_physical_alternate_cell_crosswalk_v1"
)
PHYSICAL_ALTERNATE_DOMAIN_SCHEMA: Final = (
    "partition_free_v1_signed_work_physical_alternate_domain_v1"
)
PHYSICAL_ALTERNATE_UPSTREAM_REQUIREMENT_SCHEMA: Final = (
    "partition_free_v1_signed_work_physical_alternate_upstream_owner_requirement_v1"
)
PHYSICAL_ALTERNATE_STRUCTURAL_CUDA_DOMAIN_SCHEMA: Final = (
    "partition_free_v1_signed_work_physical_alternate_structural_cuda_domain_v2"
)
PHYSICAL_ALTERNATE_BLOCK_SUPPORT_BINDING_SCHEMA: Final = (
    "partition_free_v1_signed_work_physical_alternate_block_support_binding_v1"
)
PHYSICAL_ALTERNATE_DOMAIN_RULE: Final = (
    "complete_registered_cell_partition_exact_removed_set_replay_cross_branch_"
    "packed_action_deduplication_and_preferred_physical_action_exclusion_v1"
)
PHYSICAL_ALTERNATE_DOMAIN_AUTHORITY: Final = (
    "D_side_physical_alternate_representation_only_no_full_U_outside_pair_FQR_"
    "C_K_O_commit_terminal_or_pareto_authority_v1"
)
PHYSICAL_ALTERNATE_MEMBERSHIP_RULE: Final = (
    "one_source_owned_uint8_prefix_suffix_shell_label_per_route_with_prefix_"
    "membership_equal_label_one_and_no_route_scale_D2H_or_digest_v1"
)
PHYSICAL_ALTERNATE_FIXED_DERIVATION_RULE: Final = (
    "frozen_label_commitment_label_one_first_endpoint_exact_subset_replay_"
    "minus_registered_shell_equals_empty_subset_fixed_removed_set_v1"
)

_PREFIX: Final = "partition-free-v1-signed-work-physical-alternate-domain-"
_FORECAST_TOKEN: Final = object()
_ACTION_TOKEN: Final = object()
_CROSSWALK_TOKEN: Final = object()
_RECEIPT_TOKEN: Final = object()
_PROVIDER_TOKEN: Final = object()
_STRUCTURAL_CUDA_RECEIPT_TOKEN: Final = object()
_STRUCTURAL_CUDA_PROVIDER_TOKEN: Final = object()
_CUDA_FIXED_MEMBERSHIP_LEASE_TOKEN: Final = object()
_BLOCK_SUPPORT_BINDING_RECEIPT_TOKEN: Final = object()
_BLOCK_SUPPORT_BINDING_TOKEN: Final = object()


class V1SignedWorkPhysicalAlternateDomainExecutionMode(str, Enum):
    CUDA_PRODUCTION = "cuda_production"
    CPU_EXACT_REFERENCE_TEST_ONLY = "cpu_exact_reference_test_only"


class V1SignedWorkPhysicalAlternateDomainRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    INVALID_DESIGN = "invalid_design"
    SOURCE_MISMATCH = "source_mismatch"
    STALE_SOURCE = "stale_source"
    INCOMPLETE_CELL_CROSSWALK = "incomplete_cell_crosswalk"
    PACKED_MATERIALIZATION_MISMATCH = "packed_materialization_mismatch"
    PREFERRED_ACTION_NOT_EXCLUDED = "preferred_action_not_excluded"
    CUDA_REQUIRED = "cuda_required"
    UPSTREAM_MEMBERSHIP_OWNER_REQUIRED = "upstream_membership_owner_required"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    ROUTE_SCALE_TRANSFER_FORBIDDEN = "route_scale_transfer_forbidden"
    EVICTED = "evicted"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SignedWorkPhysicalAlternateDomainRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SignedWorkPhysicalAlternateDomainRefusalCode,
        message: str,
        *,
        upstream_requirement: (
            "V1SignedWorkPhysicalAlternateMembershipOwnerRequirement | None"
        ) = None,
    ) -> None:
        if type(code) is not V1SignedWorkPhysicalAlternateDomainRefusalCode:
            raise TypeError("physical-alternate refusal code must be exact")
        self.code = code
        self.upstream_requirement = upstream_requirement
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SignedWorkPhysicalAlternateDomainRefusalCode,
    message: str,
    *,
    upstream_requirement: (
        "V1SignedWorkPhysicalAlternateMembershipOwnerRequirement | None"
    ) = None,
) -> NoReturn:
    raise V1SignedWorkPhysicalAlternateDomainRefusal(
        code,
        message,
        upstream_requirement=upstream_requirement,
    )


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _integer(value: object, *, name: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.INVALID_TYPE,
            f"{name} must be integral",
        )
    result = int(value)
    if result < (1 if positive else 0):
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.INVALID_DESIGN,
            f"{name} is out of range",
        )
    return result


def _fraction(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.INVALID_TYPE,
            f"{name} must be one real scalar",
        )
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 0.85:
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.INVALID_DESIGN,
            f"{name} must lie in (0, .85]",
        )
    return result


def _closed(values: Sequence[object], *, owner: str) -> None:
    if any(type(value) is not bool or value is not False for value in values):
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.AUTHORITY_VIOLATION,
            f"{owner} crossed its representation-only authority boundary",
        )


def _payload(value: object) -> dict[str, object]:
    return {
        name: (
            getattr(value, name).value
            if isinstance(getattr(value, name), Enum)
            else getattr(value, name)
        )
        for name in value.__dataclass_fields__
        if not name.startswith("_") and name != "receipt_hash"
    }


@dataclass(frozen=True, slots=True)
class V1SignedWorkPhysicalAlternateDomainDesign:
    execution_mode: V1SignedWorkPhysicalAlternateDomainExecutionMode = (
        V1SignedWorkPhysicalAlternateDomainExecutionMode.CUDA_PRODUCTION
    )
    route_tile_size: int = 1 << 20
    maximum_reference_route_count: int = 4096
    maximum_registered_cell_count: int = 512
    hbm_ceiling_fraction: float = 0.85
    deterministic_cuda_required: bool = True
    resident_membership_source_ownership_required: bool = True
    caller_attested_membership_forbidden: bool = True
    route_scale_d2h_forbidden: bool = True
    route_scale_content_digest_forbidden: bool = True
    packed_action_streaming_owner_required: bool = True
    schema: str = PHYSICAL_ALTERNATE_DOMAIN_DESIGN_SCHEMA

    def __post_init__(self) -> None:
        if type(self.execution_mode) is not (
            V1SignedWorkPhysicalAlternateDomainExecutionMode
        ):
            raise TypeError("physical-alternate execution mode must be exact")
        object.__setattr__(
            self,
            "route_tile_size",
            _integer(self.route_tile_size, name="route tile size", positive=True),
        )
        object.__setattr__(
            self,
            "maximum_reference_route_count",
            _integer(
                self.maximum_reference_route_count,
                name="maximum reference route count",
                positive=True,
            ),
        )
        cells = _integer(
            self.maximum_registered_cell_count,
            name="maximum registered cell count",
            positive=True,
        )
        if cells > 512:
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.INVALID_DESIGN,
                "the B=8 two-branch cell domain cannot exceed 512 entries",
            )
        object.__setattr__(self, "maximum_registered_cell_count", cells)
        object.__setattr__(
            self,
            "hbm_ceiling_fraction",
            _fraction(self.hbm_ceiling_fraction, name="HBM ceiling fraction"),
        )
        required = (
            self.deterministic_cuda_required,
            self.resident_membership_source_ownership_required,
            self.caller_attested_membership_forbidden,
            self.route_scale_d2h_forbidden,
            self.route_scale_content_digest_forbidden,
            self.packed_action_streaming_owner_required,
        )
        if any(type(value) is not bool or value is not True for value in required):
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.INVALID_DESIGN,
                "membership ownership, deterministic CUDA, and transfer bans are mandatory",
            )
        if self.schema != PHYSICAL_ALTERNATE_DOMAIN_DESIGN_SCHEMA:
            raise ValueError("unsupported physical-alternate design schema")

    @property
    def design_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "design-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkPhysicalAlternateDomainHBMForecast:
    design_hash: str
    route_count: int
    shell_size: int
    registered_cell_count: int
    physical_action_count_upper_bound: int
    route_tile_size: int
    borrowed_resident_membership_label_bytes: int
    live_source_bytes_in_current_reservation: int
    live_source_unaccounted_bytes: int
    owned_fixed_membership_bytes: int
    owned_cross_branch_fixed_comparison_bytes: int
    owned_parent_packed_mask_bytes: int
    owned_packed_mask_workspace_bytes: int
    owned_route_tile_workspace_bytes: int
    owned_bounded_crosswalk_bytes: int
    route_scale_h2d_payload_bytes: int
    maximum_h2d_staging_bytes: int
    incremental_device_bytes: int
    current_reserved_bytes: int
    forecast_peak_reserved_bytes: int
    device_total_memory_bytes: int
    hbm_ceiling_bytes: int
    admitted: bool
    source_colive_accounted_in_current_reservation: bool = True
    source_colive_accounted_in_forecast: bool = True
    parent_packed_h2d_explicit: bool = True
    route_scale_d2h_bytes: int = 0
    route_scale_content_digest_bytes: int = 0
    full_action_by_route_matrix_bytes: int = 0
    schema: str = PHYSICAL_ALTERNATE_DOMAIN_HBM_SCHEMA
    _issuer_token: InitVar[object] = None

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _FORECAST_TOKEN:
            raise TypeError("physical-alternate HBM forecasts are factory-issued")
        object.__setattr__(
            self, "design_hash", _text(self.design_hash, name="design hash")
        )
        for name in (
            "route_count",
            "shell_size",
            "registered_cell_count",
            "physical_action_count_upper_bound",
            "route_tile_size",
            "borrowed_resident_membership_label_bytes",
            "live_source_bytes_in_current_reservation",
            "live_source_unaccounted_bytes",
            "owned_fixed_membership_bytes",
            "owned_cross_branch_fixed_comparison_bytes",
            "owned_parent_packed_mask_bytes",
            "owned_packed_mask_workspace_bytes",
            "owned_route_tile_workspace_bytes",
            "owned_bounded_crosswalk_bytes",
            "route_scale_h2d_payload_bytes",
            "maximum_h2d_staging_bytes",
            "incremental_device_bytes",
            "current_reserved_bytes",
            "forecast_peak_reserved_bytes",
            "device_total_memory_bytes",
            "hbm_ceiling_bytes",
            "route_scale_d2h_bytes",
            "route_scale_content_digest_bytes",
            "full_action_by_route_matrix_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    name=name.replace("_", " "),
                    positive=name
                    in {
                        "route_count",
                        "shell_size",
                        "registered_cell_count",
                        "physical_action_count_upper_bound",
                        "route_tile_size",
                        "device_total_memory_bytes",
                    },
                ),
            )
        expected_incremental = (
            self.live_source_unaccounted_bytes
            + self.owned_fixed_membership_bytes
            + self.owned_cross_branch_fixed_comparison_bytes
            + self.owned_parent_packed_mask_bytes
            + self.owned_packed_mask_workspace_bytes
            + self.owned_route_tile_workspace_bytes
            + self.owned_bounded_crosswalk_bytes
        )
        if (
            self.borrowed_resident_membership_label_bytes != 2 * self.route_count
            or self.live_source_unaccounted_bytes
            != max(
                0,
                self.live_source_bytes_in_current_reservation
                - self.current_reserved_bytes,
            )
            or self.owned_fixed_membership_bytes != self.route_count
            or self.owned_cross_branch_fixed_comparison_bytes != self.route_count
            or self.owned_parent_packed_mask_bytes != (self.route_count + 7) // 8
            or self.owned_packed_mask_workspace_bytes != (self.route_count + 7) // 8
            or self.route_scale_h2d_payload_bytes
            != self.owned_packed_mask_workspace_bytes
            or self.maximum_h2d_staging_bytes != self.route_scale_h2d_payload_bytes
            or self.incremental_device_bytes != expected_incremental
            or self.forecast_peak_reserved_bytes
            != self.current_reserved_bytes + self.incremental_device_bytes
            or self.admitted
            is not (self.forecast_peak_reserved_bytes <= self.hbm_ceiling_bytes)
            or self.route_scale_d2h_bytes != 0
            or self.route_scale_content_digest_bytes != 0
            or self.full_action_by_route_matrix_bytes != 0
            or self.source_colive_accounted_in_current_reservation
            is not (self.live_source_unaccounted_bytes == 0)
            or self.source_colive_accounted_in_forecast is not True
            or self.parent_packed_h2d_explicit is not True
            or self.schema != PHYSICAL_ALTERNATE_DOMAIN_HBM_SCHEMA
        ):
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.ROUTE_SCALE_TRANSFER_FORBIDDEN,
                "physical-alternate HBM or transfer accounting changed",
            )
        if type(self.admitted) is not bool:
            raise TypeError("HBM admission must be exact bool")

    @property
    def forecast_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_PREFIX + "hbm-",
        )


def estimate_v1_signed_work_physical_alternate_domain_hbm(
    *,
    design: V1SignedWorkPhysicalAlternateDomainDesign,
    route_count: int,
    shell_size: int,
    registered_cell_count: int,
    physical_action_count_upper_bound: int,
    live_source_device_bytes: int,
    device_total_memory_bytes: int,
    current_reserved_bytes: int,
) -> V1SignedWorkPhysicalAlternateDomainHBMForecast:
    """Forecast the bounded provider workspace before any membership access."""

    if type(design) is not V1SignedWorkPhysicalAlternateDomainDesign:
        raise TypeError("HBM forecast requires the exact domain design")
    routes = _integer(route_count, name="route count", positive=True)
    shell = _integer(shell_size, name="shell size", positive=True)
    cells = _integer(registered_cell_count, name="registered cell count", positive=True)
    actions = _integer(
        physical_action_count_upper_bound,
        name="physical action count upper bound",
        positive=True,
    )
    live_source = _integer(
        live_source_device_bytes,
        name="live source device bytes",
    )
    if shell > 8 or cells > design.maximum_registered_cell_count or actions > cells:
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.INVALID_DESIGN,
            "forecast axes exceed the bounded B=8 action domain",
        )
    total = _integer(
        device_total_memory_bytes,
        name="device total memory bytes",
        positive=True,
    )
    reserved = _integer(current_reserved_bytes, name="current reserved bytes")
    if reserved > total:
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.INVALID_DESIGN,
            "current reserved bytes exceed total device memory",
        )
    tile = min(routes, design.route_tile_size)
    # One uint8 fixed-membership vector, one comparison vector while branch
    # equality is checked, one resident packed parent, one packed child
    # workspace, one bounded tile of uint8 labels plus int64 physical
    # positions, and a small device-side cell/action index crosswalk.  Any
    # co-live source footprint not covered by the registered reservation is
    # conservatively added rather than assumed away.
    fixed_bytes = routes
    fixed_comparison_bytes = routes
    parent_packed_bytes = (routes + 7) // 8
    packed_bytes = (routes + 7) // 8
    tile_bytes = tile * (1 + 8)
    crosswalk_bytes = cells * (8 * 6)
    live_source_unaccounted = max(0, live_source - reserved)
    incremental = (
        live_source_unaccounted
        + fixed_bytes
        + fixed_comparison_bytes
        + parent_packed_bytes
        + packed_bytes
        + tile_bytes
        + crosswalk_bytes
    )
    ceiling = int(math.floor(total * design.hbm_ceiling_fraction))
    peak = reserved + incremental
    return V1SignedWorkPhysicalAlternateDomainHBMForecast(
        design_hash=design.design_hash,
        route_count=routes,
        shell_size=shell,
        registered_cell_count=cells,
        physical_action_count_upper_bound=actions,
        route_tile_size=tile,
        borrowed_resident_membership_label_bytes=2 * routes,
        live_source_bytes_in_current_reservation=live_source,
        live_source_unaccounted_bytes=live_source_unaccounted,
        owned_fixed_membership_bytes=fixed_bytes,
        owned_cross_branch_fixed_comparison_bytes=fixed_comparison_bytes,
        owned_parent_packed_mask_bytes=parent_packed_bytes,
        owned_packed_mask_workspace_bytes=packed_bytes,
        owned_route_tile_workspace_bytes=tile_bytes,
        owned_bounded_crosswalk_bytes=crosswalk_bytes,
        route_scale_h2d_payload_bytes=packed_bytes,
        maximum_h2d_staging_bytes=packed_bytes,
        incremental_device_bytes=incremental,
        current_reserved_bytes=reserved,
        forecast_peak_reserved_bytes=peak,
        device_total_memory_bytes=total,
        hbm_ceiling_bytes=ceiling,
        admitted=peak <= ceiling,
        source_colive_accounted_in_current_reservation=(live_source_unaccounted == 0),
        _issuer_token=_FORECAST_TOKEN,
    )


def _retained_structural_source_device_bytes(
    *,
    registry: V1SignedWorkVariableShellActionCellRegistry,
    route_count: int,
) -> int:
    """Exact co-live source storage retained by this representation joint."""

    # Both variable-shell sources expose their exact retained compact tensor
    # footprint.  The two frozen branches each retain one uint8 label vector
    # borrowed by this provider.  Per-block D-fixed/residual coefficients are
    # intentionally absent: their consumer accounts them after structural
    # custody has been built once for all residual blocks.
    compact_branch_sources = sum(
        source.source_handle.retained_tensor_bytes for source in registry.branch_sources
    )
    resident_branch_labels = 2 * route_count
    return compact_branch_sources + resident_branch_labels


# Historical small-reference callers keep the old private spelling.  Its
# semantics are now explicitly structural rather than per-block.
_retained_source_device_bytes = _retained_structural_source_device_bytes


@dataclass(frozen=True, slots=True)
class V1SignedWorkPhysicalAlternateMembershipOwnerRequirement:
    registry_receipt_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    shell_freeze_receipt_hash: str
    shell_indices: tuple[int, ...]
    fixed_prefix_removed_route_index_hash: str
    fixed_prefix_count: int
    required_live_owner_type: str = (
        "upstream_decision_level_retained_membership_cuda_owner_v1"
    )
    membership_rule: str = PHYSICAL_ALTERNATE_MEMBERSHIP_RULE
    direct_same_tensor_ownership_transfer_required: bool = True
    caller_attestation_is_not_evidence: bool = True
    route_scale_d2h_allowed: bool = False
    route_scale_content_digest_allowed: bool = False
    schema: str = PHYSICAL_ALTERNATE_UPSTREAM_REQUIREMENT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "registry_receipt_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "shell_freeze_receipt_hash",
            "fixed_prefix_removed_route_index_hash",
            "required_live_owner_type",
            "membership_rule",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "proposal_generation",
            _integer(self.proposal_generation, name="proposal generation"),
        )
        object.__setattr__(
            self,
            "route_count",
            _integer(self.route_count, name="route count", positive=True),
        )
        object.__setattr__(
            self,
            "fixed_prefix_count",
            _integer(self.fixed_prefix_count, name="fixed prefix count"),
        )
        shell = tuple(
            _integer(value, name="shell route index") for value in self.shell_indices
        )
        if not shell or len(shell) > 8 or len(set(shell)) != len(shell):
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.INVALID_DESIGN,
                "upstream membership requirement has an invalid frozen shell",
            )
        object.__setattr__(self, "shell_indices", shell)
        if (
            self.direct_same_tensor_ownership_transfer_required is not True
            or self.caller_attestation_is_not_evidence is not True
            or self.route_scale_d2h_allowed is not False
            or self.route_scale_content_digest_allowed is not False
            or self.schema != PHYSICAL_ALTERNATE_UPSTREAM_REQUIREMENT_SCHEMA
        ):
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.AUTHORITY_VIOLATION,
                "upstream membership requirement weakened its custody boundary",
            )

    @property
    def requirement_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "upstream-requirement-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkPhysicalAlternateActionReceipt:
    physical_action_coordinate_hash: str
    packed_action_hash: str
    resulting_mask_hash: str
    resulting_mask_packed_sha256: str
    removed_route_index_hash: str
    removed_count: int
    selected_shell_offsets: tuple[int, ...]
    branch_alias_incidence_count: int
    cell_incidence_count: int
    schema: str = PHYSICAL_ALTERNATE_ACTION_SCHEMA
    _issuer_token: InitVar[object] = None

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _ACTION_TOKEN:
            raise TypeError("physical-alternate action receipts are issuer-only")
        for name in (
            "physical_action_coordinate_hash",
            "packed_action_hash",
            "resulting_mask_hash",
            "resulting_mask_packed_sha256",
            "removed_route_index_hash",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "removed_count",
            _integer(self.removed_count, name="removed count", positive=True),
        )
        offsets = tuple(
            _integer(value, name="selected shell offset")
            for value in self.selected_shell_offsets
        )
        if offsets != tuple(sorted(set(offsets))):
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.PACKED_MATERIALIZATION_MISMATCH,
                "physical action shell subset is not canonical",
            )
        object.__setattr__(self, "selected_shell_offsets", offsets)
        for name in ("branch_alias_incidence_count", "cell_incidence_count"):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name), name=name.replace("_", " "), positive=True
                ),
            )
        if len(self.resulting_mask_packed_sha256) != 64:
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.PACKED_MATERIALIZATION_MISMATCH,
                "packed mask digest is malformed",
            )
        if self.schema != PHYSICAL_ALTERNATE_ACTION_SCHEMA:
            raise ValueError("unsupported physical-alternate action schema")

    @property
    def action_receipt_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_PREFIX + "action-",
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkPhysicalAlternateCellCrosswalkReceipt:
    action_cell_hash: str
    branch: V1SignedWorkVariableShellQBranch
    resource_alias_id: str
    action_cell_ordinal: int
    selected_shell_offsets: tuple[int, ...]
    removed_route_index_hash: str
    physical_action_coordinate_hash: str
    packed_action_hash: str
    action_receipt_hash: str
    schema: str = PHYSICAL_ALTERNATE_CELL_CROSSWALK_SCHEMA
    _issuer_token: InitVar[object] = None

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _CROSSWALK_TOKEN:
            raise TypeError("physical-alternate crosswalk receipts are issuer-only")
        if type(self.branch) is not V1SignedWorkVariableShellQBranch:
            raise TypeError("physical-alternate crosswalk branch must be exact")
        for name in (
            "action_cell_hash",
            "resource_alias_id",
            "removed_route_index_hash",
            "physical_action_coordinate_hash",
            "packed_action_hash",
            "action_receipt_hash",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "action_cell_ordinal",
            _integer(self.action_cell_ordinal, name="action cell ordinal"),
        )
        offsets = tuple(
            _integer(value, name="selected shell offset")
            for value in self.selected_shell_offsets
        )
        if offsets != tuple(sorted(set(offsets))):
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.INCOMPLETE_CELL_CROSSWALK,
                "crosswalk shell subset is not canonical",
            )
        object.__setattr__(self, "selected_shell_offsets", offsets)
        if self.schema != PHYSICAL_ALTERNATE_CELL_CROSSWALK_SCHEMA:
            raise ValueError("unsupported physical-alternate crosswalk schema")

    @property
    def crosswalk_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_PREFIX + "cell-crosswalk-",
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkPhysicalAlternateDomainReceipt:
    design_hash: str
    hbm_forecast_hash: str
    registry_receipt_hash: str
    d_fixed_support_receipt_hash: str
    residual_support_receipt_hash: str
    preferred_action_binding_hash: str
    preferred_action_hash: str
    preferred_removed_route_index_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    shell_freeze_receipt_hash: str
    shell_indices: tuple[int, ...]
    fixed_outside_removed_route_index_hash: str
    fixed_outside_removed_count: int
    reference_membership_owner_hash: str
    registered_cell_count: int
    preferred_cell_count_excluded: int
    alternate_cell_count: int
    deduplicated_physical_action_count: int
    cross_branch_or_alias_duplicate_count: int
    excluded_preferred_action_cell_hashes: tuple[str, ...]
    cell_crosswalks: tuple[V1SignedWorkPhysicalAlternateCellCrosswalkReceipt, ...]
    physical_actions: tuple[V1SignedWorkPhysicalAlternateActionReceipt, ...]
    source_role_axis: tuple[V1SignedWorkBlockSupportSourceRole, ...]
    execution_mode: V1SignedWorkPhysicalAlternateDomainExecutionMode
    receipt_hash: str = field(init=False)
    complete_registered_cell_partition: bool = True
    exact_cell_to_physical_action_crosswalk: bool = True
    cross_branch_aliases_deduplicated: bool = True
    preferred_physical_action_excluded: bool = True
    resident_fixed_outside_membership_bound: bool = True
    exact_packed_action_materialization_owner_bound: bool = True
    route_scale_d2h_values: int = 0
    route_scale_device_tensor_content_hashes: int = 0
    outside_pair_execution_joined: bool = False
    full_u_action_domain_authority: bool = False
    fqr_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    rule: str = PHYSICAL_ALTERNATE_DOMAIN_RULE
    authority_scope: str = PHYSICAL_ALTERNATE_DOMAIN_AUTHORITY
    schema: str = PHYSICAL_ALTERNATE_DOMAIN_SCHEMA
    _issuer_token: InitVar[object] = None
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _RECEIPT_TOKEN:
            raise TypeError("physical-alternate domain receipts are issuer-only")
        for name in (
            "design_hash",
            "hbm_forecast_hash",
            "registry_receipt_hash",
            "d_fixed_support_receipt_hash",
            "residual_support_receipt_hash",
            "preferred_action_binding_hash",
            "preferred_action_hash",
            "preferred_removed_route_index_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "shell_freeze_receipt_hash",
            "fixed_outside_removed_route_index_hash",
            "reference_membership_owner_hash",
            "rule",
            "authority_scope",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        for name in (
            "proposal_generation",
            "route_count",
            "fixed_outside_removed_count",
            "registered_cell_count",
            "preferred_cell_count_excluded",
            "alternate_cell_count",
            "deduplicated_physical_action_count",
            "cross_branch_or_alias_duplicate_count",
            "route_scale_d2h_values",
            "route_scale_device_tensor_content_hashes",
        ):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    name=name.replace("_", " "),
                    positive=name
                    in {
                        "route_count",
                        "registered_cell_count",
                        "alternate_cell_count",
                        "deduplicated_physical_action_count",
                    },
                ),
            )
        shell = tuple(
            _integer(value, name="shell route index") for value in self.shell_indices
        )
        if not shell or len(shell) > 8 or len(set(shell)) != len(shell):
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.INCOMPLETE_CELL_CROSSWALK,
                "domain shell axis is malformed",
            )
        object.__setattr__(self, "shell_indices", shell)
        excluded = tuple(
            _text(value, name="excluded preferred cell hash")
            for value in self.excluded_preferred_action_cell_hashes
        )
        crosswalks = tuple(self.cell_crosswalks)
        actions = tuple(self.physical_actions)
        roles = tuple(self.source_role_axis)
        if (
            len(excluded) != self.preferred_cell_count_excluded
            or len(set(excluded)) != len(excluded)
            or len(crosswalks) != self.alternate_cell_count
            or any(
                type(value) is not V1SignedWorkPhysicalAlternateCellCrosswalkReceipt
                for value in crosswalks
            )
            or len({value.action_cell_hash for value in crosswalks}) != len(crosswalks)
            or len(actions) != self.deduplicated_physical_action_count
            or any(
                type(value) is not V1SignedWorkPhysicalAlternateActionReceipt
                for value in actions
            )
            or len({value.packed_action_hash for value in actions}) != len(actions)
            or roles
            != (
                V1SignedWorkBlockSupportSourceRole.D_FIXED,
                V1SignedWorkBlockSupportSourceRole.RESIDUAL_BLOCK,
            )
            or self.registered_cell_count
            != self.preferred_cell_count_excluded + self.alternate_cell_count
            or self.cross_branch_or_alias_duplicate_count
            != self.alternate_cell_count - self.deduplicated_physical_action_count
            or {value.action_receipt_hash for value in crosswalks}
            != {value.action_receipt_hash for value in actions}
        ):
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.INCOMPLETE_CELL_CROSSWALK,
                "registered cells, excluded P cells, and deduplicated actions do not partition",
            )
        object.__setattr__(self, "excluded_preferred_action_cell_hashes", excluded)
        object.__setattr__(self, "cell_crosswalks", crosswalks)
        object.__setattr__(self, "physical_actions", actions)
        object.__setattr__(self, "source_role_axis", roles)
        if (
            type(self.execution_mode)
            is not V1SignedWorkPhysicalAlternateDomainExecutionMode
        ):
            raise TypeError("domain execution mode must be exact")
        required = (
            self.complete_registered_cell_partition,
            self.exact_cell_to_physical_action_crosswalk,
            self.cross_branch_aliases_deduplicated,
            self.preferred_physical_action_excluded,
            self.resident_fixed_outside_membership_bound,
            self.exact_packed_action_materialization_owner_bound,
        )
        if any(type(value) is not bool or value is not True for value in required):
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.AUTHORITY_VIOLATION,
                "domain lost a mandatory identity or materialization invariant",
            )
        _closed(
            (
                self.outside_pair_execution_joined,
                self.full_u_action_domain_authority,
                self.fqr_authority,
                self.C_stream_open_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="physical alternate domain",
        )
        if (
            self.route_scale_d2h_values != 0
            or self.route_scale_device_tensor_content_hashes != 0
            or self.rule != PHYSICAL_ALTERNATE_DOMAIN_RULE
            or self.authority_scope != PHYSICAL_ALTERNATE_DOMAIN_AUTHORITY
            or self.schema != PHYSICAL_ALTERNATE_DOMAIN_SCHEMA
        ):
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.AUTHORITY_VIOLATION,
                "domain transfer or authority semantics changed",
            )
        seal = content_hash(_payload(self), prefix=_PREFIX + "receipt-integrity-")
        if self._integrity_seal and self._integrity_seal != seal:
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.STALE_SOURCE,
                "physical-alternate receipt integrity changed",
            )
        object.__setattr__(self, "_integrity_seal", seal)
        object.__setattr__(
            self,
            "receipt_hash",
            content_hash(_payload(self), prefix=_PREFIX + "receipt-"),
        )

    def assert_integrity(self) -> str:
        observed = content_hash(_payload(self), prefix=_PREFIX + "receipt-integrity-")
        if observed != self._integrity_seal:
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.STALE_SOURCE,
                "physical-alternate receipt integrity changed",
            )
        expected_hash = content_hash(_payload(self), prefix=_PREFIX + "receipt-")
        if expected_hash != self.receipt_hash:
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.STALE_SOURCE,
                "physical-alternate receipt content address changed",
            )
        return self.receipt_hash


def _issue_v1_signed_work_physical_alternate_structural_receipt_from_reference_for_test(
    *, provider: V1SignedWorkPhysicalAlternateDomainProvider
) -> V1SignedWorkPhysicalAlternateStructuralCUDADomainReceipt:
    """Issue block-free structural identity for small CPU hash-DAG tests.

    This private adapter does not create a structural device provider and
    therefore cannot issue membership leases or enter production execution.
    It exists only so the acyclic receipt graph can be exercised without a
    CUDA runtime.
    """

    if type(provider) is not V1SignedWorkPhysicalAlternateDomainProvider:
        raise TypeError("reference structural adapter requires the exact provider")
    provider.assert_live()
    if (
        provider.receipt.execution_mode
        is not V1SignedWorkPhysicalAlternateDomainExecutionMode.CPU_EXACT_REFERENCE_TEST_ONLY
    ):
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.AUTHORITY_VIOLATION,
            "reference structural adapter is test-only",
        )
    receipt = provider.receipt
    sources = tuple(provider.registry.branch_sources)
    empty_subsets: list[str] = []
    preferred_subsets: list[str] = []
    custody_families: list[str] = []
    branch_executions: list[str] = []
    ladders: list[str] = []
    commitments: list[str] = []
    for source in sources:
        source_receipt = source.assert_live_source().receipt
        empty = tuple(
            item for item in source.subset_identities if item.shell_offsets == ()
        )
        preferred = tuple(
            item
            for item in source.subset_identities
            if item.removed_route_index_hash
            == receipt.preferred_removed_route_index_hash
        )
        if len(empty) != 1 or len(preferred) != 1:
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.SOURCE_MISMATCH,
                "reference branch lacks unique empty/preferred subset custody",
            )
        empty_subsets.append(empty[0].subset_hash)
        preferred_subsets.append(preferred[0].subset_hash)
        custody_families.append(empty[0].streamed_hash_custody_family_hash)
        branch_executions.append(
            content_hash(
                {
                    "source_receipt_hash": source_receipt.receipt_hash,
                    "execution": "private_cpu_reference_test_only",
                },
                prefix=_PREFIX + "reference-structural-branch-execution-",
            )
        )
        ladders.append(source.registration.shell_freeze_receipt_hash)
        commitments.append(
            content_hash(
                {
                    "reference_membership_owner_hash": (
                        receipt.reference_membership_owner_hash
                    ),
                    "source_receipt_hash": source_receipt.receipt_hash,
                },
                prefix=_PREFIX + "reference-label-commitment-",
            )
        )
    action_axis = tuple(
        sorted(
            (
                action.physical_action_coordinate_hash,
                action.removed_route_index_hash,
                action.removed_count,
                action.selected_shell_offsets,
            )
            for action in receipt.physical_actions
        )
    )
    crosswalk_axis = tuple(
        (
            value.action_cell_hash,
            value.branch.value,
            value.resource_alias_id,
            value.action_cell_ordinal,
            value.physical_action_coordinate_hash,
        )
        for value in receipt.cell_crosswalks
    )
    fixed_binding_hash = content_hash(
        {
            "reference_membership_owner_hash": receipt.reference_membership_owner_hash,
            "registry_receipt_hash": receipt.registry_receipt_hash,
            "branch_empty_subset_identity_hashes": tuple(empty_subsets),
            "branch_preferred_subset_identity_hashes": tuple(preferred_subsets),
        },
        prefix=_PREFIX + "reference-structural-fixed-binding-",
    )
    return V1SignedWorkPhysicalAlternateStructuralCUDADomainReceipt(
        design_hash=receipt.design_hash,
        hbm_forecast_hash=receipt.hbm_forecast_hash,
        registry_receipt_hash=receipt.registry_receipt_hash,
        source_registration_hash=(
            provider.preferred_action_binding.source_registration_hash
        ),
        preferred_action_binding_hash=receipt.preferred_action_binding_hash,
        preferred_action_hash=receipt.preferred_action_hash,
        preferred_resulting_mask_hash=(
            provider.preferred_action_binding.preferred_resulting_mask_hash
        ),
        preferred_removed_route_index_hash=receipt.preferred_removed_route_index_hash,
        preferred_selected_count=(
            provider.preferred_action_binding.preferred_selected_count
        ),
        model_identity_hash=receipt.model_identity_hash,
        chart_identity_hash=receipt.chart_identity_hash,
        current_mask_hash=receipt.current_mask_hash,
        proposal_generation=receipt.proposal_generation,
        route_axis_hash=receipt.route_axis_hash,
        route_id_axis_hash=receipt.route_id_axis_hash,
        native_fiber_axis_receipt_hash=(
            provider.registry.receipt.native_fiber_axis_receipt_hash
        ),
        native_fiber_measure_hash=provider.registry.receipt.native_fiber_measure_hash,
        route_count=receipt.route_count,
        registered_d_prefix_hash=receipt.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=receipt.registered_d_w2_prefix_hash,
        shell_freeze_receipt_hash=receipt.shell_freeze_receipt_hash,
        shell_indices=receipt.shell_indices,
        fixed_outside_removed_route_index_hash=(
            receipt.fixed_outside_removed_route_index_hash
        ),
        fixed_outside_removed_count=receipt.fixed_outside_removed_count,
        branch_execution_hashes=tuple(branch_executions),
        branch_frozen_ladder_hashes=tuple(ladders),
        branch_label_commitments=tuple(commitments),
        branch_empty_subset_identity_hashes=tuple(empty_subsets),
        branch_first_endpoint_subset_identity_hashes=tuple(preferred_subsets),
        branch_streamed_hash_custody_family_hashes=tuple(custody_families),
        fixed_membership_binding_hash=fixed_binding_hash,
        registered_cell_count=receipt.registered_cell_count,
        preferred_cell_count_excluded=receipt.preferred_cell_count_excluded,
        alternate_cell_count=receipt.alternate_cell_count,
        deduplicated_physical_action_count=receipt.deduplicated_physical_action_count,
        excluded_preferred_action_cell_hashes=(
            receipt.excluded_preferred_action_cell_hashes
        ),
        physical_action_axis=action_axis,
        cell_crosswalk_axis=crosswalk_axis,
        route_scale_h2d_payload_bytes=0,
        maximum_h2d_staging_bytes=0,
        resident_device_identity="cpu",
        production_cuda_execution=False,
        private_cpu_reference_execution=True,
        _issuer_token=_STRUCTURAL_CUDA_RECEIPT_TOKEN,
    )


class V1SignedWorkPhysicalAlternateDomainProvider:
    """Live small-domain owner for exact physical alternate actions."""

    __slots__ = (
        "receipt",
        "forecast",
        "registry",
        "d_fixed_support",
        "residual_support",
        "preferred_action_binding",
        "_parent_mask",
        "_membership_labels",
        "_actions_by_hash",
        "_crosswalk_by_cell_hash",
        "_live_identity_hash",
        "_evicted",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        receipt: V1SignedWorkPhysicalAlternateDomainReceipt,
        forecast: V1SignedWorkPhysicalAlternateDomainHBMForecast,
        registry: V1SignedWorkVariableShellActionCellRegistry,
        d_fixed_support: V1SignedWorkBlockSupportSourceDeviceHandle,
        residual_support: V1SignedWorkBlockSupportSourceDeviceHandle,
        preferred_action_binding: V1SignedWorkBlockSupportPreferredActionBinding,
        parent_mask: HardMask,
        membership_labels: np.ndarray,
        actions_by_hash: dict[str, V1PackedRemovalAction],
        crosswalk_by_cell_hash: dict[
            str, V1SignedWorkPhysicalAlternateCellCrosswalkReceipt
        ],
        _issuer_token: object,
    ) -> None:
        if _issuer_token is not _PROVIDER_TOKEN:
            raise TypeError("physical-alternate providers are factory-issued")
        if type(receipt) is not V1SignedWorkPhysicalAlternateDomainReceipt:
            raise TypeError("provider requires its exact durable receipt")
        if type(forecast) is not V1SignedWorkPhysicalAlternateDomainHBMForecast:
            raise TypeError("provider requires its exact HBM forecast")
        labels = np.asarray(membership_labels)
        if (
            labels.dtype != np.dtype(np.uint8)
            or labels.ndim != 1
            or int(labels.size) != receipt.route_count
            or labels.flags.writeable
        ):
            raise TypeError("reference membership labels must be immutable uint8")
        self.receipt = receipt
        self.forecast = forecast
        self.registry = registry
        self.d_fixed_support = d_fixed_support
        self.residual_support = residual_support
        self.preferred_action_binding = preferred_action_binding
        self._parent_mask = parent_mask
        self._membership_labels = labels
        self._actions_by_hash = dict(actions_by_hash)
        self._crosswalk_by_cell_hash = dict(crosswalk_by_cell_hash)
        self._evicted = False
        self._live_identity_hash = self._current_live_identity_hash()
        self.assert_live()

    def _current_live_identity_hash(self) -> str:
        return content_hash(
            {
                "receipt_hash": self.receipt.receipt_hash,
                "forecast_hash": self.forecast.forecast_hash,
                "registry_receipt_hash": self.registry.receipt.receipt_hash,
                "d_fixed_support_receipt_hash": self.d_fixed_support.receipt.receipt_hash,
                "residual_support_receipt_hash": self.residual_support.receipt.receipt_hash,
                "preferred_action_binding_hash": self.preferred_action_binding.binding_hash,
                "parent_mask_hash": self._parent_mask.mask_hash,
                "membership_owner_hash": self.receipt.reference_membership_owner_hash,
                "action_axis": tuple(sorted(self._actions_by_hash)),
                "crosswalk_axis": tuple(sorted(self._crosswalk_by_cell_hash)),
            },
            prefix=_PREFIX + "live-provider-",
        )

    def assert_live(self) -> "V1SignedWorkPhysicalAlternateDomainProvider":
        if self._evicted:
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.EVICTED,
                "physical-alternate provider was evicted",
            )
        try:
            self.receipt.assert_integrity()
            self.registry.assert_live_sources()
            self.d_fixed_support.assert_live()
            self.residual_support.assert_live()
            self.preferred_action_binding.assert_live()
        except (
            V1SignedWorkBlockSupportRefusal,
            V1SignedWorkVariableShellActionSeparationRefusal,
            PermissionError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.STALE_SOURCE,
                f"one bound source is no longer live: {error}",
            )
        if (
            self._parent_mask.mask_hash != self.receipt.current_mask_hash
            or set(self._actions_by_hash)
            != {value.packed_action_hash for value in self.receipt.physical_actions}
            or set(self._crosswalk_by_cell_hash)
            != {value.action_cell_hash for value in self.receipt.cell_crosswalks}
            or self._current_live_identity_hash() != self._live_identity_hash
        ):
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.STALE_SOURCE,
                "live physical-alternate ownership changed",
            )
        return self

    def crosswalk_for_cell(
        self, action_cell_hash: str
    ) -> V1SignedWorkPhysicalAlternateCellCrosswalkReceipt:
        self.assert_live()
        key = _text(action_cell_hash, name="action cell hash")
        try:
            return self._crosswalk_by_cell_hash[key]
        except KeyError:
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.PREFERRED_ACTION_NOT_EXCLUDED,
                "cell is absent because it is P or was never registered",
            )

    def materialize_reference_action_for_cell(
        self, action_cell_hash: str
    ) -> V1PackedRemovalAction:
        crosswalk = self.crosswalk_for_cell(action_cell_hash)
        return self._actions_by_hash[crosswalk.packed_action_hash]

    def require_canonical_full_u_certifier_authority(self) -> NoReturn:
        self.assert_live()
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.AUTHORITY_VIOLATION,
            "outside-pair execution has not joined; this provider cannot enter the canonical full-U certifier",
        )

    def evict(self) -> None:
        self.assert_live()
        self._membership_labels = None
        self._actions_by_hash = {}
        self._crosswalk_by_cell_hash = {}
        self._parent_mask = None
        self._evicted = True

    def __reduce__(self) -> NoReturn:
        raise TypeError("live physical-alternate providers are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live physical-alternate providers are nonserializable")


def _validate_source_family(
    *,
    registry: V1SignedWorkVariableShellActionCellRegistry,
    d_fixed_support: V1SignedWorkBlockSupportSourceDeviceHandle,
    residual_support: V1SignedWorkBlockSupportSourceDeviceHandle,
    preferred_action_binding: V1SignedWorkBlockSupportPreferredActionBinding,
    parent_mask: HardMask,
) -> tuple[object, object, object]:
    if type(registry) is not V1SignedWorkVariableShellActionCellRegistry:
        raise TypeError("physical alternate domain requires the exact live registry")
    if (
        type(d_fixed_support) is not V1SignedWorkBlockSupportSourceDeviceHandle
        or type(residual_support) is not V1SignedWorkBlockSupportSourceDeviceHandle
        or type(preferred_action_binding)
        is not V1SignedWorkBlockSupportPreferredActionBinding
        or type(parent_mask) is not HardMask
    ):
        raise TypeError(
            "physical alternate domain requires exact source/action/mask types"
        )
    try:
        registry.assert_live_sources()
        d_fixed_support.assert_live()
        residual_support.assert_live()
        preferred_action_binding.assert_live()
    except (
        V1SignedWorkBlockSupportRefusal,
        V1SignedWorkVariableShellActionSeparationRefusal,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.STALE_SOURCE,
            f"source family is not live: {error}",
        )
    registry_receipt = registry.receipt
    d_receipt = d_fixed_support.receipt
    residual_receipt = residual_support.receipt
    if (
        d_receipt.source_role is not V1SignedWorkBlockSupportSourceRole.D_FIXED
        or residual_receipt.source_role
        is not V1SignedWorkBlockSupportSourceRole.RESIDUAL_BLOCK
    ):
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.SOURCE_MISMATCH,
            "support handles must be ordered D-fixed then residual block",
        )
    common_names = (
        "model_identity_hash",
        "chart_identity_hash",
        "current_mask_hash",
        "proposal_generation",
        "route_axis_hash",
        "route_id_axis_hash",
        "native_fiber_axis_receipt_hash",
        "native_fiber_measure_hash",
        "registered_d_prefix_hash",
        "registered_d_w2_prefix_hash",
    )
    if any(
        getattr(d_receipt, name) != getattr(registry_receipt, name)
        or getattr(residual_receipt, name) != getattr(registry_receipt, name)
        for name in common_names
    ):
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.SOURCE_MISMATCH,
            "block-support and variable-shell sources do not share one chart/prefix",
        )
    if (
        d_receipt.preferred_action_binding_hash != preferred_action_binding.binding_hash
        or residual_receipt.preferred_action_binding_hash
        != preferred_action_binding.binding_hash
        or d_receipt.preferred_action_hash
        != preferred_action_binding.preferred_action_hash
        or residual_receipt.preferred_action_hash
        != preferred_action_binding.preferred_action_hash
        or preferred_action_binding.current_mask_hash
        != registry_receipt.current_mask_hash
        or preferred_action_binding.route_axis_hash != registry_receipt.route_axis_hash
        or preferred_action_binding.route_count != d_receipt.route_count
        or parent_mask.mask_hash != registry_receipt.current_mask_hash
        or parent_mask.route_axis_hash != registry_receipt.route_axis_hash
        or parent_mask.route_count != d_receipt.route_count
        or d_receipt.route_count != residual_receipt.route_count
    ):
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.SOURCE_MISMATCH,
            "preferred packed action or parent route identity differs across sources",
        )
    return registry_receipt, d_receipt, residual_receipt


def _validate_structural_source_family(
    *,
    registry: V1SignedWorkVariableShellActionCellRegistry,
    preferred_action_binding: V1SignedWorkBlockSupportPreferredActionBinding,
    parent_mask: HardMask,
) -> object:
    """Validate only the block-invariant physical-domain source family."""

    if type(registry) is not V1SignedWorkVariableShellActionCellRegistry:
        raise TypeError("structural physical domain requires the exact live registry")
    if (
        type(preferred_action_binding)
        is not V1SignedWorkBlockSupportPreferredActionBinding
        or type(parent_mask) is not HardMask
    ):
        raise TypeError(
            "structural physical domain requires exact preferred-action and mask types"
        )
    try:
        registry.assert_live_sources()
        preferred_action_binding.assert_live()
    except (
        V1SignedWorkBlockSupportRefusal,
        V1SignedWorkVariableShellActionSeparationRefusal,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.STALE_SOURCE,
            f"structural source family is not live: {error}",
        )
    receipt = registry.receipt
    if (
        preferred_action_binding.current_mask_hash != receipt.current_mask_hash
        or preferred_action_binding.chart_identity_hash != receipt.chart_identity_hash
        or preferred_action_binding.proposal_generation != receipt.proposal_generation
        or preferred_action_binding.route_axis_hash != receipt.route_axis_hash
        or preferred_action_binding.route_count != parent_mask.route_count
        or parent_mask.mask_hash != receipt.current_mask_hash
        or parent_mask.route_axis_hash != receipt.route_axis_hash
    ):
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.SOURCE_MISMATCH,
            "preferred packed action or parent identity differs from the registry",
        )
    return receipt


def _fixed_prefix_identity(
    registry: V1SignedWorkVariableShellActionCellRegistry,
) -> tuple[str, int, tuple[int, ...]]:
    identities: list[tuple[str, int, tuple[int, ...]]] = []
    for source in registry.branch_sources:
        empty = tuple(
            value for value in source.subset_identities if value.shell_offsets == ()
        )
        if len(empty) != 1:
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.INCOMPLETE_CELL_CROSSWALK,
                "one branch lacks its unique fixed-prefix subset identity",
            )
        item = empty[0]
        identities.append(
            (
                item.removed_route_index_hash,
                item.removed_count,
                tuple(source.registration.shell_indices),
            )
        )
        if any(
            alias.fixed_outside_removed_count != item.removed_count
            for alias in source.resource_aliases
        ):
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.SOURCE_MISMATCH,
                "a Phase-A alias fixed-outside count differs from hash custody",
            )
    if len(set(identities)) != 1:
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.SOURCE_MISMATCH,
            "branches do not share one fixed prefix and shell",
        )
    return identities[0]


def _membership_owner_hash(
    *,
    registry_receipt_hash: str,
    parent_mask_hash: str,
    fixed_removed: RemovedRouteIndexSet,
    shell_indices: tuple[int, ...],
    labels: np.ndarray,
) -> str:
    return content_hash(
        {
            "execution": "CPU_exact_reference_test_only",
            "registry_receipt_hash": registry_receipt_hash,
            "parent_mask_hash": parent_mask_hash,
            "fixed_removed_route_index_hash": fixed_removed.index_hash,
            "fixed_removed_count": fixed_removed.count,
            "shell_indices": shell_indices,
            # Small reference only: this is deliberately unavailable in CUDA
            # production, where route-scale content digests are forbidden.
            "membership_labels": labels.tobytes(order="C"),
        },
        prefix=_PREFIX + "reference-membership-owner-",
    )


def build_v1_signed_work_physical_alternate_domain_reference(
    *,
    registry: V1SignedWorkVariableShellActionCellRegistry,
    d_fixed_support: V1SignedWorkBlockSupportSourceDeviceHandle,
    residual_support: V1SignedWorkBlockSupportSourceDeviceHandle,
    preferred_action_binding: V1SignedWorkBlockSupportPreferredActionBinding,
    parent_mask: HardMask,
    fixed_outside_route_indices: Sequence[int] | np.ndarray,
    design: V1SignedWorkPhysicalAlternateDomainDesign,
    device_total_memory_bytes: int,
    current_reserved_bytes: int,
) -> V1SignedWorkPhysicalAlternateDomainProvider:
    """Materialize and replay the exact physical domain for small CPU tests."""

    if type(design) is not V1SignedWorkPhysicalAlternateDomainDesign:
        raise TypeError("reference physical domain requires the exact design")
    if design.execution_mode is not (
        V1SignedWorkPhysicalAlternateDomainExecutionMode.CPU_EXACT_REFERENCE_TEST_ONLY
    ):
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.INVALID_DESIGN,
            "CPU materialization requires the explicit exact-reference mode",
        )
    registry_receipt, d_receipt, residual_receipt = _validate_source_family(
        registry=registry,
        d_fixed_support=d_fixed_support,
        residual_support=residual_support,
        preferred_action_binding=preferred_action_binding,
        parent_mask=parent_mask,
    )
    if (
        registry.design.execution_mode
        is not V1SignedWorkVariableShellActionSeparationMode.EXACT_REFERENCE_TEST_ONLY
        or d_receipt.execution_mode
        is not V1SignedWorkBlockSupportExecutionMode.CPU_REFERENCE_TEST_ONLY
        or residual_receipt.execution_mode
        is not V1SignedWorkBlockSupportExecutionMode.CPU_REFERENCE_TEST_ONLY
    ):
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.INVALID_DESIGN,
            "reference materialization cannot consume production-mode sources",
        )
    if registry_receipt.total_action_cell_count > design.maximum_registered_cell_count:
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.INVALID_DESIGN,
            "registered cell axis exceeds the reference design",
        )
    if parent_mask.route_count > design.maximum_reference_route_count:
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.INVALID_DESIGN,
            "CPU reference materialization exceeded its small-domain limit",
        )
    fixed_hash, fixed_count, shell_indices = _fixed_prefix_identity(registry)
    fixed_removed = RemovedRouteIndexSet.from_indices(
        fixed_outside_route_indices,
        route_count=parent_mask.route_count,
    )
    fixed_axis = tuple(int(value) for value in fixed_removed)
    if (
        fixed_removed.index_hash != fixed_hash
        or fixed_removed.count != fixed_count
        or set(fixed_axis).intersection(shell_indices)
    ):
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.PACKED_MATERIALIZATION_MISMATCH,
            "explicit fixed-outside membership does not replay registered hash custody",
        )
    parent_live = parent_mask.to_bool_array()
    if any(not bool(parent_live[index]) for index in fixed_axis + shell_indices):
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.PACKED_MATERIALIZATION_MISMATCH,
            "fixed or shell membership contains an already-pruned route",
        )
    labels = np.zeros((parent_mask.route_count,), dtype=np.uint8)
    if fixed_axis:
        labels[np.asarray(fixed_axis, dtype=np.int64)] = 1
    for offset, route_index in enumerate(shell_indices):
        labels[route_index] = 3 + offset
    immutable_labels = np.frombuffer(labels.tobytes(order="C"), dtype=np.uint8)
    immutable_labels.setflags(write=False)

    cells = tuple(registry.iter_action_cells())
    if len(cells) != registry_receipt.total_action_cell_count or any(
        type(value) is not V1SignedWorkVariableShellActionCellIdentity
        for value in cells
    ):
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.INCOMPLETE_CELL_CROSSWALK,
            "live registry cell iteration does not replay its receipt count",
        )
    cell_materialization: list[
        tuple[V1SignedWorkVariableShellActionCellIdentity, V1PackedRemovalAction]
    ] = []
    for cell in cells:
        removed_indices = tuple(
            sorted(
                fixed_axis
                + tuple(shell_indices[offset] for offset in cell.selected_shell_offsets)
            )
        )
        if not removed_indices:
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.PACKED_MATERIALIZATION_MISMATCH,
                "the registered action domain contains an empty pruning action",
            )
        action = V1PackedRemovalAction.from_removed_indices(
            parent_mask=parent_mask,
            removed_indices=removed_indices,
        )
        if (
            action.removed_route_index_hash != cell.removed_route_index_hash
            or action.removed_count != fixed_count + len(cell.selected_shell_offsets)
        ):
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.PACKED_MATERIALIZATION_MISMATCH,
                "one cell coordinate does not replay to its exact packed physical action",
            )
        cell_materialization.append((cell, action))

    preferred_hash = preferred_action_binding.preferred_action_hash
    excluded = tuple(
        cell.action_cell_hash
        for cell, action in cell_materialization
        if action.action_hash == preferred_hash
    )
    alternate_pairs = tuple(
        (cell, action)
        for cell, action in cell_materialization
        if action.action_hash != preferred_hash
    )
    if not alternate_pairs:
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.PREFERRED_ACTION_NOT_EXCLUDED,
            "P exclusion left no physical alternate",
        )
    if any(action.action_hash == preferred_hash for _cell, action in alternate_pairs):
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.PREFERRED_ACTION_NOT_EXCLUDED,
            "preferred packed action remains in the alternate domain",
        )

    incidence: dict[str, list[V1SignedWorkVariableShellActionCellIdentity]] = {}
    actions_by_hash: dict[str, V1PackedRemovalAction] = {}
    coordinate_by_action: dict[str, str] = {}
    offsets_by_action: dict[str, tuple[int, ...]] = {}
    for cell, action in alternate_pairs:
        actions_by_hash.setdefault(action.action_hash, action)
        incidence.setdefault(action.action_hash, []).append(cell)
        previous_coordinate = coordinate_by_action.setdefault(
            action.action_hash, cell.physical_action_coordinate_hash
        )
        previous_offsets = offsets_by_action.setdefault(
            action.action_hash, cell.selected_shell_offsets
        )
        if (
            previous_coordinate != cell.physical_action_coordinate_hash
            or previous_offsets != cell.selected_shell_offsets
        ):
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.INCOMPLETE_CELL_CROSSWALK,
                "aliases disagree about one physical packed action coordinate",
            )

    action_receipts: list[V1SignedWorkPhysicalAlternateActionReceipt] = []
    action_receipt_by_hash: dict[str, V1SignedWorkPhysicalAlternateActionReceipt] = {}
    for action_hash in sorted(actions_by_hash):
        action = actions_by_hash[action_hash]
        cells_for_action = incidence[action_hash]
        branch_aliases = {
            (cell.branch.value, cell.resource_alias_id) for cell in cells_for_action
        }
        receipt = V1SignedWorkPhysicalAlternateActionReceipt(
            physical_action_coordinate_hash=coordinate_by_action[action_hash],
            packed_action_hash=action_hash,
            resulting_mask_hash=action.resulting_mask.mask_hash,
            resulting_mask_packed_sha256=hashlib.sha256(
                action.resulting_mask.packed_bits
            ).hexdigest(),
            removed_route_index_hash=action.removed_route_index_hash,
            removed_count=action.removed_count,
            selected_shell_offsets=offsets_by_action[action_hash],
            branch_alias_incidence_count=len(branch_aliases),
            cell_incidence_count=len(cells_for_action),
            _issuer_token=_ACTION_TOKEN,
        )
        action_receipts.append(receipt)
        action_receipt_by_hash[action_hash] = receipt

    crosswalks: list[V1SignedWorkPhysicalAlternateCellCrosswalkReceipt] = []
    for cell, action in alternate_pairs:
        action_receipt = action_receipt_by_hash[action.action_hash]
        crosswalks.append(
            V1SignedWorkPhysicalAlternateCellCrosswalkReceipt(
                action_cell_hash=cell.action_cell_hash,
                branch=cell.branch,
                resource_alias_id=cell.resource_alias_id,
                action_cell_ordinal=cell.action_cell_ordinal,
                selected_shell_offsets=cell.selected_shell_offsets,
                removed_route_index_hash=cell.removed_route_index_hash,
                physical_action_coordinate_hash=cell.physical_action_coordinate_hash,
                packed_action_hash=action.action_hash,
                action_receipt_hash=action_receipt.action_receipt_hash,
                _issuer_token=_CROSSWALK_TOKEN,
            )
        )
    forecast = estimate_v1_signed_work_physical_alternate_domain_hbm(
        design=design,
        route_count=parent_mask.route_count,
        shell_size=len(shell_indices),
        registered_cell_count=len(cells),
        physical_action_count_upper_bound=len(cells),
        live_source_device_bytes=_retained_source_device_bytes(
            registry=registry,
            route_count=parent_mask.route_count,
        ),
        device_total_memory_bytes=device_total_memory_bytes,
        current_reserved_bytes=current_reserved_bytes,
    )
    if not forecast.admitted:
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.HBM_FORECAST_EXCEEDED,
            "physical-alternate reference forecast exceeds its registered ceiling",
        )
    membership_hash = _membership_owner_hash(
        registry_receipt_hash=registry_receipt.receipt_hash,
        parent_mask_hash=parent_mask.mask_hash,
        fixed_removed=fixed_removed,
        shell_indices=shell_indices,
        labels=immutable_labels,
    )
    receipt = V1SignedWorkPhysicalAlternateDomainReceipt(
        design_hash=design.design_hash,
        hbm_forecast_hash=forecast.forecast_hash,
        registry_receipt_hash=registry_receipt.receipt_hash,
        d_fixed_support_receipt_hash=d_receipt.receipt_hash,
        residual_support_receipt_hash=residual_receipt.receipt_hash,
        preferred_action_binding_hash=preferred_action_binding.binding_hash,
        preferred_action_hash=preferred_hash,
        preferred_removed_route_index_hash=(
            preferred_action_binding.preferred_removed_route_index_hash
        ),
        model_identity_hash=registry_receipt.model_identity_hash,
        chart_identity_hash=registry_receipt.chart_identity_hash,
        current_mask_hash=registry_receipt.current_mask_hash,
        proposal_generation=registry_receipt.proposal_generation,
        route_axis_hash=registry_receipt.route_axis_hash,
        route_id_axis_hash=registry_receipt.route_id_axis_hash,
        route_count=parent_mask.route_count,
        registered_d_prefix_hash=registry_receipt.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=registry_receipt.registered_d_w2_prefix_hash,
        shell_freeze_receipt_hash=registry_receipt.shell_freeze_receipt_hash,
        shell_indices=shell_indices,
        fixed_outside_removed_route_index_hash=fixed_removed.index_hash,
        fixed_outside_removed_count=fixed_removed.count,
        reference_membership_owner_hash=membership_hash,
        registered_cell_count=len(cells),
        preferred_cell_count_excluded=len(excluded),
        alternate_cell_count=len(crosswalks),
        deduplicated_physical_action_count=len(action_receipts),
        cross_branch_or_alias_duplicate_count=(len(crosswalks) - len(action_receipts)),
        excluded_preferred_action_cell_hashes=tuple(sorted(excluded)),
        cell_crosswalks=tuple(crosswalks),
        physical_actions=tuple(action_receipts),
        source_role_axis=(d_receipt.source_role, residual_receipt.source_role),
        execution_mode=design.execution_mode,
        _issuer_token=_RECEIPT_TOKEN,
    )
    crosswalk_by_cell = {value.action_cell_hash: value for value in crosswalks}
    return V1SignedWorkPhysicalAlternateDomainProvider(
        receipt=receipt,
        forecast=forecast,
        registry=registry,
        d_fixed_support=d_fixed_support,
        residual_support=residual_support,
        preferred_action_binding=preferred_action_binding,
        parent_mask=HardMask(
            route_axis_hash=parent_mask.route_axis_hash,
            route_count=parent_mask.route_count,
            packed_bits=bytes(parent_mask.packed_bits),
        ),
        membership_labels=immutable_labels,
        actions_by_hash=actions_by_hash,
        crosswalk_by_cell_hash=crosswalk_by_cell,
        _issuer_token=_PROVIDER_TOKEN,
    )


def describe_v1_signed_work_physical_alternate_membership_owner_requirement(
    *, registry: V1SignedWorkVariableShellActionCellRegistry
) -> V1SignedWorkPhysicalAlternateMembershipOwnerRequirement:
    """Describe the exact source-derived CUDA membership ownership contract."""

    if type(registry) is not V1SignedWorkVariableShellActionCellRegistry:
        raise TypeError("membership requirement requires the exact live registry")
    try:
        registry.assert_live_sources()
    except (V1SignedWorkVariableShellActionSeparationRefusal, RuntimeError) as error:
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.STALE_SOURCE,
            f"registry sources are stale: {error}",
        )
    fixed_hash, fixed_count, shell_indices = _fixed_prefix_identity(registry)
    receipt = registry.receipt
    route_count = registry.branch_sources[0].assert_live_source().receipt.route_count
    return V1SignedWorkPhysicalAlternateMembershipOwnerRequirement(
        registry_receipt_hash=receipt.receipt_hash,
        chart_identity_hash=receipt.chart_identity_hash,
        current_mask_hash=receipt.current_mask_hash,
        proposal_generation=receipt.proposal_generation,
        route_axis_hash=receipt.route_axis_hash,
        route_id_axis_hash=receipt.route_id_axis_hash,
        route_count=route_count,
        registered_d_prefix_hash=receipt.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=receipt.registered_d_w2_prefix_hash,
        shell_freeze_receipt_hash=receipt.shell_freeze_receipt_hash,
        shell_indices=shell_indices,
        fixed_prefix_removed_route_index_hash=fixed_hash,
        fixed_prefix_count=fixed_count,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkPhysicalAlternateStructuralCUDADomainReceipt:
    """Block-invariant CUDA custody for the physical alternate domain.

    This receipt is deliberately upstream of every D-fixed/residual support
    block.  It owns only structural identities and therefore remains reusable
    while the registered D_W2 block changes.
    """

    design_hash: str
    hbm_forecast_hash: str
    registry_receipt_hash: str
    source_registration_hash: str
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
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    route_count: int
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    shell_freeze_receipt_hash: str
    shell_indices: tuple[int, ...]
    fixed_outside_removed_route_index_hash: str
    fixed_outside_removed_count: int
    branch_execution_hashes: tuple[str, str]
    branch_frozen_ladder_hashes: tuple[str, str]
    branch_label_commitments: tuple[str, str]
    branch_empty_subset_identity_hashes: tuple[str, str]
    branch_first_endpoint_subset_identity_hashes: tuple[str, str]
    branch_streamed_hash_custody_family_hashes: tuple[str, str]
    fixed_membership_binding_hash: str
    registered_cell_count: int
    preferred_cell_count_excluded: int
    alternate_cell_count: int
    deduplicated_physical_action_count: int
    excluded_preferred_action_cell_hashes: tuple[str, ...]
    physical_action_axis: tuple[tuple[str, str, int, tuple[int, ...]], ...]
    cell_crosswalk_axis: tuple[tuple[str, str, str, int, str], ...]
    route_scale_h2d_payload_bytes: int
    maximum_h2d_staging_bytes: int
    resident_device_identity: str
    receipt_hash: str = field(init=False)
    production_cuda_execution: bool = True
    private_cpu_reference_execution: bool = False
    exact_endpoint_to_subset_replay: bool = True
    cross_branch_fixed_membership_equal: bool = True
    resident_fixed_membership_owner_bound: bool = True
    tiled_packed_child_materialization_owner_bound: bool = True
    resident_parent_packed_mask_owner_bound: bool = True
    canonical_packed_action_hash_authority: bool = False
    route_scale_d2h_values: int = 0
    route_scale_device_tensor_content_hashes: int = 0
    outside_pair_execution_joined: bool = False
    full_u_action_domain_authority: bool = False
    fqr_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = PHYSICAL_ALTERNATE_DOMAIN_AUTHORITY
    fixed_membership_derivation_rule: str = PHYSICAL_ALTERNATE_FIXED_DERIVATION_RULE
    schema: str = PHYSICAL_ALTERNATE_STRUCTURAL_CUDA_DOMAIN_SCHEMA
    _issuer_token: InitVar[object] = None
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _STRUCTURAL_CUDA_RECEIPT_TOKEN:
            raise TypeError(
                "physical-alternate structural CUDA receipts are issuer-only"
            )
        for name in (
            "design_hash",
            "hbm_forecast_hash",
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
            "shell_freeze_receipt_hash",
            "fixed_outside_removed_route_index_hash",
            "fixed_membership_binding_hash",
            "authority_scope",
            "fixed_membership_derivation_rule",
            "resident_device_identity",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        for name in (
            "proposal_generation",
            "preferred_selected_count",
            "route_count",
            "fixed_outside_removed_count",
            "registered_cell_count",
            "preferred_cell_count_excluded",
            "alternate_cell_count",
            "deduplicated_physical_action_count",
            "route_scale_d2h_values",
            "route_scale_device_tensor_content_hashes",
            "route_scale_h2d_payload_bytes",
            "maximum_h2d_staging_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    name=name.replace("_", " "),
                    positive=name
                    in {
                        "route_count",
                        "preferred_selected_count",
                        "registered_cell_count",
                        "preferred_cell_count_excluded",
                        "alternate_cell_count",
                        "deduplicated_physical_action_count",
                    },
                ),
            )
        shell = tuple(
            _integer(value, name="shell route index") for value in self.shell_indices
        )
        executions = tuple(
            _text(value, name="branch execution hash")
            for value in self.branch_execution_hashes
        )
        ladders = tuple(
            _text(value, name="frozen ladder hash")
            for value in self.branch_frozen_ladder_hashes
        )
        commitments = tuple(
            _text(value, name="label commitment")
            for value in self.branch_label_commitments
        )
        empty_subsets = tuple(
            _text(value, name="empty-subset identity hash")
            for value in self.branch_empty_subset_identity_hashes
        )
        first_subsets = tuple(
            _text(value, name="first-endpoint subset identity hash")
            for value in self.branch_first_endpoint_subset_identity_hashes
        )
        custody_families = tuple(
            _text(value, name="streamed hash-custody family hash")
            for value in self.branch_streamed_hash_custody_family_hashes
        )
        excluded = tuple(
            _text(value, name="excluded P cell hash")
            for value in self.excluded_preferred_action_cell_hashes
        )
        actions = tuple(
            (
                (
                    _text(value[0], name="physical action coordinate hash"),
                    _text(value[1], name="removed-route index hash"),
                    _integer(value[2], name="removed count", positive=True),
                    tuple(
                        _integer(offset, name="selected shell offset")
                        for offset in value[3]
                    ),
                )
                if type(value) is tuple and len(value) == 4 and type(value[3]) is tuple
                else value
            )
            for value in self.physical_action_axis
        )
        crosswalk = tuple(
            (
                (
                    _text(value[0], name="action cell hash"),
                    _text(value[1], name="action cell branch"),
                    _text(value[2], name="resource alias id"),
                    _integer(value[3], name="action cell ordinal"),
                    _text(value[4], name="physical action coordinate hash"),
                )
                if type(value) is tuple and len(value) == 5
                else value
            )
            for value in self.cell_crosswalk_axis
        )
        if (
            not 1 <= len(shell) <= 8
            or len(set(shell)) != len(shell)
            or any(route >= self.route_count for route in shell)
            or len(executions) != 2
            or len(ladders) != 2
            or len(commitments) != 2
            or len(empty_subsets) != 2
            or len(first_subsets) != 2
            or len(custody_families) != 2
            or len(excluded) != self.preferred_cell_count_excluded
            or len(set(excluded)) != len(excluded)
            or len(actions) != self.deduplicated_physical_action_count
            or len({value[0] for value in actions}) != len(actions)
            or any(
                type(value) is not tuple
                or len(value) != 4
                or type(value[0]) is not str
                or type(value[1]) is not str
                or isinstance(value[2], bool)
                or not isinstance(value[2], Integral)
                or type(value[3]) is not tuple
                or value[3] != tuple(sorted(set(value[3])))
                or any(offset >= len(shell) for offset in value[3])
                or value[2] != self.fixed_outside_removed_count + len(value[3])
                for value in actions
            )
            or len(crosswalk) != self.alternate_cell_count
            or len({value[0] for value in crosswalk}) != len(crosswalk)
            or any(
                type(value) is not tuple
                or len(value) != 5
                or any(type(value[index]) is not str for index in (0, 1, 2, 4))
                or isinstance(value[3], bool)
                or not isinstance(value[3], Integral)
                or value[1]
                not in {
                    V1SignedWorkVariableShellQBranch.SPECTRAL.value,
                    V1SignedWorkVariableShellQBranch.DIAGONAL_L1.value,
                }
                for value in crosswalk
            )
            or {value[4] for value in crosswalk} != {value[0] for value in actions}
            or self.registered_cell_count
            != self.preferred_cell_count_excluded + self.alternate_cell_count
        ):
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.INCOMPLETE_CELL_CROSSWALK,
                "CUDA physical action and cell axes do not close",
            )
        object.__setattr__(self, "shell_indices", shell)
        object.__setattr__(self, "branch_execution_hashes", executions)
        object.__setattr__(self, "branch_frozen_ladder_hashes", ladders)
        object.__setattr__(self, "branch_label_commitments", commitments)
        object.__setattr__(self, "branch_empty_subset_identity_hashes", empty_subsets)
        object.__setattr__(
            self,
            "branch_first_endpoint_subset_identity_hashes",
            first_subsets,
        )
        object.__setattr__(
            self,
            "branch_streamed_hash_custody_family_hashes",
            custody_families,
        )
        object.__setattr__(self, "excluded_preferred_action_cell_hashes", excluded)
        object.__setattr__(self, "physical_action_axis", actions)
        object.__setattr__(self, "cell_crosswalk_axis", crosswalk)
        required = (
            self.exact_endpoint_to_subset_replay,
            self.cross_branch_fixed_membership_equal,
            self.resident_fixed_membership_owner_bound,
            self.tiled_packed_child_materialization_owner_bound,
            self.resident_parent_packed_mask_owner_bound,
        )
        if any(type(value) is not bool or value is not True for value in required):
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.AUTHORITY_VIOLATION,
                "CUDA physical-domain ownership invariant was weakened",
            )
        _closed(
            (
                self.canonical_packed_action_hash_authority,
                self.outside_pair_execution_joined,
                self.full_u_action_domain_authority,
                self.fqr_authority,
                self.C_stream_open_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="physical alternate CUDA domain",
        )
        if (
            type(self.production_cuda_execution) is not bool
            or type(self.private_cpu_reference_execution) is not bool
            or self.production_cuda_execution == self.private_cpu_reference_execution
            or (
                self.production_cuda_execution
                and not self.resident_device_identity.startswith("cuda")
            )
            or (
                self.private_cpu_reference_execution
                and self.resident_device_identity != "cpu"
            )
            or self.route_scale_d2h_values != 0
            or self.route_scale_device_tensor_content_hashes != 0
            or (
                self.production_cuda_execution
                and self.route_scale_h2d_payload_bytes != (self.route_count + 7) // 8
            )
            or (
                self.private_cpu_reference_execution
                and self.route_scale_h2d_payload_bytes != 0
            )
            or self.maximum_h2d_staging_bytes != self.route_scale_h2d_payload_bytes
            or self.authority_scope != PHYSICAL_ALTERNATE_DOMAIN_AUTHORITY
            or self.fixed_membership_derivation_rule
            != PHYSICAL_ALTERNATE_FIXED_DERIVATION_RULE
            or self.schema != PHYSICAL_ALTERNATE_STRUCTURAL_CUDA_DOMAIN_SCHEMA
        ):
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.AUTHORITY_VIOLATION,
                "CUDA physical-domain transfer or authority semantics changed",
            )
        seal = content_hash(
            _payload(self), prefix=_PREFIX + "structural-cuda-receipt-integrity-"
        )
        if self._integrity_seal and self._integrity_seal != seal:
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.STALE_SOURCE,
                "CUDA physical-domain receipt integrity changed",
            )
        object.__setattr__(self, "_integrity_seal", seal)
        object.__setattr__(
            self,
            "receipt_hash",
            content_hash(_payload(self), prefix=_PREFIX + "structural-cuda-receipt-"),
        )

    def assert_integrity(self) -> str:
        seal = content_hash(
            _payload(self), prefix=_PREFIX + "structural-cuda-receipt-integrity-"
        )
        expected = content_hash(
            _payload(self), prefix=_PREFIX + "structural-cuda-receipt-"
        )
        if seal != self._integrity_seal or expected != self.receipt_hash:
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.STALE_SOURCE,
                "CUDA physical-domain receipt does not replay",
            )
        return self.receipt_hash


class V1SignedWorkPhysicalAlternateCUDAPackedActionLease:
    """Ephemeral device-packed child owned by one live physical domain."""

    __slots__ = ("_owner", "_tensor", "_coordinate_hash", "_invariant", "_closed")

    def __init__(
        self,
        *,
        owner: "V1SignedWorkPhysicalAlternateStructuralCUDADomainProvider",
        tensor: Any,
        coordinate_hash: str,
    ) -> None:
        self._owner = owner
        self._tensor = tensor
        self._coordinate_hash = coordinate_hash
        self._invariant = (
            int(tensor.data_ptr()),
            int(tensor._version),
            tuple(tensor.shape),
            str(tensor.dtype),
            str(tensor.device),
        )
        self._closed = False

    @property
    def physical_action_coordinate_hash(self) -> str:
        self._require_live()
        return self._coordinate_hash

    @property
    def packed_child(self) -> Any:
        self._require_live()
        return self._tensor

    def _require_live(self) -> None:
        if self._closed or self._owner is None or self._tensor is None:
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.EVICTED,
                "CUDA packed-action lease is closed",
            )
        self._owner.assert_live()
        observed = (
            int(self._tensor.data_ptr()),
            int(self._tensor._version),
            tuple(self._tensor.shape),
            str(self._tensor.dtype),
            str(self._tensor.device),
        )
        if observed != self._invariant:
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.STALE_SOURCE,
                "leased CUDA packed child was mutated",
            )

    def close(self) -> None:
        if self._closed:
            return
        owner = self._owner
        try:
            self._require_live()
        finally:
            self._owner = None
            self._tensor = None
            self._closed = True
            if owner is not None:
                owner._close_materialization_lease()

    def __enter__(self) -> "V1SignedWorkPhysicalAlternateCUDAPackedActionLease":
        self._require_live()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __reduce__(self) -> NoReturn:
        raise TypeError("CUDA packed-action leases are nonserializable")


class V1SignedWorkPhysicalAlternateStructuralCUDAFixedMembershipLease:
    """Authority-inert live borrow of the source-derived fixed-prefix vector."""

    __slots__ = (
        "_closed",
        "_fixed_invariant",
        "_owner",
        "_fixed_membership",
        "_parent_invariant",
        "_resident_parent_packed",
        "_token",
        "borrow_purpose_hash",
        "fixed_membership_binding_hash",
        "physical_domain_receipt_hash",
        "route_count",
        "route_axis_hash",
        "parent_mask_hash",
        "shell_indices",
        "device_identity",
        "downstream_authority",
    )

    def __init__(
        self,
        *,
        owner: "V1SignedWorkPhysicalAlternateStructuralCUDADomainProvider",
        fixed_membership: Any,
        resident_parent_packed: Any,
        token: object,
        borrow_purpose_hash: str,
        _issuer_token: object,
    ) -> None:
        if _issuer_token is not _CUDA_FIXED_MEMBERSHIP_LEASE_TOKEN:
            raise TypeError("fixed-membership leases are provider-issued")
        self._owner = owner
        self._fixed_membership = fixed_membership
        self._resident_parent_packed = resident_parent_packed
        self._token = token
        self._closed = False
        self._fixed_invariant = (
            int(fixed_membership.data_ptr()),
            int(fixed_membership._version),
            tuple(fixed_membership.shape),
            str(fixed_membership.dtype),
            str(fixed_membership.device),
        )
        self._parent_invariant = (
            int(resident_parent_packed.data_ptr()),
            int(resident_parent_packed._version),
            tuple(resident_parent_packed.shape),
            str(resident_parent_packed.dtype),
            str(resident_parent_packed.device),
        )
        self.borrow_purpose_hash = _text(
            borrow_purpose_hash,
            name="fixed-membership borrow purpose hash",
        )
        self.fixed_membership_binding_hash = owner.receipt.fixed_membership_binding_hash
        self.physical_domain_receipt_hash = owner.receipt.receipt_hash
        self.route_count = owner.receipt.route_count
        self.route_axis_hash = owner.receipt.route_axis_hash
        self.parent_mask_hash = owner.receipt.current_mask_hash
        self.shell_indices = owner.receipt.shell_indices
        self.device_identity = str(fixed_membership.device)
        self.downstream_authority = False

    def _require_live(self) -> None:
        if (
            self._closed
            or self._owner is None
            or self._fixed_membership is None
            or self._resident_parent_packed is None
        ):
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.EVICTED,
                "fixed-membership lease is closed",
            )
        self._owner._assert_fixed_membership_lease(
            token=self._token,
            fixed_invariant=self._fixed_invariant,
            parent_invariant=self._parent_invariant,
        )

    def fixed_membership(self) -> Any:
        """Return the existing CUDA bool vector without copying or authority."""

        self._require_live()
        return self._fixed_membership

    def resident_parent_packed(self) -> Any:
        """Return the existing packed parent tensor without copy or digest."""

        self._require_live()
        return self._resident_parent_packed

    def close(self) -> None:
        if self._closed:
            return
        owner = self._owner
        token = self._token
        fixed_invariant = self._fixed_invariant
        parent_invariant = self._parent_invariant
        try:
            owner._close_fixed_membership_lease(
                token=token,
                fixed_invariant=fixed_invariant,
                parent_invariant=parent_invariant,
            )
        finally:
            self._owner = None
            self._fixed_membership = None
            self._resident_parent_packed = None
            self._token = None
            self._closed = True

    def __enter__(
        self,
    ) -> "V1SignedWorkPhysicalAlternateStructuralCUDAFixedMembershipLease":
        self._require_live()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __reduce__(self) -> NoReturn:
        raise TypeError("fixed-membership leases are nonserializable")


def _materialize_packed_child_from_resident_membership(
    *,
    resident_parent_packed: Any,
    route_count: int,
    fixed_membership: Any,
    shell_indices: tuple[int, ...],
    selected_shell_offsets: tuple[int, ...],
    route_tile_size: int,
) -> Any:
    """Tile one exact child without a route-by-action matrix or route D2H.

    The helper is device-polymorphic solely so the bit-ordering kernel can be
    compared with the exact CPU mask contract in focused tests.  Production
    admission occurs only through the CUDA-only provider factory.
    """

    if (
        torch is None
        or type(fixed_membership) is not torch.Tensor
        or type(resident_parent_packed) is not torch.Tensor
    ):
        raise TypeError(
            "packed-child materialization requires exact resident torch tensors"
        )
    routes = _integer(route_count, name="route count", positive=True)
    tile_requested = _integer(route_tile_size, name="route tile size", positive=True)
    if (
        fixed_membership.dtype is not torch.bool
        or tuple(fixed_membership.shape) != (routes,)
        or not fixed_membership.is_contiguous()
        or resident_parent_packed.dtype is not torch.uint8
        or tuple(resident_parent_packed.shape) != ((routes + 7) // 8,)
        or not resident_parent_packed.is_contiguous()
        or resident_parent_packed.device != fixed_membership.device
    ):
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.PACKED_MATERIALIZATION_MISMATCH,
            "resident fixed membership or parent packed mask has the wrong shape",
        )
    shell = tuple(_integer(value, name="shell route index") for value in shell_indices)
    offsets = tuple(
        _integer(value, name="selected shell offset")
        for value in selected_shell_offsets
    )
    if (
        not shell
        or len(set(shell)) != len(shell)
        or any(route >= routes for route in shell)
        or offsets != tuple(sorted(set(offsets)))
        or any(offset >= len(shell) for offset in offsets)
    ):
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.PACKED_MATERIALIZATION_MISMATCH,
            "shell coordinate is not canonical for packed materialization",
        )
    device = fixed_membership.device
    child = resident_parent_packed.clone()
    selected_routes = tuple(shell[offset] for offset in offsets)
    tile = max(8, (tile_requested // 8) * 8)
    weights = torch.tensor(
        (128, 64, 32, 16, 8, 4, 2, 1),
        dtype=torch.uint8,
        device=device,
    )
    for start in range(0, routes, tile):
        stop = min(routes, start + tile)
        remove = fixed_membership[start:stop].clone()
        for route in selected_routes:
            if start <= route < stop:
                remove[route - start] = True
        padding = (-int(remove.numel())) % 8
        if padding:
            remove = torch.cat(
                (remove, torch.zeros((padding,), dtype=torch.bool, device=device))
            )
        clear = (
            (remove.reshape(-1, 8).to(torch.uint8) * weights).sum(dim=1).to(torch.uint8)
        )
        byte_start = start // 8
        byte_stop = byte_start + int(clear.numel())
        child[byte_start:byte_stop] = torch.bitwise_and(
            child[byte_start:byte_stop], torch.bitwise_not(clear)
        )
    return child.contiguous()


class V1SignedWorkPhysicalAlternateStructuralCUDADomainProvider:
    """Source-owned CUDA fixed membership and tiled packed-child materializer."""

    __slots__ = (
        "receipt",
        "forecast",
        "registry",
        "preferred_action_binding",
        "branch_bundles",
        "design",
        "_parent_mask",
        "_fixed_membership",
        "_fixed_invariant",
        "_action_by_coordinate",
        "_resident_parent_packed",
        "_parent_packed_invariant",
        "_active_materializations",
        "_fixed_membership_leases",
        "_live_identity_hash",
        "_evicted",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        receipt: V1SignedWorkPhysicalAlternateStructuralCUDADomainReceipt,
        forecast: V1SignedWorkPhysicalAlternateDomainHBMForecast,
        registry: V1SignedWorkVariableShellActionCellRegistry,
        preferred_action_binding: V1SignedWorkBlockSupportPreferredActionBinding,
        branch_bundles: tuple[V1SignedWorkDeviceShellExecutionBundle, ...],
        design: V1SignedWorkPhysicalAlternateDomainDesign,
        parent_mask: HardMask,
        fixed_membership: Any,
        resident_parent_packed: Any,
        _issuer_token: object,
    ) -> None:
        if _issuer_token is not _STRUCTURAL_CUDA_PROVIDER_TOKEN:
            raise TypeError("physical-alternate CUDA providers are factory-issued")
        if (
            type(receipt)
            is not V1SignedWorkPhysicalAlternateStructuralCUDADomainReceipt
            or type(forecast) is not V1SignedWorkPhysicalAlternateDomainHBMForecast
            or type(registry) is not V1SignedWorkVariableShellActionCellRegistry
            or type(preferred_action_binding)
            is not V1SignedWorkBlockSupportPreferredActionBinding
            or type(design) is not V1SignedWorkPhysicalAlternateDomainDesign
            or type(parent_mask) is not HardMask
            or type(branch_bundles) is not tuple
            or len(branch_bundles) != 2
            or any(
                type(value) is not V1SignedWorkDeviceShellExecutionBundle
                for value in branch_bundles
            )
        ):
            raise TypeError("structural CUDA provider source types changed")
        if (
            torch is None
            or type(fixed_membership) is not torch.Tensor
            or type(resident_parent_packed) is not torch.Tensor
        ):
            raise TypeError("structural CUDA provider requires exact torch tensors")
        receipt.assert_integrity()
        if (
            receipt.production_cuda_execution is not True
            or receipt.private_cpu_reference_execution is not False
            or design.execution_mode
            is not V1SignedWorkPhysicalAlternateDomainExecutionMode.CUDA_PRODUCTION
            or receipt.design_hash != design.design_hash
            or receipt.hbm_forecast_hash != forecast.forecast_hash
            or receipt.registry_receipt_hash != registry.receipt.receipt_hash
            or receipt.preferred_action_binding_hash
            != preferred_action_binding.binding_hash
            or receipt.current_mask_hash != parent_mask.mask_hash
            or receipt.route_axis_hash != parent_mask.route_axis_hash
            or receipt.route_count != parent_mask.route_count
            or receipt.branch_execution_hashes
            != tuple(value.execution_hash for value in branch_bundles)
            or fixed_membership.dtype is not torch.bool
            or tuple(fixed_membership.shape) != (receipt.route_count,)
            or not fixed_membership.is_contiguous()
            or resident_parent_packed.dtype is not torch.uint8
            or tuple(resident_parent_packed.shape) != ((receipt.route_count + 7) // 8,)
            or not resident_parent_packed.is_contiguous()
            or fixed_membership.device != resident_parent_packed.device
            or str(fixed_membership.device) != receipt.resident_device_identity
        ):
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.SOURCE_MISMATCH,
                "structural CUDA provider sources do not replay its receipt",
            )
        self.receipt = receipt
        self.forecast = forecast
        self.registry = registry
        self.preferred_action_binding = preferred_action_binding
        self.branch_bundles = branch_bundles
        self.design = design
        self._parent_mask = parent_mask
        self._fixed_membership = fixed_membership
        self._resident_parent_packed = resident_parent_packed
        self._fixed_invariant = (
            int(fixed_membership.data_ptr()),
            int(fixed_membership._version),
            tuple(fixed_membership.shape),
            str(fixed_membership.dtype),
            str(fixed_membership.device),
        )
        self._parent_packed_invariant = (
            int(resident_parent_packed.data_ptr()),
            int(resident_parent_packed._version),
            tuple(resident_parent_packed.shape),
            str(resident_parent_packed.dtype),
            str(resident_parent_packed.device),
        )
        self._action_by_coordinate = {
            value[0]: value for value in receipt.physical_action_axis
        }
        self._active_materializations = 0
        self._fixed_membership_leases: set[object] = set()
        self._evicted = False
        self._live_identity_hash = self._current_live_identity_hash()
        self.assert_live()

    def _current_live_identity_hash(self) -> str:
        return content_hash(
            {
                "receipt_hash": self.receipt.receipt_hash,
                "forecast_hash": self.forecast.forecast_hash,
                "registry_receipt_hash": self.registry.receipt.receipt_hash,
                "preferred_binding_hash": self.preferred_action_binding.binding_hash,
                "branch_execution_hashes": tuple(
                    value.execution_hash for value in self.branch_bundles
                ),
                "parent_mask_hash": self._parent_mask.mask_hash,
            },
            prefix=_PREFIX + "cuda-live-provider-",
        )

    def assert_live(
        self,
    ) -> "V1SignedWorkPhysicalAlternateStructuralCUDADomainProvider":
        if self._evicted:
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.EVICTED,
                "CUDA physical domain was evicted",
            )
        try:
            self.receipt.assert_integrity()
            self.registry.assert_live_sources()
            self.preferred_action_binding.assert_live()
            for bundle in self.branch_bundles:
                if bundle.label_handle.route_count != self.receipt.route_count:
                    raise ValueError("branch label route count changed")
        except (RuntimeError, PermissionError, TypeError, ValueError) as error:
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.STALE_SOURCE,
                f"CUDA physical-domain source is no longer live: {error}",
            )
        observed = (
            int(self._fixed_membership.data_ptr()),
            int(self._fixed_membership._version),
            tuple(self._fixed_membership.shape),
            str(self._fixed_membership.dtype),
            str(self._fixed_membership.device),
        )
        observed_parent = (
            int(self._resident_parent_packed.data_ptr()),
            int(self._resident_parent_packed._version),
            tuple(self._resident_parent_packed.shape),
            str(self._resident_parent_packed.dtype),
            str(self._resident_parent_packed.device),
        )
        if (
            observed != self._fixed_invariant
            or observed_parent != self._parent_packed_invariant
            or self._current_live_identity_hash() != self._live_identity_hash
        ):
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.STALE_SOURCE,
                "CUDA fixed-membership ownership changed",
            )
        return self

    def materialize_packed_child(
        self, *, physical_action_coordinate_hash: str
    ) -> V1SignedWorkPhysicalAlternateCUDAPackedActionLease:
        self.assert_live()
        if self._active_materializations:
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.HBM_FORECAST_EXCEEDED,
                "the HBM forecast authorizes one live packed child at a time",
            )
        coordinate = _text(
            physical_action_coordinate_hash, name="physical action coordinate hash"
        )
        try:
            _coordinate, _removed_hash, _removed_count, offsets = (
                self._action_by_coordinate[coordinate]
            )
        except KeyError:
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.INCOMPLETE_CELL_CROSSWALK,
                "physical action coordinate is absent or is preferred P",
            )
        child = _materialize_packed_child_from_resident_membership(
            resident_parent_packed=self._resident_parent_packed,
            route_count=self.receipt.route_count,
            fixed_membership=self._fixed_membership,
            shell_indices=self.receipt.shell_indices,
            selected_shell_offsets=offsets,
            route_tile_size=self.design.route_tile_size,
        )
        self._active_materializations += 1
        return V1SignedWorkPhysicalAlternateCUDAPackedActionLease(
            owner=self,
            tensor=child.contiguous(),
            coordinate_hash=coordinate,
        )

    def open_fixed_membership_lease(
        self, *, borrow_purpose_hash: str
    ) -> V1SignedWorkPhysicalAlternateStructuralCUDAFixedMembershipLease:
        """Borrow fixed-prefix membership for a source-bound device consumer."""

        self.assert_live()
        purpose = _text(
            borrow_purpose_hash,
            name="fixed-membership borrow purpose hash",
        )
        token = object()
        self._fixed_membership_leases.add(token)
        try:
            return V1SignedWorkPhysicalAlternateStructuralCUDAFixedMembershipLease(
                owner=self,
                fixed_membership=self._fixed_membership,
                resident_parent_packed=self._resident_parent_packed,
                token=token,
                borrow_purpose_hash=purpose,
                _issuer_token=_CUDA_FIXED_MEMBERSHIP_LEASE_TOKEN,
            )
        except BaseException:
            self._fixed_membership_leases.discard(token)
            raise

    def _assert_fixed_membership_lease(
        self,
        *,
        token: object,
        fixed_invariant: tuple[object, ...],
        parent_invariant: tuple[object, ...],
    ) -> None:
        self.assert_live()
        if token not in self._fixed_membership_leases:
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.STALE_SOURCE,
                "fixed-membership lease token is not active",
            )
        if (
            fixed_invariant != self._fixed_invariant
            or parent_invariant != self._parent_packed_invariant
        ):
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.STALE_SOURCE,
                "fixed-membership lease storage identity changed",
            )

    def _close_fixed_membership_lease(
        self,
        *,
        token: object,
        fixed_invariant: tuple[object, ...],
        parent_invariant: tuple[object, ...],
    ) -> None:
        try:
            self._assert_fixed_membership_lease(
                token=token,
                fixed_invariant=fixed_invariant,
                parent_invariant=parent_invariant,
            )
        finally:
            self._fixed_membership_leases.discard(token)

    def _close_materialization_lease(self) -> None:
        if self._active_materializations <= 0:
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.STALE_SOURCE,
                "unknown CUDA materialization lease",
            )
        self._active_materializations -= 1

    def require_canonical_full_u_certifier_authority(self) -> NoReturn:
        self.assert_live()
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.AUTHORITY_VIOLATION,
            "outside-pair execution has not joined the CUDA physical domain",
        )

    def evict(self) -> None:
        self.assert_live()
        if self._active_materializations or self._fixed_membership_leases:
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.AUTHORITY_VIOLATION,
                "cannot evict a CUDA physical domain with active device leases",
            )
        self._fixed_membership = None
        self._resident_parent_packed = None
        self._parent_mask = None
        self._action_by_coordinate = {}
        self._fixed_membership_leases = set()
        self._evicted = True

    def __reduce__(self) -> NoReturn:
        raise TypeError("live CUDA physical-alternate providers are nonserializable")


def build_v1_signed_work_physical_alternate_structural_domain_cuda(
    *,
    registry: V1SignedWorkVariableShellActionCellRegistry,
    preferred_action_binding: V1SignedWorkBlockSupportPreferredActionBinding,
    parent_mask: HardMask,
    branch_bundles: Sequence[V1SignedWorkDeviceShellExecutionBundle],
    design: V1SignedWorkPhysicalAlternateDomainDesign,
    device_total_memory_bytes: int,
    current_reserved_bytes: int,
) -> V1SignedWorkPhysicalAlternateStructuralCUDADomainProvider:
    """Bind both exact resident branch ladders to one physical action domain."""

    if type(design) is not V1SignedWorkPhysicalAlternateDomainDesign:
        raise TypeError("CUDA physical domain requires the exact design")
    if (
        design.execution_mode
        is not V1SignedWorkPhysicalAlternateDomainExecutionMode.CUDA_PRODUCTION
    ):
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.INVALID_DESIGN,
            "production constructor requires CUDA mode",
        )
    registry_receipt = _validate_structural_source_family(
        registry=registry,
        preferred_action_binding=preferred_action_binding,
        parent_mask=parent_mask,
    )
    if (
        registry.design.execution_mode
        is not V1SignedWorkVariableShellActionSeparationMode.CUDA_PRODUCTION
        or torch is None
        or not bool(torch.cuda.is_available())
    ):
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.CUDA_REQUIRED,
            "production physical domain forbids CPU fallback",
        )
    if (
        design.deterministic_cuda_required
        and not torch.are_deterministic_algorithms_enabled()
    ):
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.CUDA_REQUIRED,
            "deterministic CUDA algorithms are required",
        )
    bundles = tuple(branch_bundles)
    if len(bundles) != 2 or any(
        type(value) is not V1SignedWorkDeviceShellExecutionBundle for value in bundles
    ):
        raise TypeError("CUDA physical domain requires two exact branch bundles")
    fixed_hash, fixed_count, shell_indices = _fixed_prefix_identity(registry)
    forecast = estimate_v1_signed_work_physical_alternate_domain_hbm(
        design=design,
        route_count=parent_mask.route_count,
        shell_size=len(shell_indices),
        registered_cell_count=registry_receipt.total_action_cell_count,
        physical_action_count_upper_bound=registry_receipt.total_action_cell_count,
        live_source_device_bytes=_retained_structural_source_device_bytes(
            registry=registry,
            route_count=parent_mask.route_count,
        ),
        device_total_memory_bytes=device_total_memory_bytes,
        current_reserved_bytes=current_reserved_bytes,
    )
    if not forecast.admitted:
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.HBM_FORECAST_EXCEEDED,
            "CUDA physical-domain forecast exceeds HBM ceiling",
        )

    fixed_membership: Any | None = None
    execution_hashes: list[str] = []
    ladder_hashes: list[str] = []
    label_commitments: list[str] = []
    empty_subset_identity_hashes: list[str] = []
    first_endpoint_subset_identity_hashes: list[str] = []
    streamed_hash_custody_family_hashes: list[str] = []
    operation = V1DeviceShellBorrowOperation.BLOCK_SUPPORT_PHYSICAL_ALTERNATE_DOMAIN
    for source, bundle in zip(registry.branch_sources, bundles, strict=True):
        source_receipt = source.assert_live_source().receipt
        branch_identity = bundle.branch_identity
        frozen = bundle.frozen_ladder
        if (
            branch_identity.branch.value != source.branch.value
            or branch_identity.signed_work_branch_receipt_hash
            != source_receipt.kkt_branch_source_receipt_hash
            or frozen.parent_mask_hash != registry_receipt.current_mask_hash
            or frozen.route_axis_hash != registry_receipt.route_axis_hash
            or frozen.route_count != parent_mask.route_count
            or bundle.production_cuda_execution is not True
            or bundle.private_cpu_reference_execution is not False
            or bundle.label_handle.backend_kind != "torch_cuda"
        ):
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.SOURCE_MISMATCH,
                "branch bundle does not bind the registry source and parent",
            )
        source_device_probe = content_hash(
            {
                "source_receipt_hash": source_receipt.receipt_hash,
                "branch_execution_hash": bundle.execution_hash,
                "operation": "physical_alternate_domain_device_identity",
            },
            prefix=_PREFIX + "cuda-source-device-probe-",
        )
        with source.source_handle.borrow_compact_tensors(
            borrow_purpose_hash=source_device_probe
        ) as compact_lease:
            compact_device = compact_lease.compact_tensors()[0].device
        matched_offsets: list[tuple[int, ...]] = []
        matched_subset_hashes: list[str] = []
        for endpoint in frozen.actions:
            matches = tuple(
                value
                for value in source.subset_identities
                if value.removed_count == endpoint.cumulative_removed_count
                and value.removed_route_index_hash == endpoint.removed_route_index_hash
            )
            if len(matches) != 1:
                _refuse(
                    V1SignedWorkPhysicalAlternateDomainRefusalCode.SOURCE_MISMATCH,
                    "frozen endpoint does not replay to exactly one registered subset",
                )
            matched_offsets.append(matches[0].shell_offsets)
            matched_subset_hashes.append(matches[0].subset_hash)
        empty_subset = tuple(
            value for value in source.subset_identities if value.shell_offsets == ()
        )
        if (
            len(empty_subset) != 1
            or empty_subset[0].removed_route_index_hash != fixed_hash
            or empty_subset[0].removed_count != fixed_count
        ):
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.SOURCE_MISMATCH,
                "branch empty-subset custody differs from the fixed removed set",
            )
        with bundle.label_handle.borrow_for_registered_operation(
            operation=operation
        ) as label_lease:
            labels = label_lease._labels_for_registered_operation(operation=operation)
            if (
                type(labels) is not torch.Tensor
                or labels.device.type != "cuda"
                or labels.dtype is not torch.uint8
                or tuple(labels.shape) != (parent_mask.route_count,)
                or not labels.is_contiguous()
            ):
                _refuse(
                    V1SignedWorkPhysicalAlternateDomainRefusalCode.CUDA_REQUIRED,
                    "resident branch labels are not canonical CUDA uint8",
                )
            if labels.device != compact_device:
                _refuse(
                    V1SignedWorkPhysicalAlternateDomainRefusalCode.SOURCE_MISMATCH,
                    "branch labels and compact source tensors occupy different devices",
                )
            first_offsets = set(matched_offsets[0])
            for offset, route in enumerate(shell_indices):
                observed_first = bool((labels[route] == 1).item())
                if observed_first is not (offset in first_offsets):
                    _refuse(
                        V1SignedWorkPhysicalAlternateDomainRefusalCode.SOURCE_MISMATCH,
                        "first endpoint label disagrees with subset custody",
                    )
            if (
                int((labels == 1).sum(dtype=torch.int64).item())
                != frozen.cumulative_counts[0]
            ):
                _refuse(
                    V1SignedWorkPhysicalAlternateDomainRefusalCode.SOURCE_MISMATCH,
                    "label-one census differs from first frozen endpoint",
                )
            fixed = labels == 1
            shell_tensor = torch.tensor(
                shell_indices, dtype=torch.int64, device=labels.device
            )
            fixed.index_fill_(0, shell_tensor, False)
            if int(fixed.sum(dtype=torch.int64).item()) != fixed_count:
                _refuse(
                    V1SignedWorkPhysicalAlternateDomainRefusalCode.SOURCE_MISMATCH,
                    "derived fixed membership count differs from hash custody",
                )
            if fixed_membership is None:
                fixed_membership = fixed
            elif fixed_membership.device != fixed.device or not bool(
                torch.equal(fixed_membership, fixed)
            ):
                _refuse(
                    V1SignedWorkPhysicalAlternateDomainRefusalCode.SOURCE_MISMATCH,
                    "Q branches derive different fixed-prefix membership",
                )
        execution_hashes.append(bundle.execution_hash)
        ladder_hashes.append(frozen.receipt_hash)
        label_commitments.append(frozen.label_commitment)
        empty_subset_identity_hashes.append(empty_subset[0].subset_hash)
        first_endpoint_subset_identity_hashes.append(matched_subset_hashes[0])
        streamed_hash_custody_family_hashes.append(
            empty_subset[0].streamed_hash_custody_family_hash
        )
    if fixed_membership is None:  # pragma: no cover - exact two-branch guard.
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.SOURCE_MISMATCH,
            "no branch produced fixed membership",
        )
    del fixed
    parent_host = np.frombuffer(parent_mask.packed_bits, dtype=np.uint8).copy()
    resident_parent_packed = torch.from_numpy(parent_host).to(
        device=fixed_membership.device,
        non_blocking=False,
    )

    cells = tuple(registry.iter_action_cells())
    excluded: list[str] = []
    crosswalk: list[tuple[str, str, str, int, str]] = []
    actions: dict[str, tuple[str, str, int, tuple[int, ...]]] = {}
    for cell in cells:
        if (
            cell.removed_route_index_hash
            == preferred_action_binding.preferred_removed_route_index_hash
            and fixed_count + len(cell.selected_shell_offsets)
            == preferred_action_binding.preferred_selected_count
        ):
            excluded.append(cell.action_cell_hash)
            continue
        if fixed_count + len(cell.selected_shell_offsets) <= 0:
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.PACKED_MATERIALIZATION_MISMATCH,
                "the physical alternate domain contains an empty pruning action",
            )
        action = (
            cell.physical_action_coordinate_hash,
            cell.removed_route_index_hash,
            fixed_count + len(cell.selected_shell_offsets),
            cell.selected_shell_offsets,
        )
        previous = actions.setdefault(cell.physical_action_coordinate_hash, action)
        if previous != action:
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.INCOMPLETE_CELL_CROSSWALK,
                "aliases disagree about one physical action",
            )
        crosswalk.append(
            (
                cell.action_cell_hash,
                cell.branch.value,
                cell.resource_alias_id,
                cell.action_cell_ordinal,
                cell.physical_action_coordinate_hash,
            )
        )
    if (
        len(cells) != registry_receipt.total_action_cell_count
        or not excluded
        or not crosswalk
    ):
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.INCOMPLETE_CELL_CROSSWALK,
            "registered cell partition is incomplete",
        )
    binding_hash = content_hash(
        {
            "registry_receipt_hash": registry_receipt.receipt_hash,
            "branch_execution_hashes": tuple(execution_hashes),
            "branch_frozen_ladder_hashes": tuple(ladder_hashes),
            "branch_label_commitments": tuple(label_commitments),
            "branch_empty_subset_identity_hashes": tuple(empty_subset_identity_hashes),
            "branch_first_endpoint_subset_identity_hashes": tuple(
                first_endpoint_subset_identity_hashes
            ),
            "branch_streamed_hash_custody_family_hashes": tuple(
                streamed_hash_custody_family_hashes
            ),
            "fixed_removed_route_index_hash": fixed_hash,
            "fixed_count": fixed_count,
            "derivation": PHYSICAL_ALTERNATE_FIXED_DERIVATION_RULE,
        },
        prefix=_PREFIX + "cuda-fixed-membership-binding-",
    )
    receipt = V1SignedWorkPhysicalAlternateStructuralCUDADomainReceipt(
        design_hash=design.design_hash,
        hbm_forecast_hash=forecast.forecast_hash,
        registry_receipt_hash=registry_receipt.receipt_hash,
        source_registration_hash=preferred_action_binding.source_registration_hash,
        preferred_action_binding_hash=preferred_action_binding.binding_hash,
        preferred_action_hash=preferred_action_binding.preferred_action_hash,
        preferred_resulting_mask_hash=(
            preferred_action_binding.preferred_resulting_mask_hash
        ),
        preferred_removed_route_index_hash=preferred_action_binding.preferred_removed_route_index_hash,
        preferred_selected_count=preferred_action_binding.preferred_selected_count,
        model_identity_hash=registry_receipt.model_identity_hash,
        chart_identity_hash=registry_receipt.chart_identity_hash,
        current_mask_hash=registry_receipt.current_mask_hash,
        proposal_generation=registry_receipt.proposal_generation,
        route_axis_hash=registry_receipt.route_axis_hash,
        route_id_axis_hash=registry_receipt.route_id_axis_hash,
        native_fiber_axis_receipt_hash=(
            registry_receipt.native_fiber_axis_receipt_hash
        ),
        native_fiber_measure_hash=registry_receipt.native_fiber_measure_hash,
        route_count=parent_mask.route_count,
        registered_d_prefix_hash=registry_receipt.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=registry_receipt.registered_d_w2_prefix_hash,
        shell_freeze_receipt_hash=registry_receipt.shell_freeze_receipt_hash,
        shell_indices=shell_indices,
        fixed_outside_removed_route_index_hash=fixed_hash,
        fixed_outside_removed_count=fixed_count,
        branch_execution_hashes=tuple(execution_hashes),
        branch_frozen_ladder_hashes=tuple(ladder_hashes),
        branch_label_commitments=tuple(label_commitments),
        branch_empty_subset_identity_hashes=tuple(empty_subset_identity_hashes),
        branch_first_endpoint_subset_identity_hashes=tuple(
            first_endpoint_subset_identity_hashes
        ),
        branch_streamed_hash_custody_family_hashes=tuple(
            streamed_hash_custody_family_hashes
        ),
        fixed_membership_binding_hash=binding_hash,
        registered_cell_count=len(cells),
        preferred_cell_count_excluded=len(excluded),
        alternate_cell_count=len(crosswalk),
        deduplicated_physical_action_count=len(actions),
        excluded_preferred_action_cell_hashes=tuple(sorted(excluded)),
        physical_action_axis=tuple(sorted(actions.values())),
        cell_crosswalk_axis=tuple(crosswalk),
        route_scale_h2d_payload_bytes=forecast.route_scale_h2d_payload_bytes,
        maximum_h2d_staging_bytes=forecast.maximum_h2d_staging_bytes,
        resident_device_identity=str(fixed_membership.device),
        _issuer_token=_STRUCTURAL_CUDA_RECEIPT_TOKEN,
    )
    parent_snapshot = HardMask(
        route_axis_hash=parent_mask.route_axis_hash,
        route_count=parent_mask.route_count,
        packed_bits=bytes(parent_mask.packed_bits),
    )
    return V1SignedWorkPhysicalAlternateStructuralCUDADomainProvider(
        receipt=receipt,
        forecast=forecast,
        registry=registry,
        preferred_action_binding=preferred_action_binding,
        branch_bundles=bundles,
        design=design,
        parent_mask=parent_snapshot,
        fixed_membership=fixed_membership,
        resident_parent_packed=resident_parent_packed,
        _issuer_token=_STRUCTURAL_CUDA_PROVIDER_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkPhysicalAlternateBlockSupportBindingReceipt:
    """One acyclic residual-block join onto reusable structural custody."""

    structural_domain_receipt_hash: str
    source_registration_hash: str
    d_fixed_support_receipt_hash: str
    residual_support_receipt_hash: str
    d_fixed_registration_hash: str
    residual_registration_hash: str
    d_fixed_context_group_id: str
    residual_context_group_id: str
    d_fixed_block_partition_hash: str
    residual_block_partition_hash: str
    d_fixed_block_ordinal: int
    residual_block_ordinal: int
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
    fixed_membership_binding_hash: str
    resident_device_identity: str
    support_source_live_device_bytes: int
    receipt_hash: str = field(init=False)
    source_role_axis: tuple[V1SignedWorkBlockSupportSourceRole, ...] = (
        V1SignedWorkBlockSupportSourceRole.D_FIXED,
        V1SignedWorkBlockSupportSourceRole.RESIDUAL_BLOCK,
    )
    structural_membership_copied: bool = False
    owned_structural_device_bytes: int = 0
    outside_pair_execution_joined: bool = False
    full_u_action_domain_authority: bool = False
    fqr_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = PHYSICAL_ALTERNATE_BLOCK_SUPPORT_BINDING_SCHEMA
    _issuer_token: InitVar[object] = None
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _BLOCK_SUPPORT_BINDING_RECEIPT_TOKEN:
            raise TypeError("physical-alternate block bindings are issuer-only")
        for name in (
            "structural_domain_receipt_hash",
            "source_registration_hash",
            "d_fixed_support_receipt_hash",
            "residual_support_receipt_hash",
            "d_fixed_registration_hash",
            "residual_registration_hash",
            "d_fixed_context_group_id",
            "residual_context_group_id",
            "d_fixed_block_partition_hash",
            "residual_block_partition_hash",
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
            "fixed_membership_binding_hash",
            "resident_device_identity",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        for name in (
            "d_fixed_block_ordinal",
            "residual_block_ordinal",
            "preferred_selected_count",
            "proposal_generation",
            "route_count",
            "support_source_live_device_bytes",
            "owned_structural_device_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    name=name.replace("_", " "),
                    positive=name
                    in {
                        "preferred_selected_count",
                        "route_count",
                        "support_source_live_device_bytes",
                    },
                ),
            )
        roles = tuple(self.source_role_axis)
        if roles != (
            V1SignedWorkBlockSupportSourceRole.D_FIXED,
            V1SignedWorkBlockSupportSourceRole.RESIDUAL_BLOCK,
        ):
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.SOURCE_MISMATCH,
                "block-support binding roles must be ordered D-fixed/residual",
            )
        object.__setattr__(self, "source_role_axis", roles)
        _closed(
            (
                self.structural_membership_copied,
                self.outside_pair_execution_joined,
                self.full_u_action_domain_authority,
                self.fqr_authority,
                self.C_stream_open_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="physical alternate per-block binding",
        )
        if (
            self.owned_structural_device_bytes != 0
            or self.support_source_live_device_bytes != 4 * self.route_count * 8
            or self.schema != PHYSICAL_ALTERNATE_BLOCK_SUPPORT_BINDING_SCHEMA
        ):
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.AUTHORITY_VIOLATION,
                "per-block binding copied structural state or mis-accounted support sources",
            )
        seal = content_hash(_payload(self), prefix=_PREFIX + "block-binding-integrity-")
        if self._integrity_seal and self._integrity_seal != seal:
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.STALE_SOURCE,
                "per-block binding receipt integrity changed",
            )
        object.__setattr__(self, "_integrity_seal", seal)
        object.__setattr__(
            self,
            "receipt_hash",
            content_hash(_payload(self), prefix=_PREFIX + "block-binding-"),
        )

    def assert_integrity(self) -> str:
        seal = content_hash(_payload(self), prefix=_PREFIX + "block-binding-integrity-")
        expected = content_hash(_payload(self), prefix=_PREFIX + "block-binding-")
        if seal != self._integrity_seal or expected != self.receipt_hash:
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.STALE_SOURCE,
                "per-block binding receipt does not replay",
            )
        return self.receipt_hash


def _issue_v1_signed_work_physical_alternate_block_support_binding_receipt(
    *,
    structural_receipt: V1SignedWorkPhysicalAlternateStructuralCUDADomainReceipt,
    d_fixed_support: V1SignedWorkBlockSupportSourceDeviceHandle,
    residual_support: V1SignedWorkBlockSupportSourceDeviceHandle,
    resident_device_identity: str,
) -> V1SignedWorkPhysicalAlternateBlockSupportBindingReceipt:
    """Pure receipt join used by the live binder and hash-DAG tests."""

    if (
        type(structural_receipt)
        is not V1SignedWorkPhysicalAlternateStructuralCUDADomainReceipt
        or type(d_fixed_support) is not V1SignedWorkBlockSupportSourceDeviceHandle
        or type(residual_support) is not V1SignedWorkBlockSupportSourceDeviceHandle
    ):
        raise TypeError("block binding requires exact structural/support sources")
    try:
        structural_receipt.assert_integrity()
        d_fixed_support.assert_live()
        residual_support.assert_live()
    except (RuntimeError, PermissionError, TypeError, ValueError) as error:
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.STALE_SOURCE,
            f"one per-block source is stale: {error}",
        )
    d_receipt = d_fixed_support.receipt
    residual_receipt = residual_support.receipt
    if (
        d_receipt.source_role is not V1SignedWorkBlockSupportSourceRole.D_FIXED
        or residual_receipt.source_role
        is not V1SignedWorkBlockSupportSourceRole.RESIDUAL_BLOCK
    ):
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.SOURCE_MISMATCH,
            "support handles must be ordered D-fixed then residual block",
        )
    structural_names = (
        "model_identity_hash",
        "source_registration_hash",
        "chart_identity_hash",
        "current_mask_hash",
        "proposal_generation",
        "route_axis_hash",
        "route_id_axis_hash",
        "native_fiber_axis_receipt_hash",
        "native_fiber_measure_hash",
        "registered_d_prefix_hash",
        "registered_d_w2_prefix_hash",
    )
    if any(
        getattr(d_receipt, name) != getattr(structural_receipt, name)
        or getattr(residual_receipt, name) != getattr(structural_receipt, name)
        for name in structural_names
    ):
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.SOURCE_MISMATCH,
            "block-support source differs from structural chart/mask/prefix custody",
        )
    if (
        d_receipt.route_count != structural_receipt.route_count
        or residual_receipt.route_count != structural_receipt.route_count
        or d_receipt.preferred_action_binding_hash
        != structural_receipt.preferred_action_binding_hash
        or residual_receipt.preferred_action_binding_hash
        != structural_receipt.preferred_action_binding_hash
        or d_receipt.preferred_action_hash != structural_receipt.preferred_action_hash
        or residual_receipt.preferred_action_hash
        != structural_receipt.preferred_action_hash
        or d_receipt.preferred_resulting_mask_hash
        != structural_receipt.preferred_resulting_mask_hash
        or residual_receipt.preferred_resulting_mask_hash
        != structural_receipt.preferred_resulting_mask_hash
        or d_receipt.preferred_removed_route_index_hash
        != structural_receipt.preferred_removed_route_index_hash
        or residual_receipt.preferred_removed_route_index_hash
        != structural_receipt.preferred_removed_route_index_hash
        or d_receipt.preferred_selected_count
        != residual_receipt.preferred_selected_count
        or d_receipt.preferred_selected_count
        != structural_receipt.preferred_selected_count
        or d_receipt.block_partition_hash != residual_receipt.block_partition_hash
        or (
            structural_receipt.production_cuda_execution
            and (
                d_receipt.execution_mode
                is not V1SignedWorkBlockSupportExecutionMode.CUDA_PRODUCTION
                or residual_receipt.execution_mode
                is not V1SignedWorkBlockSupportExecutionMode.CUDA_PRODUCTION
            )
        )
        or (
            structural_receipt.private_cpu_reference_execution
            and (
                d_receipt.execution_mode
                is not V1SignedWorkBlockSupportExecutionMode.CPU_REFERENCE_TEST_ONLY
                or residual_receipt.execution_mode
                is not V1SignedWorkBlockSupportExecutionMode.CPU_REFERENCE_TEST_ONLY
            )
        )
    ):
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.SOURCE_MISMATCH,
            "native-fiber or preferred-action support identity differs",
        )
    device = _text(resident_device_identity, name="resident device identity")
    if device != structural_receipt.resident_device_identity:
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.SOURCE_MISMATCH,
            "binding device differs from structural membership device",
        )
    with (
        d_fixed_support.open_coefficient_lease() as d_lease,
        residual_support.open_coefficient_lease() as residual_lease,
    ):
        if (
            str(d_lease.tensor.device) != device
            or str(residual_lease.tensor.device) != device
        ):
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.SOURCE_MISMATCH,
                "support coefficients are not colocated with structural membership",
            )
    return V1SignedWorkPhysicalAlternateBlockSupportBindingReceipt(
        structural_domain_receipt_hash=structural_receipt.receipt_hash,
        source_registration_hash=structural_receipt.source_registration_hash,
        d_fixed_support_receipt_hash=d_receipt.receipt_hash,
        residual_support_receipt_hash=residual_receipt.receipt_hash,
        d_fixed_registration_hash=d_receipt.registration_hash,
        residual_registration_hash=residual_receipt.registration_hash,
        d_fixed_context_group_id=d_receipt.context_group_id,
        residual_context_group_id=residual_receipt.context_group_id,
        d_fixed_block_partition_hash=d_receipt.block_partition_hash,
        residual_block_partition_hash=residual_receipt.block_partition_hash,
        d_fixed_block_ordinal=d_receipt.block_ordinal,
        residual_block_ordinal=residual_receipt.block_ordinal,
        preferred_action_binding_hash=structural_receipt.preferred_action_binding_hash,
        preferred_action_hash=structural_receipt.preferred_action_hash,
        preferred_resulting_mask_hash=structural_receipt.preferred_resulting_mask_hash,
        preferred_removed_route_index_hash=(
            structural_receipt.preferred_removed_route_index_hash
        ),
        preferred_selected_count=d_receipt.preferred_selected_count,
        model_identity_hash=structural_receipt.model_identity_hash,
        chart_identity_hash=structural_receipt.chart_identity_hash,
        current_mask_hash=structural_receipt.current_mask_hash,
        proposal_generation=structural_receipt.proposal_generation,
        route_axis_hash=structural_receipt.route_axis_hash,
        route_id_axis_hash=structural_receipt.route_id_axis_hash,
        route_count=structural_receipt.route_count,
        native_fiber_axis_receipt_hash=d_receipt.native_fiber_axis_receipt_hash,
        native_fiber_measure_hash=d_receipt.native_fiber_measure_hash,
        registered_d_prefix_hash=structural_receipt.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=structural_receipt.registered_d_w2_prefix_hash,
        fixed_membership_binding_hash=structural_receipt.fixed_membership_binding_hash,
        resident_device_identity=device,
        support_source_live_device_bytes=4 * structural_receipt.route_count * 8,
        _issuer_token=_BLOCK_SUPPORT_BINDING_RECEIPT_TOKEN,
    )


class V1SignedWorkPhysicalAlternateBlockSupportBinding:
    """Live per-block source join with no structural membership copy."""

    __slots__ = (
        "receipt",
        "structural_provider",
        "d_fixed_support",
        "residual_support",
        "_live_identity_hash",
        "_evicted",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        receipt: V1SignedWorkPhysicalAlternateBlockSupportBindingReceipt,
        structural_provider: V1SignedWorkPhysicalAlternateStructuralCUDADomainProvider,
        d_fixed_support: V1SignedWorkBlockSupportSourceDeviceHandle,
        residual_support: V1SignedWorkBlockSupportSourceDeviceHandle,
        _issuer_token: object,
    ) -> None:
        if _issuer_token is not _BLOCK_SUPPORT_BINDING_TOKEN:
            raise TypeError("live physical-alternate block bindings are factory-issued")
        if (
            type(receipt) is not V1SignedWorkPhysicalAlternateBlockSupportBindingReceipt
            or type(structural_provider)
            is not V1SignedWorkPhysicalAlternateStructuralCUDADomainProvider
            or type(d_fixed_support) is not V1SignedWorkBlockSupportSourceDeviceHandle
            or type(residual_support) is not V1SignedWorkBlockSupportSourceDeviceHandle
        ):
            raise TypeError("live per-block binding source types changed")
        receipt.assert_integrity()
        if (
            receipt.structural_domain_receipt_hash
            != structural_provider.receipt.receipt_hash
            or receipt.d_fixed_support_receipt_hash
            != d_fixed_support.receipt.receipt_hash
            or receipt.residual_support_receipt_hash
            != residual_support.receipt.receipt_hash
        ):
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.SOURCE_MISMATCH,
                "live per-block sources do not replay the binding receipt",
            )
        self.receipt = receipt
        self.structural_provider = structural_provider
        self.d_fixed_support = d_fixed_support
        self.residual_support = residual_support
        self._evicted = False
        self._live_identity_hash = self._current_live_identity_hash()
        self.assert_live()

    def _current_live_identity_hash(self) -> str:
        return content_hash(
            {
                "binding_receipt_hash": self.receipt.receipt_hash,
                "structural_receipt_hash": self.structural_provider.receipt.receipt_hash,
                "d_fixed_support_receipt_hash": self.d_fixed_support.receipt.receipt_hash,
                "residual_support_receipt_hash": self.residual_support.receipt.receipt_hash,
            },
            prefix=_PREFIX + "live-block-binding-",
        )

    def assert_live(self) -> "V1SignedWorkPhysicalAlternateBlockSupportBinding":
        if self._evicted:
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.EVICTED,
                "physical-alternate block binding was evicted",
            )
        try:
            self.receipt.assert_integrity()
            self.structural_provider.assert_live()
            self.d_fixed_support.assert_live()
            self.residual_support.assert_live()
        except (RuntimeError, PermissionError, TypeError, ValueError) as error:
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.STALE_SOURCE,
                f"one bound per-block source is stale: {error}",
            )
        if self._current_live_identity_hash() != self._live_identity_hash:
            _refuse(
                V1SignedWorkPhysicalAlternateDomainRefusalCode.STALE_SOURCE,
                "live per-block binding identity changed",
            )
        return self

    def open_fixed_membership_lease(
        self, *, borrow_purpose_hash: str
    ) -> V1SignedWorkPhysicalAlternateStructuralCUDAFixedMembershipLease:
        self.assert_live()
        return self.structural_provider.open_fixed_membership_lease(
            borrow_purpose_hash=borrow_purpose_hash
        )

    def require_canonical_full_u_certifier_authority(self) -> NoReturn:
        self.assert_live()
        _refuse(
            V1SignedWorkPhysicalAlternateDomainRefusalCode.AUTHORITY_VIOLATION,
            "outside-pair execution has not joined this per-block source binding",
        )

    def evict(self) -> None:
        self.assert_live()
        self.structural_provider = None
        self.d_fixed_support = None
        self.residual_support = None
        self._evicted = True

    def __reduce__(self) -> NoReturn:
        raise TypeError("live physical-alternate block bindings are nonserializable")


def bind_v1_signed_work_physical_alternate_structural_domain_to_block_support(
    *,
    structural_provider: V1SignedWorkPhysicalAlternateStructuralCUDADomainProvider,
    d_fixed_support: V1SignedWorkBlockSupportSourceDeviceHandle,
    residual_support: V1SignedWorkBlockSupportSourceDeviceHandle,
) -> V1SignedWorkPhysicalAlternateBlockSupportBinding:
    """Join one D-fixed/residual pair without rebuilding structural custody."""

    if (
        type(structural_provider)
        is not V1SignedWorkPhysicalAlternateStructuralCUDADomainProvider
    ):
        raise TypeError("block support must bind to the exact structural provider")
    structural_provider.assert_live()
    receipt = _issue_v1_signed_work_physical_alternate_block_support_binding_receipt(
        structural_receipt=structural_provider.receipt,
        d_fixed_support=d_fixed_support,
        residual_support=residual_support,
        resident_device_identity=structural_provider.receipt.resident_device_identity,
    )
    return V1SignedWorkPhysicalAlternateBlockSupportBinding(
        receipt=receipt,
        structural_provider=structural_provider,
        d_fixed_support=d_fixed_support,
        residual_support=residual_support,
        _issuer_token=_BLOCK_SUPPORT_BINDING_TOKEN,
    )


def require_v1_signed_work_physical_alternate_domain_full_u_authority(
    provider: (
        V1SignedWorkPhysicalAlternateDomainProvider
        | V1SignedWorkPhysicalAlternateStructuralCUDADomainProvider
        | V1SignedWorkPhysicalAlternateBlockSupportBinding
    ),
) -> NoReturn:
    """Reject this within-shell provider at the canonical full-U boundary."""

    if type(provider) not in {
        V1SignedWorkPhysicalAlternateDomainProvider,
        V1SignedWorkPhysicalAlternateStructuralCUDADomainProvider,
        V1SignedWorkPhysicalAlternateBlockSupportBinding,
    }:
        raise TypeError("canonical authority check requires the exact provider")
    provider.require_canonical_full_u_certifier_authority()


__all__ = [
    "PHYSICAL_ALTERNATE_DOMAIN_AUTHORITY",
    "PHYSICAL_ALTERNATE_DOMAIN_RULE",
    "PHYSICAL_ALTERNATE_FIXED_DERIVATION_RULE",
    "PHYSICAL_ALTERNATE_BLOCK_SUPPORT_BINDING_SCHEMA",
    "PHYSICAL_ALTERNATE_STRUCTURAL_CUDA_DOMAIN_SCHEMA",
    "V1SignedWorkPhysicalAlternateActionReceipt",
    "V1SignedWorkPhysicalAlternateCellCrosswalkReceipt",
    "V1SignedWorkPhysicalAlternateBlockSupportBinding",
    "V1SignedWorkPhysicalAlternateBlockSupportBindingReceipt",
    "V1SignedWorkPhysicalAlternateStructuralCUDADomainProvider",
    "V1SignedWorkPhysicalAlternateStructuralCUDADomainReceipt",
    "V1SignedWorkPhysicalAlternateStructuralCUDAFixedMembershipLease",
    "V1SignedWorkPhysicalAlternateCUDAPackedActionLease",
    "V1SignedWorkPhysicalAlternateDomainDesign",
    "V1SignedWorkPhysicalAlternateDomainExecutionMode",
    "V1SignedWorkPhysicalAlternateDomainHBMForecast",
    "V1SignedWorkPhysicalAlternateDomainProvider",
    "V1SignedWorkPhysicalAlternateDomainReceipt",
    "V1SignedWorkPhysicalAlternateDomainRefusal",
    "V1SignedWorkPhysicalAlternateDomainRefusalCode",
    "V1SignedWorkPhysicalAlternateMembershipOwnerRequirement",
    "bind_v1_signed_work_physical_alternate_structural_domain_to_block_support",
    "build_v1_signed_work_physical_alternate_structural_domain_cuda",
    "build_v1_signed_work_physical_alternate_domain_reference",
    "describe_v1_signed_work_physical_alternate_membership_owner_requirement",
    "estimate_v1_signed_work_physical_alternate_domain_hbm",
    "require_v1_signed_work_physical_alternate_domain_full_u_authority",
]
