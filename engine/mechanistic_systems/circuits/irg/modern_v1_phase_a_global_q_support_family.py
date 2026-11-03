"""Source-bound base-direction support family for the global true-Q screen.

The registered joint literal-J family already freezes a minimal set of valid
support-plane anchors before route-scale outcomes exist: one outside vector
per physical action followed by the shared shell basis.  This module binds
those anchors to their population-Loewner upper endpoint ``Q_U(v_d)`` and the
canonical right-hand side ``tau_Q + Q_U(v_d)``.  Routewise coefficients are
*not* retained here; the population ``Mv`` stream supplies them later and the
global scanner applies the structural factor two.

The result is a compact source/custody joint only.  It grants no optimizer,
D-stability, C/K/O, commit, runtime, terminal, or Pareto authority.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from fractions import Fraction
import hashlib
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_literal_j_d_fixed_residual_population_projection import (
    V1LiteralJDFixedResidualPopulationProjectionRegistration,
    V1LiteralJJointAxisResidualPopulationMeasureRegistration,
    V1LiteralJJointAxisResidualPopulationMeasureRegistrationReceipt,
)
from .modern_v1_literal_j_joint_affine_shell_family_cuda import (
    V1LiteralJJointAffineShellPreoutcomeRegistration,
)
from .modern_v1_literal_j_joint_affine_shell_loewner_stream_bridge import (
    V1LiteralJJointAffineShellD1Pass1LoewnerReceipt,
    V1LiteralJJointAffineShellPass1LoewnerReceipt,
    V1LiteralJJointAffineShellLoewnerStreamResult,
)
from .modern_v1_relaxed_kkt import V1RelaxedKKTProblem

try:  # pragma: no cover - runtime-image dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


PHASE_A_GLOBAL_Q_SUPPORT_FAMILY_SCHEMA: Final = (
    "partition_free_v1_phase_a_global_true_q_base_direction_support_family_v1"
)
PHASE_A_GLOBAL_Q_SUPPORT_FAMILY_ALGORITHM: Final = (
    "preoutcome_joint_base_directions_with_population_loewner_q_upper_and_"
    "factor_two_routewise_mjv_support_rows_v1"
)
PHASE_A_GLOBAL_Q_SUPPORT_FAMILY_AUTHORITY: Final = (
    "compact_true_q_support_anchor_rhs_custody_only_no_solver_D_C_K_O_commit_"
    "runtime_terminal_or_pareto_authority_v1"
)
PHASE_A_GLOBAL_Q_D1_SUPPORT_FAMILY_SCHEMA: Final = (
    "partition_free_v1_phase_a_global_true_q_full_population_D1_base_"
    "direction_support_family_v1"
)
PHASE_A_GLOBAL_Q_D1_SUPPORT_FAMILY_ALGORITHM: Final = (
    "preoutcome_joint_base_directions_with_action_complete_D1_residual_"
    "population_measure_loewner_q_upper_and_factor_two_routewise_mjv_"
    "support_rows_v1"
)
PHASE_A_GLOBAL_Q_D1_SUPPORT_FAMILY_AUTHORITY: Final = (
    "measure_bound_action_complete_D1_residual_true_q_support_anchor_rhs_"
    "custody_only_no_solver_D_C_K_O_commit_runtime_terminal_or_pareto_"
    "authority_v1"
)

_PREFIX: Final = "partition-free-v1-phase-a-global-q-support-family-"
_RECEIPT_TOKEN: Final = object()
_HANDLE_TOKEN: Final = object()

__all__ = (
    "PHASE_A_GLOBAL_Q_SUPPORT_FAMILY_SCHEMA",
    "PHASE_A_GLOBAL_Q_SUPPORT_FAMILY_ALGORITHM",
    "PHASE_A_GLOBAL_Q_SUPPORT_FAMILY_AUTHORITY",
    "PHASE_A_GLOBAL_Q_D1_SUPPORT_FAMILY_SCHEMA",
    "PHASE_A_GLOBAL_Q_D1_SUPPORT_FAMILY_ALGORITHM",
    "PHASE_A_GLOBAL_Q_D1_SUPPORT_FAMILY_AUTHORITY",
    "V1PhaseAGlobalQSupportFamilyReceipt",
    "V1PhaseAGlobalQD1SupportFamilyReceipt",
    "V1PhaseAGlobalQSupportFamilyLease",
    "V1PhaseAGlobalQSupportFamilyHandle",
    "V1PhaseAGlobalQSupportFamilyResult",
    "V1PhaseAGlobalQD1SupportFamilyResult",
    "issue_v1_phase_a_global_q_base_direction_support_family",
    "issue_v1_phase_a_global_q_d1_base_direction_support_family",
)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be one nonempty exact string")
    return value


def _identity_axis(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not tuple or not value:
        raise TypeError(f"{name} must be one nonempty immutable tuple")
    result = tuple(_identity(item, name=f"{name} entry") for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


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


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQSupportFamilyReceipt:
    joint_preoutcome_registration_hash: str
    joint_direction_axis_hash: str
    anchor_direction_ids: tuple[str, ...]
    population_projection_registration_hash: str
    pass1_loewner_receipt_hash: str
    relaxed_kkt_problem_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    action_count: int
    shell_size: int
    anchor_count: int
    q_budget: float
    q_upper_digest: str
    support_rhs_upper_digest: str
    measurement_device_identity: str
    compact_d2h_scalar_count: int
    production_source_authority: bool
    private_cpu_reference: bool
    anchors_frozen_before_routewise_mjv_outcomes: bool = True
    base_direction_family_complete: bool = True
    outside_directions_precede_shared_shell_basis: bool = True
    q_upper_from_population_loewner_diagonals: bool = True
    redundant_shell_q_upper_intersected: bool = True
    support_plane_factor_two_required: bool = True
    routewise_coefficient_payload_present: bool = False
    support_plane_scan_authority: bool = False
    solver_authority: bool = False
    d_stability_authority: bool = False
    c_k_o_commit_authority: bool = False
    runtime_terminal_or_pareto_authority: bool = False
    algorithm: str = PHASE_A_GLOBAL_Q_SUPPORT_FAMILY_ALGORITHM
    authority_scope: str = PHASE_A_GLOBAL_Q_SUPPORT_FAMILY_AUTHORITY
    schema: str = PHASE_A_GLOBAL_Q_SUPPORT_FAMILY_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _RECEIPT_TOKEN:
            raise PermissionError("global-Q support-family receipts require their issuer")
        self._validate()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _validate(self) -> None:
        for name in (
            "joint_preoutcome_registration_hash",
            "joint_direction_axis_hash",
            "population_projection_registration_hash",
            "pass1_loewner_receipt_hash",
            "relaxed_kkt_problem_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "q_upper_digest",
            "support_rhs_upper_digest",
            "measurement_device_identity",
        ):
            _identity(getattr(self, name), name=name)
        anchors = _identity_axis(self.anchor_direction_ids, name="anchor_direction_ids")
        for name in ("route_count", "action_count", "shell_size", "anchor_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if (
            self.anchor_count != self.action_count + self.shell_size
            or self.anchor_count != len(anchors)
            or self.compact_d2h_scalar_count != 2 * self.anchor_count
        ):
            raise ValueError("support-family action, shell, anchor, or transfer counts diverged")
        _finite_nonnegative(self.q_budget, name="q_budget")
        positive = (
            self.anchors_frozen_before_routewise_mjv_outcomes,
            self.base_direction_family_complete,
            self.outside_directions_precede_shared_shell_basis,
            self.q_upper_from_population_loewner_diagonals,
            self.redundant_shell_q_upper_intersected,
            self.support_plane_factor_two_required,
        )
        closed = (
            self.routewise_coefficient_payload_present,
            self.support_plane_scan_authority,
            self.solver_authority,
            self.d_stability_authority,
            self.c_k_o_commit_authority,
            self.runtime_terminal_or_pareto_authority,
        )
        if (
            any(type(value) is not bool or value is not True for value in positive)
            or any(type(value) is not bool or value is not False for value in closed)
            or type(self.production_source_authority) is not bool
            or type(self.private_cpu_reference) is not bool
            or self.production_source_authority == self.private_cpu_reference
            or (self.production_source_authority and not self.measurement_device_identity.startswith("cuda:"))
            or (self.private_cpu_reference and self.measurement_device_identity != "cpu")
            or self.algorithm != PHASE_A_GLOBAL_Q_SUPPORT_FAMILY_ALGORITHM
            or self.authority_scope != PHASE_A_GLOBAL_Q_SUPPORT_FAMILY_AUTHORITY
            or self.schema != PHASE_A_GLOBAL_Q_SUPPORT_FAMILY_SCHEMA
        ):
            raise ValueError("support-family receipt crossed its compact source boundary")

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
            raise ValueError("global-Q support-family receipt no longer replays")
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


def _fraction_pair(value: object, *, name: str) -> tuple[int, int]:
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(type(item) is not int for item in value)
        or value[0] < 0
        or value[1] <= 0
    ):
        raise TypeError(f"{name} must be one exact nonnegative rational pair")
    reduced = Fraction(value[0], value[1])
    return (reduced.numerator, reduced.denominator)


def _nested_identity_pair(
    value: object,
    *,
    name: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(type(axis) is not tuple or not axis for axis in value)
    ):
        raise TypeError(f"{name} must be one nonempty target/complement identity pair")
    return tuple(
        tuple(_identity(item, name=f"{name} entry") for item in axis)
        for axis in value
    )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQD1SupportFamilyReceipt:
    """Support anchors for the action-complete D1/residual population measure.

    This is deliberately a sibling of the legacy projection receipt.  The
    registered population *measure* is source custody, while the independently
    produced D1/residual Loewner receipt owns the numerical projections.  No
    field calls the measure a projection or lets either representation enter
    the other's scan path.
    """

    population_measure_registration_hash: str
    population_measure_receipt_hash: str
    resident_population_registration_hash: str
    population_measure_design_hash: str
    population_measure_weight_receipt_hash: str
    joint_preoutcome_registration_hash: str
    joint_family_receipt_hash: str
    joint_postsource_custody_axis_hash: str
    population_measure_action_axis_preoutcome_receipt_hash: str
    action_axis_preoutcome_receipt_hash: str
    action_axis_source_receipt_hash: str
    physical_action_axis_hash: str
    physical_action_identity_hashes: tuple[str, ...]
    physical_action_hashes: tuple[str, ...]
    packed_action_hashes: tuple[str, ...]
    joint_direction_axis_hash: str
    anchor_direction_ids: tuple[str, ...]
    atomic_source_receipt_hashes: tuple[str, ...]
    atomic_preoutcome_receipt_hashes: tuple[str, ...]
    d1_stream_receipt_hash: str
    d1_q_receipt_hash: str
    d1_compact_shell_m_receipt_hash: str
    d1_source_axis_hash: str
    d1_literal_j_shard_receipt_hashes: tuple[str, ...]
    d1_literal_j_content_sha256s: tuple[str, ...]
    residual_stream_registration_hash: str
    residual_q_receipt_hash: str
    residual_compact_shell_m_receipt_hash: str
    residual_source_proof_closure_hash_axes: tuple[tuple[str, ...], tuple[str, ...]]
    residual_source_shard_receipt_hash_axes: tuple[tuple[str, ...], tuple[str, ...]]
    residual_source_content_sha256_axes: tuple[tuple[str, ...], tuple[str, ...]]
    pass1_loewner_receipt_hash: str
    relaxed_kkt_problem_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    population_count: int
    d1_count: int
    d1_mass_fraction: tuple[int, int]
    residual_population_counts: tuple[int, int]
    residual_observed_counts: tuple[int, int]
    residual_mass_fractions: tuple[tuple[int, int], tuple[int, int]]
    action_count: int
    shell_size: int
    anchor_count: int
    q_budget: float
    q_upper_digest: str
    support_rhs_upper_digest: str
    measurement_device_identity: str
    compact_d2h_scalar_count: int
    production_source_authority: bool
    private_cpu_reference: bool
    measure_is_source_custody_not_projection: bool = True
    independent_d1_residual_projection_bound: bool = True
    action_complete_source_binding_present: bool = True
    private_atomic_reference_nonauthoritative: bool = False
    anchors_frozen_before_routewise_mjv_outcomes: bool = True
    base_direction_family_complete: bool = True
    outside_directions_precede_shared_shell_basis: bool = True
    q_upper_from_population_loewner_diagonals: bool = True
    redundant_shell_q_upper_intersected: bool = True
    support_plane_factor_two_required: bool = True
    routewise_coefficient_payload_present: bool = False
    support_plane_scan_authority: bool = False
    solver_authority: bool = False
    d_stability_authority: bool = False
    c_k_o_commit_authority: bool = False
    runtime_terminal_or_pareto_authority: bool = False
    algorithm: str = PHASE_A_GLOBAL_Q_D1_SUPPORT_FAMILY_ALGORITHM
    authority_scope: str = PHASE_A_GLOBAL_Q_D1_SUPPORT_FAMILY_AUTHORITY
    schema: str = PHASE_A_GLOBAL_Q_D1_SUPPORT_FAMILY_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _RECEIPT_TOKEN:
            raise PermissionError("D1 global-Q support-family receipts require their issuer")
        self._validate()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _validate(self) -> None:
        for name in (
            "population_measure_registration_hash",
            "population_measure_receipt_hash",
            "resident_population_registration_hash",
            "population_measure_design_hash",
            "population_measure_weight_receipt_hash",
            "joint_preoutcome_registration_hash",
            "joint_family_receipt_hash",
            "joint_postsource_custody_axis_hash",
            "population_measure_action_axis_preoutcome_receipt_hash",
            "action_axis_preoutcome_receipt_hash",
            "action_axis_source_receipt_hash",
            "physical_action_axis_hash",
            "joint_direction_axis_hash",
            "d1_stream_receipt_hash",
            "d1_q_receipt_hash",
            "d1_compact_shell_m_receipt_hash",
            "d1_source_axis_hash",
            "residual_stream_registration_hash",
            "residual_q_receipt_hash",
            "residual_compact_shell_m_receipt_hash",
            "pass1_loewner_receipt_hash",
            "relaxed_kkt_problem_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "q_upper_digest",
            "support_rhs_upper_digest",
            "measurement_device_identity",
        ):
            _identity(getattr(self, name), name=name)
        action_ids = _identity_axis(
            self.physical_action_identity_hashes,
            name="physical_action_identity_hashes",
        )
        action_hashes = _identity_axis(
            self.physical_action_hashes,
            name="physical_action_hashes",
        )
        packed_hashes = _identity_axis(
            self.packed_action_hashes,
            name="packed_action_hashes",
        )
        anchors = _identity_axis(self.anchor_direction_ids, name="anchor_direction_ids")
        atomic = _identity_axis(
            self.atomic_source_receipt_hashes,
            name="atomic_source_receipt_hashes",
        )
        atomic_pre = _identity_axis(
            self.atomic_preoutcome_receipt_hashes,
            name="atomic_preoutcome_receipt_hashes",
        )
        d1_shards = _identity_axis(
            self.d1_literal_j_shard_receipt_hashes,
            name="d1_literal_j_shard_receipt_hashes",
        )
        d1_contents = _identity_axis(
            self.d1_literal_j_content_sha256s,
            name="d1_literal_j_content_sha256s",
        )
        proof_axes = _nested_identity_pair(
            self.residual_source_proof_closure_hash_axes,
            name="residual_source_proof_closure_hash_axes",
        )
        shard_axes = _nested_identity_pair(
            self.residual_source_shard_receipt_hash_axes,
            name="residual_source_shard_receipt_hash_axes",
        )
        content_axes = _nested_identity_pair(
            self.residual_source_content_sha256_axes,
            name="residual_source_content_sha256_axes",
        )
        for name in (
            "route_count",
            "population_count",
            "d1_count",
            "action_count",
            "shell_size",
            "anchor_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if type(self.proposal_generation) is not int or self.proposal_generation < 0:
            raise ValueError("proposal_generation must be a nonnegative exact integer")
        if (
            type(self.residual_population_counts) is not tuple
            or type(self.residual_observed_counts) is not tuple
            or len(self.residual_population_counts) != 2
            or len(self.residual_observed_counts) != 2
            or any(
                type(value) is not int or value <= 0
                for value in self.residual_population_counts
            )
            or any(
                type(value) is not int or value <= 0
                for value in self.residual_observed_counts
            )
            or any(
                self.residual_observed_counts[index]
                > self.residual_population_counts[index]
                for index in (0, 1)
            )
        ):
            raise ValueError("D1 support-family residual population counts are invalid")
        d1_mass = Fraction(
            *_fraction_pair(self.d1_mass_fraction, name="d1_mass_fraction")
        )
        residual_masses = tuple(
            Fraction(*_fraction_pair(value, name="residual_mass_fraction"))
            for value in self.residual_mass_fractions
        )
        if (
            len(residual_masses) != 2
            or self.d1_count + sum(self.residual_population_counts)
            != self.population_count
            or d1_mass != Fraction(self.d1_count, self.population_count)
            or any(
                residual_masses[index]
                != Fraction(self.residual_population_counts[index], self.population_count)
                for index in (0, 1)
            )
            or d1_mass + residual_masses[0] + residual_masses[1] != 1
            or self.anchor_count != self.action_count + self.shell_size
            or self.anchor_count != len(anchors)
            or any(
                len(axis) != self.action_count
                for axis in (action_ids, action_hashes, packed_hashes, atomic, atomic_pre)
            )
            or len(d1_shards) != len(d1_contents)
            or any(
                not (
                    len(proof_axes[index])
                    == len(shard_axes[index])
                    == len(content_axes[index])
                )
                for index in (0, 1)
            )
            or self.compact_d2h_scalar_count != 2 * self.anchor_count
        ):
            raise ValueError("D1 support-family action, source, or population axes diverged")
        _finite_nonnegative(self.q_budget, name="q_budget")
        positive = (
            self.measure_is_source_custody_not_projection,
            self.independent_d1_residual_projection_bound,
            self.anchors_frozen_before_routewise_mjv_outcomes,
            self.base_direction_family_complete,
            self.outside_directions_precede_shared_shell_basis,
            self.q_upper_from_population_loewner_diagonals,
            self.redundant_shell_q_upper_intersected,
            self.support_plane_factor_two_required,
        )
        closed = (
            self.routewise_coefficient_payload_present,
            self.support_plane_scan_authority,
            self.solver_authority,
            self.d_stability_authority,
            self.c_k_o_commit_authority,
            self.runtime_terminal_or_pareto_authority,
        )
        if (
            any(type(value) is not bool or value is not True for value in positive)
            or any(type(value) is not bool or value is not False for value in closed)
            or type(self.production_source_authority) is not bool
            or type(self.private_cpu_reference) is not bool
            or self.production_source_authority == self.private_cpu_reference
            or self.action_complete_source_binding_present
            != self.production_source_authority
            or self.private_atomic_reference_nonauthoritative
            != self.private_cpu_reference
            or (
                self.production_source_authority
                and not self.measurement_device_identity.startswith("cuda:")
            )
            or (
                self.private_cpu_reference
                and self.measurement_device_identity != "cpu"
            )
            or self.algorithm != PHASE_A_GLOBAL_Q_D1_SUPPORT_FAMILY_ALGORITHM
            or self.authority_scope != PHASE_A_GLOBAL_Q_D1_SUPPORT_FAMILY_AUTHORITY
            or self.schema != PHASE_A_GLOBAL_Q_D1_SUPPORT_FAMILY_SCHEMA
        ):
            raise ValueError("D1 support-family receipt crossed its source boundary")

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                item.name: getattr(self, item.name)
                for item in fields(self)
                if not item.name.startswith("_")
            }
        )

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="d1-receipt")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self._validate()
        observed = self._compute_hash()
        if observed != self._sealed_hash or (
            expected_hash is not None and observed != expected_hash
        ):
            raise ValueError("D1 global-Q support-family receipt no longer replays")
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class V1PhaseAGlobalQSupportFamilyLease:
    __slots__ = ("_owner", "_token", "_closed", "q_upper", "rhs_upper")

    def __init__(
        self,
        owner: "V1PhaseAGlobalQSupportFamilyHandle",
        token: object,
        q_upper: Any,
        rhs_upper: Any,
    ) -> None:
        self._owner = owner
        self._token = token
        self._closed = False
        self.q_upper = q_upper
        self.rhs_upper = rhs_upper

    def __enter__(self) -> "V1PhaseAGlobalQSupportFamilyLease":
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
        raise TypeError("global-Q support-family leases are nonserializable")


class V1PhaseAGlobalQSupportFamilyHandle:
    __slots__ = (
        "receipt",
        "_q_upper",
        "_rhs_upper",
        "_guards",
        "_invariants",
        "_tokens",
        "_evicted",
    )

    def __init__(
        self,
        *,
        receipt: (
            V1PhaseAGlobalQSupportFamilyReceipt
            | V1PhaseAGlobalQD1SupportFamilyReceipt
        ),
        q_upper: Any,
        rhs_upper: Any,
        _issuance_token: object | None = None,
    ) -> None:
        if _issuance_token is not _HANDLE_TOKEN:
            raise PermissionError("global-Q support-family handles require their issuer")
        if type(receipt) not in (
            V1PhaseAGlobalQSupportFamilyReceipt,
            V1PhaseAGlobalQD1SupportFamilyReceipt,
        ):
            raise TypeError("support-family handle requires its exact receipt")
        if (
            torch is None
            or type(q_upper) is not torch.Tensor
            or type(rhs_upper) is not torch.Tensor
        ):
            raise TypeError("support-family numerics require exact torch tensors")
        expected = (receipt.anchor_count,)
        if (
            tuple(q_upper.shape) != expected
            or tuple(rhs_upper.shape) != expected
            or q_upper.dtype is not torch.float64
            or rhs_upper.dtype is not torch.float64
            or q_upper.device != rhs_upper.device
            or not bool(torch.isfinite(q_upper).all().item())
            or not bool(torch.isfinite(rhs_upper).all().item())
            or bool((q_upper < 0.0).any().item())
            or bool((rhs_upper < 0.0).any().item())
        ):
            raise ValueError("support-family compact numerics do not match their receipt")
        self.receipt = receipt
        self._q_upper = q_upper
        self._rhs_upper = rhs_upper
        self._guards = (q_upper.detach().clone(), rhs_upper.detach().clone())
        self._invariants = (_tensor_invariant(q_upper), _tensor_invariant(rhs_upper))
        self._tokens: set[object] = set()
        self._evicted = False
        self.assert_live()

    @property
    def is_evicted(self) -> bool:
        return self._evicted

    def assert_live(self) -> None:
        if self._evicted:
            raise RuntimeError("global-Q support-family handle was evicted")
        self.receipt.assert_integrity()
        values = (self._q_upper, self._rhs_upper)
        if any(
            _tensor_invariant(value) != invariant
            or not bool(torch.equal(value, guard))
            for value, invariant, guard in zip(
                values, self._invariants, self._guards, strict=True
            )
        ):
            raise RuntimeError("global-Q support-family compact numerics mutated")

    def borrow(self) -> V1PhaseAGlobalQSupportFamilyLease:
        self.assert_live()
        token = object()
        self._tokens.add(token)
        return V1PhaseAGlobalQSupportFamilyLease(
            self, token, self._q_upper, self._rhs_upper
        )

    def _close(self, token: object) -> None:
        if token not in self._tokens:
            raise RuntimeError("global-Q support-family lease token is stale")
        try:
            self.assert_live()
        finally:
            self._tokens.remove(token)

    def evict(self) -> None:
        if self._evicted:
            raise RuntimeError("global-Q support-family handle was already evicted")
        if self._tokens:
            raise RuntimeError("cannot evict support-family numerics under an active lease")
        self.assert_live()
        self._q_upper = None
        self._rhs_upper = None
        self._guards = ()
        self._invariants = ()
        self._evicted = True

    def __reduce__(self) -> NoReturn:
        raise TypeError("global-Q support-family handles are nonserializable")


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQSupportFamilyResult:
    receipt: V1PhaseAGlobalQSupportFamilyReceipt
    device_handle: V1PhaseAGlobalQSupportFamilyHandle = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            type(self.receipt) is not V1PhaseAGlobalQSupportFamilyReceipt
            or type(self.device_handle) is not V1PhaseAGlobalQSupportFamilyHandle
        ):
            raise TypeError("support-family result requires exact components")
        self.device_handle.assert_live()
        if self.device_handle.receipt.receipt_hash != self.receipt.receipt_hash:
            raise ValueError("support-family receipt and live numerics diverged")


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQD1SupportFamilyResult:
    receipt: V1PhaseAGlobalQD1SupportFamilyReceipt
    device_handle: V1PhaseAGlobalQSupportFamilyHandle = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            type(self.receipt) is not V1PhaseAGlobalQD1SupportFamilyReceipt
            or type(self.device_handle) is not V1PhaseAGlobalQSupportFamilyHandle
        ):
            raise TypeError("D1 support-family result requires exact components")
        self.device_handle.assert_live()
        if self.device_handle.receipt.receipt_hash != self.receipt.receipt_hash:
            raise ValueError("D1 support-family receipt and live numerics diverged")


def issue_v1_phase_a_global_q_base_direction_support_family(
    *,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    pass1_loewner_result: V1LiteralJJointAffineShellLoewnerStreamResult,
    relaxed_kkt_problem: V1RelaxedKKTProblem,
    allow_private_cpu_reference_for_tests: bool = False,
) -> V1PhaseAGlobalQSupportFamilyResult:
    """Bind the pre-outcome basis anchors to compact true-Q upper endpoints."""

    if torch is None:
        raise RuntimeError("global-Q support-family issuance requires PyTorch")
    if (
        type(joint_preoutcome_registration)
        is not V1LiteralJJointAffineShellPreoutcomeRegistration
        or type(population_registration)
        is not V1LiteralJDFixedResidualPopulationProjectionRegistration
        or type(pass1_loewner_result) is not V1LiteralJJointAffineShellLoewnerStreamResult
        or type(relaxed_kkt_problem) is not V1RelaxedKKTProblem
        or type(allow_private_cpu_reference_for_tests) is not bool
    ):
        raise TypeError("support-family issuance requires exact source owners")
    prereg = joint_preoutcome_registration
    prereg.assert_integrity()
    population_registration.assert_integrity()
    pass1_loewner_result.device_handle.assert_live()
    loewner = pass1_loewner_result.receipt
    if type(loewner) is not V1LiteralJJointAffineShellPass1LoewnerReceipt:
        raise ValueError(
            "support-family issuance requires the pass-1 Loewner receipt while "
            "routewise Mv remains unopened"
        )
    loewner.assert_integrity()
    production = bool(loewner.population_loewner_authority)
    if not production and not allow_private_cpu_reference_for_tests:
        raise PermissionError("private CPU Loewner geometry cannot enter production issuance")
    if (
        loewner.joint_preoutcome_registration_hash != prereg.registration_hash
        or loewner.population_projection_registration_hash
        != population_registration.registration_hash
        or loewner.joint_direction_axis_hash != prereg.joint_direction_axis_hash
        or loewner.joint_direction_ids != prereg.joint_direction_ids
        or loewner.route_axis_hash != prereg.route_axis_hash
        or loewner.route_id_axis_hash != prereg.route_id_axis_hash
        or loewner.route_count != prereg.route_count
        or loewner.action_count != prereg.action_count
        or loewner.shell_size != prereg.shell_size
        or loewner.model_identity_hash != prereg.model_identity_hash
        or loewner.chart_identity_hash != prereg.chart_identity_hash
        or loewner.current_mask_hash != prereg.current_mask_hash
        or loewner.pass2_unopened_at_issuance is not True
        or loewner.pass2_unopened_after_issuance is not True
        or relaxed_kkt_problem.route_axis_hash != prereg.route_axis_hash
        or relaxed_kkt_problem.route_count != prereg.route_count
    ):
        raise ValueError("support-family sources do not share one chart, route, and population axis")
    with pass1_loewner_result.device_handle.borrow() as lease:
        upper = lease.loewner_upper
        expected_shape = (
            prereg.action_count,
            prereg.shell_size + 1,
            prereg.shell_size + 1,
        )
        if tuple(upper.shape) != expected_shape or upper.dtype is not torch.float64:
            raise ValueError("pass-1 Loewner upper tensor has the wrong compact shape")
        action_q = torch.diagonal(upper[:, -1:, -1:], dim1=-2, dim2=-1).reshape(-1)
        shell_diagonals = torch.diagonal(
            upper[:, : prereg.shell_size, : prereg.shell_size],
            dim1=-2,
            dim2=-1,
        )
        shell_q = torch.min(shell_diagonals, dim=0).values
        q_upper = torch.cat((action_q, shell_q), dim=0).clone().contiguous()
    if (
        not bool(torch.isfinite(q_upper).all().item())
        or bool((q_upper < 0.0).any().item())
    ):
        raise ValueError("population-Loewner diagonal produced an invalid Q upper endpoint")
    q_budget = _finite_nonnegative(
        relaxed_kkt_problem.quadratic_budget,
        name="quadratic_budget",
    )
    rhs_sum = q_upper + q_budget
    rhs_upper = torch.nextafter(
        rhs_sum,
        torch.full_like(rhs_sum, math.inf),
    ).contiguous()
    if not bool(torch.isfinite(rhs_upper).all().item()):
        raise ValueError("support-family Q budget addition overflowed binary64")
    device_identity = str(q_upper.device)
    if production and not device_identity.startswith("cuda:"):
        raise PermissionError("production support-family issuance requires CUDA source custody")
    if not production and device_identity != "cpu":
        raise ValueError("private reference support-family issuance must remain on CPU")
    receipt = V1PhaseAGlobalQSupportFamilyReceipt(
        joint_preoutcome_registration_hash=prereg.registration_hash,
        joint_direction_axis_hash=prereg.joint_direction_axis_hash,
        anchor_direction_ids=prereg.joint_direction_ids,
        population_projection_registration_hash=population_registration.registration_hash,
        pass1_loewner_receipt_hash=loewner.receipt_hash,
        relaxed_kkt_problem_hash=relaxed_kkt_problem.problem_hash,
        model_identity_hash=prereg.model_identity_hash,
        chart_identity_hash=prereg.chart_identity_hash,
        current_mask_hash=prereg.current_mask_hash,
        route_axis_hash=prereg.route_axis_hash,
        route_id_axis_hash=prereg.route_id_axis_hash,
        route_count=prereg.route_count,
        action_count=prereg.action_count,
        shell_size=prereg.shell_size,
        anchor_count=prereg.direction_count,
        q_budget=q_budget,
        q_upper_digest=_compact_tensor_digest(q_upper, role="q-upper"),
        support_rhs_upper_digest=_compact_tensor_digest(
            rhs_upper, role="rhs-upper"
        ),
        measurement_device_identity=device_identity,
        compact_d2h_scalar_count=2 * prereg.direction_count,
        production_source_authority=production,
        private_cpu_reference=not production,
        _issuance_token=_RECEIPT_TOKEN,
    )
    handle = V1PhaseAGlobalQSupportFamilyHandle(
        receipt=receipt,
        q_upper=q_upper,
        rhs_upper=rhs_upper,
        _issuance_token=_HANDLE_TOKEN,
    )
    return V1PhaseAGlobalQSupportFamilyResult(
        receipt=receipt,
        device_handle=handle,
    )


def issue_v1_phase_a_global_q_d1_base_direction_support_family(
    *,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    population_measure_registration: V1LiteralJJointAxisResidualPopulationMeasureRegistration,
    pass1_loewner_result: V1LiteralJJointAffineShellLoewnerStreamResult,
    relaxed_kkt_problem: V1RelaxedKKTProblem,
    allow_private_cpu_reference_for_tests: bool = False,
) -> V1PhaseAGlobalQD1SupportFamilyResult:
    """Bind full-population D1/residual Loewner anchors to their measure.

    The population registration supplies outcome-independent measure and action-axis
    custody only.  Numerical Q endpoints must descend from the exact D1 pass-1
    Loewner receipt; the measure is never interpreted as a projection.
    """

    if torch is None:
        raise RuntimeError("D1 global-Q support-family issuance requires PyTorch")
    if (
        type(joint_preoutcome_registration)
        is not V1LiteralJJointAffineShellPreoutcomeRegistration
        or type(population_measure_registration)
        is not V1LiteralJJointAxisResidualPopulationMeasureRegistration
        or type(pass1_loewner_result)
        is not V1LiteralJJointAffineShellLoewnerStreamResult
        or type(relaxed_kkt_problem) is not V1RelaxedKKTProblem
        or type(allow_private_cpu_reference_for_tests) is not bool
    ):
        raise TypeError("D1 support-family issuance requires exact source owners")
    prereg = joint_preoutcome_registration
    prereg.assert_integrity()
    population_hash = population_measure_registration.assert_live()
    population = population_measure_registration.receipt
    if (
        type(population)
        is not V1LiteralJJointAxisResidualPopulationMeasureRegistrationReceipt
    ):
        raise TypeError("D1 support-family population measure receipt must be exact")
    population.assert_integrity()
    pass1_loewner_result.device_handle.assert_live()
    loewner = pass1_loewner_result.receipt
    if type(loewner) is not V1LiteralJJointAffineShellD1Pass1LoewnerReceipt:
        raise ValueError(
            "D1 support-family issuance requires the exact D1/residual pass-1 "
            "Loewner receipt"
        )
    loewner.assert_integrity()
    production = bool(loewner.population_loewner_authority)
    if (
        production != loewner.production_cuda_execution
        or production != loewner.kkt_geometry_uncertainty_input_authority
        or loewner.private_cpu_reference_execution == production
    ):
        raise ValueError("D1 pass-1 Loewner authority mode is internally inconsistent")
    if not production and not allow_private_cpu_reference_for_tests:
        raise PermissionError("private CPU D1 Loewner geometry cannot enter production issuance")

    common = (
        (population_hash, population.receipt_hash),
        (population.joint_preoutcome_registration_hash, prereg.registration_hash),
        (loewner.joint_preoutcome_registration_hash, prereg.registration_hash),
        (population.joint_direction_axis_hash, prereg.joint_direction_axis_hash),
        (loewner.joint_direction_axis_hash, prereg.joint_direction_axis_hash),
        (population.joint_direction_ids, prereg.joint_direction_ids),
        (loewner.joint_direction_ids, prereg.joint_direction_ids),
        (population.physical_action_axis_hash, prereg.physical_action_axis_hash),
        (
            population.physical_action_identity_hashes,
            prereg.physical_action_identity_hashes,
        ),
        (population.physical_action_hashes, prereg.physical_action_hashes),
        (population.packed_action_hashes, prereg.packed_action_hashes),
        (loewner.model_identity_hash, prereg.model_identity_hash),
        (population.model_identity_hash, prereg.model_identity_hash),
        (loewner.chart_identity_hash, prereg.chart_identity_hash),
        (population.chart_identity_hash, prereg.chart_identity_hash),
        (loewner.current_mask_hash, prereg.current_mask_hash),
        (population.current_mask_hash, prereg.current_mask_hash),
        (loewner.proposal_generation, prereg.proposal_generation),
        (population.proposal_generation, prereg.proposal_generation),
        (loewner.route_axis_hash, prereg.route_axis_hash),
        (population.route_axis_hash, prereg.route_axis_hash),
        (loewner.route_id_axis_hash, prereg.route_id_axis_hash),
        (population.route_id_axis_hash, prereg.route_id_axis_hash),
        (loewner.route_count, prereg.route_count),
        (population.route_count, prereg.route_count),
        (loewner.action_count, prereg.action_count),
        (population.action_count, prereg.action_count),
        (loewner.shell_size, prereg.shell_size),
        (population.shell_size, prereg.shell_size),
        (population.direction_count, prereg.direction_count),
        (loewner.registered_d_prefix_hash, prereg.registered_d_prefix_hash),
        (loewner.registered_d_w2_prefix_hash, prereg.registered_d_w2_prefix_hash),
        (population.registered_d_prefix_hash, prereg.registered_d_prefix_hash),
        (population.registered_d_w2_prefix_hash, prereg.registered_d_w2_prefix_hash),
        (population.d1_source_axis_hash, loewner.d1_source_axis_hash),
        (population.d1_context_count, loewner.d1_count),
        (population.d1_mass_fraction, loewner.d1_mass_fraction),
        (
            population.residual_population_stratum_counts,
            loewner.residual_population_counts,
        ),
        (
            population.registered_residual_sample_stratum_counts,
            loewner.residual_observed_counts,
        ),
        (population.residual_stratum_mass_fractions, loewner.residual_mass_fractions),
        (population.population_count, loewner.population_count),
        (relaxed_kkt_problem.route_axis_hash, prereg.route_axis_hash),
        (relaxed_kkt_problem.route_count, prereg.route_count),
    )
    if any(left != right for left, right in common):
        raise ValueError(
            "D1 support-family sources do not share one action axis, chart, "
            "population measure, and numerical D1/residual graph"
        )
    if production and (
        loewner.action_axis_preoutcome_receipt_hash
        != population.action_axis_preoutcome_receipt_hash
        or loewner.action_axis_source_receipt_hash.startswith("private-reference:")
        or loewner.action_complete_source_binding_present is not True
        or loewner.atomic_source_tuple_nonauthoritative is not False
    ):
        raise ValueError(
            "production D1 support-family issuance requires one exact PRE/POST "
            "action-complete source binding"
        )
    if not production and (
        loewner.action_complete_source_binding_present is not False
        or loewner.atomic_source_tuple_nonauthoritative is not True
    ):
        raise ValueError("private D1 support-family reference widened its source authority")

    with pass1_loewner_result.device_handle.borrow() as lease:
        upper = lease.loewner_upper
        expected_shape = (
            prereg.action_count,
            prereg.shell_size + 1,
            prereg.shell_size + 1,
        )
        if tuple(upper.shape) != expected_shape or upper.dtype is not torch.float64:
            raise ValueError("D1 pass-1 Loewner upper tensor has the wrong compact shape")
        action_q = torch.diagonal(
            upper[:, -1:, -1:],
            dim1=-2,
            dim2=-1,
        ).reshape(-1)
        shell_diagonals = torch.diagonal(
            upper[:, : prereg.shell_size, : prereg.shell_size],
            dim1=-2,
            dim2=-1,
        )
        shell_q = torch.min(shell_diagonals, dim=0).values
        q_upper = torch.cat((action_q, shell_q), dim=0).clone().contiguous()
    if (
        not bool(torch.isfinite(q_upper).all().item())
        or bool((q_upper < 0.0).any().item())
    ):
        raise ValueError("D1 population-Loewner diagonal produced an invalid Q upper endpoint")
    q_budget = _finite_nonnegative(
        relaxed_kkt_problem.quadratic_budget,
        name="quadratic_budget",
    )
    rhs_upper = torch.nextafter(
        q_upper + q_budget,
        torch.full_like(q_upper, math.inf),
    ).contiguous()
    if not bool(torch.isfinite(rhs_upper).all().item()):
        raise ValueError("D1 support-family Q budget addition overflowed binary64")
    device_identity = str(q_upper.device)
    if production and not device_identity.startswith("cuda:"):
        raise PermissionError("production D1 support-family issuance requires CUDA custody")
    if not production and device_identity != "cpu":
        raise ValueError("private D1 support-family reference must remain on CPU")

    receipt = V1PhaseAGlobalQD1SupportFamilyReceipt(
        population_measure_registration_hash=population_hash,
        population_measure_receipt_hash=population.receipt_hash,
        resident_population_registration_hash=(
            population.resident_population_registration_hash
        ),
        population_measure_design_hash=population.population_measure_design_hash,
        population_measure_weight_receipt_hash=(
            population.population_measure_weight_receipt_hash
        ),
        joint_preoutcome_registration_hash=prereg.registration_hash,
        joint_family_receipt_hash=loewner.joint_family_receipt_hash,
        joint_postsource_custody_axis_hash=loewner.joint_postsource_custody_axis_hash,
        population_measure_action_axis_preoutcome_receipt_hash=(
            population.action_axis_preoutcome_receipt_hash
        ),
        action_axis_preoutcome_receipt_hash=(
            loewner.action_axis_preoutcome_receipt_hash
        ),
        action_axis_source_receipt_hash=loewner.action_axis_source_receipt_hash,
        physical_action_axis_hash=population.physical_action_axis_hash,
        physical_action_identity_hashes=population.physical_action_identity_hashes,
        physical_action_hashes=population.physical_action_hashes,
        packed_action_hashes=population.packed_action_hashes,
        joint_direction_axis_hash=prereg.joint_direction_axis_hash,
        anchor_direction_ids=prereg.joint_direction_ids,
        atomic_source_receipt_hashes=loewner.atomic_source_receipt_hashes,
        atomic_preoutcome_receipt_hashes=loewner.atomic_preoutcome_receipt_hashes,
        d1_stream_receipt_hash=loewner.d1_stream_receipt_hash,
        d1_q_receipt_hash=loewner.d1_q_receipt_hash,
        d1_compact_shell_m_receipt_hash=loewner.d1_compact_shell_m_receipt_hash,
        d1_source_axis_hash=loewner.d1_source_axis_hash,
        d1_literal_j_shard_receipt_hashes=(
            loewner.d1_literal_j_shard_receipt_hashes
        ),
        d1_literal_j_content_sha256s=loewner.d1_literal_j_content_sha256s,
        residual_stream_registration_hash=loewner.residual_stream_registration_hash,
        residual_q_receipt_hash=loewner.residual_q_receipt_hash,
        residual_compact_shell_m_receipt_hash=(
            loewner.residual_compact_shell_m_receipt_hash
        ),
        residual_source_proof_closure_hash_axes=(
            loewner.residual_source_proof_closure_hash_axes
        ),
        residual_source_shard_receipt_hash_axes=(
            loewner.residual_source_shard_receipt_hash_axes
        ),
        residual_source_content_sha256_axes=(
            loewner.residual_source_content_sha256_axes
        ),
        pass1_loewner_receipt_hash=loewner.receipt_hash,
        relaxed_kkt_problem_hash=relaxed_kkt_problem.problem_hash,
        model_identity_hash=prereg.model_identity_hash,
        chart_identity_hash=prereg.chart_identity_hash,
        current_mask_hash=prereg.current_mask_hash,
        proposal_generation=prereg.proposal_generation,
        route_axis_hash=prereg.route_axis_hash,
        route_id_axis_hash=prereg.route_id_axis_hash,
        route_count=prereg.route_count,
        registered_d_prefix_hash=prereg.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=prereg.registered_d_w2_prefix_hash,
        population_count=loewner.population_count,
        d1_count=loewner.d1_count,
        d1_mass_fraction=loewner.d1_mass_fraction,
        residual_population_counts=loewner.residual_population_counts,
        residual_observed_counts=loewner.residual_observed_counts,
        residual_mass_fractions=loewner.residual_mass_fractions,
        action_count=prereg.action_count,
        shell_size=prereg.shell_size,
        anchor_count=prereg.direction_count,
        q_budget=q_budget,
        q_upper_digest=_compact_tensor_digest(q_upper, role="d1-q-upper"),
        support_rhs_upper_digest=_compact_tensor_digest(
            rhs_upper,
            role="d1-rhs-upper",
        ),
        measurement_device_identity=device_identity,
        compact_d2h_scalar_count=2 * prereg.direction_count,
        production_source_authority=production,
        private_cpu_reference=not production,
        action_complete_source_binding_present=production,
        private_atomic_reference_nonauthoritative=not production,
        _issuance_token=_RECEIPT_TOKEN,
    )
    handle = V1PhaseAGlobalQSupportFamilyHandle(
        receipt=receipt,
        q_upper=q_upper,
        rhs_upper=rhs_upper,
        _issuance_token=_HANDLE_TOKEN,
    )
    return V1PhaseAGlobalQD1SupportFamilyResult(
        receipt=receipt,
        device_handle=handle,
    )
