"""Bounded structured literal-J directions for a frozen variable shell.

The Phase-A shell calculation needs the affine family

``(z_outside, e_b1, ..., e_bB)`` with ``B <= 8``.

Materialising that family as a ``(B+1) x routes`` tensor would duplicate the
same route-scale outside vector B times.  This module owns exactly one dense
binary64 outside vector and one bounded int64 shell-index vector.  Consumers
must gather basis columns from literal-J tiles; no basis bank is available.

This is process-local numerical custody only.  Its receipts do not certify a
population moment, FD replacement, D stability, a KKT face, or any C/K/O or
commit decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
from typing import TYPE_CHECKING, Any, Final, NoReturn, Sequence

from mechanistic_systems.contracts.content_address import content_hash

try:  # pragma: no cover - runtime-image dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from .modern_v1_literal_j_joint_population_source import (
        V1LiteralJJointPopulationSource,
    )


LITERAL_J_AFFINE_SHELL_DIRECTION_SCHEMA: Final = (
    "partition_free_v1_literal_J_affine_shell_direction_receipt_v4"
)
LITERAL_J_AFFINE_SHELL_DIRECTION_FORECAST_SCHEMA: Final = (
    "partition_free_v1_literal_J_affine_shell_direction_forecast_v1"
)
LITERAL_J_AFFINE_SHELL_DIRECTION_AUTHORITY: Final = (
    "structured_direction_custody_only_no_population_FD_solver_D_C_K_O_commit_"
    "terminal_or_pareto_authority_v1"
)
LITERAL_J_AFFINE_SHELL_DIRECTION_ALGORITHM: Final = (
    "one_dense_z_outside_plus_bounded_shell_indices_no_basis_bank_v1"
)
_PREFIX: Final = "partition-free-v1-literal-j-affine-shell-direction-"
_MAX_SHELL: Final = 8
_F64_BYTES: Final = 8
_I64_BYTES: Final = 8
_ISSUER: Final = object()


class V1LiteralJAffineShellDirectionExecutionMode(str, Enum):
    CUDA_PRODUCTION = "cuda_production"
    CPU_REFERENCE_TEST_ONLY = "cpu_reference_test_only"


class V1LiteralJAffineShellDirectionConstructionCustodyKind(str, Enum):
    """Typed provenance for the route-scale outside-vector construction.

    ``CALLER_VECTOR_COMPATIBILITY`` is deliberately noncanonical.  The two
    canonical kinds distinguish historical projection-replay construction
    from current-source joint-population construction without putting the
    latter's receipt hash in a field whose name claims projection replay.
    """

    CALLER_VECTOR_COMPATIBILITY = "caller_vector_compatibility_v1"
    LEGACY_PROJECTION_REPLAY = "legacy_projection_replay_v1"
    JOINT_POPULATION_SOURCE = "joint_population_source_v1"


class V1LiteralJAffineShellDirectionRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    INVALID_DESIGN = "invalid_design"
    STALE_IDENTITY = "stale_identity"
    SHELL_TOO_LARGE = "shell_too_large"
    DEVICE_REQUIRED = "device_required"
    NONFINITE_DIRECTION = "nonfinite_direction"
    MUTATED_CUSTODY = "mutated_custody"
    ACTIVE_LEASE = "active_lease"
    EVICTED = "evicted"
    AUTHORITY_VIOLATION = "authority_violation"
    PREOUTCOME_REGISTRATION_MISMATCH = "preoutcome_registration_mismatch"
    POSTSOURCE_CUSTODY_MISMATCH = "postsource_custody_mismatch"
    ACTION_CUSTODY_MISMATCH = "action_custody_mismatch"


class V1LiteralJAffineShellDirectionRefusal(RuntimeError):
    def __init__(
        self,
        code: V1LiteralJAffineShellDirectionRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1LiteralJAffineShellDirectionRefusalCode:
            raise TypeError("affine-shell direction refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1LiteralJAffineShellDirectionRefusalCode,
    message: str,
) -> NoReturn:
    raise V1LiteralJAffineShellDirectionRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1LiteralJAffineShellDirectionRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _optional_identity(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _identity(value, name=name)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _refuse(
            V1LiteralJAffineShellDirectionRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative integer",
        )
    return int(value)


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        _refuse(
            V1LiteralJAffineShellDirectionRefusalCode.INVALID_DESIGN,
            f"{name} must be positive",
        )
    return result


def _string_axis(values: object, *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        _refuse(
            V1LiteralJAffineShellDirectionRefusalCode.INVALID_TYPE,
            f"{name} must be one ordered string axis",
        )
    result = tuple(_identity(item, name=f"{name} item") for item in values)
    if not result or len(result) != len(set(result)):
        _refuse(
            V1LiteralJAffineShellDirectionRefusalCode.INVALID_DESIGN,
            f"{name} must be nonempty and unique",
        )
    return result


def _tensor_invariant(value: Any) -> tuple[object, ...]:
    return (
        id(value),
        int(value._version),
        int(value.data_ptr()),
        tuple(int(item) for item in value.shape),
        str(value.dtype),
        str(value.device),
    )


def _exact_cpu_reference_digest(value: Any, *, role: str) -> str:
    array = value.detach().contiguous().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(role.encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(repr(tuple(int(item) for item in array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return content_hash(
        {
            "schema": "bounded_exact_tensor_digest_v1",
            "role": role,
            "shape": tuple(int(item) for item in array.shape),
            "payload_sha256": digest.hexdigest(),
        },
        prefix=_PREFIX + "tensor-",
    )


@dataclass(frozen=True, slots=True)
class V1LiteralJAffineShellDirectionLineage:
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
    shell_route_ids: tuple[str, ...]
    literal_j_stream_source_hash: str
    same_capture_vjp_source_hash: str
    source_receipt_hash: str
    alpha_lineage_hash: str
    preoutcome_source_registration_hash: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "shell_freeze_receipt_hash",
            "literal_j_stream_source_hash",
            "same_capture_vjp_source_hash",
            "source_receipt_hash",
            "alpha_lineage_hash",
        ):
            _identity(getattr(self, name), name=name)
        object.__setattr__(
            self,
            "preoutcome_source_registration_hash",
            _optional_identity(
                self.preoutcome_source_registration_hash,
                name="preoutcome_source_registration_hash",
            ),
        )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal_generation"),
        )
        routes = _positive_int(self.route_count, name="route_count")
        object.__setattr__(self, "route_count", routes)
        if type(self.shell_indices) is not tuple or type(self.shell_route_ids) is not tuple:
            _refuse(
                V1LiteralJAffineShellDirectionRefusalCode.INVALID_TYPE,
                "shell axes must be immutable tuples",
            )
        indices = tuple(
            _nonnegative_int(item, name="shell index") for item in self.shell_indices
        )
        ids = _string_axis(self.shell_route_ids, name="shell_route_ids")
        if (
            not 1 <= len(indices) <= _MAX_SHELL
            or len(indices) != len(ids)
            or len(indices) != len(set(indices))
            or any(item >= routes for item in indices)
        ):
            _refuse(
                V1LiteralJAffineShellDirectionRefusalCode.SHELL_TOO_LARGE,
                "shell must contain one to eight unique in-axis routes",
            )
        object.__setattr__(self, "shell_indices", indices)
        object.__setattr__(self, "shell_route_ids", ids)

    @property
    def direction_ids(self) -> tuple[str, ...]:
        return ("literal-j:outside",) + tuple(
            f"literal-j:shell-basis:{route_id}" for route_id in self.shell_route_ids
        )

    @property
    def lineage_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "lineage-")


@dataclass(frozen=True, slots=True)
class V1LiteralJAffineShellDirectionMemoryForecast:
    route_count: int
    shell_size: int
    outside_float64_bytes: int
    shell_index_int64_bytes: int
    mutation_guard_bytes: int
    retained_device_bytes: int
    dense_basis_bank_bytes_avoided: int
    packed_action_h2d_bytes: int = 0
    construction_tile_logical_payload_upper_bound_bytes: int = 0
    construction_logical_payload_upper_bound_bytes: int = 0
    gpu_packed_materializer_used: bool = False
    route_scale_numpy_unpack_used: bool = False
    no_basis_bank_materialized: bool = True
    schema: str = LITERAL_J_AFFINE_SHELL_DIRECTION_FORECAST_SCHEMA

    def __post_init__(self) -> None:
        routes = _positive_int(self.route_count, name="route_count")
        shell = _positive_int(self.shell_size, name="shell_size")
        if shell > _MAX_SHELL:
            _refuse(
                V1LiteralJAffineShellDirectionRefusalCode.SHELL_TOO_LARGE,
                "forecast shell exceeds eight routes",
            )
        expected_outside = routes * _F64_BYTES
        expected_indices = shell * _I64_BYTES
        expected_guard = expected_outside + expected_indices
        expected_retained = 2 * (expected_outside + expected_indices)
        expected_avoided = shell * routes * _F64_BYTES
        h2d = _nonnegative_int(
            self.packed_action_h2d_bytes,
            name="packed_action_h2d_bytes",
        )
        tile_logical_payload = _nonnegative_int(
            self.construction_tile_logical_payload_upper_bound_bytes,
            name="construction_tile_logical_payload_upper_bound_bytes",
        )
        construction_logical_payload = _nonnegative_int(
            self.construction_logical_payload_upper_bound_bytes,
            name="construction_logical_payload_upper_bound_bytes",
        )
        if (
            self.outside_float64_bytes != expected_outside
            or self.shell_index_int64_bytes != expected_indices
            or self.mutation_guard_bytes != expected_guard
            or self.retained_device_bytes != expected_retained
            or self.dense_basis_bank_bytes_avoided != expected_avoided
            or type(self.gpu_packed_materializer_used) is not bool
            or type(self.route_scale_numpy_unpack_used) is not bool
            or (
                self.gpu_packed_materializer_used
                and (
                    self.route_scale_numpy_unpack_used
                    or h2d <= 0
                    or tile_logical_payload <= 0
                    or construction_logical_payload < expected_retained
                )
            )
            or (
                not self.gpu_packed_materializer_used
                and not self.route_scale_numpy_unpack_used
                and (
                    h2d != 0
                    or tile_logical_payload != 0
                    or construction_logical_payload != 0
                )
            )
            or (
                self.route_scale_numpy_unpack_used
                and (
                    h2d != 0
                    or tile_logical_payload != 0
                    or construction_logical_payload != 0
                )
            )
            or self.no_basis_bank_materialized is not True
            or self.schema != LITERAL_J_AFFINE_SHELL_DIRECTION_FORECAST_SCHEMA
        ):
            _refuse(
                V1LiteralJAffineShellDirectionRefusalCode.INVALID_DESIGN,
                "structured-direction memory forecast does not replay",
            )

    @property
    def forecast_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "forecast-")


@dataclass(frozen=True, slots=True)
class V1LiteralJAffineShellDirectionReceipt:
    lineage_hash: str
    forecast_hash: str
    direction_source_hash: str
    outside_direction_source_hash: str
    outside_value_digest: str
    shell_index_value_digest: str
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
    shell_route_ids: tuple[str, ...]
    direction_ids: tuple[str, ...]
    literal_j_stream_source_hash: str
    same_capture_vjp_source_hash: str
    source_receipt_hash: str
    alpha_lineage_hash: str
    preoutcome_source_registration_hash: str | None
    execution_mode: str
    retained_device_bytes: int
    source_registration_hash: str | None = None
    physical_action_hash: str | None = None
    packed_action_hash: str | None = None
    literal_j_cell_hash: str | None = None
    literal_j_cell_scope_hash: str | None = None
    literal_j_stream_binding_hash: str | None = None
    exact_postsource_custody_hash: str | None = None
    preoutcome_fanout_receipt_hash: str | None = None
    preoutcome_destination_binding_hash: str | None = None
    preoutcome_binding_axis_hash: str | None = None
    preoutcome_physical_action_axis_hash: str | None = None
    postsource_receipt_hash: str | None = None
    projection_replay_registration_receipt_hash: str | None = None
    construction_custody_kind: str = (
        V1LiteralJAffineShellDirectionConstructionCustodyKind.
        CALLER_VECTOR_COMPATIBILITY.value
    )
    construction_custody_receipt_hash: str | None = None
    array_free_durable_receipt: bool = True
    structured_direction_custody_only: bool = True
    outside_direction_zero_on_shell: bool = True
    route_scale_direction_d2h: bool = False
    preoutcome_source_registration_bound: bool = False
    exact_preoutcome_postsource_custody_bound: bool = False
    outside_direction_exact_action_custody: bool = False
    packed_action_h2d_bytes: int = 0
    construction_tile_logical_payload_upper_bound_bytes: int = 0
    construction_logical_payload_upper_bound_bytes: int = 0
    gpu_packed_materializer_used: bool = False
    route_scale_numpy_unpack_used: bool = False
    population_moment_authority: bool = False
    registered_fd_replacement_authority: bool = False
    fd_q_residual_charged: bool = False
    solver_authority: bool = False
    d_stability_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = LITERAL_J_AFFINE_SHELL_DIRECTION_ALGORITHM
    authority_scope: str = LITERAL_J_AFFINE_SHELL_DIRECTION_AUTHORITY
    schema: str = LITERAL_J_AFFINE_SHELL_DIRECTION_SCHEMA
    _issuer: object = field(default=None, repr=False, compare=False)
    _sealed_receipt_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self._issuer is not _ISSUER:
            _refuse(
                V1LiteralJAffineShellDirectionRefusalCode.INVALID_TYPE,
                "direction receipts are issuer-sealed",
            )
        for name in (
            "lineage_hash",
            "forecast_hash",
            "direction_source_hash",
            "outside_direction_source_hash",
            "outside_value_digest",
            "shell_index_value_digest",
        ):
            _identity(getattr(self, name), name=name)
        object.__setattr__(
            self,
            "preoutcome_source_registration_hash",
            _optional_identity(
                self.preoutcome_source_registration_hash,
                name="preoutcome_source_registration_hash",
            ),
        )
        common_exact_identities = (
            "source_registration_hash",
            "physical_action_hash",
            "packed_action_hash",
            "literal_j_cell_hash",
            "literal_j_cell_scope_hash",
            "literal_j_stream_binding_hash",
            "exact_postsource_custody_hash",
            "preoutcome_fanout_receipt_hash",
            "preoutcome_destination_binding_hash",
            "preoutcome_binding_axis_hash",
            "preoutcome_physical_action_axis_hash",
            "postsource_receipt_hash",
        )
        for name in common_exact_identities + (
            "projection_replay_registration_receipt_hash",
            "construction_custody_receipt_hash",
        ):
            object.__setattr__(
                self,
                name,
                _optional_identity(getattr(self, name), name=name),
            )
        try:
            custody_kind = V1LiteralJAffineShellDirectionConstructionCustodyKind(
                self.construction_custody_kind
            )
        except (TypeError, ValueError):
            _refuse(
                V1LiteralJAffineShellDirectionRefusalCode.INVALID_DESIGN,
                "direction construction custody kind is unsupported",
            )
        if (
            self.array_free_durable_receipt is not True
            or self.structured_direction_custody_only is not True
            or self.outside_direction_zero_on_shell is not True
        ):
            _refuse(
                V1LiteralJAffineShellDirectionRefusalCode.AUTHORITY_VIOLATION,
                "receipt lost array-free source-only semantics",
            )
        closed = (
            self.route_scale_direction_d2h,
            self.population_moment_authority,
            self.registered_fd_replacement_authority,
            self.fd_q_residual_charged,
            self.solver_authority,
            self.d_stability_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            type(self.preoutcome_source_registration_bound) is not bool
            or self.preoutcome_source_registration_bound
            != (self.preoutcome_source_registration_hash is not None)
        ):
            _refuse(
                V1LiteralJAffineShellDirectionRefusalCode.AUTHORITY_VIOLATION,
                "pre-outcome source registration flag must match its hash custody",
            )
        exact_present = tuple(
            getattr(self, name) is not None for name in common_exact_identities
        )
        canonical = self.exact_preoutcome_postsource_custody_bound
        legacy_projection = (
            custody_kind
            is V1LiteralJAffineShellDirectionConstructionCustodyKind.
            LEGACY_PROJECTION_REPLAY
        )
        joint_population = (
            custody_kind
            is V1LiteralJAffineShellDirectionConstructionCustodyKind.
            JOINT_POPULATION_SOURCE
        )
        if (
            type(self.exact_preoutcome_postsource_custody_bound) is not bool
            or type(self.outside_direction_exact_action_custody) is not bool
            or self.exact_preoutcome_postsource_custody_bound
            is not self.outside_direction_exact_action_custody
            or canonical is not all(exact_present)
            or (not canonical and any(exact_present))
            or (
                canonical and not self.preoutcome_source_registration_bound
            )
            or (
                canonical
                and not (legacy_projection or joint_population)
            )
            or (
                canonical
                and self.construction_custody_receipt_hash is None
            )
            or (
                legacy_projection
                and self.projection_replay_registration_receipt_hash
                != self.construction_custody_receipt_hash
            )
            or (
                joint_population
                and self.projection_replay_registration_receipt_hash is not None
            )
            or (
                not canonical
                and (
                    custody_kind
                    is not V1LiteralJAffineShellDirectionConstructionCustodyKind.
                    CALLER_VECTOR_COMPATIBILITY
                    or self.construction_custody_receipt_hash is not None
                    or self.projection_replay_registration_receipt_hash is not None
                )
            )
        ):
            _refuse(
                V1LiteralJAffineShellDirectionRefusalCode.AUTHORITY_VIOLATION,
                "exact pre-outcome/postsource/action custody is incomplete",
            )
        object.__setattr__(
            self,
            "packed_action_h2d_bytes",
            _nonnegative_int(
                self.packed_action_h2d_bytes,
                name="packed_action_h2d_bytes",
            ),
        )
        object.__setattr__(
            self,
            "construction_logical_payload_upper_bound_bytes",
            _nonnegative_int(
                self.construction_logical_payload_upper_bound_bytes,
                name="construction_logical_payload_upper_bound_bytes",
            ),
        )
        object.__setattr__(
            self,
            "construction_tile_logical_payload_upper_bound_bytes",
            _nonnegative_int(
                self.construction_tile_logical_payload_upper_bound_bytes,
                name="construction_tile_logical_payload_upper_bound_bytes",
            ),
        )
        if (
            type(self.gpu_packed_materializer_used) is not bool
            or type(self.route_scale_numpy_unpack_used) is not bool
            or (
                self.execution_mode
                == V1LiteralJAffineShellDirectionExecutionMode.CUDA_PRODUCTION.value
                and self.exact_preoutcome_postsource_custody_bound
                and (
                    self.gpu_packed_materializer_used is not True
                    or self.route_scale_numpy_unpack_used is not False
                    or self.packed_action_h2d_bytes <= 0
                    or self.construction_tile_logical_payload_upper_bound_bytes
                    <= 0
                    or self.construction_logical_payload_upper_bound_bytes
                    < self.retained_device_bytes
                )
            )
            or (
                self.execution_mode
                == V1LiteralJAffineShellDirectionExecutionMode.CPU_REFERENCE_TEST_ONLY.value
                and self.exact_preoutcome_postsource_custody_bound
                and (
                    self.gpu_packed_materializer_used is not False
                    or self.route_scale_numpy_unpack_used is not True
                    or self.packed_action_h2d_bytes != 0
                    or self.construction_tile_logical_payload_upper_bound_bytes != 0
                    or self.construction_logical_payload_upper_bound_bytes != 0
                )
            )
            or (
                not self.exact_preoutcome_postsource_custody_bound
                and (
                    self.gpu_packed_materializer_used
                    or self.route_scale_numpy_unpack_used
                    or self.packed_action_h2d_bytes != 0
                    or self.construction_tile_logical_payload_upper_bound_bytes != 0
                    or self.construction_logical_payload_upper_bound_bytes != 0
                )
            )
        ):
            _refuse(
                V1LiteralJAffineShellDirectionRefusalCode.AUTHORITY_VIOLATION,
                "structured-direction construction telemetry is inconsistent",
            )
        if any(type(item) is not bool or item is not False for item in closed):
            _refuse(
                V1LiteralJAffineShellDirectionRefusalCode.AUTHORITY_VIOLATION,
                "structured direction receipt leaked downstream authority",
            )
        if (
            self.algorithm != LITERAL_J_AFFINE_SHELL_DIRECTION_ALGORITHM
            or self.authority_scope != LITERAL_J_AFFINE_SHELL_DIRECTION_AUTHORITY
            or self.schema != LITERAL_J_AFFINE_SHELL_DIRECTION_SCHEMA
        ):
            _refuse(
                V1LiteralJAffineShellDirectionRefusalCode.INVALID_DESIGN,
                "unsupported structured direction semantics",
            )

        object.__setattr__(self, "_sealed_receipt_hash", self._compute_hash())

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "receipt-")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_receipt_hash or (
            expected_hash is not None and observed != expected_hash
        ):
            _refuse(
                V1LiteralJAffineShellDirectionRefusalCode.MUTATED_CUSTODY,
                "structured direction receipt changed after issuer sealing",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class V1LiteralJAffineShellDirectionLease:
    __slots__ = ("_owner", "_token", "_closed", "receipt_hash")
    __hash__ = None

    def __init__(
        self,
        owner: "V1LiteralJAffineShellDirectionDeviceHandle",
        token: object,
    ) -> None:
        self._owner = owner
        self._token = token
        self._closed = False
        self.receipt_hash = owner.receipt.receipt_hash

    def structured_tensors(self) -> tuple[Any, Any]:
        if self._closed:
            _refuse(
                V1LiteralJAffineShellDirectionRefusalCode.EVICTED,
                "structured direction lease is closed",
            )
        if self._owner.receipt.receipt_hash != self.receipt_hash:
            _refuse(
                V1LiteralJAffineShellDirectionRefusalCode.STALE_IDENTITY,
                "structured direction receipt changed while leased",
            )
        return self._owner._require_live()

    def close(self) -> None:
        if self._closed:
            return
        self._owner._close_lease(self._token)
        self._closed = True

    def __enter__(self) -> "V1LiteralJAffineShellDirectionLease":
        self.structured_tensors()
        return self

    def __exit__(self, *_: object) -> bool:
        self.close()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("structured direction leases are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("structured direction leases are nonserializable")


class V1LiteralJAffineShellDirectionDeviceHandle:
    __slots__ = (
        "_outside",
        "_shell_indices",
        "_guards",
        "_invariants",
        "_tokens",
        "_receipt",
        "_forecast",
        "_receipt_hash",
    )

    def __init__(
        self,
        *,
        outside: Any,
        shell_indices: Any,
        receipt: V1LiteralJAffineShellDirectionReceipt,
        forecast: V1LiteralJAffineShellDirectionMemoryForecast,
    ) -> None:
        self._outside = outside
        self._shell_indices = shell_indices
        self._guards = (outside.detach().clone(), shell_indices.detach().clone())
        self._invariants = (_tensor_invariant(outside), _tensor_invariant(shell_indices))
        self._tokens: set[object] = set()
        self._receipt = receipt
        self._forecast = forecast
        self._receipt_hash = receipt.assert_integrity()
        if (
            receipt.forecast_hash != forecast.forecast_hash
            or receipt.retained_device_bytes != forecast.retained_device_bytes
            or receipt.packed_action_h2d_bytes != forecast.packed_action_h2d_bytes
            or receipt.construction_tile_logical_payload_upper_bound_bytes
            != forecast.construction_tile_logical_payload_upper_bound_bytes
            or receipt.construction_logical_payload_upper_bound_bytes
            != forecast.construction_logical_payload_upper_bound_bytes
            or receipt.gpu_packed_materializer_used
            != forecast.gpu_packed_materializer_used
            or receipt.route_scale_numpy_unpack_used
            != forecast.route_scale_numpy_unpack_used
        ):
            _refuse(
                V1LiteralJAffineShellDirectionRefusalCode.STALE_IDENTITY,
                "direction receipt and construction forecast differ",
            )
        self._require_live()

    def _require_live(self) -> tuple[Any, Any]:
        self._receipt.assert_integrity(self._receipt_hash)
        if self._outside is None or self._shell_indices is None:
            _refuse(
                V1LiteralJAffineShellDirectionRefusalCode.EVICTED,
                "structured direction custody was evicted",
            )
        tensors = (self._outside, self._shell_indices)
        if tuple(_tensor_invariant(item) for item in tensors) != self._invariants or any(
            not bool(torch.equal(item, guard))
            for item, guard in zip(tensors, self._guards)
        ):
            _refuse(
                V1LiteralJAffineShellDirectionRefusalCode.MUTATED_CUSTODY,
                "structured direction tensors changed after sealing",
            )
        return tensors

    @property
    def receipt(self) -> V1LiteralJAffineShellDirectionReceipt:
        self._require_live()
        return self._receipt

    @property
    def forecast(self) -> V1LiteralJAffineShellDirectionMemoryForecast:
        self._require_live()
        return self._forecast

    @property
    def is_evicted(self) -> bool:
        return self._outside is None

    def borrow(self) -> V1LiteralJAffineShellDirectionLease:
        self._require_live()
        token = object()
        self._tokens.add(token)
        return V1LiteralJAffineShellDirectionLease(self, token)

    def _close_lease(self, token: object) -> None:
        if token not in self._tokens:
            _refuse(
                V1LiteralJAffineShellDirectionRefusalCode.STALE_IDENTITY,
                "structured direction lease token is stale",
            )
        try:
            self._require_live()
        except Exception:
            self._tokens.remove(token)
            raise
        self._tokens.remove(token)

    def evict(self) -> None:
        mutation_error: Exception | None = None
        try:
            self._require_live()
        except V1LiteralJAffineShellDirectionRefusal as error:
            if error.code is not V1LiteralJAffineShellDirectionRefusalCode.MUTATED_CUSTODY:
                raise
            mutation_error = error
        if self._tokens:
            _refuse(
                V1LiteralJAffineShellDirectionRefusalCode.ACTIVE_LEASE,
                "cannot evict structured directions while leased",
            )
        self._outside = None
        self._shell_indices = None
        self._guards = ()
        self._invariants = ()
        if mutation_error is not None:
            raise mutation_error

    def __reduce__(self) -> NoReturn:
        raise TypeError("structured direction handles are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("structured direction handles are nonserializable")


def _issue_v1_literal_j_affine_shell_direction_cuda(
    *,
    lineage: V1LiteralJAffineShellDirectionLineage,
    z_outside: Any,
    outside_direction_source_hash: str,
    execution_mode: V1LiteralJAffineShellDirectionExecutionMode,
    exact_binding: dict[str, object] | None = None,
    construction_telemetry: dict[str, object] | None = None,
    take_direction_ownership: bool = False,
) -> V1LiteralJAffineShellDirectionDeviceHandle:
    """Seal one dense outside vector and one bounded shell index vector."""

    if torch is None or type(lineage) is not V1LiteralJAffineShellDirectionLineage:
        _refuse(
            V1LiteralJAffineShellDirectionRefusalCode.INVALID_TYPE,
            "issuance requires exact lineage and PyTorch",
        )
    if type(execution_mode) is not V1LiteralJAffineShellDirectionExecutionMode:
        _refuse(
            V1LiteralJAffineShellDirectionRefusalCode.INVALID_TYPE,
            "execution mode must be exact",
        )
    if type(take_direction_ownership) is not bool:
        _refuse(
            V1LiteralJAffineShellDirectionRefusalCode.INVALID_TYPE,
            "direction ownership marker must be exact",
        )
    source_hash = _identity(
        outside_direction_source_hash, name="outside_direction_source_hash"
    )
    if exact_binding is not None:
        legacy_exact = {
            "source_registration_hash",
            "physical_action_hash",
            "packed_action_hash",
            "literal_j_cell_hash",
            "literal_j_cell_scope_hash",
            "literal_j_stream_binding_hash",
            "exact_postsource_custody_hash",
            "preoutcome_fanout_receipt_hash",
            "preoutcome_destination_binding_hash",
            "preoutcome_binding_axis_hash",
            "preoutcome_physical_action_axis_hash",
            "postsource_receipt_hash",
            "projection_replay_registration_receipt_hash",
        }
        required_exact = legacy_exact | {
            "construction_custody_kind",
            "construction_custody_receipt_hash",
        }
        if type(exact_binding) is not dict:
            _refuse(
                V1LiteralJAffineShellDirectionRefusalCode.INVALID_TYPE,
                "exact postsource binding has an unsupported coordinate axis",
            )
        # Private historical callers supplied only the projection-replay
        # coordinate.  Normalize that exact legacy shape into the generic
        # construction-custody contract so old issuers and replay tests remain
        # valid without allowing a joint-population hash to masquerade as a
        # projection registration.
        if set(exact_binding) == legacy_exact:
            projection_hash = _identity(
                exact_binding["projection_replay_registration_receipt_hash"],
                name="projection_replay_registration_receipt_hash",
            )
            exact_binding = {
                **exact_binding,
                "construction_custody_kind": (
                    V1LiteralJAffineShellDirectionConstructionCustodyKind.
                    LEGACY_PROJECTION_REPLAY.value
                ),
                "construction_custody_receipt_hash": projection_hash,
            }
        if set(exact_binding) != required_exact:
            _refuse(
                V1LiteralJAffineShellDirectionRefusalCode.INVALID_TYPE,
                "exact postsource binding has an unsupported coordinate axis",
            )
        try:
            custody_kind = V1LiteralJAffineShellDirectionConstructionCustodyKind(
                exact_binding["construction_custody_kind"]
            )
        except (TypeError, ValueError):
            _refuse(
                V1LiteralJAffineShellDirectionRefusalCode.INVALID_DESIGN,
                "canonical direction construction custody kind is unsupported",
            )
        if custody_kind is (
            V1LiteralJAffineShellDirectionConstructionCustodyKind.
            CALLER_VECTOR_COMPATIBILITY
        ):
            _refuse(
                V1LiteralJAffineShellDirectionRefusalCode.AUTHORITY_VIOLATION,
                "caller-authored custody cannot be promoted to canonical construction",
            )
        normalized: dict[str, object] = {}
        for name, value in exact_binding.items():
            if name == "projection_replay_registration_receipt_hash" and value is None:
                normalized[name] = None
            elif name == "construction_custody_kind":
                normalized[name] = custody_kind.value
            else:
                normalized[name] = _identity(value, name=name)
        if (
            custody_kind
            is V1LiteralJAffineShellDirectionConstructionCustodyKind.
            LEGACY_PROJECTION_REPLAY
            and normalized["projection_replay_registration_receipt_hash"]
            != normalized["construction_custody_receipt_hash"]
        ) or (
            custody_kind
            is V1LiteralJAffineShellDirectionConstructionCustodyKind.
            JOINT_POPULATION_SOURCE
            and normalized["projection_replay_registration_receipt_hash"] is not None
        ):
            _refuse(
                V1LiteralJAffineShellDirectionRefusalCode.AUTHORITY_VIOLATION,
                "construction custody kind and receipt coordinates disagree",
            )
        exact_binding = normalized
    if exact_binding is None:
        if construction_telemetry is not None or take_direction_ownership:
            _refuse(
                V1LiteralJAffineShellDirectionRefusalCode.AUTHORITY_VIOLATION,
                "compatibility directions cannot claim issuer-owned construction",
            )
        telemetry = {
            "packed_action_h2d_bytes": 0,
            "construction_tile_logical_payload_upper_bound_bytes": 0,
            "construction_logical_payload_upper_bound_bytes": 0,
            "gpu_packed_materializer_used": False,
            "route_scale_numpy_unpack_used": False,
        }
    else:
        required_telemetry = {
            "packed_action_h2d_bytes",
            "construction_tile_logical_payload_upper_bound_bytes",
            "construction_logical_payload_upper_bound_bytes",
            "gpu_packed_materializer_used",
            "route_scale_numpy_unpack_used",
        }
        if (
            type(construction_telemetry) is not dict
            or set(construction_telemetry) != required_telemetry
            or not take_direction_ownership
        ):
            _refuse(
                V1LiteralJAffineShellDirectionRefusalCode.AUTHORITY_VIOLATION,
                "canonical direction construction lacks exact bounded telemetry",
            )
        telemetry = dict(construction_telemetry)
    if (
        type(z_outside) is not torch.Tensor
        or z_outside.dtype != torch.float64
        or tuple(z_outside.shape) != (lineage.route_count,)
        or not z_outside.is_contiguous()
        or bool(z_outside.requires_grad)
    ):
        _refuse(
            V1LiteralJAffineShellDirectionRefusalCode.INVALID_TYPE,
            "z_outside must be detached contiguous float64 on the registered route axis",
        )
    if not bool(torch.isfinite(z_outside).all().item()):
        _refuse(
            V1LiteralJAffineShellDirectionRefusalCode.NONFINITE_DIRECTION,
            "z_outside contains nonfinite values",
        )
    shell_probe = torch.tensor(
        lineage.shell_indices,
        dtype=torch.int64,
        device=z_outside.device,
    )
    if bool((z_outside.index_select(0, shell_probe) != 0.0).any().item()):
        _refuse(
            V1LiteralJAffineShellDirectionRefusalCode.INVALID_DESIGN,
            "z_outside must be exactly zero on every frozen shell coordinate",
        )
    if (
        execution_mode is V1LiteralJAffineShellDirectionExecutionMode.CUDA_PRODUCTION
        and z_outside.device.type != "cuda"
    ):
        _refuse(
            V1LiteralJAffineShellDirectionRefusalCode.DEVICE_REQUIRED,
            "production structured directions must remain CUDA-resident",
        )
    outside = (
        z_outside
        if take_direction_ownership
        else z_outside.detach().clone(memory_format=torch.contiguous_format)
    )
    shell = torch.tensor(
        lineage.shell_indices,
        dtype=torch.int64,
        device=outside.device,
    ).contiguous()
    forecast = V1LiteralJAffineShellDirectionMemoryForecast(
        route_count=lineage.route_count,
        shell_size=len(lineage.shell_indices),
        outside_float64_bytes=lineage.route_count * _F64_BYTES,
        shell_index_int64_bytes=len(lineage.shell_indices) * _I64_BYTES,
        mutation_guard_bytes=(
            lineage.route_count * _F64_BYTES
            + len(lineage.shell_indices) * _I64_BYTES
        ),
        retained_device_bytes=2
        * (
            lineage.route_count * _F64_BYTES
            + len(lineage.shell_indices) * _I64_BYTES
        ),
        dense_basis_bank_bytes_avoided=(
            len(lineage.shell_indices) * lineage.route_count * _F64_BYTES
        ),
        packed_action_h2d_bytes=telemetry["packed_action_h2d_bytes"],
        construction_tile_logical_payload_upper_bound_bytes=telemetry[
            "construction_tile_logical_payload_upper_bound_bytes"
        ],
        construction_logical_payload_upper_bound_bytes=telemetry[
            "construction_logical_payload_upper_bound_bytes"
        ],
        gpu_packed_materializer_used=telemetry["gpu_packed_materializer_used"],
        route_scale_numpy_unpack_used=telemetry[
            "route_scale_numpy_unpack_used"
        ],
    )
    if execution_mode is V1LiteralJAffineShellDirectionExecutionMode.CPU_REFERENCE_TEST_ONLY:
        outside_digest = _exact_cpu_reference_digest(outside, role="z_outside")
    else:
        # Production identity comes from the upstream owner.  Re-hashing a
        # route-scale CUDA vector would perform precisely the D2H transfer this
        # structured owner exists to avoid.
        outside_digest = content_hash(
            {
                "schema": "literal_J_outside_direction_source_bound_value_v1",
                "outside_direction_source_hash": source_hash,
                "route_axis_hash": lineage.route_axis_hash,
                "route_id_axis_hash": lineage.route_id_axis_hash,
                "route_count": lineage.route_count,
                "dtype": "float64",
                "route_scale_direction_d2h": False,
            },
            prefix=_PREFIX + "outside-source-bound-",
        )
    shell_digest = _exact_cpu_reference_digest(shell, role="shell_indices")
    direction_source_hash = content_hash(
        {
            "schema": "literal_J_affine_shell_direction_source_v2",
            "lineage_hash": lineage.lineage_hash,
            "outside_direction_source_hash": source_hash,
            "outside_value_digest": outside_digest,
            "shell_index_value_digest": shell_digest,
            "direction_ids": lineage.direction_ids,
            "preoutcome_source_registration_hash": (
                lineage.preoutcome_source_registration_hash
            ),
            "exact_postsource_binding": exact_binding,
            "algorithm": LITERAL_J_AFFINE_SHELL_DIRECTION_ALGORITHM,
        },
        prefix=_PREFIX + "source-",
    )
    receipt = V1LiteralJAffineShellDirectionReceipt(
        lineage_hash=lineage.lineage_hash,
        forecast_hash=forecast.forecast_hash,
        direction_source_hash=direction_source_hash,
        outside_direction_source_hash=source_hash,
        outside_value_digest=outside_digest,
        shell_index_value_digest=shell_digest,
        model_identity_hash=lineage.model_identity_hash,
        chart_identity_hash=lineage.chart_identity_hash,
        current_mask_hash=lineage.current_mask_hash,
        proposal_generation=lineage.proposal_generation,
        route_axis_hash=lineage.route_axis_hash,
        route_id_axis_hash=lineage.route_id_axis_hash,
        route_count=lineage.route_count,
        registered_d_prefix_hash=lineage.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=lineage.registered_d_w2_prefix_hash,
        shell_freeze_receipt_hash=lineage.shell_freeze_receipt_hash,
        shell_indices=lineage.shell_indices,
        shell_route_ids=lineage.shell_route_ids,
        direction_ids=lineage.direction_ids,
        literal_j_stream_source_hash=lineage.literal_j_stream_source_hash,
        same_capture_vjp_source_hash=lineage.same_capture_vjp_source_hash,
        source_receipt_hash=lineage.source_receipt_hash,
        alpha_lineage_hash=lineage.alpha_lineage_hash,
        preoutcome_source_registration_hash=(
            lineage.preoutcome_source_registration_hash
        ),
        execution_mode=execution_mode.value,
        retained_device_bytes=forecast.retained_device_bytes,
        preoutcome_source_registration_bound=(
            lineage.preoutcome_source_registration_hash is not None
        ),
        source_registration_hash=(
            None if exact_binding is None else exact_binding["source_registration_hash"]
        ),
        physical_action_hash=(
            None if exact_binding is None else exact_binding["physical_action_hash"]
        ),
        packed_action_hash=(
            None if exact_binding is None else exact_binding["packed_action_hash"]
        ),
        literal_j_cell_hash=(
            None if exact_binding is None else exact_binding["literal_j_cell_hash"]
        ),
        literal_j_cell_scope_hash=(
            None if exact_binding is None else exact_binding["literal_j_cell_scope_hash"]
        ),
        literal_j_stream_binding_hash=(
            None
            if exact_binding is None
            else exact_binding["literal_j_stream_binding_hash"]
        ),
        exact_postsource_custody_hash=(
            None
            if exact_binding is None
            else exact_binding["exact_postsource_custody_hash"]
        ),
        preoutcome_fanout_receipt_hash=(
            None
            if exact_binding is None
            else exact_binding["preoutcome_fanout_receipt_hash"]
        ),
        preoutcome_destination_binding_hash=(
            None
            if exact_binding is None
            else exact_binding["preoutcome_destination_binding_hash"]
        ),
        preoutcome_binding_axis_hash=(
            None
            if exact_binding is None
            else exact_binding["preoutcome_binding_axis_hash"]
        ),
        preoutcome_physical_action_axis_hash=(
            None
            if exact_binding is None
            else exact_binding["preoutcome_physical_action_axis_hash"]
        ),
        postsource_receipt_hash=(
            None if exact_binding is None else exact_binding["postsource_receipt_hash"]
        ),
        projection_replay_registration_receipt_hash=(
            None
            if exact_binding is None
            else exact_binding["projection_replay_registration_receipt_hash"]
        ),
        construction_custody_kind=(
            V1LiteralJAffineShellDirectionConstructionCustodyKind.
            CALLER_VECTOR_COMPATIBILITY.value
            if exact_binding is None
            else exact_binding["construction_custody_kind"]
        ),
        construction_custody_receipt_hash=(
            None
            if exact_binding is None
            else exact_binding["construction_custody_receipt_hash"]
        ),
        exact_preoutcome_postsource_custody_bound=(exact_binding is not None),
        outside_direction_exact_action_custody=(exact_binding is not None),
        packed_action_h2d_bytes=telemetry["packed_action_h2d_bytes"],
        construction_tile_logical_payload_upper_bound_bytes=telemetry[
            "construction_tile_logical_payload_upper_bound_bytes"
        ],
        construction_logical_payload_upper_bound_bytes=telemetry[
            "construction_logical_payload_upper_bound_bytes"
        ],
        gpu_packed_materializer_used=telemetry["gpu_packed_materializer_used"],
        route_scale_numpy_unpack_used=telemetry[
            "route_scale_numpy_unpack_used"
        ],
        _issuer=_ISSUER,
    )
    return V1LiteralJAffineShellDirectionDeviceHandle(
        outside=outside,
        shell_indices=shell,
        receipt=receipt,
        forecast=forecast,
    )


def issue_v1_literal_j_affine_shell_direction_cuda(
    *,
    lineage: V1LiteralJAffineShellDirectionLineage,
    z_outside: Any,
    outside_direction_source_hash: str,
    execution_mode: V1LiteralJAffineShellDirectionExecutionMode,
) -> V1LiteralJAffineShellDirectionDeviceHandle:
    """Compatibility issuer without canonical postsource authority.

    A caller-authored lineage or outside-source string can still exercise the
    small-domain numerical owner.  It can never set the exact custody marker
    required by source-bound structured shard replay.
    """

    return _issue_v1_literal_j_affine_shell_direction_cuda(
        lineage=lineage,
        z_outside=z_outside,
        outside_direction_source_hash=outside_direction_source_hash,
        execution_mode=execution_mode,
        exact_binding=None,
        construction_telemetry=None,
        take_direction_ownership=False,
    )


def _materialize_canonical_action_outside(
    *,
    action_owner: object,
    route_count: int,
    shell_indices: tuple[int, ...],
    execution_mode: V1LiteralJAffineShellDirectionExecutionMode,
    device: object,
    packed_byte_tile_size: int,
) -> tuple[Any, dict[str, object]]:
    """Materialize one exact packed action and separate its frozen shell.

    Both canonical construction owners expose the same deliberately narrow
    materializer.  Keeping this operation here makes the numerical result and
    telemetry independent of which typed source supplied the action.
    """

    tile = _positive_int(packed_byte_tile_size, name="packed_byte_tile_size")
    production_cuda = (
        execution_mode is V1LiteralJAffineShellDirectionExecutionMode.CUDA_PRODUCTION
    )
    materialize = getattr(action_owner, "materialize_action_float64", None)
    if materialize is None:
        # The historical action binding uses the shorter method name.
        materialize = getattr(action_owner, "materialize_float64", None)
    if not callable(materialize):
        _refuse(
            V1LiteralJAffineShellDirectionRefusalCode.ACTION_CUSTODY_MISMATCH,
            "canonical action owner exposes no exact FP64 materializer",
        )
    try:
        outside, returned_h2d_bytes, tile_payload = materialize(
            device=device,
            tile_bytes=tile,
            production_cuda_required=production_cuda,
        )
    except V1LiteralJAffineShellDirectionRefusal:
        raise
    except Exception as error:
        _refuse(
            V1LiteralJAffineShellDirectionRefusalCode.ACTION_CUSTODY_MISMATCH,
            "packed action could not be materialised from its live issuer: "
            f"{type(error).__name__}: {error}",
        )
    if (
        torch is None
        or type(outside) is not torch.Tensor
        or outside.dtype != torch.float64
        or tuple(outside.shape) != (route_count,)
        or not outside.is_contiguous()
        or bool(outside.requires_grad)
        or not bool(torch.isfinite(outside).all().item())
    ):
        _refuse(
            V1LiteralJAffineShellDirectionRefusalCode.ACTION_CUSTODY_MISMATCH,
            "packed action issuer returned a noncanonical finite FP64 direction",
        )
    expected_packed_h2d_bytes = 2 * ((route_count + 7) // 8)
    if production_cuda and (
        type(returned_h2d_bytes) is not int
        or returned_h2d_bytes != expected_packed_h2d_bytes
        or type(tile_payload) is not int
        or tile_payload <= 0
    ):
        _refuse(
            V1LiteralJAffineShellDirectionRefusalCode.ACTION_CUSTODY_MISMATCH,
            "packed CUDA action materializer returned inconsistent transfer or workspace accounting",
        )
    if not production_cuda and (
        type(tile_payload) is not int or tile_payload != 0
    ):
        _refuse(
            V1LiteralJAffineShellDirectionRefusalCode.ACTION_CUSTODY_MISMATCH,
            "CPU reference action materializer returned CUDA construction accounting",
        )
    shell = torch.tensor(shell_indices, dtype=torch.int64, device=outside.device)
    outside.index_fill_(0, shell, 0.0)
    outside = outside.contiguous()
    retained_bytes = 2 * (
        route_count * _F64_BYTES + len(shell_indices) * _I64_BYTES
    )
    construction_payload = (
        max(
            retained_bytes,
            route_count * _F64_BYTES
            + int(tile_payload)
            + len(shell_indices) * _I64_BYTES,
        )
        if production_cuda
        else 0
    )
    return outside, {
        "packed_action_h2d_bytes": (
            int(returned_h2d_bytes) if production_cuda else 0
        ),
        "construction_tile_logical_payload_upper_bound_bytes": (
            int(tile_payload) if production_cuda else 0
        ),
        "construction_logical_payload_upper_bound_bytes": construction_payload,
        "gpu_packed_materializer_used": production_cuda,
        "route_scale_numpy_unpack_used": not production_cuda,
    }


def issue_v1_literal_j_affine_shell_direction_from_exact_postsource_cuda(
    *,
    preoutcome_fanout_receipt: V1SampledV7AffineShellPreoutcomeFanoutReceipt,
    preoutcome_destination_binding: V1SampledV7AffineShellPreoutcomeDestinationBinding,
    preoutcome_registration: V1LiteralJAffineShellDirectionPreoutcomeRegistration,
    postsource_receipt: V1SignedWorkVariableShellSourceReceipt,
    projection_replay_registration_receipt: V1SampledBlockSupportReplayRegistrationReceipt,
    action_custody: V1SignedWorkBlockSupportPreferredActionBinding,
    execution_mode: V1LiteralJAffineShellDirectionExecutionMode,
    device: object,
    packed_byte_tile_size: int = 1 << 20,
) -> V1LiteralJAffineShellDirectionDeviceHandle:
    """Issue the canonical structured direction from exact typed custody.

    The dense outside direction is materialised by the issuer-owned packed
    parent/child binding, not supplied by a caller.  The already-spent
    sampled-v7 literal-J cell is only identity-bound here; this function does
    not allocate or spend alpha.
    """

    # Local imports avoid a cycle through the two-pass projection owner, which
    # imports this module for its structured numerical handle.
    from .modern_v1_literal_j_affine_shell_preoutcome_registration import (
        V1LiteralJAffineShellDirectionPreoutcomeRegistration,
    )
    from .modern_v1_signed_work_block_support_source_cuda import (
        V1SignedWorkBlockSupportPreferredActionBinding,
    )
    from .modern_v1_signed_work_variable_shell_source_cuda import (
        V1SignedWorkVariableShellSourceReceipt,
    )
    from .modern_v1_sampled_v7_affine_shell_preoutcome_fanout import (
        V1SampledV7AffineShellPreoutcomeDestinationBinding,
        V1SampledV7AffineShellPreoutcomeFanoutReceipt,
    )
    from .modern_v1_sampled_block_support_replay_registration import (
        V1SampledBlockSupportReplayRegistrationReceipt,
    )

    if (
        type(preoutcome_fanout_receipt)
        is not V1SampledV7AffineShellPreoutcomeFanoutReceipt
        or type(preoutcome_destination_binding)
        is not V1SampledV7AffineShellPreoutcomeDestinationBinding
        or type(preoutcome_registration)
        is not V1LiteralJAffineShellDirectionPreoutcomeRegistration
        or type(postsource_receipt) is not V1SignedWorkVariableShellSourceReceipt
        or type(projection_replay_registration_receipt)
        is not V1SampledBlockSupportReplayRegistrationReceipt
        or type(action_custody)
        is not V1SignedWorkBlockSupportPreferredActionBinding
        or type(execution_mode) is not V1LiteralJAffineShellDirectionExecutionMode
    ):
        _refuse(
            V1LiteralJAffineShellDirectionRefusalCode.INVALID_TYPE,
            "canonical issuance requires the exact complete-axis receipt, destination binding, preregistration, postsource, action, and mode types",
        )
    try:
        pre_hash = preoutcome_registration.assert_integrity()
        fanout_hash = preoutcome_fanout_receipt.assert_integrity()
        destination_binding_hash = preoutcome_destination_binding.assert_integrity()
        projection_registration_hash = (
            projection_replay_registration_receipt.assert_integrity()
        )
        action_custody.assert_live()
    except V1LiteralJAffineShellDirectionRefusal:
        raise
    except Exception as error:
        _refuse(
            V1LiteralJAffineShellDirectionRefusalCode.PREOUTCOME_REGISTRATION_MISMATCH,
            "one exact pre-provider or replay parent failed integrity replay: "
            f"{type(error).__name__}: {error}",
        )
    source = postsource_receipt
    pre = preoutcome_registration
    source_hash = source.receipt_hash
    projection_registration = projection_replay_registration_receipt
    ordinal = preoutcome_destination_binding.physical_action_ordinal
    if (
        ordinal <= 0
        or ordinal > preoutcome_fanout_receipt.destination_count
        or preoutcome_fanout_receipt.destination_bindings[ordinal - 1]
        is not preoutcome_destination_binding
        or preoutcome_fanout_receipt.preoutcome_registrations[ordinal - 1]
        is not preoutcome_registration
        or preoutcome_destination_binding.preregistration_hash != pre_hash
        or preoutcome_destination_binding.source_registration_hash
        != pre.source_registration_hash
        or preoutcome_destination_binding.physical_action_hash
        != pre.physical_action_hash
        or preoutcome_destination_binding.physical_action_identity_hash
        != pre.physical_action_identity_hash
        or preoutcome_destination_binding.direction_family_hash
        != pre.direction_family_hash
        or preoutcome_destination_binding.literal_j_owner_binding_receipt_hash
        != pre.literal_j_owner_binding_receipt_hash
        or preoutcome_destination_binding.literal_j_cell_hash
        != pre.literal_j_cell_hash
        or preoutcome_destination_binding.registered_look_index
        != pre.registered_look_index
        or preoutcome_fanout_receipt.literal_j_owner_binding_receipt_hash
        != pre.literal_j_owner_binding_receipt_hash
        or preoutcome_fanout_receipt.literal_j_cell_hash != pre.literal_j_cell_hash
        or preoutcome_fanout_receipt.registered_look_index
        != pre.registered_look_index
    ):
        _refuse(
            V1LiteralJAffineShellDirectionRefusalCode.PREOUTCOME_REGISTRATION_MISMATCH,
            "preregistration is not the exact destination member of one complete pre-provider action axis",
        )
    if (
        projection_registration.structural_registration_hash
        != pre.source_registration_hash
        or projection_registration.model_identity_hash != pre.model_identity_hash
        or projection_registration.chart_identity_hash != pre.chart_identity_hash
        or projection_registration.current_mask_hash != pre.current_mask_hash
        or projection_registration.proposal_generation != pre.proposal_generation
        or projection_registration.route_axis_hash != pre.route_axis_hash
        or projection_registration.route_id_axis_hash != pre.route_id_axis_hash
        or projection_registration.conditioned_d1_registered_prefix_hash
        != pre.registered_d_prefix_hash
        or projection_registration.residual_registered_dw2_prefix_hash
        != pre.registered_d_w2_prefix_hash
        or projection_registration.shell_freeze_receipt_hash
        != pre.shell_freeze_receipt_hash
        or projection_registration.shell_indices != pre.shell_indices
        or projection_registration.shell_route_ids != pre.shell_route_ids
        or projection_registration.composite_same_capture_vjp_source_hash
        != pre.same_capture_vjp_source_hash
        or projection_registration.composite_literal_j_stream_source_hash
        != pre.literal_j_stream_source_hash
        or projection_registration.alpha_lineage_hash
        != pre.historical_alpha_lineage_hash
    ):
        _refuse(
            V1LiteralJAffineShellDirectionRefusalCode.POSTSOURCE_CUSTODY_MISMATCH,
            "D-fixed replay registration left the exact postsource chart, prefix, shell, stream, or alpha lineage",
        )
    if (
        source.registration_hash != pre.source_registration_hash
        or source.model_identity_hash != pre.model_identity_hash
        or source.chart_identity_hash != pre.chart_identity_hash
        or source.current_mask_hash != pre.current_mask_hash
        or source.proposal_generation != pre.proposal_generation
        or source.route_axis_hash != pre.route_axis_hash
        or source.route_id_axis_hash != pre.route_id_axis_hash
        or source.route_count != pre.route_count
        or source.registered_d_prefix_hash != pre.registered_d_prefix_hash
        or source.registered_d_w2_prefix_hash != pre.registered_d_w2_prefix_hash
        or source.shell_freeze_receipt_hash != pre.shell_freeze_receipt_hash
        or source.shell_indices != pre.shell_indices
        or source.shell_route_ids != pre.shell_route_ids
        or source.native_fiber_axis_receipt_hash
        != pre.native_fiber_axis_receipt_hash
        or source.native_fiber_measure_hash != pre.native_fiber_measure_hash
        or source.kkt_branch_source_receipt_hash != pre.physical_action_hash
        or source.same_capture_vjp_source_hash != pre.same_capture_vjp_source_hash
        or source.literal_j_stream_source_hash != pre.literal_j_stream_source_hash
        or source.corrected_evidence_stream_source_hash
        != pre.corrected_evidence_stream_source_hash
        or source.alpha_lineage_hash != pre.historical_alpha_lineage_hash
    ):
        _refuse(
            V1LiteralJAffineShellDirectionRefusalCode.POSTSOURCE_CUSTODY_MISMATCH,
            "post-D_W2 source receipt left the pre-registered chart, prefix, shell, stream, or action",
        )
    if (
        action_custody.source_registration_hash != pre.source_registration_hash
        or action_custody.model_identity_hash != pre.model_identity_hash
        or action_custody.chart_identity_hash != pre.chart_identity_hash
        or action_custody.current_mask_hash != pre.parent_mask_hash
        or action_custody.proposal_generation != pre.proposal_generation
        or action_custody.route_axis_hash != pre.route_axis_hash
        or action_custody.route_count != pre.route_count
        or action_custody.preferred_action_hash != pre.packed_action_hash
        or action_custody.preferred_resulting_mask_hash != pre.resulting_mask_hash
        or action_custody.preferred_removed_route_index_hash
        != pre.removed_route_index_hash
        or action_custody.preferred_selected_count != pre.removed_count
    ):
        _refuse(
            V1LiteralJAffineShellDirectionRefusalCode.ACTION_CUSTODY_MISMATCH,
            "live packed action differs from the pre-registered irreversible direction",
        )
    outside, construction_telemetry = _materialize_canonical_action_outside(
        action_owner=action_custody,
        route_count=pre.route_count,
        shell_indices=pre.shell_indices,
        execution_mode=execution_mode,
        device=device,
        packed_byte_tile_size=packed_byte_tile_size,
    )
    lineage = V1LiteralJAffineShellDirectionLineage(
        model_identity_hash=pre.model_identity_hash,
        chart_identity_hash=pre.chart_identity_hash,
        current_mask_hash=pre.current_mask_hash,
        proposal_generation=pre.proposal_generation,
        route_axis_hash=pre.route_axis_hash,
        route_id_axis_hash=pre.route_id_axis_hash,
        route_count=pre.route_count,
        registered_d_prefix_hash=pre.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=pre.registered_d_w2_prefix_hash,
        shell_freeze_receipt_hash=pre.shell_freeze_receipt_hash,
        shell_indices=pre.shell_indices,
        shell_route_ids=pre.shell_route_ids,
        literal_j_stream_source_hash=pre.literal_j_stream_source_hash,
        same_capture_vjp_source_hash=pre.same_capture_vjp_source_hash,
        source_receipt_hash=(
            projection_registration.full_population_compact_source_receipt_hash
        ),
        alpha_lineage_hash=pre.historical_alpha_lineage_hash,
        preoutcome_source_registration_hash=pre_hash,
    )
    binding_hash = content_hash(
        {
            "schema": "literal_J_affine_shell_exact_postsource_custody_v1",
            "preoutcome_registration_hash": pre_hash,
            "preoutcome_fanout_receipt_hash": fanout_hash,
            "preoutcome_destination_binding_hash": destination_binding_hash,
            "preoutcome_binding_axis_hash": preoutcome_fanout_receipt.binding_axis_hash,
            "preoutcome_physical_action_axis_hash": (
                preoutcome_fanout_receipt.physical_action_axis_hash
            ),
            "postsource_receipt_hash": source_hash,
            "projection_replay_registration_receipt_hash": (
                projection_registration_hash
            ),
            "source_registration_hash": pre.source_registration_hash,
            "physical_action_hash": pre.physical_action_hash,
            "packed_action_hash": pre.packed_action_hash,
            "action_custody_hash": action_custody.binding_hash,
            "outside_direction_source_hash": pre.outside_direction_source_hash,
            "literal_j_cell_hash": pre.literal_j_cell_hash,
            "literal_j_cell_scope_hash": pre.literal_j_cell_scope_hash,
            "literal_j_stream_binding_hash": pre.literal_j_stream_binding_hash,
            "no_new_alpha": True,
        },
        prefix=_PREFIX + "exact-postsource-",
    )
    return _issue_v1_literal_j_affine_shell_direction_cuda(
        lineage=lineage,
        z_outside=outside,
        outside_direction_source_hash=pre.outside_direction_source_hash,
        execution_mode=execution_mode,
        exact_binding={
            "source_registration_hash": pre.source_registration_hash,
            "physical_action_hash": pre.physical_action_hash,
            "packed_action_hash": pre.packed_action_hash,
            "literal_j_cell_hash": pre.literal_j_cell_hash,
            "literal_j_cell_scope_hash": pre.literal_j_cell_scope_hash,
            "literal_j_stream_binding_hash": pre.literal_j_stream_binding_hash,
            "exact_postsource_custody_hash": binding_hash,
            "preoutcome_fanout_receipt_hash": fanout_hash,
            "preoutcome_destination_binding_hash": destination_binding_hash,
            "preoutcome_binding_axis_hash": preoutcome_fanout_receipt.binding_axis_hash,
            "preoutcome_physical_action_axis_hash": (
                preoutcome_fanout_receipt.physical_action_axis_hash
            ),
            "postsource_receipt_hash": source_hash,
            "projection_replay_registration_receipt_hash": (
                projection_registration_hash
            ),
            "construction_custody_kind": (
                V1LiteralJAffineShellDirectionConstructionCustodyKind.
                LEGACY_PROJECTION_REPLAY.value
            ),
            "construction_custody_receipt_hash": projection_registration_hash,
        },
        construction_telemetry=construction_telemetry,
        take_direction_ownership=True,
    )


def issue_v1_literal_j_affine_shell_direction_from_joint_population_source_cuda(
    *,
    joint_population_source: V1LiteralJJointPopulationSource,
    execution_mode: V1LiteralJAffineShellDirectionExecutionMode,
    device: object,
    packed_byte_tile_size: int = 1 << 20,
) -> V1LiteralJAffineShellDirectionDeviceHandle:
    """Issue one direction from the exact current-source population owner.

    The joint source has already bound the complete pre-outcome action member,
    the literal-J D1/residual measure lineage, the exact logical result, and the
    residual postsource.  This issuer therefore needs no obsolete uniform
    D-fixed projection-replay registration.  It reads only the source-owned
    packed action, removes the frozen shell from that outside vector, and seals
    the source receipt as construction custody.
    """

    from .modern_v1_literal_j_joint_population_source import (
        V1LiteralJJointPopulationSource,
    )

    if (
        type(joint_population_source) is not V1LiteralJJointPopulationSource
        or type(execution_mode) is not V1LiteralJAffineShellDirectionExecutionMode
    ):
        _refuse(
            V1LiteralJAffineShellDirectionRefusalCode.INVALID_TYPE,
            "current-source issuance requires the exact joint-population owner and execution mode",
        )
    try:
        source_hash = joint_population_source.assert_live()
        source_receipt = joint_population_source.receipt
        source_receipt.assert_integrity()
        pre = joint_population_source.preoutcome_receipt
        pre_hash = pre.assert_integrity()
        postsource = joint_population_source.postsource_receipt
        postsource_hash = postsource.receipt_hash
    except V1LiteralJAffineShellDirectionRefusal:
        raise
    except Exception as error:
        _refuse(
            V1LiteralJAffineShellDirectionRefusalCode.POSTSOURCE_CUSTODY_MISMATCH,
            "joint-population construction source failed exact live replay: "
            f"{type(error).__name__}: {error}",
        )
    if (
        source_receipt.preoutcome_receipt_hash != pre_hash
        or source_receipt.residual_postsource_receipt_hash != postsource_hash
        or source_receipt.residual_preoutcome_registration_hash
        != pre.residual_preoutcome_registration_hash
        or source_receipt.residual_source_registration_hash
        != pre.residual_source_registration_hash
        or source_receipt.action_custody_hash != pre.action_custody_hash
        or source_receipt.joint_preoutcome_registration_hash
        != pre.joint_preoutcome_registration_hash
        or source_receipt.joint_direction_axis_hash != pre.joint_direction_axis_hash
        or source_receipt.physical_action_ordinal != pre.physical_action_ordinal
        or source_receipt.physical_action_identity_hash
        != pre.physical_action_identity_hash
        or source_receipt.physical_action_hash != pre.physical_action_hash
        or source_receipt.packed_action_hash != pre.packed_action_hash
    ):
        _refuse(
            V1LiteralJAffineShellDirectionRefusalCode.POSTSOURCE_CUSTODY_MISMATCH,
            "joint-population POST receipt left its exact PRE action/source coordinates",
        )
    outside, construction_telemetry = _materialize_canonical_action_outside(
        action_owner=joint_population_source,
        route_count=pre.route_count,
        shell_indices=pre.shell_indices,
        execution_mode=execution_mode,
        device=device,
        packed_byte_tile_size=packed_byte_tile_size,
    )
    lineage = V1LiteralJAffineShellDirectionLineage(
        model_identity_hash=pre.model_identity_hash,
        chart_identity_hash=pre.chart_identity_hash,
        current_mask_hash=pre.current_mask_hash,
        proposal_generation=pre.proposal_generation,
        route_axis_hash=pre.route_axis_hash,
        route_id_axis_hash=pre.route_id_axis_hash,
        route_count=pre.route_count,
        registered_d_prefix_hash=pre.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=pre.registered_d_w2_prefix_hash,
        shell_freeze_receipt_hash=pre.shell_freeze_receipt_hash,
        shell_indices=pre.shell_indices,
        shell_route_ids=pre.shell_route_ids,
        literal_j_stream_source_hash=pre.literal_j_stream_source_hash,
        same_capture_vjp_source_hash=pre.same_capture_vjp_source_hash,
        source_receipt_hash=source_hash,
        alpha_lineage_hash=pre.alpha_lineage_hash,
        preoutcome_source_registration_hash=(
            pre.residual_preoutcome_registration_hash
        ),
    )
    custody_hash = content_hash(
        {
            "schema": "literal_J_affine_shell_joint_population_construction_custody_v1",
            "joint_population_source_receipt_hash": source_hash,
            "joint_population_preoutcome_receipt_hash": pre_hash,
            "residual_postsource_receipt_hash": postsource_hash,
            "joint_preoutcome_registration_hash": (
                pre.joint_preoutcome_registration_hash
            ),
            "joint_direction_axis_hash": pre.joint_direction_axis_hash,
            "action_custody_hash": pre.action_custody_hash,
            "physical_action_ordinal": pre.physical_action_ordinal,
            "physical_action_hash": pre.physical_action_hash,
            "packed_action_hash": pre.packed_action_hash,
            "outside_direction_source_hash": pre.outside_direction_source_hash,
            "literal_j_cell_hash": pre.literal_j_cell_hash,
            "literal_j_cell_scope_hash": pre.literal_j_cell_scope_hash,
            "literal_j_stream_binding_hash": pre.literal_j_stream_binding_hash,
            "no_new_alpha": True,
            "no_provider_or_row_reacquisition": True,
        },
        prefix=_PREFIX + "joint-population-custody-",
    )
    return _issue_v1_literal_j_affine_shell_direction_cuda(
        lineage=lineage,
        z_outside=outside,
        outside_direction_source_hash=pre.outside_direction_source_hash,
        execution_mode=execution_mode,
        exact_binding={
            "source_registration_hash": pre.residual_source_registration_hash,
            "physical_action_hash": pre.physical_action_hash,
            "packed_action_hash": pre.packed_action_hash,
            "literal_j_cell_hash": pre.literal_j_cell_hash,
            "literal_j_cell_scope_hash": pre.literal_j_cell_scope_hash,
            "literal_j_stream_binding_hash": pre.literal_j_stream_binding_hash,
            "exact_postsource_custody_hash": custody_hash,
            "preoutcome_fanout_receipt_hash": pre.fanout_receipt_hash,
            "preoutcome_destination_binding_hash": pre.destination_binding_hash,
            "preoutcome_binding_axis_hash": pre.fanout_binding_axis_hash,
            "preoutcome_physical_action_axis_hash": pre.physical_action_axis_hash,
            "postsource_receipt_hash": postsource_hash,
            "projection_replay_registration_receipt_hash": None,
            "construction_custody_kind": (
                V1LiteralJAffineShellDirectionConstructionCustodyKind.
                JOINT_POPULATION_SOURCE.value
            ),
            "construction_custody_receipt_hash": source_hash,
        },
        construction_telemetry=construction_telemetry,
        take_direction_ownership=True,
    )


__all__ = [
    "LITERAL_J_AFFINE_SHELL_DIRECTION_ALGORITHM",
    "LITERAL_J_AFFINE_SHELL_DIRECTION_AUTHORITY",
    "V1LiteralJAffineShellDirectionExecutionMode",
    "V1LiteralJAffineShellDirectionConstructionCustodyKind",
    "V1LiteralJAffineShellDirectionRefusalCode",
    "V1LiteralJAffineShellDirectionRefusal",
    "V1LiteralJAffineShellDirectionLineage",
    "V1LiteralJAffineShellDirectionMemoryForecast",
    "V1LiteralJAffineShellDirectionReceipt",
    "V1LiteralJAffineShellDirectionLease",
    "V1LiteralJAffineShellDirectionDeviceHandle",
    "issue_v1_literal_j_affine_shell_direction_cuda",
    "issue_v1_literal_j_affine_shell_direction_from_exact_postsource_cuda",
    "issue_v1_literal_j_affine_shell_direction_from_joint_population_source_cuda",
]
