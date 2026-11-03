"""Source-bound analytic supports for one affine-shell literal-J family.

For the structured direction family ``(z_outside, e_b1, ..., e_bB)``, the
coordinatewise analytic support is

``(sum_j literal_j_abs_bound[j] * abs(z_outside[j]),
   literal_j_abs_bound[b1], ..., literal_j_abs_bound[bB])``.

The route-scale inputs stay under their existing live owners.  This module
borrows them, performs one bounded FP64 reduction and one shell gather, and
retains only the resulting ``B+1 <= 9`` support vector plus its mutation
guard.  Durable receipts contain scalars and source identities, never arrays.

This is analytic-support custody only.  It does not estimate a population
quantity, spend alpha, issue a Loewner band, stabilize D, or authorize any
C/K/O, mask-commit, terminal, performance, or Pareto claim.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
import math
from numbers import Real
from types import MappingProxyType
from typing import Any, Final, Mapping, NoReturn, Sequence

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_alexnet_literal_j_support_bounds_cuda import (
    V1AlexNetLiteralJAnalyticSupportSourceResult,
)
from .modern_v1_literal_j_affine_shell_direction_cuda import (
    V1LiteralJAffineShellDirectionDeviceHandle,
)
from .modern_v1_literal_j_residual_population_projection_cuda import (
    V1LiteralJResidualAnalyticSupportSourceCUDAHandle,
    V1LiteralJResidualAnalyticSupportSourceCUDAReceipt,
    V1LiteralJResidualAnalyticSupportSourceCUDAResult,
)

try:  # pragma: no cover - runtime image dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


LITERAL_J_AFFINE_SHELL_ANALYTIC_SUPPORT_SCHEMA: Final = (
    "partition_free_v1_literal_J_affine_shell_analytic_support_receipt_v1"
)
LITERAL_J_AFFINE_SHELL_ANALYTIC_SUPPORT_AUTHORITY: Final = (
    "source_bound_affine_shell_literal_J_analytic_support_only_no_population_"
    "alpha_Loewner_D_C_K_O_commit_terminal_performance_or_pareto_authority_v1"
)
LITERAL_J_AFFINE_SHELL_ANALYTIC_SUPPORT_ALGORITHM: Final = (
    "outward_route_bound_dot_abs_z_outside_plus_shell_gather_no_basis_bank_v1"
)
LITERAL_J_AFFINE_SHELL_ANALYTIC_SUPPORT_EQUATION: Final = (
    "s=(up(sum_j(abs(J_bound_j)*abs(z_outside_j))),J_bound[B])"
)
_PREFIX: Final = "partition-free-v1-literal-j-affine-shell-analytic-support-"
_ISSUER_TOKEN: Final = object()
_F64_BYTES: Final = 8
_F64_UNIT_ROUNDOFF: Final = 2.0**-53
_F64_MIN_SUBNORMAL: Final = float.fromhex("0x0.0000000000001p-1022")


class V1LiteralJAffineShellAnalyticSupportExecutionMode(str, Enum):
    CUDA_PRODUCTION = "cuda_production"
    CPU_REFERENCE_TEST_ONLY = "cpu_reference_test_only"


class V1LiteralJAffineShellAnalyticSupportRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    INVALID_DESIGN = "invalid_design"
    SOURCE_MISMATCH = "source_mismatch"
    LINEAGE_MISMATCH = "lineage_mismatch"
    DEVICE_REQUIRED = "device_required"
    NUMERICAL_REFUSAL = "numerical_refusal"
    MUTATED_CUSTODY = "mutated_custody"
    ACTIVE_LEASE = "active_lease"
    EVICTED = "evicted"
    AUTHORITY_VIOLATION = "authority_violation"


class V1LiteralJAffineShellAnalyticSupportRefusal(RuntimeError):
    def __init__(
        self,
        code: V1LiteralJAffineShellAnalyticSupportRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1LiteralJAffineShellAnalyticSupportRefusalCode:
            raise TypeError("affine-shell support refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1LiteralJAffineShellAnalyticSupportRefusalCode,
    message: str,
) -> NoReturn:
    raise V1LiteralJAffineShellAnalyticSupportRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1LiteralJAffineShellAnalyticSupportRefusalCode.INVALID_TYPE,
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
            V1LiteralJAffineShellAnalyticSupportRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative exact integer",
        )
    return int(value)


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        _refuse(
            V1LiteralJAffineShellAnalyticSupportRefusalCode.INVALID_DESIGN,
            f"{name} must be positive",
        )
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1LiteralJAffineShellAnalyticSupportRefusalCode.INVALID_TYPE,
            f"{name} must be one real scalar",
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _refuse(
            V1LiteralJAffineShellAnalyticSupportRefusalCode.NUMERICAL_REFUSAL,
            f"{name} must be finite and nonnegative",
        )
    return result


def _identity_axis(value: object, *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _refuse(
            V1LiteralJAffineShellAnalyticSupportRefusalCode.INVALID_TYPE,
            f"{name} must be one ordered identity axis",
        )
    result = tuple(_identity(item, name=f"{name} item") for item in value)
    if not result or len(result) != len(set(result)):
        _refuse(
            V1LiteralJAffineShellAnalyticSupportRefusalCode.INVALID_DESIGN,
            f"{name} must be nonempty and unique",
        )
    return result


def _bound_axis(value: object, *, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _refuse(
            V1LiteralJAffineShellAnalyticSupportRefusalCode.INVALID_TYPE,
            f"{name} must be one ordered scalar axis",
        )
    return tuple(
        _finite_nonnegative(item, name=f"{name} item") for item in value
    )


def _down(value: float) -> float:
    return math.nextafter(float(value), -math.inf)


def _up(value: float) -> float:
    return math.nextafter(float(value), math.inf)


def _mul_up(left: float, right: float) -> float:
    value = float(left) * float(right)
    if not math.isfinite(value):
        _refuse(
            V1LiteralJAffineShellAnalyticSupportRefusalCode.NUMERICAL_REFUSAL,
            "outward multiplication overflowed",
        )
    return _up(value)


def _add_up(left: float, right: float) -> float:
    value = float(left) + float(right)
    if not math.isfinite(value):
        _refuse(
            V1LiteralJAffineShellAnalyticSupportRefusalCode.NUMERICAL_REFUSAL,
            "outward addition overflowed",
        )
    return _up(value)


def _div_up(numerator: float, denominator: float) -> float:
    if not math.isfinite(denominator) or denominator <= 0.0:
        _refuse(
            V1LiteralJAffineShellAnalyticSupportRefusalCode.NUMERICAL_REFUSAL,
            "outward division left its positive finite regime",
        )
    value = float(numerator) / float(denominator)
    if not math.isfinite(value):
        _refuse(
            V1LiteralJAffineShellAnalyticSupportRefusalCode.NUMERICAL_REFUSAL,
            "outward division overflowed",
        )
    return _up(value)


def _gamma_up(operation_count: int) -> float:
    count = _positive_int(operation_count, name="operation_count")
    product = _mul_up(float(count), _F64_UNIT_ROUNDOFF)
    if product >= 0.5:
        _refuse(
            V1LiteralJAffineShellAnalyticSupportRefusalCode.NUMERICAL_REFUSAL,
            "route-scale reduction exceeds the finite gamma regime",
        )
    return _div_up(product, _down(1.0 - product))


def _outward_nonnegative_dot_upper(
    route_abs_bound: Any,
    z_outside: Any,
) -> tuple[float, int]:
    """Return a scalar upper bound for the exact dot of binary64 inputs.

    The CUDA reduction supplies the center.  A standard-model gamma envelope
    covers one product and the reduction work per route; an explicit minimum-
    subnormal allowance covers gradual-underflow loss.  Only the scalar center
    and one boolean support bit cross D2H.
    """

    terms = torch.abs(route_abs_bound) * torch.abs(z_outside)
    if not bool(torch.isfinite(terms).all().item()):
        _refuse(
            V1LiteralJAffineShellAnalyticSupportRefusalCode.NUMERICAL_REFUSAL,
            "analytic support products overflowed",
        )
    center_tensor = torch.sum(terms)
    if not bool(torch.isfinite(center_tensor).item()):
        _refuse(
            V1LiteralJAffineShellAnalyticSupportRefusalCode.NUMERICAL_REFUSAL,
            "analytic support reduction overflowed",
        )
    possible_nonzero = bool(
        ((route_abs_bound != 0.0) & (z_outside != 0.0)).any().item()
    )
    operation_count = 2 * int(route_abs_bound.numel()) + 1
    center = float(center_tensor.item())
    if not possible_nonzero:
        return 0.0, operation_count
    gamma = _gamma_up(operation_count)
    normal_upper = _div_up(center, _down(1.0 - gamma))
    underflow_upper = _mul_up(float(operation_count), _F64_MIN_SUBNORMAL)
    return _add_up(normal_upper, underflow_upper), operation_count


def _tensor_invariant(value: Any) -> tuple[object, ...]:
    return (
        id(value),
        int(value._version),
        int(value.data_ptr()),
        tuple(int(item) for item in value.shape),
        str(value.dtype),
        str(value.device),
    )


@dataclass(frozen=True, slots=True)
class V1LiteralJAffineShellAnalyticSupportReceipt:
    alexnet_support_result_hash: str
    alexnet_support_derivation_receipt_hash: str
    generic_support_source_receipt_hash: str
    affine_direction_receipt_hash: str
    affine_direction_source_hash: str
    affine_direction_current_source_receipt_hash: str
    affine_direction_preoutcome_source_registration_hash: str | None
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    population_projection_registration_hash: str
    population_alpha_lineage_hash: str
    literal_j_abs_bound_source_hash: str
    analytic_source_owner_identity_hash: str
    backend_execution_receipt_hash: str
    shell_freeze_receipt_hash: str
    shell_indices: tuple[int, ...]
    shell_route_ids: tuple[str, ...]
    direction_ids: tuple[str, ...]
    literal_j_stream_source_hash: str
    outside_abs_upper_bound: float
    shell_abs_upper_bounds: tuple[float, ...]
    support_values: tuple[float, ...]
    support_value_hash: str
    reduction_operation_count: int
    retained_float64_count: int
    retained_device_bytes: int
    execution_mode: str
    array_free_durable_receipt: bool = True
    exact_typed_parent_join: bool = True
    source_bound_analytic_support: bool = True
    preoutcome_source_registration_bound: bool = False
    outward_fp64_outside_reduction: bool = True
    bounded_scalar_shell_d2h_only: bool = True
    route_scale_d2h_performed: bool = False
    dense_basis_bank_materialized: bool = False
    caller_authored_hash_or_bound_accepted: bool = False
    population_authority: bool = False
    alpha_authority: bool = False
    loewner_authority: bool = False
    d_stability_authority: bool = False
    c_stream_authority: bool = False
    calibration_k_authority: bool = False
    paired_oracle_o_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    equation: str = LITERAL_J_AFFINE_SHELL_ANALYTIC_SUPPORT_EQUATION
    algorithm: str = LITERAL_J_AFFINE_SHELL_ANALYTIC_SUPPORT_ALGORITHM
    authority_scope: str = LITERAL_J_AFFINE_SHELL_ANALYTIC_SUPPORT_AUTHORITY
    schema: str = LITERAL_J_AFFINE_SHELL_ANALYTIC_SUPPORT_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _ISSUER_TOKEN:
            raise PermissionError("affine-shell support receipts require their issuer")
        for name in (
            "alexnet_support_result_hash",
            "alexnet_support_derivation_receipt_hash",
            "generic_support_source_receipt_hash",
            "affine_direction_receipt_hash",
            "affine_direction_source_hash",
            "affine_direction_current_source_receipt_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "population_projection_registration_hash",
            "population_alpha_lineage_hash",
            "literal_j_abs_bound_source_hash",
            "analytic_source_owner_identity_hash",
            "backend_execution_receipt_hash",
            "shell_freeze_receipt_hash",
            "literal_j_stream_source_hash",
            "support_value_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "affine_direction_preoutcome_source_registration_hash",
            _optional_identity(
                self.affine_direction_preoutcome_source_registration_hash,
                name="affine_direction_preoutcome_source_registration_hash",
            ),
        )
        generation = _nonnegative_int(
            self.proposal_generation, name="proposal_generation"
        )
        routes = _positive_int(self.route_count, name="route_count")
        operations = _positive_int(
            self.reduction_operation_count, name="reduction_operation_count"
        )
        retained = _positive_int(
            self.retained_float64_count, name="retained_float64_count"
        )
        indices = tuple(
            _nonnegative_int(item, name="shell index") for item in self.shell_indices
        )
        shell_ids = _identity_axis(self.shell_route_ids, name="shell route IDs")
        directions = _identity_axis(self.direction_ids, name="direction IDs")
        shell_bounds = _bound_axis(
            self.shell_abs_upper_bounds, name="shell absolute bounds"
        )
        values = _bound_axis(self.support_values, name="support values")
        outside = _finite_nonnegative(
            self.outside_abs_upper_bound, name="outside absolute bound"
        )
        if (
            not 1 <= len(indices) <= 8
            or len(indices) != len(set(indices))
            or any(item >= routes for item in indices)
        ):
            _refuse(
                V1LiteralJAffineShellAnalyticSupportRefusalCode.INVALID_DESIGN,
                "shell axes do not form one bounded in-route family",
            )
        if len(indices) != len(shell_ids) or len(indices) != len(shell_bounds):
            _refuse(
                V1LiteralJAffineShellAnalyticSupportRefusalCode.INVALID_DESIGN,
                "shell indices, route IDs, and support bounds must align",
            )
        if (
            len(directions) != len(indices) + 1
            or directions
            != ("literal-j:outside",)
            + tuple(f"literal-j:shell-basis:{item}" for item in shell_ids)
            or values != (outside,) + shell_bounds
            or retained != len(values)
            or self.retained_device_bytes != 2 * retained * _F64_BYTES
            or operations != 2 * routes + 1
            or self.support_value_hash
            != content_hash(
                {
                    "direction_ids": directions,
                    "support_values": values,
                    "equation": self.equation,
                },
                prefix=_PREFIX + "values-",
            )
        ):
            _refuse(
                V1LiteralJAffineShellAnalyticSupportRefusalCode.INVALID_DESIGN,
                "support value axis or memory arithmetic does not replay",
            )
        try:
            mode = V1LiteralJAffineShellAnalyticSupportExecutionMode(
                self.execution_mode
            )
        except ValueError as error:
            raise V1LiteralJAffineShellAnalyticSupportRefusal(
                V1LiteralJAffineShellAnalyticSupportRefusalCode.INVALID_DESIGN,
                "execution mode is not registered",
            ) from error
        positive = (
            self.array_free_durable_receipt,
            self.exact_typed_parent_join,
            self.source_bound_analytic_support,
            self.outward_fp64_outside_reduction,
            self.bounded_scalar_shell_d2h_only,
        )
        closed = (
            self.route_scale_d2h_performed,
            self.dense_basis_bank_materialized,
            self.caller_authored_hash_or_bound_accepted,
            self.population_authority,
            self.alpha_authority,
            self.loewner_authority,
            self.d_stability_authority,
            self.c_stream_authority,
            self.calibration_k_authority,
            self.paired_oracle_o_authority,
            self.commit_authority,
            self.terminal_authority,
            self.performance_authority,
            self.pareto_authority,
        )
        if (
            type(self.preoutcome_source_registration_bound) is not bool
            or self.preoutcome_source_registration_bound
            != (self.affine_direction_preoutcome_source_registration_hash is not None)
        ):
            _refuse(
                V1LiteralJAffineShellAnalyticSupportRefusalCode.AUTHORITY_VIOLATION,
                "pre-outcome support flag must match direction registration custody",
            )
        if (
            any(type(item) is not bool or item is not True for item in positive)
            or any(type(item) is not bool or item is not False for item in closed)
            or self.equation != LITERAL_J_AFFINE_SHELL_ANALYTIC_SUPPORT_EQUATION
            or self.algorithm != LITERAL_J_AFFINE_SHELL_ANALYTIC_SUPPORT_ALGORITHM
            or self.authority_scope != LITERAL_J_AFFINE_SHELL_ANALYTIC_SUPPORT_AUTHORITY
            or self.schema != LITERAL_J_AFFINE_SHELL_ANALYTIC_SUPPORT_SCHEMA
        ):
            _refuse(
                V1LiteralJAffineShellAnalyticSupportRefusalCode.AUTHORITY_VIOLATION,
                "affine-shell support receipt crossed its analytic-only boundary",
            )
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "reduction_operation_count", operations)
        object.__setattr__(self, "retained_float64_count", retained)
        object.__setattr__(self, "shell_indices", indices)
        object.__setattr__(self, "shell_route_ids", shell_ids)
        object.__setattr__(self, "direction_ids", directions)
        object.__setattr__(self, "outside_abs_upper_bound", outside)
        object.__setattr__(self, "shell_abs_upper_bounds", shell_bounds)
        object.__setattr__(self, "support_values", values)
        object.__setattr__(self, "execution_mode", mode.value)
        self.assert_integrity()

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            }
        )

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_PREFIX + "receipt-")
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1LiteralJAffineShellAnalyticSupportRefusalCode.MUTATED_CUSTODY,
                "affine-shell support receipt mutated",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1LiteralJAffineShellAnalyticSupportRefusalCode.SOURCE_MISMATCH,
                "affine-shell support receipt hash differs",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class V1LiteralJAffineShellAnalyticSupportLease:
    __slots__ = ("_owner", "_token", "_closed", "receipt_hash")
    __hash__ = None

    def __init__(
        self,
        owner: "V1LiteralJAffineShellAnalyticSupportDeviceHandle",
        token: object,
    ) -> None:
        self._owner = owner
        self._token = token
        self._closed = False
        self.receipt_hash = owner.receipt.receipt_hash

    def support_vector(self) -> Any:
        if self._closed:
            _refuse(
                V1LiteralJAffineShellAnalyticSupportRefusalCode.EVICTED,
                "affine-shell support lease is closed",
            )
        if self._owner.receipt.receipt_hash != self.receipt_hash:
            _refuse(
                V1LiteralJAffineShellAnalyticSupportRefusalCode.SOURCE_MISMATCH,
                "affine-shell support receipt changed while leased",
            )
        return self._owner._require_live()

    def close(self) -> None:
        if self._closed:
            return
        self._owner._close_lease(self._token)
        self._closed = True

    def __enter__(self) -> "V1LiteralJAffineShellAnalyticSupportLease":
        self.support_vector()
        return self

    def __exit__(self, *_: object) -> bool:
        self.close()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("affine-shell support leases are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("affine-shell support leases are nonserializable")


class V1LiteralJAffineShellAnalyticSupportDeviceHandle:
    __slots__ = (
        "_support",
        "_guard",
        "_invariant",
        "_tokens",
        "_receipt",
    )

    def __init__(
        self,
        *,
        support: Any,
        receipt: V1LiteralJAffineShellAnalyticSupportReceipt,
        _issuance_token: object | None = None,
    ) -> None:
        if _issuance_token is not _ISSUER_TOKEN:
            raise PermissionError("affine-shell support handles require their issuer")
        if type(receipt) is not V1LiteralJAffineShellAnalyticSupportReceipt:
            raise TypeError("affine-shell support handle requires its exact receipt")
        receipt.assert_integrity()
        if (
            torch is None
            or type(support) is not torch.Tensor
            or support.dtype != torch.float64
            or tuple(support.shape) != (receipt.retained_float64_count,)
            or not support.is_contiguous()
            or bool(support.requires_grad)
            or not bool(torch.isfinite(support).all().item())
            or bool((support < 0.0).any().item())
        ):
            _refuse(
                V1LiteralJAffineShellAnalyticSupportRefusalCode.INVALID_TYPE,
                "support custody requires one finite nonnegative contiguous FP64 vector",
            )
        self._support = support.detach().clone(memory_format=torch.contiguous_format)
        self._guard = self._support.detach().clone()
        self._invariant = _tensor_invariant(self._support)
        self._tokens: set[object] = set()
        self._receipt = receipt
        self._require_live()

    def _require_live(self) -> Any:
        if self._support is None:
            _refuse(
                V1LiteralJAffineShellAnalyticSupportRefusalCode.EVICTED,
                "affine-shell support custody was evicted",
            )
        if (
            _tensor_invariant(self._support) != self._invariant
            or not bool(torch.equal(self._support, self._guard))
        ):
            _refuse(
                V1LiteralJAffineShellAnalyticSupportRefusalCode.MUTATED_CUSTODY,
                "affine-shell support vector changed after sealing",
            )
        self._receipt.assert_integrity()
        return self._support

    @property
    def receipt(self) -> V1LiteralJAffineShellAnalyticSupportReceipt:
        self._require_live()
        return self._receipt

    @property
    def is_evicted(self) -> bool:
        return self._support is None

    @property
    def device(self) -> Any:
        return self._require_live().device

    def borrow(self) -> V1LiteralJAffineShellAnalyticSupportLease:
        self._require_live()
        token = object()
        self._tokens.add(token)
        return V1LiteralJAffineShellAnalyticSupportLease(self, token)

    def _close_lease(self, token: object) -> None:
        if token not in self._tokens:
            _refuse(
                V1LiteralJAffineShellAnalyticSupportRefusalCode.SOURCE_MISMATCH,
                "affine-shell support lease token is stale",
            )
        self._require_live()
        self._tokens.remove(token)

    def evict(self) -> None:
        self._require_live()
        if self._tokens:
            _refuse(
                V1LiteralJAffineShellAnalyticSupportRefusalCode.ACTIVE_LEASE,
                "cannot evict affine-shell support under an active lease",
            )
        self._support = None
        self._guard = None
        self._invariant = ()

    def __reduce__(self) -> NoReturn:
        raise TypeError("affine-shell support handles are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("affine-shell support handles are nonserializable")


def _assert_parent_join(
    *,
    analytic_support: V1AlexNetLiteralJAnalyticSupportSourceResult,
    affine_directions: V1LiteralJAffineShellDirectionDeviceHandle,
) -> tuple[Any, Any, Any]:
    if (
        type(analytic_support) is not V1AlexNetLiteralJAnalyticSupportSourceResult
        or type(affine_directions)
        is not V1LiteralJAffineShellDirectionDeviceHandle
    ):
        _refuse(
            V1LiteralJAffineShellAnalyticSupportRefusalCode.INVALID_TYPE,
            "affine-shell support requires exact analytic and direction owners",
        )
    derivation = analytic_support.derivation_receipt
    generic_result = analytic_support.analytic_support_source
    if (
        type(generic_result)
        is not V1LiteralJResidualAnalyticSupportSourceCUDAResult
        or type(generic_result.receipt)
        is not V1LiteralJResidualAnalyticSupportSourceCUDAReceipt
        or type(generic_result.source_handle)
        is not V1LiteralJResidualAnalyticSupportSourceCUDAHandle
    ):
        _refuse(
            V1LiteralJAffineShellAnalyticSupportRefusalCode.INVALID_TYPE,
            "AlexNet support result lost its exact generic source custody",
        )
    source_receipt = generic_result.receipt
    source_handle = generic_result.source_handle
    derivation.assert_integrity()
    analytic_support.hbm_forecast.assert_integrity(derivation.hbm_forecast_hash)
    source_receipt.assert_integrity()
    source_handle.assert_live()
    direction_receipt = affine_directions.receipt
    coordinates = (
        ("model identity", direction_receipt.model_identity_hash, source_receipt.model_identity_hash),
        ("chart identity", direction_receipt.chart_identity_hash, source_receipt.chart_identity_hash),
        ("current mask", direction_receipt.current_mask_hash, source_receipt.current_mask_hash),
        ("proposal generation", direction_receipt.proposal_generation, source_receipt.proposal_generation),
        ("route axis", direction_receipt.route_axis_hash, source_receipt.route_axis_hash),
        ("route-ID axis", direction_receipt.route_id_axis_hash, source_receipt.route_id_axis_hash),
        ("route count", direction_receipt.route_count, source_receipt.route_count),
    )
    if any(left != right for _name, left, right in coordinates):
        failed = next(name for name, left, right in coordinates if left != right)
        _refuse(
            V1LiteralJAffineShellAnalyticSupportRefusalCode.LINEAGE_MISMATCH,
            f"affine direction and analytic support differ in {failed}",
        )
    if (
        derivation.receipt_hash
        != source_receipt.analytic_bound_derivation_receipt_hash
        or derivation.model_identity_hash != source_receipt.model_identity_hash
        or derivation.current_mask_hash != source_receipt.current_mask_hash
        or derivation.proposal_generation != source_receipt.proposal_generation
        or derivation.route_axis_hash != source_receipt.route_axis_hash
        or derivation.route_id_axis_hash != source_receipt.route_id_axis_hash
        or derivation.route_count != source_receipt.route_count
        or derivation.literal_j_abs_bound_source_hash
        != source_receipt.literal_j_abs_bound_source_hash
        or derivation.source_owner_identity_hash
        != source_receipt.source_owner_identity_hash
        or derivation.backend_execution_receipt_hash
        != source_receipt.backend_execution_receipt_hash
        or direction_receipt.alpha_lineage_hash
        != derivation.population_alpha_lineage_hash
    ):
        _refuse(
            V1LiteralJAffineShellAnalyticSupportRefusalCode.LINEAGE_MISMATCH,
            "analytic support parents or population-alpha lineage do not close",
        )
    return derivation, source_receipt, source_handle


def issue_v1_literal_j_affine_shell_analytic_support_cuda(
    *,
    analytic_support: V1AlexNetLiteralJAnalyticSupportSourceResult,
    affine_directions: V1LiteralJAffineShellDirectionDeviceHandle,
    execution_mode: V1LiteralJAffineShellAnalyticSupportExecutionMode,
) -> V1LiteralJAffineShellAnalyticSupportDeviceHandle:
    """Derive and seal the bounded analytic support vector for one family."""

    if torch is None or type(execution_mode) is not V1LiteralJAffineShellAnalyticSupportExecutionMode:
        _refuse(
            V1LiteralJAffineShellAnalyticSupportRefusalCode.INVALID_TYPE,
            "issuance requires PyTorch and one exact execution mode",
        )
    derivation, source_receipt, source_handle = _assert_parent_join(
        analytic_support=analytic_support,
        affine_directions=affine_directions,
    )
    direction_receipt = affine_directions.receipt
    if direction_receipt.execution_mode != execution_mode.value:
        _refuse(
            V1LiteralJAffineShellAnalyticSupportRefusalCode.DEVICE_REQUIRED,
            "analytic-support and affine-direction execution modes differ",
        )

    with source_handle.borrow() as source_lease, affine_directions.borrow() as direction_lease:
        route_bound = source_lease.literal_j_abs_bound
        z_outside, shell_indices = direction_lease.structured_tensors()
        if (
            type(route_bound) is not torch.Tensor
            or type(z_outside) is not torch.Tensor
            or type(shell_indices) is not torch.Tensor
            or route_bound.dtype != torch.float64
            or z_outside.dtype != torch.float64
            or shell_indices.dtype != torch.int64
            or tuple(route_bound.shape) != (source_receipt.route_count,)
            or tuple(z_outside.shape) != (source_receipt.route_count,)
            or tuple(shell_indices.shape) != (len(direction_receipt.shell_indices),)
            or route_bound.device != z_outside.device
            or route_bound.device != shell_indices.device
            or bool(route_bound.requires_grad)
            or bool(z_outside.requires_grad)
            or not route_bound.is_contiguous()
            or not z_outside.is_contiguous()
            or not shell_indices.is_contiguous()
        ):
            _refuse(
                V1LiteralJAffineShellAnalyticSupportRefusalCode.SOURCE_MISMATCH,
                "live support and direction tensors do not share one exact route axis/device",
            )
        if (
            not bool(torch.isfinite(route_bound).all().item())
            or bool((route_bound < 0.0).any().item())
            or not bool(torch.isfinite(z_outside).all().item())
        ):
            _refuse(
                V1LiteralJAffineShellAnalyticSupportRefusalCode.NUMERICAL_REFUSAL,
                "live analytic support or outside direction is invalid",
            )
        if tuple(int(item) for item in shell_indices.detach().cpu().tolist()) != tuple(
            direction_receipt.shell_indices
        ):
            _refuse(
                V1LiteralJAffineShellAnalyticSupportRefusalCode.SOURCE_MISMATCH,
                "live shell index vector differs from its source-bound receipt",
            )
        if (
            execution_mode
            is V1LiteralJAffineShellAnalyticSupportExecutionMode.CUDA_PRODUCTION
            and route_bound.device.type != "cuda"
        ):
            _refuse(
                V1LiteralJAffineShellAnalyticSupportRefusalCode.DEVICE_REQUIRED,
                "production affine-shell support must remain CUDA-resident",
            )
        if (
            execution_mode
            is V1LiteralJAffineShellAnalyticSupportExecutionMode.CPU_REFERENCE_TEST_ONLY
            and route_bound.device.type != "cpu"
        ):
            _refuse(
                V1LiteralJAffineShellAnalyticSupportRefusalCode.DEVICE_REQUIRED,
                "CPU reference affine-shell support requires CPU-resident inputs",
            )
        outside_bound, operation_count = _outward_nonnegative_dot_upper(
            route_bound,
            z_outside,
        )
        shell_bound_tensor = route_bound.index_select(0, shell_indices).contiguous()
        shell_bounds = tuple(
            float(item) for item in shell_bound_tensor.detach().cpu().tolist()
        )
        support_values = (outside_bound,) + shell_bounds
        support = torch.cat(
            (
                torch.tensor(
                    (outside_bound,), dtype=torch.float64, device=route_bound.device
                ),
                shell_bound_tensor,
            )
        ).contiguous()

    value_hash = content_hash(
        {
            "direction_ids": direction_receipt.direction_ids,
            "support_values": support_values,
            "equation": LITERAL_J_AFFINE_SHELL_ANALYTIC_SUPPORT_EQUATION,
        },
        prefix=_PREFIX + "values-",
    )
    receipt = V1LiteralJAffineShellAnalyticSupportReceipt(
        alexnet_support_result_hash=analytic_support.result_hash,
        alexnet_support_derivation_receipt_hash=derivation.receipt_hash,
        generic_support_source_receipt_hash=source_receipt.receipt_hash,
        affine_direction_receipt_hash=direction_receipt.receipt_hash,
        affine_direction_source_hash=direction_receipt.direction_source_hash,
        affine_direction_current_source_receipt_hash=(
            direction_receipt.source_receipt_hash
        ),
        affine_direction_preoutcome_source_registration_hash=(
            direction_receipt.preoutcome_source_registration_hash
        ),
        model_identity_hash=source_receipt.model_identity_hash,
        chart_identity_hash=source_receipt.chart_identity_hash,
        current_mask_hash=source_receipt.current_mask_hash,
        proposal_generation=source_receipt.proposal_generation,
        route_axis_hash=source_receipt.route_axis_hash,
        route_id_axis_hash=source_receipt.route_id_axis_hash,
        route_count=source_receipt.route_count,
        population_projection_registration_hash=(
            derivation.population_projection_registration_hash
        ),
        population_alpha_lineage_hash=derivation.population_alpha_lineage_hash,
        literal_j_abs_bound_source_hash=source_receipt.literal_j_abs_bound_source_hash,
        analytic_source_owner_identity_hash=source_receipt.source_owner_identity_hash,
        backend_execution_receipt_hash=source_receipt.backend_execution_receipt_hash,
        shell_freeze_receipt_hash=direction_receipt.shell_freeze_receipt_hash,
        shell_indices=direction_receipt.shell_indices,
        shell_route_ids=direction_receipt.shell_route_ids,
        direction_ids=direction_receipt.direction_ids,
        literal_j_stream_source_hash=direction_receipt.literal_j_stream_source_hash,
        outside_abs_upper_bound=outside_bound,
        shell_abs_upper_bounds=shell_bounds,
        support_values=support_values,
        support_value_hash=value_hash,
        reduction_operation_count=operation_count,
        retained_float64_count=len(support_values),
        retained_device_bytes=2 * len(support_values) * _F64_BYTES,
        execution_mode=execution_mode.value,
        preoutcome_source_registration_bound=(
            direction_receipt.preoutcome_source_registration_bound
        ),
        _issuance_token=_ISSUER_TOKEN,
    )
    return V1LiteralJAffineShellAnalyticSupportDeviceHandle(
        support=support,
        receipt=receipt,
        _issuance_token=_ISSUER_TOKEN,
    )


__all__ = [
    "LITERAL_J_AFFINE_SHELL_ANALYTIC_SUPPORT_SCHEMA",
    "LITERAL_J_AFFINE_SHELL_ANALYTIC_SUPPORT_AUTHORITY",
    "LITERAL_J_AFFINE_SHELL_ANALYTIC_SUPPORT_ALGORITHM",
    "LITERAL_J_AFFINE_SHELL_ANALYTIC_SUPPORT_EQUATION",
    "V1LiteralJAffineShellAnalyticSupportExecutionMode",
    "V1LiteralJAffineShellAnalyticSupportRefusalCode",
    "V1LiteralJAffineShellAnalyticSupportRefusal",
    "V1LiteralJAffineShellAnalyticSupportReceipt",
    "V1LiteralJAffineShellAnalyticSupportLease",
    "V1LiteralJAffineShellAnalyticSupportDeviceHandle",
    "issue_v1_literal_j_affine_shell_analytic_support_cuda",
]
