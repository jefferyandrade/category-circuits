"""Fused, authority-inert source for a native-fiber signed-work plane.

For a pre-frozen preferred action ``p`` and one registered D_W2 context group,
this module streams the *same live corrected-A route chunks* already bound by
``V1SignedWorkVariableShellSourceChunkBinding``.  Complete native fibers permit
the subgradient sign of ``(A p)_xf`` and the route coefficients

    c_j = E_x[A_xj y_xf(j)] / number_of_cuts

to be accumulated in one route pass.  No context-by-route row is retained.
An uncertified sign is replaced by zero and its entire preferred response is
charged to a scalar deficit.  Binary64 coefficient-construction error is also
charged in L1, yielding, for every Boolean action ``z``,

    F(z) - F(p) >= c_center @ (z - p) - preferred_plane_deficit_upper.

The only private import, ``_tensor_source_identity_hash``, is deliberately at
the source-custody boundary: it verifies that the corrected tensor is exactly
the live tensor named by the issuer-only chunk binding.  It is process-local,
never a content hash or durable replay authority, and performs no D2H.

This source is intentionally unable to certify population separation, Q, C,
K, O, commits, terminal state, or Pareto claims.  A later statistical module
must turn independently registered block margins into a valid confidence or
e-process statement.  In particular, raw D_W2 context axes never imply that
the full ImageNet train population is owned: production use requires a later
typed join to ``CompleteImageNetPopulationPairReceipt`` (or its canonical
successor), which this source deliberately does not accept or manufacture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
import math
from numbers import Integral, Real
from typing import Any, Final, NoReturn, Sequence

import numpy as np

from mechanistic_systems.contracts.hard_mask_actions import HardMask

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from .contracts import content_hash
from .modern_v1_candidate_composition import V1PackedRemovalAction
from .modern_v1_native_fiber_axis import V1NativeDestinationFiberAxisReceipt
from .modern_v1_sampled_residual_population_measure import (
    V1SampledResidualPopulationMeasureDesign,
    V1SampledResidualPopulationWeightReceipt,
)
from .modern_v1_signed_work_variable_shell_source_cuda import (
    V1SignedWorkVariableShellSourceChunkBinding,
    V1SignedWorkVariableShellSourceRegistration,
    _tensor_source_identity_hash,
)

try:  # pragma: no cover - runtime-image dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


_PREFIX: Final = "partition-free-v1-block-support-source-"
_SCHEMA: Final = "partition_free_v1_signed_work_block_support_source_v1"
_REGISTERED_MEASURE_SCHEMA: Final = (
    "partition_free_v1_signed_work_block_support_source_v2"
)
_MULTI_SEGMENT_SCHEMA: Final = (
    "partition_free_v1_signed_work_block_support_multi_segment_source_v1"
)
_MULTI_SEGMENT_REGISTERED_MEASURE_SCHEMA: Final = (
    "partition_free_v1_signed_work_block_support_multi_segment_source_v2"
)
_MEASURE_CONTRACT_SCHEMA: Final = (
    "partition_free_v1_signed_work_block_support_measure_contract_v1"
)
_FORECAST_SCHEMA: Final = (
    "partition_free_v1_signed_work_block_support_source_hbm_forecast_v2"
)
_RECEIPT_SCHEMA: Final = "partition_free_v1_signed_work_block_support_source_receipt_v2"
_MULTI_SEGMENT_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_signed_work_block_support_source_receipt_v3"
)
_REGISTERED_MEASURE_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_signed_work_block_support_source_receipt_v4"
)
_F64_EPS: Final = 2.0**-52
_F64_MIN: Final = math.nextafter(0.0, math.inf)
_ISSUER: Final = object()
_FORECAST_ISSUER: Final = object()
_RECEIPT_ISSUER: Final = object()
_HANDLE_ISSUER: Final = object()
_ACTION_BINDING_ISSUER: Final = object()
_MULTI_SEGMENT_REGISTRATION_ISSUER: Final = object()
_MEASURE_CONTRACT_ISSUER: Final = object()


class V1SignedWorkBlockSupportSourceRole(str, Enum):
    D_FIXED = "d_fixed"
    RESIDUAL_BLOCK = "residual_equal_block"


class V1SignedWorkBlockSupportExecutionMode(str, Enum):
    CUDA_PRODUCTION = "cuda_production"
    CPU_REFERENCE_TEST_ONLY = "cpu_reference_test_only"


class V1SignedWorkBlockSupportMeasureKind(str, Enum):
    """The estimand measure carried by a support plane.

    ``LEGACY_UNIFORM_LOCAL_EMPIRICAL`` preserves the historical conditional
    group mean as typed replay provenance.  It is deliberately not canonical
    full-population signed-F/D evidence.  ``FULL_POPULATION_STRATIFIED`` binds
    the exact D1-plus-residual finite-population decomposition.
    """

    LEGACY_UNIFORM_LOCAL_EMPIRICAL = "legacy_uniform_local_empirical"
    FULL_POPULATION_STRATIFIED = "full_population_stratified_D1_residual"


class V1SignedWorkBlockSupportRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    INVALID_DESIGN = "invalid_design"
    STALE_IDENTITY = "stale_identity"
    AXIS_MISMATCH = "axis_mismatch"
    ORDER_MISMATCH = "order_mismatch"
    PARTIAL_SOURCE = "partial_source"
    NONFINITE_SOURCE = "nonfinite_source"
    NUMERICAL_REFUSAL = "numerical_refusal"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    DETERMINISM_DISABLED = "determinism_disabled"
    MUTATED_CUSTODY = "mutated_custody"
    ACTIVE_LEASE = "active_lease"
    EVICTED = "evicted"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SignedWorkBlockSupportRefusal(RuntimeError):
    def __init__(self, code: V1SignedWorkBlockSupportRefusalCode, message: str) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(code: V1SignedWorkBlockSupportRefusalCode, message: str) -> NoReturn:
    raise V1SignedWorkBlockSupportRefusal(code, message)


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE, f"{name} must be nonempty"
        )
    return value


def _integer(value: object, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE, f"{name} must be integral"
        )
    answer = int(value)
    if answer < (1 if positive else 0):
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.INVALID_DESIGN,
            f"{name} is out of range",
        )
    return answer


def _axis(value: Sequence[str], name: str) -> tuple[str, ...]:
    if type(value) not in {tuple, list}:
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
            f"{name} must be a sequence",
        )
    answer = tuple(_text(item, f"{name} item") for item in value)
    if not answer or len(set(answer)) != len(answer):
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.AXIS_MISMATCH,
            f"{name} is empty or nonunique",
        )
    return answer


def _weights(value: Sequence[float], count: int) -> tuple[float, ...]:
    if type(value) not in {tuple, list} or len(value) != count:
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.AXIS_MISMATCH, "weights do not align"
        )
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) or item < 0.0 for item in result):
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.INVALID_DESIGN,
            "weights must be finite and nonnegative",
        )
    if not math.isclose(math.fsum(result), 1.0, rel_tol=0.0, abs_tol=128 * _F64_EPS):
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.INVALID_DESIGN,
            "weights must sum to one",
        )
    return result


def _stratum_axis(value: object, count: int) -> tuple[str, ...]:
    if type(value) is not tuple or len(value) != count:
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.AXIS_MISMATCH,
            "context stratum axis does not align",
        )
    return tuple(_text(item, "context stratum id") for item in value)


def _count_axis(value: object, name: str) -> tuple[int, ...]:
    if type(value) is not tuple:
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
            f"{name} must be an immutable count axis",
        )
    return tuple(_integer(item, f"{name} item") for item in value)


def _fraction_axis(value: object, count: int) -> tuple[tuple[int, int], ...]:
    if type(value) is not tuple or len(value) != count:
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.AXIS_MISMATCH,
            "exact context-weight fractions do not align",
        )
    result: list[tuple[int, int]] = []
    for item in value:
        if type(item) is not tuple or len(item) != 2:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
                "each exact context weight must be one rational pair",
            )
        numerator = _integer(item[0], "context-weight numerator")
        denominator = _integer(
            item[1], "context-weight denominator", positive=True
        )
        exact = Fraction(numerator, denominator)
        canonical = (exact.numerator, exact.denominator)
        if canonical != (numerator, denominator):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.INVALID_DESIGN,
                "exact context-weight fractions must be reduced",
            )
        result.append(canonical)
    return tuple(result)


def _fraction_center_and_radius(value: Fraction) -> tuple[float, float]:
    """Return deterministic FP64 center plus an outward absolute-error bound."""

    center = float(value)
    if not math.isfinite(center):
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.NUMERICAL_REFUSAL,
            "exact context weight overflowed binary64",
        )
    error = abs(Fraction.from_float(center) - value)
    radius = float(error)
    if Fraction.from_float(radius) < error:
        radius = math.nextafter(radius, math.inf)
    radius = math.nextafter(radius, math.inf)
    if not math.isfinite(radius) or Fraction.from_float(radius) < error:
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.NUMERICAL_REFUSAL,
            "context-weight binary64 radius is not outward",
        )
    return center, radius


def _fraction_axis_hash(values: Sequence[tuple[int, int]]) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_block_support_exact_weight_axis_v1",
            "values": tuple(values),
        },
        prefix=_PREFIX + "exact-context-weight-axis-",
    )


def _stratum_axis_hash(values: Sequence[str]) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_block_support_stratum_axis_v1",
            "values": tuple(values),
        },
        prefix=_PREFIX + "context-stratum-axis-",
    )


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
            f"{name} must be a real scalar",
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.NUMERICAL_REFUSAL,
            f"{name} must be finite and nonnegative",
        )
    return result


def _source_axis_hash(role: str, values: Sequence[str]) -> str:
    # Exact public wire format used by the source binding's active packet axes.
    return content_hash(
        {
            "schema": "partition_free_v1_variable_shell_registered_axis_v1",
            "role": role,
            "values": tuple(values),
        },
        prefix="partition-free-v1-variable-shell-source-axis-",
    )


def _gamma(operations: int) -> float:
    operations = _integer(operations, "operation count")
    product = operations * _F64_EPS
    if product >= 0.125:
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.NUMERICAL_REFUSAL,
            "binary64 error model is unsafe",
        )
    return product / (1.0 - product)


def _up(value: Any) -> Any:
    return torch.nextafter(value, torch.full_like(value, math.inf))


def _down(value: Any) -> Any:
    return torch.nextafter(value, torch.full_like(value, -math.inf))


def _tensor_invariant(value: Any) -> tuple[object, ...]:
    return (
        id(value),
        int(value.data_ptr()),
        int(getattr(value, "_version", -1)),
        tuple(value.shape),
        tuple(value.stride()),
        str(value.dtype),
        str(value.device),
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkBlockSupportMeasureContract:
    """Exact estimand measure plus bounded FP64 kernel representation.

    Exact rational weights, not their binary64 centers, define the measure.
    The centers and outward representation radii exist only so the numerical
    kernel can consume the contract without silently changing that estimand.
    """

    measure_kind: V1SignedWorkBlockSupportMeasureKind
    measure_design_hash: str
    measure_weight_receipt_hash: str
    registered_context_axis_hash: str
    context_stratum_ids: tuple[str, ...]
    exact_context_weight_fractions: tuple[tuple[int, int], ...]
    context_weight_binary64_centers: tuple[float, ...]
    context_weight_binary64_radii_upper: tuple[float, ...]
    population_count: int
    population_stratum_counts: tuple[int, ...]
    d1_stratum_counts: tuple[int, ...]
    residual_sample_stratum_counts: tuple[int, ...]
    d1_context_count: int
    exact_weights_sum_to_one: bool = True
    full_population_uniform_estimand: bool = False
    canonical_f_d_measure_authority: bool = False
    uniform_local_empirical_witness_only: bool = True
    schema: str = _MEASURE_CONTRACT_SCHEMA
    _issuer: object = field(default=None, repr=False, compare=False)
    _residual_measure_design: V1SampledResidualPopulationMeasureDesign | None = field(
        default=None, repr=False, compare=False
    )
    _residual_weight_receipt: V1SampledResidualPopulationWeightReceipt | None = field(
        default=None, repr=False, compare=False
    )
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuer is not _MEASURE_CONTRACT_ISSUER:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
                "block-support measure contracts are factory-issued only",
            )
        if type(self.measure_kind) is not V1SignedWorkBlockSupportMeasureKind:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
                "block-support measure kind has the wrong enum type",
            )
        for name in (
            "measure_design_hash",
            "measure_weight_receipt_hash",
            "registered_context_axis_hash",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        centers = _weights(
            self.context_weight_binary64_centers,
            len(self.context_weight_binary64_centers),
        )
        count = len(centers)
        if count <= 0:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.INVALID_DESIGN,
                "block-support measure cannot be empty",
            )
        strata = _stratum_axis(self.context_stratum_ids, count)
        fractions = _fraction_axis(self.exact_context_weight_fractions, count)
        if type(self.context_weight_binary64_radii_upper) is not tuple or len(
            self.context_weight_binary64_radii_upper
        ) != count:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.AXIS_MISMATCH,
                "context-weight radii do not align",
            )
        radii = tuple(
            _finite_nonnegative(item, "context-weight radius")
            for item in self.context_weight_binary64_radii_upper
        )
        exact_values = tuple(Fraction(*item) for item in fractions)
        expected_numeric = tuple(
            _fraction_center_and_radius(value) for value in exact_values
        )
        if (
            centers != tuple(item[0] for item in expected_numeric)
            or radii != tuple(item[1] for item in expected_numeric)
            or sum(exact_values, Fraction(0)) != 1
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.NUMERICAL_REFUSAL,
                "binary64 weights/radii do not represent the exact normalized measure",
            )
        populations = _count_axis(
            self.population_stratum_counts, "population stratum counts"
        )
        d1 = _count_axis(self.d1_stratum_counts, "D1 stratum counts")
        sampled = _count_axis(
            self.residual_sample_stratum_counts,
            "residual sample stratum counts",
        )
        population_count = _integer(
            self.population_count, "population count", positive=True
        )
        d1_count = _integer(self.d1_context_count, "D1 context count")
        if (
            self.exact_weights_sum_to_one is not True
            or self.schema != _MEASURE_CONTRACT_SCHEMA
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.AUTHORITY_VIOLATION,
                "measure contract lost exact-normalization authority",
            )
        if (
            self.measure_kind
            is V1SignedWorkBlockSupportMeasureKind.LEGACY_UNIFORM_LOCAL_EMPIRICAL
        ):
            expected = Fraction(1, count)
            if (
                strata != ("legacy_local",) * count
                or any(value != expected for value in exact_values)
                or population_count != count
                or populations != (count,)
                or d1
                or sampled
                or d1_count != 0
                or self.full_population_uniform_estimand is not False
                or self.canonical_f_d_measure_authority is not False
                or self.uniform_local_empirical_witness_only is not True
                or self._residual_measure_design is not None
                or self._residual_weight_receipt is not None
            ):
                _refuse(
                    V1SignedWorkBlockSupportRefusalCode.AUTHORITY_VIOLATION,
                    "legacy local empirical measure acquired canonical F/D authority",
                )
        else:
            if (
                type(self._residual_measure_design)
                is not V1SampledResidualPopulationMeasureDesign
                or type(self._residual_weight_receipt)
                is not V1SampledResidualPopulationWeightReceipt
            ):
                _refuse(
                    V1SignedWorkBlockSupportRefusalCode.AUTHORITY_VIOLATION,
                    "canonical population measure requires typed live measure custody",
                )
            measure_design = self._residual_measure_design
            weight_receipt = self._residual_weight_receipt
            design_hash = measure_design.assert_integrity()
            receipt_hash = weight_receipt.assert_integrity()
            if len(populations) != 2 or len(d1) != 2 or len(sampled) != 2:
                _refuse(
                    V1SignedWorkBlockSupportRefusalCode.AXIS_MISMATCH,
                    "canonical full-population measure requires target/complement counts",
                )
            if (
                population_count != sum(populations)
                or d1_count != sum(d1)
                or count != d1_count + sum(sampled)
                or any(d1[index] > populations[index] for index in (0, 1))
                or any(
                    populations[index] > d1[index] and sampled[index] <= 0
                    for index in (0, 1)
                )
                or tuple(strata[:d1_count]).count("target") != d1[0]
                or tuple(strata[:d1_count]).count("complement") != d1[1]
                or tuple(strata[d1_count:]).count("target") != sampled[0]
                or tuple(strata[d1_count:]).count("complement") != sampled[1]
                or any(item not in {"target", "complement"} for item in strata)
                or self.full_population_uniform_estimand is not True
                or self.canonical_f_d_measure_authority is not True
                or self.uniform_local_empirical_witness_only is not False
                or self.measure_design_hash != design_hash
                or self.measure_weight_receipt_hash != receipt_hash
                or weight_receipt.measure_design_hash != design_hash
                or weight_receipt.suffix_registration_hash
                != measure_design.suffix_registration_hash
                or populations != measure_design.population_stratum_counts
                or populations != weight_receipt.population_stratum_counts
                or d1 != measure_design.d1_stratum_counts
                or d1 != weight_receipt.d1_stratum_counts
                or sampled != weight_receipt.cumulative_dw2_stratum_counts
            ):
                _refuse(
                    V1SignedWorkBlockSupportRefusalCode.AUTHORITY_VIOLATION,
                    "canonical stratified measure counts, strata, or authority changed",
                )
            expected_d1 = Fraction(1, population_count)
            expected_residual = (
                Fraction(populations[0] - d1[0], population_count * sampled[0])
                if populations[0] > d1[0]
                else Fraction(0),
                Fraction(populations[1] - d1[1], population_count * sampled[1])
                if populations[1] > d1[1]
                else Fraction(0),
            )
            expected_values = tuple(
                expected_d1
                if index < d1_count
                else expected_residual[0]
                if strata[index] == "target"
                else expected_residual[1]
                for index in range(count)
            )
            if exact_values != expected_values:
                _refuse(
                    V1SignedWorkBlockSupportRefusalCode.INVALID_DESIGN,
                    "context weights are not the exact D1-plus-residual population measure",
                )
        object.__setattr__(self, "context_stratum_ids", strata)
        object.__setattr__(self, "exact_context_weight_fractions", fractions)
        object.__setattr__(self, "context_weight_binary64_centers", centers)
        object.__setattr__(self, "context_weight_binary64_radii_upper", radii)
        object.__setattr__(self, "population_count", population_count)
        object.__setattr__(self, "population_stratum_counts", populations)
        object.__setattr__(self, "d1_stratum_counts", d1)
        object.__setattr__(self, "residual_sample_stratum_counts", sampled)
        object.__setattr__(self, "d1_context_count", d1_count)
        expected_seal = content_hash(
            self._payload(), prefix=_PREFIX + "measure-contract-integrity-"
        )
        if self._integrity_seal and self._integrity_seal != expected_seal:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.MUTATED_CUSTODY,
                "block-support measure contract changed",
            )
        object.__setattr__(self, "_integrity_seal", expected_seal)

    def _payload(self) -> dict[str, object]:
        return {
            item.name: (
                getattr(self, item.name).value
                if isinstance(getattr(self, item.name), Enum)
                else getattr(self, item.name)
            )
            for item in self.__dataclass_fields__.values()
            if not item.name.startswith("_")
        }

    def assert_integrity(self) -> str:
        if self.measure_kind is V1SignedWorkBlockSupportMeasureKind.FULL_POPULATION_STRATIFIED:
            if (
                type(self._residual_measure_design)
                is not V1SampledResidualPopulationMeasureDesign
                or type(self._residual_weight_receipt)
                is not V1SampledResidualPopulationWeightReceipt
                or self._residual_measure_design.assert_integrity()
                != self.measure_design_hash
                or self._residual_weight_receipt.assert_integrity()
                != self.measure_weight_receipt_hash
            ):
                _refuse(
                    V1SignedWorkBlockSupportRefusalCode.MUTATED_CUSTODY,
                    "canonical block-support measure lost typed source custody",
                )
        expected = content_hash(
            self._payload(), prefix=_PREFIX + "measure-contract-integrity-"
        )
        if expected != self._integrity_seal:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.MUTATED_CUSTODY,
                "block-support measure contract changed",
            )
        return expected

    @property
    def contract_hash(self) -> str:
        self.assert_integrity()
        return content_hash(self._payload(), prefix=_PREFIX + "measure-contract-")

    @property
    def context_stratum_axis_hash(self) -> str:
        return _stratum_axis_hash(self.context_stratum_ids)

    @property
    def exact_context_weight_fraction_axis_hash(self) -> str:
        return _fraction_axis_hash(self.exact_context_weight_fractions)


def _issue_legacy_uniform_local_measure(
    *, context_group_id: str, context_ids: Sequence[str]
) -> V1SignedWorkBlockSupportMeasureContract:
    contexts = tuple(context_ids)
    count = len(contexts)
    if count <= 0:
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.INVALID_DESIGN,
            "legacy local empirical measure cannot be empty",
        )
    exact = (Fraction(1, count),) * count
    numeric = tuple(_fraction_center_and_radius(value) for value in exact)
    source = content_hash(
        {
            "schema": "partition_free_v1_legacy_local_block_measure_v1",
            "context_group_id": context_group_id,
            "context_ids": contexts,
        },
        prefix=_PREFIX + "legacy-local-measure-",
    )
    return V1SignedWorkBlockSupportMeasureContract(
        measure_kind=(
            V1SignedWorkBlockSupportMeasureKind.LEGACY_UNIFORM_LOCAL_EMPIRICAL
        ),
        measure_design_hash=source,
        measure_weight_receipt_hash=source,
        registered_context_axis_hash=_source_axis_hash(
            "block_support_context", contexts
        ),
        context_stratum_ids=("legacy_local",) * count,
        exact_context_weight_fractions=tuple(
            (value.numerator, value.denominator) for value in exact
        ),
        context_weight_binary64_centers=tuple(value[0] for value in numeric),
        context_weight_binary64_radii_upper=tuple(value[1] for value in numeric),
        population_count=count,
        population_stratum_counts=(count,),
        d1_stratum_counts=(),
        residual_sample_stratum_counts=(),
        d1_context_count=0,
        _issuer=_MEASURE_CONTRACT_ISSUER,
    )


def _issue_v1_signed_work_block_support_full_population_measure(
    *,
    measure_design: V1SampledResidualPopulationMeasureDesign,
    measure_weight_receipt: V1SampledResidualPopulationWeightReceipt,
    context_ids: Sequence[str],
    context_stratum_ids: Sequence[str],
) -> V1SignedWorkBlockSupportMeasureContract:
    """Bind the exact uniform-train D1-plus-residual signed-F measure.

    Canonical F/D authority is inherited only by replaying the exact typed
    residual-measure owners.  Caller-authored hashes or counts cannot mint it.
    The returned contract retains those owners and replays both integrity
    seals whenever its own integrity is checked.
    """

    if type(measure_design) is not V1SampledResidualPopulationMeasureDesign:
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
            "canonical block measure requires the exact residual measure design",
        )
    if type(measure_weight_receipt) is not V1SampledResidualPopulationWeightReceipt:
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
            "canonical block measure requires the exact residual weight receipt",
        )
    design_hash = measure_design.assert_integrity()
    receipt_hash = measure_weight_receipt.assert_integrity()
    if (
        measure_weight_receipt.measure_design_hash != design_hash
        or measure_weight_receipt.suffix_registration_hash
        != measure_design.suffix_registration_hash
        or measure_weight_receipt.population_stratum_counts
        != measure_design.population_stratum_counts
        or measure_weight_receipt.d1_stratum_counts != measure_design.d1_stratum_counts
        or any(
            measure_weight_receipt.cumulative_dw2_stratum_counts[index]
            > measure_design.residual_stratum_counts[index]
            for index in (0, 1)
        )
    ):
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.STALE_IDENTITY,
            "residual measure design and exact weight receipt do not share one source graph",
        )

    contexts = tuple(context_ids)
    strata = tuple(context_stratum_ids)
    if (
        not contexts
        or len(contexts) != len(strata)
        or any(type(value) is not str or not value for value in contexts)
        or len(set(contexts)) != len(contexts)
    ):
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.AXIS_MISMATCH,
            "canonical measure context identities and strata do not align",
        )
    populations = measure_weight_receipt.population_stratum_counts
    d1 = measure_weight_receipt.d1_stratum_counts
    sampled = measure_weight_receipt.cumulative_dw2_stratum_counts
    population_count = measure_design.population_count
    d1_count = sum(d1)
    d1_weight = Fraction(*measure_weight_receipt.d1_context_weight_fraction)
    residual_weights = tuple(
        Fraction(*item) for item in measure_weight_receipt.dw2_context_weight_fractions
    )
    exact = tuple(
        d1_weight
        if index < d1_count
        else residual_weights[0]
        if strata[index] == "target"
        else residual_weights[1]
        if strata[index] == "complement"
        else Fraction(-1)
        for index in range(len(strata))
    )
    if any(value < 0 for value in exact):
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.AXIS_MISMATCH,
            "canonical block measure contains an unknown stratum",
        )
    numeric = tuple(_fraction_center_and_radius(value) for value in exact)
    return V1SignedWorkBlockSupportMeasureContract(
        measure_kind=V1SignedWorkBlockSupportMeasureKind.FULL_POPULATION_STRATIFIED,
        measure_design_hash=design_hash,
        measure_weight_receipt_hash=receipt_hash,
        registered_context_axis_hash=_source_axis_hash(
            "block_support_context", contexts
        ),
        context_stratum_ids=strata,
        exact_context_weight_fractions=tuple(
            (value.numerator, value.denominator) for value in exact
        ),
        context_weight_binary64_centers=tuple(value[0] for value in numeric),
        context_weight_binary64_radii_upper=tuple(value[1] for value in numeric),
        population_count=population_count,
        population_stratum_counts=populations,
        d1_stratum_counts=d1,
        residual_sample_stratum_counts=sampled,
        d1_context_count=d1_count,
        full_population_uniform_estimand=True,
        canonical_f_d_measure_authority=True,
        uniform_local_empirical_witness_only=False,
        _issuer=_MEASURE_CONTRACT_ISSUER,
        _residual_measure_design=measure_design,
        _residual_weight_receipt=measure_weight_receipt,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkBlockSupportCUDAExecutionDesign:
    route_chunk_size: int = 131_072
    maximum_batch_size: int = 8
    packed_byte_tile_size: int = 1 << 20
    hbm_ceiling_fraction: float = 0.85
    execution_mode: V1SignedWorkBlockSupportExecutionMode = (
        V1SignedWorkBlockSupportExecutionMode.CUDA_PRODUCTION
    )
    deterministic_fp64_required: bool = True
    no_context_route_retention: bool = True
    no_route_scale_d2h_or_content_hash: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "route_chunk_size",
            _integer(self.route_chunk_size, "route chunk size", positive=True),
        )
        object.__setattr__(
            self,
            "maximum_batch_size",
            _integer(self.maximum_batch_size, "maximum batch size", positive=True),
        )
        object.__setattr__(
            self,
            "packed_byte_tile_size",
            _integer(
                self.packed_byte_tile_size,
                "packed-byte tile size",
                positive=True,
            ),
        )
        fraction = float(self.hbm_ceiling_fraction)
        if not math.isfinite(fraction) or not (0.0 < fraction <= 0.85):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.INVALID_DESIGN,
                "HBM ceiling must be in (0,.85]",
            )
        object.__setattr__(self, "hbm_ceiling_fraction", fraction)
        if type(self.execution_mode) is not V1SignedWorkBlockSupportExecutionMode:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
                "execution mode has wrong type",
            )
        if (
            self.deterministic_fp64_required is not True
            or self.no_context_route_retention is not True
            or self.no_route_scale_d2h_or_content_hash is not True
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.INVALID_DESIGN,
                "all production invariants are mandatory",
            )

    @property
    def design_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "design-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkBlockSupportSourceRegistration:
    registration_id: str
    source_role: V1SignedWorkBlockSupportSourceRole
    source_registration_hash: str
    source_context_start_ordinal: int
    context_group_id: str
    block_partition_hash: str
    block_ordinal: int
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
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    witness_stream_source_hash: str
    same_capture_vjp_source_hash: str
    corrected_evidence_stream_source_hash: str
    literal_j_stream_source_hash: str
    alpha_ledger_id: str
    alpha_coordinate_id: str
    alpha_lineage_hash: str
    context_ids: tuple[str, ...]
    draw_ids: tuple[str, ...]
    measurement_ids: tuple[str, ...]
    capture_ids: tuple[str, ...]
    source_packet_hashes: tuple[str, ...]
    context_weights: tuple[float, ...]
    measure_contract_hash: str
    measure_registered_context_axis_hash: str
    measure_kind: V1SignedWorkBlockSupportMeasureKind
    context_stratum_axis_hash: str
    exact_context_weight_fraction_axis_hash: str
    full_population_canonical_f_d_measure_authority: bool
    frozen_before_group_outcomes: bool = True
    same_live_d_w2_chunk_binding_required: bool = True
    uniform_finite_population_group_mean: bool = True
    schema: str = _SCHEMA
    _issuer: object = field(default=None, repr=False, compare=False)
    _measure_contract: V1SignedWorkBlockSupportMeasureContract | None = field(
        default=None, repr=False, compare=False
    )
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuer is not _ISSUER:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
                "registration must be issued by the factory",
            )
        for name in (
            "registration_id",
            "source_registration_hash",
            "context_group_id",
            "block_partition_hash",
            "preferred_action_binding_hash",
            "preferred_action_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "preferred_resulting_mask_hash",
            "preferred_removed_route_index_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "witness_stream_source_hash",
            "same_capture_vjp_source_hash",
            "corrected_evidence_stream_source_hash",
            "literal_j_stream_source_hash",
            "alpha_ledger_id",
            "alpha_coordinate_id",
            "alpha_lineage_hash",
            "measure_contract_hash",
            "measure_registered_context_axis_hash",
            "context_stratum_axis_hash",
            "exact_context_weight_fraction_axis_hash",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if type(self.source_role) is not V1SignedWorkBlockSupportSourceRole:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
                "source role has wrong type",
            )
        if (
            type(self.measure_kind) is not V1SignedWorkBlockSupportMeasureKind
            or type(self._measure_contract)
            is not V1SignedWorkBlockSupportMeasureContract
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
                "source registration lost its exact measure contract",
            )
        measure = self._measure_contract
        measure.assert_integrity()
        for name in (
            "source_context_start_ordinal",
            "block_ordinal",
            "preferred_selected_count",
            "proposal_generation",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        axes = []
        for name in (
            "context_ids",
            "draw_ids",
            "measurement_ids",
            "capture_ids",
            "source_packet_hashes",
        ):
            item = _axis(getattr(self, name), name)
            object.__setattr__(self, name, item)
            axes.append(item)
        if len({len(item) for item in axes}) != 1:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.AXIS_MISMATCH,
                "registered group axes do not align",
            )
        object.__setattr__(
            self, "context_weights", _weights(self.context_weights, len(axes[0]))
        )
        if (
            measure.contract_hash != self.measure_contract_hash
            or measure.registered_context_axis_hash
            != self.measure_registered_context_axis_hash
            or measure.registered_context_axis_hash
            != _source_axis_hash("block_support_context", axes[0])
            or measure.measure_kind is not self.measure_kind
            or measure.context_stratum_axis_hash != self.context_stratum_axis_hash
            or measure.exact_context_weight_fraction_axis_hash
            != self.exact_context_weight_fraction_axis_hash
            or measure.context_weight_binary64_centers != self.context_weights
            or len(measure.context_stratum_ids) != len(axes[0])
            or self.full_population_canonical_f_d_measure_authority
            is not measure.canonical_f_d_measure_authority
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.STALE_IDENTITY,
                "source registration and exact measure contract differ",
            )
        legacy_measure = (
            self.measure_kind
            is V1SignedWorkBlockSupportMeasureKind.LEGACY_UNIFORM_LOCAL_EMPIRICAL
            and self.uniform_finite_population_group_mean is True
            and self.full_population_canonical_f_d_measure_authority is False
            and self.schema == _SCHEMA
        )
        canonical_measure = (
            self.measure_kind
            is V1SignedWorkBlockSupportMeasureKind.FULL_POPULATION_STRATIFIED
            and self.uniform_finite_population_group_mean is False
            and self.full_population_canonical_f_d_measure_authority is True
            and self.schema == _REGISTERED_MEASURE_SCHEMA
        )
        if (
            self.frozen_before_group_outcomes is not True
            or self.same_live_d_w2_chunk_binding_required is not True
            or type(self.uniform_finite_population_group_mean) is not bool
            or type(self.full_population_canonical_f_d_measure_authority) is not bool
            or not (legacy_measure or canonical_measure)
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.AUTHORITY_VIOLATION,
                "source freeze/binding invariants are mandatory",
            )
        expected_seal = content_hash(
            {
                item.name: getattr(self, item.name)
                for item in self.__dataclass_fields__.values()
                if not item.name.startswith("_")
            },
            prefix=_PREFIX + "registration-integrity-",
        )
        if self._integrity_seal and self._integrity_seal != expected_seal:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.MUTATED_CUSTODY,
                "support registration integrity changed",
            )
        object.__setattr__(self, "_integrity_seal", expected_seal)

    def assert_integrity(self) -> None:
        self.measure_contract.assert_integrity()
        expected = content_hash(
            {
                item.name: getattr(self, item.name)
                for item in self.__dataclass_fields__.values()
                if not item.name.startswith("_")
            },
            prefix=_PREFIX + "registration-integrity-",
        )
        if expected != self._integrity_seal:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.MUTATED_CUSTODY,
                "support registration integrity changed",
            )

    @property
    def context_count(self) -> int:
        return len(self.context_ids)

    @property
    def measure_contract(self) -> V1SignedWorkBlockSupportMeasureContract:
        if type(self._measure_contract) is not V1SignedWorkBlockSupportMeasureContract:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.MUTATED_CUSTODY,
                "source registration lost measure-contract custody",
            )
        self._measure_contract.assert_integrity()
        if self._measure_contract.contract_hash != self.measure_contract_hash:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.MUTATED_CUSTODY,
                "source registration measure-contract custody changed",
            )
        return self._measure_contract

    @property
    def registration_hash(self) -> str:
        return content_hash(
            {
                item.name: getattr(self, item.name)
                for item in self.__dataclass_fields__.values()
                if not item.name.startswith("_")
            },
            prefix=_PREFIX + "registration-",
        )


class V1SignedWorkBlockSupportPreferredActionBinding:
    """One deep-validated packed action reused by all registered D_W2 groups."""

    __slots__ = (
        "source_registration_hash",
        "model_identity_hash",
        "chart_identity_hash",
        "current_mask_hash",
        "proposal_generation",
        "route_axis_hash",
        "route_count",
        "preferred_action_hash",
        "preferred_resulting_mask_hash",
        "preferred_removed_route_index_hash",
        "preferred_selected_count",
        "packed_bitorder",
        "removal_semantics",
        "_parent_packed_bits",
        "_child_packed_bits",
        "_binding_hash",
        "_frozen",
        "_evicted",
    )

    def __init__(
        self,
        *,
        source_registration: V1SignedWorkVariableShellSourceRegistration,
        parent_mask: HardMask,
        preferred_action: V1PackedRemovalAction,
        _issuer: object,
    ) -> None:
        if (
            _issuer is not _ACTION_BINDING_ISSUER
            or type(source_registration)
            is not V1SignedWorkVariableShellSourceRegistration
            or type(parent_mask) is not HardMask
            or type(preferred_action) is not V1PackedRemovalAction
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
                "preferred-action bindings are factory-issued only",
            )
        child = preferred_action.resulting_mask
        self.source_registration_hash = source_registration.registration_hash
        self.model_identity_hash = source_registration.model_identity_hash
        self.chart_identity_hash = source_registration.chart_identity_hash
        self.current_mask_hash = parent_mask.mask_hash
        self.proposal_generation = source_registration.proposal_generation
        self.route_axis_hash = source_registration.route_axis_hash
        self.route_count = parent_mask.route_count
        self.preferred_action_hash = preferred_action.action_hash
        self.preferred_resulting_mask_hash = child.mask_hash
        self.preferred_removed_route_index_hash = (
            preferred_action.removed_route_index_hash
        )
        self.preferred_selected_count = preferred_action.removed_count
        self.packed_bitorder = "big"
        self.removal_semantics = "z=parent_and_not_child"
        self._parent_packed_bits = parent_mask.packed_bits
        self._child_packed_bits = child.packed_bits
        self._binding_hash = content_hash(
            self._payload(), prefix=_PREFIX + "preferred-action-binding-"
        )
        self._evicted = False
        self._frozen = True

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_frozen", False):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.MUTATED_CUSTODY,
                "preferred-action binding is immutable",
            )
        object.__setattr__(self, name, value)

    def __reduce__(self) -> NoReturn:
        raise TypeError("live preferred-action bindings are not serializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live preferred-action bindings are not serializable")

    @property
    def binding_hash(self) -> str:
        self.assert_live()
        return self._binding_hash

    def _payload(self) -> dict[str, object]:
        return {
            "source_registration_hash": self.source_registration_hash,
            "model_identity_hash": self.model_identity_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "current_mask_hash": self.current_mask_hash,
            "proposal_generation": self.proposal_generation,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "preferred_action_hash": self.preferred_action_hash,
            "preferred_resulting_mask_hash": self.preferred_resulting_mask_hash,
            "preferred_removed_route_index_hash": (
                self.preferred_removed_route_index_hash
            ),
            "preferred_selected_count": self.preferred_selected_count,
            "packed_bitorder": self.packed_bitorder,
            "removal_semantics": self.removal_semantics,
        }

    def assert_live(self) -> None:
        if self._evicted:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.EVICTED,
                "preferred-action packed binding was evicted",
            )
        if (
            type(self._parent_packed_bits) is not bytes
            or type(self._child_packed_bits) is not bytes
            or len(self._parent_packed_bits) != (self.route_count + 7) // 8
            or len(self._child_packed_bits) != (self.route_count + 7) // 8
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.MUTATED_CUSTODY,
                "preferred-action packed custody changed",
            )

    def materialize(self, *, device: Any, tile_bytes: int) -> tuple[Any, int]:
        self.assert_live()
        return _materialize_packed_preferred_action(
            parent_packed_bits=self._parent_packed_bits,
            child_packed_bits=self._child_packed_bits,
            route_count=self.route_count,
            device=device,
            tile_bytes=tile_bytes,
        )

    def materialize_float64(
        self,
        *,
        device: Any,
        tile_bytes: int,
        production_cuda_required: bool,
    ) -> tuple[Any, int, int]:
        """Materialize ``z=parent & ~child`` directly in FP64.

        Production transfers only the packed parent/child bytes and unpacks
        them on CUDA.  The returned integers are exact H2D payload bytes and a
        deterministic upper bound on live logical tile-tensor payload bytes,
        respectively; allocator/compiler workspace still belongs to the
        measured outer HBM admission.  No route-scale host array is created.
        CPU execution is an explicit reference-only path.
        """

        self.assert_live()
        return _materialize_packed_preferred_action_float64(
            parent_packed_bits=self._parent_packed_bits,
            child_packed_bits=self._child_packed_bits,
            route_count=self.route_count,
            device=device,
            tile_bytes=tile_bytes,
            production_cuda_required=production_cuda_required,
        )

    def evict(self) -> None:
        self.assert_live()
        object.__setattr__(self, "_parent_packed_bits", None)
        object.__setattr__(self, "_child_packed_bits", None)
        object.__setattr__(self, "_evicted", True)


def bind_v1_signed_work_block_support_preferred_action(
    *,
    source_registration: V1SignedWorkVariableShellSourceRegistration,
    preferred_action: V1PackedRemovalAction,
    parent_mask: HardMask,
) -> V1SignedWorkBlockSupportPreferredActionBinding:
    """Deep-validate and snapshot one physical action exactly once."""

    if (
        type(source_registration) is not V1SignedWorkVariableShellSourceRegistration
        or type(preferred_action) is not V1PackedRemovalAction
        or type(parent_mask) is not HardMask
    ):
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
            "preferred action requires exact source, packed action, and parent types",
        )
    # Snapshot immutable bytes before validation, then validate the snapshots.
    # This neither trusts cached mask hashes nor leaves caller-owned objects as
    # the source's long-lived custody boundary.
    parent_snapshot = HardMask(
        route_axis_hash=parent_mask.route_axis_hash,
        route_count=parent_mask.route_count,
        packed_bits=bytes(parent_mask.packed_bits),
    )
    child = preferred_action.resulting_mask
    child_snapshot = HardMask(
        route_axis_hash=child.route_axis_hash,
        route_count=child.route_count,
        packed_bits=bytes(child.packed_bits),
    )
    action_snapshot = V1PackedRemovalAction(
        resulting_mask=child_snapshot,
        removed_count=preferred_action.removed_count,
        removed_route_index_hash=preferred_action.removed_route_index_hash,
    )
    _validate_packed_preferred_action(
        packed_action=action_snapshot,
        parent_mask=parent_snapshot,
        source_registration=source_registration,
    )
    return V1SignedWorkBlockSupportPreferredActionBinding(
        source_registration=source_registration,
        parent_mask=parent_snapshot,
        preferred_action=action_snapshot,
        _issuer=_ACTION_BINDING_ISSUER,
    )


def register_v1_signed_work_block_support_source(
    *,
    source_registration: V1SignedWorkVariableShellSourceRegistration,
    source_role: V1SignedWorkBlockSupportSourceRole,
    source_context_start_ordinal: int,
    context_group_id: str,
    block_partition_hash: str,
    block_ordinal: int,
    preferred_action_binding: V1SignedWorkBlockSupportPreferredActionBinding,
    context_ids: Sequence[str],
    draw_ids: Sequence[str],
    measurement_ids: Sequence[str],
    capture_ids: Sequence[str],
    source_packet_hashes: Sequence[str],
    measure_contract: V1SignedWorkBlockSupportMeasureContract | None = None,
) -> V1SignedWorkBlockSupportSourceRegistration:
    if type(source_registration) is not V1SignedWorkVariableShellSourceRegistration:
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
            "source registration has wrong type",
        )
    start = _integer(source_context_start_ordinal, "source context start")
    group_axes = tuple(
        tuple(item)
        for item in (
            context_ids,
            draw_ids,
            measurement_ids,
            capture_ids,
            source_packet_hashes,
        )
    )
    count = len(group_axes[0])
    stop = start + count
    source_axes = (
        source_registration.registered_context_ids,
        source_registration.registered_draw_ids,
        source_registration.registered_measurement_ids,
        source_registration.registered_capture_ids,
        source_registration.registered_source_packet_hashes,
    )
    if (
        count <= 0
        or stop > source_registration.context_count
        or any(
            group != tuple(source[start:stop])
            for group, source in zip(group_axes, source_axes)
        )
    ):
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.ORDER_MISMATCH,
            "group is not an exact contiguous D_W2 source slice",
        )
    if (
        type(preferred_action_binding)
        is not V1SignedWorkBlockSupportPreferredActionBinding
    ):
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
            "support registration requires one issued preferred-action binding",
        )
    preferred_action_binding.assert_live()
    if (
        preferred_action_binding.source_registration_hash
        != source_registration.registration_hash
        or preferred_action_binding.model_identity_hash
        != source_registration.model_identity_hash
        or preferred_action_binding.chart_identity_hash
        != source_registration.chart_identity_hash
        or preferred_action_binding.current_mask_hash
        != source_registration.current_mask_hash
        or preferred_action_binding.proposal_generation
        != source_registration.proposal_generation
        or preferred_action_binding.route_axis_hash
        != source_registration.route_axis_hash
    ):
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.STALE_IDENTITY,
            "preferred-action binding belongs to another registered D_W2 source",
        )
    if measure_contract is None:
        measure = _issue_legacy_uniform_local_measure(
            context_group_id=context_group_id,
            context_ids=group_axes[0],
        )
    elif type(measure_contract) is V1SignedWorkBlockSupportMeasureContract:
        measure = measure_contract
        measure.assert_integrity()
    else:
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
            "measure contract has the wrong exact type",
        )
    if len(measure.context_stratum_ids) != count:
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.AXIS_MISMATCH,
            "measure contract does not cover the registered context group",
        )
    legacy = (
        measure.measure_kind
        is V1SignedWorkBlockSupportMeasureKind.LEGACY_UNIFORM_LOCAL_EMPIRICAL
    )
    return V1SignedWorkBlockSupportSourceRegistration(
        registration_id=f"block-support:{context_group_id}",
        source_role=source_role,
        source_registration_hash=source_registration.registration_hash,
        source_context_start_ordinal=start,
        context_group_id=context_group_id,
        block_partition_hash=block_partition_hash,
        block_ordinal=block_ordinal,
        preferred_action_binding_hash=preferred_action_binding.binding_hash,
        preferred_action_hash=preferred_action_binding.preferred_action_hash,
        preferred_resulting_mask_hash=(
            preferred_action_binding.preferred_resulting_mask_hash
        ),
        preferred_removed_route_index_hash=(
            preferred_action_binding.preferred_removed_route_index_hash
        ),
        preferred_selected_count=preferred_action_binding.preferred_selected_count,
        model_identity_hash=source_registration.model_identity_hash,
        chart_identity_hash=source_registration.chart_identity_hash,
        current_mask_hash=source_registration.current_mask_hash,
        proposal_generation=source_registration.proposal_generation,
        route_axis_hash=source_registration.route_axis_hash,
        route_id_axis_hash=source_registration.route_id_axis_hash,
        native_fiber_axis_receipt_hash=source_registration.native_fiber_axis_receipt_hash,
        native_fiber_measure_hash=source_registration.native_fiber_measure_hash,
        registered_d_prefix_hash=source_registration.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=source_registration.registered_d_w2_prefix_hash,
        witness_stream_source_hash=source_registration.witness_stream_source_hash,
        same_capture_vjp_source_hash=source_registration.same_capture_vjp_source_hash,
        corrected_evidence_stream_source_hash=source_registration.corrected_evidence_stream_source_hash,
        literal_j_stream_source_hash=source_registration.literal_j_stream_source_hash,
        alpha_ledger_id=source_registration.alpha_ledger_id,
        alpha_coordinate_id=source_registration.alpha_coordinate_id,
        alpha_lineage_hash=source_registration.alpha_lineage_hash,
        context_ids=group_axes[0],
        draw_ids=group_axes[1],
        measurement_ids=group_axes[2],
        capture_ids=group_axes[3],
        source_packet_hashes=group_axes[4],
        context_weights=measure.context_weight_binary64_centers,
        measure_contract_hash=measure.contract_hash,
        measure_registered_context_axis_hash=(
            measure.registered_context_axis_hash
        ),
        measure_kind=measure.measure_kind,
        context_stratum_axis_hash=measure.context_stratum_axis_hash,
        exact_context_weight_fraction_axis_hash=(
            measure.exact_context_weight_fraction_axis_hash
        ),
        full_population_canonical_f_d_measure_authority=(
            measure.canonical_f_d_measure_authority
        ),
        uniform_finite_population_group_mean=legacy,
        schema=_SCHEMA if legacy else _REGISTERED_MEASURE_SCHEMA,
        _issuer=_ISSUER,
        _measure_contract=measure,
    )


def _multi_segment_registration_axis_hash(values: Sequence[str]) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_block_support_segment_registration_axis_v1",
            "values": tuple(values),
        },
        prefix=_PREFIX + "segment-registration-axis-",
    )


def _multi_segment_source_chain_hash(values: Sequence[str]) -> str:
    chain = content_hash(
        {
            "schema": "partition_free_v1_block_support_numerical_source_chain_v1",
            "empty": True,
        },
        prefix=_PREFIX + "numerical-source-chain-",
    )
    for ordinal, value in enumerate(values):
        chain = content_hash(
            {
                "schema": "partition_free_v1_block_support_numerical_source_link_v1",
                "prior": chain,
                "ordinal": ordinal,
                "segment_numerical_source_hash": value,
            },
            prefix=_PREFIX + "numerical-source-chain-",
        )
    return chain


def _segment_numerical_source_hash(
    registration: V1SignedWorkVariableShellSourceRegistration,
) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_block_support_segment_numerical_source_v1",
            "registration_hash": registration.registration_hash,
            "context_axis_hash": _source_axis_hash(
                "segment_context", registration.registered_context_ids
            ),
            "draw_axis_hash": _source_axis_hash(
                "segment_draw", registration.registered_draw_ids
            ),
            "measurement_axis_hash": _source_axis_hash(
                "segment_measurement", registration.registered_measurement_ids
            ),
            "capture_axis_hash": _source_axis_hash(
                "segment_capture", registration.registered_capture_ids
            ),
            "source_packet_axis_hash": _source_axis_hash(
                "segment_source_packet", registration.registered_source_packet_hashes
            ),
            "context_weight_hash": content_hash(
                registration.context_weights,
                prefix=_PREFIX + "segment-source-weights-",
            ),
        },
        prefix=_PREFIX + "segment-numerical-source-",
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkBlockSupportMultiSegmentSourceRegistration:
    """One logical support group backed by several source-local row owners.

    ``structural_group_registration`` remains the identity exposed to the
    downstream solver.  The ordered physical registration axis and numerical
    chain separately bind which live owners may supply the corrected-A rows.
    No source row is retained here.
    """

    structural_group_registration: V1SignedWorkBlockSupportSourceRegistration = field(
        repr=False, compare=False
    )
    structural_group_registration_hash: str
    structural_source_registration_hash: str
    segment_source_registration_hashes: tuple[str, ...]
    segment_context_counts: tuple[int, ...]
    segment_numerical_source_hashes: tuple[str, ...]
    segment_registration_axis_hash: str
    numerical_source_chain_hash: str
    context_count: int
    measure_contract_hash: str
    measure_kind: V1SignedWorkBlockSupportMeasureKind
    full_population_canonical_f_d_measure_authority: bool
    exact_axis_concatenation: bool = True
    unique_contexts_across_segments: bool = True
    uniform_conditional_group_mean: bool = True
    source_local_chunk_binding_only: bool = True
    schema: str = _MULTI_SEGMENT_SCHEMA
    _issuer: object = field(default=None, repr=False, compare=False)
    _measure_contract: V1SignedWorkBlockSupportMeasureContract | None = field(
        default=None, repr=False, compare=False
    )
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self._issuer is not _MULTI_SEGMENT_REGISTRATION_ISSUER
            or type(self.structural_group_registration)
            is not V1SignedWorkBlockSupportSourceRegistration
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
                "multi-segment registrations are factory-issued only",
            )
        self.structural_group_registration.assert_integrity()
        for name in (
            "structural_group_registration_hash",
            "structural_source_registration_hash",
            "segment_registration_axis_hash",
            "numerical_source_chain_hash",
            "measure_contract_hash",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        hashes = _axis(
            self.segment_source_registration_hashes,
            "segment source registration hashes",
        )
        numerical = _axis(
            self.segment_numerical_source_hashes,
            "segment numerical source hashes",
        )
        counts = tuple(
            _integer(item, "segment context count", positive=True)
            for item in self.segment_context_counts
        )
        object.__setattr__(self, "segment_source_registration_hashes", hashes)
        object.__setattr__(self, "segment_numerical_source_hashes", numerical)
        object.__setattr__(self, "segment_context_counts", counts)
        object.__setattr__(
            self,
            "context_count",
            _integer(self.context_count, "context count", positive=True),
        )
        structural = self.structural_group_registration
        if (
            type(self.measure_kind) is not V1SignedWorkBlockSupportMeasureKind
            or type(self._measure_contract)
            is not V1SignedWorkBlockSupportMeasureContract
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
                "multi-segment registration lost exact measure custody",
            )
        measure = self._measure_contract
        measure.assert_integrity()
        required_true = (
            self.exact_axis_concatenation,
            self.unique_contexts_across_segments,
            self.uniform_conditional_group_mean,
            self.source_local_chunk_binding_only,
        )
        legacy_measure = (
            self.measure_kind
            is V1SignedWorkBlockSupportMeasureKind.LEGACY_UNIFORM_LOCAL_EMPIRICAL
            and self.full_population_canonical_f_d_measure_authority is False
            and self.schema == _MULTI_SEGMENT_SCHEMA
        )
        canonical_measure = (
            self.measure_kind
            is V1SignedWorkBlockSupportMeasureKind.FULL_POPULATION_STRATIFIED
            and self.full_population_canonical_f_d_measure_authority is True
            and self.schema == _MULTI_SEGMENT_REGISTERED_MEASURE_SCHEMA
        )
        if (
            len(hashes) < 2
            or len(hashes) != len(counts)
            or len(hashes) != len(numerical)
            or sum(counts) != self.context_count
            or self.context_count != structural.context_count
            or self.structural_group_registration_hash != structural.registration_hash
            or self.structural_source_registration_hash
            != structural.source_registration_hash
            or self.segment_registration_axis_hash
            != _multi_segment_registration_axis_hash(hashes)
            or self.numerical_source_chain_hash
            != _multi_segment_source_chain_hash(numerical)
            or self.measure_contract_hash != measure.contract_hash
            or self.measure_kind is not measure.measure_kind
            or self.full_population_canonical_f_d_measure_authority
            is not measure.canonical_f_d_measure_authority
            or structural.measure_contract_hash != measure.contract_hash
            or any(type(value) is not bool or not value for value in required_true)
            or not (legacy_measure or canonical_measure)
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.AUTHORITY_VIOLATION,
                "multi-segment registration lost its exact source decomposition",
            )
        expected = content_hash(
            self._payload(), prefix=_PREFIX + "multi-segment-registration-integrity-"
        )
        if self._integrity_seal and self._integrity_seal != expected:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.MUTATED_CUSTODY,
                "multi-segment registration integrity changed",
            )
        object.__setattr__(self, "_integrity_seal", expected)

    def _payload(self) -> dict[str, object]:
        return {
            "structural_group_registration_hash": (
                self.structural_group_registration_hash
            ),
            "structural_source_registration_hash": (
                self.structural_source_registration_hash
            ),
            "segment_source_registration_hashes": (
                self.segment_source_registration_hashes
            ),
            "segment_context_counts": self.segment_context_counts,
            "segment_numerical_source_hashes": self.segment_numerical_source_hashes,
            "segment_registration_axis_hash": self.segment_registration_axis_hash,
            "numerical_source_chain_hash": self.numerical_source_chain_hash,
            "context_count": self.context_count,
            "measure_contract_hash": self.measure_contract_hash,
            "measure_kind": self.measure_kind.value,
            "full_population_canonical_f_d_measure_authority": (
                self.full_population_canonical_f_d_measure_authority
            ),
            "exact_axis_concatenation": self.exact_axis_concatenation,
            "unique_contexts_across_segments": self.unique_contexts_across_segments,
            "uniform_conditional_group_mean": self.uniform_conditional_group_mean,
            "source_local_chunk_binding_only": self.source_local_chunk_binding_only,
            "schema": self.schema,
        }

    def assert_integrity(self) -> None:
        self.structural_group_registration.assert_integrity()
        self.measure_contract.assert_integrity()
        if (
            self.structural_group_registration.registration_hash
            != self.structural_group_registration_hash
            or content_hash(
                self._payload(),
                prefix=_PREFIX + "multi-segment-registration-integrity-",
            )
            != self._integrity_seal
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.MUTATED_CUSTODY,
                "multi-segment registration integrity changed",
            )

    @property
    def registration_hash(self) -> str:
        return content_hash(
            self._payload(), prefix=_PREFIX + "multi-segment-registration-"
        )

    @property
    def segment_count(self) -> int:
        return len(self.segment_source_registration_hashes)

    @property
    def measure_contract(self) -> V1SignedWorkBlockSupportMeasureContract:
        if type(self._measure_contract) is not V1SignedWorkBlockSupportMeasureContract:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.MUTATED_CUSTODY,
                "multi-segment registration lost measure custody",
            )
        self._measure_contract.assert_integrity()
        if self._measure_contract.contract_hash != self.measure_contract_hash:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.MUTATED_CUSTODY,
                "multi-segment measure custody changed",
            )
        return self._measure_contract


def register_v1_signed_work_block_support_multi_segment_source(
    *,
    structural_source_registration: V1SignedWorkVariableShellSourceRegistration,
    segment_source_registrations: Sequence[V1SignedWorkVariableShellSourceRegistration],
    source_role: V1SignedWorkBlockSupportSourceRole,
    context_group_id: str,
    block_partition_hash: str,
    block_ordinal: int,
    preferred_action_binding: V1SignedWorkBlockSupportPreferredActionBinding,
    context_ids: Sequence[str],
    draw_ids: Sequence[str],
    measurement_ids: Sequence[str],
    capture_ids: Sequence[str],
    source_packet_hashes: Sequence[str],
    measure_contract: V1SignedWorkBlockSupportMeasureContract | None = None,
) -> V1SignedWorkBlockSupportMultiSegmentSourceRegistration:
    """Register one conditional group whose rows live in ordered segments."""

    if (
        type(structural_source_registration)
        is not V1SignedWorkVariableShellSourceRegistration
        or type(segment_source_registrations) not in {tuple, list}
        or any(
            type(item) is not V1SignedWorkVariableShellSourceRegistration
            for item in segment_source_registrations
        )
    ):
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
            "multi-segment registration requires exact variable-source registrations",
        )
    segments = tuple(segment_source_registrations)
    if len(segments) < 2:
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.INVALID_DESIGN,
            "multi-segment groups require at least two physical segments",
        )
    supplied_axes = tuple(
        tuple(axis)
        for axis in (
            context_ids,
            draw_ids,
            measurement_ids,
            capture_ids,
            source_packet_hashes,
        )
    )
    segment_axis_names = (
        "registered_context_ids",
        "registered_draw_ids",
        "registered_measurement_ids",
        "registered_capture_ids",
        "registered_source_packet_hashes",
    )
    concatenated = tuple(
        tuple(value for segment in segments for value in getattr(segment, name))
        for name in segment_axis_names
    )
    structural_axes = tuple(
        tuple(getattr(structural_source_registration, name))
        for name in segment_axis_names
    )
    if supplied_axes != concatenated or supplied_axes != structural_axes:
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.ORDER_MISMATCH,
            "physical segments do not exactly concatenate to the structural group axes",
        )
    # The legacy group registration remains the semantic/downstream identity;
    # this sibling only changes where its numerical rows are physically owned.
    structural_group = register_v1_signed_work_block_support_source(
        source_registration=structural_source_registration,
        source_role=source_role,
        source_context_start_ordinal=0,
        context_group_id=context_group_id,
        block_partition_hash=block_partition_hash,
        block_ordinal=block_ordinal,
        preferred_action_binding=preferred_action_binding,
        context_ids=context_ids,
        draw_ids=draw_ids,
        measurement_ids=measurement_ids,
        capture_ids=capture_ids,
        source_packet_hashes=source_packet_hashes,
        measure_contract=measure_contract,
    )
    common_names = (
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
    if any(
        getattr(segment, name) != getattr(structural_source_registration, name)
        for segment in segments
        for name in common_names
    ):
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.STALE_IDENTITY,
            "a numerical segment belongs to another chart, action domain, or stream",
        )
    if any(
        any(
            weight != 1.0 / float(segment.context_count)
            for weight in segment.context_weights
        )
        for segment in segments
    ):
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.INVALID_DESIGN,
            "each source segment must represent its exact uniform conditional mean",
        )
    segment_hashes = tuple(segment.registration_hash for segment in segments)
    if len(set(segment_hashes)) != len(segment_hashes):
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.AXIS_MISMATCH,
            "physical segment registrations must be unique",
        )
    numerical_hashes = tuple(_segment_numerical_source_hash(item) for item in segments)
    measure = structural_group.measure_contract
    legacy = (
        measure.measure_kind
        is V1SignedWorkBlockSupportMeasureKind.LEGACY_UNIFORM_LOCAL_EMPIRICAL
    )
    return V1SignedWorkBlockSupportMultiSegmentSourceRegistration(
        structural_group_registration=structural_group,
        structural_group_registration_hash=structural_group.registration_hash,
        structural_source_registration_hash=(
            structural_source_registration.registration_hash
        ),
        segment_source_registration_hashes=segment_hashes,
        segment_context_counts=tuple(item.context_count for item in segments),
        segment_numerical_source_hashes=numerical_hashes,
        segment_registration_axis_hash=_multi_segment_registration_axis_hash(
            segment_hashes
        ),
        numerical_source_chain_hash=_multi_segment_source_chain_hash(numerical_hashes),
        context_count=structural_group.context_count,
        measure_contract_hash=measure.contract_hash,
        measure_kind=measure.measure_kind,
        full_population_canonical_f_d_measure_authority=(
            measure.canonical_f_d_measure_authority
        ),
        schema=(
            _MULTI_SEGMENT_SCHEMA
            if legacy
            else _MULTI_SEGMENT_REGISTERED_MEASURE_SCHEMA
        ),
        _issuer=_MULTI_SEGMENT_REGISTRATION_ISSUER,
        _measure_contract=measure,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkBlockSupportHBMForecast:
    registration_hash: str
    design_hash: str
    route_count: int
    maximum_chunk_routes: int
    maximum_batch_size: int
    persistent_coefficient_bytes: int
    persistent_coefficient_guard_bytes: int
    owned_preferred_action_bytes: int
    preferred_action_guard_bytes: int
    persistent_auxiliary_bytes: int
    source_colive_corrected_and_literal_bytes: int
    temporary_workspace_bytes: int
    incremental_bytes: int
    current_reserved_bytes: int
    projected_bytes: int
    device_total_bytes: int
    device_free_bytes: int
    ceiling_bytes: int
    fits: bool
    route_scale_d2h_bytes: int = 0
    route_scale_content_hash_bytes: int = 0
    schema: str = _FORECAST_SCHEMA
    _issuer: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuer is not _FORECAST_ISSUER:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
                "HBM forecasts are factory-issued only",
            )
        for name in ("registration_hash", "design_hash"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in (
            "route_count",
            "maximum_chunk_routes",
            "maximum_batch_size",
            "persistent_coefficient_bytes",
            "persistent_coefficient_guard_bytes",
            "owned_preferred_action_bytes",
            "preferred_action_guard_bytes",
            "persistent_auxiliary_bytes",
            "source_colive_corrected_and_literal_bytes",
            "temporary_workspace_bytes",
            "incremental_bytes",
            "current_reserved_bytes",
            "projected_bytes",
            "device_total_bytes",
            "device_free_bytes",
            "ceiling_bytes",
            "route_scale_d2h_bytes",
            "route_scale_content_hash_bytes",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        expected_incremental = (
            self.persistent_coefficient_bytes
            + self.persistent_coefficient_guard_bytes
            + self.owned_preferred_action_bytes
            + self.preferred_action_guard_bytes
            + self.persistent_auxiliary_bytes
            + self.source_colive_corrected_and_literal_bytes
            + self.temporary_workspace_bytes
        )
        if (
            min(self.route_count, self.maximum_chunk_routes, self.maximum_batch_size)
            <= 0
            or self.incremental_bytes != expected_incremental
            or self.projected_bytes
            != self.current_reserved_bytes + self.incremental_bytes
            or self.route_scale_d2h_bytes != 0
            or self.route_scale_content_hash_bytes != 0
            or type(self.fits) is not bool
            or self.schema != _FORECAST_SCHEMA
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.INVALID_DESIGN,
                "HBM forecast arithmetic or transfer prohibition changed",
            )

    @property
    def forecast_hash(self) -> str:
        return content_hash(
            {
                item.name: getattr(self, item.name)
                for item in self.__dataclass_fields__.values()
                if not item.name.startswith("_")
            },
            prefix=_PREFIX + "hbm-",
        )


def build_v1_signed_work_block_support_hbm_forecast(
    *,
    registration: V1SignedWorkBlockSupportSourceRegistration,
    native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt,
    design: V1SignedWorkBlockSupportCUDAExecutionDesign,
    device_total_memory_bytes: int,
    device_free_memory_bytes: int,
    current_reserved_device_bytes: int,
    _admission_memory: tuple[int, int] | None = None,
) -> V1SignedWorkBlockSupportHBMForecast:
    if (
        type(registration) is not V1SignedWorkBlockSupportSourceRegistration
        or type(native_fiber_axis_receipt) is not V1NativeDestinationFiberAxisReceipt
        or type(design) is not V1SignedWorkBlockSupportCUDAExecutionDesign
    ):
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
            "forecast inputs have wrong type",
        )
    registration.assert_integrity()
    native_fiber_axis_receipt.assert_integrity()
    if (
        registration.native_fiber_axis_receipt_hash
        != native_fiber_axis_receipt.receipt_hash
        or registration.native_fiber_measure_hash
        != native_fiber_axis_receipt.measure_hash
        or registration.route_axis_hash != native_fiber_axis_receipt.route_axis_hash
        or registration.route_id_axis_hash
        != native_fiber_axis_receipt.route_id_axis_hash
    ):
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.STALE_IDENTITY,
            "forecast native-fiber axis differs from registration",
        )
    total = _integer(device_total_memory_bytes, "total device memory", positive=True)
    free = _integer(device_free_memory_bytes, "free device memory")
    reserved = _integer(current_reserved_device_bytes, "reserved device memory")
    if free > total or reserved > total:
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.INVALID_DESIGN,
            "incoherent memory counters",
        )
    routes = native_fiber_axis_receipt.route_count
    chunk = min(routes, design.route_chunk_size)
    batch = min(registration.context_count, design.maximum_batch_size)
    # The smallest route-per-fiber cut maximizes the number of simultaneously
    # live fiber cells for a fixed route chunk.
    min_rpf = min(cut.routes_per_fiber for cut in native_fiber_axis_receipt.cuts)
    fibers = (chunk + min_rpf - 1) // min_rpf
    persistent = routes * 8
    coefficient_guard = routes * 8
    # The physical action is Boolean.  FP64 conversion is tile-local during
    # contraction; retaining two route-scale FP64 copies would waste 14 bytes
    # per route without changing the action geometry.
    owned_preferred = routes
    preferred_guard = routes
    # Exact rational weights are represented by one FP64 center and one
    # outward absolute-error radius per context.
    persistent_auxiliary = (2 * registration.context_count + 8) * 8
    # Both issuer-bound A and J must be live even though only A is consumed here.
    source_colive = 2 * batch * chunk * 8
    # Product/abs/weighted coefficient banks plus fiber centers/radii/signs,
    # coefficient/radius vectors, weights, and conservative reduction scalars.
    workspace = 8 * (
        5 * batch * chunk + 16 * batch * fibers + 16 * chunk + 2 * batch + 64
    )
    incremental = (
        persistent
        + coefficient_guard
        + owned_preferred
        + preferred_guard
        + persistent_auxiliary
        + source_colive
        + workspace
    )
    projected = reserved + incremental
    ceiling = math.floor(total * design.hbm_ceiling_fraction)
    available, allocated = _admission_memory or (free, reserved)
    fits = allocated + incremental <= ceiling and incremental <= available
    return V1SignedWorkBlockSupportHBMForecast(
        registration_hash=registration.registration_hash,
        design_hash=design.design_hash,
        route_count=routes,
        maximum_chunk_routes=chunk,
        maximum_batch_size=batch,
        persistent_coefficient_bytes=persistent,
        persistent_coefficient_guard_bytes=coefficient_guard,
        owned_preferred_action_bytes=owned_preferred,
        preferred_action_guard_bytes=preferred_guard,
        persistent_auxiliary_bytes=persistent_auxiliary,
        source_colive_corrected_and_literal_bytes=source_colive,
        temporary_workspace_bytes=workspace,
        incremental_bytes=incremental,
        current_reserved_bytes=reserved,
        projected_bytes=projected,
        device_total_bytes=total,
        device_free_bytes=free,
        ceiling_bytes=ceiling,
        fits=fits,
        _issuer=_FORECAST_ISSUER,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkBlockSupportSourceReceipt:
    registration_hash: str
    source_registration_hash: str
    source_role: V1SignedWorkBlockSupportSourceRole
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    alpha_ledger_id: str
    alpha_coordinate_id: str
    alpha_lineage_hash: str
    context_group_id: str
    block_partition_hash: str
    block_ordinal: int
    preferred_action_binding_hash: str
    preferred_action_hash: str
    preferred_resulting_mask_hash: str
    preferred_removed_route_index_hash: str
    preferred_selected_count: int
    context_axis_hash: str
    draw_axis_hash: str
    measurement_axis_hash: str
    capture_axis_hash: str
    source_packet_axis_hash: str
    context_weight_hash: str
    context_count: int
    route_count: int
    fiber_count: int
    cut_count: int
    certified_positive_cell_count: int
    certified_negative_cell_count: int
    ambiguous_cell_count: int
    ambiguous_preferred_response_upper: float
    coefficient_construction_l1_radius_upper: float
    preferred_plane_deficit_upper: float
    consumed_chunk_binding_count: int
    consumed_durable_chunk_binding_chain_hash: str
    hbm_forecast_hash: str
    execution_mode: V1SignedWorkBlockSupportExecutionMode
    preferred_action_representation: str
    packed_host_bytes_read: int
    scalar_device_to_host_transfer_bytes: int
    retained_context_route_rows: int = 0
    route_scale_d2h_values: int = 0
    route_scale_device_tensor_content_hashes: int = 0
    packed_action_bind_time_full_axis_validations: int = 1
    structural_group_registration_hash: str | None = None
    segment_registration_axis_hash: str | None = None
    numerical_source_chain_hash: str | None = None
    segment_registration_count: int = 1
    multi_segment_numerical_source: bool = False
    measure_contract_hash: str | None = None
    measure_registered_context_axis_hash: str | None = None
    measure_kind: V1SignedWorkBlockSupportMeasureKind = (
        V1SignedWorkBlockSupportMeasureKind.LEGACY_UNIFORM_LOCAL_EMPIRICAL
    )
    measure_design_hash: str | None = None
    measure_weight_receipt_hash: str | None = None
    context_stratum_axis_hash: str | None = None
    exact_context_weight_fraction_axis_hash: str | None = None
    exact_context_weights_sum_to_one: bool = True
    canonical_full_population_f_d_measure_authority: bool = False
    complete_imagenet_population_pair_receipt_hash: str | None = None
    uniform_finite_population_group_mean: bool = True
    preferred_action_content_bound: bool = True
    source_local_chunk_binding_only: bool = True
    deferred_source_finiteness_reduction: bool = True
    per_chunk_finiteness_device_to_host_syncs: int = 0
    upstream_dual_stream_custody_authority: bool = False
    production_imagenet_population_authority: bool = False
    complete_imagenet_population_pair_owner_join_required: bool = True
    population_separation_authority: bool = False
    q_authority: bool = False
    c_authority: bool = False
    k_authority: bool = False
    o_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    array_free_receipt: bool = True
    schema: str = _RECEIPT_SCHEMA
    _issuer: object = field(default=None, repr=False, compare=False)
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuer is not _RECEIPT_ISSUER:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
                "support-source receipts are issuer-only",
            )
        for name in (
            "registration_hash",
            "source_registration_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "alpha_ledger_id",
            "alpha_coordinate_id",
            "alpha_lineage_hash",
            "context_group_id",
            "block_partition_hash",
            "preferred_action_binding_hash",
            "preferred_action_hash",
            "preferred_resulting_mask_hash",
            "preferred_removed_route_index_hash",
            "context_axis_hash",
            "draw_axis_hash",
            "measurement_axis_hash",
            "capture_axis_hash",
            "source_packet_axis_hash",
            "context_weight_hash",
            "consumed_durable_chunk_binding_chain_hash",
            "hbm_forecast_hash",
            "preferred_action_representation",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in (
            "structural_group_registration_hash",
            "segment_registration_axis_hash",
            "numerical_source_chain_hash",
            "measure_contract_hash",
            "measure_registered_context_axis_hash",
            "measure_design_hash",
            "measure_weight_receipt_hash",
            "context_stratum_axis_hash",
            "exact_context_weight_fraction_axis_hash",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _text(value, name))
        if (
            type(self.source_role) is not V1SignedWorkBlockSupportSourceRole
            or type(self.execution_mode) is not V1SignedWorkBlockSupportExecutionMode
            or type(self.measure_kind) is not V1SignedWorkBlockSupportMeasureKind
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
                "support-source receipt enum type changed",
            )
        for name in (
            "proposal_generation",
            "block_ordinal",
            "preferred_selected_count",
            "context_count",
            "route_count",
            "fiber_count",
            "cut_count",
            "certified_positive_cell_count",
            "certified_negative_cell_count",
            "ambiguous_cell_count",
            "consumed_chunk_binding_count",
            "packed_host_bytes_read",
            "scalar_device_to_host_transfer_bytes",
            "retained_context_route_rows",
            "route_scale_d2h_values",
            "route_scale_device_tensor_content_hashes",
            "packed_action_bind_time_full_axis_validations",
            "segment_registration_count",
            "per_chunk_finiteness_device_to_host_syncs",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        for name in (
            "ambiguous_preferred_response_upper",
            "coefficient_construction_l1_radius_upper",
            "preferred_plane_deficit_upper",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name),
            )
        closed = (
            self.upstream_dual_stream_custody_authority,
            self.production_imagenet_population_authority,
            self.population_separation_authority,
            self.q_authority,
            self.c_authority,
            self.k_authority,
            self.o_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        required_true = (
            self.preferred_action_content_bound,
            self.source_local_chunk_binding_only,
            self.deferred_source_finiteness_reduction,
            self.complete_imagenet_population_pair_owner_join_required,
            self.array_free_receipt,
            self.exact_context_weights_sum_to_one,
        )
        legacy_provenance = (
            self.structural_group_registration_hash is None
            and self.segment_registration_axis_hash is None
            and self.numerical_source_chain_hash is None
            and self.segment_registration_count == 1
            and self.multi_segment_numerical_source is False
            and self.schema == _RECEIPT_SCHEMA
        )
        multi_segment_provenance = (
            self.structural_group_registration_hash is not None
            and self.segment_registration_axis_hash is not None
            and self.numerical_source_chain_hash is not None
            and self.segment_registration_count >= 2
            and self.multi_segment_numerical_source is True
            and self.schema == _MULTI_SEGMENT_RECEIPT_SCHEMA
        )
        registered_measure_provenance = (
            self.structural_group_registration_hash is not None
            and self.segment_registration_axis_hash is not None
            and self.numerical_source_chain_hash is not None
            and self.segment_registration_count >= 2
            and self.multi_segment_numerical_source is True
            and self.schema == _REGISTERED_MEASURE_RECEIPT_SCHEMA
        )
        registered_single_measure_provenance = (
            self.structural_group_registration_hash is None
            and self.segment_registration_axis_hash is None
            and self.numerical_source_chain_hash is None
            and self.segment_registration_count == 1
            and self.multi_segment_numerical_source is False
            and self.schema == _REGISTERED_MEASURE_RECEIPT_SCHEMA
        )
        measure_fields = (
            self.measure_contract_hash,
            self.measure_registered_context_axis_hash,
            self.measure_design_hash,
            self.measure_weight_receipt_hash,
            self.context_stratum_axis_hash,
            self.exact_context_weight_fraction_axis_hash,
        )
        typed_legacy_measure = (
            self.measure_kind
            is V1SignedWorkBlockSupportMeasureKind.LEGACY_UNIFORM_LOCAL_EMPIRICAL
            and self.uniform_finite_population_group_mean is True
            and self.canonical_full_population_f_d_measure_authority is False
            and (all(value is None for value in measure_fields) or all(
                type(value) is str and bool(value) for value in measure_fields
            ))
            and not registered_measure_provenance
        )
        canonical_measure = (
            self.measure_kind
            is V1SignedWorkBlockSupportMeasureKind.FULL_POPULATION_STRATIFIED
            and self.uniform_finite_population_group_mean is False
            and self.canonical_full_population_f_d_measure_authority is True
            and all(type(value) is str and bool(value) for value in measure_fields)
            and (
                registered_measure_provenance
                or registered_single_measure_provenance
            )
        )
        if (
            min(self.context_count, self.route_count, self.fiber_count, self.cut_count)
            <= 0
            or self.preferred_selected_count > self.route_count
            or self.certified_positive_cell_count
            + self.certified_negative_cell_count
            + self.ambiguous_cell_count
            != self.context_count * self.fiber_count
            or self.preferred_plane_deficit_upper
            < self.ambiguous_preferred_response_upper
            + self.coefficient_construction_l1_radius_upper
            or self.consumed_chunk_binding_count <= 0
            or self.preferred_action_representation
            != "canonical_packed_parent_child_delta_v1"
            or self.packed_host_bytes_read <= 0
            or (
                self.execution_mode
                is V1SignedWorkBlockSupportExecutionMode.CPU_REFERENCE_TEST_ONLY
                and self.scalar_device_to_host_transfer_bytes != 0
            )
            or self.retained_context_route_rows != 0
            or self.route_scale_d2h_values != 0
            or self.route_scale_device_tensor_content_hashes != 0
            or self.packed_action_bind_time_full_axis_validations != 1
            or self.per_chunk_finiteness_device_to_host_syncs != 0
            or self.complete_imagenet_population_pair_receipt_hash is not None
            or type(self.multi_segment_numerical_source) is not bool
            or type(self.uniform_finite_population_group_mean) is not bool
            or type(self.exact_context_weights_sum_to_one) is not bool
            or type(self.canonical_full_population_f_d_measure_authority) is not bool
            or (
                self.measure_registered_context_axis_hash is not None
                and self.measure_registered_context_axis_hash
                != self.context_axis_hash
            )
            or any(type(value) is not bool or value for value in closed)
            or any(type(value) is not bool or not value for value in required_true)
            or not (
                legacy_provenance
                or multi_segment_provenance
                or registered_measure_provenance
                or registered_single_measure_provenance
            )
            or not (typed_legacy_measure or canonical_measure)
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.AUTHORITY_VIOLATION,
                "support-source receipt crossed its source-only authority boundary",
            )
        expected_seal = content_hash(
            self._payload(), prefix=_PREFIX + "receipt-integrity-"
        )
        if self._integrity_seal and self._integrity_seal != expected_seal:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.MUTATED_CUSTODY,
                "support-source receipt integrity changed",
            )
        object.__setattr__(self, "_integrity_seal", expected_seal)

    def assert_integrity(self) -> None:
        expected = content_hash(self._payload(), prefix=_PREFIX + "receipt-integrity-")
        if expected != self._integrity_seal:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.MUTATED_CUSTODY,
                "support-source receipt integrity changed",
            )

    def _payload(self) -> dict[str, object]:
        return {
            item.name: (
                getattr(self, item.name).value
                if isinstance(getattr(self, item.name), Enum)
                else getattr(self, item.name)
            )
            for item in self.__dataclass_fields__.values()
            if not item.name.startswith("_")
        }

    @property
    def receipt_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "receipt-")


class V1SignedWorkBlockSupportCoefficientLease:
    __slots__ = ("_owner", "_token", "tensor", "_closed")

    def __init__(
        self,
        owner: "V1SignedWorkBlockSupportSourceDeviceHandle",
        token: str,
        tensor: Any,
    ) -> None:
        self._owner, self._token, self.tensor, self._closed = (
            owner,
            token,
            tensor,
            False,
        )

    def close(self) -> None:
        if not self._closed:
            try:
                self._owner._close_lease(self._token)
            finally:
                self._closed = True

    def __enter__(self) -> "V1SignedWorkBlockSupportCoefficientLease":
        self._owner.assert_live()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __reduce__(self) -> NoReturn:
        raise TypeError("live device leases are not serializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live device leases are not serializable")


class V1SignedWorkBlockSupportSourceDeviceHandle:
    __slots__ = (
        "receipt",
        "_coefficients",
        "_guard",
        "_invariant",
        "_guard_invariant",
        "_leases",
        "_lease_counter",
        "_evicted",
    )

    def __init__(
        self,
        *,
        coefficients: Any,
        receipt: V1SignedWorkBlockSupportSourceReceipt,
        _issuer: object,
    ) -> None:
        if (
            _issuer is not _HANDLE_ISSUER
            or type(receipt) is not V1SignedWorkBlockSupportSourceReceipt
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
                "support-source handles are issuer-only",
            )
        receipt.assert_integrity()
        if (
            type(coefficients) is not torch.Tensor
            or coefficients.dtype != torch.float64
            or tuple(coefficients.shape) != (receipt.route_count,)
            or not coefficients.is_contiguous()
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
                "support coefficients are malformed",
            )
        self.receipt = receipt
        self._coefficients = coefficients
        self._guard = coefficients.detach().clone()
        self._invariant = _tensor_invariant(coefficients)
        self._guard_invariant = _tensor_invariant(self._guard)
        self._leases: set[str] = set()
        self._lease_counter = 0
        self._evicted = False

    def __reduce__(self) -> NoReturn:
        raise TypeError("live device handles are not serializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live device handles are not serializable")

    def assert_live(self) -> None:
        if self._evicted:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.EVICTED,
                "coefficient handle was evicted",
            )
        self.receipt.assert_integrity()
        if (
            _tensor_invariant(self._coefficients) != self._invariant
            or _tensor_invariant(self._guard) != self._guard_invariant
            or not bool(torch.equal(self._coefficients, self._guard))
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.MUTATED_CUSTODY,
                "route coefficient tensor mutated",
            )

    def open_coefficient_lease(self) -> V1SignedWorkBlockSupportCoefficientLease:
        self.assert_live()
        self._lease_counter += 1
        token = content_hash(
            {"receipt": self.receipt.receipt_hash, "counter": self._lease_counter},
            prefix=_PREFIX + "lease-",
        )
        self._leases.add(token)
        return V1SignedWorkBlockSupportCoefficientLease(self, token, self._coefficients)

    def _close_lease(self, token: str) -> None:
        if token not in self._leases:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.STALE_IDENTITY,
                "unknown coefficient lease",
            )
        try:
            self.assert_live()
        finally:
            self._leases.discard(token)

    def evict(self) -> None:
        self.assert_live()
        if self._leases:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.ACTIVE_LEASE,
                "cannot evict with active coefficient lease",
            )
        self._coefficients = None
        self._guard = None
        self._invariant = ()
        self._guard_invariant = ()
        self._evicted = True


@dataclass(frozen=True, slots=True)
class V1SignedWorkBlockSupportCPUReferenceEvaluation:
    candidate_minus_preferred_exact: Fraction
    retained_plane_center_exact: Fraction
    certified_plane_lower_exact: Fraction
    lower_plane_inequality_holds: bool


def _validate_packed_preferred_action(
    *,
    packed_action: V1PackedRemovalAction,
    parent_mask: HardMask,
    source_registration: V1SignedWorkVariableShellSourceRegistration,
) -> None:
    """Bind the irreversible Boolean action to canonical packed-mask content."""

    if (
        type(packed_action) is not V1PackedRemovalAction
        or type(parent_mask) is not HardMask
        or type(source_registration) is not V1SignedWorkVariableShellSourceRegistration
    ):
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
            "preferred action requires exact packed action, parent, and source types",
        )
    child = packed_action.resulting_mask
    if (
        parent_mask.mask_hash != source_registration.current_mask_hash
        or parent_mask.route_axis_hash != source_registration.route_axis_hash
        or child.route_axis_hash != source_registration.route_axis_hash
        or parent_mask.route_count != child.route_count
    ):
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.STALE_IDENTITY,
            "packed preferred action does not share the registered parent/route axis",
        )
    try:
        removed_count, removed_hash = child.removed_index_count_and_hash_from(
            parent_mask
        )
    except ValueError as error:
        raise V1SignedWorkBlockSupportRefusal(
            V1SignedWorkBlockSupportRefusalCode.STALE_IDENTITY,
            "packed preferred action is not an irreversible child of the parent",
        ) from error
    if (
        removed_count != packed_action.removed_count
        or removed_hash != packed_action.removed_route_index_hash
    ):
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.STALE_IDENTITY,
            "packed preferred-action delta differs from its declared identity",
        )


def _materialize_packed_preferred_action(
    *,
    parent_packed_bits: bytes,
    child_packed_bits: bytes,
    route_count: int,
    device: Any,
    tile_bytes: int,
) -> tuple[Any, int]:
    action = torch.empty(
        route_count,
        dtype=torch.bool,
        device=device,
    )
    parent_bytes = np.frombuffer(parent_packed_bits, dtype=np.uint8)
    child_bytes = np.frombuffer(child_packed_bits, dtype=np.uint8)
    for byte_start in range(0, int(parent_bytes.size), tile_bytes):
        byte_stop = min(int(parent_bytes.size), byte_start + tile_bytes)
        removed_bytes = np.bitwise_and(
            parent_bytes[byte_start:byte_stop],
            np.bitwise_not(child_bytes[byte_start:byte_stop]),
        )
        unpacked = np.unpackbits(removed_bytes, bitorder="big")
        route_start = byte_start * 8
        route_stop = min(route_count, byte_stop * 8)
        slab = torch.from_numpy(
            unpacked[: route_stop - route_start].astype(np.bool_, copy=False)
        )
        action[route_start:route_stop].copy_(slab.to(device=device))
    return action.contiguous(), int(parent_bytes.size + child_bytes.size)


def _materialize_packed_preferred_action_float64(
    *,
    parent_packed_bits: bytes,
    child_packed_bits: bytes,
    route_count: int,
    device: Any,
    tile_bytes: int,
    production_cuda_required: bool,
) -> tuple[Any, int, int]:
    """Unpack one packed irreversible action without a route-scale CPU slab."""

    if torch is None or type(production_cuda_required) is not bool:
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
            "FP64 packed-action materialization requires PyTorch and an exact mode flag",
        )
    routes = _integer(route_count, "route_count", positive=True)
    tile = _integer(tile_bytes, "tile_bytes", positive=True)
    expected_bytes = (routes + 7) // 8
    if (
        type(parent_packed_bits) is not bytes
        or type(child_packed_bits) is not bytes
        or len(parent_packed_bits) != expected_bytes
        or len(child_packed_bits) != expected_bytes
    ):
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.MUTATED_CUSTODY,
            "packed parent/child byte custody no longer matches the route axis",
        )
    target = torch.device(device)
    if production_cuda_required and target.type != "cuda":
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.AUTHORITY_VIOLATION,
            "production FP64 action materialization requires CUDA",
        )
    if target.type == "cuda" and not torch.cuda.is_available():
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.AUTHORITY_VIOLATION,
            "CUDA action materialization requires an available CUDA runtime",
        )

    parent_bytes = np.frombuffer(parent_packed_bits, dtype=np.uint8)
    child_bytes = np.frombuffer(child_packed_bits, dtype=np.uint8)
    if target.type != "cuda":
        action, transfer_bytes = _materialize_packed_preferred_action(
            parent_packed_bits=parent_packed_bits,
            child_packed_bits=child_packed_bits,
            route_count=routes,
            device=target,
            tile_bytes=tile,
        )
        return action.to(dtype=torch.float64).contiguous(), transfer_bytes, 0

    output = torch.empty(routes, dtype=torch.float64, device=target)
    shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=target)
    maximum_tile_bytes = min(tile, expected_bytes)
    # A conservative live logical-payload ledger includes parent, child,
    # bitwise-not child, removed, shifted and unpacked 8-way uint8 slabs, the
    # FP64 slab, and the eight-byte shift vector.  CUDA allocator/compiler
    # workspace is deliberately left to the measured outer HBM admission.
    logical_tile_payload_upper_bound = 8 + maximum_tile_bytes * (
        4 + 8 + 8 + 8 * 8
    )
    for byte_start in range(0, expected_bytes, tile):
        byte_stop = min(expected_bytes, byte_start + tile)
        parent_tile = torch.from_numpy(
            np.array(parent_bytes[byte_start:byte_stop], copy=True)
        ).to(device=target)
        child_tile = torch.from_numpy(
            np.array(child_bytes[byte_start:byte_stop], copy=True)
        ).to(device=target)
        removed = torch.bitwise_and(parent_tile, torch.bitwise_not(child_tile))
        unpacked = torch.bitwise_and(
            torch.bitwise_right_shift(removed[:, None], shifts[None, :]),
            1,
        ).reshape(-1)
        route_start = byte_start * 8
        route_stop = min(routes, byte_stop * 8)
        output[route_start:route_stop].copy_(
            unpacked[: route_stop - route_start].to(dtype=torch.float64)
        )
    return output.contiguous(), 2 * expected_bytes, logical_tile_payload_upper_bound


def evaluate_v1_signed_work_block_support_cpu_reference(
    *,
    handle: V1SignedWorkBlockSupportSourceDeviceHandle,
    corrected_rows: Any,
    native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt,
    context_weights: Sequence[float],
    preferred_action: Any,
    candidate_action: Any,
) -> V1SignedWorkBlockSupportCPUReferenceEvaluation:
    if (
        handle.receipt.execution_mode
        is not V1SignedWorkBlockSupportExecutionMode.CPU_REFERENCE_TEST_ONLY
    ):
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.AUTHORITY_VIOLATION,
            "exact evaluator is test-only",
        )
    handle.assert_live()
    if any(
        type(item) is not torch.Tensor
        or item.device.type != "cpu"
        or item.dtype != torch.float64
        for item in (corrected_rows, preferred_action, candidate_action)
    ):
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
            "reference operands must be CPU float64 tensors",
        )
    rows = corrected_rows
    count, routes = rows.shape
    if (
        routes != native_fiber_axis_receipt.route_count
        or tuple(preferred_action.shape) != (routes,)
        or tuple(candidate_action.shape) != (routes,)
    ):
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.AXIS_MISMATCH,
            "reference axes do not align",
        )
    weights = _weights(context_weights, count)
    if not bool(
        torch.logical_or(candidate_action == 0.0, candidate_action == 1.0).all().item()
    ):
        _refuse(
            V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
            "candidate must be Boolean",
        )

    def work(action: Any) -> Fraction:
        total = Fraction(0)
        for x in range(count):
            fiber_sum = Fraction(0)
            for cut in native_fiber_axis_receipt.cuts:
                for fiber in range(cut.fiber_count):
                    start, stop = cut.route_span_for_fiber(fiber)
                    response = sum(
                        (
                            Fraction.from_float(float(rows[x, j]))
                            * Fraction.from_float(float(action[j]))
                            for j in range(start, stop)
                        ),
                        Fraction(0),
                    )
                    fiber_sum += abs(response) / native_fiber_axis_receipt.cut_count
            total += Fraction.from_float(weights[x]) * fiber_sum
        return total

    with handle.open_coefficient_lease() as lease:
        coefficients = tuple(
            Fraction.from_float(float(item)) for item in lease.tensor.tolist()
        )
    plane = sum(
        (
            coefficients[j]
            * (
                Fraction.from_float(float(candidate_action[j]))
                - Fraction.from_float(float(preferred_action[j]))
            )
            for j in range(routes)
        ),
        Fraction(0),
    )
    difference = work(candidate_action) - work(preferred_action)
    lower = plane - Fraction.from_float(handle.receipt.preferred_plane_deficit_upper)
    return V1SignedWorkBlockSupportCPUReferenceEvaluation(
        difference, plane, lower, difference >= lower
    )


class V1SignedWorkBlockSupportSourceCUDAAccumulator:
    def __init__(
        self,
        *,
        registration: V1SignedWorkBlockSupportSourceRegistration,
        source_registration: V1SignedWorkVariableShellSourceRegistration,
        native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt,
        preferred_action_binding: V1SignedWorkBlockSupportPreferredActionBinding,
        design: V1SignedWorkBlockSupportCUDAExecutionDesign,
        declared_device: str,
        device_total_memory_bytes: int,
        device_free_memory_bytes: int,
        current_reserved_device_bytes: int,
    ) -> None:
        if torch is None:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE, "torch is required"
            )
        if (
            type(registration) is not V1SignedWorkBlockSupportSourceRegistration
            or type(source_registration)
            is not V1SignedWorkVariableShellSourceRegistration
            or type(native_fiber_axis_receipt)
            is not V1NativeDestinationFiberAxisReceipt
            or type(preferred_action_binding)
            is not V1SignedWorkBlockSupportPreferredActionBinding
            or type(design) is not V1SignedWorkBlockSupportCUDAExecutionDesign
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
                "accumulator inputs have wrong type",
            )
        registration.assert_integrity()
        if (
            registration.source_registration_hash
            != source_registration.registration_hash
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.STALE_IDENTITY,
                "source registration changed",
            )
        if (
            registration.native_fiber_axis_receipt_hash
            != native_fiber_axis_receipt.receipt_hash
            or registration.native_fiber_measure_hash
            != native_fiber_axis_receipt.measure_hash
            or registration.route_axis_hash != native_fiber_axis_receipt.route_axis_hash
            or registration.route_id_axis_hash
            != native_fiber_axis_receipt.route_id_axis_hash
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.STALE_IDENTITY,
                "native-fiber/route identity changed",
            )
        native_fiber_axis_receipt.assert_integrity()
        device = torch.device(_text(declared_device, "declared device"))
        if (
            design.execution_mode
            is V1SignedWorkBlockSupportExecutionMode.CUDA_PRODUCTION
        ):
            if (
                device.type != "cuda"
                or not torch.are_deterministic_algorithms_enabled()
                or torch.is_deterministic_algorithms_warn_only_enabled()
            ):
                _refuse(
                    V1SignedWorkBlockSupportRefusalCode.DETERMINISM_DISABLED,
                    "production requires deterministic CUDA",
                )
        elif device.type != "cpu":
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.INVALID_DESIGN,
                "reference mode requires CPU",
            )
        preferred_action_binding.assert_live()
        if (
            registration.preferred_action_binding_hash
            != preferred_action_binding.binding_hash
            or registration.preferred_action_hash
            != preferred_action_binding.preferred_action_hash
            or registration.preferred_resulting_mask_hash
            != preferred_action_binding.preferred_resulting_mask_hash
            or registration.preferred_removed_route_index_hash
            != preferred_action_binding.preferred_removed_route_index_hash
            or registration.preferred_selected_count
            != preferred_action_binding.preferred_selected_count
            or registration.current_mask_hash
            != preferred_action_binding.current_mask_hash
            or registration.route_axis_hash != preferred_action_binding.route_axis_hash
            or preferred_action_binding.route_count
            != native_fiber_axis_receipt.route_count
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.STALE_IDENTITY,
                "preferred-action binding differs from the frozen support registration",
            )
        routes = native_fiber_axis_receipt.route_count
        admission_memory = (
            cuda_reusable_memory_bytes(
                device, framework=torch, free_bytes=device_free_memory_bytes
            ),
            int(torch.cuda.memory_allocated(device)),
        ) if device.type == "cuda" else None
        forecast = build_v1_signed_work_block_support_hbm_forecast(
            registration=registration,
            native_fiber_axis_receipt=native_fiber_axis_receipt,
            design=design,
            device_total_memory_bytes=device_total_memory_bytes,
            device_free_memory_bytes=device_free_memory_bytes,
            current_reserved_device_bytes=current_reserved_device_bytes,
            _admission_memory=admission_memory,
        )
        if not forecast.fits:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.HBM_FORECAST_EXCEEDED,
                "deterministic HBM forecast is unsafe",
            )
        preferred_tensor, packed_host_bytes_read = preferred_action_binding.materialize(
            device=device,
            tile_bytes=design.packed_byte_tile_size,
        )
        self.registration, self.source_registration, self.axis, self.design = (
            registration,
            source_registration,
            native_fiber_axis_receipt,
            design,
        )
        self.forecast, self._device, self._preferred = (
            forecast,
            device,
            preferred_tensor,
        )
        self._preferred_guard = preferred_tensor.detach().clone()
        self._preferred_invariant = _tensor_invariant(preferred_tensor)
        self._preferred_guard_invariant = _tensor_invariant(self._preferred_guard)
        self._packed_host_bytes_read = packed_host_bytes_read
        self._scalar_device_observation_count = 0
        self._coefficients = torch.zeros(routes, dtype=torch.float64, device=device)
        self._weights = torch.tensor(
            registration.context_weights, dtype=torch.float64, device=device
        )
        self._weight_radii = torch.tensor(
            registration.measure_contract.context_weight_binary64_radii_upper,
            dtype=torch.float64,
            device=device,
        )
        self._ambiguous_upper = torch.zeros((), dtype=torch.float64, device=device)
        self._coefficient_radius_upper = torch.zeros(
            (), dtype=torch.float64, device=device
        )
        self._source_finite = torch.ones((), dtype=torch.bool, device=device)
        self._positive = torch.zeros((), dtype=torch.int64, device=device)
        self._negative = torch.zeros((), dtype=torch.int64, device=device)
        self._ambiguous = torch.zeros((), dtype=torch.int64, device=device)
        self._context_cursor = self._active_count = self._route_cursor = 0
        self._active_hashes: dict[str, str] | None = None
        self._binding_chain = content_hash(
            {"registration": registration.registration_hash, "empty": True},
            prefix=_PREFIX + "binding-chain-",
        )
        self._binding_count = 0
        self._aborted = False
        self._sealed = False

    @property
    def is_aborted(self) -> bool:
        """Report whether partial accumulator-owned tensor custody was released."""

        return bool(self._aborted)

    def _require_unaborted(self) -> None:
        if self._aborted:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.EVICTED,
                "block-support accumulator was aborted",
            )

    def _check_preferred(self, *, deep: bool = False) -> None:
        self._require_unaborted()
        if (
            _tensor_invariant(self._preferred) != self._preferred_invariant
            or _tensor_invariant(self._preferred_guard)
            != self._preferred_guard_invariant
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.MUTATED_CUSTODY,
                "preferred action mutated",
            )
        if deep:
            equal = bool(torch.equal(self._preferred, self._preferred_guard))
            if self._device.type == "cuda":
                self._scalar_device_observation_count += 1
            if not equal:
                _refuse(
                    V1SignedWorkBlockSupportRefusalCode.MUTATED_CUSTODY,
                    "preferred action content mutated",
                )

    def begin_packet(
        self,
        *,
        context_ids: Sequence[str],
        draw_ids: Sequence[str],
        measurement_ids: Sequence[str],
        capture_ids: Sequence[str],
        source_packet_hashes: Sequence[str],
    ) -> None:
        self._require_unaborted()
        if self._sealed or self._active_count:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.ORDER_MISMATCH,
                "packet cannot begin now",
            )
        axes = tuple(
            tuple(item)
            for item in (
                context_ids,
                draw_ids,
                measurement_ids,
                capture_ids,
                source_packet_hashes,
            )
        )
        if len({len(item) for item in axes}) != 1:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.AXIS_MISMATCH,
                "packet axes do not align",
            )
        count = len(axes[0])
        start = self._context_cursor
        stop = start + count
        registered = (
            self.registration.context_ids,
            self.registration.draw_ids,
            self.registration.measurement_ids,
            self.registration.capture_ids,
            self.registration.source_packet_hashes,
        )
        if (
            count <= 0
            or count > self.design.maximum_batch_size
            or stop > self.registration.context_count
            or any(
                axis != tuple(owner[start:stop])
                for axis, owner in zip(axes, registered)
            )
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.ORDER_MISMATCH,
                "packet is not the next registered group slice",
            )
        self._active_count, self._route_cursor = count, 0
        self._active_hashes = {
            "context": _source_axis_hash("active_packet_context", axes[0]),
            "draw": _source_axis_hash("active_packet_draw", axes[1]),
            "measurement": _source_axis_hash("active_packet_measurement", axes[2]),
            "capture": _source_axis_hash("active_packet_capture", axes[3]),
            "source_packet": _source_axis_hash("active_packet_source_packet", axes[4]),
        }

    def consume_route_chunk(
        self,
        *,
        route_start: int,
        corrected_evidence_rows: Any,
        chunk_binding: V1SignedWorkVariableShellSourceChunkBinding,
    ) -> None:
        self._require_unaborted()
        self._check_preferred()
        if not self._active_count or self._active_hashes is None:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.PARTIAL_SOURCE,
                "begin_packet is required",
            )
        start = _integer(route_start, "route start")
        if (
            start != self._route_cursor
            or type(chunk_binding) is not V1SignedWorkVariableShellSourceChunkBinding
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.ORDER_MISMATCH,
                "chunk is out of order or unbound",
            )
        if (
            type(corrected_evidence_rows) is not torch.Tensor
            or corrected_evidence_rows.dtype != torch.float64
            or corrected_evidence_rows.device != self._device
            or corrected_evidence_rows.ndim != 2
            or not corrected_evidence_rows.is_contiguous()
            or corrected_evidence_rows.requires_grad
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
                "corrected rows must be contiguous detached device FP64",
            )
        batch, width = map(int, corrected_evidence_rows.shape)
        stop = start + width
        if (
            batch != self._active_count
            or width <= 0
            or width > self.design.route_chunk_size
            or stop > self.axis.route_count
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.AXIS_MISMATCH,
                "corrected chunk shape/span is invalid",
            )
        b = chunk_binding
        h = self._active_hashes
        expected_ordinal = (
            self.registration.source_context_start_ordinal + self._context_cursor
        )
        if (
            b.registration_hash != self.source_registration.registration_hash
            or b.active_packet_ordinal != expected_ordinal
            or b.active_packet_context_axis_hash != h["context"]
            or b.active_packet_draw_axis_hash != h["draw"]
            or b.active_packet_measurement_axis_hash != h["measurement"]
            or b.active_packet_capture_axis_hash != h["capture"]
            or b.active_packet_source_packet_axis_hash != h["source_packet"]
            or b.route_start != start
            or b.route_stop != stop
            or b.route_axis_hash != self.registration.route_axis_hash
            or b.route_id_axis_hash != self.registration.route_id_axis_hash
            or b.witness_stream_source_hash
            != self.registration.witness_stream_source_hash
            or b.same_capture_vjp_source_hash
            != self.registration.same_capture_vjp_source_hash
            or b.corrected_evidence_stream_source_hash
            != self.registration.corrected_evidence_stream_source_hash
            or b.literal_j_stream_source_hash
            != self.registration.literal_j_stream_source_hash
            or b.corrected_evidence_tensor_identity_hash
            != _tensor_source_identity_hash(
                corrected_evidence_rows, name="corrected evidence rows"
            )
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.STALE_IDENTITY,
                "chunk differs from exact live D_W2 binding",
            )
        self._accumulate_bound_chunk(
            start=start,
            stop=stop,
            corrected_evidence_rows=corrected_evidence_rows,
            durable_binding_hash=b.durable_binding_hash,
        )

    def _accumulate_bound_chunk(
        self,
        *,
        start: int,
        stop: int,
        corrected_evidence_rows: Any,
        durable_binding_hash: str,
    ) -> None:
        """Apply the common bounded numerical kernel after custody validation."""

        batch = int(corrected_evidence_rows.shape[0])
        width = stop - start
        self._source_finite &= torch.isfinite(corrected_evidence_rows).all()
        cut = self.axis.cut_for_route(start)
        if (
            stop > cut.route_stop
            or (start - cut.route_offset) % cut.routes_per_fiber
            or width % cut.routes_per_fiber
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.AXIS_MISMATCH,
                "chunk must be cut-contained and complete-fiber aligned",
            )
        fibers, rpf = width // cut.routes_per_fiber, cut.routes_per_fiber
        a = corrected_evidence_rows.reshape(batch, fibers, rpf)
        p = self._preferred[start:stop].reshape(fibers, rpf).to(torch.float64)
        center = torch.einsum("bfr,fr->bf", a, p)
        abs_mass = torch.einsum("bfr,fr->bf", torch.abs(a), torch.abs(p))
        dot_ops = 2 * rpf + 1
        gamma = _gamma(dot_ops)
        abs_upper = _up(abs_mass / (1.0 - gamma))
        radius = _up(gamma * abs_upper + _F64_MIN * float(dot_ops))
        lower, upper = _down(center - radius), _up(center + radius)
        positive, negative = lower > 0.0, upper < 0.0
        y = positive.to(torch.float64) - negative.to(torch.float64)
        ambiguous = ~(positive | negative)
        packet_weights = self._weights[
            self._context_cursor : self._context_cursor + batch
        ]
        packet_weight_radii = self._weight_radii[
            self._context_cursor : self._context_cursor + batch
        ]
        signed_weights = packet_weights[:, None] * y
        numerator = torch.einsum("bf,bfr->fr", signed_weights, a).reshape(width)
        numerator_abs = torch.einsum(
            "bf,bfr->fr", torch.abs(signed_weights), torch.abs(a)
        ).reshape(width)
        coeff_ops = 2 * batch + 2
        coeff_gamma = _gamma(coeff_ops)
        num_upper = _up(numerator_abs / (1.0 - coeff_gamma))
        num_radius = _up(coeff_gamma * num_upper + _F64_MIN * float(coeff_ops))
        # The exact measure is rational.  Charge the complete error incurred by
        # consuming its deterministic binary64 centers, rather than silently
        # redefining the signed-F estimand in floating-point units.
        weight_error = torch.einsum(
            "bf,bfr->fr",
            packet_weight_radii[:, None] * torch.abs(y),
            torch.abs(a),
        ).reshape(width)
        weight_error_gamma = _gamma(2 * batch + 2)
        weight_error_upper = _up(
            weight_error / (1.0 - weight_error_gamma)
            + _F64_MIN * float(2 * batch + 2)
        )
        num_radius = _up(num_radius + weight_error_upper)
        chunk_coeff = numerator / float(self.axis.cut_count)
        div_radius = _up(
            _F64_EPS * torch.abs(chunk_coeff) / (1.0 - _F64_EPS) + _F64_MIN
        )
        chunk_radius = _up(num_radius / float(self.axis.cut_count) + div_radius)
        old = self._coefficients[start:stop]
        add_radius = _up(
            _F64_EPS * (torch.abs(old) + torch.abs(chunk_coeff)) / (1.0 - _F64_EPS)
            + _F64_MIN
        )
        self._coefficients[start:stop] = old + chunk_coeff
        radius_values = _up(chunk_radius + add_radius)
        rg = _gamma(width + 1)
        self._coefficient_radius_upper = _up(
            self._coefficient_radius_upper
            + _up(torch.sum(radius_values) / (1.0 - rg) + _F64_MIN * float(width + 1))
        )
        response_upper = _up(torch.abs(center) + radius)
        ambiguous_terms = (
            _up(packet_weights + packet_weight_radii)[:, None]
            * torch.where(ambiguous, response_upper, torch.zeros_like(response_upper))
            / float(self.axis.cut_count)
        )
        cells = batch * fibers
        # Each term includes a weight multiplication and cut-count division in
        # addition to the reduction; use the complete operation envelope.
        ag = _gamma(3 * cells + 3)
        self._ambiguous_upper = _up(
            self._ambiguous_upper
            + _up(torch.sum(ambiguous_terms) / (1.0 - ag) + _F64_MIN * float(cells + 1))
        )
        self._positive += torch.sum(positive, dtype=torch.int64)
        self._negative += torch.sum(negative, dtype=torch.int64)
        self._ambiguous += torch.sum(ambiguous, dtype=torch.int64)
        self._route_cursor = stop
        self._binding_count += 1
        self._binding_chain = content_hash(
            {
                "prior": self._binding_chain,
                "ordinal": self._binding_count - 1,
                "durable_binding_hash": durable_binding_hash,
            },
            prefix=_PREFIX + "binding-chain-",
        )

    def end_packet(self) -> None:
        self._require_unaborted()
        if not self._active_count or self._route_cursor != self.axis.route_count:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.PARTIAL_SOURCE,
                "packet did not cover the full route axis",
            )
        self._context_cursor += self._active_count
        self._active_count = self._route_cursor = 0
        self._active_hashes = None

    def seal(self) -> V1SignedWorkBlockSupportSourceDeviceHandle:
        self._require_unaborted()
        self._check_preferred(deep=True)
        if (
            self._sealed
            or self._active_count
            or self._context_cursor != self.registration.context_count
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.PARTIAL_SOURCE,
                "registered group is incomplete or already sealed",
            )
        finite = bool(
            (
                torch.isfinite(self._ambiguous_upper)
                & torch.isfinite(self._coefficient_radius_upper)
                & torch.isfinite(self._coefficients).all()
                & self._source_finite
            ).item()
        )
        if self._device.type == "cuda":
            self._scalar_device_observation_count += 1
        if not finite:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.NUMERICAL_REFUSAL,
                "support construction overflowed",
            )
        ambiguous = float(self._ambiguous_upper.item())
        coeff_radius = float(self._coefficient_radius_upper.item())
        positive_count = int(self._positive.item())
        negative_count = int(self._negative.item())
        ambiguous_count = int(self._ambiguous.item())
        if self._device.type == "cuda":
            self._scalar_device_observation_count += 5
        deficit = math.nextafter(ambiguous + coeff_radius, math.inf)
        if not math.isfinite(deficit):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.NUMERICAL_REFUSAL,
                "preferred-plane deficit overflowed binary64",
            )
        r = self.registration
        measure = r.measure_contract
        receipt = V1SignedWorkBlockSupportSourceReceipt(
            registration_hash=r.registration_hash,
            source_registration_hash=r.source_registration_hash,
            source_role=r.source_role,
            model_identity_hash=r.model_identity_hash,
            chart_identity_hash=r.chart_identity_hash,
            current_mask_hash=r.current_mask_hash,
            proposal_generation=r.proposal_generation,
            route_axis_hash=r.route_axis_hash,
            route_id_axis_hash=r.route_id_axis_hash,
            native_fiber_axis_receipt_hash=r.native_fiber_axis_receipt_hash,
            native_fiber_measure_hash=r.native_fiber_measure_hash,
            registered_d_prefix_hash=r.registered_d_prefix_hash,
            registered_d_w2_prefix_hash=r.registered_d_w2_prefix_hash,
            alpha_ledger_id=r.alpha_ledger_id,
            alpha_coordinate_id=r.alpha_coordinate_id,
            alpha_lineage_hash=r.alpha_lineage_hash,
            context_group_id=r.context_group_id,
            block_partition_hash=r.block_partition_hash,
            block_ordinal=r.block_ordinal,
            preferred_action_binding_hash=r.preferred_action_binding_hash,
            preferred_action_hash=r.preferred_action_hash,
            preferred_resulting_mask_hash=r.preferred_resulting_mask_hash,
            preferred_removed_route_index_hash=(r.preferred_removed_route_index_hash),
            preferred_selected_count=r.preferred_selected_count,
            context_axis_hash=_source_axis_hash("block_support_context", r.context_ids),
            draw_axis_hash=_source_axis_hash("block_support_draw", r.draw_ids),
            measurement_axis_hash=_source_axis_hash(
                "block_support_measurement", r.measurement_ids
            ),
            capture_axis_hash=_source_axis_hash("block_support_capture", r.capture_ids),
            source_packet_axis_hash=_source_axis_hash(
                "block_support_source_packet", r.source_packet_hashes
            ),
            context_weight_hash=content_hash(
                r.context_weights, prefix=_PREFIX + "weights-"
            ),
            context_count=r.context_count,
            route_count=self.axis.route_count,
            fiber_count=self.axis.fiber_count,
            cut_count=self.axis.cut_count,
            certified_positive_cell_count=positive_count,
            certified_negative_cell_count=negative_count,
            ambiguous_cell_count=ambiguous_count,
            ambiguous_preferred_response_upper=ambiguous,
            coefficient_construction_l1_radius_upper=coeff_radius,
            preferred_plane_deficit_upper=deficit,
            consumed_chunk_binding_count=self._binding_count,
            consumed_durable_chunk_binding_chain_hash=self._binding_chain,
            hbm_forecast_hash=self.forecast.forecast_hash,
            execution_mode=self.design.execution_mode,
            preferred_action_representation="canonical_packed_parent_child_delta_v1",
            packed_host_bytes_read=self._packed_host_bytes_read,
            scalar_device_to_host_transfer_bytes=(
                8 * self._scalar_device_observation_count
            ),
            measure_contract_hash=measure.contract_hash,
            measure_registered_context_axis_hash=(
                measure.registered_context_axis_hash
            ),
            measure_kind=measure.measure_kind,
            measure_design_hash=measure.measure_design_hash,
            measure_weight_receipt_hash=measure.measure_weight_receipt_hash,
            context_stratum_axis_hash=measure.context_stratum_axis_hash,
            exact_context_weight_fraction_axis_hash=(
                measure.exact_context_weight_fraction_axis_hash
            ),
            uniform_finite_population_group_mean=(
                r.uniform_finite_population_group_mean
            ),
            canonical_full_population_f_d_measure_authority=(
                r.full_population_canonical_f_d_measure_authority
            ),
            schema=(
                _REGISTERED_MEASURE_RECEIPT_SCHEMA
                if r.full_population_canonical_f_d_measure_authority
                else _RECEIPT_SCHEMA
            ),
            _issuer=_RECEIPT_ISSUER,
        )
        coefficients = self._coefficients
        handle = V1SignedWorkBlockSupportSourceDeviceHandle(
            coefficients=coefficients,
            receipt=receipt,
            _issuer=_HANDLE_ISSUER,
        )
        self._coefficients = None
        self._preferred = None
        self._preferred_guard = None
        self._preferred_invariant = ()
        self._preferred_guard_invariant = ()
        self._sealed = True
        return handle

    def abort_unsealed(self) -> None:
        """Idempotently release all tensors still owned before ``seal``.

        Once sealed, coefficient ownership belongs to the returned device
        handle and this method refuses instead of invalidating published
        custody.  Partial packet state is deliberately discarded: callers use
        this path only while unwinding a failed pre-outcome construction.
        """

        if self._aborted:
            return
        if self._sealed:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.AUTHORITY_VIOLATION,
                "sealed block-support custody must be released through its handle",
            )
        for name in (
            "_preferred",
            "_preferred_guard",
            "_coefficients",
            "_weights",
            "_weight_radii",
            "_ambiguous_upper",
            "_coefficient_radius_upper",
            "_source_finite",
            "_positive",
            "_negative",
            "_ambiguous",
        ):
            setattr(self, name, None)
        self._preferred_invariant = ()
        self._preferred_guard_invariant = ()
        self._active_hashes = None
        self._active_count = 0
        self._route_cursor = 0
        self._aborted = True


class V1SignedWorkBlockSupportMultiSegmentSourceCUDAAccumulator(
    V1SignedWorkBlockSupportSourceCUDAAccumulator
):
    """Stream one logical support group from ordered source-local owners."""

    def __init__(
        self,
        *,
        registration: V1SignedWorkBlockSupportMultiSegmentSourceRegistration,
        structural_source_registration: V1SignedWorkVariableShellSourceRegistration,
        segment_source_registrations: Sequence[
            V1SignedWorkVariableShellSourceRegistration
        ],
        native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt,
        preferred_action_binding: V1SignedWorkBlockSupportPreferredActionBinding,
        design: V1SignedWorkBlockSupportCUDAExecutionDesign,
        declared_device: str,
        device_total_memory_bytes: int,
        device_free_memory_bytes: int,
        current_reserved_device_bytes: int,
    ) -> None:
        if (
            type(registration)
            is not V1SignedWorkBlockSupportMultiSegmentSourceRegistration
            or type(structural_source_registration)
            is not V1SignedWorkVariableShellSourceRegistration
            or type(segment_source_registrations) not in {tuple, list}
            or any(
                type(item) is not V1SignedWorkVariableShellSourceRegistration
                for item in segment_source_registrations
            )
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
                "multi-segment accumulator inputs have wrong type",
            )
        registration.assert_integrity()
        segments = tuple(segment_source_registrations)
        segment_hashes = tuple(item.registration_hash for item in segments)
        numerical_hashes = tuple(
            _segment_numerical_source_hash(item) for item in segments
        )
        if (
            structural_source_registration.registration_hash
            != registration.structural_source_registration_hash
            or segment_hashes != registration.segment_source_registration_hashes
            or numerical_hashes != registration.segment_numerical_source_hashes
            or _multi_segment_source_chain_hash(numerical_hashes)
            != registration.numerical_source_chain_hash
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.STALE_IDENTITY,
                "structural or physical source registrations changed",
            )
        super().__init__(
            registration=registration.structural_group_registration,
            source_registration=structural_source_registration,
            native_fiber_axis_receipt=native_fiber_axis_receipt,
            preferred_action_binding=preferred_action_binding,
            design=design,
            declared_device=declared_device,
            device_total_memory_bytes=device_total_memory_bytes,
            device_free_memory_bytes=device_free_memory_bytes,
            current_reserved_device_bytes=current_reserved_device_bytes,
        )
        self.multi_segment_registration = registration
        self.segment_source_registrations = segments
        self._segment_cursor = 0
        self._segment_local_cursor = 0
        self._active_segment_ordinal: int | None = None
        self._binding_chain = content_hash(
            {
                "registration": registration.registration_hash,
                "numerical_source_chain_hash": registration.numerical_source_chain_hash,
                "empty": True,
            },
            prefix=_PREFIX + "multi-segment-binding-chain-",
        )

    def begin_packet(
        self,
        *,
        segment_ordinal: int,
        context_ids: Sequence[str],
        draw_ids: Sequence[str],
        measurement_ids: Sequence[str],
        capture_ids: Sequence[str],
        source_packet_hashes: Sequence[str],
    ) -> None:
        self._require_unaborted()
        if self._sealed or self._active_count:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.ORDER_MISMATCH,
                "packet cannot begin now",
            )
        ordinal = _integer(segment_ordinal, "segment ordinal")
        if ordinal != self._segment_cursor or ordinal >= len(
            self.segment_source_registrations
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.ORDER_MISMATCH,
                "packet does not belong to the next registered source segment",
            )
        axes = tuple(
            tuple(item)
            for item in (
                context_ids,
                draw_ids,
                measurement_ids,
                capture_ids,
                source_packet_hashes,
            )
        )
        if len({len(item) for item in axes}) != 1:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.AXIS_MISMATCH,
                "packet axes do not align",
            )
        count = len(axes[0])
        segment = self.segment_source_registrations[ordinal]
        local_start = self._segment_local_cursor
        local_stop = local_start + count
        global_start = self._context_cursor
        global_stop = global_start + count
        segment_axes = (
            segment.registered_context_ids,
            segment.registered_draw_ids,
            segment.registered_measurement_ids,
            segment.registered_capture_ids,
            segment.registered_source_packet_hashes,
        )
        structural = self.registration
        structural_axes = (
            structural.context_ids,
            structural.draw_ids,
            structural.measurement_ids,
            structural.capture_ids,
            structural.source_packet_hashes,
        )
        if (
            count <= 0
            or count > self.design.maximum_batch_size
            or local_stop > segment.context_count
            or global_stop > structural.context_count
            or any(
                axis != tuple(owner[local_start:local_stop])
                for axis, owner in zip(axes, segment_axes)
            )
            or any(
                axis != tuple(owner[global_start:global_stop])
                for axis, owner in zip(axes, structural_axes)
            )
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.ORDER_MISMATCH,
                "packet is not the next exact source-local and structural slice",
            )
        self._active_count, self._route_cursor = count, 0
        self._active_segment_ordinal = ordinal
        self._active_hashes = {
            "context": _source_axis_hash("active_packet_context", axes[0]),
            "draw": _source_axis_hash("active_packet_draw", axes[1]),
            "measurement": _source_axis_hash("active_packet_measurement", axes[2]),
            "capture": _source_axis_hash("active_packet_capture", axes[3]),
            "source_packet": _source_axis_hash("active_packet_source_packet", axes[4]),
        }

    def consume_route_chunk(
        self,
        *,
        route_start: int,
        corrected_evidence_rows: Any,
        chunk_binding: V1SignedWorkVariableShellSourceChunkBinding,
    ) -> None:
        self._check_preferred()
        if (
            not self._active_count
            or self._active_hashes is None
            or self._active_segment_ordinal is None
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.PARTIAL_SOURCE,
                "begin_packet is required",
            )
        start = _integer(route_start, "route start")
        if (
            start != self._route_cursor
            or type(chunk_binding) is not V1SignedWorkVariableShellSourceChunkBinding
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.ORDER_MISMATCH,
                "chunk is out of order or unbound",
            )
        if (
            type(corrected_evidence_rows) is not torch.Tensor
            or corrected_evidence_rows.dtype != torch.float64
            or corrected_evidence_rows.device != self._device
            or corrected_evidence_rows.ndim != 2
            or not corrected_evidence_rows.is_contiguous()
            or corrected_evidence_rows.requires_grad
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.INVALID_TYPE,
                "corrected rows must be contiguous detached device FP64",
            )
        batch, width = map(int, corrected_evidence_rows.shape)
        stop = start + width
        if (
            batch != self._active_count
            or width <= 0
            or width > self.design.route_chunk_size
            or stop > self.axis.route_count
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.AXIS_MISMATCH,
                "corrected chunk shape/span is invalid",
            )
        segment_ordinal = self._active_segment_ordinal
        segment = self.segment_source_registrations[segment_ordinal]
        binding = chunk_binding
        hashes = self._active_hashes
        if (
            binding.registration_hash != segment.registration_hash
            or binding.active_packet_ordinal != self._segment_local_cursor
            or binding.active_packet_context_axis_hash != hashes["context"]
            or binding.active_packet_draw_axis_hash != hashes["draw"]
            or binding.active_packet_measurement_axis_hash != hashes["measurement"]
            or binding.active_packet_capture_axis_hash != hashes["capture"]
            or binding.active_packet_source_packet_axis_hash != hashes["source_packet"]
            or binding.route_start != start
            or binding.route_stop != stop
            or binding.route_axis_hash != segment.route_axis_hash
            or binding.route_id_axis_hash != segment.route_id_axis_hash
            or binding.witness_stream_source_hash != segment.witness_stream_source_hash
            or binding.same_capture_vjp_source_hash
            != segment.same_capture_vjp_source_hash
            or binding.corrected_evidence_stream_source_hash
            != segment.corrected_evidence_stream_source_hash
            or binding.literal_j_stream_source_hash
            != segment.literal_j_stream_source_hash
            or binding.corrected_evidence_tensor_identity_hash
            != _tensor_source_identity_hash(
                corrected_evidence_rows, name="corrected evidence rows"
            )
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.STALE_IDENTITY,
                "chunk differs from its exact source-local segment binding",
            )
        durable = content_hash(
            {
                "segment_ordinal": segment_ordinal,
                "segment_registration_hash": segment.registration_hash,
                "durable_binding_hash": binding.durable_binding_hash,
            },
            prefix=_PREFIX + "multi-segment-durable-binding-",
        )
        self._accumulate_bound_chunk(
            start=start,
            stop=stop,
            corrected_evidence_rows=corrected_evidence_rows,
            durable_binding_hash=durable,
        )

    def end_packet(self) -> None:
        if self._active_segment_ordinal is None:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.PARTIAL_SOURCE,
                "no source-local packet is active",
            )
        completed = self._active_count
        segment = self.segment_source_registrations[self._active_segment_ordinal]
        super().end_packet()
        self._segment_local_cursor += completed
        if self._segment_local_cursor == segment.context_count:
            self._segment_cursor += 1
            self._segment_local_cursor = 0
        self._active_segment_ordinal = None

    def seal(self) -> V1SignedWorkBlockSupportSourceDeviceHandle:
        self._check_preferred(deep=True)
        multi = self.multi_segment_registration
        if (
            self._sealed
            or self._active_count
            or self._active_segment_ordinal is not None
            or self._segment_cursor != multi.segment_count
            or self._segment_local_cursor != 0
            or self._context_cursor != multi.context_count
        ):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.PARTIAL_SOURCE,
                "registered multi-segment group is incomplete or already sealed",
            )
        finite = bool(
            (
                torch.isfinite(self._ambiguous_upper)
                & torch.isfinite(self._coefficient_radius_upper)
                & torch.isfinite(self._coefficients).all()
                & self._source_finite
            ).item()
        )
        if self._device.type == "cuda":
            self._scalar_device_observation_count += 1
        if not finite:
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.NUMERICAL_REFUSAL,
                "support construction overflowed",
            )
        ambiguous = float(self._ambiguous_upper.item())
        coefficient_radius = float(self._coefficient_radius_upper.item())
        positive_count = int(self._positive.item())
        negative_count = int(self._negative.item())
        ambiguous_count = int(self._ambiguous.item())
        if self._device.type == "cuda":
            self._scalar_device_observation_count += 5
        deficit = math.nextafter(ambiguous + coefficient_radius, math.inf)
        if not math.isfinite(deficit):
            _refuse(
                V1SignedWorkBlockSupportRefusalCode.NUMERICAL_REFUSAL,
                "preferred-plane deficit overflowed binary64",
            )
        structural = multi.structural_group_registration
        measure = multi.measure_contract
        receipt = V1SignedWorkBlockSupportSourceReceipt(
            registration_hash=multi.registration_hash,
            source_registration_hash=multi.structural_source_registration_hash,
            source_role=structural.source_role,
            model_identity_hash=structural.model_identity_hash,
            chart_identity_hash=structural.chart_identity_hash,
            current_mask_hash=structural.current_mask_hash,
            proposal_generation=structural.proposal_generation,
            route_axis_hash=structural.route_axis_hash,
            route_id_axis_hash=structural.route_id_axis_hash,
            native_fiber_axis_receipt_hash=(structural.native_fiber_axis_receipt_hash),
            native_fiber_measure_hash=structural.native_fiber_measure_hash,
            registered_d_prefix_hash=structural.registered_d_prefix_hash,
            registered_d_w2_prefix_hash=structural.registered_d_w2_prefix_hash,
            alpha_ledger_id=structural.alpha_ledger_id,
            alpha_coordinate_id=structural.alpha_coordinate_id,
            alpha_lineage_hash=structural.alpha_lineage_hash,
            context_group_id=structural.context_group_id,
            block_partition_hash=structural.block_partition_hash,
            block_ordinal=structural.block_ordinal,
            preferred_action_binding_hash=(structural.preferred_action_binding_hash),
            preferred_action_hash=structural.preferred_action_hash,
            preferred_resulting_mask_hash=(structural.preferred_resulting_mask_hash),
            preferred_removed_route_index_hash=(
                structural.preferred_removed_route_index_hash
            ),
            preferred_selected_count=structural.preferred_selected_count,
            context_axis_hash=_source_axis_hash(
                "block_support_context", structural.context_ids
            ),
            draw_axis_hash=_source_axis_hash("block_support_draw", structural.draw_ids),
            measurement_axis_hash=_source_axis_hash(
                "block_support_measurement", structural.measurement_ids
            ),
            capture_axis_hash=_source_axis_hash(
                "block_support_capture", structural.capture_ids
            ),
            source_packet_axis_hash=_source_axis_hash(
                "block_support_source_packet", structural.source_packet_hashes
            ),
            context_weight_hash=content_hash(
                structural.context_weights, prefix=_PREFIX + "weights-"
            ),
            context_count=structural.context_count,
            route_count=self.axis.route_count,
            fiber_count=self.axis.fiber_count,
            cut_count=self.axis.cut_count,
            certified_positive_cell_count=positive_count,
            certified_negative_cell_count=negative_count,
            ambiguous_cell_count=ambiguous_count,
            ambiguous_preferred_response_upper=ambiguous,
            coefficient_construction_l1_radius_upper=coefficient_radius,
            preferred_plane_deficit_upper=deficit,
            consumed_chunk_binding_count=self._binding_count,
            consumed_durable_chunk_binding_chain_hash=self._binding_chain,
            hbm_forecast_hash=self.forecast.forecast_hash,
            execution_mode=self.design.execution_mode,
            preferred_action_representation="canonical_packed_parent_child_delta_v1",
            packed_host_bytes_read=self._packed_host_bytes_read,
            scalar_device_to_host_transfer_bytes=(
                8 * self._scalar_device_observation_count
            ),
            structural_group_registration_hash=(
                multi.structural_group_registration_hash
            ),
            segment_registration_axis_hash=multi.segment_registration_axis_hash,
            numerical_source_chain_hash=multi.numerical_source_chain_hash,
            segment_registration_count=multi.segment_count,
            multi_segment_numerical_source=True,
            measure_contract_hash=measure.contract_hash,
            measure_registered_context_axis_hash=(
                measure.registered_context_axis_hash
            ),
            measure_kind=measure.measure_kind,
            measure_design_hash=measure.measure_design_hash,
            measure_weight_receipt_hash=measure.measure_weight_receipt_hash,
            context_stratum_axis_hash=measure.context_stratum_axis_hash,
            exact_context_weight_fraction_axis_hash=(
                measure.exact_context_weight_fraction_axis_hash
            ),
            uniform_finite_population_group_mean=(
                structural.uniform_finite_population_group_mean
            ),
            canonical_full_population_f_d_measure_authority=(
                multi.full_population_canonical_f_d_measure_authority
            ),
            schema=(
                _REGISTERED_MEASURE_RECEIPT_SCHEMA
                if multi.full_population_canonical_f_d_measure_authority
                else _MULTI_SEGMENT_RECEIPT_SCHEMA
            ),
            _issuer=_RECEIPT_ISSUER,
        )
        coefficients = self._coefficients
        handle = V1SignedWorkBlockSupportSourceDeviceHandle(
            coefficients=coefficients,
            receipt=receipt,
            _issuer=_HANDLE_ISSUER,
        )
        self._coefficients = None
        self._preferred = None
        self._preferred_guard = None
        self._preferred_invariant = ()
        self._preferred_guard_invariant = ()
        self._sealed = True
        return handle


__all__ = [
    name
    for name in globals()
    if name.startswith("V1SignedWorkBlockSupport")
    or name.startswith("bind_v1_")
    or name.startswith("build_v1_")
    or name.startswith("register_v1_")
    or name.startswith("evaluate_v1_")
]
