"""Current-source D1 prefix producer for sampled-V7 D stability.

This module owns the narrow production path from the live same-capture D1 row
prefix to the four source leaves needed by the D-only measurement owner:

* resident D1 literal-J population;
* fixed-primal shell union;
* D_W2 suffix registration;
* residual-population measure design.

It consumes no C/K/O rows, spends no alpha, chooses no action beyond replaying
the already registered two-branch relaxed primals, and cannot certify D
stability or commit a mask.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
from typing import Any, Final, NoReturn

from .contracts import content_hash
from .modern_v1_sampled_residual_population_measure import (
    register_v1_sampled_residual_population_measure,
)
from .modern_v1_sampled_v7_d_fixed_source import (
    V1SampledV7DFixedSourceRefusal,
    V1SampledV7DFixedSourceRefusalCode,
    derive_v1_sampled_v7_d1_canonical_axis_plan,
)
from .modern_v1_sampled_v7_fixed_primal_shell import (
    issue_v1_sampled_v7_fixed_primal_shell_cuda,
)
from .modern_v1_sampled_v7_resident_d1_literal_j_population import (
    prepare_v1_sampled_v7_resident_d1_literal_j_population_capture,
)
from .modern_v1_sampled_variable_shell_dw2_suffix import (
    register_v1_sampled_variable_shell_dw2_suffix,
)
from .modern_v1_signed_work_literal_j_private_primal_bridge_cuda import (
    issue_v1_signed_work_literal_j_private_primal_bridge_cuda,
)


SAMPLED_V7_CURRENT_SOURCE_D1_PREFIX_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_current_source_D1_prefix_v1"
)
SAMPLED_V7_CURRENT_SOURCE_D1_PREFIX_AUTHORITY: Final = (
    "same_capture_D1_prefix_to_resident_literal_J_shell_suffix_and_residual_"
    "measure_only_no_D_stability_C_K_O_alpha_spend_commit_terminal_or_pareto_"
    "authority_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-v7-current-source-D1-prefix-"


class V1SampledV7CurrentSourceD1PrefixRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_LINEAGE = "stale_lineage"
    MISSING_SOURCE = "missing_source"
    CLEANUP_FAILURE = "cleanup_failure"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SampledV7CurrentSourceD1PrefixRefusal(RuntimeError):
    """Typed fail-closed refusal for the current-source D1 prefix producer."""

    def __init__(
        self,
        code: V1SampledV7CurrentSourceD1PrefixRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7CurrentSourceD1PrefixRefusalCode:
            raise TypeError("current-source D1 prefix code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7CurrentSourceD1PrefixRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7CurrentSourceD1PrefixRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7CurrentSourceD1PrefixRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty identity string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SampledV7CurrentSourceD1PrefixRefusalCode.INVALID_TYPE,
            f"{name} must be one positive exact integer",
        )
    result = int(value)
    if result <= 0:
        _refuse(
            V1SampledV7CurrentSourceD1PrefixRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _maybe_hash(value: Any, *names: str) -> str:
    for name in names:
        candidate = getattr(value, name, None)
        if type(candidate) is str and candidate:
            return candidate
    assert_integrity = getattr(value, "assert_integrity", None)
    if callable(assert_integrity):
        observed = assert_integrity()
        if type(observed) is str and observed:
            return observed
    assert_live = getattr(value, "assert_live", None)
    if callable(assert_live):
        observed = assert_live()
        if type(observed) is str and observed:
            return observed
    _refuse(
        V1SampledV7CurrentSourceD1PrefixRefusalCode.INVALID_TYPE,
        f"{type(value).__name__} lacks a stable content identity",
    )


def _try_evict(value: Any | None) -> Exception | None:
    if value is None:
        return None
    evict = getattr(value, "evict", None)
    if not callable(evict):
        return None
    try:
        evict()
    except Exception as error:  # pragma: no cover - defensive cleanup surface.
        return error
    return None


def _try_evict_d1_plan(value: Any | None) -> Exception | None:
    """Evict a still-live D1 plan, accepting prior exact shell consumption."""

    error = _try_evict(value)
    if (
        type(error) is V1SampledV7DFixedSourceRefusal
        and error.code is V1SampledV7DFixedSourceRefusalCode.EVICTED
    ):
        return None
    return error


@dataclass(frozen=True, slots=True)
class V1SampledV7CurrentSourceD1PrefixResult:
    """Live D1 prefix products for the next source-only V7 owner."""

    d1_axis_plan_hash: str
    private_primal_source_hash: str
    private_primal_descriptor_axis_hash: str
    private_primal_receipt_hash: str
    shell_result_hash: str
    shell_union_receipt_hash: str
    resident_d1_receipt_hash: str
    suffix_registration_hash: str
    residual_measure_design_hash: str
    d1_stratum_counts: tuple[int, int]
    resident_d1_population: Any = field(repr=False, compare=False)
    shell_result: Any = field(repr=False, compare=False)
    shell_union: Any = field(repr=False, compare=False)
    suffix_registration: Any = field(repr=False, compare=False)
    residual_measure_design: Any = field(repr=False, compare=False)
    d_stability_authority: bool = False
    c_k_o_or_validation_authority: bool = False
    alpha_spend_authority: bool = False
    mask_or_pareto_authority: bool = False
    authority_scope: str = SAMPLED_V7_CURRENT_SOURCE_D1_PREFIX_AUTHORITY
    schema: str = SAMPLED_V7_CURRENT_SOURCE_D1_PREFIX_SCHEMA
    _result_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "d1_axis_plan_hash",
            "private_primal_source_hash",
            "private_primal_descriptor_axis_hash",
            "private_primal_receipt_hash",
            "shell_result_hash",
            "shell_union_receipt_hash",
            "resident_d1_receipt_hash",
            "suffix_registration_hash",
            "residual_measure_design_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        counts = tuple(self.d1_stratum_counts)
        if (
            len(counts) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0
                for value in counts
            )
            or self.shell_union is not getattr(self.shell_result, "shell_union", None)
            or _maybe_hash(self.shell_union, "receipt_hash")
            != self.shell_union_receipt_hash
            or _maybe_hash(self.suffix_registration, "receipt_hash")
            != self.suffix_registration_hash
            or _maybe_hash(self.residual_measure_design, "design_hash")
            != self.residual_measure_design_hash
            or _maybe_hash(self.resident_d1_population.receipt, "receipt_hash")
            != self.resident_d1_receipt_hash
            or tuple(getattr(self.resident_d1_population.receipt, "d1_stratum_counts", ()))
            != counts
            or tuple(getattr(self.suffix_registration, "consumed_d1_stratum_counts", ()))
            != counts
            or getattr(self.suffix_registration, "d1_pair_prefix_axis_hash", None)
            != getattr(self.shell_result.receipt, "d1_pair_prefix_axis_hash", None)
            or getattr(self.suffix_registration, "provisional_shell_hash", None)
            != getattr(self.shell_union, "provisional_shell_hash", None)
            or getattr(
                self.suffix_registration,
                "provisional_action_family_hash",
                None,
            )
            != getattr(self.shell_union, "provisional_action_family_hash", None)
            or getattr(self.residual_measure_design, "suffix_registration_hash", None)
            != self.suffix_registration_hash
            or tuple(getattr(self.residual_measure_design, "d1_stratum_counts", ()))
            != counts
            or self.d_stability_authority is not False
            or self.c_k_o_or_validation_authority is not False
            or self.alpha_spend_authority is not False
            or self.mask_or_pareto_authority is not False
            or self.authority_scope != SAMPLED_V7_CURRENT_SOURCE_D1_PREFIX_AUTHORITY
            or self.schema != SAMPLED_V7_CURRENT_SOURCE_D1_PREFIX_SCHEMA
        ):
            _refuse(
                V1SampledV7CurrentSourceD1PrefixRefusalCode.AUTHORITY_VIOLATION,
                "current-source D1 prefix result crossed its source-only boundary",
            )
        handle = getattr(self.resident_d1_population, "device_handle", None)
        assert_live_identity = getattr(handle, "assert_live_identity", None)
        if callable(assert_live_identity):
            assert_live_identity()
        object.__setattr__(
            self,
            "d1_stratum_counts",
            (int(counts[0]), int(counts[1])),
        )
        self.assert_live()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
            and name
            not in {
                "resident_d1_population",
                "shell_result",
                "shell_union",
                "suffix_registration",
                "residual_measure_design",
            }
        }

    def assert_live(self) -> str:
        observed = content_hash(self._payload(), prefix=_PREFIX + "result-")
        if self._result_hash is not None and observed != self._result_hash:
            _refuse(
                V1SampledV7CurrentSourceD1PrefixRefusalCode.STALE_LINEAGE,
                "current-source D1 prefix result identity changed",
            )
        handle = getattr(self.resident_d1_population, "device_handle", None)
        assert_live_identity = getattr(handle, "assert_live_identity", None)
        if callable(assert_live_identity):
            assert_live_identity()
        object.__setattr__(self, "_result_hash", observed)
        return observed

    @property
    def result_hash(self) -> str:
        return self.assert_live()

    def evict_resident_d1(self) -> None:
        handle = getattr(self.resident_d1_population, "device_handle", None)
        evict = getattr(handle, "evict", None)
        if callable(evict):
            evict()


def issue_v1_sampled_v7_current_source_d1_prefix(
    *,
    row_prefix: Any,
    original_capture_binding: Any,
    prepared_source_problem: Any,
    source_registration: Any,
    common_coordinate: Any,
    route_ids: Any,
    generation_sampling_frame: Any,
    discovery_sampling_stream: Any,
    transport_packet_size: int,
) -> V1SampledV7CurrentSourceD1PrefixResult:
    """Produce the live D1 prefix leaves from the current source graph."""

    packet_size = _positive_int(transport_packet_size, name="transport packet size")
    d1_axis_plan = None
    private_primal_bridge = None
    resident_d1_population = None
    shell_result = None
    try:
        d1_axis_plan = derive_v1_sampled_v7_d1_canonical_axis_plan(
            row_prefix=row_prefix,
            original_capture_binding=original_capture_binding,
        )
        d1_axis_plan_hash = _maybe_hash(d1_axis_plan, "plan_hash")
        private_primal_bridge = issue_v1_signed_work_literal_j_private_primal_bridge_cuda(
            problem=common_coordinate.problem,
            policy=common_coordinate.policies.phase_b,
            spectral_execution=common_coordinate.branch_executions[0],
            diagonal_l1_execution=common_coordinate.branch_executions[1],
        )
        capture = prepare_v1_sampled_v7_resident_d1_literal_j_population_capture(
            branch_executions=common_coordinate.branch_executions,
            private_primal_bridge=private_primal_bridge,
        )
        shell_result = issue_v1_sampled_v7_fixed_primal_shell_cuda(
            prepared_source_problem=prepared_source_problem,
            source_registration=source_registration,
            common_coordinate=common_coordinate,
            private_primal_bridge=private_primal_bridge,
            d1_axis_plan=d1_axis_plan,
            route_ids=route_ids,
            d1_projection_consumer=capture,
        )
        resident_d1_population = capture.take_result()
        private_primal_source_hash = private_primal_bridge.source_receipt.source_hash
        private_primal_descriptor_axis_hash = (
            private_primal_bridge.descriptor_axis.descriptor_axis_hash
        )
        private_primal_receipt_hash = private_primal_bridge.receipt.receipt_hash
        bridge_cleanup_error = _try_evict(private_primal_bridge)
        if bridge_cleanup_error is not None:
            raise V1SampledV7CurrentSourceD1PrefixRefusal(
                V1SampledV7CurrentSourceD1PrefixRefusalCode.CLEANUP_FAILURE,
                "private-primal GPU custody could not be released after D1 replay",
            ) from bridge_cleanup_error
        private_primal_bridge = None
        shell_union = shell_result.shell_union
        provisional_shell_hash = getattr(
            shell_union,
            "provisional_shell_hash",
            shell_union.receipt_hash,
        )
        provisional_action_family_hash = getattr(
            shell_union,
            "provisional_action_family_hash",
            shell_union.receipt_hash,
        )
        suffix_registration = register_v1_sampled_variable_shell_dw2_suffix(
            frame=generation_sampling_frame,
            discovery_stream=discovery_sampling_stream,
            consumed_d1_stratum_counts=resident_d1_population.receipt.d1_stratum_counts,
            provisional_shell_hash=provisional_shell_hash,
            provisional_action_family_hash=provisional_action_family_hash,
            transport_packet_size=packet_size,
        )
        residual_measure_design = register_v1_sampled_residual_population_measure(
            suffix_registration=suffix_registration,
        )
        result = V1SampledV7CurrentSourceD1PrefixResult(
            d1_axis_plan_hash=d1_axis_plan_hash,
            private_primal_source_hash=private_primal_source_hash,
            private_primal_descriptor_axis_hash=private_primal_descriptor_axis_hash,
            private_primal_receipt_hash=private_primal_receipt_hash,
            shell_result_hash=_maybe_hash(getattr(shell_result, "receipt", None), "receipt_hash"),
            shell_union_receipt_hash=shell_union.receipt_hash,
            resident_d1_receipt_hash=resident_d1_population.receipt.receipt_hash,
            suffix_registration_hash=suffix_registration.receipt_hash,
            residual_measure_design_hash=residual_measure_design.design_hash,
            d1_stratum_counts=resident_d1_population.receipt.d1_stratum_counts,
            resident_d1_population=resident_d1_population,
            shell_result=shell_result,
            shell_union=shell_union,
            suffix_registration=suffix_registration,
            residual_measure_design=residual_measure_design,
        )
        resident_d1_population = None
        return result
    except BaseException as primary_error:
        cleanup_errors = []
        error = _try_evict(private_primal_bridge)
        if error is not None:
            cleanup_errors.append(error)
        if shell_result is None:
            error = _try_evict_d1_plan(d1_axis_plan)
            if error is not None:
                cleanup_errors.append(error)
        if resident_d1_population is not None:
            handle = getattr(resident_d1_population, "device_handle", None)
            error = _try_evict(handle)
            if error is not None:
                cleanup_errors.append(error)
        if cleanup_errors:
            primary_error.add_note(
                "current-source D1 prefix cleanup also failed: "
                + "; ".join(
                    f"{type(error).__name__}: {error}"
                    for error in cleanup_errors
                )
            )
        raise


__all__ = (
    "SAMPLED_V7_CURRENT_SOURCE_D1_PREFIX_AUTHORITY",
    "SAMPLED_V7_CURRENT_SOURCE_D1_PREFIX_SCHEMA",
    "V1SampledV7CurrentSourceD1PrefixRefusal",
    "V1SampledV7CurrentSourceD1PrefixRefusalCode",
    "V1SampledV7CurrentSourceD1PrefixResult",
    "issue_v1_sampled_v7_current_source_d1_prefix",
)
