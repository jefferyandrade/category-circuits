"""Packed relaxed-gate execution for canonical modern-v1 directional secants.

This provider is deliberately narrower than the historical singleton finite-dose
field.  It executes only a pre-frozen, strictly nested packed ladder on a sealed
C-stream prefix, at the canonical doses ``(0.5, 1.0)``.  The parent raw target
logit is evaluated once and shared by every endpoint.  The provider then emits
the typed :class:`V1DirectionalSecantRecord` observations consumed by F/Q/R
certification.

The canonical Jacobian row is the positive-removal covector
``j=-partial_m r``, so a positive removal direction ``z`` predicts
``+eta * jz``.  Consequently the signed nonlinear remainder is
``delta - eta * jz``.  The typed record owns that
arithmetic, including the canonical quadratic-cancelling diagnostic
``R_1 - 4 R_1/2``; this module only supplies source-bound inputs.

No object in this module can select or reorder a candidate, inspect a BCE
oracle, authorize a hard-mask commit, or claim terminal validation authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

import numpy as np

from mechanistic_systems.backends.torch_masks import CompiledTorchCircuitMask
from mechanistic_systems.circuits.irg.modern_v1_directional_remainder import (
    DIRECTIONAL_SECANT_DOSES,
    V1DirectionalSecantDesign,
    V1DirectionalSecantRecord,
)
from mechanistic_systems.circuits.irg.modern_v1_fibered_directional_record import (
    V1FiberedDirectionalProjectionClosure,
    V1FiberedDirectionalRecord,
    close_v1_fibered_directional_record,
)
from mechanistic_systems.circuits.irg.modern_v1_native_fiber_axis import (
    V1NativeDestinationFiberAxisReceipt,
)
from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask


PACKED_DIRECTIONAL_SECANT_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_packed_directional_secant_execution_receipt_v2"
)
PACKED_DIRECTIONAL_SECANT_AUTHORITY_SCOPE: Final = (
    "directional_secant_measurement_only_no_selection_reordering_oracle_"
    "mask_commit_or_terminal_authority_v1"
)
PACKED_FIBERED_DIRECTIONAL_SECANT_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_packed_native_fibered_directional_secant_execution_receipt_v1"
)
PACKED_FIBERED_DIRECTIONAL_SECANT_AUTHORITY_SCOPE: Final = (
    "native_fibered_directional_secant_measurement_only_no_selection_"
    "reordering_calibration_oracle_mask_commit_or_terminal_authority_v1"
)
DEFAULT_BINARY64_ENCLOSURE_SOURCE_HASH: Final = content_hash(
    {
        "method": "one_outward_binary64_spacing_radius_around_reported_center_v1",
        "claim": "numerical_center_enclosure_only_not_an_analytic_network_bound",
    },
    prefix="modern-v1-directional-default-enclosure-source-",
)

_PROJECTION_PREFIX: Final = "modern-v1-directional-projection-"
_PREFIX_PREFIX: Final = "modern-v1-directional-context-prefix-"
_SLICE_PREFIX: Final = "modern-v1-directional-context-slice-"
_DIRECTION_PREFIX: Final = "modern-v1-packed-removal-direction-"
_EXECUTION_PREFIX: Final = "modern-v1-directional-execution-proof-"
_INTERVAL_PREFIX: Final = "modern-v1-directional-interval-proof-"
_RECEIPT_PREFIX: Final = "modern-v1-directional-execution-receipt-"
_FIBERED_RECEIPT_PREFIX: Final = (
    "modern-v1-native-fibered-directional-execution-receipt-"
)
_FIBERED_BOUNDS_PREFIX: Final = (
    "modern-v1-native-fibered-directional-observation-bounds-"
)
_PARENT_EXECUTION_PREFIX: Final = (
    "modern-v1-native-fibered-parent-execution-"
)
_FIBERED_EXECUTION_PREFIX: Final = (
    "modern-v1-native-fibered-directional-execution-proof-"
)


def directional_secant_provider_source_hash() -> str:
    """Hash the exact provider source used by a pre-outcome design."""

    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be one real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _immutable_f64(
    value: object,
    *,
    shape: tuple[int, ...],
    name: str,
    nonnegative: bool = False,
) -> np.ndarray:
    try:
        array = np.array(value, dtype=np.dtype("<f8"), order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a float64 array") from error
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    if nonnegative and bool(np.any(array < 0.0)):
        raise ValueError(f"{name} must be nonnegative")
    array[array == 0.0] = 0.0
    result = np.frombuffer(array.tobytes(order="C"), dtype=np.dtype("<f8")).reshape(
        shape
    )
    result.setflags(write=False)
    return result


def sealed_directional_context_prefix_hash(
    *,
    design_hash: str,
    context_ids_by_stratum: Sequence[tuple[str, Sequence[str]]],
    draw_ids: Sequence[str],
) -> str:
    """Content-address a zero-origin registered C-stream prefix.

    This legacy-named helper remains the exact zero-origin specialization of
    :func:`sealed_directional_context_slice_hash`.  Adaptive production looks
    should use the slice helper so already-consumed ImageNet contexts are not
    executed again.
    """

    axes = tuple(
        (
            _identity(name, name="stratum id"),
            tuple(_identity(item, name="context id") for item in contexts),
        )
        for name, contexts in context_ids_by_stratum
    )
    return sealed_directional_context_slice_hash(
        design_hash=design_hash,
        context_ids_by_stratum=axes,
        sealed_order_start_indices_by_stratum=tuple(
            (name, 0) for name, _axis in axes
        ),
        draw_ids=draw_ids,
        _zero_origin_prefix_compatibility=True,
    )


def sealed_directional_context_slice_hash(
    *,
    design_hash: str,
    context_ids_by_stratum: Sequence[tuple[str, Sequence[str]]],
    sealed_order_start_indices_by_stratum: Sequence[tuple[str, int]],
    draw_ids: Sequence[str],
    _zero_origin_prefix_compatibility: bool = False,
) -> str:
    """Content-address one exact incremental slice of the sealed C stream."""

    axes = tuple(
        (
            _identity(name, name="stratum id"),
            tuple(_identity(item, name="context id") for item in contexts),
        )
        for name, contexts in context_ids_by_stratum
    )
    starts_list: list[tuple[str, int]] = []
    for name, value in sealed_order_start_indices_by_stratum:
        stratum = _identity(name, name="slice-start stratum id")
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("sealed C-stream slice starts must be exact integers")
        start = int(value)
        if start < 0:
            raise ValueError("sealed C-stream slice starts must be nonnegative")
        starts_list.append((stratum, start))
    starts = tuple(starts_list)
    draws = tuple(_identity(value, name="draw id") for value in draw_ids)
    flat = tuple(item for _, axis in axes for item in axis)
    if (
        not axes
        or len({name for name, _ in axes}) != len(axes)
        or tuple(name for name, _ in starts) != tuple(name for name, _ in axes)
        or len(starts) != len({name for name, _ in starts})
        or not flat
        or len(flat) != len(set(flat))
        or len(draws) != len(flat)
        or len(draws) != len(set(draws))
    ):
        raise ValueError("sealed directional context-slice/draw axes are invalid")
    if _zero_origin_prefix_compatibility:
        if any(start != 0 for _name, start in starts):  # pragma: no cover
            raise ValueError("zero-origin compatibility requires zero starts")
        return content_hash(
            {
                "design_hash": _identity(design_hash, name="design hash"),
                "context_ids_by_stratum": axes,
                "draw_ids": draws,
            },
            prefix=_PREFIX_PREFIX,
        )
    return content_hash(
        {
            "design_hash": _identity(design_hash, name="design hash"),
            "context_ids_by_stratum": axes,
            "sealed_order_start_indices_by_stratum": starts,
            "draw_ids": draws,
        },
        prefix=_SLICE_PREFIX,
    )


def fibered_directional_context_slice_axes(
    *,
    design: V1DirectionalSecantDesign,
    projection_closures: Sequence[V1FiberedDirectionalProjectionClosure],
    sealed_order_start_indices_by_stratum: Sequence[tuple[str, int]],
) -> tuple[
    tuple[tuple[str, tuple[str, ...]], ...],
    tuple[tuple[str, int], ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    """Validate and expose one exact ordered native-fibered C-stream slice."""

    if type(design) is not V1DirectionalSecantDesign:
        raise TypeError("fibered C-slice validation requires the exact design")
    design.assert_integrity()
    closures = tuple(projection_closures)
    if not closures or any(
        type(closure) is not V1FiberedDirectionalProjectionClosure
        for closure in closures
    ):
        raise TypeError("fibered C slice requires nonempty exact closure rows")
    for closure in closures:
        closure.assert_integrity()
    starts_list: list[tuple[str, int]] = []
    for name, value in sealed_order_start_indices_by_stratum:
        stratum = _identity(name, name="fibered C-slice start stratum id")
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("fibered C-slice starts must be exact integers")
        start = int(value)
        if start < 0:
            raise ValueError("fibered C-slice starts must be nonnegative")
        starts_list.append((stratum, start))
    starts = tuple(starts_list)
    registered = design.certification_context_ids_by_stratum
    if tuple(name for name, _axis in registered) != tuple(
        name for name, _start in starts
    ):
        raise ValueError("fibered C-slice start strata differ from the design")
    rows: list[tuple[str, tuple[str, ...]]] = []
    cursor = 0
    for (stratum, complete_axis), (_start_name, start) in zip(
        registered,
        starts,
        strict=True,
    ):
        selected: list[str] = []
        local_offset = 0
        while cursor < len(closures) and closures[cursor].stratum_id == stratum:
            closure = closures[cursor]
            expected_index = start + local_offset
            if (
                expected_index >= len(complete_axis)
                or closure.sealed_order_index != expected_index
                or closure.context_id != complete_axis[expected_index]
            ):
                raise ValueError("fibered closure rows are not an exact C slice")
            selected.append(closure.context_id)
            local_offset += 1
            cursor += 1
        if start > len(complete_axis) or start + len(selected) > len(complete_axis):
            raise ValueError("fibered C-slice coordinate exceeds its population")
        rows.append((stratum, tuple(selected)))
    if cursor != len(closures):
        raise ValueError("fibered closure stratum order differs from the design")
    contexts = tuple(closure.context_id for closure in closures)
    draws = tuple(closure.draw_id for closure in closures)
    if (
        len(contexts) != len(set(contexts))
        or len(draws) != len(set(draws))
        or tuple(item for _name, axis in rows for item in axis) != contexts
    ):
        raise ValueError("fibered C-slice context/draw axes repeat or reorder")
    return tuple(rows), starts, contexts, draws


def directional_projection_hash(
    *,
    kind: str,
    design_hash: str,
    chart_identity_hash: str,
    context_ids: Sequence[str],
    endpoint_action_hashes: Sequence[str],
    centers: object,
    radii: object,
) -> str:
    """Hash a corrected-work or literal-Jacobian endpoint projection table."""

    if kind not in {"corrected_evidence", "literal_positive_removal_jacobian"}:
        raise ValueError("directional projection kind is not canonical")
    contexts = tuple(
        _identity(value, name="projection context id") for value in context_ids
    )
    endpoints = tuple(
        _identity(value, name="projection endpoint action hash")
        for value in endpoint_action_hashes
    )
    if (
        not contexts
        or len(contexts) != len(set(contexts))
        or not endpoints
        or len(endpoints) != len(set(endpoints))
    ):
        raise ValueError("directional projection axes are invalid")
    shape = (len(contexts), len(endpoints))
    center_array = _immutable_f64(centers, shape=shape, name="projection centers")
    radius_array = _immutable_f64(
        radii,
        shape=shape,
        name="projection radii",
        nonnegative=True,
    )
    return content_hash(
        {
            "kind": kind,
            "design_hash": _identity(design_hash, name="design hash"),
            "chart_identity_hash": _identity(
                chart_identity_hash,
                name="chart identity hash",
            ),
            "context_ids": contexts,
            "endpoint_action_hashes": endpoints,
            "centers": center_array,
            "radii": radius_array,
        },
        prefix=_PROJECTION_PREFIX,
    )


@dataclass(frozen=True, slots=True)
class V1DirectionalSecantObservationBounds:
    """Optional finite, pre-outcome envelope checked against every observation."""

    design_hash: str
    endpoint_action_hashes: tuple[str, ...]
    raw_logit_absolute_upper: float
    corrected_projection_absolute_upper_by_endpoint: np.ndarray = field(
        repr=False,
        compare=False,
    )
    jacobian_projection_absolute_upper_by_endpoint: np.ndarray = field(
        repr=False,
        compare=False,
    )
    remainder_absolute_upper_by_endpoint_and_dose: np.ndarray = field(
        repr=False,
        compare=False,
    )
    bound_source_hash: str
    registered_before_secant_outcomes: bool = True
    _bounds_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        design_hash = _identity(self.design_hash, name="bounds design hash")
        endpoints = tuple(
            _identity(value, name="bounds endpoint action hash")
            for value in self.endpoint_action_hashes
        )
        if not endpoints or len(endpoints) != len(set(endpoints)):
            raise ValueError("observation-bound endpoint axis is invalid")
        raw = _finite_nonnegative(
            self.raw_logit_absolute_upper,
            name="raw-logit absolute bound",
        )
        corrected = _immutable_f64(
            self.corrected_projection_absolute_upper_by_endpoint,
            shape=(len(endpoints),),
            name="corrected projection bounds",
            nonnegative=True,
        )
        jacobian = _immutable_f64(
            self.jacobian_projection_absolute_upper_by_endpoint,
            shape=(len(endpoints),),
            name="Jacobian projection bounds",
            nonnegative=True,
        )
        remainder = _immutable_f64(
            self.remainder_absolute_upper_by_endpoint_and_dose,
            shape=(len(endpoints), 2),
            name="remainder bounds",
            nonnegative=True,
        )
        source = _identity(self.bound_source_hash, name="observation-bound source hash")
        if self.registered_before_secant_outcomes is not True:
            raise PermissionError("directional observation bounds were post-selected")
        payload = {
            "design_hash": design_hash,
            "endpoint_action_hashes": endpoints,
            "raw_logit_absolute_upper": raw,
            "corrected_projection_absolute_upper_by_endpoint": corrected,
            "jacobian_projection_absolute_upper_by_endpoint": jacobian,
            "remainder_absolute_upper_by_endpoint_and_dose": remainder,
            "bound_source_hash": source,
            "registered_before_secant_outcomes": True,
        }
        object.__setattr__(self, "design_hash", design_hash)
        object.__setattr__(self, "endpoint_action_hashes", endpoints)
        object.__setattr__(self, "raw_logit_absolute_upper", raw)
        object.__setattr__(
            self,
            "corrected_projection_absolute_upper_by_endpoint",
            corrected,
        )
        object.__setattr__(
            self,
            "jacobian_projection_absolute_upper_by_endpoint",
            jacobian,
        )
        object.__setattr__(
            self,
            "remainder_absolute_upper_by_endpoint_and_dose",
            remainder,
        )
        object.__setattr__(self, "bound_source_hash", source)
        object.__setattr__(
            self,
            "_bounds_hash",
            content_hash(payload, prefix="modern-v1-directional-observation-bounds-"),
        )

    @property
    def bounds_hash(self) -> str:
        return self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "design_hash": self.design_hash,
            "endpoint_action_hashes": self.endpoint_action_hashes,
            "raw_logit_absolute_upper": self.raw_logit_absolute_upper,
            "corrected_projection_absolute_upper_by_endpoint": (
                self.corrected_projection_absolute_upper_by_endpoint
            ),
            "jacobian_projection_absolute_upper_by_endpoint": (
                self.jacobian_projection_absolute_upper_by_endpoint
            ),
            "remainder_absolute_upper_by_endpoint_and_dose": (
                self.remainder_absolute_upper_by_endpoint_and_dose
            ),
            "bound_source_hash": self.bound_source_hash,
            "registered_before_secant_outcomes": (
                self.registered_before_secant_outcomes
            ),
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(
            self._payload(),
            prefix="modern-v1-directional-observation-bounds-",
        )
        if observed != self._bounds_hash:
            raise ValueError("directional observation bounds were tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("directional observation-bounds hash does not match")
        return observed


@dataclass(frozen=True, slots=True)
class V1FiberedDirectionalSecantObservationBounds:
    """Pre-outcome finite envelope for native-fibered F/Q/R observations.

    The F coordinate is already nonnegative native-fiber work.  It is not a
    bound on, and cannot be populated from, a globally signed corrected
    projection.
    """

    design_hash: str
    endpoint_action_hashes: tuple[str, ...]
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    raw_logit_absolute_upper: float
    fibered_work_upper_by_endpoint: np.ndarray = field(
        repr=False,
        compare=False,
    )
    global_literal_jacobian_absolute_upper_by_endpoint: np.ndarray = field(
        repr=False,
        compare=False,
    )
    remainder_absolute_upper_by_endpoint_and_dose: np.ndarray = field(
        repr=False,
        compare=False,
    )
    bound_source_hash: str
    registered_before_secant_outcomes: bool = True
    global_corrected_scalar_projection_used: bool = False
    _bounds_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        design_hash = _identity(self.design_hash, name="fibered bounds design hash")
        endpoints = tuple(
            _identity(value, name="fibered bounds endpoint action hash")
            for value in self.endpoint_action_hashes
        )
        if not endpoints or len(endpoints) != len(set(endpoints)):
            raise ValueError("fibered observation-bound endpoint axis is invalid")
        fiber_receipt_hash = _identity(
            self.native_fiber_axis_receipt_hash,
            name="native-fiber axis receipt hash",
        )
        fiber_measure_hash = _identity(
            self.native_fiber_measure_hash,
            name="native-fiber measure hash",
        )
        raw = _finite_nonnegative(
            self.raw_logit_absolute_upper,
            name="raw-logit absolute bound",
        )
        fibered_work = _immutable_f64(
            self.fibered_work_upper_by_endpoint,
            shape=(len(endpoints),),
            name="fibered-work upper bounds",
            nonnegative=True,
        )
        jacobian = _immutable_f64(
            self.global_literal_jacobian_absolute_upper_by_endpoint,
            shape=(len(endpoints),),
            name="global literal-J projection bounds",
            nonnegative=True,
        )
        remainder = _immutable_f64(
            self.remainder_absolute_upper_by_endpoint_and_dose,
            shape=(len(endpoints), 2),
            name="fibered directional remainder bounds",
            nonnegative=True,
        )
        source = _identity(
            self.bound_source_hash,
            name="fibered observation-bound source hash",
        )
        if self.registered_before_secant_outcomes is not True:
            raise PermissionError("fibered directional bounds were post-selected")
        if self.global_corrected_scalar_projection_used is not False:
            raise ValueError("fibered F bounds cannot use a global corrected scalar")
        payload = {
            "design_hash": design_hash,
            "endpoint_action_hashes": endpoints,
            "native_fiber_axis_receipt_hash": fiber_receipt_hash,
            "native_fiber_measure_hash": fiber_measure_hash,
            "raw_logit_absolute_upper": raw,
            "fibered_work_upper_by_endpoint": fibered_work,
            "global_literal_jacobian_absolute_upper_by_endpoint": jacobian,
            "remainder_absolute_upper_by_endpoint_and_dose": remainder,
            "bound_source_hash": source,
            "registered_before_secant_outcomes": True,
            "global_corrected_scalar_projection_used": False,
        }
        object.__setattr__(self, "design_hash", design_hash)
        object.__setattr__(self, "endpoint_action_hashes", endpoints)
        object.__setattr__(
            self,
            "native_fiber_axis_receipt_hash",
            fiber_receipt_hash,
        )
        object.__setattr__(self, "native_fiber_measure_hash", fiber_measure_hash)
        object.__setattr__(self, "raw_logit_absolute_upper", raw)
        object.__setattr__(self, "fibered_work_upper_by_endpoint", fibered_work)
        object.__setattr__(
            self,
            "global_literal_jacobian_absolute_upper_by_endpoint",
            jacobian,
        )
        object.__setattr__(
            self,
            "remainder_absolute_upper_by_endpoint_and_dose",
            remainder,
        )
        object.__setattr__(self, "bound_source_hash", source)
        object.__setattr__(
            self,
            "_bounds_hash",
            content_hash(payload, prefix=_FIBERED_BOUNDS_PREFIX),
        )

    @property
    def bounds_hash(self) -> str:
        return self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "design_hash": self.design_hash,
            "endpoint_action_hashes": self.endpoint_action_hashes,
            "native_fiber_axis_receipt_hash": self.native_fiber_axis_receipt_hash,
            "native_fiber_measure_hash": self.native_fiber_measure_hash,
            "raw_logit_absolute_upper": self.raw_logit_absolute_upper,
            "fibered_work_upper_by_endpoint": self.fibered_work_upper_by_endpoint,
            "global_literal_jacobian_absolute_upper_by_endpoint": (
                self.global_literal_jacobian_absolute_upper_by_endpoint
            ),
            "remainder_absolute_upper_by_endpoint_and_dose": (
                self.remainder_absolute_upper_by_endpoint_and_dose
            ),
            "bound_source_hash": self.bound_source_hash,
            "registered_before_secant_outcomes": True,
            "global_corrected_scalar_projection_used": False,
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_FIBERED_BOUNDS_PREFIX)
        if observed != self._bounds_hash:
            raise ValueError("fibered directional observation bounds were tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError(
                "fibered directional observation-bounds hash does not match"
            )
        return observed


@dataclass(frozen=True, slots=True)
class V1PackedDirectionalSecantExecutionReceipt:
    """Fail-closed execution receipt with no downstream decision authority."""

    design_hash: str
    ladder_hash: str
    chart_identity_hash: str
    parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    endpoint_action_hashes: tuple[str, ...]
    endpoint_mask_hashes: tuple[str, ...]
    packed_direction_hashes: tuple[str, ...]
    sealed_context_prefix_hash: str
    context_ids_by_stratum: tuple[tuple[str, tuple[str, ...]], ...]
    sealed_order_start_indices_by_stratum: tuple[tuple[str, int], ...]
    draw_ids: tuple[str, ...]
    corrected_projection_hash: str
    jacobian_projection_hash: str
    dual_chart_projection_proof_hash: str
    provider_source_hash: str
    relaxed_gate_executor_source_hash: str
    raw_logit_interval_verifier_source_hash: str
    raw_logit_interval_proof_hash: str
    observation_bounds_hash: str | None
    maximum_packed_lanes: int
    gate_workspace_forecast_bytes: int
    maximum_gate_workspace_bytes: int
    parent_lane_evaluations: int
    endpoint_lane_evaluations: int
    fused_forward_calls: int
    lane_packet_sizes: tuple[int, ...]
    parent_raw_logit_hash: str
    dosed_raw_logit_hash: str
    records: tuple[V1DirectionalSecantRecord, ...] = field(repr=False, compare=False)
    execution_proof_hash: str
    doses: tuple[float, float] = DIRECTIONAL_SECANT_DOSES
    endpoint_order_unchanged_after_outcomes: bool = True
    candidate_selection_performed: bool = False
    candidate_reordering_performed: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = PACKED_DIRECTIONAL_SECANT_AUTHORITY_SCOPE
    schema: str = PACKED_DIRECTIONAL_SECANT_RECEIPT_SCHEMA
    _receipt_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "design_hash",
            "ladder_hash",
            "chart_identity_hash",
            "parent_mask_hash",
            "route_axis_hash",
            "sealed_context_prefix_hash",
            "corrected_projection_hash",
            "jacobian_projection_hash",
            "dual_chart_projection_proof_hash",
            "provider_source_hash",
            "relaxed_gate_executor_source_hash",
            "raw_logit_interval_verifier_source_hash",
            "raw_logit_interval_proof_hash",
            "parent_raw_logit_hash",
            "dosed_raw_logit_hash",
            "execution_proof_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        endpoints = tuple(
            _identity(value, name="endpoint action hash")
            for value in self.endpoint_action_hashes
        )
        endpoint_masks = tuple(
            _identity(value, name="endpoint mask hash")
            for value in self.endpoint_mask_hashes
        )
        directions = tuple(
            _identity(value, name="packed direction hash")
            for value in self.packed_direction_hashes
        )
        if not endpoints or not (
            len(endpoints) == len(endpoint_masks) == len(directions)
        ):
            raise ValueError("receipt endpoint/direction axes are not aligned")
        contexts = tuple(
            (
                _identity(name, name="receipt stratum id"),
                tuple(_identity(item, name="receipt context id") for item in axis),
            )
            for name, axis in self.context_ids_by_stratum
        )
        starts_list: list[tuple[str, int]] = []
        for name, value in self.sealed_order_start_indices_by_stratum:
            stratum = _identity(name, name="receipt slice-start stratum id")
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError("receipt slice starts must be exact integers")
            start = int(value)
            if start < 0:
                raise ValueError("receipt slice starts must be nonnegative")
            starts_list.append((stratum, start))
        starts = tuple(starts_list)
        flat_contexts = tuple(item for _, axis in contexts for item in axis)
        draws = tuple(
            _identity(value, name="receipt draw id") for value in self.draw_ids
        )
        records = tuple(self.records)
        if (
            not contexts
            or tuple(name for name, _ in starts)
            != tuple(name for name, _ in contexts)
            or not flat_contexts
            or len(flat_contexts) != len(draws)
            or len(records) != len(flat_contexts)
            or any(type(record) is not V1DirectionalSecantRecord for record in records)
        ):
            raise ValueError("receipt context, draw, and record axes are not aligned")
        record_hashes: list[str] = []
        expected_rows = tuple(
            (stratum, start + offset, context)
            for (stratum, axis), (_start_name, start) in zip(
                contexts,
                starts,
                strict=True,
            )
            for offset, context in enumerate(axis)
        )
        for (stratum, sealed_index, context), draw, record in zip(
            expected_rows,
            draws,
            records,
            strict=True,
        ):
            record.assert_integrity()
            if (
                record.design_hash != self.design_hash
                or record.ladder_hash != self.ladder_hash
                or record.chart_identity_hash != self.chart_identity_hash
                or record.endpoint_action_hashes != endpoints
                or record.stratum_id != stratum
                or record.sealed_order_index != sealed_index
                or record.context_id != context
                or record.draw_id != draw
                or record.packed_direction_execution_proof_hash
                != self.execution_proof_hash
                or record.raw_logit_interval_proof_hash
                != self.raw_logit_interval_proof_hash
            ):
                raise ValueError(
                    "directional record was recontextualized in its receipt"
                )
            expected_row_proof = content_hash(
                {
                    "dual_chart_projection_proof_hash": (
                        self.dual_chart_projection_proof_hash
                    ),
                    "context_id": context,
                    "draw_id": draw,
                    "corrected_projection_hash": self.corrected_projection_hash,
                    "jacobian_projection_hash": self.jacobian_projection_hash,
                    "row_index": len(record_hashes),
                },
                prefix="modern-v1-directional-dual-chart-row-proof-",
            )
            if record.dual_chart_projection_proof_hash != expected_row_proof:
                raise ValueError("directional record has a stale dual-chart row proof")
            record_hashes.append(record.record_hash)
        integers = {
            "route_count": self.route_count,
            "maximum_packed_lanes": self.maximum_packed_lanes,
            "gate_workspace_forecast_bytes": self.gate_workspace_forecast_bytes,
            "maximum_gate_workspace_bytes": self.maximum_gate_workspace_bytes,
            "parent_lane_evaluations": self.parent_lane_evaluations,
            "endpoint_lane_evaluations": self.endpoint_lane_evaluations,
            "fused_forward_calls": self.fused_forward_calls,
        }
        if any(
            isinstance(value, bool) or int(value) != value or int(value) < 0
            for value in integers.values()
        ):
            raise ValueError(
                "receipt execution counts must be exact nonnegative integers"
            )
        if (
            int(self.route_count) <= 0
            or int(self.maximum_packed_lanes) <= 0
            or int(self.maximum_gate_workspace_bytes) <= 0
            or int(self.parent_lane_evaluations) != 1
            or int(self.endpoint_lane_evaluations) != 2 * len(endpoints)
            or sum(self.lane_packet_sizes) != int(self.endpoint_lane_evaluations)
            or any(
                int(value) <= 0 or int(value) > int(self.maximum_packed_lanes)
                for value in self.lane_packet_sizes
            )
            or int(self.fused_forward_calls) != 1 + len(self.lane_packet_sizes)
            or int(self.gate_workspace_forecast_bytes)
            > int(self.maximum_gate_workspace_bytes)
            or int(self.gate_workspace_forecast_bytes)
            != int(self.route_count) * (2 + 4 * int(self.maximum_packed_lanes))
        ):
            raise ValueError("directional execution accounting is invalid")
        if tuple(float(value) for value in self.doses) != DIRECTIONAL_SECANT_DOSES:
            raise ValueError("directional execution receipt has noncanonical doses")
        if (
            self.endpoint_order_unchanged_after_outcomes is not True
            or self.candidate_selection_performed is not False
            or self.candidate_reordering_performed is not False
            or self.bce_oracle_authority is not False
            or self.mask_commit_authority is not False
            or self.terminal_authority is not False
            or self.authority_scope != PACKED_DIRECTIONAL_SECANT_AUTHORITY_SCOPE
            or self.schema != PACKED_DIRECTIONAL_SECANT_RECEIPT_SCHEMA
        ):
            raise PermissionError(
                "directional execution receipt exceeds measurement authority"
            )
        expected_prefix_hash = (
            sealed_directional_context_prefix_hash(
                design_hash=self.design_hash,
                context_ids_by_stratum=contexts,
                draw_ids=draws,
            )
            if all(start == 0 for _name, start in starts)
            else sealed_directional_context_slice_hash(
                design_hash=self.design_hash,
                context_ids_by_stratum=contexts,
                sealed_order_start_indices_by_stratum=starts,
                draw_ids=draws,
            )
        )
        if self.sealed_context_prefix_hash != expected_prefix_hash:
            raise ValueError("directional receipt has a stale context-prefix hash")
        corrected_centers = np.stack(
            [record.corrected_projection_center_by_endpoint for record in records],
            axis=0,
        )
        corrected_radii = np.stack(
            [record.corrected_projection_radius_by_endpoint for record in records],
            axis=0,
        )
        jacobian_centers = np.stack(
            [record.jacobian_projection_center_by_endpoint for record in records],
            axis=0,
        )
        jacobian_radii = np.stack(
            [record.jacobian_projection_radius_by_endpoint for record in records],
            axis=0,
        )
        if self.corrected_projection_hash != directional_projection_hash(
            kind="corrected_evidence",
            design_hash=self.design_hash,
            chart_identity_hash=self.chart_identity_hash,
            context_ids=flat_contexts,
            endpoint_action_hashes=endpoints,
            centers=corrected_centers,
            radii=corrected_radii,
        ):
            raise ValueError("receipt corrected projection hash is stale")
        if self.jacobian_projection_hash != directional_projection_hash(
            kind="literal_positive_removal_jacobian",
            design_hash=self.design_hash,
            chart_identity_hash=self.chart_identity_hash,
            context_ids=flat_contexts,
            endpoint_action_hashes=endpoints,
            centers=jacobian_centers,
            radii=jacobian_radii,
        ):
            raise ValueError("receipt Jacobian projection hash is stale")
        parent_centers = np.asarray(
            [record.parent_raw_target_logit_center for record in records],
            dtype=np.float64,
        )
        dosed_centers = np.stack(
            [record.dosed_raw_target_logit_center_by_endpoint for record in records],
            axis=0,
        )
        if self.parent_raw_logit_hash != content_hash(parent_centers):
            raise ValueError("receipt parent raw-logit hash is stale")
        if self.dosed_raw_logit_hash != content_hash(dosed_centers):
            raise ValueError("receipt dosed raw-logit hash is stale")
        payload = {
            "schema": self.schema,
            "design_hash": self.design_hash,
            "ladder_hash": self.ladder_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "parent_mask_hash": self.parent_mask_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": int(self.route_count),
            "endpoint_action_hashes": endpoints,
            "endpoint_mask_hashes": endpoint_masks,
            "packed_direction_hashes": directions,
            "sealed_context_prefix_hash": self.sealed_context_prefix_hash,
            "context_ids_by_stratum": contexts,
            "sealed_order_start_indices_by_stratum": starts,
            "draw_ids": draws,
            "corrected_projection_hash": self.corrected_projection_hash,
            "jacobian_projection_hash": self.jacobian_projection_hash,
            "dual_chart_projection_proof_hash": self.dual_chart_projection_proof_hash,
            "provider_source_hash": self.provider_source_hash,
            "relaxed_gate_executor_source_hash": self.relaxed_gate_executor_source_hash,
            "raw_logit_interval_verifier_source_hash": (
                self.raw_logit_interval_verifier_source_hash
            ),
            "raw_logit_interval_proof_hash": self.raw_logit_interval_proof_hash,
            "observation_bounds_hash": self.observation_bounds_hash,
            "maximum_packed_lanes": int(self.maximum_packed_lanes),
            "gate_workspace_forecast_bytes": int(self.gate_workspace_forecast_bytes),
            "maximum_gate_workspace_bytes": int(self.maximum_gate_workspace_bytes),
            "parent_lane_evaluations": int(self.parent_lane_evaluations),
            "endpoint_lane_evaluations": int(self.endpoint_lane_evaluations),
            "fused_forward_calls": int(self.fused_forward_calls),
            "lane_packet_sizes": tuple(int(value) for value in self.lane_packet_sizes),
            "parent_raw_logit_hash": self.parent_raw_logit_hash,
            "dosed_raw_logit_hash": self.dosed_raw_logit_hash,
            "record_hashes": tuple(record_hashes),
            "execution_proof_hash": self.execution_proof_hash,
            "doses": DIRECTIONAL_SECANT_DOSES,
            "endpoint_order_unchanged_after_outcomes": True,
            "candidate_selection_performed": False,
            "candidate_reordering_performed": False,
            "bce_oracle_authority": False,
            "mask_commit_authority": False,
            "terminal_authority": False,
            "authority_scope": self.authority_scope,
        }
        object.__setattr__(self, "endpoint_action_hashes", endpoints)
        object.__setattr__(self, "endpoint_mask_hashes", endpoint_masks)
        object.__setattr__(self, "packed_direction_hashes", directions)
        object.__setattr__(self, "context_ids_by_stratum", contexts)
        object.__setattr__(
            self,
            "sealed_order_start_indices_by_stratum",
            starts,
        )
        object.__setattr__(self, "draw_ids", draws)
        object.__setattr__(self, "records", records)
        object.__setattr__(
            self,
            "_receipt_hash",
            content_hash(payload, prefix=_RECEIPT_PREFIX),
        )

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "design_hash": self.design_hash,
            "ladder_hash": self.ladder_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "parent_mask_hash": self.parent_mask_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": int(self.route_count),
            "endpoint_action_hashes": self.endpoint_action_hashes,
            "endpoint_mask_hashes": self.endpoint_mask_hashes,
            "packed_direction_hashes": self.packed_direction_hashes,
            "sealed_context_prefix_hash": self.sealed_context_prefix_hash,
            "context_ids_by_stratum": self.context_ids_by_stratum,
            "sealed_order_start_indices_by_stratum": (
                self.sealed_order_start_indices_by_stratum
            ),
            "draw_ids": self.draw_ids,
            "corrected_projection_hash": self.corrected_projection_hash,
            "jacobian_projection_hash": self.jacobian_projection_hash,
            "dual_chart_projection_proof_hash": self.dual_chart_projection_proof_hash,
            "provider_source_hash": self.provider_source_hash,
            "relaxed_gate_executor_source_hash": (
                self.relaxed_gate_executor_source_hash
            ),
            "raw_logit_interval_verifier_source_hash": (
                self.raw_logit_interval_verifier_source_hash
            ),
            "raw_logit_interval_proof_hash": self.raw_logit_interval_proof_hash,
            "observation_bounds_hash": self.observation_bounds_hash,
            "maximum_packed_lanes": int(self.maximum_packed_lanes),
            "gate_workspace_forecast_bytes": int(
                self.gate_workspace_forecast_bytes
            ),
            "maximum_gate_workspace_bytes": int(self.maximum_gate_workspace_bytes),
            "parent_lane_evaluations": int(self.parent_lane_evaluations),
            "endpoint_lane_evaluations": int(self.endpoint_lane_evaluations),
            "fused_forward_calls": int(self.fused_forward_calls),
            "lane_packet_sizes": tuple(int(value) for value in self.lane_packet_sizes),
            "parent_raw_logit_hash": self.parent_raw_logit_hash,
            "dosed_raw_logit_hash": self.dosed_raw_logit_hash,
            "record_hashes": tuple(record.record_hash for record in self.records),
            "execution_proof_hash": self.execution_proof_hash,
            "doses": DIRECTIONAL_SECANT_DOSES,
            "endpoint_order_unchanged_after_outcomes": True,
            "candidate_selection_performed": False,
            "candidate_reordering_performed": False,
            "bce_oracle_authority": False,
            "mask_commit_authority": False,
            "terminal_authority": False,
            "authority_scope": self.authority_scope,
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for record in self.records:
            record.assert_integrity()
        observed = content_hash(self._payload(), prefix=_RECEIPT_PREFIX)
        if observed != self._receipt_hash:
            raise ValueError("directional execution receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("directional execution receipt hash does not match")
        return observed


@dataclass(frozen=True, slots=True)
class V1PackedFiberedDirectionalSecantExecutionReceipt:
    """Versioned execution receipt for native-fibered F/global-J/whole-R rows."""

    design_hash: str
    ladder_hash: str
    chart_identity_hash: str
    parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    proposal_generation: int
    endpoint_action_hashes: tuple[str, ...]
    endpoint_mask_hashes: tuple[str, ...]
    packed_direction_hashes: tuple[str, ...]
    sealed_context_prefix_hash: str
    context_ids_by_stratum: tuple[tuple[str, tuple[str, ...]], ...]
    sealed_order_start_indices_by_stratum: tuple[tuple[str, int], ...]
    draw_ids: tuple[str, ...]
    projection_closure_hashes: tuple[str, ...]
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    fibered_work_projection_receipt_hash: str
    literal_jacobian_projection_receipt_hash: str
    corrected_evidence_stream_identity_hash: str
    literal_jacobian_stream_identity_hash: str
    raw_logit_stream_identity_hash: str
    certification_stream_commitment_hash: str
    joint_fqr_family_registration_hash: str
    same_capture_and_vjp_execution_proof_hash: str
    parent_execution_identity_hash: str
    provider_source_hash: str
    relaxed_gate_executor_source_hash: str
    raw_logit_interval_verifier_source_hash: str
    raw_logit_interval_proof_hash: str
    observation_bounds_hash: str | None
    maximum_packed_lanes: int
    gate_workspace_forecast_bytes: int
    maximum_gate_workspace_bytes: int
    parent_lane_evaluations: int
    endpoint_lane_evaluations: int
    fused_forward_calls: int
    lane_packet_sizes: tuple[int, ...]
    parent_raw_logit_hash: str
    dosed_raw_logit_hash: str
    records: tuple[V1FiberedDirectionalRecord, ...] = field(
        repr=False,
        compare=False,
    )
    execution_proof_hash: str
    doses: tuple[float, float] = DIRECTIONAL_SECANT_DOSES
    fibered_work_values_are_nonnegative: bool = True
    global_corrected_scalar_projection_used: bool = False
    literal_jacobian_projection_is_global_signed: bool = True
    secants_execute_complete_actions: bool = True
    endpoint_order_unchanged_after_outcomes: bool = True
    candidate_selection_performed: bool = False
    candidate_reordering_performed: bool = False
    calibration_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = PACKED_FIBERED_DIRECTIONAL_SECANT_AUTHORITY_SCOPE
    schema: str = PACKED_FIBERED_DIRECTIONAL_SECANT_RECEIPT_SCHEMA
    _receipt_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        identity_names = (
            "design_hash",
            "ladder_hash",
            "chart_identity_hash",
            "parent_mask_hash",
            "route_axis_hash",
            "sealed_context_prefix_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "fibered_work_projection_receipt_hash",
            "literal_jacobian_projection_receipt_hash",
            "corrected_evidence_stream_identity_hash",
            "literal_jacobian_stream_identity_hash",
            "raw_logit_stream_identity_hash",
            "certification_stream_commitment_hash",
            "joint_fqr_family_registration_hash",
            "same_capture_and_vjp_execution_proof_hash",
            "parent_execution_identity_hash",
            "provider_source_hash",
            "relaxed_gate_executor_source_hash",
            "raw_logit_interval_verifier_source_hash",
            "raw_logit_interval_proof_hash",
            "parent_raw_logit_hash",
            "dosed_raw_logit_hash",
            "execution_proof_hash",
        )
        for name in identity_names:
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        endpoints = tuple(
            _identity(value, name="fibered endpoint action hash")
            for value in self.endpoint_action_hashes
        )
        endpoint_masks = tuple(
            _identity(value, name="fibered endpoint mask hash")
            for value in self.endpoint_mask_hashes
        )
        directions = tuple(
            _identity(value, name="fibered packed direction hash")
            for value in self.packed_direction_hashes
        )
        if not endpoints or not (
            len(endpoints) == len(endpoint_masks) == len(directions)
        ) or len(endpoints) != len(set(endpoints)):
            raise ValueError("fibered receipt endpoint/direction axes are invalid")
        contexts = tuple(
            (
                _identity(name, name="fibered receipt stratum id"),
                tuple(
                    _identity(item, name="fibered receipt context id")
                    for item in axis
                ),
            )
            for name, axis in self.context_ids_by_stratum
        )
        starts_list: list[tuple[str, int]] = []
        for name, value in self.sealed_order_start_indices_by_stratum:
            stratum = _identity(name, name="fibered receipt start stratum id")
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError("fibered receipt slice starts must be exact integers")
            start = int(value)
            if start < 0:
                raise ValueError("fibered receipt slice starts must be nonnegative")
            starts_list.append((stratum, start))
        starts = tuple(starts_list)
        flat_contexts = tuple(item for _name, axis in contexts for item in axis)
        draws = tuple(
            _identity(value, name="fibered draw id") for value in self.draw_ids
        )
        closure_hashes = tuple(
            _identity(value, name="fibered projection closure hash")
            for value in self.projection_closure_hashes
        )
        records = tuple(self.records)
        if (
            not contexts
            or tuple(name for name, _axis in contexts)
            != tuple(name for name, _start in starts)
            or not flat_contexts
            or len(flat_contexts) != len(draws)
            or len(records) != len(flat_contexts)
            or len(closure_hashes) != len(flat_contexts)
            or len(flat_contexts) != len(set(flat_contexts))
            or len(draws) != len(set(draws))
            or len(closure_hashes) != len(set(closure_hashes))
            or any(type(record) is not V1FiberedDirectionalRecord for record in records)
        ):
            raise ValueError("fibered receipt C, closure, draw, and record axes differ")
        expected_rows = tuple(
            (stratum, start + offset, context)
            for (stratum, axis), (_start_name, start) in zip(
                contexts,
                starts,
                strict=True,
            )
            for offset, context in enumerate(axis)
        )
        record_hashes: list[str] = []
        for (stratum, sealed_index, context), draw, closure_hash, record in zip(
            expected_rows,
            draws,
            closure_hashes,
            records,
            strict=True,
        ):
            record.assert_integrity()
            expected = {
                "projection_closure_hash": closure_hash,
                "design_hash": self.design_hash,
                "ladder_hash": self.ladder_hash,
                "chart_identity_hash": self.chart_identity_hash,
                "parent_mask_hash": self.parent_mask_hash,
                "parent_execution_identity_hash": self.parent_execution_identity_hash,
                "proposal_generation": int(self.proposal_generation),
                "route_count": int(self.route_count),
                "endpoint_action_hashes": endpoints,
                "stratum_id": stratum,
                "sealed_order_index": sealed_index,
                "context_id": context,
                "draw_id": draw,
                "certification_stream_commitment_hash": (
                    self.certification_stream_commitment_hash
                ),
                "joint_fqr_family_registration_hash": (
                    self.joint_fqr_family_registration_hash
                ),
                "corrected_evidence_stream_identity_hash": (
                    self.corrected_evidence_stream_identity_hash
                ),
                "literal_jacobian_stream_identity_hash": (
                    self.literal_jacobian_stream_identity_hash
                ),
                "raw_logit_stream_identity_hash": self.raw_logit_stream_identity_hash,
                "same_capture_and_vjp_execution_proof_hash": (
                    self.same_capture_and_vjp_execution_proof_hash
                ),
                "native_fiber_axis_receipt_hash": (
                    self.native_fiber_axis_receipt_hash
                ),
                "native_fiber_measure_hash": self.native_fiber_measure_hash,
                "fibered_work_projection_receipt_hash": (
                    self.fibered_work_projection_receipt_hash
                ),
                "literal_jacobian_projection_receipt_hash": (
                    self.literal_jacobian_projection_receipt_hash
                ),
                "packed_direction_execution_proof_hash": self.execution_proof_hash,
                "raw_logit_interval_proof_hash": self.raw_logit_interval_proof_hash,
            }
            if any(getattr(record, name) != value for name, value in expected.items()):
                raise ValueError("fibered directional record was recontextualized")
            record_hashes.append(record.record_hash)
        integers = {
            "route_count": self.route_count,
            "proposal_generation": self.proposal_generation,
            "maximum_packed_lanes": self.maximum_packed_lanes,
            "gate_workspace_forecast_bytes": self.gate_workspace_forecast_bytes,
            "maximum_gate_workspace_bytes": self.maximum_gate_workspace_bytes,
            "parent_lane_evaluations": self.parent_lane_evaluations,
            "endpoint_lane_evaluations": self.endpoint_lane_evaluations,
            "fused_forward_calls": self.fused_forward_calls,
        }
        if any(
            isinstance(value, bool) or int(value) != value or int(value) < 0
            for value in integers.values()
        ):
            raise ValueError("fibered execution counts must be exact nonnegative integers")
        packets: list[int] = []
        for value in self.lane_packet_sizes:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError("fibered lane packet sizes must be exact integers")
            packets.append(int(value))
        lane_packet_sizes = tuple(packets)
        if (
            int(self.route_count) <= 0
            or int(self.maximum_packed_lanes) <= 0
            or int(self.maximum_gate_workspace_bytes) <= 0
            or int(self.parent_lane_evaluations) != 1
            or int(self.endpoint_lane_evaluations) != 2 * len(endpoints)
            or sum(lane_packet_sizes) != int(self.endpoint_lane_evaluations)
            or any(
                int(value) <= 0 or int(value) > int(self.maximum_packed_lanes)
                for value in lane_packet_sizes
            )
            or int(self.fused_forward_calls) != 1 + len(lane_packet_sizes)
            or int(self.gate_workspace_forecast_bytes)
            > int(self.maximum_gate_workspace_bytes)
            or int(self.gate_workspace_forecast_bytes)
            != int(self.route_count) * (2 + 4 * int(self.maximum_packed_lanes))
        ):
            raise ValueError("fibered directional execution accounting is invalid")
        if tuple(float(value) for value in self.doses) != DIRECTIONAL_SECANT_DOSES:
            raise ValueError("fibered directional receipt has noncanonical doses")
        if (
            self.corrected_evidence_stream_identity_hash
            == self.literal_jacobian_stream_identity_hash
        ):
            raise ValueError("corrected-evidence and literal-J streams collapsed")
        if (
            self.fibered_work_values_are_nonnegative is not True
            or self.global_corrected_scalar_projection_used is not False
            or self.literal_jacobian_projection_is_global_signed is not True
            or self.secants_execute_complete_actions is not True
            or self.endpoint_order_unchanged_after_outcomes is not True
            or self.candidate_selection_performed is not False
            or self.candidate_reordering_performed is not False
            or self.calibration_authority is not False
            or self.bce_oracle_authority is not False
            or self.mask_commit_authority is not False
            or self.terminal_authority is not False
            or self.authority_scope != PACKED_FIBERED_DIRECTIONAL_SECANT_AUTHORITY_SCOPE
            or self.schema != PACKED_FIBERED_DIRECTIONAL_SECANT_RECEIPT_SCHEMA
        ):
            raise PermissionError("fibered directional receipt exceeds evidence authority")
        expected_prefix_hash = (
            sealed_directional_context_prefix_hash(
                design_hash=self.design_hash,
                context_ids_by_stratum=contexts,
                draw_ids=draws,
            )
            if all(start == 0 for _name, start in starts)
            else sealed_directional_context_slice_hash(
                design_hash=self.design_hash,
                context_ids_by_stratum=contexts,
                sealed_order_start_indices_by_stratum=starts,
                draw_ids=draws,
            )
        )
        if self.sealed_context_prefix_hash != expected_prefix_hash:
            raise ValueError("fibered receipt has a stale context-slice hash")
        parent_centers = np.asarray(
            [record.parent_raw_target_logit_center for record in records],
            dtype=np.float64,
        )
        dosed_centers = np.stack(
            [record.dosed_raw_target_logit_center_by_endpoint for record in records],
            axis=0,
        )
        if self.parent_raw_logit_hash != content_hash(parent_centers):
            raise ValueError("fibered receipt parent raw-logit hash is stale")
        if self.dosed_raw_logit_hash != content_hash(dosed_centers):
            raise ValueError("fibered receipt dosed raw-logit hash is stale")
        bounds_hash = (
            None
            if self.observation_bounds_hash is None
            else _identity(
                self.observation_bounds_hash,
                name="fibered observation-bounds hash",
            )
        )
        object.__setattr__(self, "endpoint_action_hashes", endpoints)
        object.__setattr__(self, "endpoint_mask_hashes", endpoint_masks)
        object.__setattr__(self, "packed_direction_hashes", directions)
        object.__setattr__(self, "context_ids_by_stratum", contexts)
        object.__setattr__(self, "sealed_order_start_indices_by_stratum", starts)
        object.__setattr__(self, "draw_ids", draws)
        object.__setattr__(self, "projection_closure_hashes", closure_hashes)
        object.__setattr__(self, "observation_bounds_hash", bounds_hash)
        object.__setattr__(self, "lane_packet_sizes", lane_packet_sizes)
        for name, value in integers.items():
            object.__setattr__(self, name, int(value))
        object.__setattr__(self, "records", records)
        object.__setattr__(
            self,
            "_receipt_hash",
            content_hash(
                self._payload(record_hashes=tuple(record_hashes)),
                prefix=_FIBERED_RECEIPT_PREFIX,
            ),
        )

    def _payload(
        self,
        *,
        record_hashes: tuple[str, ...] | None = None,
    ) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "design_hash",
                "ladder_hash",
                "chart_identity_hash",
                "parent_mask_hash",
                "route_axis_hash",
                "route_count",
                "proposal_generation",
                "endpoint_action_hashes",
                "endpoint_mask_hashes",
                "packed_direction_hashes",
                "sealed_context_prefix_hash",
                "context_ids_by_stratum",
                "sealed_order_start_indices_by_stratum",
                "draw_ids",
                "projection_closure_hashes",
                "native_fiber_axis_receipt_hash",
                "native_fiber_measure_hash",
                "fibered_work_projection_receipt_hash",
                "literal_jacobian_projection_receipt_hash",
                "corrected_evidence_stream_identity_hash",
                "literal_jacobian_stream_identity_hash",
                "raw_logit_stream_identity_hash",
                "certification_stream_commitment_hash",
                "joint_fqr_family_registration_hash",
                "same_capture_and_vjp_execution_proof_hash",
                "parent_execution_identity_hash",
                "provider_source_hash",
                "relaxed_gate_executor_source_hash",
                "raw_logit_interval_verifier_source_hash",
                "raw_logit_interval_proof_hash",
                "observation_bounds_hash",
                "maximum_packed_lanes",
                "gate_workspace_forecast_bytes",
                "maximum_gate_workspace_bytes",
                "parent_lane_evaluations",
                "endpoint_lane_evaluations",
                "fused_forward_calls",
                "lane_packet_sizes",
                "parent_raw_logit_hash",
                "dosed_raw_logit_hash",
                "execution_proof_hash",
                "doses",
                "fibered_work_values_are_nonnegative",
                "global_corrected_scalar_projection_used",
                "literal_jacobian_projection_is_global_signed",
                "secants_execute_complete_actions",
                "endpoint_order_unchanged_after_outcomes",
                "candidate_selection_performed",
                "candidate_reordering_performed",
                "calibration_authority",
                "bce_oracle_authority",
                "mask_commit_authority",
                "terminal_authority",
                "authority_scope",
            )
        } | {
            "record_hashes": (
                tuple(record.record_hash for record in self.records)
                if record_hashes is None
                else record_hashes
            )
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for record in self.records:
            record.assert_integrity()
        observed = content_hash(self._payload(), prefix=_FIBERED_RECEIPT_PREFIX)
        if observed != self._receipt_hash:
            raise ValueError("fibered directional execution receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("fibered directional execution receipt hash does not match")
        return observed


RawLogitIntervalVerifier = Callable[
    [np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, str],
]


@dataclass(frozen=True, slots=True)
class _PackedCompleteActionSecants:
    parent: np.ndarray = field(repr=False)
    parent_radius: np.ndarray = field(repr=False)
    dosed: np.ndarray = field(repr=False)
    dosed_radius: np.ndarray = field(repr=False)
    interval_proof_hash: str
    packet_sizes: tuple[int, ...]
    packed_direction_hashes: tuple[str, ...]
    endpoint_mask_hashes: tuple[str, ...]


class NativeTorchV1DirectionalSecantProvider:
    """Execute one frozen packed ladder over one sealed C-stream prefix."""

    def __init__(
        self,
        *,
        evaluator: Any,
        design: V1DirectionalSecantDesign,
        batch_factory: Callable[[tuple[str, ...]], Any],
        parent_mask: HardMask,
        sealed_context_prefix_ids_by_stratum: Sequence[tuple[str, Sequence[str]]],
        sealed_order_start_indices_by_stratum: (
            Sequence[tuple[str, int]] | None
        ) = None,
        draw_ids: Sequence[str],
        sealed_context_prefix_hash: str,
        corrected_projection_center_by_context_endpoint: object,
        corrected_projection_radius_by_context_endpoint: object,
        corrected_projection_hash: str,
        jacobian_projection_center_by_context_endpoint: object,
        jacobian_projection_radius_by_context_endpoint: object,
        jacobian_projection_hash: str,
        dual_chart_projection_proof_hash: str,
        relaxed_gate_executor_source_hash: str,
        raw_logit_interval_verifier_source_hash: str,
        raw_logit_interval_verifier: RawLogitIntervalVerifier | None = None,
        observation_bounds: V1DirectionalSecantObservationBounds | None = None,
        maximum_packed_lanes: int = 8,
        maximum_gate_workspace_bytes: int = 8 * 1024**3,
        target_readout_index: int = 0,
    ) -> None:
        if type(design) is not V1DirectionalSecantDesign:
            raise TypeError("directional secants require the exact typed design")
        design_hash = design.assert_integrity()
        if not design.ladder.is_packed:
            raise ValueError("directional secants require packed frozen directions")
        if type(parent_mask) is not HardMask:
            raise TypeError("directional secants require the exact HardMask parent")
        if design.ladder.parent_mask is None:
            raise ValueError("packed directional ladder lost its parent mask")
        if (
            parent_mask.mask_hash != design.ladder.current_mask_hash
            or parent_mask.mask_hash != design.ladder.parent_mask.mask_hash
            or parent_mask.route_axis_hash != design.ladder.route_axis_hash
            or parent_mask.route_count != design.ladder.route_count
        ):
            raise ValueError("directional parent mask differs from the frozen design")
        if (
            design.packed_direction_provider_source_hash
            != directional_secant_provider_source_hash()
        ):
            raise ValueError("directional design names a stale provider source")
        executor_hash = _identity(
            relaxed_gate_executor_source_hash,
            name="relaxed-gate executor source hash",
        )
        interval_source_hash = _identity(
            raw_logit_interval_verifier_source_hash,
            name="raw-logit interval verifier source hash",
        )
        if executor_hash != design.relaxed_gate_executor_source_hash:
            raise ValueError("relaxed-gate executor source hash is stale")
        if interval_source_hash != design.raw_logit_interval_verifier_source_hash:
            raise ValueError("raw-logit interval verifier source hash is stale")
        if (
            raw_logit_interval_verifier is None
            and interval_source_hash != DEFAULT_BINARY64_ENCLOSURE_SOURCE_HASH
        ):
            raise ValueError("a nondefault interval source requires its bound verifier")

        design_axes = dict(design.certification_context_ids_by_stratum)
        axes = tuple(
            (
                _identity(name, name="sealed prefix stratum id"),
                tuple(
                    _identity(item, name="sealed prefix context id")
                    for item in axis
                ),
            )
            for name, axis in sealed_context_prefix_ids_by_stratum
        )
        if tuple(name for name, _ in axes) != tuple(design_axes):
            raise ValueError("sealed slice strata differ from the frozen C design")
        if sealed_order_start_indices_by_stratum is None:
            starts = tuple((name, 0) for name, _axis in axes)
        else:
            normalized_starts: list[tuple[str, int]] = []
            for name, value in sealed_order_start_indices_by_stratum:
                stratum = _identity(name, name="sealed slice-start stratum id")
                if isinstance(value, bool) or not isinstance(
                    value, (int, np.integer)
                ):
                    raise ValueError("sealed slice starts must be exact integers")
                start = int(value)
                if start < 0:
                    raise ValueError("sealed slice starts must be nonnegative")
                normalized_starts.append((stratum, start))
            starts = tuple(normalized_starts)
        if tuple(name for name, _ in starts) != tuple(name for name, _ in axes):
            raise ValueError("sealed slice-start strata differ from the C design")
        start_by_name = dict(starts)
        for name, axis in axes:
            start = start_by_name[name]
            stop = start + len(axis)
            if (
                start > len(design_axes[name])
                or stop > len(design_axes[name])
                or tuple(design_axes[name][start:stop]) != axis
            ):
                raise ValueError(
                    "directional contexts must be exact sealed C-stream slices"
                )
        contexts = tuple(item for _, axis in axes for item in axis)
        if not contexts or len(contexts) != len(set(contexts)):
            raise ValueError("sealed directional context slice is empty or repeats")
        draws = tuple(
            _identity(value, name="directional draw id") for value in draw_ids
        )
        if len(draws) != len(contexts) or len(draws) != len(set(draws)):
            raise ValueError("directional draw ledger differs from the context slice")
        expected_prefix_hash = (
            sealed_directional_context_prefix_hash(
                design_hash=design_hash,
                context_ids_by_stratum=axes,
                draw_ids=draws,
            )
            if all(start == 0 for _name, start in starts)
            else sealed_directional_context_slice_hash(
                design_hash=design_hash,
                context_ids_by_stratum=axes,
                sealed_order_start_indices_by_stratum=starts,
                draw_ids=draws,
            )
        )
        if sealed_context_prefix_hash != expected_prefix_hash:
            raise ValueError("sealed directional context-slice hash is stale")

        endpoint_hashes = design.endpoint_action_hashes
        shape = (len(contexts), len(endpoint_hashes))
        corrected = _immutable_f64(
            corrected_projection_center_by_context_endpoint,
            shape=shape,
            name="corrected endpoint projections",
        )
        corrected_radius = _immutable_f64(
            corrected_projection_radius_by_context_endpoint,
            shape=shape,
            name="corrected endpoint projection radii",
            nonnegative=True,
        )
        jacobian = _immutable_f64(
            jacobian_projection_center_by_context_endpoint,
            shape=shape,
            name="literal Jacobian endpoint projections",
        )
        jacobian_radius = _immutable_f64(
            jacobian_projection_radius_by_context_endpoint,
            shape=shape,
            name="literal Jacobian endpoint projection radii",
            nonnegative=True,
        )
        expected_corrected_hash = directional_projection_hash(
            kind="corrected_evidence",
            design_hash=design_hash,
            chart_identity_hash=design.chart_identity_hash,
            context_ids=contexts,
            endpoint_action_hashes=endpoint_hashes,
            centers=corrected,
            radii=corrected_radius,
        )
        expected_jacobian_hash = directional_projection_hash(
            kind="literal_positive_removal_jacobian",
            design_hash=design_hash,
            chart_identity_hash=design.chart_identity_hash,
            context_ids=contexts,
            endpoint_action_hashes=endpoint_hashes,
            centers=jacobian,
            radii=jacobian_radius,
        )
        if corrected_projection_hash != expected_corrected_hash:
            raise ValueError("corrected endpoint projection hash is stale")
        if jacobian_projection_hash != expected_jacobian_hash:
            raise ValueError("literal Jacobian endpoint projection hash is stale")

        lanes = int(maximum_packed_lanes)
        ceiling = int(maximum_gate_workspace_bytes)
        target_index = int(target_readout_index)
        if (
            isinstance(maximum_packed_lanes, bool)
            or lanes != maximum_packed_lanes
            or lanes <= 0
            or isinstance(maximum_gate_workspace_bytes, bool)
            or ceiling != maximum_gate_workspace_bytes
            or ceiling <= 0
            or isinstance(target_readout_index, bool)
            or target_index != target_readout_index
            or target_index < 0
        ):
            raise ValueError("directional execution limits/readout index are invalid")
        blocks = tuple(getattr(evaluator, "route_blocks", ()))
        if (
            not blocks
            or len({str(block.layer_id) for block in blocks}) != len(blocks)
            or sum(int(np.prod(block.shape)) for block in blocks)
            != design.ladder.route_count
        ):
            raise ValueError("directional evaluator route axis is invalid")
        # One resident parent Boolean axis, one transient child Boolean axis, and
        # at most ``lanes`` float32 gate axes.  Compiled backends may allocate
        # their own registered workspace; this receipt forecasts this provider's
        # route-scale materialization only.
        forecast = int(design.ladder.route_count) * (1 + 1 + 4 * lanes)
        if forecast > ceiling:
            raise MemoryError("directional gate workspace forecast exceeds its ceiling")

        bounds = observation_bounds
        if bounds is not None:
            if type(bounds) is not V1DirectionalSecantObservationBounds:
                raise TypeError("directional observation bounds use the wrong type")
            if (
                bounds.design_hash != design_hash
                or bounds.endpoint_action_hashes != endpoint_hashes
            ):
                raise ValueError("directional observation bounds name another design")
            bounds.assert_integrity()

        self.evaluator = evaluator
        self.design = design
        self.batch_factory = batch_factory
        self.parent_mask = parent_mask
        self.context_axes = axes
        self.sealed_order_start_indices_by_stratum = starts
        self.sealed_order_start_index_by_stratum = start_by_name
        self.context_ids = contexts
        self.draw_ids = draws
        self.sealed_context_prefix_hash = expected_prefix_hash
        self.corrected = corrected
        self.corrected_radius = corrected_radius
        self.jacobian = jacobian
        self.jacobian_radius = jacobian_radius
        self.corrected_projection_hash = expected_corrected_hash
        self.jacobian_projection_hash = expected_jacobian_hash
        self.dual_chart_projection_proof_hash = _identity(
            dual_chart_projection_proof_hash,
            name="dual-chart projection proof hash",
        )
        self.relaxed_gate_executor_source_hash = executor_hash
        self.raw_logit_interval_verifier_source_hash = interval_source_hash
        self.raw_logit_interval_verifier = raw_logit_interval_verifier
        self.observation_bounds = bounds
        self.maximum_packed_lanes = lanes
        self.maximum_gate_workspace_bytes = ceiling
        self.gate_workspace_forecast_bytes = forecast
        self.target_readout_index = target_index
        self.route_blocks = blocks

    def _gate_mapping(
        self,
        *,
        endpoint_mask: HardMask | None,
        dose: float,
    ) -> dict[str, np.ndarray]:
        parent = self.parent_mask.to_bool_array()
        child = None if endpoint_mask is None else endpoint_mask.to_bool_array()
        if child is not None and np.any(child & ~parent):
            raise ValueError("a frozen directional endpoint reactivates a parent route")
        gates: dict[str, np.ndarray] = {}
        offset = 0
        for block in self.route_blocks:
            count = int(np.prod(block.shape))
            parent_block = parent[offset : offset + count].astype(np.float32, copy=True)
            if child is not None:
                child_block = child[offset : offset + count]
                removed = parent[offset : offset + count] & ~child_block
                parent_block -= np.float32(dose) * removed.astype(np.float32)
            if (
                not np.isfinite(parent_block).all()
                or np.any(parent_block < 0.0)
                or np.any(parent_block > 1.0)
            ):
                raise ValueError("relaxed directional gates left [0,1]")
            gates[str(block.layer_id)] = parent_block.reshape(block.shape)
            offset += count
        return gates

    def _compiled(self, gates: Mapping[str, np.ndarray]) -> CompiledTorchCircuitMask:
        compiler = getattr(self.evaluator, "compile_gate_values", None)
        if callable(compiler):
            compiled = compiler(dict(gates))
            if type(compiled) is not CompiledTorchCircuitMask:
                raise TypeError("relaxed-gate compiler returned a noncanonical mask")
            return compiled
        return CompiledTorchCircuitMask(
            parameter_overrides={},
            residual_gates={},
            gate_values=dict(gates),
            float_gate_values=dict(gates),
        )

    def _raw_target_logits(self, response: Any) -> np.ndarray:
        if tuple(map(str, response.context_ids)) != self.context_ids:
            raise ValueError("directional evaluator changed the sealed context axis")
        values = np.asarray(response.values, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[0] != len(self.context_ids)
            or self.target_readout_index >= values.shape[1]
            or not np.isfinite(values).all()
        ):
            raise ValueError("directional evaluator returned invalid raw logits")
        return values[:, self.target_readout_index].copy()

    @staticmethod
    def _default_radii(values: np.ndarray) -> np.ndarray:
        upper = np.nextafter(values, np.where(values >= 0.0, np.inf, -np.inf))
        radii = np.abs(upper - values)
        radii[values == 0.0] = np.nextafter(0.0, 1.0)
        return radii

    def _raw_logit_enclosures(
        self,
        parent: np.ndarray,
        dosed: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        if self.raw_logit_interval_verifier is None:
            parent_radius = self._default_radii(parent)
            dosed_radius = self._default_radii(dosed)
            proof = content_hash(
                {
                    "source_hash": DEFAULT_BINARY64_ENCLOSURE_SOURCE_HASH,
                    "parent_centers": parent,
                    "parent_radii": parent_radius,
                    "dosed_centers": dosed,
                    "dosed_radii": dosed_radius,
                    "claim": "binary64_reported_center_numerical_enclosure_only",
                },
                prefix=_INTERVAL_PREFIX,
            )
            return parent_radius, dosed_radius, proof
        parent_radius, dosed_radius, proof = self.raw_logit_interval_verifier(
            parent.copy(),
            dosed.copy(),
        )
        return (
            _immutable_f64(
                parent_radius,
                shape=parent.shape,
                name="parent raw-logit radii",
                nonnegative=True,
            ),
            _immutable_f64(
                dosed_radius,
                shape=dosed.shape,
                name="dosed raw-logit radii",
                nonnegative=True,
            ),
            _identity(proof, name="raw-logit interval proof hash"),
        )

    def _validate_bounds(
        self,
        *,
        parent: np.ndarray,
        parent_radius: np.ndarray,
        dosed: np.ndarray,
        dosed_radius: np.ndarray,
    ) -> None:
        bounds = self.observation_bounds
        if bounds is None:
            return
        if (
            np.any(np.abs(parent) + parent_radius > bounds.raw_logit_absolute_upper)
            or np.any(
                np.abs(dosed) + dosed_radius
                > bounds.raw_logit_absolute_upper
            )
            or np.any(
                np.abs(self.corrected) + self.corrected_radius
                > bounds.corrected_projection_absolute_upper_by_endpoint[None, :]
            )
            or np.any(
                np.abs(self.jacobian) + self.jacobian_radius
                > bounds.jacobian_projection_absolute_upper_by_endpoint[None, :]
            )
        ):
            raise ValueError(
                "directional observation exceeded a pre-outcome finite bound"
            )
        delta = dosed - parent[:, None, None]
        remainder = delta - self.jacobian[:, :, None] * np.asarray(
            DIRECTIONAL_SECANT_DOSES,
            dtype=np.float64,
        )[None, None, :]
        remainder_radius = (
            dosed_radius
            + parent_radius[:, None, None]
            + self.jacobian_radius[:, :, None]
            * np.asarray(DIRECTIONAL_SECANT_DOSES, dtype=np.float64)[None, None, :]
        )
        if np.any(
            np.abs(remainder) + remainder_radius
            > bounds.remainder_absolute_upper_by_endpoint_and_dose[None, :, :]
        ):
            raise ValueError("directional remainder exceeded its pre-outcome bound")

    def _execute_complete_action_secants(self) -> _PackedCompleteActionSecants:
        """Execute the common packed parent/eta={1/2,1} finite-action lanes."""

        self.design.assert_integrity()
        endpoint_masks = self.design.ladder.endpoint_masks
        endpoint_hashes = self.design.endpoint_action_hashes
        pairs = tuple(
            (endpoint_index, dose_index, dose)
            for endpoint_index in range(len(endpoint_masks))
            for dose_index, dose in enumerate(DIRECTIONAL_SECANT_DOSES)
        )
        parent_compiled = self._compiled(
            self._gate_mapping(endpoint_mask=None, dose=0.0)
        )
        parent_response = self.evaluator.readout_responses_from_batches(
            self.batch_factory(self.context_ids),
            compiled_masks=(parent_compiled,),
            lane_keys=(f"directional-parent:{self.design.design_hash}",),
        )
        if len(parent_response) != 1:
            raise ValueError(
                "directional evaluator returned the wrong parent lane count"
            )
        parent = self._raw_target_logits(parent_response[0])
        dosed = np.empty(
            (len(self.context_ids), len(endpoint_masks), 2),
            dtype=np.float64,
        )
        packet_sizes: list[int] = []
        for start in range(0, len(pairs), self.maximum_packed_lanes):
            packet = pairs[start : start + self.maximum_packed_lanes]
            compiled = tuple(
                self._compiled(
                    self._gate_mapping(
                        endpoint_mask=endpoint_masks[endpoint_index],
                        dose=dose,
                    )
                )
                for endpoint_index, _dose_index, dose in packet
            )
            responses = self.evaluator.readout_responses_from_batches(
                self.batch_factory(self.context_ids),
                compiled_masks=compiled,
                lane_keys=tuple(
                    f"directional:{endpoint_hashes[endpoint_index]}:eta:{dose}"
                    for endpoint_index, _dose_index, dose in packet
                ),
            )
            if len(responses) != len(packet):
                raise ValueError(
                    "directional evaluator returned the wrong packed lane count"
                )
            packet_sizes.append(len(packet))
            for (endpoint_index, dose_index, _dose), response in zip(
                packet,
                responses,
                strict=True,
            ):
                dosed[:, endpoint_index, dose_index] = self._raw_target_logits(response)

        parent_radius, dosed_radius, interval_proof_hash = (
            self._raw_logit_enclosures(parent, dosed)
        )
        self._validate_bounds(
            parent=parent,
            parent_radius=parent_radius,
            dosed=dosed,
            dosed_radius=dosed_radius,
        )
        direction_hashes: list[str] = []
        endpoint_mask_hashes: list[str] = []
        for action, mask in zip(
            self.design.ladder.actions,
            endpoint_masks,
            strict=True,
        ):
            count, index_hash = mask.removed_index_count_and_hash_from(self.parent_mask)
            direction_hashes.append(
                content_hash(
                    {
                        "design_hash": self.design.design_hash,
                        "ladder_hash": self.design.ladder_hash,
                        "parent_mask_hash": self.parent_mask.mask_hash,
                        "endpoint_action_hash": action.action_hash,
                        "endpoint_mask_hash": mask.mask_hash,
                        "removed_count": count,
                        "removed_route_index_hash": index_hash,
                    },
                    prefix=_DIRECTION_PREFIX,
                )
            )
            endpoint_mask_hashes.append(mask.mask_hash)
        return _PackedCompleteActionSecants(
            parent=parent,
            parent_radius=parent_radius,
            dosed=dosed,
            dosed_radius=dosed_radius,
            interval_proof_hash=interval_proof_hash,
            packet_sizes=tuple(packet_sizes),
            packed_direction_hashes=tuple(direction_hashes),
            endpoint_mask_hashes=tuple(endpoint_mask_hashes),
        )

    def execute(self) -> V1PackedDirectionalSecantExecutionReceipt:
        """Execute every frozen endpoint/dose without selecting or reordering it."""

        execution = self._execute_complete_action_secants()
        parent = execution.parent
        parent_radius = execution.parent_radius
        dosed = execution.dosed
        dosed_radius = execution.dosed_radius
        interval_proof_hash = execution.interval_proof_hash
        packet_sizes = execution.packet_sizes
        direction_hashes = execution.packed_direction_hashes
        endpoint_mask_hashes = execution.endpoint_mask_hashes
        endpoint_hashes = self.design.endpoint_action_hashes
        execution_proof_hash = content_hash(
            {
                "provider_source_hash": directional_secant_provider_source_hash(),
                "design_hash": self.design.design_hash,
                "ladder_hash": self.design.ladder_hash,
                "parent_mask_hash": self.parent_mask.mask_hash,
                "endpoint_action_hashes": endpoint_hashes,
                "endpoint_mask_hashes": endpoint_mask_hashes,
                "packed_direction_hashes": direction_hashes,
                "sealed_context_prefix_hash": self.sealed_context_prefix_hash,
                "sealed_order_start_indices_by_stratum": (
                    self.sealed_order_start_indices_by_stratum
                ),
                "corrected_projection_hash": self.corrected_projection_hash,
                "jacobian_projection_hash": self.jacobian_projection_hash,
                "relaxed_gate_executor_source_hash": (
                    self.relaxed_gate_executor_source_hash
                ),
                "raw_logit_interval_verifier_source_hash": (
                    self.raw_logit_interval_verifier_source_hash
                ),
                "raw_logit_interval_proof_hash": interval_proof_hash,
                "doses": DIRECTIONAL_SECANT_DOSES,
                "maximum_packed_lanes": self.maximum_packed_lanes,
                "gate_workspace_forecast_bytes": self.gate_workspace_forecast_bytes,
                "lane_packet_sizes": packet_sizes,
                "parent_raw_logit_hash": content_hash(parent),
                "dosed_raw_logit_hash": content_hash(dosed),
            },
            prefix=_EXECUTION_PREFIX,
        )

        record_rows: list[V1DirectionalSecantRecord] = []
        context_cursor = 0
        for stratum, contexts in self.context_axes:
            start_index = self.sealed_order_start_index_by_stratum[stratum]
            for local_index, context in enumerate(contexts):
                sealed_index = start_index + local_index
                row_projection_proof = content_hash(
                    {
                        "dual_chart_projection_proof_hash": (
                            self.dual_chart_projection_proof_hash
                        ),
                        "context_id": context,
                        "draw_id": self.draw_ids[context_cursor],
                        "corrected_projection_hash": self.corrected_projection_hash,
                        "jacobian_projection_hash": self.jacobian_projection_hash,
                        "row_index": context_cursor,
                    },
                    prefix="modern-v1-directional-dual-chart-row-proof-",
                )
                record_rows.append(
                    V1DirectionalSecantRecord(
                        design_hash=self.design.design_hash,
                        ladder_hash=self.design.ladder_hash,
                        chart_identity_hash=self.design.chart_identity_hash,
                        endpoint_action_hashes=endpoint_hashes,
                        stratum_id=stratum,
                        sealed_order_index=sealed_index,
                        context_id=context,
                        draw_id=self.draw_ids[context_cursor],
                        parent_raw_target_logit_center=float(parent[context_cursor]),
                        parent_raw_target_logit_radius=float(
                            parent_radius[context_cursor]
                        ),
                        corrected_projection_center_by_endpoint=self.corrected[
                            context_cursor
                        ],
                        corrected_projection_radius_by_endpoint=self.corrected_radius[
                            context_cursor
                        ],
                        jacobian_projection_center_by_endpoint=self.jacobian[
                            context_cursor
                        ],
                        jacobian_projection_radius_by_endpoint=self.jacobian_radius[
                            context_cursor
                        ],
                        dosed_raw_target_logit_center_by_endpoint=dosed[context_cursor],
                        dosed_raw_target_logit_radius_by_endpoint=dosed_radius[
                            context_cursor
                        ],
                        dual_chart_projection_proof_hash=row_projection_proof,
                        packed_direction_execution_proof_hash=execution_proof_hash,
                        raw_logit_interval_proof_hash=interval_proof_hash,
                    )
                )
                context_cursor += 1
        return V1PackedDirectionalSecantExecutionReceipt(
            design_hash=self.design.design_hash,
            ladder_hash=self.design.ladder_hash,
            chart_identity_hash=self.design.chart_identity_hash,
            parent_mask_hash=self.parent_mask.mask_hash,
            route_axis_hash=self.parent_mask.route_axis_hash,
            route_count=self.parent_mask.route_count,
            endpoint_action_hashes=endpoint_hashes,
            endpoint_mask_hashes=endpoint_mask_hashes,
            packed_direction_hashes=direction_hashes,
            sealed_context_prefix_hash=self.sealed_context_prefix_hash,
            context_ids_by_stratum=self.context_axes,
            sealed_order_start_indices_by_stratum=(
                self.sealed_order_start_indices_by_stratum
            ),
            draw_ids=self.draw_ids,
            corrected_projection_hash=self.corrected_projection_hash,
            jacobian_projection_hash=self.jacobian_projection_hash,
            dual_chart_projection_proof_hash=self.dual_chart_projection_proof_hash,
            provider_source_hash=directional_secant_provider_source_hash(),
            relaxed_gate_executor_source_hash=self.relaxed_gate_executor_source_hash,
            raw_logit_interval_verifier_source_hash=(
                self.raw_logit_interval_verifier_source_hash
            ),
            raw_logit_interval_proof_hash=interval_proof_hash,
            observation_bounds_hash=(
                None
                if self.observation_bounds is None
                else self.observation_bounds.bounds_hash
            ),
            maximum_packed_lanes=self.maximum_packed_lanes,
            gate_workspace_forecast_bytes=self.gate_workspace_forecast_bytes,
            maximum_gate_workspace_bytes=self.maximum_gate_workspace_bytes,
            parent_lane_evaluations=1,
            endpoint_lane_evaluations=2 * len(endpoint_hashes),
            fused_forward_calls=1 + len(packet_sizes),
            lane_packet_sizes=packet_sizes,
            parent_raw_logit_hash=content_hash(parent),
            dosed_raw_logit_hash=content_hash(dosed),
            records=tuple(record_rows),
            execution_proof_hash=execution_proof_hash,
        )


class NativeTorchV1FiberedDirectionalSecantProvider(
    NativeTorchV1DirectionalSecantProvider
):
    """Execute complete-action secants for ordered native-fibered C rows.

    This sibling deliberately bypasses the legacy constructor: it has no
    globally signed corrected-projection field.  Only the shared packed gate
    and raw-logit execution mechanics are inherited.
    """

    def __init__(
        self,
        *,
        evaluator: Any,
        design: V1DirectionalSecantDesign,
        batch_factory: Callable[[tuple[str, ...]], Any],
        parent_mask: HardMask,
        projection_closures: Sequence[V1FiberedDirectionalProjectionClosure],
        native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt,
        sealed_order_start_indices_by_stratum: Sequence[tuple[str, int]],
        sealed_context_prefix_hash: str,
        relaxed_gate_executor_source_hash: str,
        raw_logit_interval_verifier_source_hash: str,
        raw_logit_interval_verifier: RawLogitIntervalVerifier | None = None,
        observation_bounds: (
            V1FiberedDirectionalSecantObservationBounds | None
        ) = None,
        maximum_packed_lanes: int = 8,
        maximum_gate_workspace_bytes: int = 8 * 1024**3,
        target_readout_index: int = 0,
    ) -> None:
        if type(design) is not V1DirectionalSecantDesign:
            raise TypeError("fibered secants require the exact typed design")
        design_hash = design.assert_integrity()
        if not design.ladder.is_packed:
            raise ValueError("fibered secants require packed frozen directions")
        if type(parent_mask) is not HardMask:
            raise TypeError("fibered secants require the exact HardMask parent")
        if design.ladder.parent_mask is None:
            raise ValueError("fibered directional ladder lost its parent mask")
        if (
            parent_mask.mask_hash != design.ladder.current_mask_hash
            or parent_mask.mask_hash != design.ladder.parent_mask.mask_hash
            or parent_mask.route_axis_hash != design.ladder.route_axis_hash
            or parent_mask.route_count != design.ladder.route_count
        ):
            raise ValueError("fibered parent mask differs from the frozen design")
        if (
            design.packed_direction_provider_source_hash
            != directional_secant_provider_source_hash()
        ):
            raise ValueError("fibered design names a stale provider source")
        if type(native_fiber_axis_receipt) is not V1NativeDestinationFiberAxisReceipt:
            raise TypeError("fibered secants require the exact native-fiber receipt")
        native_fiber_axis_receipt.assert_integrity()
        if (
            native_fiber_axis_receipt.route_count != design.ladder.route_count
            or native_fiber_axis_receipt.route_axis_hash
            != design.ladder.route_axis_hash
        ):
            raise ValueError("native-fiber receipt differs from the frozen route axis")

        closures = tuple(projection_closures)
        axes, starts, contexts, draws = fibered_directional_context_slice_axes(
            design=design,
            projection_closures=closures,
            sealed_order_start_indices_by_stratum=(
                sealed_order_start_indices_by_stratum
            ),
        )
        expected_prefix_hash = (
            sealed_directional_context_prefix_hash(
                design_hash=design_hash,
                context_ids_by_stratum=axes,
                draw_ids=draws,
            )
            if all(start == 0 for _name, start in starts)
            else sealed_directional_context_slice_hash(
                design_hash=design_hash,
                context_ids_by_stratum=axes,
                sealed_order_start_indices_by_stratum=starts,
                draw_ids=draws,
            )
        )
        if sealed_context_prefix_hash != expected_prefix_hash:
            raise ValueError("fibered directional context-slice hash is stale")
        first = closures[0]
        expected_design_coordinates = {
            "design_hash": design_hash,
            "ladder_hash": design.ladder_hash,
            "chart_identity_hash": design.chart_identity_hash,
            "parent_mask_hash": design.ladder.current_mask_hash,
            "proposal_generation": design.ladder.proposal_generation,
            "route_count": design.ladder.route_count,
            "endpoint_action_hashes": design.endpoint_action_hashes,
            "certification_stream_commitment_hash": (
                design.certification_stream_commitment_hash
            ),
            "joint_fqr_family_registration_hash": (
                design.joint_fqr_family_registration_hash
            ),
            "raw_logit_stream_identity_hash": (
                design.ladder.raw_logit_stream_source_hash
            ),
            "native_fiber_axis_receipt_hash": (
                native_fiber_axis_receipt.receipt_hash
            ),
            "native_fiber_measure_hash": native_fiber_axis_receipt.measure_hash,
        }
        family_coordinates = {
            "corrected_evidence_stream_identity_hash": (
                first.corrected_evidence_stream_identity_hash
            ),
            "literal_jacobian_stream_identity_hash": (
                first.literal_jacobian_stream_identity_hash
            ),
            "same_capture_and_vjp_execution_proof_hash": (
                first.same_capture_and_vjp_execution_proof_hash
            ),
            "fibered_work_projection_receipt_hash": (
                first.fibered_work_projection_receipt_hash
            ),
            "literal_jacobian_projection_receipt_hash": (
                first.literal_jacobian_projection_receipt_hash
            ),
        }
        for closure in closures:
            if any(
                getattr(closure, name) != value
                for name, value in expected_design_coordinates.items()
            ):
                raise ValueError("fibered closure differs from the frozen design")
            if any(
                getattr(closure, name) != value
                for name, value in family_coordinates.items()
            ):
                raise ValueError("fibered closure rows do not share one VJP family")
            if (
                closure.same_capture_and_vjp_certified is not True
                or closure.global_corrected_scalar_projection_used is not False
                or closure.literal_jacobian_projection_is_global_signed is not True
                or closure.fibered_work_values_are_nonnegative is not True
                or closure.postselection_occurred is not False
            ):
                raise PermissionError("fibered closure lacks canonical evidence scope")
        if (
            first.corrected_evidence_stream_identity_hash
            == first.literal_jacobian_stream_identity_hash
        ):
            raise ValueError("corrected-evidence and literal-J streams collapsed")

        executor_hash = _identity(
            relaxed_gate_executor_source_hash,
            name="relaxed-gate executor source hash",
        )
        interval_source_hash = _identity(
            raw_logit_interval_verifier_source_hash,
            name="raw-logit interval verifier source hash",
        )
        if executor_hash != design.relaxed_gate_executor_source_hash:
            raise ValueError("relaxed-gate executor source hash is stale")
        if interval_source_hash != design.raw_logit_interval_verifier_source_hash:
            raise ValueError("raw-logit interval verifier source hash is stale")
        if (
            raw_logit_interval_verifier is None
            and interval_source_hash != DEFAULT_BINARY64_ENCLOSURE_SOURCE_HASH
        ):
            raise ValueError("a nondefault interval source requires its verifier")

        lanes = int(maximum_packed_lanes)
        ceiling = int(maximum_gate_workspace_bytes)
        target_index = int(target_readout_index)
        if (
            isinstance(maximum_packed_lanes, bool)
            or lanes != maximum_packed_lanes
            or lanes <= 0
            or isinstance(maximum_gate_workspace_bytes, bool)
            or ceiling != maximum_gate_workspace_bytes
            or ceiling <= 0
            or isinstance(target_readout_index, bool)
            or target_index != target_readout_index
            or target_index < 0
        ):
            raise ValueError("fibered directional execution limits are invalid")
        blocks = tuple(getattr(evaluator, "route_blocks", ()))
        if (
            not blocks
            or len({str(block.layer_id) for block in blocks}) != len(blocks)
            or sum(int(np.prod(block.shape)) for block in blocks)
            != design.ladder.route_count
        ):
            raise ValueError("fibered directional evaluator route axis is invalid")
        forecast = int(design.ladder.route_count) * (1 + 1 + 4 * lanes)
        if forecast > ceiling:
            raise MemoryError("fibered directional gate forecast exceeds its ceiling")
        bounds = observation_bounds
        if bounds is not None:
            if type(bounds) is not V1FiberedDirectionalSecantObservationBounds:
                raise TypeError("fibered observation bounds use the wrong type")
            if (
                bounds.design_hash != design_hash
                or bounds.endpoint_action_hashes != design.endpoint_action_hashes
                or bounds.native_fiber_axis_receipt_hash
                != native_fiber_axis_receipt.receipt_hash
                or bounds.native_fiber_measure_hash
                != native_fiber_axis_receipt.measure_hash
            ):
                raise ValueError("fibered observation bounds name another design")
            bounds.assert_integrity()

        self.evaluator = evaluator
        self.design = design
        self.batch_factory = batch_factory
        self.parent_mask = parent_mask
        self.projection_closures = closures
        self.native_fiber_axis_receipt = native_fiber_axis_receipt
        self.context_axes = axes
        self.sealed_order_start_indices_by_stratum = starts
        self.sealed_order_start_index_by_stratum = dict(starts)
        self.context_ids = contexts
        self.draw_ids = draws
        self.sealed_context_prefix_hash = expected_prefix_hash
        self.jacobian = _immutable_f64(
            np.stack(
                [
                    closure.global_literal_jacobian_center_by_endpoint
                    for closure in closures
                ],
                axis=0,
            ),
            shape=(len(closures), len(design.endpoint_action_hashes)),
            name="fibered global literal-J centers",
        )
        self.jacobian_radius = _immutable_f64(
            np.stack(
                [
                    closure.global_literal_jacobian_radius_by_endpoint
                    for closure in closures
                ],
                axis=0,
            ),
            shape=(len(closures), len(design.endpoint_action_hashes)),
            name="fibered global literal-J radii",
            nonnegative=True,
        )
        self.relaxed_gate_executor_source_hash = executor_hash
        self.raw_logit_interval_verifier_source_hash = interval_source_hash
        self.raw_logit_interval_verifier = raw_logit_interval_verifier
        self.observation_bounds = bounds
        self.maximum_packed_lanes = lanes
        self.maximum_gate_workspace_bytes = ceiling
        self.gate_workspace_forecast_bytes = forecast
        self.target_readout_index = target_index
        self.route_blocks = blocks

    def _validate_bounds(
        self,
        *,
        parent: np.ndarray,
        parent_radius: np.ndarray,
        dosed: np.ndarray,
        dosed_radius: np.ndarray,
    ) -> None:
        bounds = self.observation_bounds
        if bounds is None:
            return
        fibered_upper = np.stack(
            [closure.fibered_work_upper_by_endpoint for closure in self.projection_closures],
            axis=0,
        )
        raw_parent_upper = np.nextafter(
            np.abs(parent) + parent_radius,
            np.inf,
        )
        raw_dosed_upper = np.nextafter(
            np.abs(dosed) + dosed_radius,
            np.inf,
        )
        jacobian_upper = np.nextafter(
            np.abs(self.jacobian) + self.jacobian_radius,
            np.inf,
        )
        if (
            np.any(raw_parent_upper > bounds.raw_logit_absolute_upper)
            or np.any(raw_dosed_upper > bounds.raw_logit_absolute_upper)
            or np.any(
                fibered_upper > bounds.fibered_work_upper_by_endpoint[None, :]
            )
            or np.any(
                jacobian_upper
                > bounds.global_literal_jacobian_absolute_upper_by_endpoint[None, :]
            )
        ):
            raise ValueError("fibered directional observation exceeded its bound")

    def execute(self) -> V1PackedFiberedDirectionalSecantExecutionReceipt:
        """Execute all frozen endpoints and close native-fibered F/Q/R rows."""

        execution = self._execute_complete_action_secants()
        closure_hashes = tuple(
            closure.assert_integrity() for closure in self.projection_closures
        )
        first = self.projection_closures[0]
        parent_execution_identity_hash = content_hash(
            {
                "provider_source_hash": directional_secant_provider_source_hash(),
                "design_hash": self.design.design_hash,
                "parent_mask_hash": self.parent_mask.mask_hash,
                "sealed_context_prefix_hash": self.sealed_context_prefix_hash,
                "raw_logit_stream_identity_hash": (
                    first.raw_logit_stream_identity_hash
                ),
                "relaxed_gate_executor_source_hash": (
                    self.relaxed_gate_executor_source_hash
                ),
                "raw_logit_interval_verifier_source_hash": (
                    self.raw_logit_interval_verifier_source_hash
                ),
                "parent_raw_target_logit_center": execution.parent,
                "parent_raw_target_logit_radius": execution.parent_radius,
                "target_readout_index": self.target_readout_index,
            },
            prefix=_PARENT_EXECUTION_PREFIX,
        )
        execution_proof_hash = content_hash(
            {
                "provider_source_hash": directional_secant_provider_source_hash(),
                "design_hash": self.design.design_hash,
                "ladder_hash": self.design.ladder_hash,
                "parent_mask_hash": self.parent_mask.mask_hash,
                "endpoint_action_hashes": self.design.endpoint_action_hashes,
                "endpoint_mask_hashes": execution.endpoint_mask_hashes,
                "packed_direction_hashes": execution.packed_direction_hashes,
                "sealed_context_prefix_hash": self.sealed_context_prefix_hash,
                "sealed_order_start_indices_by_stratum": (
                    self.sealed_order_start_indices_by_stratum
                ),
                "projection_closure_hashes": closure_hashes,
                "native_fiber_axis_receipt_hash": (
                    first.native_fiber_axis_receipt_hash
                ),
                "native_fiber_measure_hash": first.native_fiber_measure_hash,
                "fibered_work_projection_receipt_hash": (
                    first.fibered_work_projection_receipt_hash
                ),
                "literal_jacobian_projection_receipt_hash": (
                    first.literal_jacobian_projection_receipt_hash
                ),
                "same_capture_and_vjp_execution_proof_hash": (
                    first.same_capture_and_vjp_execution_proof_hash
                ),
                "parent_execution_identity_hash": parent_execution_identity_hash,
                "relaxed_gate_executor_source_hash": (
                    self.relaxed_gate_executor_source_hash
                ),
                "raw_logit_interval_verifier_source_hash": (
                    self.raw_logit_interval_verifier_source_hash
                ),
                "raw_logit_interval_proof_hash": execution.interval_proof_hash,
                "doses": DIRECTIONAL_SECANT_DOSES,
                "maximum_packed_lanes": self.maximum_packed_lanes,
                "gate_workspace_forecast_bytes": self.gate_workspace_forecast_bytes,
                "lane_packet_sizes": execution.packet_sizes,
                "parent_raw_logit_hash": content_hash(execution.parent),
                "dosed_raw_logit_hash": content_hash(execution.dosed),
            },
            prefix=_FIBERED_EXECUTION_PREFIX,
        )
        records = tuple(
            close_v1_fibered_directional_record(
                design=self.design,
                native_fiber_axis_receipt=self.native_fiber_axis_receipt,
                projection_closure=closure,
                parent_execution_identity_hash=parent_execution_identity_hash,
                parent_raw_target_logit_center=float(execution.parent[row]),
                parent_raw_target_logit_radius=float(execution.parent_radius[row]),
                dosed_raw_target_logit_center_by_endpoint=execution.dosed[row],
                dosed_raw_target_logit_radius_by_endpoint=execution.dosed_radius[row],
                packed_direction_execution_proof_hash=execution_proof_hash,
                raw_logit_interval_proof_hash=execution.interval_proof_hash,
            )
            for row, closure in enumerate(self.projection_closures)
        )
        if self.observation_bounds is not None and any(
            bool(
                np.any(
                    record.r_upper_observation_by_endpoint
                    > self.observation_bounds.remainder_absolute_upper_by_endpoint_and_dose
                )
            )
            for record in records
        ):
            raise ValueError("fibered directional remainder exceeded its bound")
        return V1PackedFiberedDirectionalSecantExecutionReceipt(
            design_hash=self.design.design_hash,
            ladder_hash=self.design.ladder_hash,
            chart_identity_hash=self.design.chart_identity_hash,
            parent_mask_hash=self.parent_mask.mask_hash,
            route_axis_hash=self.parent_mask.route_axis_hash,
            route_count=self.parent_mask.route_count,
            proposal_generation=self.design.ladder.proposal_generation,
            endpoint_action_hashes=self.design.endpoint_action_hashes,
            endpoint_mask_hashes=execution.endpoint_mask_hashes,
            packed_direction_hashes=execution.packed_direction_hashes,
            sealed_context_prefix_hash=self.sealed_context_prefix_hash,
            context_ids_by_stratum=self.context_axes,
            sealed_order_start_indices_by_stratum=(
                self.sealed_order_start_indices_by_stratum
            ),
            draw_ids=self.draw_ids,
            projection_closure_hashes=closure_hashes,
            native_fiber_axis_receipt_hash=first.native_fiber_axis_receipt_hash,
            native_fiber_measure_hash=first.native_fiber_measure_hash,
            fibered_work_projection_receipt_hash=(
                first.fibered_work_projection_receipt_hash
            ),
            literal_jacobian_projection_receipt_hash=(
                first.literal_jacobian_projection_receipt_hash
            ),
            corrected_evidence_stream_identity_hash=(
                first.corrected_evidence_stream_identity_hash
            ),
            literal_jacobian_stream_identity_hash=(
                first.literal_jacobian_stream_identity_hash
            ),
            raw_logit_stream_identity_hash=first.raw_logit_stream_identity_hash,
            certification_stream_commitment_hash=(
                first.certification_stream_commitment_hash
            ),
            joint_fqr_family_registration_hash=(
                first.joint_fqr_family_registration_hash
            ),
            same_capture_and_vjp_execution_proof_hash=(
                first.same_capture_and_vjp_execution_proof_hash
            ),
            parent_execution_identity_hash=parent_execution_identity_hash,
            provider_source_hash=directional_secant_provider_source_hash(),
            relaxed_gate_executor_source_hash=self.relaxed_gate_executor_source_hash,
            raw_logit_interval_verifier_source_hash=(
                self.raw_logit_interval_verifier_source_hash
            ),
            raw_logit_interval_proof_hash=execution.interval_proof_hash,
            observation_bounds_hash=(
                None
                if self.observation_bounds is None
                else self.observation_bounds.bounds_hash
            ),
            maximum_packed_lanes=self.maximum_packed_lanes,
            gate_workspace_forecast_bytes=self.gate_workspace_forecast_bytes,
            maximum_gate_workspace_bytes=self.maximum_gate_workspace_bytes,
            parent_lane_evaluations=1,
            endpoint_lane_evaluations=2 * len(self.design.endpoint_action_hashes),
            fused_forward_calls=1 + len(execution.packet_sizes),
            lane_packet_sizes=execution.packet_sizes,
            parent_raw_logit_hash=content_hash(execution.parent),
            dosed_raw_logit_hash=content_hash(execution.dosed),
            records=records,
            execution_proof_hash=execution_proof_hash,
        )


__all__ = [
    "DEFAULT_BINARY64_ENCLOSURE_SOURCE_HASH",
    "PACKED_DIRECTIONAL_SECANT_AUTHORITY_SCOPE",
    "PACKED_DIRECTIONAL_SECANT_RECEIPT_SCHEMA",
    "PACKED_FIBERED_DIRECTIONAL_SECANT_AUTHORITY_SCOPE",
    "PACKED_FIBERED_DIRECTIONAL_SECANT_RECEIPT_SCHEMA",
    "NativeTorchV1FiberedDirectionalSecantProvider",
    "NativeTorchV1DirectionalSecantProvider",
    "V1DirectionalSecantObservationBounds",
    "V1FiberedDirectionalSecantObservationBounds",
    "V1PackedFiberedDirectionalSecantExecutionReceipt",
    "V1PackedDirectionalSecantExecutionReceipt",
    "directional_projection_hash",
    "directional_secant_provider_source_hash",
    "fibered_directional_context_slice_axes",
    "sealed_directional_context_prefix_hash",
    "sealed_directional_context_slice_hash",
]
