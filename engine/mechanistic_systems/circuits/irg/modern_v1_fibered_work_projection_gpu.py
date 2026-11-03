"""Deterministic native-fiber corrected-work projection for modern v1.

The legacy frozen-ladder projection collapses every selected route to one
signed scalar before taking magnitude.  That is not the canonical corrected
work functional.  Corrected evidence may cancel only inside one registered
native downstream-gate fiber and inside one context.  Distinct fibers are
measured by counting/total variation, while repeated cuts are probability
averaged::

    F_x(z) = (1 / number_of_cuts) * sum_cut sum_fiber
             abs(sum_selected_route_in_fiber a_j(x)).

This module is a sibling of ``modern_v1_ladder_projection_gpu``.  It wraps an
exact ``V1FrozenActionProjectionRequest`` and an exact source-issued
``V1NativeDestinationFiberAxisReceipt``.  The wrapper admits identity-ordinal
transport only when the native physical and route-ID axes exactly match the
two independently frozen projection axes.

The shell-to-fiber run CSR contains selected route ordinals and one compact
record per nonempty ``(shell, native fiber)`` intersection.  It never stores a
route-length fiber-ID vector.  At execution, each run is reduced through a
fixed padded binary tree.  Unique-destination ``index_copy_`` operations are
the only fiber-state writes; no floating atomic or scatter-add participates.

CPU execution is an explicit reference mode.  Production mode requires an
available CUDA device and never falls back to CPU.  Both paths emit a
conservative outward binary64 enclosure for the exact real fibered functional
of the binary64 estimator rows consumed.  The global signed projection and
singleton triangle are diagnostics only and carry no work, F/Q, oracle, or
mask-commit authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from mechanistic_systems.circuits.irg.modern_v1_ladder_projection_gpu import (
    LADDER_PROJECTION_COORDINATE,
    LADDER_PROJECTION_COTANGENT,
    LADDER_PROJECTION_ORIENTATION,
    LADDER_PROJECTION_UNITS,
    V1DirectionalActionProjectionRequest,
    V1FrozenActionProjectionRequest,
)
from mechanistic_systems.circuits.irg.modern_v1_native_fiber_axis import (
    UNIFORM_CUT_COUNTING_FIBER_MEASURE_EQUATION,
    V1NativeDestinationFiberAxisReceipt,
)
from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.core.axis_identity import (
    ConstructiveOrdinalRouteAxis,
    ordered_string_axis_hash,
)
from mechanistic_systems.measurement.optimized_irg_providers import (
    _RouteChunkedLossOnlyConsumer,
)

try:  # pragma: no cover - availability is environment dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


FIBERED_WORK_PROJECTION_REQUEST_SCHEMA: Final = (
    "partition_free_v1_fibered_corrected_work_projection_request_v1"
)
FIBERED_WORK_RUN_CSR_SCHEMA: Final = (
    "partition_free_v1_shell_to_native_fiber_run_csr_v1"
)
FIBERED_WORK_PACKET_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_fibered_corrected_work_packet_receipt_v1"
)
FIBERED_WORK_DERIVATION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_fibered_corrected_work_derivation_receipt_v1"
)
FIBERED_WORK_RESULT_SCHEMA: Final = (
    "partition_free_v1_fibered_corrected_work_projection_result_v1"
)
FIBERED_WORK_ARITHMETIC_ID: Final = (
    "selected_shell_native_fiber_run_fixed_tree_unique_writer_"
    "cut_counting_tv_float64_outward_v1"
)
FIBERED_WORK_DIAGNOSTIC_ID: Final = (
    "global_signed_and_singleton_triangle_non_authoritative_diagnostics_v1"
)
FIBERED_WORK_OUTWARD_ENCLOSURE_ID: Final = (
    "analytic_gamma_n_with_subnormal_allowance_over_exact_fixed_tree_"
    "operation_depth_upper_bound_v1"
)
FIBERED_WORK_MEMORY_SCOPE: Final = (
    "selected_routes_plus_shell_fiber_runs_plus_context_native_fiber_state_"
    "excluding_scorer_allocator_framework_and_compiler_workspace_v1"
)
FIBERED_WORK_AUTHORITY: Final = (
    "fibered_corrected_work_estimator_projection_and_numerical_enclosure_only_"
    "no_fqr_or_order_or_oracle_or_commit_authority_v1"
)
FIBERED_WORK_DIRECTIONAL_ROLE: Final = "corrected_evidence"
FIBERED_WORK_DIRECTIONAL_ROW_SEMANTICS: Final = (
    "corrected_aumann_shapley_positive_removal_-G_phi_c_v1"
)
FIBERED_WORK_FROZEN_REFERENCE_KIND: Final = "frozen_reference"
FIBERED_WORK_DIRECTIONAL_KIND: Final = "directional_corrected_evidence"

_REQUEST_PREFIX: Final = "partition-free-v1-fibered-work-request-"
_CSR_PREFIX: Final = "partition-free-v1-fibered-work-run-csr-"
_PACKET_PREFIX: Final = "partition-free-v1-fibered-work-packet-"
_RECEIPT_PREFIX: Final = "partition-free-v1-fibered-work-receipt-"
_MATRIX_PREFIX: Final = "partition-free-v1-fibered-work-matrix-"
_WINDOW_PREFIX: Final = "partition-free-v1-fibered-work-route-windows-"
_NEGATIVE_ZERO_U64: Final = np.uint64(1 << 63)
_FLOAT64_BYTES: Final = 8
_UNIT_ROUNDOFF: Final = float(np.finfo(np.float64).eps / 2.0)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _optional_identity(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _identity(value, name=name)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be one exact nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be one exact nonnegative integer")
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be one exact positive integer")
    return result


def _identity_axis(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(_identity(value, name=f"{name} item") for value in tuple(values))
    return result


def _immutable_vector(value: Any, *, dtype: str, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.dtype(dtype), order="C", copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one vector")
    result = np.frombuffer(array.tobytes(order="C"), dtype=np.dtype(dtype))
    result.setflags(write=False)
    return result


def _array_digest(array: np.ndarray) -> dict[str, object]:
    encoded = memoryview(np.ascontiguousarray(array)).cast("B")
    return {
        "dtype": array.dtype.str,
        "shape": tuple(int(value) for value in array.shape),
        "bytes_sha256": hashlib.sha256(encoded).hexdigest(),
        "length": encoded.nbytes,
    }


def _canonical_matrix(
    value: Any,
    *,
    shape: tuple[int, int],
    name: str,
) -> tuple[np.ndarray, str]:
    array = np.array(value, dtype="<f8", order="C", copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be one finite float64 matrix of shape {shape}")
    bits = array.reshape(-1).view(np.uint64)
    bits[bits == _NEGATIVE_ZERO_U64] = np.uint64(0)
    encoded = array.tobytes(order="C")
    result = np.frombuffer(encoded, dtype="<f8").reshape(shape)
    result.setflags(write=False)
    digest = content_hash(
        {
            "shape": shape,
            "dtype": "little_endian_float64",
            "encoded": {
                "bytes_sha256": hashlib.sha256(encoded).hexdigest(),
                "length": len(encoded),
            },
        },
        prefix=_MATRIX_PREFIX,
    )
    return result, digest


def fibered_work_projection_kernel_source_hash() -> str:
    """Hash the exact projection source used by a derivation receipt."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _projection_semantics_payload(
    request: V1FrozenActionProjectionRequest | V1DirectionalActionProjectionRequest,
) -> dict[str, object]:
    """Return the exact role/chart binding admitted by fibered corrected work."""

    if type(request) is V1FrozenActionProjectionRequest:
        request.assert_integrity()
        return {
            "projection_request_kind": FIBERED_WORK_FROZEN_REFERENCE_KIND,
            "projection_role": None,
            "row_semantics": LADDER_PROJECTION_ORIENTATION,
            "chart_identity_hash": None,
            "joint_fqr_family_registration_hash": None,
            "certification_stream_commitment_hash": (
                request.design.certification_stream_commitment_hash
            ),
            "corrected_evidence_stream_identity_hash": (
                request.raw_logit_stream_identity_hash
            ),
            "literal_jacobian_stream_identity_hash": None,
            "same_capture_and_vjp_registration_hash": None,
            "d_c_causal_registration_hash": None,
            "directional_commit_eligible": False,
        }
    if type(request) is not V1DirectionalActionProjectionRequest:
        raise TypeError("unknown projection request type is not fibered-work eligible")
    request.assert_integrity()
    if (
        request.projection_role != FIBERED_WORK_DIRECTIONAL_ROLE
        or request.row_semantics != FIBERED_WORK_DIRECTIONAL_ROW_SEMANTICS
        or request.raw_logit_stream_identity_hash
        != request.corrected_evidence_stream_identity_hash
    ):
        raise PermissionError(
            "fibered corrected work accepts only the exact corrected-evidence "
            "directional role; literal-J remains globally signed"
        )
    return {
        "projection_request_kind": FIBERED_WORK_DIRECTIONAL_KIND,
        "projection_role": request.projection_role,
        "row_semantics": request.row_semantics,
        "chart_identity_hash": request.design.chart_identity_hash,
        "joint_fqr_family_registration_hash": (
            request.design.joint_fqr_family_registration_hash
        ),
        "certification_stream_commitment_hash": (
            request.design.certification_stream_commitment_hash
        ),
        "corrected_evidence_stream_identity_hash": (
            request.corrected_evidence_stream_identity_hash
        ),
        "literal_jacobian_stream_identity_hash": (
            request.literal_jacobian_stream_identity_hash
        ),
        "same_capture_and_vjp_registration_hash": (
            request.same_capture_and_vjp_registration_hash
        ),
        "d_c_causal_registration_hash": request.d_c_causal_registration_hash,
        "directional_commit_eligible": True,
    }


def _projection_semantics_binding_hash(
    request: V1FrozenActionProjectionRequest | V1DirectionalActionProjectionRequest,
) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_fibered_work_projection_semantics_v1",
            "projection_request_hash": request.request_hash,
            **_projection_semantics_payload(request),
        },
        prefix="partition-free-v1-fibered-work-semantics-",
    )


@dataclass(frozen=True, slots=True)
class V1FiberedWorkRunCSR:
    """Selected shell membership grouped into native-fiber runs.

    ``selected_route_indices`` is final-action scale, not route-axis scale.
    ``run_member_starts`` partitions that selected axis.  Every run belongs to
    exactly one shell and one native fiber; a shell can name a fiber at most
    once because physical route order is target-major inside every cut.
    """

    projection_request_hash: str
    ladder_hash: str
    projection_shell_csr_hash: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    route_count: int
    fiber_count: int
    cut_count: int
    endpoint_action_hashes: tuple[str, ...]
    shell_hashes: tuple[str, ...]
    shell_removed_counts: tuple[int, ...]
    selected_route_indices: np.ndarray = field(repr=False, compare=False)
    shell_selected_starts: np.ndarray = field(repr=False, compare=False)
    shell_run_starts: np.ndarray = field(repr=False, compare=False)
    run_member_starts: np.ndarray = field(repr=False, compare=False)
    run_fiber_indices: np.ndarray = field(repr=False, compare=False)
    run_cut_indices: np.ndarray = field(repr=False, compare=False)
    schema: str = FIBERED_WORK_RUN_CSR_SCHEMA
    _csr_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "projection_request_hash",
            "ladder_hash",
            "projection_shell_csr_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        route_count = _positive_int(self.route_count, name="run-CSR route count")
        fiber_count = _positive_int(self.fiber_count, name="run-CSR fiber count")
        cut_count = _positive_int(self.cut_count, name="run-CSR cut count")
        actions = _identity_axis(
            self.endpoint_action_hashes, name="run-CSR endpoint action axis"
        )
        shells = _identity_axis(self.shell_hashes, name="run-CSR shell axis")
        shell_counts = tuple(
            _positive_int(value, name="run-CSR shell removed count")
            for value in self.shell_removed_counts
        )
        if not actions or not (
            len(actions) == len(shells) == len(shell_counts)
        ):
            raise ValueError("run-CSR shell identity axes are not aligned")
        selected_index_dtype = (
            "<u4" if route_count <= np.iinfo(np.uint32).max else "<u8"
        )
        arrays = {
            "selected_route_indices": (
                selected_index_dtype,
                self.selected_route_indices,
            ),
            "shell_selected_starts": ("<u8", self.shell_selected_starts),
            "shell_run_starts": ("<u8", self.shell_run_starts),
            "run_member_starts": ("<u8", self.run_member_starts),
            "run_fiber_indices": ("<u8", self.run_fiber_indices),
            "run_cut_indices": ("<u4", self.run_cut_indices),
        }
        for name, (dtype, value) in arrays.items():
            object.__setattr__(
                self,
                name,
                _immutable_vector(value, dtype=dtype, name=name),
            )
        selected_count = int(self.selected_route_indices.size)
        run_count = int(self.run_fiber_indices.size)
        endpoint_count = len(actions)
        if (
            self.shell_selected_starts.size != endpoint_count + 1
            or self.shell_run_starts.size != endpoint_count + 1
            or self.run_member_starts.size != run_count + 1
            or self.run_cut_indices.size != run_count
            or int(self.shell_selected_starts[0]) != 0
            or int(self.shell_selected_starts[-1]) != selected_count
            or int(self.shell_run_starts[0]) != 0
            or int(self.shell_run_starts[-1]) != run_count
            or int(self.run_member_starts[0]) != 0
            or int(self.run_member_starts[-1]) != selected_count
            or selected_count <= 0
            or run_count <= 0
        ):
            raise ValueError("run-CSR offsets do not close their compact axes")
        for axis in (
            self.shell_selected_starts,
            self.shell_run_starts,
            self.run_member_starts,
        ):
            if np.any(axis[1:] <= axis[:-1]):
                raise ValueError("run-CSR offsets must be strictly increasing")
        if (
            np.any(self.selected_route_indices >= route_count)
            or np.any(self.run_fiber_indices >= fiber_count)
            or np.any(self.run_cut_indices >= cut_count)
        ):
            raise ValueError("run-CSR membership left a registered axis")
        for shell in range(endpoint_count):
            selected_left = int(self.shell_selected_starts[shell])
            selected_right = int(self.shell_selected_starts[shell + 1])
            run_left = int(self.shell_run_starts[shell])
            run_right = int(self.shell_run_starts[shell + 1])
            selected = self.selected_route_indices[selected_left:selected_right]
            fibers = self.run_fiber_indices[run_left:run_right]
            if (
                selected_right - selected_left != shell_counts[shell]
                or (len(selected) > 1 and np.any(selected[1:] <= selected[:-1]))
                or len(fibers) <= 0
                or (len(fibers) > 1 and np.any(fibers[1:] <= fibers[:-1]))
                or int(self.run_member_starts[run_left]) != selected_left
                or int(self.run_member_starts[run_right]) != selected_right
            ):
                raise ValueError("run-CSR shell membership is not canonical")
        if self.schema != FIBERED_WORK_RUN_CSR_SCHEMA:
            raise ValueError("run-CSR schema is invalid")
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "fiber_count", fiber_count)
        object.__setattr__(self, "cut_count", cut_count)
        object.__setattr__(self, "endpoint_action_hashes", actions)
        object.__setattr__(self, "shell_hashes", shells)
        object.__setattr__(self, "shell_removed_counts", shell_counts)
        self.assert_integrity()

    @property
    def endpoint_count(self) -> int:
        return len(self.endpoint_action_hashes)

    @property
    def run_count(self) -> int:
        return int(self.run_fiber_indices.size)

    @property
    def selected_count(self) -> int:
        return int(self.selected_route_indices.size)

    @property
    def host_storage_bytes(self) -> int:
        return sum(
            int(getattr(self, name).nbytes)
            for name in (
                "selected_route_indices",
                "shell_selected_starts",
                "shell_run_starts",
                "run_member_starts",
                "run_fiber_indices",
                "run_cut_indices",
            )
        )

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "projection_request_hash": self.projection_request_hash,
            "ladder_hash": self.ladder_hash,
            "projection_shell_csr_hash": self.projection_shell_csr_hash,
            "native_fiber_axis_receipt_hash": self.native_fiber_axis_receipt_hash,
            "native_fiber_measure_hash": self.native_fiber_measure_hash,
            "route_count": self.route_count,
            "fiber_count": self.fiber_count,
            "cut_count": self.cut_count,
            "endpoint_action_hashes": self.endpoint_action_hashes,
            "shell_hashes": self.shell_hashes,
            "shell_removed_counts": self.shell_removed_counts,
            "arrays": {
                name: _array_digest(getattr(self, name))
                for name in (
                    "selected_route_indices",
                    "shell_selected_starts",
                    "shell_run_starts",
                    "run_member_starts",
                    "run_fiber_indices",
                    "run_cut_indices",
                )
            },
            "route_scale_fiber_id_array_materialized": False,
        }

    @property
    def csr_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._hash_payload(), prefix=_CSR_PREFIX)
        if self._csr_hash is not None and observed != self._csr_hash:
            raise ValueError("fibered-work run CSR was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("fibered-work run CSR hash does not match")
        object.__setattr__(self, "_csr_hash", observed)
        return observed

    def assert_against_axis(
        self,
        axis: V1NativeDestinationFiberAxisReceipt,
    ) -> None:
        """Replay every compact run against the source-issued cut arithmetic."""

        if type(axis) is not V1NativeDestinationFiberAxisReceipt:
            raise TypeError("run-CSR replay requires the exact native-fiber receipt")
        axis.assert_integrity()
        self.assert_integrity()
        if (
            self.native_fiber_axis_receipt_hash != axis.receipt_hash
            or self.native_fiber_measure_hash != axis.measure_hash
            or self.route_count != axis.route_count
            or self.fiber_count != axis.fiber_count
            or self.cut_count != axis.cut_count
        ):
            raise ValueError("run-CSR names another native-fiber axis")
        for shell in range(self.endpoint_count):
            run_left = int(self.shell_run_starts[shell])
            run_right = int(self.shell_run_starts[shell + 1])
            for run in range(run_left, run_right):
                member_left = int(self.run_member_starts[run])
                member_right = int(self.run_member_starts[run + 1])
                routes = self.selected_route_indices[member_left:member_right]
                fiber = int(self.run_fiber_indices[run])
                cut_index = int(self.run_cut_indices[run])
                cut = axis.cut_for_fiber(fiber)
                fiber_local = fiber - cut.fiber_offset
                span_left, span_right = cut.route_span_for_fiber(fiber_local)
                if (
                    cut.cut_index != cut_index
                    or len(routes) <= 0
                    or int(routes[0]) < span_left
                    or int(routes[-1]) >= span_right
                ):
                    raise ValueError("run-CSR run crosses a native destination fiber")


def _selected_shell_indices(
    request: V1FrozenActionProjectionRequest,
) -> tuple[np.ndarray, ...]:
    ladder = request.design.ladder
    index_dtype = np.dtype(
        "<u4" if request.route_count <= np.iinfo(np.uint32).max else "<u8"
    )
    shells: list[np.ndarray] = []
    for action, expected_count in zip(
        ladder.actions,
        request.shell_removed_counts,
        strict=True,
    ):
        parts = tuple(
            np.asarray(chunk).astype(index_dtype, copy=False)
            for chunk in ladder.iter_shell_index_chunks(
                endpoint_index=action.endpoint_index
            )
            if len(chunk)
        )
        if not parts:
            raise ValueError("fibered-work shell is empty")
        selected = np.concatenate(parts).astype(index_dtype, copy=False)
        if (
            int(selected.size) != expected_count
            or int(selected[-1]) >= request.route_count
            or (len(selected) > 1 and np.any(selected[1:] <= selected[:-1]))
        ):
            raise ValueError("fibered-work shell membership is not canonical")
        selected.setflags(write=False)
        shells.append(selected)
    return tuple(shells)


def build_v1_fibered_work_run_csr(
    projection_request: V1FrozenActionProjectionRequest | V1DirectionalActionProjectionRequest,
    native_fiber_axis: V1NativeDestinationFiberAxisReceipt,
) -> V1FiberedWorkRunCSR:
    """Lower selected shell routes into compact native-fiber runs."""

    if type(projection_request) not in {
        V1FrozenActionProjectionRequest,
        V1DirectionalActionProjectionRequest,
    }:
        raise TypeError("fibered-work lowering requires the exact projection request")
    _projection_semantics_payload(projection_request)
    if type(native_fiber_axis) is not V1NativeDestinationFiberAxisReceipt:
        raise TypeError("fibered-work lowering requires the exact native-fiber receipt")
    projection_request.assert_integrity()
    native_fiber_axis.assert_integrity()
    if projection_request.route_count != native_fiber_axis.route_count:
        raise ValueError("projection and native-fiber route counts differ")
    if native_fiber_axis.route_axis_hash != projection_request.physical_mask_axis_hash:
        raise ValueError("native physical route axis differs from packed mask axis")
    if (
        native_fiber_axis.route_id_axis_hash
        != projection_request.ordered_score_route_axis_hash
    ):
        raise ValueError("native route-ID axis differs from ordered score-route axis")

    shells = _selected_shell_indices(projection_request)
    shell_selected_starts = [0]
    shell_run_starts = [0]
    run_member_starts = [0]
    run_fibers: list[int] = []
    run_cuts: list[int] = []
    flattened: list[np.ndarray] = []
    selected_cursor = 0
    for selected in shells:
        flattened.append(selected)
        for cut in native_fiber_axis.cuts:
            left = int(np.searchsorted(selected, cut.route_offset, side="left"))
            right = int(np.searchsorted(selected, cut.route_stop, side="left"))
            if left == right:
                continue
            # O(fibers), not O(selected routes): locate each native-fiber
            # boundary directly in the sorted selected route axis.  This
            # avoids even a temporary selected-scale fiber-ID vector.
            route_boundaries = (
                np.arange(cut.fiber_count + 1, dtype=np.uint64)
                * np.uint64(cut.routes_per_fiber)
                + np.uint64(cut.route_offset)
            )
            positions = np.searchsorted(selected, route_boundaries, side="left")
            nonempty = np.flatnonzero(positions[1:] > positions[:-1])
            for local_fiber_raw in nonempty:
                local_fiber = int(local_fiber_raw)
                run_fibers.append(cut.fiber_offset + local_fiber)
                run_cuts.append(cut.cut_index)
                run_member_starts.append(
                    selected_cursor + int(positions[local_fiber + 1])
                )
        selected_cursor += int(selected.size)
        shell_selected_starts.append(selected_cursor)
        shell_run_starts.append(len(run_fibers))
    selected_index_dtype = np.dtype(
        "<u4"
        if projection_request.route_count <= np.iinfo(np.uint32).max
        else "<u8"
    )
    selected_routes = np.concatenate(flattened).astype(
        selected_index_dtype,
        copy=False,
    )
    result = V1FiberedWorkRunCSR(
        projection_request_hash=projection_request.request_hash,
        ladder_hash=projection_request.ladder_hash,
        projection_shell_csr_hash=projection_request.csr.csr_hash,
        native_fiber_axis_receipt_hash=native_fiber_axis.receipt_hash,
        native_fiber_measure_hash=native_fiber_axis.measure_hash,
        route_count=projection_request.route_count,
        fiber_count=native_fiber_axis.fiber_count,
        cut_count=native_fiber_axis.cut_count,
        endpoint_action_hashes=projection_request.endpoint_action_hashes,
        shell_hashes=projection_request.shell_hashes,
        shell_removed_counts=projection_request.shell_removed_counts,
        selected_route_indices=selected_routes,
        shell_selected_starts=np.asarray(shell_selected_starts, dtype="<u8"),
        shell_run_starts=np.asarray(shell_run_starts, dtype="<u8"),
        run_member_starts=np.asarray(run_member_starts, dtype="<u8"),
        run_fiber_indices=np.asarray(run_fibers, dtype="<u8"),
        run_cut_indices=np.asarray(run_cuts, dtype="<u4"),
    )
    result.assert_against_axis(native_fiber_axis)
    return result


@dataclass(frozen=True, slots=True)
class V1FiberedWorkProjectionRequest:
    """Pre-outcome binding of a frozen ladder to its native work measure."""

    projection_request: (
        V1FrozenActionProjectionRequest | V1DirectionalActionProjectionRequest
    ) = field(
        repr=False,
        compare=False,
    )
    native_fiber_axis: V1NativeDestinationFiberAxisReceipt = field(
        repr=False,
        compare=False,
    )
    maximum_temporary_device_bytes: int
    maximum_registered_device_bytes: int
    schema: str = FIBERED_WORK_PROJECTION_REQUEST_SCHEMA
    _run_csr: V1FiberedWorkRunCSR = field(init=False, repr=False, compare=False)
    _request_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.projection_request) not in {
            V1FrozenActionProjectionRequest,
            V1DirectionalActionProjectionRequest,
        }:
            raise TypeError("fibered-work wrapper requires an exact projection request")
        if type(self.native_fiber_axis) is not V1NativeDestinationFiberAxisReceipt:
            raise TypeError("fibered-work wrapper requires an exact native-fiber receipt")
        self.projection_request.assert_integrity()
        semantics = _projection_semantics_payload(self.projection_request)
        self.native_fiber_axis.assert_integrity()
        if (
            self.native_fiber_axis.route_axis_hash
            != self.projection_request.physical_mask_axis_hash
        ):
            raise ValueError("native route-axis hash differs from physical mask axis")
        if (
            self.native_fiber_axis.route_id_axis_hash
            != self.projection_request.ordered_score_route_axis_hash
        ):
            raise ValueError("native route-ID hash differs from ordered score-route axis")
        if self.schema != FIBERED_WORK_PROJECTION_REQUEST_SCHEMA:
            raise ValueError("fibered-work request schema is invalid")
        object.__setattr__(
            self,
            "maximum_temporary_device_bytes",
            _positive_int(
                self.maximum_temporary_device_bytes,
                name="maximum temporary device bytes",
            ),
        )
        object.__setattr__(
            self,
            "maximum_registered_device_bytes",
            _positive_int(
                self.maximum_registered_device_bytes,
                name="maximum registered device bytes",
            ),
        )
        if (
            self.maximum_temporary_device_bytes
            > self.maximum_registered_device_bytes
        ):
            raise ValueError(
                "temporary workspace cap cannot exceed total registered-device cap"
            )
        csr = build_v1_fibered_work_run_csr(
            self.projection_request,
            self.native_fiber_axis,
        )
        object.__setattr__(self, "_run_csr", csr)
        if semantics["projection_request_kind"] not in {
            FIBERED_WORK_FROZEN_REFERENCE_KIND,
            FIBERED_WORK_DIRECTIONAL_KIND,
        }:
            raise AssertionError("unreachable fibered-work projection semantics")
        self.assert_integrity()

    @property
    def run_csr(self) -> V1FiberedWorkRunCSR:
        return self._run_csr

    @property
    def request_hash(self) -> str:
        return self.assert_integrity()

    @property
    def route_count(self) -> int:
        return self.projection_request.route_count

    @property
    def context_count(self) -> int:
        return self.projection_request.context_count

    @property
    def endpoint_count(self) -> int:
        return self.projection_request.endpoint_count

    @property
    def physical_mask_axis_hash(self) -> str:
        return self.projection_request.physical_mask_axis_hash

    @property
    def ordered_score_route_axis_hash(self) -> str:
        return self.projection_request.ordered_score_route_axis_hash

    @property
    def native_fiber_axis_receipt_hash(self) -> str:
        return self.native_fiber_axis.receipt_hash

    @property
    def native_fiber_measure_hash(self) -> str:
        return self.native_fiber_axis.measure_hash

    @property
    def projection_semantics(self) -> Mapping[str, object]:
        return MappingProxyType(_projection_semantics_payload(self.projection_request))

    @property
    def projection_semantics_binding_hash(self) -> str:
        return _projection_semantics_binding_hash(self.projection_request)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "projection_request_hash": self.projection_request.request_hash,
            "native_fiber_axis_receipt_hash": self.native_fiber_axis.receipt_hash,
            "native_fiber_measure_hash": self.native_fiber_axis.measure_hash,
            "run_csr_hash": self.run_csr.csr_hash,
            "projection_semantics_binding_hash": (
                self.projection_semantics_binding_hash
            ),
            "projection_semantics": dict(self.projection_semantics),
            "physical_mask_axis_hash": self.physical_mask_axis_hash,
            "ordered_score_route_axis_hash": self.ordered_score_route_axis_hash,
            "route_count": self.route_count,
            "fiber_count": self.native_fiber_axis.fiber_count,
            "cut_count": self.native_fiber_axis.cut_count,
            "measure_equation": self.native_fiber_axis.measure_equation,
            "maximum_temporary_device_bytes": self.maximum_temporary_device_bytes,
            "maximum_registered_device_bytes": (
                self.maximum_registered_device_bytes
            ),
            "projection_kernel_source_hash": fibered_work_projection_kernel_source_hash(),
            "projection_arithmetic_id": FIBERED_WORK_ARITHMETIC_ID,
            "coordinate": LADDER_PROJECTION_COORDINATE,
            "cotangent": LADDER_PROJECTION_COTANGENT,
            "orientation": LADDER_PROJECTION_ORIENTATION,
            "units": LADDER_PROJECTION_UNITS,
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.projection_request.assert_integrity()
        _projection_semantics_payload(self.projection_request)
        self.native_fiber_axis.assert_integrity()
        self.run_csr.assert_integrity()
        self.run_csr.assert_against_axis(self.native_fiber_axis)
        if (
            self.native_fiber_axis.route_axis_hash != self.physical_mask_axis_hash
            or self.native_fiber_axis.route_id_axis_hash
            != self.ordered_score_route_axis_hash
        ):
            raise ValueError("fibered-work ordinal transport became stale")
        observed = content_hash(self._hash_payload(), prefix=_REQUEST_PREFIX)
        if self._request_hash is not None and observed != self._request_hash:
            raise ValueError("fibered-work request was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("fibered-work request hash does not match")
        object.__setattr__(self, "_request_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FiberedWorkProjectionPacketReceipt:
    request_hash: str
    projection_request_kind: str
    projection_semantics_binding_hash: str
    packet_index: int
    context_start: int
    context_stop: int
    context_ids: tuple[str, ...]
    draw_ids: tuple[str, ...]
    stratum_ids: tuple[str, ...]
    route_windows: tuple[tuple[int, int], ...]
    route_window_ledger_hash: str
    run_tile_schedule_hash: str
    fibered_center_matrix_hash: str
    global_signed_diagnostic_matrix_hash: str
    singleton_triangle_diagnostic_matrix_hash: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    run_csr_hash: str
    numerical_backend: str
    production_cuda_execution: bool
    maximum_temporary_device_bytes: int
    maximum_registered_device_bytes: int
    preoutcome_peak_registered_device_bytes_upper: int
    maximum_run_tile_member_count: int
    peak_registered_device_bytes: int
    host_materialization_bytes: int
    device_to_host_bytes: int
    device_identity: str
    floating_atomics_used: bool = False
    schema: str = FIBERED_WORK_PACKET_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "request_hash",
            "projection_request_kind",
            "projection_semantics_binding_hash",
            "route_window_ledger_hash",
            "run_tile_schedule_hash",
            "fibered_center_matrix_hash",
            "global_signed_diagnostic_matrix_hash",
            "singleton_triangle_diagnostic_matrix_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "run_csr_hash",
            "numerical_backend",
            "device_identity",
        ):
            _identity(getattr(self, name), name=name)
        contexts = _identity_axis(self.context_ids, name="packet context axis")
        draws = _identity_axis(self.draw_ids, name="packet draw axis")
        strata = _identity_axis(self.stratum_ids, name="packet stratum axis")
        if not contexts or not (len(contexts) == len(draws) == len(strata)):
            raise ValueError("fibered-work packet ledgers are not aligned")
        if self.projection_request_kind not in {
            FIBERED_WORK_FROZEN_REFERENCE_KIND,
            FIBERED_WORK_DIRECTIONAL_KIND,
        }:
            raise ValueError("fibered-work packet request kind is invalid")
        index = _nonnegative_int(self.packet_index, name="packet index")
        start = _nonnegative_int(self.context_start, name="packet context start")
        stop = _positive_int(self.context_stop, name="packet context stop")
        if stop - start != len(contexts):
            raise ValueError("fibered-work packet context interval is inconsistent")
        windows = tuple(
            (
                _nonnegative_int(left, name="route-window start"),
                _positive_int(right, name="route-window stop"),
            )
            for left, right in self.route_windows
        )
        if not windows or windows[0][0] != 0 or any(
            left >= right
            or (position and left != windows[position - 1][1])
            for position, (left, right) in enumerate(windows)
        ):
            raise ValueError("fibered-work packet windows are not contiguous")
        if self.floating_atomics_used is not False:
            raise ValueError("fibered-work projection forbids floating atomics")
        if type(self.production_cuda_execution) is not bool:
            raise ValueError("production CUDA execution flag must be exact bool")
        if self.production_cuda_execution and not self.numerical_backend.startswith(
            "torch_cuda_"
        ):
            raise ValueError("production fibered-work receipt must name CUDA")
        for name in (
            "peak_registered_device_bytes",
            "host_materialization_bytes",
            "device_to_host_bytes",
            "maximum_run_tile_member_count",
            "preoutcome_peak_registered_device_bytes_upper",
        ):
            _nonnegative_int(getattr(self, name), name=name)
        _positive_int(
            self.maximum_temporary_device_bytes,
            name="maximum temporary device bytes",
        )
        _positive_int(
            self.maximum_registered_device_bytes,
            name="maximum registered device bytes",
        )
        if not (
            self.maximum_temporary_device_bytes
            <= self.maximum_registered_device_bytes
            and self.preoutcome_peak_registered_device_bytes_upper
            <= self.maximum_registered_device_bytes
            and self.peak_registered_device_bytes
            <= self.preoutcome_peak_registered_device_bytes_upper
        ):
            raise ValueError(
                "fibered-work packet exceeded its registered device-memory forecast"
            )
        if self.schema != FIBERED_WORK_PACKET_RECEIPT_SCHEMA:
            raise ValueError("fibered-work packet schema is invalid")
        object.__setattr__(self, "packet_index", index)
        object.__setattr__(self, "context_start", start)
        object.__setattr__(self, "context_stop", stop)
        object.__setattr__(self, "context_ids", contexts)
        object.__setattr__(self, "draw_ids", draws)
        object.__setattr__(self, "stratum_ids", strata)
        object.__setattr__(self, "route_windows", windows)
        self.assert_integrity()

    def _hash_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "request_hash",
                "projection_request_kind",
                "projection_semantics_binding_hash",
                "packet_index",
                "context_start",
                "context_stop",
                "context_ids",
                "draw_ids",
                "stratum_ids",
                "route_windows",
                "route_window_ledger_hash",
                "run_tile_schedule_hash",
                "fibered_center_matrix_hash",
                "global_signed_diagnostic_matrix_hash",
                "singleton_triangle_diagnostic_matrix_hash",
                "native_fiber_axis_receipt_hash",
                "native_fiber_measure_hash",
                "run_csr_hash",
                "numerical_backend",
                "production_cuda_execution",
                "maximum_temporary_device_bytes",
                "maximum_registered_device_bytes",
                "preoutcome_peak_registered_device_bytes_upper",
                "maximum_run_tile_member_count",
                "peak_registered_device_bytes",
                "host_materialization_bytes",
                "device_to_host_bytes",
                "device_identity",
                "floating_atomics_used",
            )
        } | {
            "projection_arithmetic_id": FIBERED_WORK_ARITHMETIC_ID,
            "diagnostic_id": FIBERED_WORK_DIAGNOSTIC_ID,
            "outward_enclosure_id": FIBERED_WORK_OUTWARD_ENCLOSURE_ID,
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._hash_payload(), prefix=_PACKET_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("fibered-work packet receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("fibered-work packet receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FiberedWorkProjectionDerivationReceipt:
    request_hash: str
    projection_request_hash: str
    projection_request_kind: str
    projection_semantics_binding_hash: str
    projection_role: str | None
    row_semantics: str
    chart_identity_hash: str | None
    joint_fqr_family_registration_hash: str | None
    certification_stream_commitment_hash: str
    corrected_evidence_stream_identity_hash: str
    literal_jacobian_stream_identity_hash: str | None
    same_capture_and_vjp_registration_hash: str | None
    d_c_causal_registration_hash: str | None
    directional_commit_eligible: bool
    design_hash: str
    ladder_hash: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    run_csr_hash: str
    physical_mask_axis_hash: str
    ordered_score_route_axis_hash: str
    context_ids: tuple[str, ...]
    draw_ids: tuple[str, ...]
    stratum_ids: tuple[str, ...]
    endpoint_action_hashes: tuple[str, ...]
    packet_receipt_hashes: tuple[str, ...]
    route_window_ledger_hashes: tuple[str, ...]
    run_tile_schedule_hashes: tuple[str, ...]
    fibered_center_matrix_hash: str
    global_signed_diagnostic_matrix_hash: str
    singleton_triangle_diagnostic_matrix_hash: str
    singleton_triangle_upper_matrix_hash: str
    numerical_radius_matrix_hash: str
    numerical_lower_matrix_hash: str
    numerical_upper_matrix_hash: str
    projection_kernel_source_hash: str
    numerical_backend: str
    production_cuda_execution: bool
    maximum_temporary_device_bytes: int
    maximum_registered_device_bytes: int
    preoutcome_peak_registered_device_bytes_upper: int
    peak_registered_device_bytes: int
    total_host_materialization_bytes: int
    exact_device_to_host_bytes: int
    packet_count: int
    route_chunk_update_count: int
    estimator_rows_exactly_projected: bool = True
    real_arithmetic_exactness_claimed: bool = False
    outward_numerical_enclosure_provided: bool = True
    within_native_fiber_cancellation_only: bool = True
    cross_context_cancellation_used: bool = False
    cross_native_fiber_work_cancellation_used: bool = False
    floating_atomics_used: bool = False
    scalar_fibered_work_derivation_authority: bool = True
    fq_certification_authority: bool = False
    score_order_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    schema: str = FIBERED_WORK_DERIVATION_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "request_hash",
            "projection_request_hash",
            "projection_request_kind",
            "projection_semantics_binding_hash",
            "design_hash",
            "ladder_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "run_csr_hash",
            "physical_mask_axis_hash",
            "ordered_score_route_axis_hash",
            "fibered_center_matrix_hash",
            "global_signed_diagnostic_matrix_hash",
            "singleton_triangle_diagnostic_matrix_hash",
            "singleton_triangle_upper_matrix_hash",
            "numerical_radius_matrix_hash",
            "numerical_lower_matrix_hash",
            "numerical_upper_matrix_hash",
            "projection_kernel_source_hash",
            "numerical_backend",
            "row_semantics",
            "certification_stream_commitment_hash",
            "corrected_evidence_stream_identity_hash",
        ):
            _identity(getattr(self, name), name=name)
        for name in (
            "projection_role",
            "chart_identity_hash",
            "joint_fqr_family_registration_hash",
            "literal_jacobian_stream_identity_hash",
            "same_capture_and_vjp_registration_hash",
            "d_c_causal_registration_hash",
        ):
            _optional_identity(getattr(self, name), name=name)
        if self.projection_request_kind == FIBERED_WORK_DIRECTIONAL_KIND:
            if not (
                self.projection_role == FIBERED_WORK_DIRECTIONAL_ROLE
                and self.row_semantics == FIBERED_WORK_DIRECTIONAL_ROW_SEMANTICS
                and self.chart_identity_hash is not None
                and self.joint_fqr_family_registration_hash is not None
                and self.literal_jacobian_stream_identity_hash is not None
                and self.same_capture_and_vjp_registration_hash is not None
                and self.d_c_causal_registration_hash is not None
                and self.directional_commit_eligible is True
            ):
                raise ValueError(
                    "directional fibered-work receipt lacks corrected-evidence "
                    "chart/same-VJP ownership"
                )
        elif self.projection_request_kind == FIBERED_WORK_FROZEN_REFERENCE_KIND:
            if not (
                self.projection_role is None
                and self.chart_identity_hash is None
                and self.joint_fqr_family_registration_hash is None
                and self.literal_jacobian_stream_identity_hash is None
                and self.same_capture_and_vjp_registration_hash is None
                and self.d_c_causal_registration_hash is None
                and self.directional_commit_eligible is False
            ):
                raise ValueError("frozen fibered-work reference claimed directional authority")
        else:
            raise ValueError("fibered-work derivation request kind is invalid")
        contexts = _identity_axis(self.context_ids, name="derivation context axis")
        draws = _identity_axis(self.draw_ids, name="derivation draw axis")
        strata = _identity_axis(self.stratum_ids, name="derivation stratum axis")
        actions = _identity_axis(
            self.endpoint_action_hashes, name="derivation endpoint action axis"
        )
        packets = _identity_axis(
            self.packet_receipt_hashes, name="derivation packet receipt axis"
        )
        windows = _identity_axis(
            self.route_window_ledger_hashes, name="derivation window ledger axis"
        )
        tile_schedules = _identity_axis(
            self.run_tile_schedule_hashes,
            name="derivation run-tile schedule axis",
        )
        if not contexts or not (len(contexts) == len(draws) == len(strata)):
            raise ValueError("fibered-work derivation ledgers are not aligned")
        if (
            not actions
            or not packets
            or not (len(packets) == len(windows) == len(tile_schedules))
        ):
            raise ValueError("fibered-work derivation axes are incomplete")
        if len(packets) != _positive_int(self.packet_count, name="packet count"):
            raise ValueError("fibered-work packet count differs from receipt axis")
        _positive_int(self.route_chunk_update_count, name="route chunk update count")
        if type(self.production_cuda_execution) is not bool:
            raise ValueError("production CUDA execution flag must be exact bool")
        if self.production_cuda_execution and not self.numerical_backend.startswith(
            "torch_cuda_"
        ):
            raise ValueError("production fibered-work derivation must name CUDA")
        for name in (
            "peak_registered_device_bytes",
            "total_host_materialization_bytes",
            "exact_device_to_host_bytes",
            "preoutcome_peak_registered_device_bytes_upper",
        ):
            _nonnegative_int(getattr(self, name), name=name)
        _positive_int(
            self.maximum_temporary_device_bytes,
            name="maximum temporary device bytes",
        )
        _positive_int(
            self.maximum_registered_device_bytes,
            name="maximum registered device bytes",
        )
        if not (
            self.maximum_temporary_device_bytes
            <= self.maximum_registered_device_bytes
            and self.preoutcome_peak_registered_device_bytes_upper
            <= self.maximum_registered_device_bytes
            and self.peak_registered_device_bytes
            <= self.preoutcome_peak_registered_device_bytes_upper
        ):
            raise ValueError(
                "fibered-work derivation exceeded its registered device-memory forecast"
            )
        if not (
            self.estimator_rows_exactly_projected is True
            and self.real_arithmetic_exactness_claimed is False
            and self.outward_numerical_enclosure_provided is True
            and self.within_native_fiber_cancellation_only is True
            and self.cross_context_cancellation_used is False
            and self.cross_native_fiber_work_cancellation_used is False
            and self.floating_atomics_used is False
            and self.scalar_fibered_work_derivation_authority is True
        ) or any(
            (
                self.fq_certification_authority,
                self.score_order_authority,
                self.bce_oracle_authority,
                self.mask_commit_authority,
            )
        ):
            raise ValueError("fibered-work derivation exceeded its scalar authority")
        if self.schema != FIBERED_WORK_DERIVATION_RECEIPT_SCHEMA:
            raise ValueError("fibered-work derivation schema is invalid")
        object.__setattr__(self, "context_ids", contexts)
        object.__setattr__(self, "draw_ids", draws)
        object.__setattr__(self, "stratum_ids", strata)
        object.__setattr__(self, "endpoint_action_hashes", actions)
        object.__setattr__(self, "packet_receipt_hashes", packets)
        object.__setattr__(self, "route_window_ledger_hashes", windows)
        object.__setattr__(self, "run_tile_schedule_hashes", tile_schedules)
        self.assert_integrity()

    def _hash_payload(self) -> dict[str, object]:
        names = (
            "schema",
            "request_hash",
            "projection_request_hash",
            "projection_request_kind",
            "projection_semantics_binding_hash",
            "projection_role",
            "row_semantics",
            "chart_identity_hash",
            "joint_fqr_family_registration_hash",
            "certification_stream_commitment_hash",
            "corrected_evidence_stream_identity_hash",
            "literal_jacobian_stream_identity_hash",
            "same_capture_and_vjp_registration_hash",
            "d_c_causal_registration_hash",
            "directional_commit_eligible",
            "design_hash",
            "ladder_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "run_csr_hash",
            "physical_mask_axis_hash",
            "ordered_score_route_axis_hash",
            "context_ids",
            "draw_ids",
            "stratum_ids",
            "endpoint_action_hashes",
            "packet_receipt_hashes",
            "route_window_ledger_hashes",
            "run_tile_schedule_hashes",
            "fibered_center_matrix_hash",
            "global_signed_diagnostic_matrix_hash",
            "singleton_triangle_diagnostic_matrix_hash",
            "singleton_triangle_upper_matrix_hash",
            "numerical_radius_matrix_hash",
            "numerical_lower_matrix_hash",
            "numerical_upper_matrix_hash",
            "projection_kernel_source_hash",
            "numerical_backend",
            "production_cuda_execution",
            "maximum_temporary_device_bytes",
            "maximum_registered_device_bytes",
            "preoutcome_peak_registered_device_bytes_upper",
            "peak_registered_device_bytes",
            "total_host_materialization_bytes",
            "exact_device_to_host_bytes",
            "packet_count",
            "route_chunk_update_count",
            "estimator_rows_exactly_projected",
            "real_arithmetic_exactness_claimed",
            "outward_numerical_enclosure_provided",
            "within_native_fiber_cancellation_only",
            "cross_context_cancellation_used",
            "cross_native_fiber_work_cancellation_used",
            "floating_atomics_used",
            "scalar_fibered_work_derivation_authority",
            "fq_certification_authority",
            "score_order_authority",
            "bce_oracle_authority",
            "mask_commit_authority",
        )
        return {name: getattr(self, name) for name in names} | {
            "measure_equation": UNIFORM_CUT_COUNTING_FIBER_MEASURE_EQUATION,
            "projection_arithmetic_id": FIBERED_WORK_ARITHMETIC_ID,
            "diagnostic_id": FIBERED_WORK_DIAGNOSTIC_ID,
            "outward_enclosure_id": FIBERED_WORK_OUTWARD_ENCLOSURE_ID,
            "memory_scope": FIBERED_WORK_MEMORY_SCOPE,
            "authority": FIBERED_WORK_AUTHORITY,
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._hash_payload(), prefix=_RECEIPT_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("fibered-work derivation receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("fibered-work derivation receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FiberedWorkProjectionResult:
    receipt: V1FiberedWorkProjectionDerivationReceipt
    packet_receipts: tuple[V1FiberedWorkProjectionPacketReceipt, ...]
    fibered_center_matrix: np.ndarray = field(repr=False, compare=False)
    global_signed_diagnostic_matrix: np.ndarray = field(repr=False, compare=False)
    singleton_triangle_diagnostic_matrix: np.ndarray = field(
        repr=False,
        compare=False,
    )
    singleton_triangle_upper_matrix: np.ndarray = field(repr=False, compare=False)
    numerical_radius_matrix: np.ndarray = field(repr=False, compare=False)
    numerical_lower_matrix: np.ndarray = field(repr=False, compare=False)
    numerical_upper_matrix: np.ndarray = field(repr=False, compare=False)
    schema: str = FIBERED_WORK_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if type(self.receipt) is not V1FiberedWorkProjectionDerivationReceipt:
            raise TypeError("fibered-work result requires its exact typed receipt")
        self.receipt.assert_integrity()
        packets = tuple(self.packet_receipts)
        if any(
            type(item) is not V1FiberedWorkProjectionPacketReceipt
            for item in packets
        ) or tuple(
            item.receipt_hash for item in packets
        ) != self.receipt.packet_receipt_hashes:
            raise ValueError("fibered-work packet receipts differ from derivation")
        if any(
            item.request_hash != self.receipt.request_hash
            or item.projection_request_kind
            != self.receipt.projection_request_kind
            or item.projection_semantics_binding_hash
            != self.receipt.projection_semantics_binding_hash
            or item.native_fiber_axis_receipt_hash
            != self.receipt.native_fiber_axis_receipt_hash
            or item.native_fiber_measure_hash
            != self.receipt.native_fiber_measure_hash
            or item.run_csr_hash != self.receipt.run_csr_hash
            or item.production_cuda_execution
            != self.receipt.production_cuda_execution
            or item.maximum_temporary_device_bytes
            != self.receipt.maximum_temporary_device_bytes
            or item.maximum_registered_device_bytes
            != self.receipt.maximum_registered_device_bytes
            for item in packets
        ):
            raise ValueError("fibered-work packet identity differs from derivation")
        shape = (
            len(self.receipt.context_ids),
            len(self.receipt.endpoint_action_hashes),
        )
        specifications = (
            ("fibered_center_matrix", self.receipt.fibered_center_matrix_hash),
            (
                "global_signed_diagnostic_matrix",
                self.receipt.global_signed_diagnostic_matrix_hash,
            ),
            (
                "singleton_triangle_diagnostic_matrix",
                self.receipt.singleton_triangle_diagnostic_matrix_hash,
            ),
            (
                "singleton_triangle_upper_matrix",
                self.receipt.singleton_triangle_upper_matrix_hash,
            ),
            ("numerical_radius_matrix", self.receipt.numerical_radius_matrix_hash),
            ("numerical_lower_matrix", self.receipt.numerical_lower_matrix_hash),
            ("numerical_upper_matrix", self.receipt.numerical_upper_matrix_hash),
        )
        for name, expected_hash in specifications:
            canonical, observed_hash = _canonical_matrix(
                getattr(self, name), shape=shape, name=name
            )
            if observed_hash != expected_hash:
                raise ValueError(f"{name} differs from its derivation hash")
            object.__setattr__(self, name, canonical)
        if np.any(self.fibered_center_matrix < 0.0):
            raise ValueError("fibered corrected work cannot be negative")
        if np.any(self.singleton_triangle_diagnostic_matrix < 0.0):
            raise ValueError("singleton triangle diagnostic cannot be negative")
        if np.any(
            self.singleton_triangle_upper_matrix
            < self.singleton_triangle_diagnostic_matrix
        ):
            raise ValueError("singleton triangle upper bound is understated")
        if np.any(self.numerical_lower_matrix < 0.0) or np.any(
            self.numerical_lower_matrix > self.fibered_center_matrix
        ) or np.any(self.numerical_upper_matrix < self.fibered_center_matrix):
            raise ValueError("fibered-work numerical enclosure is not outward")
        if self.schema != FIBERED_WORK_RESULT_SCHEMA:
            raise ValueError("fibered-work result schema is invalid")
        object.__setattr__(self, "packet_receipts", packets)


@dataclass(frozen=True, slots=True)
class _RunBucket:
    width: int
    run_local_indices: np.ndarray


@dataclass(frozen=True, slots=True)
class _ClosedPacket:
    receipt: V1FiberedWorkProjectionPacketReceipt
    center: np.ndarray = field(repr=False, compare=False)
    global_signed: np.ndarray = field(repr=False, compare=False)
    singleton_triangle: np.ndarray = field(repr=False, compare=False)


def _build_run_buckets(
    csr: V1FiberedWorkRunCSR,
) -> tuple[tuple[_RunBucket, ...], ...]:
    schedules: list[tuple[_RunBucket, ...]] = []
    for shell in range(csr.endpoint_count):
        run_left = int(csr.shell_run_starts[shell])
        run_right = int(csr.shell_run_starts[shell + 1])
        buckets: dict[int, list[int]] = {}
        for run in range(run_left, run_right):
            length = int(csr.run_member_starts[run + 1]) - int(
                csr.run_member_starts[run]
            )
            width = 1 << (length - 1).bit_length()
            buckets.setdefault(width, []).append(run)
        shell_buckets: list[_RunBucket] = []
        for width in sorted(buckets):
            runs = buckets[width]
            local_runs = np.empty(len(runs), dtype="<u8")
            for row, run in enumerate(runs):
                local_runs[row] = run - run_left
            local_runs.setflags(write=False)
            shell_buckets.append(
                _RunBucket(
                    width=width,
                    run_local_indices=local_runs,
                )
            )
        schedules.append(tuple(shell_buckets))
    return tuple(schedules)


def _canonical_device(device: Any) -> Any:
    if torch is None:
        raise RuntimeError("fibered-work projection requires PyTorch")
    result = torch.device(device)
    if result.type not in {"cpu", "cuda"}:
        raise ValueError("fibered-work projection supports only CPU and CUDA")
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA fibered-work execution is unavailable")
    if result.type == "cuda" and result.index is None:
        # ``torch.device("cuda") != torch.device("cuda:0")``: bind the
        # registered device to its concrete index so row-device identity
        # comparisons hold on every CUDA host.
        result = torch.device("cuda", int(torch.cuda.current_device()))
    return result


def _device_identity(device: Any) -> str:
    if device.type != "cuda":
        return str(device)
    index = device.index if device.index is not None else int(torch.cuda.current_device())
    return f"cuda:{index}:{torch.cuda.get_device_name(device)}"


def _fixed_tree_reduce_last(values: Any) -> Any:
    if torch is None or not isinstance(values, torch.Tensor):
        raise TypeError("fixed-tree reduction requires a torch tensor")
    if values.ndim < 2 or int(values.shape[-1]) <= 0:
        raise ValueError("fixed-tree reduction requires a nonempty final axis")
    original_width = int(values.shape[-1])
    width = 1 << (original_width - 1).bit_length()
    if width == original_width:
        work = values
    else:
        shape = tuple(int(value) for value in values.shape[:-1]) + (width,)
        work = torch.zeros(shape, dtype=torch.float64, device=values.device)
        work[..., :original_width].copy_(values)
    while width > 1:
        work = work[..., :width:2] + work[..., 1:width:2]
        width //= 2
    # Width-one runs otherwise return a view into ``values``.  The caller
    # reuses its gathered tile in place for the absolute tree, so the scalar
    # result must always own storage regardless of run width.
    return work[..., 0].clone()


def _upward_add(left: float, right: float) -> float:
    result = float(left) + float(right)
    if not math.isfinite(result) or result < 0.0:
        raise FloatingPointError("outward nonnegative addition left finite binary64")
    return math.nextafter(result, math.inf)


def _upward_multiply(left: float, right: float) -> float:
    result = float(left) * float(right)
    if not math.isfinite(result) or result < 0.0:
        raise FloatingPointError("outward nonnegative product left finite binary64")
    return math.nextafter(result, math.inf)


def _upward_divide(numerator: float, denominator: float) -> float:
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise FloatingPointError("outward division denominator is invalid")
    result = float(numerator) / float(denominator)
    if not math.isfinite(result) or result < 0.0:
        raise FloatingPointError("outward quotient left finite binary64")
    return math.nextafter(result, math.inf)


def _fibered_outward_enclosure(
    center: np.ndarray,
    singleton_triangle: np.ndarray,
    *,
    endpoint_removed_counts: tuple[int, ...],
    endpoint_count: int,
    cut_count: int,
    maximum_fibers_per_cut: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Enclose the exact fibered functional of consumed binary64 rows.

    The operation-depth bound deliberately dominates the implemented fixed
    trees and shell-prefix updates.  It is conservative: using selected route
    count instead of the much smaller longest within-fiber path avoids an
    unreceipted dependence on outcome-specific occupancy.
    """

    if center.shape != singleton_triangle.shape or center.shape[1] != endpoint_count:
        raise ValueError("fibered-work enclosure axes are inconsistent")
    triangle_upper = np.empty_like(center)
    radius = np.empty_like(center)
    lower = np.empty_like(center)
    upper = np.empty_like(center)
    for endpoint, count in enumerate(endpoint_removed_counts):
        operation_depth = count + maximum_fibers_per_cut + cut_count + endpoint + 8
        depth_u = _upward_multiply(float(operation_depth), _UNIT_ROUNDOFF)
        denominator = math.nextafter(1.0 - depth_u, -math.inf)
        if denominator <= 0.0:
            raise FloatingPointError("fibered-work roundoff bound is unresolved")
        gamma = _upward_divide(depth_u, denominator)
        eta = _upward_multiply(
            _upward_multiply(
                4.0,
                float(count + maximum_fibers_per_cut + cut_count),
            ),
            float(np.finfo(np.float64).tiny),
        )
        for row in range(center.shape[0]):
            observed_triangle = float(singleton_triangle[row, endpoint])
            unweighted_upper = _upward_divide(
                _upward_add(observed_triangle, eta),
                math.nextafter(1.0 - gamma, -math.inf),
            )
            weighted_upper = _upward_divide(unweighted_upper, float(cut_count))
            weighted_eta = _upward_divide(eta, float(cut_count))
            error = _upward_add(
                _upward_multiply(gamma, weighted_upper),
                weighted_eta,
            )
            value = float(center[row, endpoint])
            weighted_upper = max(
                weighted_upper,
                _upward_add(abs(value), error),
            )
            low = max(0.0, math.nextafter(value - error, -math.inf))
            high = math.nextafter(value + error, math.inf)
            if not all(
                math.isfinite(item)
                for item in (weighted_upper, error, low, high)
            ):
                raise FloatingPointError("fibered-work enclosure overflowed")
            # The published triangle diagnostic remains in its natural,
            # unweighted singleton units.  ``weighted_upper`` is used only to
            # bound the cut-averaged fibered functional.
            triangle_upper[row, endpoint] = unweighted_upper
            radius[row, endpoint] = error
            lower[row, endpoint] = low
            upper[row, endpoint] = high
    return triangle_upper, radius, lower, upper


class V1FiberedWorkProjectionGPUConsumer(_RouteChunkedLossOnlyConsumer):
    """Bounded deterministic C-stream consumer for canonical fibered work."""

    positive_removal_stream_coordinate = LADDER_PROJECTION_COORDINATE

    def __init__(
        self,
        request: V1FiberedWorkProjectionRequest,
        *,
        route_ids: Sequence[str],
        device: Any = "cpu",
        production: bool = False,
    ) -> None:
        if torch is None:
            raise RuntimeError("fibered-work projection requires PyTorch")
        if type(request) is not V1FiberedWorkProjectionRequest:
            raise TypeError("fibered-work consumer requires an exact request")
        if type(production) is not bool:
            raise TypeError("production mode must be one exact bool")
        request.assert_integrity()
        if type(route_ids) not in {tuple, ConstructiveOrdinalRouteAxis}:
            raise TypeError(
                "fibered-work route IDs must be an immutable tuple or compact "
                "ConstructiveOrdinalRouteAxis"
            )
        if (
            len(route_ids) != request.route_count
            or ordered_string_axis_hash(
                route_ids,
                name="fibered-work ordered score-route axis",
            )
            != request.ordered_score_route_axis_hash
        ):
            raise ValueError("fibered-work score-route IDs differ from request")
        resolved_device = _canonical_device(device)
        if production and resolved_device.type != "cuda":
            raise PermissionError(
                "production fibered-work projection requires CUDA; CPU is reference-only"
            )
        self.request = request
        # ``ConstructiveOrdinalRouteAxis`` is a compact 57M-coordinate
        # sequence.  Preserve the validated source object; tuple expansion
        # would manufacture tens of millions of Python strings.
        self.route_ids = route_ids
        self.device = resolved_device
        self.production = production
        self.loss_pullback_hash = (
            request.projection_request.raw_logit_stream_identity_hash
        )
        self.head_ids = ("positive_removal:raw_target_logit:fibered_work",)
        self.chunk_size = request.projection_request.registered_route_chunk_size
        self._route_chunk_bound = False
        self._device_identity = _device_identity(self.device)
        self._numerical_backend = (
            "torch_cuda_native_fiber_fixed_tree_unique_writer_v1"
            if self.device.type == "cuda"
            else "torch_cpu_native_fiber_fixed_tree_reference_v1"
        )
        self._shell_physical_indices = tuple(
            request.run_csr.selected_route_indices[
                int(request.run_csr.shell_selected_starts[shell]) : int(
                    request.run_csr.shell_selected_starts[shell + 1]
                )
            ]
            for shell in range(request.endpoint_count)
        )
        self._run_buckets = _build_run_buckets(request.run_csr)
        static_device_csr_bytes = request.run_csr.run_count * np.dtype(
            np.int64
        ).itemsize
        if static_device_csr_bytes > request.maximum_registered_device_bytes:
            raise MemoryError(
                "native-fiber device geometry exceeds the registered total ceiling"
            )
        self._device_run_fibers = tuple(
            torch.as_tensor(
                np.asarray(
                    request.run_csr.run_fiber_indices[
                        int(request.run_csr.shell_run_starts[shell]) : int(
                            request.run_csr.shell_run_starts[shell + 1]
                        )
                    ],
                    dtype=np.int64,
                ),
                dtype=torch.int64,
                device=self.device,
            )
            for shell in range(request.endpoint_count)
        )
        self._device_csr_bytes = sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in self._device_run_fibers
        )
        self._next_context = 0
        self._packet_index = 0
        self._packet_context_start = 0
        self._packet_context_ids: tuple[str, ...] = ()
        self._packet_draw_ids: tuple[str, ...] = ()
        self._packet_stratum_ids: tuple[str, ...] = ()
        self._packet_shell_rows: tuple[Any, ...] = ()
        self._packet_nonfinite_marker: Any | None = None
        self._packet_next_route = 0
        self._packet_windows: list[tuple[int, int]] = []
        self._packet_peak_registered_bytes = 0
        self._packet_base_registered_bytes = 0
        self._packet_run_tile_schedule: tuple[
            tuple[tuple[_RunBucket, tuple[tuple[int, int], ...]], ...], ...
        ] = ()
        self._packet_run_tile_schedule_hash = ""
        self._packet_maximum_run_tile_member_count = 0
        self._packet_preoutcome_peak_registered_device_bytes_upper = 0
        self._packet_open = False
        self._failed = False
        self._failure_reason: str | None = None
        self._closed_packets: list[_ClosedPacket] = []
        self._closed_result: V1FiberedWorkProjectionResult | None = None

    @property
    def is_failed(self) -> bool:
        return self._failed

    @property
    def delivered_record_count(self) -> int:
        return 0

    def bind_route_chunk_size(self, chunk_size: int) -> None:
        try:
            self._assert_live()
            observed = _positive_int(chunk_size, name="provider route chunk size")
            if observed != self.request.projection_request.registered_route_chunk_size:
                raise ValueError("provider route chunk size differs from frozen request")
            if self._packet_open or self._next_context:
                raise RuntimeError("cannot bind route chunks after acquisition starts")
            self.chunk_size = observed
            self._route_chunk_bound = True
        except Exception as error:
            self._abort(error)

    def _assert_live(self, *, verify_request: bool = False) -> None:
        if self._failed:
            raise RuntimeError(
                "fibered-work transaction is permanently refused after: "
                f"{self._failure_reason}"
            )
        if self._closed_result is not None:
            raise RuntimeError("fibered-work transaction is already closed")
        if verify_request:
            self.request.assert_integrity()

    def _clear_packet(self) -> None:
        self._packet_context_start = 0
        self._packet_context_ids = ()
        self._packet_draw_ids = ()
        self._packet_stratum_ids = ()
        self._packet_shell_rows = ()
        self._packet_nonfinite_marker = None
        self._packet_next_route = 0
        self._packet_windows = []
        self._packet_peak_registered_bytes = 0
        self._packet_base_registered_bytes = 0
        self._packet_run_tile_schedule = ()
        self._packet_run_tile_schedule_hash = ""
        self._packet_maximum_run_tile_member_count = 0
        self._packet_preoutcome_peak_registered_device_bytes_upper = 0
        self._packet_open = False

    def _abort(self, error: Exception) -> None:
        self._failed = True
        self._failure_reason = f"{type(error).__name__}: {error}"
        self._clear_packet()
        self._closed_packets.clear()
        self._closed_result = None
        raise error

    def _route_axis_matches(self, route_ids: Sequence[str]) -> bool:
        if route_ids is self.route_ids:
            return True
        return (
            len(route_ids) == self.request.route_count
            and ordered_string_axis_hash(
                route_ids,
                name="observed fibered-work ordered score-route axis",
            )
            == self.request.ordered_score_route_axis_hash
        )

    def _assert_registered_peak(self) -> None:
        if (
            self._packet_peak_registered_bytes
            > self.request.maximum_registered_device_bytes
            or self._packet_peak_registered_bytes
            > self._packet_preoutcome_peak_registered_device_bytes_upper
        ):
            raise MemoryError(
                "fibered-work live registered tensors exceeded their pre-outcome "
                "total-memory forecast"
            )

    def _build_packet_run_tile_schedule(
        self,
        *,
        batch_size: int,
        base_registered_bytes: int,
    ) -> tuple[
        tuple[tuple[_RunBucket, tuple[tuple[int, int], ...]], ...], ...
    ]:
        """Freeze a pre-outcome tile schedule under the registered cap."""

        batch = _positive_int(batch_size, name="packet batch size")
        transient_cap = self.request.maximum_temporary_device_bytes
        total_cap = self.request.maximum_registered_device_bytes
        base_bytes = _positive_int(
            base_registered_bytes,
            name="packet base registered bytes",
        )
        if base_bytes > total_cap:
            raise MemoryError(
                "selected-shell cache and static geometry exceed the registered "
                "total device-memory ceiling"
            )
        common_persistent = (
            batch * self.request.native_fiber_axis.fiber_count * _FLOAT64_BYTES
            + batch * self.request.endpoint_count * _FLOAT64_BYTES * 3
            + batch * _FLOAT64_BYTES * 2
        )
        maximum_fibers = max(
            cut.fiber_count for cut in self.request.native_fiber_axis.cuts
        )
        cut_reduction_transient = (
            batch * maximum_fibers * 12
            + batch * self.request.native_fiber_axis.cut_count * 16
        )
        maximum_chunk_selected = max(
            min(
                int(indices.size),
                self.request.projection_request.registered_route_chunk_size,
            )
            for indices in self._shell_physical_indices
        )
        selected_gather_transient = maximum_chunk_selected * (
            np.dtype(np.int64).itemsize + batch * _FLOAT64_BYTES
        )
        finite_check_transient = (
            batch
            * self.request.projection_request.registered_route_chunk_size
            + _FLOAT64_BYTES
        )
        route_chunk_transient = max(
            selected_gather_transient,
            finite_check_transient,
        )
        if max(cut_reduction_transient, route_chunk_transient) > transient_cap:
            raise MemoryError(
                "registered transient workspace cannot fit deterministic route "
                "gather or cut reduction"
            )
        preoutcome_peak = base_bytes + route_chunk_transient
        if preoutcome_peak > total_cap:
            raise MemoryError(
                "route-chunk gather exceeds the registered total device-memory ceiling"
            )
        schedule = []
        payload_shells: list[dict[str, object]] = []
        maximum_members = 0
        for shell, buckets in enumerate(self._run_buckets):
            run_left = int(self.request.run_csr.shell_run_starts[shell])
            run_right = int(self.request.run_csr.shell_run_starts[shell + 1])
            run_count_for_shell = run_right - run_left
            run_array_bytes = batch * run_count_for_shell * _FLOAT64_BYTES * 2
            fixed_registered = base_bytes + common_persistent + run_array_bytes
            non_tile_transient = max(
                cut_reduction_transient,
                batch * run_count_for_shell * _FLOAT64_BYTES * 2,
            )
            if (
                non_tile_transient > transient_cap
                or fixed_registered + non_tile_transient > total_cap
            ):
                raise MemoryError(
                    "fiber/run state cannot fit under the registered simultaneous "
                    "device-memory ceiling"
                )
            preoutcome_peak = max(
                preoutcome_peak,
                fixed_registered + non_tile_transient,
            )
            shell_schedule = []
            payload_buckets: list[dict[str, object]] = []
            for bucket in buckets:
                # Device members per run: gathered binary64 values plus the
                # largest first tree intermediate (1/2 gathered), one int64
                # index, and one bool validity bit.  Per-run output/schedule
                # vectors are included separately.  This is a conservative
                # live-allocation upper bound, not a measured after-the-fact
                # heuristic.
                bytes_per_run = (
                    bucket.width * (batch * 12 + 8 + 1)
                    + batch * 32
                    + 16
                )
                available_total = total_cap - fixed_registered
                tile_budget = min(transient_cap, available_total)
                runs_per_tile = tile_budget // bytes_per_run
                if runs_per_tile <= 0:
                    raise MemoryError(
                        "registered fibered-work temporary workspace cannot fit "
                        "one deterministic native-fiber run"
                    )
                run_count = int(bucket.run_local_indices.size)
                tiles = tuple(
                    (left, min(run_count, left + runs_per_tile))
                    for left in range(0, run_count, runs_per_tile)
                )
                maximum_members = max(
                    maximum_members,
                    max((right - left) * bucket.width for left, right in tiles),
                )
                maximum_tile_runs = max(right - left for left, right in tiles)
                preoutcome_peak = max(
                    preoutcome_peak,
                    fixed_registered + maximum_tile_runs * bytes_per_run,
                )
                shell_schedule.append((bucket, tiles))
                payload_buckets.append(
                    {
                        "width": bucket.width,
                        "run_local_indices": _array_digest(
                            bucket.run_local_indices
                        ),
                        "tiles": tiles,
                        "bytes_per_run_upper": bytes_per_run,
                    }
                )
            schedule.append(tuple(shell_schedule))
            payload_shells.append(
                {"shell": shell, "buckets": payload_buckets}
            )
        result = tuple(schedule)
        schedule_hash = content_hash(
            {
                "schema": "partition_free_v1_fibered_work_run_tile_schedule_v1",
                "request_hash": self.request.request_hash,
                "run_csr_hash": self.request.run_csr.csr_hash,
                "batch_size": batch,
                "maximum_temporary_device_bytes": transient_cap,
                "maximum_registered_device_bytes": total_cap,
                "base_registered_bytes": base_bytes,
                "common_persistent_bytes": common_persistent,
                "cut_reduction_transient_bytes_upper": (
                    cut_reduction_transient
                ),
                "route_chunk_transient_bytes_upper": route_chunk_transient,
                "preoutcome_peak_registered_device_bytes_upper": (
                    preoutcome_peak
                ),
                "shells": payload_shells,
                "floating_atomics_used": False,
            },
            prefix="partition-free-v1-fibered-work-run-tiles-",
        )
        self._packet_run_tile_schedule_hash = schedule_hash
        self._packet_maximum_run_tile_member_count = maximum_members
        self._packet_preoutcome_peak_registered_device_bytes_upper = (
            preoutcome_peak
        )
        return result

    def _packet_selected_shell_cache_bytes(self, *, batch_size: int) -> int:
        batch = _positive_int(batch_size, name="packet batch size")
        selected_values = sum(
            int(indices.size) for indices in self._shell_physical_indices
        )
        return (
            batch * selected_values * _FLOAT64_BYTES
            + _FLOAT64_BYTES
            + self._device_csr_bytes
        )

    def begin_packet(
        self,
        context_ids: Sequence[str],
        *,
        route_ids: Sequence[str],
        destination_transport_hash: str | None,
        loss_pullback_hash: str | None,
    ) -> None:
        try:
            self._assert_live(verify_request=self._packet_index == 0)
            if not self._route_chunk_bound:
                raise RuntimeError("provider chunk ABI must bind before acquisition")
            if self._packet_open:
                raise RuntimeError("fibered-work packet is already open")
            contexts = _identity_axis(context_ids, name="observed context axis")
            if not contexts:
                raise ValueError("fibered-work packet cannot be empty")
            base = self.request.projection_request
            start = self._next_context
            stop = start + len(contexts)
            if contexts != base.expected_context_ids[start:stop]:
                raise ValueError("fibered-work packet violates frozen context order")
            if not self._route_axis_matches(route_ids):
                raise ValueError("fibered-work packet route axis differs")
            if str(destination_transport_hash or "") != base.destination_transport_hash:
                raise ValueError("fibered-work destination transport differs")
            if str(loss_pullback_hash or "") != base.raw_logit_stream_identity_hash:
                raise ValueError("fibered-work raw-logit stream identity differs")
            batch = len(contexts)
            base_bytes = self._packet_selected_shell_cache_bytes(
                batch_size=batch
            )
            run_tile_schedule = self._build_packet_run_tile_schedule(
                batch_size=batch,
                base_registered_bytes=base_bytes,
            )
            shell_rows = tuple(
                torch.empty(
                    (batch, int(indices.size)),
                    dtype=torch.float64,
                    device=self.device,
                )
                for indices in self._shell_physical_indices
            )
            marker = torch.zeros((), dtype=torch.float64, device=self.device)
            self._packet_context_start = start
            self._packet_context_ids = contexts
            self._packet_draw_ids = base.expected_draw_ids[start:stop]
            self._packet_stratum_ids = base.expected_stratum_ids[start:stop]
            self._packet_shell_rows = shell_rows
            self._packet_nonfinite_marker = marker
            self._packet_base_registered_bytes = base_bytes
            self._packet_peak_registered_bytes = base_bytes
            self._packet_run_tile_schedule = run_tile_schedule
            self._packet_open = True
        except Exception as error:
            self._abort(error)

    def consume_positive_removal_chunk(self, start: int, stop: int, rows: Any) -> None:
        try:
            self._assert_live()
            begin = _nonnegative_int(start, name="route window start")
            end = _positive_int(stop, name="route window stop")
            span = end - begin
            contexts = len(self._packet_context_ids)
            if (
                not self._packet_open
                or not self._packet_shell_rows
                or self._packet_nonfinite_marker is None
                or begin != self._packet_next_route
                or span <= 0
                or end > self.request.route_count
                or span > self.request.projection_request.registered_route_chunk_size
            ):
                raise ValueError("fibered-work chunks must cover a bounded prefix")
            if (
                not isinstance(rows, torch.Tensor)
                or tuple(rows.shape) != (contexts, span)
                or rows.dtype != torch.float64
                or rows.device != self.device
                or not rows.is_contiguous()
            ):
                raise ValueError(
                    "fibered-work input must be contiguous float64 "
                    "[context, route-window] on the registered device"
                )
            finite = torch.isfinite(rows).all()
            finite_check_temporary = int(rows.numel()) + _FLOAT64_BYTES
            self._packet_peak_registered_bytes = max(
                self._packet_peak_registered_bytes,
                self._packet_base_registered_bytes + finite_check_temporary,
            )
            self._assert_registered_peak()
            self._packet_nonfinite_marker.add_(
                torch.where(
                    finite,
                    self._packet_nonfinite_marker.new_zeros(()),
                    self._packet_nonfinite_marker.new_full((), float("nan")),
                )
            )
            for shell, physical in enumerate(self._shell_physical_indices):
                left = int(np.searchsorted(physical, begin, side="left"))
                right = int(np.searchsorted(physical, end, side="left"))
                if left == right:
                    continue
                offsets = torch.as_tensor(
                    (physical[left:right] - begin).astype(np.int64, copy=False),
                    dtype=torch.int64,
                    device=self.device,
                )
                selected = torch.index_select(rows, 1, offsets)
                self._packet_shell_rows[shell][:, left:right].copy_(selected.detach())
                temporary = (
                    int(offsets.numel()) * int(offsets.element_size())
                    + int(selected.numel()) * int(selected.element_size())
                )
                self._packet_peak_registered_bytes = max(
                    self._packet_peak_registered_bytes,
                    self._packet_base_registered_bytes + temporary,
                )
                self._assert_registered_peak()
            self._packet_windows.append((begin, end))
            self._packet_next_route = end
        except Exception as error:
            self._abort(error)

    def consume_chunk(self, start: int, stop: int, rows: Any) -> None:
        del start, stop, rows
        self._abort(
            PermissionError(
                "fibered-work projection accepts only direct signed raw-logit "
                "positive-removal rows"
            )
        )

    def consume_chunk_fused(
        self,
        start: int,
        stop: int,
        *,
        global_raw: Any,
        gate: Any,
        pullback: Any,
        affine_scale: float,
        score_epilogue: Any | None = None,
    ) -> None:
        del start, stop, global_raw, gate, pullback, affine_scale, score_epilogue
        self._abort(
            PermissionError(
                "fibered-work projection requires scorer-emitted -G*phi*c rows"
            )
        )

    def _reduce_packet(self) -> tuple[Any, Any, Any]:
        assert torch is not None
        marker = self._packet_nonfinite_marker
        if marker is None or not self._packet_shell_rows:
            raise RuntimeError("fibered-work finalizer has no packet state")
        batch = len(self._packet_context_ids)
        endpoint_count = self.request.endpoint_count
        fiber_state = torch.zeros(
            (batch, self.request.native_fiber_axis.fiber_count),
            dtype=torch.float64,
            device=self.device,
        )
        center = torch.empty(
            (batch, endpoint_count), dtype=torch.float64, device=self.device
        )
        global_signed = torch.empty_like(center)
        singleton_triangle = torch.empty_like(center)
        running_signed = torch.zeros(batch, dtype=torch.float64, device=self.device)
        running_triangle = torch.zeros_like(running_signed)
        persistent_bytes = (
            int(fiber_state.numel())
            + int(center.numel()) * 3
            + int(running_signed.numel()) * 2
        ) * _FLOAT64_BYTES
        self._packet_peak_registered_bytes = max(
            self._packet_peak_registered_bytes,
            self._packet_base_registered_bytes + persistent_bytes,
        )
        self._assert_registered_peak()
        csr = self.request.run_csr
        for shell, values in enumerate(self._packet_shell_rows):
            run_left = int(csr.shell_run_starts[shell])
            run_right = int(csr.shell_run_starts[shell + 1])
            run_count = run_right - run_left
            run_sums = torch.empty(
                (batch, run_count), dtype=torch.float64, device=self.device
            )
            run_triangles = torch.empty_like(run_sums)
            selected_left = int(csr.shell_selected_starts[shell])
            if not self._packet_run_tile_schedule:
                raise RuntimeError("fibered-work run-tile schedule is absent")
            for bucket, tiles in self._packet_run_tile_schedule[shell]:
                width = bucket.width
                for tile_left, tile_right in tiles:
                    local_runs = np.asarray(
                        bucket.run_local_indices[tile_left:tile_right],
                        dtype=np.int64,
                    )
                    members = np.zeros(
                        (len(local_runs), width), dtype=np.int64
                    )
                    valid = np.zeros(
                        (len(local_runs), width), dtype=np.bool_
                    )
                    for row, local_run in enumerate(local_runs):
                        run = run_left + int(local_run)
                        member_left = (
                            int(csr.run_member_starts[run]) - selected_left
                        )
                        member_right = (
                            int(csr.run_member_starts[run + 1]) - selected_left
                        )
                        length = member_right - member_left
                        members[row, :length] = np.arange(
                            member_left, member_right, dtype=np.int64
                        )
                        valid[row, :length] = True
                    run_indices = torch.as_tensor(
                        local_runs,
                        dtype=torch.int64,
                        device=self.device,
                    )
                    member_indices = torch.as_tensor(
                        members.reshape(-1),
                        dtype=torch.int64,
                        device=self.device,
                    )
                    valid_device = torch.as_tensor(
                        valid,
                        dtype=torch.bool,
                        device=self.device,
                    )
                    gathered = torch.index_select(
                        values,
                        1,
                        member_indices,
                    ).reshape(batch, -1, width)
                    gathered.masked_fill_(~valid_device.unsqueeze(0), 0.0)
                    signed = _fixed_tree_reduce_last(gathered)
                    # Reuse the gathered allocation for the absolute tree;
                    # never hold a second full tile of absolute values.
                    gathered.abs_()
                    triangle = _fixed_tree_reduce_last(gathered)
                    run_sums.index_copy_(1, run_indices, signed)
                    run_triangles.index_copy_(1, run_indices, triangle)
                    temporary = (
                        int(gathered.numel()) * _FLOAT64_BYTES
                        + int(gathered.numel()) * _FLOAT64_BYTES // 2
                        + int(member_indices.numel())
                        * int(member_indices.element_size())
                        + int(valid_device.numel())
                        * int(valid_device.element_size())
                        + int(run_indices.numel())
                        * int(run_indices.element_size())
                        + int(signed.numel()) * _FLOAT64_BYTES * 2
                    )
                    if temporary > self.request.maximum_temporary_device_bytes:
                        raise MemoryError(
                            "fibered-work realized deterministic tile exceeded "
                            "its registered temporary workspace"
                        )
                    self._packet_peak_registered_bytes = max(
                        self._packet_peak_registered_bytes,
                        self._packet_base_registered_bytes
                        + persistent_bytes
                        + int(run_sums.numel()) * _FLOAT64_BYTES * 2
                        + temporary,
                    )
                    self._assert_registered_peak()
                    del (
                        gathered,
                        member_indices,
                        valid_device,
                        run_indices,
                        signed,
                        triangle,
                    )
            fibers = self._device_run_fibers[shell]
            prior = torch.index_select(fiber_state, 1, fibers)
            updated = prior + run_sums
            state_update_temporary = (
                int(prior.numel()) + int(updated.numel())
            ) * _FLOAT64_BYTES
            self._packet_peak_registered_bytes = max(
                self._packet_peak_registered_bytes,
                self._packet_base_registered_bytes
                + persistent_bytes
                + int(run_sums.numel()) * _FLOAT64_BYTES * 2
                + state_update_temporary,
            )
            self._assert_registered_peak()
            fiber_state.index_copy_(1, fibers, updated)
            del prior, updated
            shell_signed = _fixed_tree_reduce_last(run_sums)
            shell_triangle = _fixed_tree_reduce_last(run_triangles)
            running_signed = running_signed + shell_signed
            running_triangle = running_triangle + shell_triangle
            cut_values = []
            cut_reduction_temporary = (
                batch
                * max(
                    cut.fiber_count
                    for cut in self.request.native_fiber_axis.cuts
                )
                * 12
                + batch * self.request.native_fiber_axis.cut_count * 16
            )
            self._packet_peak_registered_bytes = max(
                self._packet_peak_registered_bytes,
                self._packet_base_registered_bytes
                + persistent_bytes
                + int(run_sums.numel()) * _FLOAT64_BYTES * 2
                + cut_reduction_temporary,
            )
            self._assert_registered_peak()
            for cut in self.request.native_fiber_axis.cuts:
                cut_values.append(
                    _fixed_tree_reduce_last(
                        torch.abs(fiber_state[:, cut.fiber_offset : cut.fiber_stop])
                    )
                )
            cut_matrix = torch.stack(cut_values, dim=1)
            center[:, shell].copy_(
                _fixed_tree_reduce_last(cut_matrix)
                / float(self.request.native_fiber_axis.cut_count)
            )
            global_signed[:, shell].copy_(running_signed)
            singleton_triangle[:, shell].copy_(running_triangle)
        center[0, 0].add_(marker)
        global_signed[0, 0].add_(marker)
        singleton_triangle[0, 0].add_(marker)
        return center, global_signed, singleton_triangle

    def end_packet(self) -> None:
        try:
            self._assert_live()
            if not self._packet_open or self._packet_next_route != self.request.route_count:
                raise RuntimeError("fibered-work packet did not cover route axis exactly")
            center_device, signed_device, triangle_device = self._reduce_packet()
            batch = len(self._packet_context_ids)
            endpoints = self.request.endpoint_count
            center, center_hash = _canonical_matrix(
                center_device.detach().to("cpu").numpy(),
                shape=(batch, endpoints),
                name="packet fibered-work centers",
            )
            signed, signed_hash = _canonical_matrix(
                signed_device.detach().to("cpu").numpy(),
                shape=(batch, endpoints),
                name="packet global-signed diagnostics",
            )
            triangle, triangle_hash = _canonical_matrix(
                triangle_device.detach().to("cpu").numpy(),
                shape=(batch, endpoints),
                name="packet singleton-triangle diagnostics",
            )
            windows = tuple(self._packet_windows)
            window_hash = content_hash(
                {
                    "request_hash": self.request.request_hash,
                    "packet_index": self._packet_index,
                    "route_count": self.request.route_count,
                    "ordered_score_route_axis_hash": (
                        self.request.ordered_score_route_axis_hash
                    ),
                    "route_windows": windows,
                },
                prefix=_WINDOW_PREFIX,
            )
            host_bytes = batch * endpoints * _FLOAT64_BYTES * 3
            receipt = V1FiberedWorkProjectionPacketReceipt(
                request_hash=self.request.request_hash,
                projection_request_kind=str(
                    self.request.projection_semantics[
                        "projection_request_kind"
                    ]
                ),
                projection_semantics_binding_hash=(
                    self.request.projection_semantics_binding_hash
                ),
                packet_index=self._packet_index,
                context_start=self._packet_context_start,
                context_stop=self._packet_context_start + batch,
                context_ids=self._packet_context_ids,
                draw_ids=self._packet_draw_ids,
                stratum_ids=self._packet_stratum_ids,
                route_windows=windows,
                route_window_ledger_hash=window_hash,
                run_tile_schedule_hash=self._packet_run_tile_schedule_hash,
                fibered_center_matrix_hash=center_hash,
                global_signed_diagnostic_matrix_hash=signed_hash,
                singleton_triangle_diagnostic_matrix_hash=triangle_hash,
                native_fiber_axis_receipt_hash=(
                    self.request.native_fiber_axis_receipt_hash
                ),
                native_fiber_measure_hash=self.request.native_fiber_measure_hash,
                run_csr_hash=self.request.run_csr.csr_hash,
                numerical_backend=self._numerical_backend,
                production_cuda_execution=self.production,
                maximum_temporary_device_bytes=(
                    self.request.maximum_temporary_device_bytes
                ),
                maximum_registered_device_bytes=(
                    self.request.maximum_registered_device_bytes
                ),
                preoutcome_peak_registered_device_bytes_upper=(
                    self._packet_preoutcome_peak_registered_device_bytes_upper
                ),
                maximum_run_tile_member_count=(
                    self._packet_maximum_run_tile_member_count
                ),
                peak_registered_device_bytes=self._packet_peak_registered_bytes,
                host_materialization_bytes=host_bytes,
                device_to_host_bytes=(host_bytes if self.device.type == "cuda" else 0),
                device_identity=self._device_identity,
            )
            self._closed_packets.append(
                _ClosedPacket(receipt, center, signed, triangle)
            )
            self._next_context += batch
            self._packet_index += 1
            self._clear_packet()
        except Exception as error:
            self._abort(error)

    def close(self) -> V1FiberedWorkProjectionResult:
        if self._closed_result is not None:
            return self._closed_result
        try:
            self._assert_live(verify_request=True)
            if self._packet_open or self._next_context != self.request.context_count:
                raise RuntimeError("cannot close a partial fibered-work request")
            if not self._closed_packets:
                raise RuntimeError("fibered-work request produced no packets")
            shape = (self.request.context_count, self.request.endpoint_count)
            center, center_hash = _canonical_matrix(
                np.concatenate([item.center for item in self._closed_packets], axis=0),
                shape=shape,
                name="complete fibered-work centers",
            )
            signed, signed_hash = _canonical_matrix(
                np.concatenate(
                    [item.global_signed for item in self._closed_packets], axis=0
                ),
                shape=shape,
                name="complete global-signed diagnostics",
            )
            triangle, triangle_hash = _canonical_matrix(
                np.concatenate(
                    [item.singleton_triangle for item in self._closed_packets], axis=0
                ),
                shape=shape,
                name="complete singleton-triangle diagnostics",
            )
            triangle_upper_raw, radius_raw, lower_raw, upper_raw = (
                _fibered_outward_enclosure(
                    center,
                    triangle,
                    endpoint_removed_counts=(
                        self.request.projection_request.endpoint_removed_counts
                    ),
                    endpoint_count=self.request.endpoint_count,
                    cut_count=self.request.native_fiber_axis.cut_count,
                    maximum_fibers_per_cut=max(
                        cut.fiber_count for cut in self.request.native_fiber_axis.cuts
                    ),
                )
            )
            triangle_upper, triangle_upper_hash = _canonical_matrix(
                triangle_upper_raw,
                shape=shape,
                name="singleton-triangle upper bounds",
            )
            radius, radius_hash = _canonical_matrix(
                radius_raw, shape=shape, name="fibered-work numerical radii"
            )
            lower, lower_hash = _canonical_matrix(
                lower_raw, shape=shape, name="fibered-work numerical lower bounds"
            )
            upper, upper_hash = _canonical_matrix(
                upper_raw, shape=shape, name="fibered-work numerical upper bounds"
            )
            packets = tuple(item.receipt for item in self._closed_packets)
            base = self.request.projection_request
            semantics = dict(self.request.projection_semantics)
            receipt = V1FiberedWorkProjectionDerivationReceipt(
                request_hash=self.request.request_hash,
                projection_request_hash=base.request_hash,
                projection_request_kind=str(
                    semantics["projection_request_kind"]
                ),
                projection_semantics_binding_hash=(
                    self.request.projection_semantics_binding_hash
                ),
                projection_role=semantics["projection_role"],
                row_semantics=str(semantics["row_semantics"]),
                chart_identity_hash=semantics["chart_identity_hash"],
                joint_fqr_family_registration_hash=(
                    semantics["joint_fqr_family_registration_hash"]
                ),
                certification_stream_commitment_hash=str(
                    semantics["certification_stream_commitment_hash"]
                ),
                corrected_evidence_stream_identity_hash=str(
                    semantics["corrected_evidence_stream_identity_hash"]
                ),
                literal_jacobian_stream_identity_hash=(
                    semantics["literal_jacobian_stream_identity_hash"]
                ),
                same_capture_and_vjp_registration_hash=(
                    semantics["same_capture_and_vjp_registration_hash"]
                ),
                d_c_causal_registration_hash=(
                    semantics["d_c_causal_registration_hash"]
                ),
                directional_commit_eligible=bool(
                    semantics["directional_commit_eligible"]
                ),
                design_hash=base.design_hash,
                ladder_hash=base.ladder_hash,
                native_fiber_axis_receipt_hash=(
                    self.request.native_fiber_axis_receipt_hash
                ),
                native_fiber_measure_hash=self.request.native_fiber_measure_hash,
                run_csr_hash=self.request.run_csr.csr_hash,
                physical_mask_axis_hash=self.request.physical_mask_axis_hash,
                ordered_score_route_axis_hash=(
                    self.request.ordered_score_route_axis_hash
                ),
                context_ids=base.expected_context_ids,
                draw_ids=base.expected_draw_ids,
                stratum_ids=base.expected_stratum_ids,
                endpoint_action_hashes=base.endpoint_action_hashes,
                packet_receipt_hashes=tuple(item.receipt_hash for item in packets),
                route_window_ledger_hashes=tuple(
                    item.route_window_ledger_hash for item in packets
                ),
                run_tile_schedule_hashes=tuple(
                    item.run_tile_schedule_hash for item in packets
                ),
                fibered_center_matrix_hash=center_hash,
                global_signed_diagnostic_matrix_hash=signed_hash,
                singleton_triangle_diagnostic_matrix_hash=triangle_hash,
                singleton_triangle_upper_matrix_hash=triangle_upper_hash,
                numerical_radius_matrix_hash=radius_hash,
                numerical_lower_matrix_hash=lower_hash,
                numerical_upper_matrix_hash=upper_hash,
                projection_kernel_source_hash=fibered_work_projection_kernel_source_hash(),
                numerical_backend=self._numerical_backend,
                production_cuda_execution=self.production,
                maximum_temporary_device_bytes=(
                    self.request.maximum_temporary_device_bytes
                ),
                maximum_registered_device_bytes=(
                    self.request.maximum_registered_device_bytes
                ),
                preoutcome_peak_registered_device_bytes_upper=max(
                    item.preoutcome_peak_registered_device_bytes_upper
                    for item in packets
                ),
                peak_registered_device_bytes=max(
                    item.peak_registered_device_bytes for item in packets
                ),
                total_host_materialization_bytes=sum(
                    item.host_materialization_bytes for item in packets
                ),
                exact_device_to_host_bytes=sum(
                    item.device_to_host_bytes for item in packets
                ),
                packet_count=len(packets),
                route_chunk_update_count=sum(
                    len(item.route_windows) for item in packets
                ),
            )
            result = V1FiberedWorkProjectionResult(
                receipt=receipt,
                packet_receipts=packets,
                fibered_center_matrix=center,
                global_signed_diagnostic_matrix=signed,
                singleton_triangle_diagnostic_matrix=triangle,
                singleton_triangle_upper_matrix=triangle_upper,
                numerical_radius_matrix=radius,
                numerical_lower_matrix=lower,
                numerical_upper_matrix=upper,
            )
            self._closed_packets.clear()
            self._closed_result = result
            return result
        except Exception as error:
            self._abort(error)
            raise AssertionError("unreachable")  # pragma: no cover

    def route_streaming_receipt(self) -> Mapping[str, object]:
        if (
            self._closed_result is None
            and not self._failed
            and not self._packet_open
            and self._next_context == self.request.context_count
        ):
            self.close()
        if self._closed_result is None:
            if self._failed:
                raise RuntimeError("refused fibered-work transaction has no receipt")
            raise RuntimeError("fibered-work transaction must close before receipt")
        receipt = self._closed_result.receipt
        return MappingProxyType(
            {
                "route_chunked_streaming": True,
                "route_chunk_size": (
                    self.request.projection_request.registered_route_chunk_size
                ),
                "route_chunk_update_count": receipt.route_chunk_update_count,
                "loss_only_route_streaming": True,
                "fibered_corrected_work_projection_streaming": True,
                "projection_request_kind": receipt.projection_request_kind,
                "projection_semantics_binding_hash": (
                    receipt.projection_semantics_binding_hash
                ),
                "projection_role": receipt.projection_role,
                "row_semantics": receipt.row_semantics,
                "chart_identity_hash": receipt.chart_identity_hash,
                "joint_fqr_family_registration_hash": (
                    receipt.joint_fqr_family_registration_hash
                ),
                "certification_stream_commitment_hash": (
                    receipt.certification_stream_commitment_hash
                ),
                "corrected_evidence_stream_identity_hash": (
                    receipt.corrected_evidence_stream_identity_hash
                ),
                "literal_jacobian_stream_identity_hash": (
                    receipt.literal_jacobian_stream_identity_hash
                ),
                "same_capture_and_vjp_registration_hash": (
                    receipt.same_capture_and_vjp_registration_hash
                ),
                "d_c_causal_registration_hash": (
                    receipt.d_c_causal_registration_hash
                ),
                "directional_commit_eligible": (
                    receipt.directional_commit_eligible
                ),
                "positive_removal_stream_coordinate": LADDER_PROJECTION_COORDINATE,
                "positive_removal_stream_identity_hash": (
                    self.request.projection_request.raw_logit_stream_identity_hash
                ),
                "projection_orientation": receipt.row_semantics,
                "projection_units": LADDER_PROJECTION_UNITS,
                "physical_mask_axis_hash": self.request.physical_mask_axis_hash,
                "ordered_score_route_axis_hash": (
                    self.request.ordered_score_route_axis_hash
                ),
                "native_fiber_axis_receipt_hash": (
                    self.request.native_fiber_axis_receipt_hash
                ),
                "native_fiber_measure_hash": self.request.native_fiber_measure_hash,
                "fibered_work_run_csr_hash": self.request.run_csr.csr_hash,
                "fibered_work_derivation_receipt_hash": receipt.receipt_hash,
                "projection_kernel_source_hash": receipt.projection_kernel_source_hash,
                "projection_arithmetic_id": FIBERED_WORK_ARITHMETIC_ID,
                "outward_enclosure_id": FIBERED_WORK_OUTWARD_ENCLOSURE_ID,
                "numerical_backend": receipt.numerical_backend,
                "production_cuda_execution": receipt.production_cuda_execution,
                "production_c_stream_authority": (
                    receipt.production_cuda_execution
                ),
                "cpu_reference_only": receipt.numerical_backend.startswith(
                    "torch_cpu_"
                ),
                "reference_execution_only": not receipt.production_cuda_execution,
                "maximum_temporary_device_bytes": (
                    receipt.maximum_temporary_device_bytes
                ),
                "maximum_registered_device_bytes": (
                    receipt.maximum_registered_device_bytes
                ),
                "preoutcome_peak_registered_device_bytes_upper": (
                    receipt.preoutcome_peak_registered_device_bytes_upper
                ),
                "run_tile_schedule_hashes": receipt.run_tile_schedule_hashes,
                "fixed_tree_reduction": True,
                "one_writer_fiber_updates": True,
                "floating_atomics_used": False,
                "route_scale_fiber_id_array_materialized": False,
                "within_native_fiber_cancellation_only": True,
                "cross_context_cancellation_used": False,
                "cross_native_fiber_work_cancellation_used": False,
                "global_signed_projection_is_diagnostic_only": True,
                "singleton_triangle_is_diagnostic_only": True,
                "outward_numerical_enclosure_provided": True,
                "peak_registered_device_bytes": receipt.peak_registered_device_bytes,
                "projection_device_to_host_bytes": receipt.exact_device_to_host_bytes,
                "host_context_route_rows_transferred": False,
                "scalar_fibered_work_derivation_authority": True,
                "fq_certification_authority": False,
                "score_order_authority": False,
                "bce_oracle_authority": False,
                "mask_commit_authority": False,
            }
        )


__all__ = [
    "FIBERED_WORK_ARITHMETIC_ID",
    "FIBERED_WORK_AUTHORITY",
    "FIBERED_WORK_DERIVATION_RECEIPT_SCHEMA",
    "FIBERED_WORK_DIAGNOSTIC_ID",
    "FIBERED_WORK_MEMORY_SCOPE",
    "FIBERED_WORK_PACKET_RECEIPT_SCHEMA",
    "FIBERED_WORK_PROJECTION_REQUEST_SCHEMA",
    "FIBERED_WORK_RESULT_SCHEMA",
    "FIBERED_WORK_RUN_CSR_SCHEMA",
    "V1FiberedWorkProjectionDerivationReceipt",
    "V1FiberedWorkProjectionGPUConsumer",
    "V1FiberedWorkProjectionPacketReceipt",
    "V1FiberedWorkProjectionRequest",
    "V1FiberedWorkProjectionResult",
    "V1FiberedWorkRunCSR",
    "build_v1_fibered_work_run_csr",
    "fibered_work_projection_kernel_source_hash",
]
