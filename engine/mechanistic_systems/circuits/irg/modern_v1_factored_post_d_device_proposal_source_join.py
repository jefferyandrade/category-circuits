"""Native factored route-iii source identity for spectral proposal inputs.

Joins the finalized registered source, rounded literal-J Q, native physical
coordinates and compact structural legality. It retains no signed-F operator:
Phase A can run without one; a later operator is checked against the same
source when least-work integration requests it. No sampled FD/row-bank or
response-endpoint identity is manufactured. Solver admission remains a
separate, currently unlanded consumer gate.
"""
from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from types import MappingProxyType
from typing import Mapping

from mechanistic_systems.circuits.irg.modern_v1_factored_alpha_source_binding import V1FactoredDAlphaSourceBindingReceipt
from mechanistic_systems.circuits.irg.modern_v1_factored_rounded_q_geometry import V1FactoredRoundedQDeviceHandle
from mechanistic_systems.circuits.irg.modern_v1_factored_signed_work_operator import V1FactoredSignedWorkOperator
from mechanistic_systems.circuits.irg.modern_v1_compact_device_structural_source import (
    V1CompactDeviceStructuralSource, V1CompactDeviceStructuralSourceExecutionMode,
)
from mechanistic_systems.circuits.irg.modern_v1_native_fiber_axis import V1NativeDestinationFiberAxisReceipt
from mechanistic_systems.circuits.irg.modern_v1_neutral_runtime import DeclaredRouteAxis
from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask

_PREFIX = "irg-v1-factored-post-D-native-proposal-source-"
_TOKEN = object()


def _require(ok, message):
    if not ok:
        raise ValueError(message)


@dataclass(frozen=True)
class V1FactoredPostDDeviceProposalSourceJoinReceipt:
    coordinates: Mapping[str, object]
    schema: str = "factored_post_D_native_proposal_source_join_v1"
    solver_authority: bool = False
    kkt_authority: bool = False
    production_geometry_authority: bool = False
    population_enclosure_authority: bool = False
    action_authority: bool = False
    d_decision_authority: bool = False
    c_measurement_authority: bool = False
    calibration_k_authority: bool = False
    paired_oracle_o_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    _issuance_token: InitVar[object] = None
    _receipt_hash: str = field(init=False, repr=False)

    def __post_init__(self, _issuance_token):
        _require(_issuance_token is _TOKEN, "native source join receipts are owner-issued")
        object.__setattr__(self, "coordinates", MappingProxyType(dict(self.coordinates)))
        object.__setattr__(self, "_receipt_hash", content_hash(self.to_json_payload(), prefix=_PREFIX+"receipt-"))
        self.assert_integrity()

    def to_json_payload(self):
        return {k: dict(v) if k == "coordinates" else v for k,v in vars(self).items() if not k.startswith("_")}

    @property
    def receipt_hash(self):
        return self._receipt_hash

    def assert_integrity(self):
        _require(self.schema == "factored_post_D_native_proposal_source_join_v1"
                 and all(v is False for k,v in vars(self).items() if k.endswith("_authority")),
                 "native source join authority changed")
        _require(content_hash(self.to_json_payload(), prefix=_PREFIX+"receipt-") == self._receipt_hash,
                 "native source join receipt changed")
        return self._receipt_hash


def _joined(binding, geometry, structural, physical, native, mask):
    _require(type(binding) is V1FactoredDAlphaSourceBindingReceipt
             and type(geometry) is V1FactoredRoundedQDeviceHandle
             and type(structural) is V1CompactDeviceStructuralSource
             and type(physical) is DeclaredRouteAxis
             and type(native) is V1NativeDestinationFiberAxisReceipt
             and type(mask) is HardMask, "native source join requires exact parent types")
    binding_hash = binding.assert_live()
    geometry_hash = geometry.assert_live()
    structural.assert_live_identity()
    shash = structural.receipt.assert_integrity()
    capacity = structural.capacity_receipt
    chash = capacity.assert_integrity()
    native.assert_integrity()
    physical.__post_init__()
    fresh_mask = HardMask(route_axis_hash=mask.route_axis_hash, route_count=mask.route_count,
                          packed_bits=mask.packed_bits).mask_hash
    source, q, st = binding.coordinates, geometry.receipt.coordinates, structural.receipt
    _require(geometry._binding is binding and geometry._handoff is binding._parents[2],
             "rounded Q does not own the supplied registered source")
    _require(source["source_axis_kind"] == "native_declared_physical_coordinates"
             and source["response_endpoint_family_bound"] is False
             and q["upstream_geometry_used_for_rounded_Q"] is False
             and q["diagonal_branch_admitted"] is False,
             "native spectral source law required")
    _require(q["source_binding_hash"] == binding_hash, "rounded Q source binding mismatch")
    for name in ("replay_boundary_hash", "session_design_hash", "finalization_receipt_hash",
                 "packet_receipt_hashes", "look_receipt_hashes", "current_mask_hash",
                 "chart_identity_hash", "proposal_generation", "literal_j_stream_identity_hash",
                 "physical_route_axis_hash", "route_id_axis_hash", "route_count"):
        _require(q[name] == source[name], f"rounded Q {name} differs from registered source")
    _require(source["physical_route_axis_hash"] == physical.route_axis_hash == mask.route_axis_hash
             == native.route_axis_hash == st.route_axis_hash == capacity.route_axis_hash
             and source["route_id_axis_hash"] == physical.route_id_axis_hash == native.route_id_axis_hash
             == st.route_id_axis_hash
             and source["declared_source_axis_hash"] == physical.content_hash == native.route_axis_content_hash
             == st.route_axis_content_hash == capacity.route_axis_content_hash
             and source["route_count"] == physical.route_count == native.route_count == mask.route_count
             == st.route_count == capacity.route_count,
             "physical and ordered route axes differ across native parents")
    _require(source["current_mask_hash"] == fresh_mask == mask.mask_hash == st.current_mask_hash
             == capacity.current_mask_hash, "native parent current masks differ")
    _require(q["native_fiber_axis_hash"] == native.receipt_hash == st.native_fiber_axis_receipt_hash
             and q["native_fiber_measure_hash"] == native.measure_hash,
             "native destination-fiber measure differs")
    expected = tuple((b.block_id,b.offset,b.stop) for b in physical.route_blocks)
    observed = tuple((b.block_id,b.start,b.stop) for b in capacity.block_capacities)
    _require(expected == observed and capacity.block_capacities == structural.block_capacities
             and st.block_count == len(expected), "structural capacity blocks differ from physical declaration")
    _require(q["device"] == st.declared_device, "native Q and structure devices differ")
    private = q["private_cpu_reference_execution"]
    mode = (V1CompactDeviceStructuralSourceExecutionMode.CPU_REFERENCE_TEST_ONLY if private
            else V1CompactDeviceStructuralSourceExecutionMode.CUDA_PRODUCTION)
    _require(st.execution_mode is mode and st.phase_a_structural_input_authority is (not private)
             and st.phase_b_structural_input_authority is (not private),
             "native parent execution scopes differ")
    coordinate = content_hash(dict(source_binding_hash=binding_hash,
        session_design_hash=source["session_design_hash"], finalization_receipt_hash=source["finalization_receipt_hash"],
        registered_prefix_axis_hash=source["registered_prefix_axis_hash"],
        alpha_head_cas_receipt_hash=source["alpha_head_cas_receipt_hash"],
        published_ledger_head_hash=source["published_ledger_head_hash"]), prefix=_PREFIX+"registered-D-coordinate-")
    geom = content_hash(dict(rounded_q_receipt_hash=geometry_hash, structural_source_receipt_hash=shash,
        native_fiber_axis_hash=native.receipt_hash, native_fiber_measure_hash=native.measure_hash,
        current_mask_hash=fresh_mask, branch="spectral"), prefix=_PREFIX+"geometry-")
    registration = content_hash(dict(registered_d_coordinate_hash=coordinate, source_geometry_hash=geom,
        structural_capacity_receipt_hash=chash), prefix=_PREFIX+"registration-")
    return dict(source_binding_hash=binding_hash, rounded_q_receipt_hash=geometry_hash,
        structural_source_receipt_hash=shash, structural_capacity_receipt_hash=chash,
        source_registration_hash=registration, registered_d_coordinate_hash=coordinate, source_geometry_hash=geom,
        physical_route_axis_hash=physical.route_axis_hash, route_id_axis_hash=physical.route_id_axis_hash,
        declared_source_axis_hash=physical.content_hash, route_count=physical.route_count,
        native_fiber_axis_hash=native.receipt_hash, native_fiber_measure_hash=native.measure_hash,
        current_mask_hash=fresh_mask, chart_identity_hash=source["chart_identity_hash"],
        proposal_generation=source["proposal_generation"], session_design_hash=source["session_design_hash"],
        replay_boundary_hash=source["replay_boundary_hash"], finalization_receipt_hash=source["finalization_receipt_hash"],
        corrected_a_stream_identity_hash=source["corrected_a_stream_identity_hash"],
        literal_j_stream_identity_hash=source["literal_j_stream_identity_hash"],
        packet_receipt_hashes=source["packet_receipt_hashes"], look_receipt_hashes=source["look_receipt_hashes"],
        stratum_names=q["stratum_names"], stratum_counts=q["stratum_counts"], stratum_weights=q["stratum_weights"],
        packet_strata=q["packet_strata"], packet_context_spans=q["packet_context_spans"],
        factor_shape=q["factor_shape"], roundoff_residual_spectral_bound=q["roundoff_residual_spectral_bound"],
        quadratic_budget=st.quadratic_budget, structural_registration_hash=st.structural_registration_hash,
        registered_legal_resource_maximum=st.registered_legal_resource_maximum,
        registered_required_resource_value=st.registered_required_resource_value,
        device=q["device"], private_cpu_reference_execution=private,
        admitted_q_branch="spectral", diagonal_branch_admitted=False,
        signed_f_operator_retained=False, response_endpoint_family_bound=False,
        sampled_fd_or_row_bank_source=False, route_scale_d2h_bytes=0)


class V1FactoredPostDDeviceProposalSourceJoin:
    def __init__(self, *, source_binding, geometry_handle, structural_source,
                 physical_route_axis, native_fiber_axis, current_mask):
        self._parents = (source_binding,geometry_handle,structural_source,physical_route_axis,native_fiber_axis,current_mask)
        coordinates = _joined(*self._parents)
        self._receipt = V1FactoredPostDDeviceProposalSourceJoinReceipt(coordinates, _issuance_token=_TOKEN)
        self._receipt_hash = self._receipt.receipt_hash
        self._closed = False

    @property
    def receipt(self):
        return self._receipt

    @property
    def receipt_hash(self):
        return self._receipt_hash

    def assert_live(self):
        _require(not self._closed, "native proposal source join closed")
        _require(self.receipt.assert_integrity() == self._receipt_hash, "native source join receipt replaced")
        _require(_joined(*self._parents) == dict(self.receipt.coordinates), "native proposal source parents changed")
        return self._receipt_hash

    def validate_signed_work(self, operator):
        """Check a later corrected-A owner without retaining it in Phase A."""
        self.assert_live()
        _require(type(operator) is V1FactoredSignedWorkOperator, "exact native signed-work owner required")
        operator_hash = operator.assert_live()
        binding, geometry, _, _, _, _ = self._parents
        _require(operator._binding is binding and operator._handoff is geometry._handoff,
                 "signed-work owner has another registered source")
        a,q = operator.receipt.coordinates, geometry.receipt.coordinates
        _require(a["factorization_kind"] == "factored_packet_stream_corrected_A_legacy_rows"
                 and a["logical_zero_residual"] is True and a["row_spool_receipt_hash"] is None
                 and a["route_scale_fiber_id_array_materialized"] is False
                 and a["arithmetic_law"] == "legacy_FP32_rows_FP64_RN_or_FTZ_gamma_plus_absolute_floor_v1",
                 "signed-work arithmetic or residual semantics differ")
        for name in ("source_binding_hash", "replay_boundary_hash", "finalization_receipt_hash",
                     "packet_receipt_hashes", "look_receipt_hashes", "packet_strata", "packet_context_spans",
                     "context_weights", "stratum_counts", "current_mask_hash", "chart_identity_hash",
                     "proposal_generation", "native_fiber_axis_hash", "native_fiber_measure_hash",
                     "physical_route_axis_hash", "route_id_axis_hash", "route_count", "device",
                     "private_cpu_reference_execution"):
            _require(a[name] == q[name], f"signed-work and rounded Q {name} differ")
        _require(a["corrected_a_stream_identity_hash"] == self.receipt.coordinates["corrected_a_stream_identity_hash"],
                 "signed-work corrected-A stream differs")
        return content_hash(dict(source_join_receipt_hash=self._receipt_hash,
            signed_work_receipt_hash=operator_hash), prefix=_PREFIX+"signed-work-join-")

    def close(self):
        self._closed = True
        self._parents = ()

    def __enter__(self):
        self.assert_live()
        return self

    def __exit__(self, *_exc):
        self.close()

    def __reduce_ex__(self, _protocol):
        raise TypeError("native proposal source joins are nonserializable")
