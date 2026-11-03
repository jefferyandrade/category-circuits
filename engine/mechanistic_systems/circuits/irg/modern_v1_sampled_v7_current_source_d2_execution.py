"""Live current-source D2 execution coordinator for sampled-v7.

This module owns only the narrow, pre-witness D2 graph seam:

``D1/live owner/logical look -> source-issued actions -> D1 compact support
-> shared D/W2 acquisition plan -> block-support destination axis
-> strict fanout plan -> alpha owner binding -> affine-shell preoutcome fanout
-> joint preoutcome registration -> current-source D2 preoutcome plan``.

The separate execution call then performs the physical D/W2 acquisition,
fans the rows into the predeclared block-support destinations, and accepts
those residual sources into the block-support axis.

Everything here is live custody and graph-coordination only.  It deliberately
does not issue a literal-J direction, Loewner envelope, KKT/witness receipt,
D-stability decision, C/K/O authority, mask-commit authority, validation
authority, terminal authority, or Pareto authority.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Final, NoReturn

from mechanistic_systems.circuits.irg.modern_v1_literal_j_joint_affine_shell_family_cuda import (
    register_v1_literal_j_joint_affine_shell_preoutcome,
)
from mechanistic_systems.circuits.irg.modern_v1_sampled_v7_affine_shell_preoutcome_fanout import (
    prepare_v1_sampled_v7_affine_shell_preoutcome_fanout,
)
from mechanistic_systems.circuits.irg.modern_v1_sampled_v7_branch_action_pair import (
    issue_v1_sampled_v7_branch_action_pair,
)
from mechanistic_systems.circuits.irg.modern_v1_sampled_v7_d_alpha_measurement_owner_binding import (
    issue_v1_sampled_v7_d_alpha_measurement_owner_binding,
)
from mechanistic_systems.circuits.irg.modern_v1_sampled_v7_shared_dw2_acquisition import (
    execute_v1_sampled_v7_shared_dw2_acquisition,
    prepare_v1_sampled_v7_shared_dw2_acquisition,
)
from mechanistic_systems.circuits.irg.modern_v1_sampled_v7_shared_dw2_fanout import (
    execute_v1_sampled_v7_shared_dw2_fanout,
    prepare_v1_sampled_v7_shared_dw2_block_support_fanout,
)
from mechanistic_systems.circuits.irg.modern_v1_signed_work_block_support_source_cuda import (
    V1SignedWorkBlockSupportCUDAExecutionDesign,
)
from mechanistic_systems.circuits.irg.modern_v1_signed_work_variable_shell_source_cuda import (
    V1SignedWorkVariableShellSourceCUDAExecutionDesign,
)
from mechanistic_systems.contracts.content_address import content_hash


SAMPLED_V7_CURRENT_SOURCE_D2_EXECUTION_COORDINATOR_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_current_source_D2_execution_coordinator_v1"
)
SAMPLED_V7_CURRENT_SOURCE_D2_EXECUTION_RESULT_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_current_source_D2_execution_result_v1"
)
SAMPLED_V7_CURRENT_SOURCE_D2_EXECUTION_AUTHORITY: Final = (
    "live_current_source_D2_D_W2_acquisition_and_block_support_fanout_custody_only_"
    "no_literal_J_direction_Loewner_KKT_D_stability_C_K_O_commit_validation_terminal_or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-v7-current-source-D2-execution-"


class V1SampledV7CurrentSourceD2ExecutionRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_SOURCE = "stale_source"
    MISSING_SOURCE_ATTRIBUTE = "missing_source_attribute"
    AUTHORITY_VIOLATION = "authority_violation"
    EXECUTION_REPLAY = "execution_replay"
    EXECUTION_FAILURE_CLEANUP_INCOMPLETE = "execution_failure_cleanup_incomplete"
    EVICTED = "evicted"


class V1SampledV7CurrentSourceD2ExecutionRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledV7CurrentSourceD2ExecutionRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7CurrentSourceD2ExecutionRefusalCode:
            raise TypeError("current-source D2 execution refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7CurrentSourceD2ExecutionRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7CurrentSourceD2ExecutionRefusal(code, message)


def _live_hash(value: object, *, name: str) -> str:
    method = getattr(value, "assert_live", None)
    if not callable(method):
        _refuse(
            V1SampledV7CurrentSourceD2ExecutionRefusalCode.INVALID_TYPE,
            f"{name} lacks assert_live()",
        )
    result = method()
    if type(result) is not str or not result:
        _refuse(
            V1SampledV7CurrentSourceD2ExecutionRefusalCode.STALE_SOURCE,
            f"{name} returned a noncanonical live identity",
        )
    return result


def _integrity_hash(value: object, *, name: str) -> str:
    method = getattr(value, "assert_integrity", None)
    if not callable(method):
        _refuse(
            V1SampledV7CurrentSourceD2ExecutionRefusalCode.INVALID_TYPE,
            f"{name} lacks assert_integrity()",
        )
    result = method()
    if type(result) is not str or not result:
        _refuse(
            V1SampledV7CurrentSourceD2ExecutionRefusalCode.STALE_SOURCE,
            f"{name} returned a noncanonical integrity identity",
        )
    return result


def _source_attr(source: object, name: str) -> object:
    if not hasattr(source, name):
        _refuse(
            V1SampledV7CurrentSourceD2ExecutionRefusalCode.MISSING_SOURCE_ATTRIBUTE,
            f"current_source.{name} is required by the public D/W2 coordinator seam",
        )
    value = getattr(source, name)
    if value is None:
        _refuse(
            V1SampledV7CurrentSourceD2ExecutionRefusalCode.MISSING_SOURCE_ATTRIBUTE,
            f"current_source.{name} is None",
        )
    return value


def _exact_designs(
    *,
    variable_shell_design: object,
    block_support_design: object,
) -> None:
    if type(variable_shell_design) is not V1SignedWorkVariableShellSourceCUDAExecutionDesign:
        _refuse(
            V1SampledV7CurrentSourceD2ExecutionRefusalCode.INVALID_TYPE,
            "current-source D2 execution requires an exact variable-shell CUDA execution design",
        )
    if type(block_support_design) is not V1SignedWorkBlockSupportCUDAExecutionDesign:
        _refuse(
            V1SampledV7CurrentSourceD2ExecutionRefusalCode.INVALID_TYPE,
            "current-source D2 execution requires an exact block-support CUDA execution design",
        )


def _hash_attr(value: object, names: tuple[str, ...], *, owner_name: str) -> str:
    for name in names:
        candidate = getattr(value, name, None)
        if type(candidate) is str and candidate:
            return candidate
    receipt = getattr(value, "receipt", None)
    if receipt is not None:
        nested = getattr(receipt, "receipt_hash", None)
        if type(nested) is str and nested:
            return nested
        nested_method = getattr(receipt, "assert_integrity", None)
        if callable(nested_method):
            nested_value = nested_method()
            if type(nested_value) is str and nested_value:
                return nested_value
    _refuse(
        V1SampledV7CurrentSourceD2ExecutionRefusalCode.STALE_SOURCE,
        f"{owner_name} lacks a stable registered hash",
    )


def _safe_evict(value: object | None, *, name: str) -> BaseException | None:
    if value is None:
        return None
    method = getattr(value, "evict", None)
    if not callable(method):
        return None
    try:
        method()
    except BaseException as error:  # pragma: no cover - aggregated by callers.
        error.add_note(f"while evicting {name}")
        return error
    return None


def _abort_and_evict_append_owner(
    *,
    destination: object,
    name: str,
) -> BaseException | None:
    owner = getattr(destination, "append_owner", None)
    if owner is None:
        return None
    try:
        registration = getattr(destination, "append_registration", None)
        if bool(getattr(owner, "has_pending_packet", False)):
            abort = getattr(owner, "abort_pending_packet", None)
            if callable(abort):
                if registration is None:
                    abort()
                else:
                    abort(registration=registration)
        evict = getattr(owner, "evict", None)
        if callable(evict):
            evict()
        source_handle = getattr(destination, "source_handle", None)
        if source_handle is not None and not bool(getattr(source_handle, "is_evicted", False)):
            handle_evict = getattr(source_handle, "evict", None)
            if callable(handle_evict):
                handle_evict()
        bridge = getattr(destination, "block_support_bridge_result", None)
        bridge_handle = getattr(bridge, "source_handle", None)
        if bridge_handle is not None and not bool(getattr(bridge_handle, "is_evicted", False)):
            bridge_evict = getattr(bridge_handle, "evict", None)
            if callable(bridge_evict):
                bridge_evict()
    except BaseException as error:  # pragma: no cover - surfaced as cleanup debt.
        error.add_note(f"while evicting fanout destination {name}")
        return error
    return None


def _evict_fanout_destination_owners(
    fanout_or_plan: object | None,
    *,
    name: str,
) -> list[BaseException]:
    if fanout_or_plan is None:
        return []
    errors: list[BaseException] = []
    destinations = getattr(fanout_or_plan, "destinations", ())
    for index, destination in enumerate(tuple(destinations)):
        error = _abort_and_evict_append_owner(
            destination=destination,
            name=f"{name}[{index}]",
        )
        if error is not None:
            errors.append(error)
    return errors


def _cleanup_prepared_graph(
    *,
    fanout_plan: object | None,
    block_support_axis: object | None,
    d1_compact_support_pair: object | None,
) -> list[BaseException]:
    errors = _evict_fanout_destination_owners(
        fanout_plan,
        name="prepared fanout plan",
    )
    for cleanup_name, value in (
        ("block-support axis", block_support_axis),
        ("D1 compact support pair", d1_compact_support_pair),
    ):
        error = _safe_evict(value, name=cleanup_name)
        if error is not None:
            errors.append(error)
    return errors


def _authority_payload() -> dict[str, bool | str]:
    return {
        "literal_j_direction_authority": False,
        "loewner_envelope_authority": False,
        "kkt_witness_authority": False,
        "d_stability_authority": False,
        "c_authority": False,
        "k_authority": False,
        "o_or_oracle_authority": False,
        "mask_commit_authority": False,
        "validation_authority": False,
        "terminal_or_pareto_authority": False,
        "authority_scope": SAMPLED_V7_CURRENT_SOURCE_D2_EXECUTION_AUTHORITY,
    }


def _coordinator_hash(
    *,
    live_d_context: object,
    d1_prefix: object,
    measurement_owner: object,
    packet_issuance: object,
    action_pair: object,
    d1_compact_support_pair: object,
    acquisition_plan: object,
    block_support_axis: object,
    fanout_plan: object,
    alpha_owner_binding: object,
    affine_shell_fanout_plan: object,
    joint_preoutcome_registration: object,
    d2_preoutcome_plan: object,
) -> str:
    return content_hash(
        {
            "schema": SAMPLED_V7_CURRENT_SOURCE_D2_EXECUTION_COORDINATOR_SCHEMA,
            "authority": SAMPLED_V7_CURRENT_SOURCE_D2_EXECUTION_AUTHORITY,
            "live_d_context_hash": _live_hash(
                live_d_context,
                name="live current-source D context",
            ),
            "d1_prefix_hash": _live_hash(d1_prefix, name="current-source D1 prefix"),
            "measurement_owner_hash": _live_hash(
                measurement_owner,
                name="D-only measurement owner",
            ),
            "packet_issuance_hash": _live_hash(
                packet_issuance,
                name="D2 logical-look issuance",
            ),
            "action_pair_hash": _integrity_hash(
                action_pair,
                name="source-issued branch action pair",
            ),
            "d1_compact_support_pair_hash": _live_hash(
                d1_compact_support_pair,
                name="current-source D1 compact support pair",
            ),
            "acquisition_plan_hash": _hash_attr(
                acquisition_plan,
                ("plan_hash", "receipt_hash", "acquisition_plan_hash"),
                owner_name="shared D/W2 acquisition plan",
            ),
            "block_support_axis_hash": _live_hash(
                block_support_axis,
                name="current-source block-support axis",
            ),
            "fanout_plan_hash": _hash_attr(
                fanout_plan,
                ("plan_hash", "receipt_hash", "fanout_plan_hash"),
                owner_name="shared D/W2 fanout plan",
            ),
            "alpha_owner_binding_hash": _live_hash(
                alpha_owner_binding,
                name="D alpha-owner binding",
            ),
            "affine_shell_fanout_plan_hash": _live_hash(
                affine_shell_fanout_plan,
                name="affine-shell preoutcome fanout plan",
            ),
            "joint_preoutcome_registration_hash": _integrity_hash(
                joint_preoutcome_registration,
                name="joint affine-shell preoutcome registration",
            ),
            "d2_preoutcome_plan_hash": _live_hash(
                d2_preoutcome_plan,
                name="current-source D2 preoutcome plan",
            ),
        },
        prefix=_PREFIX + "coordinator-",
    )


def _frozen_component_hashes(
    *,
    live_d_context: object,
    d1_prefix: object,
    measurement_owner: object,
    packet_issuance: object,
    action_pair: object,
    d1_compact_support_pair: object,
    acquisition_plan: object,
    block_support_axis: object,
    fanout_plan: object,
    alpha_owner_binding: object,
    affine_shell_fanout_plan: object,
    joint_preoutcome_registration: object,
    d2_preoutcome_plan: object,
) -> dict[str, str]:
    return {
        "live_d_context_hash": _live_hash(
            live_d_context,
            name="live current-source D context",
        ),
        "d1_prefix_hash": _live_hash(d1_prefix, name="current-source D1 prefix"),
        "measurement_owner_hash": _live_hash(
            measurement_owner,
            name="D-only measurement owner",
        ),
        "packet_issuance_hash": _live_hash(
            packet_issuance,
            name="D2 logical-look issuance",
        ),
        "action_pair_hash": _integrity_hash(
            action_pair,
            name="source-issued branch action pair",
        ),
        "d1_compact_support_pair_hash": _live_hash(
            d1_compact_support_pair,
            name="current-source D1 compact support pair",
        ),
        "acquisition_plan_hash": _hash_attr(
            acquisition_plan,
            ("plan_hash", "receipt_hash", "acquisition_plan_hash"),
            owner_name="shared D/W2 acquisition plan",
        ),
        "block_support_axis_hash": _live_hash(
            block_support_axis,
            name="current-source block-support axis",
        ),
        "fanout_plan_hash": _hash_attr(
            fanout_plan,
            ("plan_hash", "receipt_hash", "fanout_plan_hash"),
            owner_name="shared D/W2 fanout plan",
        ),
        "alpha_owner_binding_hash": _live_hash(
            alpha_owner_binding,
            name="D alpha-owner binding",
        ),
        "affine_shell_fanout_plan_hash": _live_hash(
            affine_shell_fanout_plan,
            name="affine-shell preoutcome fanout plan",
        ),
        "joint_preoutcome_registration_hash": _integrity_hash(
            joint_preoutcome_registration,
            name="joint affine-shell preoutcome registration",
        ),
        "d2_preoutcome_plan_hash": _live_hash(
            d2_preoutcome_plan,
            name="current-source D2 preoutcome plan",
        ),
    }


class V1SampledV7CurrentSourceD2GraphCoordinator:
    """Prepared live graph for one current-source D2 execution.

    Instances are intentionally mutable, single-use, and nonserializable.
    The caller must supply the exact variable-shell and block-support CUDA
    designs; this coordinator only threads them into the public source seams
    and never turns them into a decision authority.
    """

    __slots__ = (
        "live_d_context",
        "current_source",
        "variable_shell_design",
        "block_support_design",
        "declared_device",
        "d1_prefix",
        "measurement_owner",
        "packet_issuance",
        "action_pair",
        "d1_compact_support_pair",
        "acquisition_plan",
        "block_support_axis",
        "fanout_plan",
        "alpha_owner_binding",
        "affine_shell_fanout_plan",
        "joint_preoutcome_registration",
        "d2_preoutcome_plan",
        "literal_j_direction_authority",
        "loewner_envelope_authority",
        "kkt_witness_authority",
        "d_stability_authority",
        "c_authority",
        "k_authority",
        "o_or_oracle_authority",
        "mask_commit_authority",
        "validation_authority",
        "terminal_or_pareto_authority",
        "authority_scope",
        "_state",
        "_result",
        "_coordinator_hash",
        "_component_hashes",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        live_d_context: object,
        current_source: object,
        variable_shell_design: V1SignedWorkVariableShellSourceCUDAExecutionDesign,
        block_support_design: V1SignedWorkBlockSupportCUDAExecutionDesign,
        declared_device: object,
        d1_prefix: object,
        measurement_owner: object,
        packet_issuance: object,
        action_pair: object,
        d1_compact_support_pair: object,
        acquisition_plan: object,
        block_support_axis: object,
        fanout_plan: object,
        alpha_owner_binding: object,
        affine_shell_fanout_plan: object,
        joint_preoutcome_registration: object,
        d2_preoutcome_plan: object,
    ) -> None:
        _exact_designs(
            variable_shell_design=variable_shell_design,
            block_support_design=block_support_design,
        )
        self.live_d_context = live_d_context
        self.current_source = current_source
        self.variable_shell_design = variable_shell_design
        self.block_support_design = block_support_design
        self.declared_device = declared_device
        self.d1_prefix = d1_prefix
        self.measurement_owner = measurement_owner
        self.packet_issuance = packet_issuance
        self.action_pair = action_pair
        self.d1_compact_support_pair = d1_compact_support_pair
        self.acquisition_plan = acquisition_plan
        self.block_support_axis = block_support_axis
        self.fanout_plan = fanout_plan
        self.alpha_owner_binding = alpha_owner_binding
        self.affine_shell_fanout_plan = affine_shell_fanout_plan
        self.joint_preoutcome_registration = joint_preoutcome_registration
        self.d2_preoutcome_plan = d2_preoutcome_plan
        for name, value in _authority_payload().items():
            setattr(self, name, value)
        self._state = "prepared"
        self._result: V1SampledV7CurrentSourceD2ExecutionResult | None = None
        self._component_hashes = _frozen_component_hashes(
            live_d_context=live_d_context,
            d1_prefix=d1_prefix,
            measurement_owner=measurement_owner,
            packet_issuance=packet_issuance,
            action_pair=action_pair,
            d1_compact_support_pair=d1_compact_support_pair,
            acquisition_plan=acquisition_plan,
            block_support_axis=block_support_axis,
            fanout_plan=fanout_plan,
            alpha_owner_binding=alpha_owner_binding,
            affine_shell_fanout_plan=affine_shell_fanout_plan,
            joint_preoutcome_registration=joint_preoutcome_registration,
            d2_preoutcome_plan=d2_preoutcome_plan,
        )
        self._coordinator_hash = _coordinator_hash(
            live_d_context=live_d_context,
            d1_prefix=d1_prefix,
            measurement_owner=measurement_owner,
            packet_issuance=packet_issuance,
            action_pair=action_pair,
            d1_compact_support_pair=d1_compact_support_pair,
            acquisition_plan=acquisition_plan,
            block_support_axis=block_support_axis,
            fanout_plan=fanout_plan,
            alpha_owner_binding=alpha_owner_binding,
            affine_shell_fanout_plan=affine_shell_fanout_plan,
            joint_preoutcome_registration=joint_preoutcome_registration,
            d2_preoutcome_plan=d2_preoutcome_plan,
        )

    @property
    def coordinator_hash(self) -> str:
        return self._coordinator_hash

    @property
    def is_evicted(self) -> bool:
        return self._state == "evicted"

    @property
    def has_executed(self) -> bool:
        return self._state == "executed"

    def assert_live(self) -> str:
        if self._state == "evicted":
            _refuse(
                V1SampledV7CurrentSourceD2ExecutionRefusalCode.EVICTED,
                "current-source D2 coordinator was evicted",
            )
        if self.live_d_context.current_source is not self.current_source:
            _refuse(
                V1SampledV7CurrentSourceD2ExecutionRefusalCode.STALE_SOURCE,
                "live context current_source identity changed",
            )
        _exact_designs(
            variable_shell_design=self.variable_shell_design,
            block_support_design=self.block_support_design,
        )
        if self._state == "prepared":
            current = _coordinator_hash(
                live_d_context=self.live_d_context,
                d1_prefix=self.d1_prefix,
                measurement_owner=self.measurement_owner,
                packet_issuance=self.packet_issuance,
                action_pair=self.action_pair,
                d1_compact_support_pair=self.d1_compact_support_pair,
                acquisition_plan=self.acquisition_plan,
                block_support_axis=self.block_support_axis,
                fanout_plan=self.fanout_plan,
                alpha_owner_binding=self.alpha_owner_binding,
                affine_shell_fanout_plan=self.affine_shell_fanout_plan,
                joint_preoutcome_registration=self.joint_preoutcome_registration,
                d2_preoutcome_plan=self.d2_preoutcome_plan,
            )
            if current != self._coordinator_hash:
                _refuse(
                    V1SampledV7CurrentSourceD2ExecutionRefusalCode.STALE_SOURCE,
                    "current-source D2 coordinator source graph no longer replays",
                )
        elif self._state == "executed":
            live_checks = {
                "live_d_context_hash": _live_hash(
                    self.live_d_context,
                    name="executed coordinator live context",
                ),
                "d1_compact_support_pair_hash": _live_hash(
                    self.d1_compact_support_pair,
                    name="executed coordinator D1 compact pair",
                ),
                "block_support_axis_hash": _live_hash(
                    self.block_support_axis,
                    name="executed coordinator block-support axis",
                ),
                "acquisition_plan_hash": _hash_attr(
                    self.acquisition_plan,
                    ("plan_hash", "receipt_hash", "acquisition_plan_hash"),
                    owner_name="executed coordinator acquisition plan",
                ),
                "fanout_plan_hash": _hash_attr(
                    self.fanout_plan,
                    ("plan_hash", "receipt_hash", "fanout_plan_hash"),
                    owner_name="executed coordinator fanout plan",
                ),
            }
            for key, value in live_checks.items():
                if value != self._component_hashes[key]:
                    _refuse(
                        V1SampledV7CurrentSourceD2ExecutionRefusalCode.STALE_SOURCE,
                        f"executed coordinator {key} no longer matches frozen graph",
                    )
        else:
            _refuse(
                V1SampledV7CurrentSourceD2ExecutionRefusalCode.STALE_SOURCE,
                "current-source D2 coordinator entered an unknown lifecycle state",
            )
        if any(
            getattr(self, name) is not False
            for name in (
                "literal_j_direction_authority",
                "loewner_envelope_authority",
                "kkt_witness_authority",
                "d_stability_authority",
                "c_authority",
                "k_authority",
                "o_or_oracle_authority",
                "mask_commit_authority",
                "validation_authority",
                "terminal_or_pareto_authority",
            )
        ):
            _refuse(
                V1SampledV7CurrentSourceD2ExecutionRefusalCode.AUTHORITY_VIOLATION,
                "current-source D2 coordinator authority flags were widened",
            )
        if self.authority_scope != SAMPLED_V7_CURRENT_SOURCE_D2_EXECUTION_AUTHORITY:
            _refuse(
                V1SampledV7CurrentSourceD2ExecutionRefusalCode.AUTHORITY_VIOLATION,
                "current-source D2 coordinator authority scope changed",
            )
        return self._coordinator_hash

    def execute(self) -> "V1SampledV7CurrentSourceD2ExecutionResult":
        return execute_v1_sampled_v7_current_source_d2_graph_coordinator(
            coordinator=self,
        )

    def evict(self) -> None:
        if self._state == "evicted":
            return
        errors: list[BaseException] = []
        if self._result is not None:
            error = _safe_evict(self._result, name="current-source D2 execution result")
            if error is not None:
                errors.append(error)
        else:
            errors.extend(
                _evict_fanout_destination_owners(
                    self.fanout_plan,
                    name="coordinator prepared fanout plan",
                )
            )
            for name, value in (
                ("block-support axis", self.block_support_axis),
                ("D1 compact support pair", self.d1_compact_support_pair),
            ):
                error = _safe_evict(value, name=name)
                if error is not None:
                    errors.append(error)
        if errors:
            raise BaseExceptionGroup(
                "current-source D2 coordinator cleanup failed",
                errors,
            )
        self._state = "evicted"

    def require_d_stability(self) -> NoReturn:
        _refuse(
            V1SampledV7CurrentSourceD2ExecutionRefusalCode.AUTHORITY_VIOLATION,
            "current-source D2 execution coordinates raw source custody, not a D-stability decision",
        )

    def require_c_k_o_or_commit(self) -> NoReturn:
        _refuse(
            V1SampledV7CurrentSourceD2ExecutionRefusalCode.AUTHORITY_VIOLATION,
            "current-source D2 execution cannot open C/K/O or commit a mask",
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError("live current-source D2 coordinators are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live current-source D2 coordinators are nonserializable")


class V1SampledV7CurrentSourceD2ExecutionResult:
    """Live result retaining the executed D/W2 rows and residual fanout custody."""

    __slots__ = (
        "coordinator_hash",
        "live_d_context",
        "current_source",
        "d1_compact_support_pair",
        "acquisition",
        "fanout",
        "block_support_axis",
        "block_support_results",
        "d2_preoutcome_plan",
        "literal_j_direction_authority",
        "loewner_envelope_authority",
        "kkt_witness_authority",
        "d_stability_authority",
        "c_authority",
        "k_authority",
        "o_or_oracle_authority",
        "mask_commit_authority",
        "validation_authority",
        "terminal_or_pareto_authority",
        "authority_scope",
        "_state",
        "_result_hash",
        "_frozen_d2_preoutcome_plan_hash",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        coordinator: V1SampledV7CurrentSourceD2GraphCoordinator,
        acquisition: object,
        fanout: object,
        block_support_results: tuple[object, ...],
    ) -> None:
        self.coordinator_hash = coordinator.coordinator_hash
        self.live_d_context = coordinator.live_d_context
        self.current_source = coordinator.current_source
        self.d1_compact_support_pair = coordinator.d1_compact_support_pair
        self.acquisition = acquisition
        self.fanout = fanout
        self.block_support_axis = coordinator.block_support_axis
        self.block_support_results = tuple(block_support_results)
        self.d2_preoutcome_plan = coordinator.d2_preoutcome_plan
        self._frozen_d2_preoutcome_plan_hash = coordinator._component_hashes[
            "d2_preoutcome_plan_hash"
        ]
        for name, value in _authority_payload().items():
            setattr(self, name, value)
        self._state = "live"
        self._result_hash = content_hash(
            {
                "schema": SAMPLED_V7_CURRENT_SOURCE_D2_EXECUTION_RESULT_SCHEMA,
                "authority": SAMPLED_V7_CURRENT_SOURCE_D2_EXECUTION_AUTHORITY,
                "coordinator_hash": self.coordinator_hash,
                "acquisition_hash": _hash_attr(
                    acquisition,
                    ("result_hash", "receipt_hash"),
                    owner_name="shared D/W2 acquisition result",
                ),
                "fanout_hash": _hash_attr(
                    fanout,
                    ("result_hash", "receipt_hash"),
                    owner_name="shared D/W2 fanout result",
                ),
                "block_support_axis_hash": _live_hash(
                    self.block_support_axis,
                    name="completed current-source block-support axis",
                ),
                "block_support_result_count": len(self.block_support_results),
                "d2_preoutcome_plan_hash": self._frozen_d2_preoutcome_plan_hash,
            },
            prefix=_PREFIX + "result-",
        )

    @property
    def result_hash(self) -> str:
        return self._result_hash

    @property
    def is_evicted(self) -> bool:
        return self._state == "evicted"

    def assert_live(self) -> str:
        if self._state == "evicted":
            _refuse(
                V1SampledV7CurrentSourceD2ExecutionRefusalCode.EVICTED,
                "current-source D2 execution result was evicted",
            )
        if self.live_d_context.current_source is not self.current_source:
            _refuse(
                V1SampledV7CurrentSourceD2ExecutionRefusalCode.STALE_SOURCE,
                "result live context current_source identity changed",
            )
        _live_hash(self.live_d_context, name="result live D context")
        _live_hash(
            self.d1_compact_support_pair,
            name="result D1 compact support pair",
        )
        _live_hash(self.block_support_axis, name="result block-support axis")
        for index, result in enumerate(self.block_support_results):
            method = getattr(result, "assert_live", None)
            if callable(method):
                observed = method()
                if type(observed) is not str or not observed:
                    _refuse(
                        V1SampledV7CurrentSourceD2ExecutionRefusalCode.STALE_SOURCE,
                        f"block-support result {index} returned a noncanonical live identity",
                    )
        _hash_attr(
            self.acquisition,
            ("result_hash", "receipt_hash"),
            owner_name="result shared acquisition",
        )
        _hash_attr(
            self.fanout,
            ("result_hash", "receipt_hash"),
            owner_name="result shared fanout",
        )
        if any(
            getattr(self, name) is not False
            for name in (
                "literal_j_direction_authority",
                "loewner_envelope_authority",
                "kkt_witness_authority",
                "d_stability_authority",
                "c_authority",
                "k_authority",
                "o_or_oracle_authority",
                "mask_commit_authority",
                "validation_authority",
                "terminal_or_pareto_authority",
            )
        ):
            _refuse(
                V1SampledV7CurrentSourceD2ExecutionRefusalCode.AUTHORITY_VIOLATION,
                "current-source D2 result authority flags were widened",
            )
        if self.authority_scope != SAMPLED_V7_CURRENT_SOURCE_D2_EXECUTION_AUTHORITY:
            _refuse(
                V1SampledV7CurrentSourceD2ExecutionRefusalCode.AUTHORITY_VIOLATION,
                "current-source D2 result authority scope changed",
            )
        return self._result_hash

    def evict(self) -> None:
        if self._state == "evicted":
            return
        errors: list[BaseException] = []
        errors.extend(
            _evict_fanout_destination_owners(
                self.fanout,
                name="executed fanout result",
            )
        )
        for name, value in (
            ("block-support axis", self.block_support_axis),
            ("D1 compact support pair", self.d1_compact_support_pair),
        ):
            error = _safe_evict(value, name=name)
            if error is not None:
                errors.append(error)
        if errors:
            raise BaseExceptionGroup(
                "current-source D2 result cleanup failed",
                errors,
            )
        self._state = "evicted"

    def require_d_stability(self) -> NoReturn:
        _refuse(
            V1SampledV7CurrentSourceD2ExecutionRefusalCode.AUTHORITY_VIOLATION,
            "current-source D2 execution result is source custody, not a D-stability decision",
        )

    def require_c_k_o_or_commit(self) -> NoReturn:
        _refuse(
            V1SampledV7CurrentSourceD2ExecutionRefusalCode.AUTHORITY_VIOLATION,
            "current-source D2 execution result cannot open C/K/O or commit a mask",
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError("live current-source D2 execution results are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live current-source D2 execution results are nonserializable")


def prepare_v1_sampled_v7_current_source_d2_graph_coordinator(
    *,
    live_d_context: object,
    variable_shell_design: V1SignedWorkVariableShellSourceCUDAExecutionDesign,
    block_support_design: V1SignedWorkBlockSupportCUDAExecutionDesign,
    declared_device: object,
) -> V1SampledV7CurrentSourceD2GraphCoordinator:
    """Prepare the exact current-source D2 graph before provider outcomes exist."""

    _exact_designs(
        variable_shell_design=variable_shell_design,
        block_support_design=block_support_design,
    )
    _live_hash(live_d_context, name="live current-source D context")
    current_source = _source_attr(live_d_context, "current_source")

    d1_compact_support_pair: object | None = None
    block_support_axis: object | None = None
    fanout_plan: object | None = None
    try:
        d1_prefix = live_d_context.derive_current_source_d1_prefix()
        measurement_owner = live_d_context.derive_current_source_d_only_measurement_owner()
        packet_issuance = live_d_context.derive_current_source_d2_logical_look_issuance()
        action_pair = issue_v1_sampled_v7_branch_action_pair(
            source_context=live_d_context,
        )
        d1_compact_support_pair = (
            live_d_context.derive_current_source_d1_compact_support_pair(
                packet_issuance=packet_issuance,
                variable_shell_design=variable_shell_design,
            )
        )

        acquisition_plan = prepare_v1_sampled_v7_shared_dw2_acquisition(
            measurement_owner=measurement_owner,
            population_attachment=d1_compact_support_pair.population_attachment,
            packet_issuance=packet_issuance,
            scorer=_source_attr(current_source, "scorer"),
            route_ids=_source_attr(current_source, "constructive_route_ids"),
            declared_device=declared_device,
            scratch_directory=_source_attr(current_source, "d_w2_scratch_root"),
            prior_row_prefix=live_d_context.row_prefix,
        )

        block_support_axis = live_d_context.derive_current_source_block_support_axis(
            packet_issuance=packet_issuance,
            action_pair=action_pair,
            d1_compact_support_pair=d1_compact_support_pair,
            variable_shell_design=variable_shell_design,
            block_support_design=block_support_design,
        )
        fanout_plan = prepare_v1_sampled_v7_shared_dw2_block_support_fanout(
            acquisition_plan=acquisition_plan,
            action_pair=action_pair,
            spectral_support=d1_compact_support_pair.branch_bindings[0],
            diagonal_l1_support=d1_compact_support_pair.branch_bindings[1],
            design=variable_shell_design,
            block_support_destination_axis=block_support_axis.destination_bindings,
        )
        alpha_owner_binding = issue_v1_sampled_v7_d_alpha_measurement_owner_binding(
            alpha_family_plan=_source_attr(current_source, "alpha_family_plan"),
            alpha_family_registry=_source_attr(current_source, "alpha_family_registry"),
            alpha_head_cas=_source_attr(current_source, "alpha_head_cas"),
            measurement_owner=measurement_owner,
            ordinal_physical_axis_bridge=_source_attr(
                current_source,
                "ordinal_physical_axis_bridge",
            ),
            registered_look_index=acquisition_plan.logical_registered_look_index,
            registered_look_stratum_counts=acquisition_plan.logical_registered_look_stratum_counts,
        )
        affine_shell_fanout_plan = prepare_v1_sampled_v7_affine_shell_preoutcome_fanout(
            base_plan=fanout_plan,
            literal_j_owner_binding=alpha_owner_binding,
        )
        joint_preoutcome_registration = register_v1_literal_j_joint_affine_shell_preoutcome(
            fanout_plan=affine_shell_fanout_plan,
        )
        d2_preoutcome_plan = live_d_context.derive_current_source_d2_preoutcome_plan(
            alpha_owner_binding=alpha_owner_binding,
            affine_shell_fanout_plan=affine_shell_fanout_plan,
            joint_preoutcome_registration=joint_preoutcome_registration,
        )
        coordinator = V1SampledV7CurrentSourceD2GraphCoordinator(
            live_d_context=live_d_context,
            current_source=current_source,
            variable_shell_design=variable_shell_design,
            block_support_design=block_support_design,
            declared_device=declared_device,
            d1_prefix=d1_prefix,
            measurement_owner=measurement_owner,
            packet_issuance=packet_issuance,
            action_pair=action_pair,
            d1_compact_support_pair=d1_compact_support_pair,
            acquisition_plan=acquisition_plan,
            block_support_axis=block_support_axis,
            fanout_plan=fanout_plan,
            alpha_owner_binding=alpha_owner_binding,
            affine_shell_fanout_plan=affine_shell_fanout_plan,
            joint_preoutcome_registration=joint_preoutcome_registration,
            d2_preoutcome_plan=d2_preoutcome_plan,
        )
    except BaseException as primary_error:
        cleanup_errors = _cleanup_prepared_graph(
            fanout_plan=fanout_plan,
            block_support_axis=block_support_axis,
            d1_compact_support_pair=d1_compact_support_pair,
        )
        if cleanup_errors:
            primary_error.add_note(
                "current-source D2 preparation cleanup also failed: "
                + "; ".join(
                    f"{type(error).__name__}: {error}" for error in cleanup_errors
                )
            )
        raise

    return coordinator


def execute_v1_sampled_v7_current_source_d2_graph_coordinator(
    *,
    coordinator: V1SampledV7CurrentSourceD2GraphCoordinator,
) -> V1SampledV7CurrentSourceD2ExecutionResult:
    """Execute one prepared current-source D2 graph exactly once."""

    if type(coordinator) is not V1SampledV7CurrentSourceD2GraphCoordinator:
        _refuse(
            V1SampledV7CurrentSourceD2ExecutionRefusalCode.INVALID_TYPE,
            "execution requires an exact current-source D2 graph coordinator",
        )
    if coordinator._state != "prepared":
        _refuse(
            V1SampledV7CurrentSourceD2ExecutionRefusalCode.EXECUTION_REPLAY,
            "current-source D2 graph coordinators execute exactly once",
        )
    coordinator.assert_live()
    try:
        fanout = None
        acquisition = execute_v1_sampled_v7_shared_dw2_acquisition(
            coordinator.acquisition_plan,
        )
        fanout = execute_v1_sampled_v7_shared_dw2_fanout(
            plan=coordinator.fanout_plan,
            acquisition=acquisition,
        )
        block_support_results = coordinator.block_support_axis.accept_residual_fanout(
            fanout,
        )
        result = V1SampledV7CurrentSourceD2ExecutionResult(
            coordinator=coordinator,
            acquisition=acquisition,
            fanout=fanout,
            block_support_results=block_support_results,
        )
        coordinator._result = result
        coordinator._state = "executed"
        return result
    except BaseException as primary_error:
        primary_error.add_note("current-source D2 execution did not produce authority")
        if "fanout" in locals() and fanout is not None:
            for cleanup_error in _evict_fanout_destination_owners(
                fanout,
                name="failed execution fanout result",
            ):
                primary_error.add_note(
                    "current-source D2 fanout destination cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        try:
            coordinator.evict()
        except BaseException as cleanup_error:
            primary_error.add_note(
                "current-source D2 execution cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}",
            )
        raise


__all__ = [
    "SAMPLED_V7_CURRENT_SOURCE_D2_EXECUTION_AUTHORITY",
    "SAMPLED_V7_CURRENT_SOURCE_D2_EXECUTION_COORDINATOR_SCHEMA",
    "SAMPLED_V7_CURRENT_SOURCE_D2_EXECUTION_RESULT_SCHEMA",
    "V1SampledV7CurrentSourceD2ExecutionRefusal",
    "V1SampledV7CurrentSourceD2ExecutionRefusalCode",
    "V1SampledV7CurrentSourceD2ExecutionResult",
    "V1SampledV7CurrentSourceD2GraphCoordinator",
    "execute_v1_sampled_v7_current_source_d2_graph_coordinator",
    "prepare_v1_sampled_v7_current_source_d2_graph_coordinator",
]
