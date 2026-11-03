"""Content-addressed deployment authority for a completed glocal-v3 campaign.

Sequential proposal receipts and a terminal parity certificate are useful only
if the exact hard mask that is later exported is bound to them.  This module
can build a detached receipt from a :class:`CampaignEvidenceState`, but only
the physical sampled-path runtime may mint the nonserializable live capability
after binding its backend, masks, proposal executions, resources, and terminal
evidence.

The certificate deliberately says ``shadow``.  It proves conformance to the
current glocal-v3 runtime contract, not external promotion of that contract.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Mapping

from .campaign import CampaignEvidenceState
from .glocal_contracts import (
    EVIDENCE_FOUNDATION_GLOCAL_V3,
    JOINT_RISK_QUERY_ID,
    TerminalParityCertificate,
    content_hash,
)


GLOCAL_DEPLOYMENT_CERTIFICATE_SCHEMA = "glocal_deployment_mask_certificate_v2"
_LIVE_AUTHORITY_TOKEN = object()


def _require_sha256(value: str, label: str) -> str:
    identity = str(value)
    if len(identity) != 64 or any(
        character not in "0123456789abcdef" for character in identity
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return identity


def _require_identity(value: str, label: str) -> str:
    identity = str(value).strip()
    if not identity:
        raise ValueError(f"{label} must be nonempty")
    return identity


@dataclass(frozen=True, slots=True)
class GlocalAcceptedTransition:
    """One accepted state/calibration transition in campaign order."""

    proposal_id: str
    registration_index: int
    registration_hash: str
    decision_receipt_hash: str
    parent_state_hash: str
    candidate_state_hash: str
    parent_calibration_hash: str
    candidate_calibration_hash: str
    distortion_debit: float
    schema_name: str = "glocal_accepted_transition_v1"

    def __post_init__(self) -> None:
        _require_identity(self.proposal_id, "accepted proposal ID")
        if int(self.registration_index) <= 0:
            raise ValueError("accepted transition index must be positive")
        for value, label in (
            (self.registration_hash, "accepted registration hash"),
            (self.decision_receipt_hash, "accepted decision receipt hash"),
            (self.parent_state_hash, "accepted parent state hash"),
            (self.candidate_state_hash, "accepted candidate state hash"),
            (self.parent_calibration_hash, "accepted parent calibration hash"),
            (
                self.candidate_calibration_hash,
                "accepted candidate calibration hash",
            ),
        ):
            _require_sha256(value, label)
        debit = float(self.distortion_debit)
        if not math.isfinite(debit) or debit < 0.0:
            raise ValueError("accepted transition debit must be finite and nonnegative")
        object.__setattr__(self, "registration_index", int(self.registration_index))
        object.__setattr__(self, "distortion_debit", debit)

    @property
    def content_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class GlocalDeploymentMaskCertificate:
    """Exact mask binding for sequential safety plus terminal faithfulness."""

    certificate_id: str
    campaign_id: str
    campaign_receipt_hash: str
    risk_estimand_hash: str
    prediction_rule_hash: str
    context_population_hash: str
    query_id: str
    context_design_hash: str
    initial_state_hash: str
    final_state_hash: str
    initial_calibration_hash: str
    final_calibration_hash: str
    native_route_axis_hash: str
    native_mask_hash: str
    accepted_transitions: tuple[GlocalAcceptedTransition, ...]
    decision_receipt_hashes: tuple[str, ...]
    sampling_design_hashes: tuple[str, ...]
    distortion_budget: float
    distortion_spent: float
    terminal_precommitment_hash: str
    terminal_population_hash: str
    terminal_lock_hash: str
    terminal_opened_lock_hash: str
    terminal_certificate_hash: str
    terminal_audit_result_hash: str
    terminal_audit_evidence_receipt_hash: str
    terminal_sampling_design_hash: str
    terminal_audit_observation_set_hash: str
    terminal_upper_bound: float
    terminal_tolerance: float
    terminal_method: str
    native_mask_bridge_hashes: tuple[str, ...] = ()
    irg_final_state_hash: str | None = None
    irg_final_mask_hash: str | None = None
    external_scientific_identity_hash: str | None = None
    external_query_measure_hash: str | None = None
    foundation: str = EVIDENCE_FOUNDATION_GLOCAL_V3
    authority_status: str = "shadow"
    schema_name: str = GLOCAL_DEPLOYMENT_CERTIFICATE_SCHEMA

    def __post_init__(self) -> None:
        for value, label in (
            (self.certificate_id, "deployment certificate ID"),
            (self.campaign_id, "deployment campaign ID"),
            (self.query_id, "deployment query ID"),
            (self.native_route_axis_hash, "native route-axis identity"),
            (self.terminal_method, "terminal method"),
        ):
            _require_identity(value, label)
        for value, label in (
            (self.campaign_receipt_hash, "campaign receipt hash"),
            (self.risk_estimand_hash, "risk estimand hash"),
            (self.prediction_rule_hash, "prediction rule hash"),
            (self.context_population_hash, "context population hash"),
            (self.context_design_hash, "context design hash"),
            (self.initial_state_hash, "initial state hash"),
            (self.final_state_hash, "final state hash"),
            (self.initial_calibration_hash, "initial calibration hash"),
            (self.final_calibration_hash, "final calibration hash"),
            (self.native_mask_hash, "native mask hash"),
            (
                self.terminal_precommitment_hash,
                "terminal precommitment hash",
            ),
            (self.terminal_population_hash, "terminal population hash"),
            (self.terminal_lock_hash, "terminal validation-lock hash"),
            (self.terminal_opened_lock_hash, "opened terminal lock hash"),
            (self.terminal_certificate_hash, "terminal certificate hash"),
            (self.terminal_audit_result_hash, "terminal audit result hash"),
            (
                self.terminal_audit_evidence_receipt_hash,
                "terminal audit evidence receipt hash",
            ),
            (
                self.terminal_sampling_design_hash,
                "terminal sampling design hash",
            ),
            (
                self.terminal_audit_observation_set_hash,
                "terminal audit observation-set hash",
            ),
        ):
            _require_sha256(value, label)
        if self.foundation != EVIDENCE_FOUNDATION_GLOCAL_V3:
            raise ValueError("deployment certificate has a foreign foundation")
        if self.authority_status != "shadow":
            raise ValueError(
                "unpromoted glocal deployment authority must remain shadow"
            )
        if self.schema_name != GLOCAL_DEPLOYMENT_CERTIFICATE_SCHEMA:
            raise ValueError("unsupported glocal deployment certificate schema")
        transitions = tuple(self.accepted_transitions)
        if any(not isinstance(item, GlocalAcceptedTransition) for item in transitions):
            raise TypeError(
                "deployment accepted transitions must use the typed contract"
            )
        indices = tuple(item.registration_index for item in transitions)
        if tuple(sorted(indices)) != indices or len(set(indices)) != len(indices):
            raise ValueError(
                "deployment accepted transitions are not in unique campaign order"
            )
        expected_state = self.initial_state_hash
        expected_calibration = self.initial_calibration_hash
        for transition in transitions:
            if transition.parent_state_hash != expected_state:
                raise ValueError("deployment state transition chain is discontinuous")
            if transition.parent_calibration_hash != expected_calibration:
                raise ValueError(
                    "deployment calibration transition chain is discontinuous"
                )
            expected_state = transition.candidate_state_hash
            expected_calibration = transition.candidate_calibration_hash
        if expected_state != self.final_state_hash:
            raise ValueError(
                "deployment transition chain does not reach the final state"
            )
        if expected_calibration != self.final_calibration_hash:
            raise ValueError(
                "deployment transition chain does not reach the final calibration"
            )
        decision_hashes = tuple(str(value) for value in self.decision_receipt_hashes)
        sampling_hashes = tuple(str(value) for value in self.sampling_design_hashes)
        bridge_hashes = tuple(str(value) for value in self.native_mask_bridge_hashes)
        for values, label, allow_empty in (
            (decision_hashes, "decision receipt hash", True),
            (sampling_hashes, "sampling design hash", True),
            (bridge_hashes, "native mask bridge hash", True),
        ):
            if not allow_empty and not values:
                raise ValueError(f"deployment requires at least one {label}")
            if len(set(values)) != len(values):
                raise ValueError(f"deployment {label} values must be unique")
            for value in values:
                _require_sha256(value, label)
        accepted_receipts = {item.decision_receipt_hash for item in transitions}
        if not accepted_receipts.issubset(set(decision_hashes)):
            raise ValueError(
                "accepted transition receipts are absent from the campaign receipt axis"
            )
        budget = float(self.distortion_budget)
        spent = float(self.distortion_spent)
        if (
            not math.isfinite(budget)
            or not math.isfinite(spent)
            or budget < 0.0
            or spent < 0.0
            or spent > budget + 1e-12
        ):
            raise ValueError("deployment distortion ledger is invalid")
        transition_spent = math.fsum(item.distortion_debit for item in transitions)
        if not math.isclose(transition_spent, spent, abs_tol=1e-12):
            raise ValueError(
                "deployment accepted debits do not reconcile to campaign spend"
            )
        upper = float(self.terminal_upper_bound)
        tolerance = float(self.terminal_tolerance)
        if (
            not math.isfinite(upper)
            or not math.isfinite(tolerance)
            or not 0.0 <= upper <= 1.0
            or not 0.0 <= tolerance <= 1.0
            or upper > tolerance
        ):
            raise ValueError(
                "deployment requires a passing terminal parity certificate"
            )
        paired_optional = (
            (self.irg_final_state_hash, "IRG final state identity"),
            (self.irg_final_mask_hash, "IRG final mask identity"),
        )
        if (paired_optional[0][0] is None) != (paired_optional[1][0] is None):
            raise ValueError(
                "IRG deployment state and mask identities must be supplied together"
            )
        for value, label in paired_optional:
            if value is not None:
                _require_identity(value, label)
        external_pair = (
            (
                self.external_scientific_identity_hash,
                "external scientific identity",
            ),
            (self.external_query_measure_hash, "external query-measure identity"),
        )
        if (self.external_scientific_identity_hash is None) != (
            self.external_query_measure_hash is None
        ):
            raise ValueError(
                "external scientific and query-measure identities must be "
                "supplied together"
            )
        for value, label in external_pair:
            if value is not None:
                _require_identity(value, label)
        object.__setattr__(self, "accepted_transitions", transitions)
        object.__setattr__(self, "decision_receipt_hashes", decision_hashes)
        object.__setattr__(self, "sampling_design_hashes", sampling_hashes)
        object.__setattr__(self, "native_mask_bridge_hashes", bridge_hashes)
        object.__setattr__(self, "distortion_budget", budget)
        object.__setattr__(self, "distortion_spent", spent)
        object.__setattr__(self, "terminal_upper_bound", upper)
        object.__setattr__(self, "terminal_tolerance", tolerance)

    @property
    def content_hash(self) -> str:
        return content_hash(self)

    def validate_mask_binding(
        self,
        *,
        native_route_axis_hash: str,
        native_mask_hash: str,
        query_id: str | None = None,
        external_scientific_identity_hash: str | None = None,
        external_query_measure_hash: str | None = None,
    ) -> None:
        """Fail closed if a consumer attempts to deploy a different object."""

        if str(native_route_axis_hash) != self.native_route_axis_hash:
            raise ValueError("deployment certificate route axis differs")
        if str(native_mask_hash) != self.native_mask_hash:
            raise ValueError("deployment certificate hard mask differs")
        if query_id is not None and str(query_id) != self.query_id:
            raise ValueError("deployment certificate query differs")
        for label, expected, observed in (
            (
                "scientific identity",
                external_scientific_identity_hash,
                self.external_scientific_identity_hash,
            ),
            (
                "query measure",
                external_query_measure_hash,
                self.external_query_measure_hash,
            ),
        ):
            if expected is not None and str(expected) != observed:
                raise ValueError(f"deployment certificate {label} differs")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["content_hash"] = self.content_hash
        return payload

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "GlocalDeploymentMaskCertificate":
        data = dict(payload)
        expected_hash = data.pop("content_hash", None)
        transitions = tuple(
            (
                item
                if isinstance(item, GlocalAcceptedTransition)
                else GlocalAcceptedTransition(**dict(item))
            )
            for item in data.get("accepted_transitions", ())
        )
        data["accepted_transitions"] = transitions
        for key in (
            "decision_receipt_hashes",
            "sampling_design_hashes",
            "native_mask_bridge_hashes",
        ):
            data[key] = tuple(str(value) for value in data.get(key, ()))
        certificate = cls(**data)
        if expected_hash is not None and str(expected_hash) != certificate.content_hash:
            raise ValueError("glocal deployment certificate content hash mismatch")
        return certificate


class GlocalDeploymentAuthorityHandle:
    """Nonserializable capability backed by one live physical runtime graph."""

    __slots__ = (
        "_runtime_owner",
        "_backend_owner",
        "_campaign",
        "_certificate",
        "_build_arguments",
        "_physical_binding_hash",
        "_physical_revalidator",
    )

    def __init__(
        self,
        *,
        runtime_owner: object,
        backend_owner: object,
        campaign: CampaignEvidenceState,
        certificate: GlocalDeploymentMaskCertificate,
        build_arguments: Mapping[str, Any],
        physical_binding_hash: str,
        physical_revalidator: Any,
        _token: object,
    ) -> None:
        if _token is not _LIVE_AUTHORITY_TOKEN:
            raise TypeError(
                "live glocal deployment authority must be issued from the "
                "physical sampled-path runtime"
            )
        if runtime_owner is None or backend_owner is None:
            raise TypeError(
                "live glocal deployment authority requires runtime/backend owners"
            )
        _require_sha256(physical_binding_hash, "physical runtime binding hash")
        if not callable(physical_revalidator):
            raise TypeError(
                "live glocal deployment authority requires a physical revalidator"
            )
        self._runtime_owner = runtime_owner
        self._backend_owner = backend_owner
        self._campaign = campaign
        self._certificate = certificate
        self._build_arguments = MappingProxyType(dict(build_arguments))
        self._physical_binding_hash = str(physical_binding_hash)
        self._physical_revalidator = physical_revalidator

    def __setattr__(self, name: str, value: Any) -> None:
        if hasattr(self, name):
            raise AttributeError("live glocal deployment authority is immutable")
        object.__setattr__(self, name, value)

    @property
    def certificate(self) -> GlocalDeploymentMaskCertificate:
        return self._certificate

    @property
    def physical_binding_hash(self) -> str:
        return self._physical_binding_hash

    def require_current(self) -> GlocalDeploymentMaskCertificate:
        """Revalidate physical execution and rebuild the campaign receipt graph."""

        current = self._physical_revalidator(
            self._runtime_owner,
            self._backend_owner,
        )
        if not isinstance(current, Mapping):
            raise RuntimeError(
                "physical runtime revalidator returned no typed binding payload"
            )
        current_hash = content_hash(
            {
                "schema": "glocal_physical_runtime_binding_v1",
                "payload": dict(current),
            }
        )
        if current_hash != self._physical_binding_hash:
            raise RuntimeError(
                "live glocal deployment authority is stale relative to its "
                "physical runtime/backend graph"
            )

        recomputed = build_glocal_deployment_mask_certificate(
            campaign=self._campaign,
            **self._build_arguments,
        )
        if recomputed.content_hash != self._certificate.content_hash:
            raise RuntimeError(
                "live glocal deployment authority is stale relative to its campaign"
            )
        return self._certificate

    def __reduce__(self):
        raise TypeError("live glocal deployment authority cannot be serialized")

    def __reduce_ex__(self, protocol):
        del protocol
        raise TypeError("live glocal deployment authority cannot be serialized")

    def __copy__(self):
        raise TypeError("live glocal deployment authority cannot be copied")

    def __deepcopy__(self, memo):
        del memo
        raise TypeError("live glocal deployment authority cannot be copied")


def build_glocal_deployment_mask_certificate(
    *,
    campaign: CampaignEvidenceState,
    query_id: str,
    context_design_hash: str,
    native_route_axis_hash: str,
    native_mask_hash: str,
    terminal_audit_result_hash: str,
    terminal_sampling_design_hash: str,
    native_mask_bridge_hashes: tuple[str, ...] = (),
    irg_final_state_hash: str | None = None,
    irg_final_mask_hash: str | None = None,
    external_scientific_identity_hash: str | None = None,
    external_query_measure_hash: str | None = None,
    certificate_id: str | None = None,
) -> GlocalDeploymentMaskCertificate:
    """Build a detached shadow receipt from a terminally locked campaign.

    This mathematical builder cannot establish that Torch performed proposal
    adjudication or the terminal audit.  Its serialized result supports replay
    and inspection; possession of it never authorizes deployment.
    """

    campaign_receipt = campaign.receipt()
    if campaign_receipt.get("foundation") != EVIDENCE_FOUNDATION_GLOCAL_V3:
        raise ValueError("deployment requires a glocal-v3 campaign")
    registrations = tuple(campaign_receipt["proposal_registrations"])
    sampling_design_rows = tuple(campaign_receipt.get("sampling_designs", ()))
    design_hash_by_proposal = {
        str(item["proposal_id"]): str(item["sampling_design_hash"])
        for item in sampling_design_rows
    }
    if len(design_hash_by_proposal) != len(sampling_design_rows) or set(
        design_hash_by_proposal
    ) != {str(item["registration"]["proposal_id"]) for item in registrations}:
        raise RuntimeError(
            "campaign receipt does not retain one sampling design per proposal"
        )
    for item in registrations:
        registration = dict(item["registration"])
        proposal_id = str(registration["proposal_id"])
        if design_hash_by_proposal[proposal_id] != str(
            registration["sampling_design_hash"]
        ):
            raise RuntimeError(
                "campaign proposal registration and retained sampling design differ"
            )
    live = tuple(
        str(item["registration"]["proposal_id"])
        for item in registrations
        if item["status"] == "registered"
    )
    if live:
        raise RuntimeError(
            "deployment cannot be certified while proposals remain live: "
            + ",".join(live)
        )
    certificate: TerminalParityCertificate | None = campaign.terminal_certificate
    precommitment = campaign.terminal_precommitment
    audit_evidence = campaign.terminal_audit_evidence_receipt
    lock = campaign.terminal_lock
    if (
        precommitment is None
        or certificate is None
        or audit_evidence is None
        or not certificate.certified
    ):
        raise RuntimeError(
            "deployment requires passing typed one-shot terminal audit evidence"
        )
    if lock is None or not lock.audit_opened:
        raise RuntimeError("deployment requires the terminal lock to be opened")
    if campaign_receipt.get("terminal_audit_failure") is not None:
        raise RuntimeError("deployment refuses a failed terminal audit")
    registered_lock_hash = str(campaign_receipt["terminal_lock_hash"])
    opened_lock_hash = str(campaign_receipt["terminal_opened_lock_hash"])
    if certificate.validation_lock_hash != registered_lock_hash:
        raise RuntimeError("terminal certificate and campaign lock differ")
    if (
        lock.terminal_precommitment_hash != precommitment.content_hash
        or lock.terminal_population_hash
        != precommitment.estimand.terminal_population_hash
        or audit_evidence.terminal_precommitment_hash != precommitment.content_hash
    ):
        raise RuntimeError(
            "terminal lock/evidence differ from the campaign-birth precommitment"
        )
    if certificate.content_hash != campaign_receipt["terminal_certificate_hash"]:
        raise RuntimeError("terminal certificate and campaign receipt differ")
    if (
        audit_evidence.certificate.content_hash != certificate.content_hash
        or audit_evidence.content_hash
        != campaign_receipt["terminal_audit_evidence_receipt_hash"]
    ):
        raise RuntimeError("terminal physical evidence and campaign receipt differ")
    if audit_evidence.sampling_design_hash != str(terminal_sampling_design_hash):
        raise RuntimeError(
            "terminal sampling design differs from the physical audit receipt"
        )
    if certificate.final_state_hash != campaign.current_state_hash:
        raise RuntimeError("terminal certificate does not bind the current state")

    receipts_by_hash = {receipt.content_hash: receipt for receipt in campaign.receipts}
    accepted_receipts = {
        receipt.proposal_id: receipt
        for receipt in campaign.receipts
        if receipt.outcome == "accepted"
    }
    transitions: list[GlocalAcceptedTransition] = []
    sampling_design_hashes: list[str] = []
    for item in registrations:
        registration = dict(item["registration"])
        sampling_hash = str(registration["sampling_design_hash"])
        if sampling_hash not in sampling_design_hashes:
            sampling_design_hashes.append(sampling_hash)
        if item["status"] != "accepted":
            continue
        proposal_id = str(registration["proposal_id"])
        receipt = accepted_receipts.get(proposal_id)
        if receipt is None:
            raise RuntimeError(
                "accepted campaign proposal lacks its accepted decision receipt"
            )
        registration_hash = str(item["registration_hash"])
        if receipt.proposal_registration_hash != registration_hash:
            raise RuntimeError(
                "accepted campaign registration and decision receipt differ"
            )
        transitions.append(
            GlocalAcceptedTransition(
                proposal_id=proposal_id,
                registration_index=int(registration["registration_index"]),
                registration_hash=registration_hash,
                decision_receipt_hash=receipt.content_hash,
                parent_state_hash=str(registration["parent_state_hash"]),
                candidate_state_hash=str(registration["candidate_state_hash"]),
                parent_calibration_hash=str(registration["parent_calibration_hash"]),
                candidate_calibration_hash=str(
                    registration["candidate_calibration_hash"]
                ),
                distortion_debit=float(receipt.distortion_debit),
            )
        )
    decision_hashes = tuple(
        str(value) for value in campaign_receipt["decision_receipt_hashes"]
    )
    if set(decision_hashes) != set(receipts_by_hash):
        raise RuntimeError("campaign receipt does not enumerate its decision receipts")
    risk = campaign.risk_estimand
    query_weights = dict(risk.query_weights)
    expected_query_scope = (
        next(iter(query_weights)) if len(query_weights) == 1 else JOINT_RISK_QUERY_ID
    )
    if str(query_id) != expected_query_scope:
        raise ValueError(
            "deployment query scope differs from the complete campaign risk"
        )
    return GlocalDeploymentMaskCertificate(
        certificate_id=(
            str(certificate_id)
            if certificate_id is not None
            else f"{campaign.campaign_id}:deployment"
        ),
        campaign_id=campaign.campaign_id,
        campaign_receipt_hash=str(campaign_receipt["campaign_hash"]),
        risk_estimand_hash=risk.content_hash,
        prediction_rule_hash=risk.prediction_rule.content_hash,
        context_population_hash=risk.context_population_hash,
        query_id=str(query_id),
        context_design_hash=_require_sha256(context_design_hash, "context design hash"),
        initial_state_hash=campaign.initial_state_hash,
        final_state_hash=campaign.current_state_hash,
        initial_calibration_hash=campaign.initial_calibration_hash,
        final_calibration_hash=campaign.current_calibration_hash,
        native_route_axis_hash=str(native_route_axis_hash),
        native_mask_hash=_require_sha256(native_mask_hash, "native mask hash"),
        accepted_transitions=tuple(transitions),
        decision_receipt_hashes=decision_hashes,
        sampling_design_hashes=tuple(sampling_design_hashes),
        distortion_budget=campaign.distortion_budget,
        distortion_spent=campaign.distortion_spent,
        terminal_precommitment_hash=precommitment.content_hash,
        terminal_population_hash=precommitment.estimand.terminal_population_hash,
        terminal_lock_hash=registered_lock_hash,
        terminal_opened_lock_hash=opened_lock_hash,
        terminal_certificate_hash=certificate.content_hash,
        terminal_audit_result_hash=_require_sha256(
            terminal_audit_result_hash,
            "terminal audit result hash",
        ),
        terminal_audit_evidence_receipt_hash=audit_evidence.content_hash,
        terminal_sampling_design_hash=_require_sha256(
            terminal_sampling_design_hash,
            "terminal sampling design hash",
        ),
        terminal_audit_observation_set_hash=(certificate.audit_observation_set_hash),
        terminal_upper_bound=certificate.upper_bound,
        terminal_tolerance=certificate.tolerance,
        terminal_method=certificate.method,
        native_mask_bridge_hashes=tuple(native_mask_bridge_hashes),
        irg_final_state_hash=irg_final_state_hash,
        irg_final_mask_hash=irg_final_mask_hash,
        external_scientific_identity_hash=external_scientific_identity_hash,
        external_query_measure_hash=external_query_measure_hash,
    )


def build_live_glocal_deployment_authority(
    *,
    campaign: CampaignEvidenceState,
    query_id: str,
    context_design_hash: str,
    native_route_axis_hash: str,
    native_mask_hash: str,
    terminal_audit_result_hash: str,
    terminal_sampling_design_hash: str,
    native_mask_bridge_hashes: tuple[str, ...] = (),
    irg_final_state_hash: str | None = None,
    irg_final_mask_hash: str | None = None,
    external_scientific_identity_hash: str | None = None,
    external_query_measure_hash: str | None = None,
    certificate_id: str | None = None,
) -> GlocalDeploymentAuthorityHandle:
    """Refuse the obsolete campaign-only live-authority construction."""

    del (
        campaign,
        query_id,
        context_design_hash,
        native_route_axis_hash,
        native_mask_hash,
        terminal_audit_result_hash,
        terminal_sampling_design_hash,
        native_mask_bridge_hashes,
        irg_final_state_hash,
        irg_final_mask_hash,
        external_scientific_identity_hash,
        external_query_measure_hash,
        certificate_id,
    )
    raise RuntimeError(
        "campaign-only live deployment authority is disabled; authority must "
        "be issued by GlocalBrierCampaignRuntime after physical execution"
    )


def _issue_physical_glocal_deployment_authority(
    *,
    runtime_owner: object,
    backend_owner: object,
    campaign: CampaignEvidenceState,
    query_id: str,
    context_design_hash: str,
    native_route_axis_hash: str,
    native_mask_hash: str,
    terminal_audit_result_hash: str,
    terminal_sampling_design_hash: str,
    native_mask_bridge_hashes: tuple[str, ...] = (),
    irg_final_state_hash: str | None = None,
    irg_final_mask_hash: str | None = None,
    external_scientific_identity_hash: str | None = None,
    external_query_measure_hash: str | None = None,
    certificate_id: str | None = None,
) -> GlocalDeploymentAuthorityHandle:
    """Private issuer for the physical sampled-path runtime."""

    # Import lazily to avoid the deployment/contracts -> runtime import cycle.
    # Exact nominal types are intentional: a caller-assembled campaign or a
    # lookalike object must never mint the live capability.
    from mechanistic_systems.circuits.evidence_glocal import (
        GlocalBrierCampaignRuntime,
    )
    from mechanistic_systems.circuits.sampled_torch import (
        TorchSampledPathEvaluationBackend,
    )

    if type(runtime_owner) is not GlocalBrierCampaignRuntime:
        raise TypeError(
            "live glocal authority requires the exact physical campaign runtime"
        )
    if type(backend_owner) is not TorchSampledPathEvaluationBackend:
        raise TypeError(
            "live glocal authority requires the exact Torch sampled-path backend"
        )
    if (
        runtime_owner.campaign is not campaign
        or runtime_owner.backend is not backend_owner
    ):
        raise RuntimeError(
            "live glocal authority campaign/backend owners are not one physical graph"
        )
    physical_revalidator = GlocalBrierCampaignRuntime._revalidate_physical_authority
    physical_binding_payload = physical_revalidator(runtime_owner, backend_owner)

    arguments: dict[str, Any] = {
        "query_id": str(query_id),
        "context_design_hash": str(context_design_hash),
        "native_route_axis_hash": str(native_route_axis_hash),
        "native_mask_hash": str(native_mask_hash),
        "terminal_audit_result_hash": str(terminal_audit_result_hash),
        "terminal_sampling_design_hash": str(terminal_sampling_design_hash),
        "native_mask_bridge_hashes": tuple(native_mask_bridge_hashes),
        "irg_final_state_hash": irg_final_state_hash,
        "irg_final_mask_hash": irg_final_mask_hash,
        "external_scientific_identity_hash": external_scientific_identity_hash,
        "external_query_measure_hash": external_query_measure_hash,
        "certificate_id": certificate_id,
    }
    certificate = build_glocal_deployment_mask_certificate(
        campaign=campaign,
        **arguments,
    )
    binding_hash = content_hash(
        {
            "schema": "glocal_physical_runtime_binding_v1",
            "payload": dict(physical_binding_payload),
        }
    )
    current = physical_revalidator(runtime_owner, backend_owner)
    if (
        not isinstance(current, Mapping)
        or content_hash(
            {
                "schema": "glocal_physical_runtime_binding_v1",
                "payload": dict(current),
            }
        )
        != binding_hash
    ):
        raise RuntimeError(
            "physical runtime binding changed while deployment authority was issued"
        )
    return GlocalDeploymentAuthorityHandle(
        runtime_owner=runtime_owner,
        backend_owner=backend_owner,
        campaign=campaign,
        certificate=certificate,
        build_arguments=arguments,
        physical_binding_hash=binding_hash,
        physical_revalidator=physical_revalidator,
        _token=_LIVE_AUTHORITY_TOKEN,
    )


__all__ = [
    "GLOCAL_DEPLOYMENT_CERTIFICATE_SCHEMA",
    "GlocalAcceptedTransition",
    "GlocalDeploymentAuthorityHandle",
    "GlocalDeploymentMaskCertificate",
    "build_glocal_deployment_mask_certificate",
    "build_live_glocal_deployment_authority",
]
