"""Workflow-neutral sampling identities and evidence-role designs.

Three non-equivalent sampling objects live at this boundary:

* :class:`BinaryEvidenceRoleDesign` is a sealed, finite, disjoint partition.
  It survives for exact historical compatibility and bounded research uses;
  it is not a current IID evidence design.
* :class:`~mechanistic_systems.measurement.iid_streams.BinaryIIDSamplingDesign`
  (and its class-stratified sibling) describes independent logical draws
  with replacement from a frozen empirical law.  Repeated source contexts
  retain distinct draw identities.
* :class:`~mechanistic_systems.measurement.iid_streams.StratifiedSealedWithoutReplacementSamplingDesign`
  reveals, per (role, stream, stratum), a nested prefix of one sealed uniform
  permutation of that stratum's finite population: simple random sampling
  without replacement whose terminal state is the exact stratum census.  It
  is the modern-v0 proposal-adjudication law
  (``independent_sealed_without_replacement_class_stratified_v1``); it is
  NOT the role partition above, which splits contexts between evidence roles
  rather than sampling within one role.

The distinction is decision-bearing.  A finite role allocation cannot be
used to infer the IID filtration or sample horizon, and physical packet
naturalization must not introduce a statistical stopping horizon.  Under the
sealed without-replacement law the sample horizon is the population itself,
disclosed per stratum, and the interval carries the exact finite-population
correction rather than a horizon knob.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, ClassVar, Mapping, Protocol, Sequence, TypeVar

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.measurement.contexts import ContextAxis
from mechanistic_systems.measurement.iid_streams import (
    BinaryIIDSamplingDesign,
    binary_iid_sampling_design_from_axis,
)


BINARY_EVIDENCE_ROLE_STRATA = ("on_target", "off_target")
DISJOINT_ROLE_DESIGN_AUTHORITY_CEILING = (
    "compatibility_and_research_only_not_current_iid_evidence"
)

# These two strings are serialized history.  The namespace payload is used by
# authoritative queue-v0 artifacts, and the hash prefix occurs in historical
# role-design receipts.  Renaming either would change replay identity.  Their
# survival does not give the retired workflow carrier any current authority.
_HISTORICAL_BINARY_EVIDENCE_NAMESPACE_PAYLOAD_ID = (
    "irg_native_global_two_partition_bce_precision_v1"
)
_HISTORICAL_BINARY_ROLE_DESIGN_HASH_PREFIX = "irg-native-binary-role-design-"


class IIDPhysicalPacketPolicy(Protocol):
    """Minimum structural contract for IID physical-packet naturalization.

    The protocol deliberately excludes statistical horizon fields.  The
    naturalizer is allowed to change physical packet sizes only.
    """

    response_initial_packet: int
    response_packet: int
    response_maximum_contexts_per_stratum: int


_PacketPolicyT = TypeVar("_PacketPolicyT", bound=IIDPhysicalPacketPolicy)


@dataclass(frozen=True, slots=True)
class BinaryEvidenceRoleDesign:
    """A sealed finite partition of binary contexts into evidence roles.

    This object is retained for byte-compatible replay and explicitly bounded
    research comparisons.  It is sampling without replacement over a finite
    source axis and therefore cannot establish current IID validity, an IID
    filtration, or a superpopulation claim.

    The six dataclass fields intentionally match the historical carrier.  Its
    content hash also retains the historical prefix because existing receipts
    bind those exact bytes.  The authority metadata is a class variable so it
    clarifies use without changing that serialized identity.
    """

    authority_ceiling: ClassVar[str] = DISJOINT_ROLE_DESIGN_AUTHORITY_CEILING
    static_role_partition: ClassVar[bool] = True
    sampling_with_replacement: ClassVar[bool] = False

    target_class_id: int
    calibration_contexts: Mapping[str, tuple[str, ...]]
    response_contexts: Mapping[str, tuple[str, ...]]
    oracle_contexts: Mapping[str, tuple[str, ...]]
    source_context_axis_hash: str
    randomization_namespace: str

    def __post_init__(self) -> None:
        roles = (
            self.calibration_contexts,
            self.response_contexts,
            self.oracle_contexts,
        )
        normalized: list[dict[str, tuple[str, ...]]] = []
        for role in roles:
            values = {str(key): tuple(map(str, rows)) for key, rows in role.items()}
            if set(values) != set(BINARY_EVIDENCE_ROLE_STRATA):
                raise ValueError("each evidence role requires binary target strata")
            if any(not rows or len(set(rows)) != len(rows) for rows in values.values()):
                raise ValueError("evidence-role strata must be nonempty and unique")
            normalized.append(values)
        flat_roles = [
            {context for rows in role.values() for context in rows}
            for role in normalized
        ]
        if any(
            flat_roles[left].intersection(flat_roles[right])
            for left in range(len(flat_roles))
            for right in range(left + 1, len(flat_roles))
        ):
            raise ValueError(
                "calibration, response, and oracle contexts must be disjoint"
            )
        object.__setattr__(self, "calibration_contexts", normalized[0])
        object.__setattr__(self, "response_contexts", normalized[1])
        object.__setattr__(self, "oracle_contexts", normalized[2])

    @property
    def content_hash(self) -> str:
        """Return the exact historical content-address identity."""

        return content_hash(
            self,
            prefix=_HISTORICAL_BINARY_ROLE_DESIGN_HASH_PREFIX,
        )

    def receipt(self) -> dict[str, Any]:
        """Describe the design without granting it current IID authority."""

        roles = {
            "calibration": self.calibration_contexts,
            "response": self.response_contexts,
            "oracle": self.oracle_contexts,
        }
        return {
            "schema": "binary_disjoint_evidence_role_design_v1",
            "content_hash": self.content_hash,
            "target_class_id": int(self.target_class_id),
            "source_context_axis_hash": self.source_context_axis_hash,
            "randomization_namespace": self.randomization_namespace,
            "role_counts": {
                role: {
                    stratum: len(contexts) for stratum, contexts in populations.items()
                }
                for role, populations in roles.items()
            },
            "static_role_partition": self.static_role_partition,
            "sampling_with_replacement": self.sampling_with_replacement,
            "authority_ceiling": self.authority_ceiling,
        }


def sealed_evidence_role_order(
    context_ids: Sequence[str],
    *,
    namespace: str,
) -> tuple[str, ...]:
    """Return the historical deterministic finite-role permutation.

    This is a replay primitive, not an IID draw mechanism.  It sorts the
    finite population once and therefore cannot create new observation
    identities beyond that population.
    """

    population = tuple(sorted(map(str, context_ids)))
    return tuple(
        sorted(
            population,
            key=lambda context_id: content_hash(
                {
                    "algorithm": "sha256_role_split_v1",
                    "namespace": str(namespace),
                    "context_id": context_id,
                }
            ),
        )
    )


def binary_evidence_role_namespace(
    *,
    run_id: str,
    random_seed: int,
    context_axis_hash: str,
) -> str:
    """Return the authoritative queue-v0 binary evidence namespace.

    The payload ID is deliberately frozen to retain exact existing hashes.
    Its historical spelling is serialization compatibility, not a workflow
    alias or a claim that the static role design is current IID evidence.
    """

    if not str(run_id) or not str(context_axis_hash):
        raise ValueError(
            "binary evidence-role namespace requires run and context identity"
        )
    return content_hash(
        {
            "execution": _HISTORICAL_BINARY_EVIDENCE_NAMESPACE_PAYLOAD_ID,
            "run_id": str(run_id),
            "random_seed": int(random_seed),
            "context_axis_hash": str(context_axis_hash),
        }
    )


def compile_disjoint_binary_evidence_role_design(
    axis: ContextAxis,
    *,
    target_class_id: int,
    randomization_namespace: str,
    calibration_fraction: float = 0.25,
    response_fraction: float = 0.25,
) -> BinaryEvidenceRoleDesign:
    """Compile a compatibility-only finite disjoint role partition.

    The compiler preserves the exact historical split, including minimum role
    sizes and rounding.  Use :func:`compile_binary_iid_sampling_design` for
    current independent empirical draws.
    """

    axis.validate()
    if not str(randomization_namespace):
        raise ValueError("binary role design requires a randomization namespace")
    if (
        not 0.0 < float(calibration_fraction) < 1.0
        or not 0.0 < float(response_fraction) < 1.0
        or float(calibration_fraction) + float(response_fraction) >= 1.0
    ):
        raise ValueError("binary role fractions must leave a nonempty oracle role")
    binary = {
        "on_target": tuple(
            record.context_id
            for record in axis.records
            if int(record.label) == int(target_class_id)
        ),
        "off_target": tuple(
            record.context_id
            for record in axis.records
            if int(record.label) != int(target_class_id)
        ),
    }
    roles: dict[str, dict[str, tuple[str, ...]]] = {
        "calibration": {},
        "response": {},
        "oracle": {},
    }
    for stratum, context_ids in binary.items():
        order = sealed_evidence_role_order(
            context_ids,
            namespace=f"{randomization_namespace}:{stratum}",
        )
        if len(order) < 6:
            raise ValueError(
                "disjoint binary role design requires at least six contexts in each "
                "binary stratum to keep three evidence roles disjoint"
            )
        calibration_count = max(2, int(round(len(order) * calibration_fraction)))
        response_count = max(2, int(round(len(order) * response_fraction)))
        if calibration_count + response_count > len(order) - 2:
            excess = calibration_count + response_count - (len(order) - 2)
            response_count = max(2, response_count - excess)
        if calibration_count + response_count > len(order) - 2:
            calibration_count = len(order) - response_count - 2
        roles["calibration"][stratum] = order[:calibration_count]
        roles["response"][stratum] = order[
            calibration_count : calibration_count + response_count
        ]
        roles["oracle"][stratum] = order[calibration_count + response_count :]
    return BinaryEvidenceRoleDesign(
        target_class_id=int(target_class_id),
        calibration_contexts=roles["calibration"],
        response_contexts=roles["response"],
        oracle_contexts=roles["oracle"],
        source_context_axis_hash=axis.content_hash,
        randomization_namespace=str(randomization_namespace),
    )


def compile_binary_iid_sampling_design(
    axis: ContextAxis,
    *,
    target_class_id: int,
    randomization_namespace: str,
) -> BinaryIIDSamplingDesign:
    """Compile the current full-population IID empirical sampling design."""

    return binary_iid_sampling_design_from_axis(
        axis,
        target_class_id=int(target_class_id),
        randomization_namespace=str(randomization_namespace),
    )


def naturalize_iid_physical_packet_policy(
    policy: _PacketPolicyT,
    design: BinaryIIDSamplingDesign,
) -> _PacketPolicyT:
    """Fit physical response packets to source support without an IID horizon.

    Physical evaluation may deduplicate repeated source contexts inside one
    packet, so a response packet need not exceed the available source support.
    This operation changes only ``response_initial_packet`` and
    ``response_packet``.  In particular, it does not cap logical IID draws,
    alter oracle/calibration horizons, or reinterpret source-population size
    as a statistical sample limit.
    """

    response_capacity = min(
        min(
            len(values),
            int(policy.response_maximum_contexts_per_stratum),
        )
        for values in design.source_context_ids_by_stratum.values()
    )
    if response_capacity <= 0:
        raise ValueError("IID training distribution has no response support")
    return replace(
        policy,
        response_initial_packet=min(
            int(policy.response_initial_packet),
            response_capacity,
        ),
        response_packet=min(
            int(policy.response_packet),
            response_capacity,
        ),
    )


__all__ = [
    "BINARY_EVIDENCE_ROLE_STRATA",
    "DISJOINT_ROLE_DESIGN_AUTHORITY_CEILING",
    "BinaryEvidenceRoleDesign",
    "BinaryIIDSamplingDesign",
    "IIDPhysicalPacketPolicy",
    "binary_evidence_role_namespace",
    "compile_binary_iid_sampling_design",
    "compile_disjoint_binary_evidence_role_design",
    "naturalize_iid_physical_packet_policy",
    "sealed_evidence_role_order",
]
