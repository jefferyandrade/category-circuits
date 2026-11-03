"""Atomic shared-C acquisition across the retained convex KKT branches.

Each retained branch has a different frozen hard ladder, but every branch is
measured on the same registered C contexts and draws.  This module owns that
joint causal seam.  A shared row is committed only after the corresponding
directional-secant record for *every* retained branch has replayed against a
detached clone of its bounded sequential accumulator.  Invalid or incomplete
branch batches therefore leave the whole family unchanged.

The coordinator retains constant-cell sufficient statistics, never raw
records.  It grants no calibration, oracle, mask-commit, terminal, or Pareto
authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, Sequence

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_directional_remainder import V1DirectionalSecantRecord
from .modern_v1_fqr_branch_design import V1FQRBranchFamilyDesign
from .modern_v1_fqr_sequential_acquisition import (
    V1FQRDecisionReadinessReceipt,
    V1FQRDeterministicUncertaintyReceipt,
    V1FQRSequentialAccumulator,
    V1FQRSequentialAccumulatorSnapshot,
    V1FQRSequentialAcquisitionReceipt,
)
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch


FQR_BRANCH_FAMILY_PROGRESS_SCHEMA: Final = (
    "partition_free_v1_fqr_branch_family_C_progress_v1"
)
FQR_BRANCH_FAMILY_READINESS_SCHEMA: Final = (
    "partition_free_v1_fqr_branch_family_C_readiness_v1"
)
FQR_BRANCH_FAMILY_ACCUMULATOR_SNAPSHOT_SCHEMA: Final = (
    "partition_free_v1_fqr_branch_family_C_accumulator_snapshot_v1"
)
FQR_BRANCH_FAMILY_ACQUISITION_AUTHORITY_SCOPE: Final = (
    "atomic_shared_C_prefix_across_retained_KKT_branches_only_no_K_O_commit_"
    "terminal_or_pareto_authority_v1"
)

_PROGRESS_PREFIX: Final = "modern-v1-fqr-branch-family-C-progress-"
_READINESS_PREFIX: Final = "modern-v1-fqr-branch-family-C-readiness-"
_SNAPSHOT_PREFIX: Final = "modern-v1-fqr-branch-family-C-snapshot-"
_TRANSACTION_PREFIX: Final = "modern-v1-fqr-branch-family-C-transaction-"

FamilySamplingStatus = Literal["acquire_more_C", "sampling_stable"]
FamilyReadinessStatus = Literal[
    "acquire_more_C",
    "resolve_fd_rank",
    "resolve_kkt_numerics",
    "resolve_numerical_enclosure",
    "ready_for_cross_branch_fqr_screening",
]


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _branch_axis(
    values: Sequence[V1RelaxedKKTBranch],
) -> tuple[V1RelaxedKKTBranch, ...]:
    axis = tuple(values)
    if (
        not axis
        or any(type(value) is not V1RelaxedKKTBranch for value in axis)
        or len(set(axis)) != len(axis)
    ):
        raise ValueError("retained branch axis must be exact, unique, and nonempty")
    return axis


def _authority_firewall(value: object) -> None:
    for name in (
        "calibration_K_open_authority",
        "paired_oracle_O_open_authority",
        "mask_commit_authority",
        "terminal_authority",
        "pareto_claim_authority",
    ):
        if getattr(value, name) is not False:
            raise PermissionError("shared-C family evidence crossed its authority seam")


@dataclass(frozen=True, slots=True)
class V1FQRBranchFamilyProgressReceipt:
    """Bounded simultaneous snapshot of every retained branch C prefix."""

    branch_family_design_hash: str
    joint_family_registration_hash: str
    branches: tuple[V1RelaxedKKTBranch, ...]
    branch_receipts: tuple[V1FQRSequentialAcquisitionReceipt, ...] = field(
        repr=False,
        compare=False,
    )
    shared_observed_counts_by_stratum: tuple[tuple[str, int], ...]
    shared_next_expected_context_by_stratum: tuple[tuple[str, str | None], ...]
    shared_next_expected_draw_by_stratum: tuple[tuple[str, str | None], ...]
    shared_record_count: int
    transaction_chain_hash: str
    sampling_status: FamilySamplingStatus
    sampling_resolved_branch_count: int
    retained_branch_count: int
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_claim_authority: bool = False
    authority_scope: str = FQR_BRANCH_FAMILY_ACQUISITION_AUTHORITY_SCOPE
    schema: str = FQR_BRANCH_FAMILY_PROGRESS_SCHEMA
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _identity(self.branch_family_design_hash, name="branch-family design hash")
        _identity(
            self.joint_family_registration_hash,
            name="joint-family registration hash",
        )
        branches = _branch_axis(self.branches)
        receipts = tuple(self.branch_receipts)
        if len(receipts) != len(branches) or any(
            type(value) is not V1FQRSequentialAcquisitionReceipt
            for value in receipts
        ):
            raise TypeError("family progress requires one exact receipt per branch")
        for value in receipts:
            value.assert_integrity()
        counts = tuple(self.shared_observed_counts_by_stratum)
        next_contexts = tuple(self.shared_next_expected_context_by_stratum)
        next_draws = tuple(self.shared_next_expected_draw_by_stratum)
        if (
            not counts
            or any(value.observed_counts_by_stratum != counts for value in receipts)
            or any(
                value.next_expected_context_by_stratum != next_contexts
                for value in receipts
            )
            or any(
                value.next_expected_draw_by_stratum != next_draws
                for value in receipts
            )
        ):
            raise ValueError("retained branches do not share one exact C prefix")
        if any(
            name != context_name or name != draw_name
            for (name, _count), (context_name, _context), (draw_name, _draw) in zip(
                counts,
                next_contexts,
                next_draws,
                strict=True,
            )
        ):
            raise ValueError("shared C prefix axes are misaligned")
        shared_count = sum(count for _name, count in counts)
        if self.shared_record_count != shared_count:
            raise ValueError("family shared-record count differs from its C prefix")
        resolved = sum(not value.sampling_ambiguity_present for value in receipts)
        status: FamilySamplingStatus = (
            "sampling_stable" if resolved == len(receipts) else "acquire_more_C"
        )
        if (
            self.sampling_resolved_branch_count != resolved
            or self.retained_branch_count != len(branches)
            or self.sampling_status != status
        ):
            raise ValueError("family sampling status is not a replay of branch receipts")
        _identity(self.transaction_chain_hash, name="transaction chain hash")
        _authority_firewall(self)
        if (
            self.authority_scope != FQR_BRANCH_FAMILY_ACQUISITION_AUTHORITY_SCOPE
            or self.schema != FQR_BRANCH_FAMILY_PROGRESS_SCHEMA
        ):
            raise ValueError("family progress schema or authority scope is invalid")
        object.__setattr__(self, "branches", branches)
        object.__setattr__(self, "branch_receipts", receipts)
        object.__setattr__(self, "shared_observed_counts_by_stratum", counts)
        object.__setattr__(
            self,
            "shared_next_expected_context_by_stratum",
            next_contexts,
        )
        object.__setattr__(
            self,
            "shared_next_expected_draw_by_stratum",
            next_draws,
        )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "branch_family_design_hash": self.branch_family_design_hash,
            "joint_family_registration_hash": self.joint_family_registration_hash,
            "branches": tuple(value.value for value in self.branches),
            "branch_receipt_hashes": tuple(
                value.receipt_hash for value in self.branch_receipts
            ),
            "shared_observed_counts_by_stratum": (
                self.shared_observed_counts_by_stratum
            ),
            "shared_next_expected_context_by_stratum": (
                self.shared_next_expected_context_by_stratum
            ),
            "shared_next_expected_draw_by_stratum": (
                self.shared_next_expected_draw_by_stratum
            ),
            "shared_record_count": self.shared_record_count,
            "transaction_chain_hash": self.transaction_chain_hash,
            "sampling_status": self.sampling_status,
            "sampling_resolved_branch_count": self.sampling_resolved_branch_count,
            "retained_branch_count": self.retained_branch_count,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_claim_authority": self.pareto_claim_authority,
            "authority_scope": self.authority_scope,
        }

    @property
    def receipt_hash(self) -> str:
        cached = self._receipt_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for value in self.branch_receipts:
            value.assert_integrity()
        observed = content_hash(self._payload(), prefix=_PROGRESS_PREFIX)
        if self._receipt_hash is not None and self._receipt_hash != observed:
            raise ValueError("family C-progress receipt was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("family C-progress receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    def receipt_for_branch(
        self,
        branch: V1RelaxedKKTBranch,
    ) -> V1FQRSequentialAcquisitionReceipt:
        if type(branch) is not V1RelaxedKKTBranch:
            raise TypeError("branch lookup requires the exact KKT branch enum")
        try:
            return self.branch_receipts[self.branches.index(branch)]
        except ValueError as error:
            raise ValueError("branch is not retained by this family") from error


@dataclass(frozen=True, slots=True)
class V1FQRBranchFamilyReadinessReceipt:
    """Joint ambiguity routing after all branch-local readiness replays."""

    progress_receipt_hash: str
    branch_family_design_hash: str
    joint_family_registration_hash: str
    branches: tuple[V1RelaxedKKTBranch, ...]
    branch_acquisition_receipt_hashes: tuple[str, ...]
    branch_design_hashes: tuple[str, ...]
    branch_readiness_receipts: tuple[V1FQRDecisionReadinessReceipt, ...] = field(
        repr=False,
        compare=False,
    )
    status: FamilyReadinessStatus
    dominant_ambiguity: str | None
    ready_for_cross_branch_fqr_screening: bool
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_claim_authority: bool = False
    authority_scope: str = FQR_BRANCH_FAMILY_ACQUISITION_AUTHORITY_SCOPE
    schema: str = FQR_BRANCH_FAMILY_READINESS_SCHEMA
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "progress_receipt_hash",
            "branch_family_design_hash",
            "joint_family_registration_hash",
        ):
            _identity(getattr(self, name), name=name)
        branches = _branch_axis(self.branches)
        readiness = tuple(self.branch_readiness_receipts)
        if len(readiness) != len(branches) or any(
            type(value) is not V1FQRDecisionReadinessReceipt
            for value in readiness
        ):
            raise TypeError("family readiness requires one exact receipt per branch")
        for value in readiness:
            value.assert_integrity()
        acquisition_hashes = tuple(
            _identity(value, name="branch acquisition-receipt hash")
            for value in self.branch_acquisition_receipt_hashes
        )
        design_hashes = tuple(
            _identity(value, name="branch sequential-design hash")
            for value in self.branch_design_hashes
        )
        if (
            len(acquisition_hashes) != len(branches)
            or len(design_hashes) != len(branches)
            or tuple(value.acquisition_receipt_hash for value in readiness)
            != acquisition_hashes
            or tuple(value.design_hash for value in readiness) != design_hashes
        ):
            raise ValueError(
                "family readiness receipts differ from the registered branch progress"
            )
        statuses = tuple(value.status for value in readiness)
        if all(value.ready_for_fqr_proposal_screening for value in readiness):
            expected_status: FamilyReadinessStatus = (
                "ready_for_cross_branch_fqr_screening"
            )
            expected_dominant = None
        elif "acquire_more_C" in statuses:
            expected_status = "acquire_more_C"
            expected_dominant = "sampling"
        elif "resolve_fd_rank" in statuses:
            expected_status = "resolve_fd_rank"
            expected_dominant = "fd_rank"
        elif "resolve_kkt_numerics" in statuses:
            expected_status = "resolve_kkt_numerics"
            expected_dominant = "kkt_numerics"
        else:
            expected_status = "resolve_numerical_enclosure"
            expected_dominant = "numerical_enclosure"
        ready = expected_status == "ready_for_cross_branch_fqr_screening"
        if (
            self.status != expected_status
            or self.dominant_ambiguity != expected_dominant
            or self.ready_for_cross_branch_fqr_screening is not ready
        ):
            raise ValueError("family readiness is not a replay of branch ambiguity")
        _authority_firewall(self)
        if (
            self.authority_scope != FQR_BRANCH_FAMILY_ACQUISITION_AUTHORITY_SCOPE
            or self.schema != FQR_BRANCH_FAMILY_READINESS_SCHEMA
        ):
            raise ValueError("family readiness schema or authority scope is invalid")
        object.__setattr__(self, "branches", branches)
        object.__setattr__(
            self,
            "branch_acquisition_receipt_hashes",
            acquisition_hashes,
        )
        object.__setattr__(self, "branch_design_hashes", design_hashes)
        object.__setattr__(self, "branch_readiness_receipts", readiness)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "progress_receipt_hash": self.progress_receipt_hash,
            "branch_family_design_hash": self.branch_family_design_hash,
            "joint_family_registration_hash": self.joint_family_registration_hash,
            "branches": tuple(value.value for value in self.branches),
            "branch_acquisition_receipt_hashes": (
                self.branch_acquisition_receipt_hashes
            ),
            "branch_design_hashes": self.branch_design_hashes,
            "branch_readiness_receipt_hashes": tuple(
                value.receipt_hash for value in self.branch_readiness_receipts
            ),
            "status": self.status,
            "dominant_ambiguity": self.dominant_ambiguity,
            "ready_for_cross_branch_fqr_screening": (
                self.ready_for_cross_branch_fqr_screening
            ),
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_claim_authority": self.pareto_claim_authority,
            "authority_scope": self.authority_scope,
        }

    @property
    def receipt_hash(self) -> str:
        cached = self._receipt_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for value in self.branch_readiness_receipts:
            value.assert_integrity()
        observed = content_hash(self._payload(), prefix=_READINESS_PREFIX)
        if self._receipt_hash is not None and self._receipt_hash != observed:
            raise ValueError("family readiness receipt was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("family readiness receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FQRBranchFamilyAccumulatorSnapshot:
    """Bounded restart state for the atomic retained-branch C prefix."""

    branch_family_design_hash: str
    joint_family_registration_hash: str
    branches: tuple[V1RelaxedKKTBranch, ...]
    branch_snapshots: tuple[V1FQRSequentialAccumulatorSnapshot, ...] = field(
        repr=False,
        compare=False,
    )
    shared_record_count: int
    transaction_chain_hash: str
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_claim_authority: bool = False
    authority_scope: str = FQR_BRANCH_FAMILY_ACQUISITION_AUTHORITY_SCOPE
    schema: str = FQR_BRANCH_FAMILY_ACCUMULATOR_SNAPSHOT_SCHEMA
    _snapshot_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _identity(self.branch_family_design_hash, name="branch-family design hash")
        _identity(
            self.joint_family_registration_hash,
            name="joint-family registration hash",
        )
        branches = _branch_axis(self.branches)
        snapshots = tuple(self.branch_snapshots)
        if len(snapshots) != len(branches) or any(
            type(value) is not V1FQRSequentialAccumulatorSnapshot
            for value in snapshots
        ):
            raise TypeError("family snapshot requires one exact branch snapshot")
        for value in snapshots:
            value.assert_integrity()
        counts = tuple(value.observed_counts_by_stratum for value in snapshots)
        if not counts or any(value != counts[0] for value in counts[1:]):
            raise ValueError("family snapshot branches have different C prefixes")
        shared_count = sum(count for _name, count in counts[0])
        if self.shared_record_count != shared_count:
            raise ValueError("family snapshot record count differs from its prefix")
        _identity(self.transaction_chain_hash, name="transaction chain hash")
        _authority_firewall(self)
        if (
            self.authority_scope != FQR_BRANCH_FAMILY_ACQUISITION_AUTHORITY_SCOPE
            or self.schema != FQR_BRANCH_FAMILY_ACCUMULATOR_SNAPSHOT_SCHEMA
        ):
            raise ValueError("family snapshot schema or authority scope is invalid")
        object.__setattr__(self, "branches", branches)
        object.__setattr__(self, "branch_snapshots", snapshots)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "branch_family_design_hash": self.branch_family_design_hash,
            "joint_family_registration_hash": self.joint_family_registration_hash,
            "branches": tuple(value.value for value in self.branches),
            "branch_snapshot_hashes": tuple(
                value.snapshot_hash for value in self.branch_snapshots
            ),
            "shared_record_count": self.shared_record_count,
            "transaction_chain_hash": self.transaction_chain_hash,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_claim_authority": self.pareto_claim_authority,
            "authority_scope": self.authority_scope,
        }

    @property
    def snapshot_hash(self) -> str:
        cached = self._snapshot_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for value in self.branch_snapshots:
            value.assert_integrity()
        observed = content_hash(self._payload(), prefix=_SNAPSHOT_PREFIX)
        if self._snapshot_hash is not None and self._snapshot_hash != observed:
            raise ValueError("family accumulator snapshot was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("family accumulator snapshot hash does not match")
        object.__setattr__(self, "_snapshot_hash", observed)
        return observed


class V1FQRBranchFamilyAccumulator:
    """Atomic bounded-state coordinator for one registered retained family."""

    __slots__ = (
        "design",
        "_branches",
        "_accumulators",
        "_shared_record_count",
        "_transaction_chain_hash",
    )

    def __init__(self, design: V1FQRBranchFamilyDesign) -> None:
        if type(design) is not V1FQRBranchFamilyDesign:
            raise TypeError("family accumulator requires the exact family design")
        design_hash = design.assert_integrity()
        branches = tuple(value.branch for value in design.branch_bundles)
        self.design = design
        self._branches = branches
        self._accumulators = {
            value.branch: V1FQRSequentialAccumulator(value.sequential_design)
            for value in design.branch_bundles
        }
        self._shared_record_count = 0
        self._transaction_chain_hash = content_hash(
            {
                "branch_family_design_hash": design_hash,
                "joint_family_registration_hash": design.common_family_hash,
                "branches": tuple(value.value for value in branches),
                "shared_record_count": 0,
            },
            prefix=_TRANSACTION_PREFIX,
        )

    @classmethod
    def from_snapshot(
        cls,
        design: V1FQRBranchFamilyDesign,
        snapshot: V1FQRBranchFamilyAccumulatorSnapshot,
        *,
        expected_snapshot_hash: str | None = None,
    ) -> "V1FQRBranchFamilyAccumulator":
        if type(design) is not V1FQRBranchFamilyDesign:
            raise TypeError("family restore requires the exact family design")
        if type(snapshot) is not V1FQRBranchFamilyAccumulatorSnapshot:
            raise TypeError("family restore requires the exact family snapshot")
        if snapshot.shared_record_count > 0 and expected_snapshot_hash is None:
            raise ValueError("nonempty family restore requires an expected snapshot hash")
        snapshot.assert_integrity(expected_snapshot_hash)
        design.assert_integrity(snapshot.branch_family_design_hash)
        if (
            snapshot.joint_family_registration_hash != design.common_family_hash
            or snapshot.branches
            != tuple(value.branch for value in design.branch_bundles)
        ):
            raise ValueError("family snapshot is stale or belongs to another branch set")
        restored = cls(design)
        restored._accumulators = {
            bundle.branch: V1FQRSequentialAccumulator.from_snapshot(
                bundle.sequential_design,
                branch_snapshot,
                expected_snapshot_hash=branch_snapshot.snapshot_hash,
            )
            for bundle, branch_snapshot in zip(
                design.branch_bundles,
                snapshot.branch_snapshots,
                strict=True,
            )
        }
        restored._shared_record_count = snapshot.shared_record_count
        restored._transaction_chain_hash = snapshot.transaction_chain_hash
        restored.progress().assert_integrity()
        return restored

    @property
    def branches(self) -> tuple[V1RelaxedKKTBranch, ...]:
        return self._branches

    @property
    def shared_record_count(self) -> int:
        return self._shared_record_count

    @property
    def transaction_chain_hash(self) -> str:
        return self._transaction_chain_hash

    def record_atomic(
        self,
        records_by_branch: Sequence[
            tuple[V1RelaxedKKTBranch, V1DirectionalSecantRecord]
        ],
    ) -> V1FQRBranchFamilyProgressReceipt:
        return self.record_batch_atomic(
            tuple((branch, (record,)) for branch, record in records_by_branch)
        )

    def record_batch_atomic(
        self,
        records_by_branch: Sequence[
            tuple[V1RelaxedKKTBranch, Sequence[V1DirectionalSecantRecord]]
        ],
    ) -> V1FQRBranchFamilyProgressReceipt:
        """Commit a same-context branch batch only after complete replay."""

        rows = tuple((branch, tuple(records)) for branch, records in records_by_branch)
        if tuple(branch for branch, _records in rows) != self._branches:
            raise ValueError("C batch must cover retained branches in frozen order")
        lengths = tuple(len(records) for _branch, records in rows)
        if not lengths or lengths[0] == 0 or any(value != lengths[0] for value in lengths):
            raise ValueError("every retained branch must supply the same nonempty C batch")
        clones: dict[V1RelaxedKKTBranch, V1FQRSequentialAccumulator] = {}
        for branch in self._branches:
            current = self._accumulators[branch]
            snapshot = current.export_snapshot()
            clones[branch] = V1FQRSequentialAccumulator.from_snapshot(
                current.design,
                snapshot,
                expected_snapshot_hash=snapshot.snapshot_hash,
            )
        chain = self._transaction_chain_hash
        count = self._shared_record_count
        for offset in range(lengths[0]):
            records = tuple(records[offset] for _branch, records in rows)
            if any(type(value) is not V1DirectionalSecantRecord for value in records):
                raise TypeError("family C acquisition requires exact secant records")
            coordinates = tuple(
                (
                    value.stratum_id,
                    value.sealed_order_index,
                    value.context_id,
                    value.draw_id,
                )
                for value in records
            )
            if any(value != coordinates[0] for value in coordinates[1:]):
                raise ValueError("retained branches supplied different C coordinates")
            for branch, record in zip(self._branches, records, strict=True):
                clones[branch].record(record)
            count += 1
            chain = content_hash(
                {
                    "previous_transaction_chain_hash": chain,
                    "shared_record_count": count,
                    "shared_C_coordinate": coordinates[0],
                    "branch_record_hashes": tuple(
                        (branch.value, record.record_hash)
                        for branch, record in zip(
                            self._branches,
                            records,
                            strict=True,
                        )
                    ),
                },
                prefix=_TRANSACTION_PREFIX,
            )
        # No live state is mutated before every branch and row has replayed.
        self._accumulators = clones
        self._shared_record_count = count
        self._transaction_chain_hash = chain
        return self.progress()

    def progress(self) -> V1FQRBranchFamilyProgressReceipt:
        receipts = tuple(
            self._accumulators[branch].receipt() for branch in self._branches
        )
        first = receipts[0]
        return V1FQRBranchFamilyProgressReceipt(
            branch_family_design_hash=self.design.design_hash,
            joint_family_registration_hash=self.design.common_family_hash,
            branches=self._branches,
            branch_receipts=receipts,
            shared_observed_counts_by_stratum=first.observed_counts_by_stratum,
            shared_next_expected_context_by_stratum=(
                first.next_expected_context_by_stratum
            ),
            shared_next_expected_draw_by_stratum=first.next_expected_draw_by_stratum,
            shared_record_count=self._shared_record_count,
            transaction_chain_hash=self._transaction_chain_hash,
            sampling_status=(
                "sampling_stable"
                if all(not value.sampling_ambiguity_present for value in receipts)
                else "acquire_more_C"
            ),
            sampling_resolved_branch_count=sum(
                not value.sampling_ambiguity_present for value in receipts
            ),
            retained_branch_count=len(self._branches),
        )

    def readiness(
        self,
        deterministic_by_branch: Sequence[
            tuple[V1RelaxedKKTBranch, V1FQRDeterministicUncertaintyReceipt]
        ],
    ) -> V1FQRBranchFamilyReadinessReceipt:
        rows = tuple(deterministic_by_branch)
        if tuple(branch for branch, _receipt in rows) != self._branches:
            raise ValueError("deterministic receipts must cover retained branches in order")
        progress = self.progress()
        readiness = tuple(
            self._accumulators[branch].decision_readiness(deterministic)
            for branch, deterministic in rows
        )
        statuses = tuple(value.status for value in readiness)
        if all(value.ready_for_fqr_proposal_screening for value in readiness):
            status: FamilyReadinessStatus = "ready_for_cross_branch_fqr_screening"
            dominant = None
        elif "acquire_more_C" in statuses:
            status = "acquire_more_C"
            dominant = "sampling"
        elif "resolve_fd_rank" in statuses:
            status = "resolve_fd_rank"
            dominant = "fd_rank"
        elif "resolve_kkt_numerics" in statuses:
            status = "resolve_kkt_numerics"
            dominant = "kkt_numerics"
        else:
            status = "resolve_numerical_enclosure"
            dominant = "numerical_enclosure"
        return V1FQRBranchFamilyReadinessReceipt(
            progress_receipt_hash=progress.receipt_hash,
            branch_family_design_hash=self.design.design_hash,
            joint_family_registration_hash=self.design.common_family_hash,
            branches=self._branches,
            branch_acquisition_receipt_hashes=tuple(
                value.receipt_hash for value in progress.branch_receipts
            ),
            branch_design_hashes=tuple(
                value.design_hash for value in progress.branch_receipts
            ),
            branch_readiness_receipts=readiness,
            status=status,
            dominant_ambiguity=dominant,
            ready_for_cross_branch_fqr_screening=(
                status == "ready_for_cross_branch_fqr_screening"
            ),
        )

    def export_snapshot(self) -> V1FQRBranchFamilyAccumulatorSnapshot:
        return V1FQRBranchFamilyAccumulatorSnapshot(
            branch_family_design_hash=self.design.design_hash,
            joint_family_registration_hash=self.design.common_family_hash,
            branches=self._branches,
            branch_snapshots=tuple(
                self._accumulators[branch].export_snapshot()
                for branch in self._branches
            ),
            shared_record_count=self._shared_record_count,
            transaction_chain_hash=self._transaction_chain_hash,
        )


__all__ = [
    "FQR_BRANCH_FAMILY_ACQUISITION_AUTHORITY_SCOPE",
    "FQR_BRANCH_FAMILY_ACCUMULATOR_SNAPSHOT_SCHEMA",
    "FQR_BRANCH_FAMILY_PROGRESS_SCHEMA",
    "FQR_BRANCH_FAMILY_READINESS_SCHEMA",
    "V1FQRBranchFamilyAccumulator",
    "V1FQRBranchFamilyAccumulatorSnapshot",
    "V1FQRBranchFamilyProgressReceipt",
    "V1FQRBranchFamilyReadinessReceipt",
]
