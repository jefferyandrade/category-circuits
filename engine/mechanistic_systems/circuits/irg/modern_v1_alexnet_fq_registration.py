"""Compact pre-outcome registration for AlexNet corrected-score/F-Q sampling.

Dynamic production uses causally opened, domain-separated role streams.  D is
an adaptive prefix of an independent full-population permutation opened only
after the current mask and raw-logit transform freeze.  C is another full-
population permutation opened only after D freezes the selected shell.  K is
an append-only calibration prefix stream opened only after the candidate
freezes.  Finally, O is independently permuted *after* both affine states and
the exact K prefix freeze, over the out-of-fit population ``U \\ K``.

The older exhaustive D/C/oracle cut remains available solely to read and test
legacy artifacts.  It is not dynamic estimand authority: adaptively selecting
a candidate from the first cut and evaluating it on the residual cut does not
in general produce a lawful oracle sample.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from mechanistic_systems.experimental_design.finite_population import (
    SealedUniformPermutationReceipt,
    verify_sealed_uniform_permutation_custody,
)

from .contracts import content_hash


ALEXNET_FQ_ACQUISITION_REGISTRATION_SCHEMA_V1: Final = (
    "partition_free_v1_alexnet_fq_acquisition_registration_v1"
)
ALEXNET_FQ_ACQUISITION_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_alexnet_fq_acquisition_registration_v2"
)
ALEXNET_FQ_ALPHA_SPLIT_POLICY: Final = (
    "hierarchical_irreversibility_bonferroni_half_then_equal_path_split_v1"
)
ALEXNET_FQ_ALPHA_SPENDING_POLICY: Final = (
    "finite_horizon_normalized_inverse_square_nonrecycling_v1"
)
ALEXNET_FQ_FOLD_POLICY_V1: Final = (
    "sealed_uniform_per_stratum_three_role_train_partition_v1"
)
ALEXNET_FQ_FOLD_POLICY: Final = (
    "sealed_uniform_full_stratum_contiguous_three_role_slices_v2"
)
ALEXNET_FQ_LOOK_POLICY_V1: Final = (
    "balanced_dyadic_prefixes_then_exact_reserved_fold_census_v1"
)
ALEXNET_FQ_LOOK_POLICY: Final = (
    "balanced_dyadic_prefixes_then_registered_role_capacity_v2"
)
ALEXNET_FQ_PREFIX_SEMANTICS_V1: Final = (
    "adaptive_prefix_of_one_fixed_complete_train_role_fold_"
    "not_a_population_definition_v1"
)
ALEXNET_FQ_PREFIX_SEMANTICS: Final = (
    "adaptive_without_replacement_prefix_of_sealed_full_train_stratum_"
    "role_capacity_is_not_population_census_v2"
)
ALEXNET_FQ_FULL_POPULATION_ESTIMAND: Final = (
    "stratified_complete_imagenet_train_finite_population_mean_v1"
)
ALEXNET_FQ_TRAIN_ROLE_PARTITION_RECEIPT_SCHEMA_V1: Final = (
    "partition_free_v1_alexnet_exhaustive_train_role_partition_receipt_v1"
)
ALEXNET_FQ_TRAIN_ROLE_PARTITION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_alexnet_full_frame_contiguous_role_partition_receipt_v2"
)
ALEXNET_FQ_FOLD_REALIZATION_SCHEMA_V1: Final = (
    "partition_free_v1_alexnet_fq_fold_realization_v1"
)
ALEXNET_FQ_FOLD_REALIZATION_SCHEMA: Final = (
    "partition_free_v1_alexnet_fq_fold_realization_v3"
)
ALEXNET_FQ_FOLD_REALIZATION_SCHEMA_V2: Final = (
    "partition_free_v1_alexnet_fq_fold_realization_v2"
)
ALEXNET_FQ_GENERATION_SAMPLING_FRAME_SCHEMA: Final = (
    "partition_free_v1_alexnet_generation_full_train_sampling_frame_v1"
)
ALEXNET_FQ_CAUSAL_GENERATION_FRAME_SCHEMA: Final = (
    "partition_free_v1_alexnet_causal_generation_frame_v1"
)
ALEXNET_FQ_CAUSAL_ROLE_STREAM_SCHEMA: Final = (
    "partition_free_v1_alexnet_causal_role_stream_v1"
)
ALEXNET_FQ_DISCOVERY_PREFIX_FREEZE_SCHEMA: Final = (
    "partition_free_v1_alexnet_discovery_prefix_freeze_v1"
)
ALEXNET_FQ_CERTIFICATION_PREFIX_FREEZE_SCHEMA: Final = (
    "partition_free_v1_alexnet_certification_prefix_freeze_v1"
)
ALEXNET_FQ_CALIBRATION_PREFIX_FREEZE_SCHEMA: Final = (
    "partition_free_v1_alexnet_calibration_prefix_freeze_v1"
)
ALEXNET_FQ_ORACLE_SAMPLING_FRAME_SCHEMA: Final = (
    "partition_free_v1_alexnet_out_of_fit_oracle_sampling_frame_v1"
)
ALEXNET_FQ_CAUSAL_ROLE_POLICY: Final = (
    "domain_separated_causal_full_frame_d_c_k_then_out_of_fit_o_v1"
)
ALEXNET_FQ_CAUSAL_RANDOMIZATION_ASSUMPTION: Final = (
    "hmac_sha256_domain_separation_under_prf_assumption_v1"
)
ALEXNET_FQ_FULL_FRAME_PREFIX_SEMANTICS: Final = (
    "adaptive_without_replacement_prefix_of_independent_complete_train_"
    "stratum_per_role_v1"
)
ALEXNET_FQ_OUT_OF_FIT_ORACLE_ESTIMAND: Final = (
    "frozen_calibration_paired_bce_mean_on_complete_train_minus_exact_k_prefix_v1"
)

_REGISTRATION_PREFIX = "partition-free-v1-alexnet-fq-registration-"
_FOLD_REALIZATION_PREFIX = "partition-free-v1-alexnet-fq-fold-realization-"
_TRAIN_ROLE_PARTITION_PREFIX = (
    "partition-free-v1-alexnet-exhaustive-train-role-partition-"
)
_FULL_TRAIN_MEMBERSHIP_PREFIX = "partition-free-v1-alexnet-complete-train-membership-"
_TRAIN_ROLE_MEMBERSHIP_PREFIX = "partition-free-v1-alexnet-train-role-membership-"
_FOLD_ASSIGNMENT_KDF_SCHEMA = "partition_free_v1_alexnet_fq_fold_assignment_kdf_v2"
_GENERATION_SAMPLING_FRAME_KDF_SCHEMA = (
    "partition_free_v1_alexnet_generation_full_train_sampling_frame_kdf_v1"
)
_GENERATION_SAMPLING_FRAME_PREFIX = (
    "partition-free-v1-alexnet-generation-full-train-sampling-frame-"
)
_ORACLE_ORDER_ROOT_COMMITMENT_PREFIX = (
    "partition-free-v1-alexnet-paired-bce-oracle-order-root-commitment-"
)
_ORACLE_ORDER_KDF_SCHEMA = (
    "partition_free_v1_alexnet_paired_bce_oracle_attempt_order_kdf_v1"
)
_CAUSAL_ROLE_KDF_SCHEMA = (
    "partition_free_v1_alexnet_causal_role_permutation_kdf_v1"
)
_CAUSAL_GENERATION_FRAME_PREFIX = (
    "partition-free-v1-alexnet-causal-generation-frame-"
)
_CAUSAL_ROLE_STREAM_PREFIX = "partition-free-v1-alexnet-causal-role-stream-"
_DISCOVERY_PREFIX_FREEZE_PREFIX = (
    "partition-free-v1-alexnet-discovery-prefix-freeze-"
)
_CERTIFICATION_PREFIX_FREEZE_PREFIX = (
    "partition-free-v1-alexnet-certification-prefix-freeze-"
)
_CALIBRATION_PREFIX_FREEZE_PREFIX = (
    "partition-free-v1-alexnet-calibration-prefix-freeze-"
)
_OUT_OF_FIT_ORACLE_FRAME_PREFIX = (
    "partition-free-v1-alexnet-out-of-fit-oracle-frame-"
)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be one positive integer")
    return int(value)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be one nonnegative integer")
    return int(value)


def _stratum_counts(
    value: object,
    *,
    name: str,
    positive: bool = True,
) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must contain target then complement counts")
    validator = _positive_int if positive else _nonnegative_int
    return (
        validator(value[0], name=f"{name} target"),
        validator(value[1], name=f"{name} complement"),
    )


def _prefix_membership_hash(
    *,
    role: str,
    stratum: str,
    context_ids: Sequence[str],
) -> str:
    contexts = tuple(_identity(value, name=f"{role} {stratum} prefix") for value in context_ids)
    if len(contexts) != len(set(contexts)):
        raise ValueError(f"{role} {stratum} prefix repeats a context")
    return content_hash(
        {
            "schema": "partition_free_v1_alexnet_role_prefix_membership_v1",
            "role": role,
            "stratum": stratum,
            "ordered_context_ids": contexts,
        },
        prefix=f"partition-free-v1-alexnet-{role}-{stratum}-prefix-",
    )


def alexnet_fq_full_population_membership_hash(
    *,
    stratum: str,
    context_ids: Sequence[str],
) -> str:
    """Return the canonical set identity for one complete train stratum.

    This identity is intentionally distinct from the historical F/Q
    registration's ordered source-membership commitment.  K/O receipts use a
    sorted set identity because a fresh oracle permutation must not redefine
    the finite population it permutes.  Cross-runtime reducers must call this
    owner instead of recreating either hash domain.
    """

    contexts = tuple(
        _identity(value, name=f"{stratum} full-population context")
        for value in context_ids
    )
    if not contexts or len(contexts) != len(set(contexts)):
        raise ValueError(f"{stratum} full-population membership must be unique")
    return content_hash(
        {
            "schema": "partition_free_v1_alexnet_full_population_membership_v1",
            "stratum": _identity(stratum, name="full-population stratum"),
            "canonical_context_ids": tuple(sorted(contexts)),
        },
        prefix=f"partition-free-v1-alexnet-fq-{stratum}-full-population-membership-",
    )


def out_of_fit_population(
    full_population: Sequence[str],
    calibration_prefix: Sequence[str],
) -> tuple[str, ...]:
    """Return canonical ``U \\ K`` after validating an exact K subset.

    The canonical order is intentional.  O receives a fresh sealed
    permutation of this set; it never inherits the residual order of K or D.
    """

    population = tuple(
        _identity(value, name="out-of-fit full-population context")
        for value in full_population
    )
    calibration = tuple(
        _identity(value, name="out-of-fit calibration-prefix context")
        for value in calibration_prefix
    )
    if (
        not population
        or len(population) != len(set(population))
        or len(calibration) != len(set(calibration))
        or not set(calibration).issubset(population)
    ):
        raise ValueError("out-of-fit population requires one unique K subset of U")
    excluded = set(calibration)
    result = tuple(sorted(value for value in population if value not in excluded))
    if not result:
        raise ValueError("calibration prefix leaves no out-of-fit oracle population")
    return result


def _alpha(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be one alpha scalar")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie strictly in (0,1)")
    return result


def _entropy_root(value: object, *, name: str) -> str:
    result = _identity(value, name=name)
    try:
        decoded = bytes.fromhex(result)
    except ValueError as error:
        raise ValueError(f"{name} must be lowercase hexadecimal") from error
    if len(decoded) != 32 or result != result.lower() or decoded.hex() != result:
        raise ValueError(f"{name} must contain exactly 256 lowercase bits")
    return result


def _exact_slot_state(value: object) -> tuple[object, ...]:
    return tuple(getattr(value, item.name) for item in fields(value))


def _restore_exact_slot_state(
    value: object,
    state: object,
    *,
    artifact: str,
) -> None:
    coordinates = fields(value)
    if not isinstance(state, (tuple, list)) or len(state) != len(coordinates):
        raise ValueError(
            f"unsupported legacy {artifact} state; migration must be explicit"
        )
    for coordinate, coordinate_value in zip(coordinates, state, strict=True):
        object.__setattr__(value, coordinate.name, coordinate_value)
    value.__post_init__()  # type: ignore[attr-defined]


def hierarchical_master_alpha_split(
    master_familywise_alpha: float,
) -> tuple[float, float, float]:
    """Analytic default reflecting the irreversible oracle asymmetry.

    Half of the trajectory error stake protects the irreversible direct-loss
    gate.  The other half protects path construction and is split symmetrically
    between corrected-order and selected-shell F/Q inference.  This is the
    max-min allocation on each binary branch when no empirical cost model or
    prior reliability asymmetry is admitted.
    """

    master = _alpha(master_familywise_alpha, name="master familywise alpha")
    oracle = master / 2.0
    proposal = (master - oracle) / 2.0
    fq = master - math.fsum((oracle, proposal))
    if min(proposal, fq, oracle) <= 0.0 or math.fsum((proposal, fq, oracle)) != master:
        raise ArithmeticError("hierarchical alpha split did not close exactly")
    return proposal, fq, oracle


def finite_inverse_square_allocations(
    familywise_alpha: float,
    horizon: int,
) -> tuple[float, ...]:
    """Close one finite nonrecycling alpha family on an index axis."""

    total = _alpha(familywise_alpha, name="familywise alpha")
    count = _positive_int(horizon, name="alpha horizon")
    weights = tuple(1.0 / float((index + 1) ** 2) for index in range(count))
    normalization = math.fsum(weights)
    prefix = tuple(total * weight / normalization for weight in weights[:-1])
    final = total - math.fsum(prefix)
    if final <= 0.0:
        raise ArithmeticError("finite inverse-square allocation lost its final stake")
    result = prefix + (final,)
    if math.fsum(result) != total:
        raise ArithmeticError("finite inverse-square allocation did not close exactly")
    return result


def balanced_dyadic_census_looks(
    target_population_count: int,
    complement_population_count: int,
) -> tuple[tuple[int, int], ...]:
    """Return dyadic prefixes, saturating each stratum, then exact census.

    ImageNet's target/complement strata are highly asymmetric.  Once the
    smaller stratum is exhausted it remains fixed while the larger stratum
    continues through its dyadic prefixes; jumping directly from the smaller
    census to the much larger joint census would turn an operational packet
    policy into an avoidable inference cap.
    """

    target = _positive_int(target_population_count, name="target population count")
    complement = _positive_int(
        complement_population_count,
        name="complement population count",
    )
    looks: list[tuple[int, int]] = []
    value = 1
    final = (target, complement)
    while True:
        current = (min(value, target), min(value, complement))
        if not looks or current != looks[-1]:
            looks.append(current)
        if current == final:
            break
        value *= 2
        if value > max(target, complement):
            break
    if looks[-1] != final:
        looks.append(final)
    return tuple(looks)


def balanced_dyadic_capacity_looks(
    target_capacity_count: int,
    complement_capacity_count: int,
) -> tuple[tuple[int, int], ...]:
    """Return nested dyadic looks ending at a registered role capacity.

    This is deliberately distinct from a population census.  In particular,
    the D capacity is strictly smaller than the full train population because
    the same sealed permutation also reserves nonempty C and oracle slices.
    """

    return balanced_dyadic_census_looks(
        target_capacity_count,
        complement_capacity_count,
    )


def contiguous_three_role_cut_counts(
    population_count: int,
) -> tuple[int, int, int]:
    """Return deterministic nonempty D/C/oracle capacities for one stratum."""

    population = _positive_int(population_count, name="role-cut population count")
    if population < 3:
        raise ValueError(
            "full-train stratum requires at least three contexts for nonempty "
            "D/C/oracle slices"
        )
    discovery = population // 3
    held_out = population - discovery
    certification = held_out // 2
    oracle = held_out - certification
    if min(discovery, certification, oracle) <= 0:
        raise ArithmeticError("three-role capacity calculation produced an empty role")
    return discovery, certification, oracle


def contiguous_three_role_slices(
    full_permutation: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Take exact D/C/oracle contiguous slices of one full permutation."""

    order = tuple(
        _identity(value, name="full-stratum permutation context")
        for value in full_permutation
    )
    if len(order) != len(set(order)):
        raise ValueError("full-stratum permutation contexts must be unique")
    discovery_count, certification_count, _oracle_count = (
        contiguous_three_role_cut_counts(len(order))
    )
    certification_stop = discovery_count + certification_count
    return (
        order[:discovery_count],
        order[discovery_count:certification_stop],
        order[certification_stop:],
    )


def _role_slice_boundaries(
    *,
    target_population_count: int,
    complement_population_count: int,
) -> tuple[tuple[str, int, int, int, int], ...]:
    rows: list[tuple[str, int, int, int, int]] = []
    for stratum, population_count in (
        ("target", target_population_count),
        ("complement", complement_population_count),
    ):
        discovery, certification, _oracle = contiguous_three_role_cut_counts(
            population_count
        )
        rows.append(
            (
                stratum,
                0,
                discovery,
                discovery + certification,
                population_count,
            )
        )
    return tuple(rows)


def _role_rows(
    value: Sequence[tuple[str, Sequence[str]]],
    *,
    role: str,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    rows = tuple(
        (
            _identity(name, name=f"{role} stratum"),
            tuple(_identity(context, name=f"{role} context") for context in contexts),
        )
        for name, contexts in value
    )
    if tuple(name for name, _ in rows) != ("target", "complement"):
        raise ValueError(f"{role} rows must name target then complement")
    flat = tuple(context for _, contexts in rows for context in contexts)
    if len(flat) != len(set(flat)):
        raise ValueError(f"{role} contexts must be unique")
    return rows


@dataclass(frozen=True, slots=True)
class V1AlexNetTrainRolePartitionReceipt:
    """Proof that exact permutation slices partition the full train frame."""

    registration_hash: str
    train_dataset_identity_hash: str
    train_split_identity_hash: str
    source_context_axis_hash: str
    full_train_population_count: int
    target_population_count: int
    complement_population_count: int
    target_population_membership_hash: str
    complement_population_membership_hash: str
    full_train_membership_hash: str
    target_full_permutation_receipt_hash: str
    complement_full_permutation_receipt_hash: str
    role_slice_boundaries_by_stratum: tuple[tuple[str, int, int, int, int], ...]
    role_counts_by_stratum: tuple[tuple[str, int, int], ...]
    role_membership_hashes: tuple[tuple[str, str], ...]
    full_population_estimand: str = ALEXNET_FQ_FULL_POPULATION_ESTIMAND
    adaptive_prefix_semantics: str = ALEXNET_FQ_PREFIX_SEMANTICS
    role_assignment_is_contiguous_full_permutation_slices: bool = True
    discovery_is_without_replacement_sample_of_full_strata: bool = True
    discovery_terminal_is_full_population_census: bool = False
    upstream_context_promotion_count: int = 0
    exhaustive_partition: bool = True
    duplicate_context_count: int = 0
    population_redefinition_authority: bool = False
    schema: str = ALEXNET_FQ_TRAIN_ROLE_PARTITION_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "registration_hash",
            "train_dataset_identity_hash",
            "train_split_identity_hash",
            "source_context_axis_hash",
            "target_population_membership_hash",
            "complement_population_membership_hash",
            "full_train_membership_hash",
            "target_full_permutation_receipt_hash",
            "complement_full_permutation_receipt_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        full_count = _positive_int(
            self.full_train_population_count,
            name="full train population count",
        )
        target_count = _positive_int(
            self.target_population_count,
            name="target population count",
        )
        complement_count = _positive_int(
            self.complement_population_count,
            name="complement population count",
        )
        if full_count != target_count + complement_count:
            raise ValueError("full train count does not close its two strata")
        role_counts = tuple(self.role_counts_by_stratum)
        if tuple(row[0] for row in role_counts) != (
            "discovery",
            "certification",
            "oracle",
        ):
            raise ValueError("train role counts must name D/C/oracle exactly")
        normalized_counts: list[tuple[str, int, int]] = []
        for role, target, complement in role_counts:
            normalized_counts.append(
                (
                    role,
                    _positive_int(target, name=f"{role} target count"),
                    _positive_int(complement, name=f"{role} complement count"),
                )
            )
        if (
            sum(row[1] for row in normalized_counts) != target_count
            or sum(row[2] for row in normalized_counts) != complement_count
        ):
            raise ValueError("D/C/oracle counts do not exhaust the full train strata")
        boundaries = tuple(tuple(row) for row in self.role_slice_boundaries_by_stratum)
        expected_boundaries = _role_slice_boundaries(
            target_population_count=target_count,
            complement_population_count=complement_count,
        )
        if boundaries != expected_boundaries:
            raise ValueError(
                "train role slice boundaries are not the canonical contiguous cuts"
            )
        expected_role_counts = (
            (
                "discovery",
                expected_boundaries[0][2] - expected_boundaries[0][1],
                expected_boundaries[1][2] - expected_boundaries[1][1],
            ),
            (
                "certification",
                expected_boundaries[0][3] - expected_boundaries[0][2],
                expected_boundaries[1][3] - expected_boundaries[1][2],
            ),
            (
                "oracle",
                expected_boundaries[0][4] - expected_boundaries[0][3],
                expected_boundaries[1][4] - expected_boundaries[1][3],
            ),
        )
        if tuple(normalized_counts) != expected_role_counts:
            raise ValueError(
                "D/C/oracle counts do not match the canonical contiguous cuts"
            )
        role_hashes = tuple(
            (
                _identity(role, name="train role hash name"),
                _identity(value, name=f"{role} membership hash"),
            )
            for role, value in self.role_membership_hashes
        )
        if tuple(role for role, _ in role_hashes) != (
            "discovery",
            "certification",
            "oracle",
        ):
            raise ValueError("train role hashes must name D/C/oracle exactly")
        if (
            self.full_population_estimand != ALEXNET_FQ_FULL_POPULATION_ESTIMAND
            or self.adaptive_prefix_semantics != ALEXNET_FQ_PREFIX_SEMANTICS
            or self.role_assignment_is_contiguous_full_permutation_slices is not True
            or self.discovery_is_without_replacement_sample_of_full_strata is not True
            or self.discovery_terminal_is_full_population_census is not False
            or isinstance(self.upstream_context_promotion_count, bool)
            or self.upstream_context_promotion_count != 0
            or self.exhaustive_partition is not True
            or isinstance(self.duplicate_context_count, bool)
            or self.duplicate_context_count != 0
            or self.population_redefinition_authority is not False
            or self.schema != ALEXNET_FQ_TRAIN_ROLE_PARTITION_RECEIPT_SCHEMA
        ):
            raise PermissionError("train role receipt weakened fixed-population scope")
        object.__setattr__(self, "full_train_population_count", full_count)
        object.__setattr__(self, "target_population_count", target_count)
        object.__setattr__(self, "complement_population_count", complement_count)
        object.__setattr__(self, "role_slice_boundaries_by_stratum", boundaries)
        object.__setattr__(self, "role_counts_by_stratum", tuple(normalized_counts))
        object.__setattr__(self, "role_membership_hashes", role_hashes)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "registration_hash": self.registration_hash,
            "train_dataset_identity_hash": self.train_dataset_identity_hash,
            "train_split_identity_hash": self.train_split_identity_hash,
            "source_context_axis_hash": self.source_context_axis_hash,
            "full_train_population_count": self.full_train_population_count,
            "target_population_count": self.target_population_count,
            "complement_population_count": self.complement_population_count,
            "target_population_membership_hash": (
                self.target_population_membership_hash
            ),
            "complement_population_membership_hash": (
                self.complement_population_membership_hash
            ),
            "full_train_membership_hash": self.full_train_membership_hash,
            "target_full_permutation_receipt_hash": (
                self.target_full_permutation_receipt_hash
            ),
            "complement_full_permutation_receipt_hash": (
                self.complement_full_permutation_receipt_hash
            ),
            "role_slice_boundaries_by_stratum": (self.role_slice_boundaries_by_stratum),
            "role_counts_by_stratum": self.role_counts_by_stratum,
            "role_membership_hashes": self.role_membership_hashes,
            "full_population_estimand": self.full_population_estimand,
            "adaptive_prefix_semantics": self.adaptive_prefix_semantics,
            "role_assignment_is_contiguous_full_permutation_slices": (
                self.role_assignment_is_contiguous_full_permutation_slices
            ),
            "discovery_is_without_replacement_sample_of_full_strata": (
                self.discovery_is_without_replacement_sample_of_full_strata
            ),
            "discovery_terminal_is_full_population_census": (
                self.discovery_terminal_is_full_population_census
            ),
            "upstream_context_promotion_count": self.upstream_context_promotion_count,
            "exhaustive_partition": self.exhaustive_partition,
            "duplicate_context_count": self.duplicate_context_count,
            "population_redefinition_authority": (
                self.population_redefinition_authority
            ),
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_TRAIN_ROLE_PARTITION_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("train role partition receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("train role partition receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    def __getstate__(self) -> tuple[object, ...]:
        return _exact_slot_state(self)

    def __setstate__(self, state: object) -> None:
        _restore_exact_slot_state(
            self,
            state,
            artifact="train-role-partition receipt",
        )

    def assert_realization(
        self,
        *,
        target_assignment_receipt: SealedUniformPermutationReceipt,
        complement_assignment_receipt: SealedUniformPermutationReceipt,
        discovery: Sequence[tuple[str, Sequence[str]]],
        certification: Sequence[tuple[str, Sequence[str]]],
        oracle: Sequence[tuple[str, Sequence[str]]],
    ) -> None:
        if (
            type(target_assignment_receipt) is not SealedUniformPermutationReceipt
            or type(complement_assignment_receipt)
            is not SealedUniformPermutationReceipt
        ):
            raise TypeError("role realization requires exact full-permutation receipts")
        target_assignment_receipt.verify()
        complement_assignment_receipt.verify()
        if (
            target_assignment_receipt.content_hash
            != self.target_full_permutation_receipt_hash
            or complement_assignment_receipt.content_hash
            != self.complement_full_permutation_receipt_hash
        ):
            raise ValueError("role realization uses the wrong full permutations")
        roles = (
            ("discovery", _role_rows(discovery, role="discovery")),
            ("certification", _role_rows(certification, role="certification")),
            ("oracle", _role_rows(oracle, role="oracle")),
        )
        counts = tuple((role, len(rows[0][1]), len(rows[1][1])) for role, rows in roles)
        hashes = tuple(
            (
                role,
                content_hash(rows, prefix=_TRAIN_ROLE_MEMBERSHIP_PREFIX + role + "-"),
            )
            for role, rows in roles
        )
        flat = tuple(
            context
            for _role, rows in roles
            for _stratum, contexts in rows
            for context in contexts
        )
        target_slices = contiguous_three_role_slices(
            target_assignment_receipt.generated_order
        )
        complement_slices = contiguous_three_role_slices(
            complement_assignment_receipt.generated_order
        )
        expected_roles = tuple(
            (
                ("target", target_slices[role_index]),
                ("complement", complement_slices[role_index]),
            )
            for role_index in range(3)
        )
        if (
            counts != self.role_counts_by_stratum
            or hashes != self.role_membership_hashes
            or tuple(rows for _role, rows in roles) != expected_roles
            or len(flat) != self.full_train_population_count
            or len(flat) != len(set(flat))
            or content_hash(
                tuple(sorted(flat)),
                prefix=_FULL_TRAIN_MEMBERSHIP_PREFIX,
            )
            != self.full_train_membership_hash
        ):
            raise ValueError(
                "D/C/oracle realization is not the registered full train partition"
            )

    @classmethod
    def create(
        cls,
        *,
        registration: "V1AlexNetFQAcquisitionRegistration",
        target_context_ids: Sequence[str],
        complement_context_ids: Sequence[str],
        target_assignment_receipt: SealedUniformPermutationReceipt,
        complement_assignment_receipt: SealedUniformPermutationReceipt,
        discovery: Sequence[tuple[str, Sequence[str]]],
        certification: Sequence[tuple[str, Sequence[str]]],
        oracle: Sequence[tuple[str, Sequence[str]]],
    ) -> "V1AlexNetTrainRolePartitionReceipt":
        if type(registration) is not V1AlexNetFQAcquisitionRegistration:
            raise TypeError("train role receipt requires its exact registration")
        target = tuple(
            _identity(value, name="target population context")
            for value in target_context_ids
        )
        complement = tuple(
            _identity(value, name="complement population context")
            for value in complement_context_ids
        )
        full = target + complement
        if (
            len(target) != len(set(target))
            or len(complement) != len(set(complement))
            or len(full) != len(set(full))
        ):
            raise ValueError("registered full train strata contain duplicate contexts")
        if (
            len(target) != registration.target_population_count
            or len(complement) != registration.complement_population_count
            or content_hash(
                target,
                prefix="partition-free-v1-alexnet-fq-target-membership-",
            )
            != registration.target_population_membership_hash
            or content_hash(
                complement,
                prefix="partition-free-v1-alexnet-fq-complement-membership-",
            )
            != registration.complement_population_membership_hash
        ):
            raise ValueError("runtime train strata differ from the F/Q registration")
        normalized_roles = tuple(
            (role, _role_rows(rows, role=role))
            for role, rows in (
                ("discovery", discovery),
                ("certification", certification),
                ("oracle", oracle),
            )
        )
        role_flat = tuple(
            context
            for _role, rows in normalized_roles
            for _stratum, contexts in rows
            for context in contexts
        )
        if (
            len(role_flat) != len(full)
            or len(role_flat) != len(set(role_flat))
            or set(role_flat) != set(full)
        ):
            raise ValueError("D/C/oracle roles do not uniquely exhaust full train")
        if (
            type(target_assignment_receipt) is not SealedUniformPermutationReceipt
            or type(complement_assignment_receipt)
            is not SealedUniformPermutationReceipt
        ):
            raise TypeError("role receipt requires exact full-permutation receipts")
        target_assignment_receipt.verify()
        complement_assignment_receipt.verify()
        if target_assignment_receipt.canonical_population != tuple(
            sorted(target)
        ) or complement_assignment_receipt.canonical_population != tuple(
            sorted(complement)
        ):
            raise ValueError("role receipt full permutations use the wrong populations")
        result = cls(
            registration_hash=registration.registration_hash,
            train_dataset_identity_hash=registration.train_dataset_identity_hash,
            train_split_identity_hash=registration.train_split_identity_hash,
            source_context_axis_hash=registration.source_context_axis_hash,
            full_train_population_count=len(full),
            target_population_count=len(target),
            complement_population_count=len(complement),
            target_population_membership_hash=(
                registration.target_population_membership_hash
            ),
            complement_population_membership_hash=(
                registration.complement_population_membership_hash
            ),
            full_train_membership_hash=content_hash(
                tuple(sorted(full)),
                prefix=_FULL_TRAIN_MEMBERSHIP_PREFIX,
            ),
            target_full_permutation_receipt_hash=(
                target_assignment_receipt.content_hash
            ),
            complement_full_permutation_receipt_hash=(
                complement_assignment_receipt.content_hash
            ),
            role_slice_boundaries_by_stratum=_role_slice_boundaries(
                target_population_count=len(target),
                complement_population_count=len(complement),
            ),
            role_counts_by_stratum=tuple(
                (role, len(rows[0][1]), len(rows[1][1]))
                for role, rows in normalized_roles
            ),
            role_membership_hashes=tuple(
                (
                    role,
                    content_hash(
                        rows,
                        prefix=_TRAIN_ROLE_MEMBERSHIP_PREFIX + role + "-",
                    ),
                )
                for role, rows in normalized_roles
            ),
        )
        result.assert_realization(
            target_assignment_receipt=target_assignment_receipt,
            complement_assignment_receipt=complement_assignment_receipt,
            discovery=discovery,
            certification=certification,
            oracle=oracle,
        )
        return result


@dataclass(frozen=True, slots=True)
class V1AlexNetFQFoldRealization:
    target_assignment_receipt: SealedUniformPermutationReceipt = field(repr=False)
    complement_assignment_receipt: SealedUniformPermutationReceipt = field(repr=False)
    discovery_context_ids_by_stratum: tuple[tuple[str, tuple[str, ...]], ...]
    certification_context_ids_by_stratum: tuple[tuple[str, tuple[str, ...]], ...]
    oracle_context_ids_by_stratum: tuple[tuple[str, tuple[str, ...]], ...]
    discovery_cumulative_stratum_counts: tuple[tuple[int, int], ...]
    certification_cumulative_stratum_counts: tuple[tuple[int, int], ...]
    oracle_cumulative_stratum_counts: tuple[tuple[int, int], ...]
    train_role_partition_receipt: V1AlexNetTrainRolePartitionReceipt

    def __post_init__(self) -> None:
        if (
            type(self.target_assignment_receipt) is not SealedUniformPermutationReceipt
            or type(self.complement_assignment_receipt)
            is not SealedUniformPermutationReceipt
            or type(self.train_role_partition_receipt)
            is not V1AlexNetTrainRolePartitionReceipt
        ):
            raise TypeError("F/Q folds require exact assignment and role receipts")
        self.target_assignment_receipt.verify()
        self.complement_assignment_receipt.verify()
        self.train_role_partition_receipt.assert_integrity()
        roles = (
            _role_rows(
                self.discovery_context_ids_by_stratum,
                role="discovery",
            ),
            _role_rows(
                self.certification_context_ids_by_stratum,
                role="certification",
            ),
            _role_rows(self.oracle_context_ids_by_stratum, role="oracle"),
        )
        self.train_role_partition_receipt.assert_realization(
            target_assignment_receipt=self.target_assignment_receipt,
            complement_assignment_receipt=self.complement_assignment_receipt,
            discovery=roles[0],
            certification=roles[1],
            oracle=roles[2],
        )
        canonical_by_stratum = (
            self.target_assignment_receipt.canonical_population,
            self.complement_assignment_receipt.canonical_population,
        )
        for stratum_index, canonical in enumerate(canonical_by_stratum):
            realized = tuple(
                sorted(
                    context
                    for role_rows in roles
                    for context in role_rows[stratum_index][1]
                )
            )
            if realized != canonical:
                raise ValueError(
                    "F/Q roles differ from the sealed full train population"
                )

        def validate_looks(
            value: Sequence[tuple[int, int]],
            *,
            role_rows: tuple[tuple[str, tuple[str, ...]], ...],
            role: str,
        ) -> tuple[tuple[int, int], ...]:
            looks = tuple(tuple(row) for row in value)
            final = (len(role_rows[0][1]), len(role_rows[1][1]))
            prior = (0, 0)
            for row in looks:
                if (
                    len(row) != 2
                    or any(
                        isinstance(count, bool)
                        or not isinstance(count, int)
                        or count <= 0
                        for count in row
                    )
                    or row[0] < prior[0]
                    or row[1] < prior[1]
                    or row[0] > final[0]
                    or row[1] > final[1]
                ):
                    raise ValueError(f"{role} adaptive prefix schedule is invalid")
                prior = row
            if not looks or looks[-1] != final:
                raise ValueError(
                    f"{role} adaptive prefixes do not end at the registered capacity"
                )
            return looks  # type: ignore[return-value]

        object.__setattr__(self, "discovery_context_ids_by_stratum", roles[0])
        object.__setattr__(self, "certification_context_ids_by_stratum", roles[1])
        object.__setattr__(self, "oracle_context_ids_by_stratum", roles[2])
        object.__setattr__(
            self,
            "discovery_cumulative_stratum_counts",
            validate_looks(
                self.discovery_cumulative_stratum_counts,
                role_rows=roles[0],
                role="discovery",
            ),
        )

        object.__setattr__(
            self,
            "certification_cumulative_stratum_counts",
            validate_looks(
                self.certification_cumulative_stratum_counts,
                role_rows=roles[1],
                role="certification",
            ),
        )
        object.__setattr__(
            self,
            "oracle_cumulative_stratum_counts",
            validate_looks(
                self.oracle_cumulative_stratum_counts,
                role_rows=roles[2],
                role="oracle",
            ),
        )

    @property
    def full_population_stratum_counts(self) -> tuple[int, int]:
        return (
            len(self.target_assignment_receipt.canonical_population),
            len(self.complement_assignment_receipt.canonical_population),
        )

    @property
    def discovery_sampling_capacity_stratum_counts(self) -> tuple[int, int]:
        return (
            len(self.discovery_context_ids_by_stratum[0][1]),
            len(self.discovery_context_ids_by_stratum[1][1]),
        )

    @property
    def discovery_terminal_is_full_population_census(self) -> bool:
        return False

    @property
    def dynamic_estimand_authority(self) -> bool:
        """Static folds are a population/cut template, not dynamic authority."""

        return False

    @property
    def realization_hash(self) -> str:
        return content_hash(
            {
                "schema": ALEXNET_FQ_FOLD_REALIZATION_SCHEMA,
                "target_assignment_receipt_hash": (
                    self.target_assignment_receipt.content_hash
                ),
                "complement_assignment_receipt_hash": (
                    self.complement_assignment_receipt.content_hash
                ),
                "discovery_context_ids_by_stratum": (
                    self.discovery_context_ids_by_stratum
                ),
                "certification_context_ids_by_stratum": (
                    self.certification_context_ids_by_stratum
                ),
                "oracle_context_ids_by_stratum": (self.oracle_context_ids_by_stratum),
                "discovery_cumulative_stratum_counts": (
                    self.discovery_cumulative_stratum_counts
                ),
                "certification_cumulative_stratum_counts": (
                    self.certification_cumulative_stratum_counts
                ),
                "oracle_cumulative_stratum_counts": (
                    self.oracle_cumulative_stratum_counts
                ),
                "train_role_partition_receipt_hash": (
                    self.train_role_partition_receipt.receipt_hash
                ),
                "full_population_stratum_counts": (self.full_population_stratum_counts),
                "discovery_sampling_capacity_stratum_counts": (
                    self.discovery_sampling_capacity_stratum_counts
                ),
                "discovery_terminal_is_full_population_census": False,
                "dynamic_estimand_authority": False,
                "full_population_estimand": ALEXNET_FQ_FULL_POPULATION_ESTIMAND,
                "adaptive_prefix_semantics": ALEXNET_FQ_PREFIX_SEMANTICS,
            },
            prefix=_FOLD_REALIZATION_PREFIX,
        )

    def __getstate__(self) -> tuple[object, ...]:
        return _exact_slot_state(self)

    def __setstate__(self, state: object) -> None:
        _restore_exact_slot_state(self, state, artifact="F/Q fold realization")


@dataclass(frozen=True, slots=True)
class V1AlexNetFQCausalGenerationFrameReceipt:
    """Outcome-independent state boundary from which role streams may be opened."""

    registration_hash: str
    generation_index: int
    current_mask_hash: str
    frozen_transform_hash: str
    target_population_count: int
    complement_population_count: int
    target_population_membership_hash: str
    complement_population_membership_hash: str
    causal_role_policy: str = ALEXNET_FQ_CAUSAL_ROLE_POLICY
    randomization_assumption: str = ALEXNET_FQ_CAUSAL_RANDOMIZATION_ASSUMPTION
    full_population_estimand: str = ALEXNET_FQ_FULL_POPULATION_ESTIMAND
    registered_before_role_outcomes: bool = True
    role_streams_initially_closed: bool = True
    legacy_residual_cut_authority: bool = False
    validation_rows_seen: bool = False
    schema: str = ALEXNET_FQ_CAUSAL_GENERATION_FRAME_SCHEMA
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "registration_hash",
            "current_mask_hash",
            "frozen_transform_hash",
            "target_population_membership_hash",
            "complement_population_membership_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        generation = _nonnegative_int(
            self.generation_index,
            name="causal generation index",
        )
        target = _positive_int(
            self.target_population_count,
            name="causal target population count",
        )
        complement = _positive_int(
            self.complement_population_count,
            name="causal complement population count",
        )
        if min(target, complement) < 2:
            raise ValueError(
                "causal K/O separation requires at least two contexts per stratum"
            )
        if (
            self.causal_role_policy != ALEXNET_FQ_CAUSAL_ROLE_POLICY
            or self.randomization_assumption
            != ALEXNET_FQ_CAUSAL_RANDOMIZATION_ASSUMPTION
            or self.full_population_estimand != ALEXNET_FQ_FULL_POPULATION_ESTIMAND
            or self.registered_before_role_outcomes is not True
            or self.role_streams_initially_closed is not True
            or self.legacy_residual_cut_authority is not False
            or self.validation_rows_seen is not False
            or self.schema != ALEXNET_FQ_CAUSAL_GENERATION_FRAME_SCHEMA
        ):
            raise PermissionError("causal generation frame weakened its opening law")
        object.__setattr__(self, "generation_index", generation)
        object.__setattr__(self, "target_population_count", target)
        object.__setattr__(self, "complement_population_count", complement)
        self.assert_integrity()

    @property
    def full_population_stratum_counts(self) -> tuple[int, int]:
        return self.target_population_count, self.complement_population_count

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "_receipt_hash"
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(
            self._payload(),
            prefix=_CAUSAL_GENERATION_FRAME_PREFIX,
        )
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("causal generation frame receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("causal generation frame receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    def __getstate__(self) -> tuple[object, ...]:
        return _exact_slot_state(self)

    def __setstate__(self, state: object) -> None:
        _restore_exact_slot_state(self, state, artifact="causal generation frame")


@dataclass(frozen=True, slots=True)
class V1AlexNetFQCausalRoleSamplingStreamReceipt:
    """One domain-separated D, C, or K full-population permutation pair."""

    registration_hash: str
    generation_frame_hash: str
    generation_index: int
    role: str
    causal_parent_receipt_hash: str
    current_mask_hash: str
    frozen_transform_hash: str
    target_assignment_receipt: SealedUniformPermutationReceipt = field(repr=False)
    complement_assignment_receipt: SealedUniformPermutationReceipt = field(
        repr=False
    )
    target_population_membership_hash: str = ""
    complement_population_membership_hash: str = ""
    sampling_capacity_stratum_counts: tuple[int, int] = (1, 1)
    discovery_prefix_freeze_hash: str | None = None
    selected_shell_hash: str | None = None
    certification_prefix_freeze_hash: str | None = None
    candidate_freeze_hash: str | None = None
    dense_referent_mask_hash: str | None = None
    candidate_mask_hash: str | None = None
    candidate_attempt_index: int | None = None
    prefix_semantics: str = ALEXNET_FQ_FULL_FRAME_PREFIX_SEMANTICS
    causal_role_policy: str = ALEXNET_FQ_CAUSAL_ROLE_POLICY
    randomization_assumption: str = ALEXNET_FQ_CAUSAL_RANDOMIZATION_ASSUMPTION
    complete_train_population_permuted: bool = True
    domain_separated_from_other_roles: bool = True
    opened_after_causal_parent_freeze: bool = True
    sampling_order_outcome_independent: bool = True
    candidate_or_mask_authority: bool = False
    validation_rows_seen: bool = False
    schema: str = ALEXNET_FQ_CAUSAL_ROLE_STREAM_SCHEMA
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "registration_hash",
            "generation_frame_hash",
            "causal_parent_receipt_hash",
            "current_mask_hash",
            "frozen_transform_hash",
            "target_population_membership_hash",
            "complement_population_membership_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if self.role not in {"discovery", "certification", "calibration"}:
            raise ValueError("causal role stream must be D, C, or K")
        generation = _nonnegative_int(
            self.generation_index,
            name="causal role generation index",
        )
        if (
            type(self.target_assignment_receipt)
            is not SealedUniformPermutationReceipt
            or type(self.complement_assignment_receipt)
            is not SealedUniformPermutationReceipt
        ):
            raise TypeError("causal role stream requires exact permutation receipts")
        verify_sealed_uniform_permutation_custody(
            self.target_assignment_receipt
        )
        verify_sealed_uniform_permutation_custody(
            self.complement_assignment_receipt
        )
        target_count = len(self.target_assignment_receipt.canonical_population)
        complement_count = len(
            self.complement_assignment_receipt.canonical_population
        )
        capacities = _stratum_counts(
            self.sampling_capacity_stratum_counts,
            name="causal role sampling capacity",
        )
        expected_capacities = (
            (target_count - 1, complement_count - 1)
            if self.role == "calibration"
            else (target_count, complement_count)
        )
        if capacities != expected_capacities or min(expected_capacities) <= 0:
            raise ValueError("causal role capacity differs from its population law")

        discovery_freeze = self.discovery_prefix_freeze_hash
        selected_shell = self.selected_shell_hash
        certification_freeze = self.certification_prefix_freeze_hash
        candidate_freeze = self.candidate_freeze_hash
        dense_mask = self.dense_referent_mask_hash
        candidate_mask = self.candidate_mask_hash
        attempt = self.candidate_attempt_index
        for name, value in (
            ("discovery_prefix_freeze_hash", discovery_freeze),
            ("selected_shell_hash", selected_shell),
            ("certification_prefix_freeze_hash", certification_freeze),
            ("candidate_freeze_hash", candidate_freeze),
            ("dense_referent_mask_hash", dense_mask),
            ("candidate_mask_hash", candidate_mask),
        ):
            if value is not None:
                object.__setattr__(self, name, _identity(value, name=name))
        if attempt is not None:
            object.__setattr__(
                self,
                "candidate_attempt_index",
                _nonnegative_int(attempt, name="candidate attempt index"),
            )
        if self.role == "discovery":
            if any(
                value is not None
                for value in (
                    discovery_freeze,
                    selected_shell,
                    certification_freeze,
                    candidate_freeze,
                    dense_mask,
                    candidate_mask,
                    attempt,
                )
            ):
                raise ValueError("D stream cannot cross a later causal phase")
        elif self.role == "certification":
            if (
                discovery_freeze is None
                or selected_shell is None
                or any(
                    value is not None
                    for value in (
                        certification_freeze,
                        candidate_freeze,
                        dense_mask,
                        candidate_mask,
                        attempt,
                    )
                )
            ):
                raise ValueError("C stream requires only the frozen D shell")
        elif (
            discovery_freeze is None
            or selected_shell is None
            or certification_freeze is None
            or candidate_freeze is None
            or dense_mask is None
            or candidate_mask is None
            or attempt is None
            or dense_mask == candidate_mask
        ):
            raise ValueError("K stream requires the exact frozen candidate lineage")
        if (
            self.prefix_semantics != ALEXNET_FQ_FULL_FRAME_PREFIX_SEMANTICS
            or self.causal_role_policy != ALEXNET_FQ_CAUSAL_ROLE_POLICY
            or self.randomization_assumption
            != ALEXNET_FQ_CAUSAL_RANDOMIZATION_ASSUMPTION
            or self.complete_train_population_permuted is not True
            or self.domain_separated_from_other_roles is not True
            or self.opened_after_causal_parent_freeze is not True
            or self.sampling_order_outcome_independent is not True
            or self.candidate_or_mask_authority is not False
            or self.validation_rows_seen is not False
            or self.schema != ALEXNET_FQ_CAUSAL_ROLE_STREAM_SCHEMA
        ):
            raise PermissionError("causal role stream weakened its sampling law")
        object.__setattr__(self, "generation_index", generation)
        object.__setattr__(self, "sampling_capacity_stratum_counts", capacities)
        self.assert_integrity()

    @property
    def full_population_stratum_counts(self) -> tuple[int, int]:
        return (
            len(self.target_assignment_receipt.canonical_population),
            len(self.complement_assignment_receipt.canonical_population),
        )

    @property
    def context_ids_by_stratum(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return (
            ("target", self.target_assignment_receipt.generated_order),
            ("complement", self.complement_assignment_receipt.generated_order),
        )

    @property
    def terminal_is_complete_train_population_census(self) -> bool:
        return self.role in {"discovery", "certification"}

    @property
    def append_only_prefix_stream(self) -> bool:
        return self.role == "calibration"

    def _payload(self) -> dict[str, object]:
        return {
            name: (
                getattr(self, name).content_hash
                if name
                in {"target_assignment_receipt", "complement_assignment_receipt"}
                else getattr(self, name)
            )
            for name in self.__dataclass_fields__
            if name != "_receipt_hash"
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        verify_sealed_uniform_permutation_custody(
            self.target_assignment_receipt
        )
        verify_sealed_uniform_permutation_custody(
            self.complement_assignment_receipt
        )
        observed = content_hash(self._payload(), prefix=_CAUSAL_ROLE_STREAM_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("causal role stream receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("causal role stream receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    def __getstate__(self) -> tuple[object, ...]:
        return _exact_slot_state(self)

    def __setstate__(self, state: object) -> None:
        _restore_exact_slot_state(self, state, artifact="causal role stream")


@dataclass(frozen=True, slots=True)
class V1AlexNetFQDiscoveryPrefixFreezeReceipt:
    generation_frame_hash: str
    discovery_stream_hash: str
    generation_index: int
    cumulative_stratum_counts: tuple[int, int]
    target_prefix_membership_hash: str
    complement_prefix_membership_hash: str
    discovery_evidence_state_hash: str
    selected_shell_hash: str
    prefix_is_adaptive_full_population_sample: bool = True
    candidate_or_mask_authority: bool = False
    validation_rows_seen: bool = False
    schema: str = ALEXNET_FQ_DISCOVERY_PREFIX_FREEZE_SCHEMA
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "generation_frame_hash",
            "discovery_stream_hash",
            "target_prefix_membership_hash",
            "complement_prefix_membership_hash",
            "discovery_evidence_state_hash",
            "selected_shell_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "generation_index",
            _nonnegative_int(self.generation_index, name="D freeze generation index"),
        )
        object.__setattr__(
            self,
            "cumulative_stratum_counts",
            _stratum_counts(self.cumulative_stratum_counts, name="D freeze counts"),
        )
        if (
            self.prefix_is_adaptive_full_population_sample is not True
            or self.candidate_or_mask_authority is not False
            or self.validation_rows_seen is not False
            or self.schema != ALEXNET_FQ_DISCOVERY_PREFIX_FREEZE_SCHEMA
        ):
            raise PermissionError("D prefix freeze weakened its authority firewall")
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "_receipt_hash"
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_DISCOVERY_PREFIX_FREEZE_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("D prefix freeze receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("D prefix freeze receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    def __getstate__(self) -> tuple[object, ...]:
        return _exact_slot_state(self)

    def __setstate__(self, state: object) -> None:
        _restore_exact_slot_state(self, state, artifact="D prefix freeze")


@dataclass(frozen=True, slots=True)
class V1AlexNetFQCertificationPrefixFreezeReceipt:
    generation_frame_hash: str
    discovery_prefix_freeze_hash: str
    certification_stream_hash: str
    generation_index: int
    cumulative_stratum_counts: tuple[int, int]
    target_prefix_membership_hash: str
    complement_prefix_membership_hash: str
    selected_shell_hash: str
    certification_evidence_state_hash: str
    candidate_freeze_hash: str
    candidate_mask_hash: str
    candidate_frozen_before_calibration: bool = True
    oracle_rows_seen: bool = False
    validation_rows_seen: bool = False
    schema: str = ALEXNET_FQ_CERTIFICATION_PREFIX_FREEZE_SCHEMA
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "generation_frame_hash",
            "discovery_prefix_freeze_hash",
            "certification_stream_hash",
            "target_prefix_membership_hash",
            "complement_prefix_membership_hash",
            "selected_shell_hash",
            "certification_evidence_state_hash",
            "candidate_freeze_hash",
            "candidate_mask_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "generation_index",
            _nonnegative_int(self.generation_index, name="C freeze generation index"),
        )
        object.__setattr__(
            self,
            "cumulative_stratum_counts",
            _stratum_counts(self.cumulative_stratum_counts, name="C freeze counts"),
        )
        if (
            self.candidate_frozen_before_calibration is not True
            or self.oracle_rows_seen is not False
            or self.validation_rows_seen is not False
            or self.schema != ALEXNET_FQ_CERTIFICATION_PREFIX_FREEZE_SCHEMA
        ):
            raise PermissionError("C prefix freeze crossed its causal boundary")
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "_receipt_hash"
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(
            self._payload(),
            prefix=_CERTIFICATION_PREFIX_FREEZE_PREFIX,
        )
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("C prefix freeze receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("C prefix freeze receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    def __getstate__(self) -> tuple[object, ...]:
        return _exact_slot_state(self)

    def __setstate__(self, state: object) -> None:
        _restore_exact_slot_state(self, state, artifact="C prefix freeze")


@dataclass(frozen=True, slots=True)
class V1AlexNetFQCalibrationPrefixFreezeReceipt:
    generation_frame_hash: str
    certification_prefix_freeze_hash: str
    calibration_stream_hash: str
    generation_index: int
    candidate_attempt_index: int
    look_index: int
    prior_prefix_freeze_hash: str | None
    cumulative_stratum_counts: tuple[int, int]
    target_prefix_membership_hash: str
    complement_prefix_membership_hash: str
    candidate_freeze_hash: str
    dense_referent_mask_hash: str
    candidate_mask_hash: str
    dense_calibration_hash: str
    candidate_calibration_hash: str
    calibration_evidence_state_hash: str
    append_only_prefix: bool = True
    calibration_frozen_before_oracle_open: bool = True
    oracle_rows_seen: bool = False
    validation_rows_seen: bool = False
    schema: str = ALEXNET_FQ_CALIBRATION_PREFIX_FREEZE_SCHEMA
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "generation_frame_hash",
            "certification_prefix_freeze_hash",
            "calibration_stream_hash",
            "target_prefix_membership_hash",
            "complement_prefix_membership_hash",
            "candidate_freeze_hash",
            "dense_referent_mask_hash",
            "candidate_mask_hash",
            "dense_calibration_hash",
            "candidate_calibration_hash",
            "calibration_evidence_state_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if self.prior_prefix_freeze_hash is not None:
            object.__setattr__(
                self,
                "prior_prefix_freeze_hash",
                _identity(
                    self.prior_prefix_freeze_hash,
                    name="prior calibration prefix freeze hash",
                ),
            )
        object.__setattr__(
            self,
            "generation_index",
            _nonnegative_int(self.generation_index, name="K freeze generation index"),
        )
        object.__setattr__(
            self,
            "candidate_attempt_index",
            _nonnegative_int(
                self.candidate_attempt_index,
                name="K freeze candidate attempt index",
            ),
        )
        look = _positive_int(self.look_index, name="K freeze look index")
        counts = _stratum_counts(
            self.cumulative_stratum_counts,
            name="K freeze counts",
        )
        if (
            (look == 1) != (self.prior_prefix_freeze_hash is None)
            or self.dense_referent_mask_hash == self.candidate_mask_hash
        ):
            raise ValueError("K prefix freeze lineage is incoherent")
        if (
            self.append_only_prefix is not True
            or self.calibration_frozen_before_oracle_open is not True
            or self.oracle_rows_seen is not False
            or self.validation_rows_seen is not False
            or self.schema != ALEXNET_FQ_CALIBRATION_PREFIX_FREEZE_SCHEMA
        ):
            raise PermissionError("K prefix freeze crossed its causal boundary")
        object.__setattr__(self, "look_index", look)
        object.__setattr__(self, "cumulative_stratum_counts", counts)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "_receipt_hash"
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(
            self._payload(),
            prefix=_CALIBRATION_PREFIX_FREEZE_PREFIX,
        )
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("K prefix freeze receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("K prefix freeze receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    def __getstate__(self) -> tuple[object, ...]:
        return _exact_slot_state(self)

    def __setstate__(self, state: object) -> None:
        _restore_exact_slot_state(self, state, artifact="K prefix freeze")


@dataclass(frozen=True, slots=True)
class V1AlexNetFQOracleSamplingFrameReceipt:
    """Candidate-bound O order on the exact out-of-fit population ``U \\ K``."""

    registration_hash: str
    generation_frame_hash: str
    certification_prefix_freeze_hash: str
    calibration_stream_hash: str
    calibration_prefix_freeze_hash: str
    generation_index: int
    candidate_attempt_index: int
    candidate_freeze_hash: str
    dense_referent_mask_hash: str
    candidate_mask_hash: str
    dense_calibration_hash: str
    candidate_calibration_hash: str
    target_full_population_membership_hash: str
    complement_full_population_membership_hash: str
    calibration_prefix_context_ids_by_stratum: tuple[
        tuple[str, tuple[str, ...]], ...
    ]
    target_assignment_receipt: SealedUniformPermutationReceipt = field(repr=False)
    complement_assignment_receipt: SealedUniformPermutationReceipt = field(
        repr=False
    )
    full_population_stratum_counts: tuple[int, int] = (1, 1)
    calibration_prefix_stratum_counts: tuple[int, int] = (1, 1)
    oracle_population_stratum_counts: tuple[int, int] = (1, 1)
    estimand: str = ALEXNET_FQ_OUT_OF_FIT_ORACLE_ESTIMAND
    causal_role_policy: str = ALEXNET_FQ_CAUSAL_ROLE_POLICY
    randomization_assumption: str = ALEXNET_FQ_CAUSAL_RANDOMIZATION_ASSUMPTION
    oracle_population_is_exact_full_minus_k: bool = True
    oracle_disjoint_from_calibration_prefix: bool = True
    order_opened_after_candidate_and_calibration_freeze: bool = True
    terminal_is_out_of_fit_population_census: bool = True
    terminal_is_complete_train_population_census: bool = False
    oracle_outcomes_seen_before_open: bool = False
    validation_rows_seen: bool = False
    schema: str = ALEXNET_FQ_ORACLE_SAMPLING_FRAME_SCHEMA
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "registration_hash",
            "generation_frame_hash",
            "certification_prefix_freeze_hash",
            "calibration_stream_hash",
            "calibration_prefix_freeze_hash",
            "candidate_freeze_hash",
            "dense_referent_mask_hash",
            "candidate_mask_hash",
            "dense_calibration_hash",
            "candidate_calibration_hash",
            "target_full_population_membership_hash",
            "complement_full_population_membership_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        generation = _nonnegative_int(
            self.generation_index,
            name="O frame generation index",
        )
        attempt = _nonnegative_int(
            self.candidate_attempt_index,
            name="O frame candidate attempt index",
        )
        if (
            type(self.target_assignment_receipt)
            is not SealedUniformPermutationReceipt
            or type(self.complement_assignment_receipt)
            is not SealedUniformPermutationReceipt
        ):
            raise TypeError("O frame requires exact permutation receipts")
        verify_sealed_uniform_permutation_custody(self.target_assignment_receipt)
        verify_sealed_uniform_permutation_custody(self.complement_assignment_receipt)
        k_rows = _role_rows(
            self.calibration_prefix_context_ids_by_stratum,
            role="calibration",
        )
        full_counts = _stratum_counts(
            self.full_population_stratum_counts,
            name="O frame full-population counts",
        )
        k_counts = _stratum_counts(
            self.calibration_prefix_stratum_counts,
            name="O frame K counts",
        )
        o_counts = _stratum_counts(
            self.oracle_population_stratum_counts,
            name="O frame population counts",
        )
        oracle_populations = (
            self.target_assignment_receipt.canonical_population,
            self.complement_assignment_receipt.canonical_population,
        )
        full_population_membership_hashes = (
            self.target_full_population_membership_hash,
            self.complement_full_population_membership_hash,
        )
        for index in (0, 1):
            if (
                len(k_rows[index][1]) != k_counts[index]
                or len(oracle_populations[index]) != o_counts[index]
                or k_counts[index] + o_counts[index] != full_counts[index]
                or set(k_rows[index][1]).intersection(oracle_populations[index])
            ):
                raise ValueError("O frame is not the exact out-of-fit complement")
            expected_membership_hash = alexnet_fq_full_population_membership_hash(
                stratum=("target", "complement")[index],
                context_ids=tuple(k_rows[index][1]) + tuple(oracle_populations[index]),
            )
            if expected_membership_hash != full_population_membership_hashes[index]:
                raise ValueError(
                    "O frame full population membership differs from K/O union"
                )
        if self.dense_referent_mask_hash == self.candidate_mask_hash:
            raise ValueError("O frame candidate must differ from dense referent")
        if (
            self.estimand != ALEXNET_FQ_OUT_OF_FIT_ORACLE_ESTIMAND
            or self.causal_role_policy != ALEXNET_FQ_CAUSAL_ROLE_POLICY
            or self.randomization_assumption
            != ALEXNET_FQ_CAUSAL_RANDOMIZATION_ASSUMPTION
            or self.oracle_population_is_exact_full_minus_k is not True
            or self.oracle_disjoint_from_calibration_prefix is not True
            or self.order_opened_after_candidate_and_calibration_freeze is not True
            or self.terminal_is_out_of_fit_population_census is not True
            or self.terminal_is_complete_train_population_census is not False
            or self.oracle_outcomes_seen_before_open is not False
            or self.validation_rows_seen is not False
            or self.schema != ALEXNET_FQ_ORACLE_SAMPLING_FRAME_SCHEMA
        ):
            raise PermissionError("O frame weakened its out-of-fit causal law")
        object.__setattr__(self, "generation_index", generation)
        object.__setattr__(self, "candidate_attempt_index", attempt)
        object.__setattr__(self, "calibration_prefix_context_ids_by_stratum", k_rows)
        object.__setattr__(self, "full_population_stratum_counts", full_counts)
        object.__setattr__(self, "calibration_prefix_stratum_counts", k_counts)
        object.__setattr__(self, "oracle_population_stratum_counts", o_counts)
        self.assert_integrity()

    @property
    def oracle_context_ids_by_stratum(
        self,
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return (
            ("target", self.target_assignment_receipt.generated_order),
            ("complement", self.complement_assignment_receipt.generated_order),
        )

    @property
    def realization_hash(self) -> str:
        return self.receipt_hash

    def _payload(self) -> dict[str, object]:
        return {
            name: (
                getattr(self, name).content_hash
                if name
                in {"target_assignment_receipt", "complement_assignment_receipt"}
                else getattr(self, name)
            )
            for name in self.__dataclass_fields__
            if name != "_receipt_hash"
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        verify_sealed_uniform_permutation_custody(self.target_assignment_receipt)
        verify_sealed_uniform_permutation_custody(self.complement_assignment_receipt)
        observed = content_hash(self._payload(), prefix=_OUT_OF_FIT_ORACLE_FRAME_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("O sampling frame receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("O sampling frame receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    def __getstate__(self) -> tuple[object, ...]:
        return _exact_slot_state(self)

    def __setstate__(self, state: object) -> None:
        _restore_exact_slot_state(self, state, artifact="O sampling frame")


@dataclass(frozen=True, slots=True)
class V1AlexNetFQGenerationSamplingFrameReceipt:
    """Legacy state-bound contiguous-cut compatibility receipt.

    ``fold_realization`` carries the exact full permutations and contiguous
    role slices.  It is retained for artifact replay only.  Dynamic production
    must use :class:`V1AlexNetFQCausalGenerationFrameReceipt` and the causally
    opened role streams above.
    """

    registration_hash: str
    generation_index: int
    current_mask_hash: str
    frozen_transform_hash: str
    fold_realization: V1AlexNetFQFoldRealization = field(repr=False)
    full_population_estimand: str = ALEXNET_FQ_FULL_POPULATION_ESTIMAND
    state_bound_before_permutation_derivation: bool = True
    upstream_context_promotion_count: int = 0
    discovery_terminal_is_full_population_census: bool = False
    validation_rows_seen: bool = False
    schema: str = ALEXNET_FQ_GENERATION_SAMPLING_FRAME_SCHEMA
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "registration_hash",
            _identity(self.registration_hash, name="generation registration hash"),
        )
        object.__setattr__(
            self,
            "current_mask_hash",
            _identity(self.current_mask_hash, name="generation current-mask hash"),
        )
        object.__setattr__(
            self,
            "frozen_transform_hash",
            _identity(
                self.frozen_transform_hash,
                name="generation frozen-transform hash",
            ),
        )
        if (
            isinstance(self.generation_index, bool)
            or not isinstance(self.generation_index, int)
            or self.generation_index < 0
        ):
            raise ValueError("generation sampling frame index must be nonnegative")
        if type(self.fold_realization) is not V1AlexNetFQFoldRealization:
            raise TypeError("generation sampling frame requires exact role realization")
        if (
            self.fold_realization.train_role_partition_receipt.registration_hash
            != self.registration_hash
        ):
            raise ValueError("generation sampling frame registration binding differs")
        if (
            self.full_population_estimand != ALEXNET_FQ_FULL_POPULATION_ESTIMAND
            or self.state_bound_before_permutation_derivation is not True
            or isinstance(self.upstream_context_promotion_count, bool)
            or self.upstream_context_promotion_count != 0
            or self.discovery_terminal_is_full_population_census is not False
            or self.validation_rows_seen is not False
            or self.schema != ALEXNET_FQ_GENERATION_SAMPLING_FRAME_SCHEMA
        ):
            raise PermissionError(
                "generation sampling frame weakened its inference law"
            )
        self.assert_integrity()

    @property
    def target_assignment_receipt(self) -> SealedUniformPermutationReceipt:
        return self.fold_realization.target_assignment_receipt

    @property
    def complement_assignment_receipt(self) -> SealedUniformPermutationReceipt:
        return self.fold_realization.complement_assignment_receipt

    @property
    def discovery_context_ids_by_stratum(
        self,
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return self.fold_realization.discovery_context_ids_by_stratum

    @property
    def certification_context_ids_by_stratum(
        self,
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return self.fold_realization.certification_context_ids_by_stratum

    @property
    def oracle_context_ids_by_stratum(
        self,
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return self.fold_realization.oracle_context_ids_by_stratum

    @property
    def discovery_cumulative_stratum_counts(self) -> tuple[tuple[int, int], ...]:
        return self.fold_realization.discovery_cumulative_stratum_counts

    @property
    def certification_cumulative_stratum_counts(
        self,
    ) -> tuple[tuple[int, int], ...]:
        return self.fold_realization.certification_cumulative_stratum_counts

    @property
    def oracle_cumulative_stratum_counts(self) -> tuple[tuple[int, int], ...]:
        return self.fold_realization.oracle_cumulative_stratum_counts

    @property
    def train_role_partition_receipt(self) -> V1AlexNetTrainRolePartitionReceipt:
        return self.fold_realization.train_role_partition_receipt

    @property
    def full_population_stratum_counts(self) -> tuple[int, int]:
        return self.fold_realization.full_population_stratum_counts

    @property
    def discovery_sampling_capacity_stratum_counts(self) -> tuple[int, int]:
        return self.fold_realization.discovery_sampling_capacity_stratum_counts

    @property
    def dynamic_estimand_authority(self) -> bool:
        """Legacy residual cuts cannot authorize adaptive dynamic inference."""

        return False

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "registration_hash": self.registration_hash,
            "generation_index": self.generation_index,
            "current_mask_hash": self.current_mask_hash,
            "frozen_transform_hash": self.frozen_transform_hash,
            "fold_realization_hash": self.fold_realization.realization_hash,
            "target_full_permutation_receipt_hash": (
                self.target_assignment_receipt.content_hash
            ),
            "complement_full_permutation_receipt_hash": (
                self.complement_assignment_receipt.content_hash
            ),
            "train_role_partition_receipt_hash": (
                self.train_role_partition_receipt.receipt_hash
            ),
            "full_population_stratum_counts": self.full_population_stratum_counts,
            "discovery_sampling_capacity_stratum_counts": (
                self.discovery_sampling_capacity_stratum_counts
            ),
            "full_population_estimand": self.full_population_estimand,
            "state_bound_before_permutation_derivation": (
                self.state_bound_before_permutation_derivation
            ),
            "upstream_context_promotion_count": self.upstream_context_promotion_count,
            "discovery_terminal_is_full_population_census": (
                self.discovery_terminal_is_full_population_census
            ),
            "dynamic_estimand_authority": False,
            "validation_rows_seen": self.validation_rows_seen,
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    @property
    def realization_hash(self) -> str:
        """Compatibility alias for consumers that bind a realized frame."""

        return self.receipt_hash

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.fold_realization.train_role_partition_receipt.assert_realization(
            target_assignment_receipt=self.target_assignment_receipt,
            complement_assignment_receipt=self.complement_assignment_receipt,
            discovery=self.discovery_context_ids_by_stratum,
            certification=self.certification_context_ids_by_stratum,
            oracle=self.oracle_context_ids_by_stratum,
        )
        observed = content_hash(
            self._payload(),
            prefix=_GENERATION_SAMPLING_FRAME_PREFIX,
        )
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("generation sampling frame receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("generation sampling frame receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    def __getstate__(self) -> tuple[object, ...]:
        return _exact_slot_state(self)

    def __setstate__(self, state: object) -> None:
        _restore_exact_slot_state(self, state, artifact="generation sampling frame")


@dataclass(frozen=True, slots=True)
class V1AlexNetFQAcquisitionRegistration:
    campaign_contract_hash: str
    train_dataset_identity_hash: str
    train_split_identity_hash: str
    source_context_axis_hash: str
    source_label_map_hash: str
    source_role_registration_hash: str
    target_population_count: int
    complement_population_count: int
    target_population_membership_hash: str
    complement_population_membership_hash: str
    upstream_discovery_context_ids_by_stratum: tuple[tuple[str, tuple[str, ...]], ...]
    fold_assignment_entropy_root_hex: str = field(repr=False, compare=False)
    generation_order_entropy_root_hex: str = field(repr=False, compare=False)
    oracle_order_entropy_root_hex: str = field(repr=False, compare=False)
    maximum_proposal_generations: int = 1
    maximum_oracle_attempts: int = 1
    master_familywise_alpha: float = 0.05
    proposal_order_familywise_alpha: float = 0.0125
    fq_familywise_alpha: float = 0.0125
    oracle_familywise_alpha: float = 0.025
    registered_route_chunk_size: int = 262_144
    packet_draw_memory_budget_bytes: int = 1
    packet_draws_per_stratum: int = 1
    analytic_envelope_factory_identity_hash: str = ""
    expected_envelope_verifier_source_hash: str = ""
    expected_model_identity_hash: str = ""
    expected_model_state_hash: str = ""
    query_contract_hash: str = ""
    preprocessing_hash: str = ""
    normalized_input_box_hash: str = ""
    alpha_split_policy: str = ALEXNET_FQ_ALPHA_SPLIT_POLICY
    alpha_spending_policy: str = ALEXNET_FQ_ALPHA_SPENDING_POLICY
    fold_assignment_policy: str = ALEXNET_FQ_FOLD_POLICY
    look_schedule_policy: str = ALEXNET_FQ_LOOK_POLICY
    full_population_estimand: str = ALEXNET_FQ_FULL_POPULATION_ESTIMAND
    upstream_context_promotion_count: int = 0
    discovery_terminal_is_full_population_census: bool = False
    registered_before_outcomes: bool = True
    validation_rows_seen: bool = False
    schema: str = ALEXNET_FQ_ACQUISITION_REGISTRATION_SCHEMA
    _registration_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "train_dataset_identity_hash",
            "train_split_identity_hash",
            "source_context_axis_hash",
            "source_label_map_hash",
            "source_role_registration_hash",
            "target_population_membership_hash",
            "complement_population_membership_hash",
            "analytic_envelope_factory_identity_hash",
            "expected_envelope_verifier_source_hash",
            "expected_model_identity_hash",
            "expected_model_state_hash",
            "query_contract_hash",
            "preprocessing_hash",
            "normalized_input_box_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        upstream_rows: list[tuple[str, tuple[str, ...]]] = []
        for raw_name, raw_contexts in self.upstream_discovery_context_ids_by_stratum:
            stratum = _identity(raw_name, name="upstream discovery stratum")
            contexts = tuple(
                _identity(value, name="upstream discovery context")
                for value in raw_contexts
            )
            if len(contexts) != len(set(contexts)):
                raise ValueError("upstream discovery contexts must be unique")
            upstream_rows.append((stratum, contexts))
        upstream = tuple(upstream_rows)
        if tuple(name for name, _ in upstream) != ("target", "complement"):
            raise ValueError(
                "upstream discovery contexts must name target then complement"
            )
        upstream_flat = tuple(
            context for _, contexts in upstream for context in contexts
        )
        if len(upstream_flat) != len(set(upstream_flat)):
            raise ValueError("upstream discovery contexts must be globally unique")
        object.__setattr__(
            self,
            "upstream_discovery_context_ids_by_stratum",
            upstream,
        )
        for name in (
            "target_population_count",
            "complement_population_count",
            "maximum_proposal_generations",
            "maximum_oracle_attempts",
            "registered_route_chunk_size",
            "packet_draw_memory_budget_bytes",
            "packet_draws_per_stratum",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )
        contiguous_three_role_cut_counts(self.target_population_count)
        contiguous_three_role_cut_counts(self.complement_population_count)
        for name in (
            "fold_assignment_entropy_root_hex",
            "generation_order_entropy_root_hex",
            "oracle_order_entropy_root_hex",
        ):
            object.__setattr__(
                self,
                name,
                _entropy_root(getattr(self, name), name=name),
            )
        master = _alpha(self.master_familywise_alpha, name="master familywise alpha")
        proposal = _alpha(
            self.proposal_order_familywise_alpha,
            name="proposal-order familywise alpha",
        )
        fq = _alpha(self.fq_familywise_alpha, name="F/Q familywise alpha")
        oracle = _alpha(self.oracle_familywise_alpha, name="oracle familywise alpha")
        if math.fsum((proposal, fq, oracle)) != master:
            raise ValueError("registered alpha families must exactly close the master")
        if (proposal, fq, oracle) != hierarchical_master_alpha_split(master):
            raise ValueError("registered alpha families differ from the analytic split")
        for name, expected in (
            ("alpha_split_policy", ALEXNET_FQ_ALPHA_SPLIT_POLICY),
            ("alpha_spending_policy", ALEXNET_FQ_ALPHA_SPENDING_POLICY),
            ("fold_assignment_policy", ALEXNET_FQ_FOLD_POLICY),
            ("look_schedule_policy", ALEXNET_FQ_LOOK_POLICY),
            ("schema", ALEXNET_FQ_ACQUISITION_REGISTRATION_SCHEMA),
        ):
            if getattr(self, name) != expected:
                raise ValueError(f"unsupported F/Q registration coordinate: {name}")
        if (
            self.full_population_estimand != ALEXNET_FQ_FULL_POPULATION_ESTIMAND
            or isinstance(self.upstream_context_promotion_count, bool)
            or self.upstream_context_promotion_count != 0
            or self.discovery_terminal_is_full_population_census is not False
        ):
            raise PermissionError(
                "F/Q registration weakened the full-population sampling frame"
            )
        if (
            self.registered_before_outcomes is not True
            or self.validation_rows_seen is not False
        ):
            raise PermissionError("F/Q registration crossed its pre-outcome boundary")
        object.__setattr__(self, "master_familywise_alpha", master)
        object.__setattr__(self, "proposal_order_familywise_alpha", proposal)
        object.__setattr__(self, "fq_familywise_alpha", fq)
        object.__setattr__(self, "oracle_familywise_alpha", oracle)
        self.assert_integrity()

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "campaign_contract_hash": self.campaign_contract_hash,
            "train_dataset_identity_hash": self.train_dataset_identity_hash,
            "train_split_identity_hash": self.train_split_identity_hash,
            "source_context_axis_hash": self.source_context_axis_hash,
            "source_label_map_hash": self.source_label_map_hash,
            "source_role_registration_hash": self.source_role_registration_hash,
            "target_population_count": self.target_population_count,
            "complement_population_count": self.complement_population_count,
            "target_population_membership_hash": (
                self.target_population_membership_hash
            ),
            "complement_population_membership_hash": (
                self.complement_population_membership_hash
            ),
            "upstream_discovery_context_ids_by_stratum": (
                self.upstream_discovery_context_ids_by_stratum
            ),
            "fold_assignment_entropy_root_hex": (self.fold_assignment_entropy_root_hex),
            "generation_order_entropy_root_hex": (
                self.generation_order_entropy_root_hex
            ),
            "oracle_order_entropy_root_hex": self.oracle_order_entropy_root_hex,
            "maximum_proposal_generations": self.maximum_proposal_generations,
            "maximum_oracle_attempts": self.maximum_oracle_attempts,
            "master_familywise_alpha": self.master_familywise_alpha,
            "proposal_order_familywise_alpha": (self.proposal_order_familywise_alpha),
            "fq_familywise_alpha": self.fq_familywise_alpha,
            "oracle_familywise_alpha": self.oracle_familywise_alpha,
            "registered_route_chunk_size": self.registered_route_chunk_size,
            "packet_draw_memory_budget_bytes": (self.packet_draw_memory_budget_bytes),
            "packet_draws_per_stratum": self.packet_draws_per_stratum,
            "analytic_envelope_factory_identity_hash": (
                self.analytic_envelope_factory_identity_hash
            ),
            "expected_envelope_verifier_source_hash": (
                self.expected_envelope_verifier_source_hash
            ),
            "expected_model_identity_hash": self.expected_model_identity_hash,
            "expected_model_state_hash": self.expected_model_state_hash,
            "query_contract_hash": self.query_contract_hash,
            "preprocessing_hash": self.preprocessing_hash,
            "normalized_input_box_hash": self.normalized_input_box_hash,
            "alpha_split_policy": self.alpha_split_policy,
            "alpha_spending_policy": self.alpha_spending_policy,
            "fold_assignment_policy": self.fold_assignment_policy,
            "look_schedule_policy": self.look_schedule_policy,
            "full_population_estimand": self.full_population_estimand,
            "upstream_context_promotion_count": self.upstream_context_promotion_count,
            "discovery_terminal_is_full_population_census": (
                self.discovery_terminal_is_full_population_census
            ),
            "registered_before_outcomes": self.registered_before_outcomes,
            "validation_rows_seen": self.validation_rows_seen,
        }

    @property
    def registration_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_REGISTRATION_PREFIX)
        if self._registration_hash is not None and observed != self._registration_hash:
            raise ValueError("F/Q acquisition registration was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("F/Q acquisition registration hash does not match")
        object.__setattr__(self, "_registration_hash", observed)
        return observed

    def __getstate__(self) -> tuple[object, ...]:
        return _exact_slot_state(self)

    def __setstate__(self, state: object) -> None:
        _restore_exact_slot_state(self, state, artifact="F/Q acquisition registration")

    @property
    def proposal_order_generation_allocations(self) -> tuple[float, ...]:
        return finite_inverse_square_allocations(
            self.proposal_order_familywise_alpha,
            self.maximum_proposal_generations,
        )

    @property
    def fq_generation_allocations(self) -> tuple[float, ...]:
        return finite_inverse_square_allocations(
            self.fq_familywise_alpha,
            self.maximum_proposal_generations,
        )

    @property
    def oracle_attempt_allocations(self) -> tuple[float, ...]:
        return finite_inverse_square_allocations(
            self.oracle_familywise_alpha,
            self.maximum_oracle_attempts,
        )

    @property
    def oracle_order_entropy_root_commitment(self) -> str:
        """Public commitment to the direct-oracle-only randomization root."""

        return content_hash(
            {
                "schema": "partition_free_v1_alexnet_paired_bce_oracle_order_root_v1",
                "root_sha256": hashlib.sha256(
                    bytes.fromhex(self.oracle_order_entropy_root_hex)
                ).hexdigest(),
                "derivation_schema": _ORACLE_ORDER_KDF_SCHEMA,
                "maximum_oracle_attempts": self.maximum_oracle_attempts,
            },
            prefix=_ORACLE_ORDER_ROOT_COMMITMENT_PREFIX,
        )

    def open_causal_generation_frame(
        self,
        *,
        generation_index: int,
        current_mask_hash: str,
        frozen_transform_hash: str,
        target_context_ids: Sequence[str],
        complement_context_ids: Sequence[str],
    ) -> V1AlexNetFQCausalGenerationFrameReceipt:
        """Freeze current state before any dynamic role permutation is opened."""

        self.assert_integrity()
        generation = _nonnegative_int(
            generation_index,
            name="causal generation index",
        )
        if generation >= self.maximum_proposal_generations:
            raise ValueError("causal generation frame exceeds registered horizon")
        target, complement = self._validated_runtime_strata(
            target_context_ids=target_context_ids,
            complement_context_ids=complement_context_ids,
        )
        return V1AlexNetFQCausalGenerationFrameReceipt(
            registration_hash=self.registration_hash,
            generation_index=generation,
            current_mask_hash=_identity(current_mask_hash, name="current-mask hash"),
            frozen_transform_hash=_identity(
                frozen_transform_hash,
                name="frozen corrected-score transform hash",
            ),
            target_population_count=len(target),
            complement_population_count=len(complement),
            target_population_membership_hash=self.target_population_membership_hash,
            complement_population_membership_hash=(
                self.complement_population_membership_hash
            ),
        )

    def _assert_causal_generation_frame(
        self,
        frame: V1AlexNetFQCausalGenerationFrameReceipt,
    ) -> None:
        if type(frame) is not V1AlexNetFQCausalGenerationFrameReceipt:
            raise TypeError("causal role opening requires its exact generation frame")
        frame.assert_integrity()
        if (
            frame.registration_hash != self.registration_hash
            or frame.generation_index >= self.maximum_proposal_generations
            or frame.full_population_stratum_counts
            != (self.target_population_count, self.complement_population_count)
            or frame.target_population_membership_hash
            != self.target_population_membership_hash
            or frame.complement_population_membership_hash
            != self.complement_population_membership_hash
        ):
            raise ValueError("causal generation frame has stale registration coordinates")

    def _causal_role_entropy(
        self,
        *,
        role: str,
        frame: V1AlexNetFQCausalGenerationFrameReceipt,
        causal_parent_receipt_hash: str,
        stratum: str,
        sampling_population_membership_hash: str,
        phase_coordinates: Mapping[str, object],
        oracle: bool = False,
    ) -> str:
        message = json.dumps(
            {
                "schema": _CAUSAL_ROLE_KDF_SCHEMA,
                "registration_hash": self.registration_hash,
                "campaign_contract_hash": self.campaign_contract_hash,
                "train_dataset_identity_hash": self.train_dataset_identity_hash,
                "train_split_identity_hash": self.train_split_identity_hash,
                "source_context_axis_hash": self.source_context_axis_hash,
                "source_label_map_hash": self.source_label_map_hash,
                "source_role_registration_hash": self.source_role_registration_hash,
                "generation_frame_hash": frame.receipt_hash,
                "generation_index": frame.generation_index,
                "current_mask_hash": frame.current_mask_hash,
                "frozen_transform_hash": frame.frozen_transform_hash,
                "causal_parent_receipt_hash": causal_parent_receipt_hash,
                "role": role,
                "stratum": stratum,
                "sampling_population_membership_hash": (
                    sampling_population_membership_hash
                ),
                "phase_coordinates": dict(phase_coordinates),
                "causal_role_policy": ALEXNET_FQ_CAUSAL_ROLE_POLICY,
                "randomization_assumption": (
                    ALEXNET_FQ_CAUSAL_RANDOMIZATION_ASSUMPTION
                ),
                "oracle_root_domain": oracle,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        root = (
            self.oracle_order_entropy_root_hex
            if oracle
            else self.generation_order_entropy_root_hex
        )
        return hmac.new(bytes.fromhex(root), message, hashlib.sha256).hexdigest()

    def _open_full_population_role_stream(
        self,
        *,
        frame: V1AlexNetFQCausalGenerationFrameReceipt,
        role: str,
        causal_parent_receipt_hash: str,
        target_context_ids: Sequence[str],
        complement_context_ids: Sequence[str],
        discovery_prefix_freeze_hash: str | None = None,
        selected_shell_hash: str | None = None,
        certification_prefix_freeze_hash: str | None = None,
        candidate_freeze_hash: str | None = None,
        dense_referent_mask_hash: str | None = None,
        candidate_mask_hash: str | None = None,
        candidate_attempt_index: int | None = None,
    ) -> V1AlexNetFQCausalRoleSamplingStreamReceipt:
        self._assert_causal_generation_frame(frame)
        target, complement = self._validated_runtime_strata(
            target_context_ids=target_context_ids,
            complement_context_ids=complement_context_ids,
        )
        parent_hash = _identity(
            causal_parent_receipt_hash,
            name=f"{role} causal parent receipt hash",
        )
        phase_coordinates = {
            "discovery_prefix_freeze_hash": discovery_prefix_freeze_hash,
            "selected_shell_hash": selected_shell_hash,
            "certification_prefix_freeze_hash": certification_prefix_freeze_hash,
            "candidate_freeze_hash": candidate_freeze_hash,
            "dense_referent_mask_hash": dense_referent_mask_hash,
            "candidate_mask_hash": candidate_mask_hash,
            "candidate_attempt_index": candidate_attempt_index,
        }
        receipts: list[SealedUniformPermutationReceipt] = []
        for stratum, contexts, membership_hash in (
            ("target", target, self.target_population_membership_hash),
            (
                "complement",
                complement,
                self.complement_population_membership_hash,
            ),
        ):
            receipt = SealedUniformPermutationReceipt.create(
                contexts,
                entropy_hex=self._causal_role_entropy(
                    role=role,
                    frame=frame,
                    causal_parent_receipt_hash=parent_hash,
                    stratum=stratum,
                    sampling_population_membership_hash=membership_hash,
                    phase_coordinates=phase_coordinates,
                ),
            )
            receipt.verify()
            receipts.append(receipt)
        capacities = (
            (len(target) - 1, len(complement) - 1)
            if role == "calibration"
            else (len(target), len(complement))
        )
        return V1AlexNetFQCausalRoleSamplingStreamReceipt(
            registration_hash=self.registration_hash,
            generation_frame_hash=frame.receipt_hash,
            generation_index=frame.generation_index,
            role=role,
            causal_parent_receipt_hash=parent_hash,
            current_mask_hash=frame.current_mask_hash,
            frozen_transform_hash=frame.frozen_transform_hash,
            target_assignment_receipt=receipts[0],
            complement_assignment_receipt=receipts[1],
            target_population_membership_hash=self.target_population_membership_hash,
            complement_population_membership_hash=(
                self.complement_population_membership_hash
            ),
            sampling_capacity_stratum_counts=capacities,
            discovery_prefix_freeze_hash=discovery_prefix_freeze_hash,
            selected_shell_hash=selected_shell_hash,
            certification_prefix_freeze_hash=certification_prefix_freeze_hash,
            candidate_freeze_hash=candidate_freeze_hash,
            dense_referent_mask_hash=dense_referent_mask_hash,
            candidate_mask_hash=candidate_mask_hash,
            candidate_attempt_index=candidate_attempt_index,
        )

    def open_discovery_stream(
        self,
        *,
        frame: V1AlexNetFQCausalGenerationFrameReceipt,
        target_context_ids: Sequence[str],
        complement_context_ids: Sequence[str],
    ) -> V1AlexNetFQCausalRoleSamplingStreamReceipt:
        """Open D as a fresh full-U permutation after transform freeze."""

        return self._open_full_population_role_stream(
            frame=frame,
            role="discovery",
            causal_parent_receipt_hash=frame.receipt_hash,
            target_context_ids=target_context_ids,
            complement_context_ids=complement_context_ids,
        )

    def freeze_discovery_prefix(
        self,
        *,
        frame: V1AlexNetFQCausalGenerationFrameReceipt,
        stream: V1AlexNetFQCausalRoleSamplingStreamReceipt,
        cumulative_stratum_counts: tuple[int, int],
        discovery_evidence_state_hash: str,
        selected_shell_hash: str,
    ) -> V1AlexNetFQDiscoveryPrefixFreezeReceipt:
        """Close D before any C order is derived."""

        self._assert_causal_generation_frame(frame)
        if type(stream) is not V1AlexNetFQCausalRoleSamplingStreamReceipt:
            raise TypeError("D prefix freeze requires its exact role stream")
        stream.assert_integrity()
        counts = _stratum_counts(cumulative_stratum_counts, name="D freeze counts")
        if (
            stream.role != "discovery"
            or stream.registration_hash != self.registration_hash
            or stream.generation_frame_hash != frame.receipt_hash
            or stream.generation_index != frame.generation_index
            or any(
                counts[index] > stream.sampling_capacity_stratum_counts[index]
                for index in (0, 1)
            )
        ):
            raise ValueError("D prefix freeze has stale stream coordinates")
        prefixes = (
            stream.target_assignment_receipt.generated_order[: counts[0]],
            stream.complement_assignment_receipt.generated_order[: counts[1]],
        )
        return V1AlexNetFQDiscoveryPrefixFreezeReceipt(
            generation_frame_hash=frame.receipt_hash,
            discovery_stream_hash=stream.receipt_hash,
            generation_index=frame.generation_index,
            cumulative_stratum_counts=counts,
            target_prefix_membership_hash=_prefix_membership_hash(
                role="discovery",
                stratum="target",
                context_ids=prefixes[0],
            ),
            complement_prefix_membership_hash=_prefix_membership_hash(
                role="discovery",
                stratum="complement",
                context_ids=prefixes[1],
            ),
            discovery_evidence_state_hash=discovery_evidence_state_hash,
            selected_shell_hash=selected_shell_hash,
        )

    def open_certification_stream(
        self,
        *,
        frame: V1AlexNetFQCausalGenerationFrameReceipt,
        discovery_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt,
        discovery_freeze: V1AlexNetFQDiscoveryPrefixFreezeReceipt,
        target_context_ids: Sequence[str],
        complement_context_ids: Sequence[str],
    ) -> V1AlexNetFQCausalRoleSamplingStreamReceipt:
        """Open C independently only after D freezes its selected shell."""

        self._assert_causal_generation_frame(frame)
        if (
            type(discovery_stream)
            is not V1AlexNetFQCausalRoleSamplingStreamReceipt
            or type(discovery_freeze)
            is not V1AlexNetFQDiscoveryPrefixFreezeReceipt
        ):
            raise TypeError("C stream requires the exact D stream/freeze graph")
        discovery_stream.assert_integrity()
        discovery_freeze.assert_integrity()
        if (
            discovery_stream.role != "discovery"
            or discovery_stream.generation_frame_hash != frame.receipt_hash
            or discovery_freeze.generation_frame_hash != frame.receipt_hash
            or discovery_freeze.discovery_stream_hash
            != discovery_stream.receipt_hash
            or discovery_freeze.generation_index != frame.generation_index
        ):
            raise ValueError("C stream received a stale D prefix freeze")
        return self._open_full_population_role_stream(
            frame=frame,
            role="certification",
            causal_parent_receipt_hash=discovery_freeze.receipt_hash,
            target_context_ids=target_context_ids,
            complement_context_ids=complement_context_ids,
            discovery_prefix_freeze_hash=discovery_freeze.receipt_hash,
            selected_shell_hash=discovery_freeze.selected_shell_hash,
        )

    def freeze_certification_prefix(
        self,
        *,
        frame: V1AlexNetFQCausalGenerationFrameReceipt,
        discovery_freeze: V1AlexNetFQDiscoveryPrefixFreezeReceipt,
        stream: V1AlexNetFQCausalRoleSamplingStreamReceipt,
        cumulative_stratum_counts: tuple[int, int],
        certification_evidence_state_hash: str,
        candidate_freeze_hash: str,
        candidate_mask_hash: str,
    ) -> V1AlexNetFQCertificationPrefixFreezeReceipt:
        """Freeze the C-certified candidate before K can be opened."""

        self._assert_causal_generation_frame(frame)
        if (
            type(discovery_freeze)
            is not V1AlexNetFQDiscoveryPrefixFreezeReceipt
            or type(stream) is not V1AlexNetFQCausalRoleSamplingStreamReceipt
        ):
            raise TypeError("C prefix freeze requires exact D/C receipts")
        discovery_freeze.assert_integrity()
        stream.assert_integrity()
        counts = _stratum_counts(cumulative_stratum_counts, name="C freeze counts")
        if (
            stream.role != "certification"
            or stream.generation_frame_hash != frame.receipt_hash
            or stream.discovery_prefix_freeze_hash != discovery_freeze.receipt_hash
            or stream.selected_shell_hash != discovery_freeze.selected_shell_hash
            or any(
                counts[index] > stream.sampling_capacity_stratum_counts[index]
                for index in (0, 1)
            )
        ):
            raise ValueError("C prefix freeze has stale causal coordinates")
        prefixes = (
            stream.target_assignment_receipt.generated_order[: counts[0]],
            stream.complement_assignment_receipt.generated_order[: counts[1]],
        )
        return V1AlexNetFQCertificationPrefixFreezeReceipt(
            generation_frame_hash=frame.receipt_hash,
            discovery_prefix_freeze_hash=discovery_freeze.receipt_hash,
            certification_stream_hash=stream.receipt_hash,
            generation_index=frame.generation_index,
            cumulative_stratum_counts=counts,
            target_prefix_membership_hash=_prefix_membership_hash(
                role="certification",
                stratum="target",
                context_ids=prefixes[0],
            ),
            complement_prefix_membership_hash=_prefix_membership_hash(
                role="certification",
                stratum="complement",
                context_ids=prefixes[1],
            ),
            selected_shell_hash=discovery_freeze.selected_shell_hash,
            certification_evidence_state_hash=certification_evidence_state_hash,
            candidate_freeze_hash=candidate_freeze_hash,
            candidate_mask_hash=candidate_mask_hash,
        )

    def open_calibration_stream(
        self,
        *,
        frame: V1AlexNetFQCausalGenerationFrameReceipt,
        discovery_freeze: V1AlexNetFQDiscoveryPrefixFreezeReceipt,
        certification_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt,
        certification_freeze: V1AlexNetFQCertificationPrefixFreezeReceipt,
        dense_referent_mask_hash: str,
        candidate_mask_hash: str,
        candidate_attempt_index: int,
        target_context_ids: Sequence[str],
        complement_context_ids: Sequence[str],
    ) -> V1AlexNetFQCausalRoleSamplingStreamReceipt:
        """Open append-only K after the candidate, never before it."""

        self._assert_causal_generation_frame(frame)
        if (
            type(discovery_freeze)
            is not V1AlexNetFQDiscoveryPrefixFreezeReceipt
            or type(certification_stream)
            is not V1AlexNetFQCausalRoleSamplingStreamReceipt
            or type(certification_freeze)
            is not V1AlexNetFQCertificationPrefixFreezeReceipt
        ):
            raise TypeError("K stream requires the exact D/C receipt graph")
        discovery_freeze.assert_integrity()
        certification_stream.assert_integrity()
        certification_freeze.assert_integrity()
        attempt = _nonnegative_int(
            candidate_attempt_index,
            name="K candidate attempt index",
        )
        candidate = _identity(candidate_mask_hash, name="K candidate mask hash")
        dense = _identity(
            dense_referent_mask_hash,
            name="K dense referent mask hash",
        )
        if (
            attempt >= self.maximum_oracle_attempts
            or dense == candidate
            or certification_stream.role != "certification"
            or certification_stream.generation_frame_hash != frame.receipt_hash
            or certification_stream.discovery_prefix_freeze_hash
            != discovery_freeze.receipt_hash
            or certification_freeze.generation_frame_hash != frame.receipt_hash
            or certification_freeze.certification_stream_hash
            != certification_stream.receipt_hash
            or certification_freeze.discovery_prefix_freeze_hash
            != discovery_freeze.receipt_hash
            or certification_freeze.candidate_mask_hash != candidate
        ):
            raise ValueError("K stream received stale candidate coordinates")
        return self._open_full_population_role_stream(
            frame=frame,
            role="calibration",
            causal_parent_receipt_hash=certification_freeze.receipt_hash,
            target_context_ids=target_context_ids,
            complement_context_ids=complement_context_ids,
            discovery_prefix_freeze_hash=discovery_freeze.receipt_hash,
            selected_shell_hash=discovery_freeze.selected_shell_hash,
            certification_prefix_freeze_hash=certification_freeze.receipt_hash,
            candidate_freeze_hash=certification_freeze.candidate_freeze_hash,
            dense_referent_mask_hash=dense,
            candidate_mask_hash=candidate,
            candidate_attempt_index=attempt,
        )

    def freeze_calibration_prefix(
        self,
        *,
        frame: V1AlexNetFQCausalGenerationFrameReceipt,
        certification_freeze: V1AlexNetFQCertificationPrefixFreezeReceipt,
        stream: V1AlexNetFQCausalRoleSamplingStreamReceipt,
        cumulative_stratum_counts: tuple[int, int],
        dense_calibration_hash: str,
        candidate_calibration_hash: str,
        calibration_evidence_state_hash: str,
        prior_prefix_freeze: V1AlexNetFQCalibrationPrefixFreezeReceipt | None = None,
    ) -> V1AlexNetFQCalibrationPrefixFreezeReceipt:
        """Append and freeze one K prefix; shrinking or rewriting is forbidden."""

        self._assert_causal_generation_frame(frame)
        if (
            type(certification_freeze)
            is not V1AlexNetFQCertificationPrefixFreezeReceipt
            or type(stream) is not V1AlexNetFQCausalRoleSamplingStreamReceipt
        ):
            raise TypeError("K prefix freeze requires exact C/K receipts")
        certification_freeze.assert_integrity()
        stream.assert_integrity()
        counts = _stratum_counts(cumulative_stratum_counts, name="K freeze counts")
        if (
            stream.role != "calibration"
            or stream.generation_frame_hash != frame.receipt_hash
            or stream.certification_prefix_freeze_hash
            != certification_freeze.receipt_hash
            or stream.candidate_freeze_hash
            != certification_freeze.candidate_freeze_hash
            or stream.candidate_mask_hash != certification_freeze.candidate_mask_hash
            or any(
                counts[index] > stream.sampling_capacity_stratum_counts[index]
                for index in (0, 1)
            )
        ):
            raise ValueError("K prefix freeze has stale causal coordinates")
        prior_hash: str | None = None
        look_index = 1
        if prior_prefix_freeze is not None:
            if type(prior_prefix_freeze) is not V1AlexNetFQCalibrationPrefixFreezeReceipt:
                raise TypeError("prior K prefix freeze has the wrong exact type")
            prior_prefix_freeze.assert_integrity()
            prior_counts = prior_prefix_freeze.cumulative_stratum_counts
            if (
                prior_prefix_freeze.generation_frame_hash != frame.receipt_hash
                or prior_prefix_freeze.calibration_stream_hash != stream.receipt_hash
                or prior_prefix_freeze.candidate_freeze_hash
                != stream.candidate_freeze_hash
                or prior_prefix_freeze.candidate_mask_hash != stream.candidate_mask_hash
                or counts[0] < prior_counts[0]
                or counts[1] < prior_counts[1]
                or counts == prior_counts
            ):
                raise ValueError("K prefix extension is not append-only")
            prior_hash = prior_prefix_freeze.receipt_hash
            look_index = prior_prefix_freeze.look_index + 1
        prefixes = (
            stream.target_assignment_receipt.generated_order[: counts[0]],
            stream.complement_assignment_receipt.generated_order[: counts[1]],
        )
        assert stream.candidate_attempt_index is not None
        assert stream.candidate_freeze_hash is not None
        assert stream.dense_referent_mask_hash is not None
        assert stream.candidate_mask_hash is not None
        return V1AlexNetFQCalibrationPrefixFreezeReceipt(
            generation_frame_hash=frame.receipt_hash,
            certification_prefix_freeze_hash=certification_freeze.receipt_hash,
            calibration_stream_hash=stream.receipt_hash,
            generation_index=frame.generation_index,
            candidate_attempt_index=stream.candidate_attempt_index,
            look_index=look_index,
            prior_prefix_freeze_hash=prior_hash,
            cumulative_stratum_counts=counts,
            target_prefix_membership_hash=_prefix_membership_hash(
                role="calibration",
                stratum="target",
                context_ids=prefixes[0],
            ),
            complement_prefix_membership_hash=_prefix_membership_hash(
                role="calibration",
                stratum="complement",
                context_ids=prefixes[1],
            ),
            candidate_freeze_hash=stream.candidate_freeze_hash,
            dense_referent_mask_hash=stream.dense_referent_mask_hash,
            candidate_mask_hash=stream.candidate_mask_hash,
            dense_calibration_hash=dense_calibration_hash,
            candidate_calibration_hash=candidate_calibration_hash,
            calibration_evidence_state_hash=calibration_evidence_state_hash,
        )

    def open_oracle_stream(
        self,
        *,
        frame: V1AlexNetFQCausalGenerationFrameReceipt,
        certification_freeze: V1AlexNetFQCertificationPrefixFreezeReceipt,
        calibration_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt,
        calibration_prefix_freeze: V1AlexNetFQCalibrationPrefixFreezeReceipt,
        candidate_attempt_index: int,
        candidate_mask_hash: str,
        dense_calibration_hash: str,
        candidate_calibration_hash: str,
        target_context_ids: Sequence[str],
        complement_context_ids: Sequence[str],
    ) -> V1AlexNetFQOracleSamplingFrameReceipt:
        """Open O over ``U \\ K`` after candidate and affine-state freeze."""

        self._assert_causal_generation_frame(frame)
        if (
            type(certification_freeze)
            is not V1AlexNetFQCertificationPrefixFreezeReceipt
            or type(calibration_stream)
            is not V1AlexNetFQCausalRoleSamplingStreamReceipt
            or type(calibration_prefix_freeze)
            is not V1AlexNetFQCalibrationPrefixFreezeReceipt
        ):
            raise TypeError("O stream requires exact C/K freeze lineage")
        certification_freeze.assert_integrity()
        calibration_stream.assert_integrity()
        calibration_prefix_freeze.assert_integrity()
        attempt = _nonnegative_int(
            candidate_attempt_index,
            name="O candidate attempt index",
        )
        candidate = _identity(candidate_mask_hash, name="O candidate mask hash")
        dense_calibration = _identity(
            dense_calibration_hash,
            name="O dense calibration hash",
        )
        candidate_calibration = _identity(
            candidate_calibration_hash,
            name="O candidate calibration hash",
        )
        if (
            attempt >= self.maximum_oracle_attempts
            or calibration_stream.role != "calibration"
            or calibration_stream.generation_frame_hash != frame.receipt_hash
            or calibration_stream.certification_prefix_freeze_hash
            != certification_freeze.receipt_hash
            or calibration_prefix_freeze.calibration_stream_hash
            != calibration_stream.receipt_hash
            or calibration_prefix_freeze.certification_prefix_freeze_hash
            != certification_freeze.receipt_hash
            or calibration_prefix_freeze.candidate_attempt_index != attempt
            or calibration_prefix_freeze.candidate_mask_hash != candidate
            or calibration_prefix_freeze.dense_calibration_hash != dense_calibration
            or calibration_prefix_freeze.candidate_calibration_hash
            != candidate_calibration
        ):
            raise ValueError("O stream received stale candidate/calibration coordinates")
        target, complement = self._validated_runtime_strata(
            target_context_ids=target_context_ids,
            complement_context_ids=complement_context_ids,
        )
        counts = calibration_prefix_freeze.cumulative_stratum_counts
        k_prefixes = (
            calibration_stream.target_assignment_receipt.generated_order[: counts[0]],
            calibration_stream.complement_assignment_receipt.generated_order[
                : counts[1]
            ],
        )
        oracle_populations = (
            out_of_fit_population(target, k_prefixes[0]),
            out_of_fit_population(complement, k_prefixes[1]),
        )
        phase_coordinates = {
            "certification_prefix_freeze_hash": certification_freeze.receipt_hash,
            "calibration_stream_hash": calibration_stream.receipt_hash,
            "calibration_prefix_freeze_hash": calibration_prefix_freeze.receipt_hash,
            "candidate_attempt_index": attempt,
            "candidate_freeze_hash": calibration_prefix_freeze.candidate_freeze_hash,
            "dense_referent_mask_hash": (
                calibration_prefix_freeze.dense_referent_mask_hash
            ),
            "candidate_mask_hash": candidate,
            "dense_calibration_hash": dense_calibration,
            "candidate_calibration_hash": candidate_calibration,
            "target_k_prefix_membership_hash": (
                calibration_prefix_freeze.target_prefix_membership_hash
            ),
            "complement_k_prefix_membership_hash": (
                calibration_prefix_freeze.complement_prefix_membership_hash
            ),
        }
        receipts: list[SealedUniformPermutationReceipt] = []
        for stratum, population in zip(
            ("target", "complement"),
            oracle_populations,
            strict=True,
        ):
            membership_hash = content_hash(
                population,
                prefix=f"partition-free-v1-alexnet-{stratum}-out-of-fit-population-",
            )
            receipt = SealedUniformPermutationReceipt.create(
                population,
                entropy_hex=self._causal_role_entropy(
                    role="oracle",
                    frame=frame,
                    causal_parent_receipt_hash=calibration_prefix_freeze.receipt_hash,
                    stratum=stratum,
                    sampling_population_membership_hash=membership_hash,
                    phase_coordinates=phase_coordinates,
                    oracle=True,
                ),
            )
            receipt.verify()
            receipts.append(receipt)
        return V1AlexNetFQOracleSamplingFrameReceipt(
            registration_hash=self.registration_hash,
            generation_frame_hash=frame.receipt_hash,
            certification_prefix_freeze_hash=certification_freeze.receipt_hash,
            calibration_stream_hash=calibration_stream.receipt_hash,
            calibration_prefix_freeze_hash=calibration_prefix_freeze.receipt_hash,
            generation_index=frame.generation_index,
            candidate_attempt_index=attempt,
            candidate_freeze_hash=calibration_prefix_freeze.candidate_freeze_hash,
            dense_referent_mask_hash=(
                calibration_prefix_freeze.dense_referent_mask_hash
            ),
            candidate_mask_hash=candidate,
            dense_calibration_hash=dense_calibration,
            candidate_calibration_hash=candidate_calibration,
            target_full_population_membership_hash=(
                alexnet_fq_full_population_membership_hash(
                    stratum="target",
                    context_ids=target,
                )
            ),
            complement_full_population_membership_hash=(
                alexnet_fq_full_population_membership_hash(
                    stratum="complement",
                    context_ids=complement,
                )
            ),
            calibration_prefix_context_ids_by_stratum=(
                ("target", k_prefixes[0]),
                ("complement", k_prefixes[1]),
            ),
            target_assignment_receipt=receipts[0],
            complement_assignment_receipt=receipts[1],
            full_population_stratum_counts=(len(target), len(complement)),
            calibration_prefix_stratum_counts=counts,
            oracle_population_stratum_counts=(
                len(oracle_populations[0]),
                len(oracle_populations[1]),
            ),
        )

    def oracle_attempt_order_receipts(
        self,
        *,
        candidate_attempt_index: int,
        folds: V1AlexNetFQFoldRealization | V1AlexNetFQGenerationSamplingFrameReceipt,
    ) -> tuple[SealedUniformPermutationReceipt, SealedUniformPermutationReceipt]:
        """Derive independent target/complement orders for one oracle attempt."""

        self.assert_integrity()
        if type(folds) not in (
            V1AlexNetFQFoldRealization,
            V1AlexNetFQGenerationSamplingFrameReceipt,
        ):
            raise TypeError("oracle attempt orders require exact F/Q folds")
        if (
            isinstance(candidate_attempt_index, bool)
            or not isinstance(candidate_attempt_index, int)
            or not 0 <= candidate_attempt_index < self.maximum_oracle_attempts
        ):
            raise ValueError("oracle attempt order exceeds the registered horizon")
        if tuple(name for name, _ in folds.oracle_context_ids_by_stratum) != (
            "target",
            "complement",
        ):
            raise ValueError("oracle folds must be target then complement")
        rows: list[SealedUniformPermutationReceipt] = []
        for stratum, contexts in folds.oracle_context_ids_by_stratum:
            if not contexts:
                raise ValueError("oracle fold cannot be empty")
            membership_hash = content_hash(
                (stratum, tuple(sorted(contexts))),
                prefix="partition-free-v1-alexnet-paired-bce-oracle-membership-",
            )
            message = json.dumps(
                {
                    "schema": _ORACLE_ORDER_KDF_SCHEMA,
                    "campaign_contract_hash": self.campaign_contract_hash,
                    "train_dataset_identity_hash": self.train_dataset_identity_hash,
                    "train_split_identity_hash": self.train_split_identity_hash,
                    "source_role_registration_hash": self.source_role_registration_hash,
                    "fold_realization_hash": folds.realization_hash,
                    "oracle_order_entropy_root_commitment": (
                        self.oracle_order_entropy_root_commitment
                    ),
                    "maximum_oracle_attempts": self.maximum_oracle_attempts,
                    "candidate_attempt_index": candidate_attempt_index,
                    "role": "paired_bce_irreversible_acceptance",
                    "stratum": stratum,
                    "fixed_membership_hash": membership_hash,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            entropy = hmac.new(
                bytes.fromhex(self.oracle_order_entropy_root_hex),
                message,
                hashlib.sha256,
            ).hexdigest()
            receipt = SealedUniformPermutationReceipt.create(
                contexts,
                entropy_hex=entropy,
            )
            receipt.verify()
            rows.append(receipt)
        return rows[0], rows[1]

    def _fold_entropy(self, *, stratum: str, membership_hash: str) -> str:
        message = json.dumps(
            {
                "schema": _FOLD_ASSIGNMENT_KDF_SCHEMA,
                "campaign_contract_hash": self.campaign_contract_hash,
                "train_dataset_identity_hash": self.train_dataset_identity_hash,
                "train_split_identity_hash": self.train_split_identity_hash,
                "source_context_axis_hash": self.source_context_axis_hash,
                "source_label_map_hash": self.source_label_map_hash,
                "source_role_registration_hash": self.source_role_registration_hash,
                "stratum": stratum,
                "membership_hash": membership_hash,
                "fold_assignment_policy": self.fold_assignment_policy,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hmac.new(
            bytes.fromhex(self.fold_assignment_entropy_root_hex),
            message,
            hashlib.sha256,
        ).hexdigest()

    def _validated_runtime_strata(
        self,
        *,
        target_context_ids: Sequence[str],
        complement_context_ids: Sequence[str],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        target = tuple(
            _identity(value, name="runtime target context")
            for value in target_context_ids
        )
        complement = tuple(
            _identity(value, name="runtime complement context")
            for value in complement_context_ids
        )
        full_train_contexts = target + complement
        if (
            len(target) != self.target_population_count
            or len(complement) != self.complement_population_count
            or len(target) != len(set(target))
            or len(complement) != len(set(complement))
            or len(full_train_contexts) != len(set(full_train_contexts))
            or content_hash(
                target,
                prefix="partition-free-v1-alexnet-fq-target-membership-",
            )
            != self.target_population_membership_hash
            or content_hash(
                complement,
                prefix="partition-free-v1-alexnet-fq-complement-membership-",
            )
            != self.complement_population_membership_hash
        ):
            raise ValueError("runtime train strata differ from the F/Q registration")
        upstream = dict(self.upstream_discovery_context_ids_by_stratum)
        if not set(upstream["target"]).issubset(target) or not set(
            upstream["complement"]
        ).issubset(complement):
            raise ValueError("upstream contexts leave the registered population")
        return target, complement

    def _realize_contiguous_role_slices(
        self,
        *,
        target: tuple[str, ...],
        complement: tuple[str, ...],
        target_receipt: SealedUniformPermutationReceipt,
        complement_receipt: SealedUniformPermutationReceipt,
    ) -> V1AlexNetFQFoldRealization:
        if target_receipt.canonical_population != tuple(
            sorted(target)
        ) or complement_receipt.canonical_population != tuple(sorted(complement)):
            raise ValueError(
                "full permutations differ from the registered train strata"
            )
        target_receipt.verify()
        complement_receipt.verify()

        # Upstream identities are population-integrity anchors only.  They are
        # deliberately absent from this slicing map: promoting them into D
        # would change inclusion probabilities for the full-population score.
        target_discovery, target_certification, target_oracle = (
            contiguous_three_role_slices(target_receipt.generated_order)
        )
        (
            complement_discovery,
            complement_certification,
            complement_oracle,
        ) = contiguous_three_role_slices(complement_receipt.generated_order)
        discovery = (
            ("target", target_discovery),
            ("complement", complement_discovery),
        )
        certification = (
            ("target", target_certification),
            ("complement", complement_certification),
        )
        oracle = (
            ("target", target_oracle),
            ("complement", complement_oracle),
        )
        role_receipt = V1AlexNetTrainRolePartitionReceipt.create(
            registration=self,
            target_context_ids=target,
            complement_context_ids=complement,
            target_assignment_receipt=target_receipt,
            complement_assignment_receipt=complement_receipt,
            discovery=discovery,
            certification=certification,
            oracle=oracle,
        )
        return V1AlexNetFQFoldRealization(
            target_assignment_receipt=target_receipt,
            complement_assignment_receipt=complement_receipt,
            discovery_context_ids_by_stratum=discovery,
            certification_context_ids_by_stratum=certification,
            oracle_context_ids_by_stratum=oracle,
            discovery_cumulative_stratum_counts=balanced_dyadic_capacity_looks(
                len(discovery[0][1]),
                len(discovery[1][1]),
            ),
            certification_cumulative_stratum_counts=balanced_dyadic_capacity_looks(
                len(certification[0][1]),
                len(certification[1][1]),
            ),
            oracle_cumulative_stratum_counts=balanced_dyadic_capacity_looks(
                len(oracle[0][1]),
                len(oracle[1][1]),
            ),
            train_role_partition_receipt=role_receipt,
        )

    def reify_folds(
        self,
        *,
        target_context_ids: Sequence[str],
        complement_context_ids: Sequence[str],
    ) -> V1AlexNetFQFoldRealization:
        """Materialize a static population/cut template from the compact seal.

        The returned D slice is unbiased for the full registered population,
        but this static object is not the sampling-frame authority for a
        dynamic generation.  Dynamic callers must use
        :meth:`reify_generation_sampling_frame` after freezing state.
        """

        self.assert_integrity()
        target, complement = self._validated_runtime_strata(
            target_context_ids=target_context_ids,
            complement_context_ids=complement_context_ids,
        )
        target_receipt = SealedUniformPermutationReceipt.create(
            target,
            entropy_hex=self._fold_entropy(
                stratum="target",
                membership_hash=self.target_population_membership_hash,
            ),
        )
        complement_receipt = SealedUniformPermutationReceipt.create(
            complement,
            entropy_hex=self._fold_entropy(
                stratum="complement",
                membership_hash=self.complement_population_membership_hash,
            ),
        )
        return self._realize_contiguous_role_slices(
            target=target,
            complement=complement,
            target_receipt=target_receipt,
            complement_receipt=complement_receipt,
        )

    def _generation_sampling_frame_entropy(
        self,
        *,
        generation_index: int,
        current_mask_hash: str,
        frozen_transform_hash: str,
        stratum: str,
        membership_hash: str,
    ) -> str:
        message = json.dumps(
            {
                "schema": _GENERATION_SAMPLING_FRAME_KDF_SCHEMA,
                "registration_hash": self.registration_hash,
                "campaign_contract_hash": self.campaign_contract_hash,
                "train_dataset_identity_hash": self.train_dataset_identity_hash,
                "train_split_identity_hash": self.train_split_identity_hash,
                "source_context_axis_hash": self.source_context_axis_hash,
                "source_label_map_hash": self.source_label_map_hash,
                "source_role_registration_hash": self.source_role_registration_hash,
                "generation_index": generation_index,
                "current_mask_hash": current_mask_hash,
                "frozen_transform_hash": frozen_transform_hash,
                "stratum": stratum,
                "full_population_membership_hash": membership_hash,
                "fold_assignment_policy": self.fold_assignment_policy,
                "full_population_estimand": ALEXNET_FQ_FULL_POPULATION_ESTIMAND,
                "upstream_context_promotion_count": 0,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hmac.new(
            bytes.fromhex(self.generation_order_entropy_root_hex),
            message,
            hashlib.sha256,
        ).hexdigest()

    def reify_generation_sampling_frame(
        self,
        *,
        generation_index: int,
        current_mask_hash: str,
        frozen_transform_hash: str,
        target_context_ids: Sequence[str],
        complement_context_ids: Sequence[str],
    ) -> V1AlexNetFQGenerationSamplingFrameReceipt:
        """Replay the legacy same-permutation residual-cut frame.

        This method remains solely for compatibility evidence.  It carries no
        dynamic estimand authority; production callers must use
        :meth:`open_causal_generation_frame` and the causal role-opening API.
        """

        self.assert_integrity()
        if (
            isinstance(generation_index, bool)
            or not isinstance(generation_index, int)
            or not 0 <= generation_index < self.maximum_proposal_generations
        ):
            raise ValueError("generation sampling frame exceeds registered horizon")
        mask_hash = _identity(current_mask_hash, name="current-mask hash")
        transform_hash = _identity(
            frozen_transform_hash,
            name="frozen corrected-score transform hash",
        )
        target, complement = self._validated_runtime_strata(
            target_context_ids=target_context_ids,
            complement_context_ids=complement_context_ids,
        )
        target_receipt = SealedUniformPermutationReceipt.create(
            target,
            entropy_hex=self._generation_sampling_frame_entropy(
                generation_index=generation_index,
                current_mask_hash=mask_hash,
                frozen_transform_hash=transform_hash,
                stratum="target",
                membership_hash=self.target_population_membership_hash,
            ),
        )
        complement_receipt = SealedUniformPermutationReceipt.create(
            complement,
            entropy_hex=self._generation_sampling_frame_entropy(
                generation_index=generation_index,
                current_mask_hash=mask_hash,
                frozen_transform_hash=transform_hash,
                stratum="complement",
                membership_hash=self.complement_population_membership_hash,
            ),
        )
        realization = self._realize_contiguous_role_slices(
            target=target,
            complement=complement,
            target_receipt=target_receipt,
            complement_receipt=complement_receipt,
        )
        return V1AlexNetFQGenerationSamplingFrameReceipt(
            registration_hash=self.registration_hash,
            generation_index=generation_index,
            current_mask_hash=mask_hash,
            frozen_transform_hash=transform_hash,
            fold_realization=realization,
        )

    def to_json_payload(self) -> dict[str, Any]:
        return {**self._payload(), "registration_hash": self.registration_hash}

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "V1AlexNetFQAcquisitionRegistration":
        values = dict(payload)
        declared = values.pop("registration_hash", None)
        values["upstream_discovery_context_ids_by_stratum"] = tuple(
            (str(name), tuple(str(value) for value in contexts))
            for name, contexts in values.get(
                "upstream_discovery_context_ids_by_stratum",
                (),
            )
        )
        result = cls(**values)
        if declared is None:
            raise ValueError("F/Q acquisition registration lacks its content hash")
        result.assert_integrity(str(declared))
        return result

    @classmethod
    def load(cls, path: str | Path) -> "V1AlexNetFQAcquisitionRegistration":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("F/Q acquisition registration JSON must be an object")
        return cls.from_mapping(payload)


__all__ = [
    "ALEXNET_FQ_ACQUISITION_REGISTRATION_SCHEMA",
    "ALEXNET_FQ_ACQUISITION_REGISTRATION_SCHEMA_V1",
    "ALEXNET_FQ_ALPHA_SPENDING_POLICY",
    "ALEXNET_FQ_ALPHA_SPLIT_POLICY",
    "ALEXNET_FQ_CALIBRATION_PREFIX_FREEZE_SCHEMA",
    "ALEXNET_FQ_CAUSAL_GENERATION_FRAME_SCHEMA",
    "ALEXNET_FQ_CAUSAL_RANDOMIZATION_ASSUMPTION",
    "ALEXNET_FQ_CAUSAL_ROLE_POLICY",
    "ALEXNET_FQ_CAUSAL_ROLE_STREAM_SCHEMA",
    "ALEXNET_FQ_CERTIFICATION_PREFIX_FREEZE_SCHEMA",
    "ALEXNET_FQ_DISCOVERY_PREFIX_FREEZE_SCHEMA",
    "ALEXNET_FQ_FOLD_POLICY",
    "ALEXNET_FQ_FOLD_POLICY_V1",
    "ALEXNET_FQ_FOLD_REALIZATION_SCHEMA",
    "ALEXNET_FQ_FOLD_REALIZATION_SCHEMA_V1",
    "ALEXNET_FQ_FOLD_REALIZATION_SCHEMA_V2",
    "ALEXNET_FQ_FULL_POPULATION_ESTIMAND",
    "ALEXNET_FQ_FULL_FRAME_PREFIX_SEMANTICS",
    "ALEXNET_FQ_GENERATION_SAMPLING_FRAME_SCHEMA",
    "ALEXNET_FQ_LOOK_POLICY",
    "ALEXNET_FQ_LOOK_POLICY_V1",
    "ALEXNET_FQ_PREFIX_SEMANTICS",
    "ALEXNET_FQ_PREFIX_SEMANTICS_V1",
    "ALEXNET_FQ_ORACLE_SAMPLING_FRAME_SCHEMA",
    "ALEXNET_FQ_OUT_OF_FIT_ORACLE_ESTIMAND",
    "ALEXNET_FQ_TRAIN_ROLE_PARTITION_RECEIPT_SCHEMA",
    "ALEXNET_FQ_TRAIN_ROLE_PARTITION_RECEIPT_SCHEMA_V1",
    "V1AlexNetFQAcquisitionRegistration",
    "V1AlexNetFQCalibrationPrefixFreezeReceipt",
    "V1AlexNetFQCausalGenerationFrameReceipt",
    "V1AlexNetFQCausalRoleSamplingStreamReceipt",
    "V1AlexNetFQCertificationPrefixFreezeReceipt",
    "V1AlexNetFQDiscoveryPrefixFreezeReceipt",
    "V1AlexNetFQFoldRealization",
    "V1AlexNetFQGenerationSamplingFrameReceipt",
    "V1AlexNetFQOracleSamplingFrameReceipt",
    "V1AlexNetTrainRolePartitionReceipt",
    "alexnet_fq_full_population_membership_hash",
    "balanced_dyadic_capacity_looks",
    "balanced_dyadic_census_looks",
    "contiguous_three_role_cut_counts",
    "contiguous_three_role_slices",
    "finite_inverse_square_allocations",
    "hierarchical_master_alpha_split",
    "out_of_fit_population",
]
