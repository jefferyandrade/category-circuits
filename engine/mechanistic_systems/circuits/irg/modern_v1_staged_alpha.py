"""Per-step staged Student-t decision engine for modern partition-free v1.

The staged-alpha family was selected and cross-query shrinkage
deferred.  This module implements the selected statistical
object but deliberately stops at a certificate *candidate*: the generic J0
campaign schema is implemented, while a production instance and an
external authorization boundary remain absent.

For each candidate attempt and registered look ``j`` the error stake is

    alpha[j] = alpha_attempt / L,

where ``L`` is the fixed maximum number of looks within that attempt.  The
production campaign derives normalized stakes proportional to
``1 / (attempt + 1)^2`` before any outcomes, over the immutable maximum
candidate-attempt count. Thus the within-attempt look union and the
across-attempt union spend at most ``alpha_campaign`` even under adaptive
proposal selection. Conditional on
the attempt's pre-draw freeze, its sealed categorical law makes the
importance-weighted, genuinely paired loss-referent-to-candidate damage rows
IID under the registered sampling law. In the production v1 AlexNet runtime the
loss referent is the dense initial mask; the current mask is instead the
proposal/relinearization and commit parent recorded in certificate metadata.
The same Student-t interval family used by modern v0 is applied to those rows.
An upper endpoint below the total dense-baseline budget is an accept candidate;
a lower endpoint above it is a reject candidate;
otherwise the step remains unresolved.

Every attempt directly compares its proposed mask with the dense baseline, so
accepted endpoints are never summed.  The finite-horizon union controls false
acceptance of any proposed mask in the campaign; it does not turn Student-t
model evidence into a distribution-free bound. Exact finite-sample coverage
still requires the registered Student-t observation model; outside it the
interval has the same model/CLT evidence ceiling as the v0 mechanism it
matches operationally.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import secrets
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from itertools import accumulate
from numbers import Integral
from types import MappingProxyType
from typing import Any, Final, Literal

import numpy as np

from mechanistic_systems.evidence import student_t_interval

from mechanistic_systems.measurement.request_providers import (
    PairedMaskOracleProvider,
    validate_outcome_for_request,
)

from .contracts import MeasurementOutcome, MeasurementRequest, content_hash
from .modern_v1_core import (
    MODERN_V1_WORKFLOW_IDENTITY,
    V1CertificateFreeze,
    V1EvidencePacket,
    V1OwnerRegistration,
    assert_packet_matches_certificate_freeze,
)

STAGED_ALPHA_COORDINATE_SCHEMA: Final = "partition_free_v1_staged_alpha_coordinates_v4"
STAGED_ALPHA_STEP_SCHEMA: Final = "partition_free_v1_staged_alpha_step_freeze_v3"
STAGED_ALPHA_PACKET_SCHEMA: Final = "partition_free_v1_staged_alpha_packet_envelope_v1"
STAGED_ALPHA_LEDGER_SCHEMA: Final = "partition_free_v1_staged_alpha_ledger_v1"
STAGED_ALPHA_DECISION_SCHEMA: Final = "partition_free_v1_staged_alpha_decision_v2"
STAGED_ALPHA_TRAJECTORY_SCHEMA: Final = (
    "partition_free_v1_staged_alpha_trajectory_ledger_v1"
)
STAGED_ALPHA_SAMPLING_PLAN_SCHEMA: Final = (
    "partition_free_v1_staged_alpha_sampling_plan_v2"
)
STAGED_ALPHA_ENTROPY_TAPE_SCHEMA: Final = (
    "partition_free_v1_staged_alpha_counter_entropy_stream_v2"
)
STAGED_ALPHA_LEGACY_ENTROPY_TAPE_SCHEMA: Final = (
    "partition_free_v1_staged_alpha_system_entropy_tape_v1"
)
STAGED_ALPHA_PROOF_ID: Final = (
    "finite_horizon_quadratic_cross_attempt_student_t_look_union_v1"
)
STAGED_ALPHA_STEP_SPENDING_ID: Final = (
    "finite_horizon_normalized_inverse_square_cross_attempt_spending_v1"
)
STAGED_ALPHA_LOOK_SPENDING_ID: Final = "equal_within_step_registered_looks_v1"
STAGED_ALPHA_BOUND_ID: Final = "two_sided_student_t_increment_interval_v1"
STAGED_ALPHA_SAMPLING_ID: Final = (
    "receipted_u64_exact_categorical_with_replacement_importance_weighted_v1"
)
STAGED_ALPHA_SAMPLING_MASS_TOTAL: Final = 1 << 64
_SAMPLING_MASS_TOTAL: Final = STAGED_ALPHA_SAMPLING_MASS_TOTAL
STAGED_ALPHA_SYSTEM_ENTROPY_SOURCE_ID: Final = (
    "python_secrets_hmac_sha256_counter_stream_v2"
)
STAGED_ALPHA_LEGACY_SYSTEM_ENTROPY_SOURCE_ID: Final = (
    "python_secrets_system_entropy_v1"
)
STAGED_ALPHA_EXTERNAL_ENTROPY_SOURCE_ID: Final = "external_entropy_replay_only_v1"
STAGED_ALPHA_EXHAUSTION_ACTION_ID: Final = "refuse_unresolved_no_commit_v1"
_PACKET_TOKEN = object()
_TRANSFORMED_PACKET_TOKEN = object()
_SYSTEM_SAMPLING_TOKEN = object()
_EXTERNAL_SAMPLING_TOKEN = object()


def _nonempty(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _finite_positive(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _freeze(value: Any, label: str) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise TypeError(f"{label} keys must be nonempty strings")
            frozen[key] = _freeze(item, f"{label}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item, f"{label}[{index}]") for index, item in enumerate(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a nonfinite float")
        return float(value)
    if isinstance(value, (str, bytes, int, bool)) or value is None:
        return value
    raise TypeError(f"{label} contains an unsupported value")


def _probabilities(
    value: object,
    label: str,
    count: int,
    *,
    allow_zero: bool = False,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    invalid_support = np.any(array < 0.0) if allow_zero else np.any(array <= 0.0)
    if (
        array.shape != (count,)
        or not np.isfinite(array).all()
        or bool(invalid_support)
    ):
        support = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{label} must be finite, {support}, and row-aligned")
    return array


def _entropy_hex(value: object, *, draw_count: int) -> str:
    entropy = _nonempty(value, "sampling entropy").lower()
    if len(entropy) != 16 * draw_count:
        raise ValueError("sampling entropy must encode exactly 64 bits per draw")
    try:
        bytes.fromhex(entropy)
    except ValueError as exc:
        raise ValueError("sampling entropy must be lowercase hexadecimal") from exc
    if entropy != str(value):
        raise ValueError("sampling entropy must be canonical lowercase hexadecimal")
    return entropy


def _entropy_root_hex(value: object) -> str:
    root = _nonempty(value, "counter entropy root")
    if len(root) != 64:
        raise ValueError("counter entropy root must encode exactly 256 bits")
    try:
        decoded = bytes.fromhex(root)
    except ValueError as exc:
        raise ValueError(
            "counter entropy root must be lowercase hexadecimal"
        ) from exc
    if root != root.lower() or decoded.hex() != root:
        raise ValueError(
            "counter entropy root must be canonical lowercase hexadecimal"
        )
    return root


def _positive_sampling_masses(probabilities: np.ndarray) -> tuple[int, ...]:
    """Quantize a declared law to an exact positive 64-bit categorical law."""

    with localcontext() as context:
        context.prec = 80
        decimal_probabilities = [Decimal.from_float(float(value)) for value in probabilities]
        total = sum(decimal_probabilities, Decimal(0))
        scaled = [value * _SAMPLING_MASS_TOTAL / total for value in decimal_probabilities]
    floors = [int(value) for value in scaled]
    remainder = _SAMPLING_MASS_TOTAL - sum(floors)
    order = sorted(
        range(len(floors)),
        key=lambda index: (-(scaled[index] - Decimal(floors[index])), index),
    )
    for index in order[:remainder]:
        floors[index] += 1
    if any(value <= 0 for value in floors) or sum(floors) != _SAMPLING_MASS_TOTAL:
        raise ValueError(
            "sampling probabilities are below the positive 64-bit mass resolution"
        )
    return tuple(floors)


def canonicalize_v1_staged_alpha_sampling_probabilities(
    sampling_population_probabilities: Sequence[float],
) -> tuple[float, ...]:
    """Return the exact float law induced by the registered uint64 masses.

    The staged-alpha sampler realizes a declared categorical law through a
    positive uint64 mass partition. Candidate-specific predictable allocation
    policies must bind the resulting exact law—not a nearby pre-quantization
    vector—so the request, sampling plan, importance weights, and replay
    receipt all name one object.
    """

    masses = quantize_v1_staged_alpha_sampling_masses(
        sampling_population_probabilities
    )
    return tuple(float(value / _SAMPLING_MASS_TOTAL) for value in masses)


def quantize_v1_staged_alpha_sampling_masses(
    sampling_population_probabilities: Sequence[float],
) -> tuple[int, ...]:
    """Return the exact positive uint64 mass partition for a declared law."""

    probabilities = np.asarray(
        tuple(float(value) for value in sampling_population_probabilities),
        dtype=np.float64,
    ).reshape(-1)
    if (
        not probabilities.size
        or not np.isfinite(probabilities).all()
        or bool(np.any(probabilities <= 0.0))
        or not math.isclose(
            float(np.sum(probabilities, dtype=np.float64)),
            1.0,
            rel_tol=0.0,
            abs_tol=64.0 * np.finfo(np.float64).eps,
        )
    ):
        raise ValueError(
            "sampling population probabilities must be finite, positive, and sum to one"
        )
    return _positive_sampling_masses(probabilities)


def _sampling_source_indices(
    *,
    entropy_hex: str,
    draw_ids: Sequence[str],
    masses: Sequence[int],
) -> tuple[int, ...]:
    cumulative = tuple(accumulate(masses))
    result: list[int] = []
    for index, _draw_id in enumerate(draw_ids):
        uniform_u64 = int(entropy_hex[16 * index : 16 * (index + 1)], 16)
        source_index = bisect_right(cumulative, uniform_u64)
        if source_index == len(cumulative):
            # Preserve the former linear-search refusal for an invalid mass
            # partition whose cumulative support does not cover this draw.
            raise StopIteration
        result.append(source_index)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class V1StagedAlphaEntropyTape:
    """Preissued counter-entropy stream for exact interruption/resume replay.

    New instances store one 256-bit OS-random root and derive addressed
    64-bit words through domain-separated HMAC-SHA256 counter blocks.  The
    root is born before candidate outcomes and remains in the trusted runtime
    checkpoint, so future words are fixed pre-outcome and replay exactly while
    storage stays constant in the registered horizon.  The legacy bounded
    tape identity remains readable solely for historical checkpoint replay.
    """

    campaign_contract_hash: str
    controller_run_id: str
    executor_identity_hash: str
    maximum_candidate_count: int
    draws_per_candidate: int
    entropy_hex: str
    entropy_source_id: str = STAGED_ALPHA_SYSTEM_ENTROPY_SOURCE_ID
    schema: str = STAGED_ALPHA_ENTROPY_TAPE_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "controller_run_id",
            "executor_identity_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "maximum_candidate_count",
            _positive_int(self.maximum_candidate_count, "maximum_candidate_count"),
        )
        object.__setattr__(
            self,
            "draws_per_candidate",
            _positive_int(self.draws_per_candidate, "draws_per_candidate"),
        )
        identity = (self.entropy_source_id, self.schema)
        if identity == (
            STAGED_ALPHA_SYSTEM_ENTROPY_SOURCE_ID,
            STAGED_ALPHA_ENTROPY_TAPE_SCHEMA,
        ):
            entropy = _entropy_root_hex(self.entropy_hex)
        elif identity == (
            STAGED_ALPHA_LEGACY_SYSTEM_ENTROPY_SOURCE_ID,
            STAGED_ALPHA_LEGACY_ENTROPY_TAPE_SCHEMA,
        ):
            expected_words = (
                self.maximum_candidate_count * self.draws_per_candidate
            )
            entropy = _entropy_hex(
                self.entropy_hex,
                draw_count=expected_words,
            )
        else:
            raise ValueError(
                "unsupported staged-alpha system-entropy stream identity"
            )
        object.__setattr__(self, "entropy_hex", entropy)

    @property
    def tape_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-staged-alpha-entropy-tape-")

    def segment(
        self,
        *,
        candidate_attempt_index: int,
        first_draw_index: int,
        draw_count: int,
    ) -> tuple[str, str]:
        if any(isinstance(value, bool) for value in (
            candidate_attempt_index, first_draw_index, draw_count
        )):
            raise TypeError("entropy-tape indices must be integers")
        attempt = int(candidate_attempt_index)
        first = int(first_draw_index)
        count = int(draw_count)
        if (
            attempt < 0
            or attempt >= self.maximum_candidate_count
            or first < 0
            or count <= 0
            or first + count > self.draws_per_candidate
        ):
            raise ValueError("entropy-tape segment is outside the registered campaign bounds")
        if self.entropy_source_id == STAGED_ALPHA_LEGACY_SYSTEM_ENTROPY_SOURCE_ID:
            word_offset = attempt * self.draws_per_candidate + first
            start = 16 * word_offset
            stop = start + 16 * count
            entropy = self.entropy_hex[start:stop]
        else:
            first_block = first // 4
            final_block = (first + count - 1) // 4
            root = bytes.fromhex(self.entropy_hex)
            blocks = bytearray()
            root_commitment = content_hash(
                self.entropy_hex,
                prefix="partition-free-v1-staged-alpha-counter-root-",
            )
            for block_index in range(first_block, final_block + 1):
                block_binding = content_hash(
                    {
                        "schema": "partition_free_v1_staged_alpha_counter_block_v1",
                        "root_commitment": root_commitment,
                        "campaign_contract_hash": self.campaign_contract_hash,
                        "controller_run_id": self.controller_run_id,
                        "executor_identity_hash": self.executor_identity_hash,
                        "maximum_candidate_count": self.maximum_candidate_count,
                        "draws_per_candidate": self.draws_per_candidate,
                        "candidate_attempt_index": attempt,
                        "uint64_block_index": block_index,
                    },
                    prefix="partition-free-v1-staged-alpha-counter-block-",
                )
                blocks.extend(
                    hmac.new(
                        root,
                        block_binding.encode("ascii"),
                        hashlib.sha256,
                    ).digest()
                )
            within_first_block = 8 * (first % 4)
            byte_count = 8 * count
            entropy = bytes(
                blocks[
                    within_first_block : within_first_block + byte_count
                ]
            ).hex()
        receipt = content_hash(
            {
                "tape_hash": self.tape_hash,
                "candidate_attempt_index": attempt,
                "first_draw_index": first,
                "draw_count": count,
                "entropy_hex": entropy,
            },
            prefix="partition-free-v1-staged-alpha-entropy-segment-",
        )
        return entropy, receipt


def issue_v1_staged_alpha_entropy_tape(
    *,
    campaign_contract_hash: str,
    controller_run_id: str,
    executor_identity_hash: str,
    maximum_candidate_count: int,
    draws_per_candidate: int,
) -> V1StagedAlphaEntropyTape:
    """Issue one constant-storage counter stream before the trajectory begins."""

    maximum = _positive_int(maximum_candidate_count, "maximum_candidate_count")
    per_candidate = _positive_int(draws_per_candidate, "draws_per_candidate")
    return V1StagedAlphaEntropyTape(
        campaign_contract_hash=_nonempty(campaign_contract_hash, "campaign_contract_hash"),
        controller_run_id=_nonempty(controller_run_id, "controller_run_id"),
        executor_identity_hash=_nonempty(executor_identity_hash, "executor_identity_hash"),
        maximum_candidate_count=maximum,
        draws_per_candidate=per_candidate,
        entropy_hex=secrets.token_bytes(32).hex(),
    )


@dataclass(frozen=True, slots=True, init=False)
class V1StagedAlphaSamplingPlan:
    """Sealed, exactly reproducible with-replacement categorical draws."""

    population_context_ids: tuple[str, ...]
    sampling_masses_u64: tuple[int, ...]
    entropy_hex: str
    entropy_source_receipt_hash: str
    entropy_source_id: str
    birth_binding: tuple[tuple[str, str], ...]
    sampling_law_hash: str
    draw_ids: tuple[str, ...]
    source_context_ids: tuple[str, ...]
    algorithm_id: str = STAGED_ALPHA_SAMPLING_ID
    schema: str = STAGED_ALPHA_SAMPLING_PLAN_SCHEMA

    def __init__(self, *, _issuer_token: object, **values: object) -> None:
        if _issuer_token not in {_SYSTEM_SAMPLING_TOKEN, _EXTERNAL_SAMPLING_TOKEN}:
            raise TypeError("staged-alpha sampling plans are factory-issued only")
        expected = tuple(
            name
            for name in self.__dataclass_fields__
            if name not in {"algorithm_id", "schema"}
        )
        if set(values) != set(expected):
            raise TypeError("staged-alpha sampling-plan fields differ")
        for name in expected:
            object.__setattr__(self, name, values[name])
        object.__setattr__(self, "algorithm_id", STAGED_ALPHA_SAMPLING_ID)
        object.__setattr__(self, "schema", STAGED_ALPHA_SAMPLING_PLAN_SCHEMA)
        self.__post_init__()
        if (
            _issuer_token is _SYSTEM_SAMPLING_TOKEN
            and self.entropy_source_id != STAGED_ALPHA_SYSTEM_ENTROPY_SOURCE_ID
        ) or (
            _issuer_token is _EXTERNAL_SAMPLING_TOKEN
            and self.entropy_source_id != STAGED_ALPHA_EXTERNAL_ENTROPY_SOURCE_ID
        ):
            raise ValueError("sampling-plan issuer and entropy source differ")

    def __post_init__(self) -> None:
        population = tuple(
            _nonempty(value, "population_context_id")
            for value in self.population_context_ids
        )
        draws = tuple(_nonempty(value, "draw_id") for value in self.draw_ids)
        sources = tuple(
            _nonempty(value, "source_context_id") for value in self.source_context_ids
        )
        masses = tuple(
            _positive_int(value, "sampling mass") for value in self.sampling_masses_u64
        )
        population_set = set(population)
        if not population or len(population_set) != len(population):
            raise ValueError("sampling-plan population must be nonempty and unique")
        if len(masses) != len(population) or sum(masses) != _SAMPLING_MASS_TOTAL:
            raise ValueError("sampling-plan masses must form an exact 64-bit law")
        if not draws or len(set(draws)) != len(draws) or len(draws) != len(sources):
            raise ValueError("sampling-plan draw/source axes are invalid")
        if any(value not in population_set for value in sources):
            raise ValueError("sampling-plan source is outside its population")
        object.__setattr__(self, "population_context_ids", population)
        object.__setattr__(self, "sampling_masses_u64", masses)
        object.__setattr__(
            self,
            "entropy_hex",
            _entropy_hex(self.entropy_hex, draw_count=len(draws)),
        )
        object.__setattr__(
            self,
            "entropy_source_receipt_hash",
            _nonempty(self.entropy_source_receipt_hash, "entropy_source_receipt_hash"),
        )
        source = _nonempty(self.entropy_source_id, "entropy_source_id")
        if source not in {
            STAGED_ALPHA_SYSTEM_ENTROPY_SOURCE_ID,
            STAGED_ALPHA_EXTERNAL_ENTROPY_SOURCE_ID,
        }:
            raise ValueError("unsupported staged-alpha entropy source")
        binding = tuple(
            sorted(
                (
                    _nonempty(key, "birth binding key"),
                    _nonempty(value, "birth binding value"),
                )
                for key, value in self.birth_binding
            )
        )
        if (
            not binding
            or len({key for key, _value in binding}) != len(binding)
            or dict(binding).get("sampling_law_hash") != self.sampling_law_hash
        ):
            raise ValueError("staged-alpha sampling requires a bound birth identity")
        if source == STAGED_ALPHA_SYSTEM_ENTROPY_SOURCE_ID and (
            dict(binding).get("pre_draw_filtration_hash") is None
            or dict(binding).get("step_freeze_hash") is None
        ):
            raise ValueError("system entropy requires pre-draw step and filtration binding")
        object.__setattr__(self, "entropy_source_id", source)
        object.__setattr__(self, "birth_binding", binding)
        object.__setattr__(
            self,
            "sampling_law_hash",
            _nonempty(self.sampling_law_hash, "sampling_law_hash"),
        )
        object.__setattr__(self, "draw_ids", draws)
        object.__setattr__(self, "source_context_ids", sources)
        if (
            self.algorithm_id != STAGED_ALPHA_SAMPLING_ID
            or self.schema != STAGED_ALPHA_SAMPLING_PLAN_SCHEMA
        ):
            raise ValueError("unsupported staged-alpha sampling-plan identity")
        self.verify()

    @classmethod
    def create(
        cls,
        *,
        population_context_ids: Sequence[str],
        sampling_population_probabilities: Sequence[float],
        entropy_hex: str,
        entropy_source_receipt_hash: str,
        birth_binding: Mapping[str, str],
        sampling_law_hash: str,
        draw_ids: Sequence[str],
    ) -> "V1StagedAlphaSamplingPlan":
        population = tuple(str(value) for value in population_context_ids)
        probabilities = _probabilities(
            sampling_population_probabilities,
            "sampling population probabilities",
            len(population),
        )
        if not math.isclose(
            math.fsum(float(value) for value in probabilities),
            1.0,
            rel_tol=0.0,
            abs_tol=64.0 * np.finfo(np.float64).eps,
        ):
            raise ValueError("sampling population probabilities must sum to one")
        masses = _positive_sampling_masses(probabilities)
        return cls._create_external_from_masses(
            population_context_ids=population,
            sampling_masses_u64=masses,
            entropy_hex=entropy_hex,
            entropy_source_receipt_hash=entropy_source_receipt_hash,
            birth_binding=birth_binding,
            sampling_law_hash=sampling_law_hash,
            draw_ids=draw_ids,
        )

    @classmethod
    def _create_external_from_masses(
        cls,
        *,
        population_context_ids: Sequence[str],
        sampling_masses_u64: Sequence[int],
        entropy_hex: str,
        entropy_source_receipt_hash: str,
        birth_binding: Mapping[str, str],
        sampling_law_hash: str,
        draw_ids: Sequence[str],
    ) -> "V1StagedAlphaSamplingPlan":
        population = tuple(str(value) for value in population_context_ids)
        masses = tuple(
            _positive_int(value, "sampling mass") for value in sampling_masses_u64
        )
        if len(masses) != len(population) or sum(masses) != _SAMPLING_MASS_TOTAL:
            raise ValueError("sampling-plan masses must form an exact 64-bit law")
        draws = tuple(str(value) for value in draw_ids)
        entropy = _entropy_hex(entropy_hex, draw_count=len(draws))
        law_hash = _nonempty(sampling_law_hash, "sampling_law_hash")
        indices = _sampling_source_indices(
            entropy_hex=entropy,
            draw_ids=draws,
            masses=masses,
        )
        return cls(
            _issuer_token=_EXTERNAL_SAMPLING_TOKEN,
            population_context_ids=population,
            sampling_masses_u64=masses,
            entropy_hex=entropy,
            entropy_source_receipt_hash=entropy_source_receipt_hash,
            entropy_source_id=STAGED_ALPHA_EXTERNAL_ENTROPY_SOURCE_ID,
            birth_binding=tuple(
                sorted((str(key), str(value)) for key, value in birth_binding.items())
            ),
            sampling_law_hash=law_hash,
            draw_ids=draws,
            source_context_ids=tuple(population[index] for index in indices),
        )

    @classmethod
    def draw_system_entropy(
        cls,
        *,
        _issuer_token: object,
        population_context_ids: Sequence[str],
        sampling_population_probabilities: Sequence[float],
        birth_binding: Mapping[str, str],
        sampling_law_hash: str,
        draw_ids: Sequence[str],
    ) -> "V1StagedAlphaSamplingPlan":
        """Issue fresh OS-backed entropy after the pre-draw birth is frozen."""

        if _issuer_token is not _SYSTEM_SAMPLING_TOKEN:
            raise TypeError("system-entropy sampling plans are factory-issued only")
        draws = tuple(str(value) for value in draw_ids)
        entropy = "".join(secrets.token_bytes(8).hex() for _draw in draws)
        binding = tuple(
            sorted((str(key), str(value)) for key, value in birth_binding.items())
        )
        entropy_receipt = content_hash(
            {
                "entropy_source_id": STAGED_ALPHA_SYSTEM_ENTROPY_SOURCE_ID,
                "birth_binding": binding,
                "draw_ids": draws,
                "entropy_hex": entropy,
            },
            prefix="partition-free-v1-staged-alpha-system-entropy-",
        )
        external = cls.create(
            population_context_ids=population_context_ids,
            sampling_population_probabilities=sampling_population_probabilities,
            entropy_hex=entropy,
            entropy_source_receipt_hash=entropy_receipt,
            birth_binding=dict(binding),
            sampling_law_hash=sampling_law_hash,
            draw_ids=draws,
        )
        return cls(
            _issuer_token=_SYSTEM_SAMPLING_TOKEN,
            population_context_ids=external.population_context_ids,
            sampling_masses_u64=external.sampling_masses_u64,
            entropy_hex=external.entropy_hex,
            entropy_source_receipt_hash=external.entropy_source_receipt_hash,
            entropy_source_id=STAGED_ALPHA_SYSTEM_ENTROPY_SOURCE_ID,
            birth_binding=external.birth_binding,
            sampling_law_hash=external.sampling_law_hash,
            draw_ids=external.draw_ids,
            source_context_ids=external.source_context_ids,
        )

    @classmethod
    def from_preissued_system_entropy(
        cls,
        *,
        _issuer_token: object,
        population_context_ids: Sequence[str],
        sampling_population_probabilities: Sequence[float] | None = None,
        sampling_masses_u64: Sequence[int] | None = None,
        entropy_hex: str,
        entropy_source_receipt_hash: str,
        birth_binding: Mapping[str, str],
        sampling_law_hash: str,
        draw_ids: Sequence[str],
    ) -> "V1StagedAlphaSamplingPlan":
        if _issuer_token is not _SYSTEM_SAMPLING_TOKEN:
            raise TypeError("preissued system-entropy plans are factory-issued only")
        if (sampling_population_probabilities is None) == (sampling_masses_u64 is None):
            raise ValueError(
                "preissued sampling requires exactly one probability or exact-mass law"
            )
        if sampling_masses_u64 is None:
            assert sampling_population_probabilities is not None
            external = cls.create(
                population_context_ids=population_context_ids,
                sampling_population_probabilities=sampling_population_probabilities,
                entropy_hex=entropy_hex,
                entropy_source_receipt_hash=entropy_source_receipt_hash,
                birth_binding=birth_binding,
                sampling_law_hash=sampling_law_hash,
                draw_ids=draw_ids,
            )
        else:
            external = cls._create_external_from_masses(
                population_context_ids=population_context_ids,
                sampling_masses_u64=sampling_masses_u64,
                entropy_hex=entropy_hex,
                entropy_source_receipt_hash=entropy_source_receipt_hash,
                birth_binding=birth_binding,
                sampling_law_hash=sampling_law_hash,
                draw_ids=draw_ids,
            )
        return cls(
            _issuer_token=_SYSTEM_SAMPLING_TOKEN,
            population_context_ids=external.population_context_ids,
            sampling_masses_u64=external.sampling_masses_u64,
            entropy_hex=external.entropy_hex,
            entropy_source_receipt_hash=external.entropy_source_receipt_hash,
            entropy_source_id=STAGED_ALPHA_SYSTEM_ENTROPY_SOURCE_ID,
            birth_binding=external.birth_binding,
            sampling_law_hash=external.sampling_law_hash,
            draw_ids=external.draw_ids,
            source_context_ids=external.source_context_ids,
        )

    @property
    def sampling_population_probabilities(self) -> tuple[float, ...]:
        return tuple(value / _SAMPLING_MASS_TOTAL for value in self.sampling_masses_u64)

    @property
    def plan_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-staged-alpha-sampling-plan-")

    def verify(self) -> None:
        indices = _sampling_source_indices(
            entropy_hex=self.entropy_hex,
            draw_ids=self.draw_ids,
            masses=self.sampling_masses_u64,
        )
        regenerated = tuple(self.population_context_ids[index] for index in indices)
        if regenerated != self.source_context_ids:
            raise ValueError("staged-alpha sampling-plan reveal is not reproducible")

    @property
    def runtime_generated(self) -> bool:
        return self.entropy_source_id == STAGED_ALPHA_SYSTEM_ENTROPY_SOURCE_ID


def draw_v1_staged_alpha_sampling_plan(
    *,
    step_freeze: "V1StagedAlphaStepFreeze",
    population_context_ids: Sequence[str],
    sampling_population_probabilities: Sequence[float],
    draw_ids: Sequence[str],
    executor_identity_hash: str,
) -> V1StagedAlphaSamplingPlan:
    """Public factory for post-freeze, pre-outcome system-randomized draws."""

    if type(step_freeze) is not V1StagedAlphaStepFreeze:
        raise TypeError("sampling-plan draw requires an exact staged-alpha step freeze")
    freeze = step_freeze.certificate_freeze
    binding = {
        "registration_hash": step_freeze.registration_hash,
        "campaign_contract_hash": step_freeze.campaign_contract_hash,
        "step_freeze_hash": step_freeze.step_freeze_hash,
        "pre_draw_filtration_hash": freeze.pre_draw_filtration_hash,
        "sampling_law_hash": freeze.sampling_law_hash,
        "importance_weight_hash": freeze.importance_weight_hash,
        "executor_identity_hash": _nonempty(
            executor_identity_hash,
            "executor_identity_hash",
        ),
    }
    return V1StagedAlphaSamplingPlan.draw_system_entropy(
        _issuer_token=_SYSTEM_SAMPLING_TOKEN,
        population_context_ids=population_context_ids,
        sampling_population_probabilities=sampling_population_probabilities,
        birth_binding=binding,
        sampling_law_hash=freeze.sampling_law_hash,
        draw_ids=draw_ids,
    )


def draw_v1_staged_alpha_sampling_plan_from_tape(
    *,
    step_freeze: "V1StagedAlphaStepFreeze",
    entropy_tape: V1StagedAlphaEntropyTape,
    candidate_attempt_index: int,
    first_draw_index: int,
    population_context_ids: Sequence[str],
    sampling_population_probabilities: Sequence[float] | None = None,
    sampling_masses_u64: Sequence[int] | None = None,
    draw_ids: Sequence[str],
    executor_identity_hash: str,
) -> V1StagedAlphaSamplingPlan:
    """Address one exact system-entropy segment after the step freeze."""

    if type(step_freeze) is not V1StagedAlphaStepFreeze:
        raise TypeError("sampling-plan draw requires an exact staged-alpha step freeze")
    if type(entropy_tape) is not V1StagedAlphaEntropyTape:
        raise TypeError("sampling-plan draw requires an exact system-entropy tape")
    executor = _nonempty(executor_identity_hash, "executor_identity_hash")
    if (
        entropy_tape.campaign_contract_hash != step_freeze.campaign_contract_hash
        or entropy_tape.executor_identity_hash != executor
    ):
        raise ValueError("system-entropy tape names another campaign or executor")
    if (
        isinstance(candidate_attempt_index, bool)
        or not isinstance(candidate_attempt_index, Integral)
        or int(candidate_attempt_index) < 0
    ):
        raise ValueError("candidate_attempt_index must be a nonnegative integer")
    attempt_index = int(candidate_attempt_index)
    if attempt_index != step_freeze.candidate_attempt_index:
        raise ValueError(
            "system-entropy segment differs from the frozen candidate attempt"
        )
    draws = tuple(str(value) for value in draw_ids)
    entropy, receipt = entropy_tape.segment(
        candidate_attempt_index=attempt_index,
        first_draw_index=first_draw_index,
        draw_count=len(draws),
    )
    freeze = step_freeze.certificate_freeze
    binding = {
        "registration_hash": step_freeze.registration_hash,
        "campaign_contract_hash": step_freeze.campaign_contract_hash,
        "step_freeze_hash": step_freeze.step_freeze_hash,
        "pre_draw_filtration_hash": freeze.pre_draw_filtration_hash,
        "sampling_law_hash": freeze.sampling_law_hash,
        "importance_weight_hash": freeze.importance_weight_hash,
        "executor_identity_hash": executor,
    }
    return V1StagedAlphaSamplingPlan.from_preissued_system_entropy(
        _issuer_token=_SYSTEM_SAMPLING_TOKEN,
        population_context_ids=population_context_ids,
        sampling_population_probabilities=sampling_population_probabilities,
        sampling_masses_u64=sampling_masses_u64,
        entropy_hex=entropy,
        entropy_source_receipt_hash=receipt,
        birth_binding=binding,
        sampling_law_hash=freeze.sampling_law_hash,
        draw_ids=draws,
    )


@dataclass(frozen=True, slots=True)
class V1StagedAlphaCoordinates:
    """Owner-selected per-attempt stake from one finite campaign alpha budget."""

    campaign_wide_alpha: float
    per_step_alpha: float
    maximum_candidate_count: int
    maximum_looks_per_step: int
    maximum_importance_weight: float
    campaign_contract_hash: str
    registered_cumulative_draw_counts: tuple[int, ...]
    look_schedule_policy_hash: str
    proof_artifact_hash: str | None = None
    owner_decision_record_hash: str | None = None
    importance_weight_bound_proof_hash: str | None = None
    entropy_source_policy_hash: str | None = None
    deferred_family: Literal["grow_eprocess"] = "grow_eprocess"
    cross_query_shrinkage: Literal["defer"] = "defer"
    step_spending_id: str = STAGED_ALPHA_STEP_SPENDING_ID
    look_spending_id: str = STAGED_ALPHA_LOOK_SPENDING_ID
    bound_id: str = STAGED_ALPHA_BOUND_ID
    sampling_id: str = STAGED_ALPHA_SAMPLING_ID
    proof_id: str = STAGED_ALPHA_PROOF_ID
    exhaustion_action_id: str = STAGED_ALPHA_EXHAUSTION_ACTION_ID
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = STAGED_ALPHA_COORDINATE_SCHEMA

    def __post_init__(self) -> None:
        campaign_alpha = float(self.campaign_wide_alpha)
        alpha = float(self.per_step_alpha)
        if not math.isfinite(campaign_alpha) or not 0.0 < campaign_alpha < 1.0:
            raise ValueError("campaign_wide_alpha must lie in (0,1)")
        if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("per_step_alpha must lie in (0,1)")
        maximum_candidates = _positive_int(
            self.maximum_candidate_count,
            "maximum_candidate_count",
        )
        first_attempt_alpha = campaign_alpha / math.fsum(
            1.0 / float((index + 1) ** 2)
            for index in range(maximum_candidates)
        )
        if alpha != first_attempt_alpha:
            raise ValueError(
                "per_step_alpha must equal the registered first-attempt "
                "inverse-square campaign stake"
            )
        object.__setattr__(self, "campaign_wide_alpha", campaign_alpha)
        object.__setattr__(self, "per_step_alpha", alpha)
        object.__setattr__(self, "maximum_candidate_count", maximum_candidates)
        object.__setattr__(
            self,
            "maximum_looks_per_step",
            _positive_int(self.maximum_looks_per_step, "maximum_looks_per_step"),
        )
        object.__setattr__(
            self,
            "maximum_importance_weight",
            _finite_positive(self.maximum_importance_weight, "maximum_importance_weight"),
        )
        object.__setattr__(
            self,
            "campaign_contract_hash",
            _nonempty(self.campaign_contract_hash, "campaign_contract_hash"),
        )
        counts = tuple(
            _positive_int(value, "registered cumulative draw count")
            for value in self.registered_cumulative_draw_counts
        )
        if (
            not counts
            or any(right <= left for left, right in zip(counts, counts[1:]))
            or len(counts) != self.maximum_looks_per_step
        ):
            raise ValueError(
                "registered cumulative draw counts must be strictly increasing "
                "and match maximum_looks_per_step"
            )
        object.__setattr__(self, "registered_cumulative_draw_counts", counts)
        object.__setattr__(
            self,
            "look_schedule_policy_hash",
            _nonempty(self.look_schedule_policy_hash, "look_schedule_policy_hash"),
        )
        expected = {
            "step_spending_id": STAGED_ALPHA_STEP_SPENDING_ID,
            "look_spending_id": STAGED_ALPHA_LOOK_SPENDING_ID,
            "bound_id": STAGED_ALPHA_BOUND_ID,
            "sampling_id": STAGED_ALPHA_SAMPLING_ID,
            "proof_id": STAGED_ALPHA_PROOF_ID,
            "exhaustion_action_id": STAGED_ALPHA_EXHAUSTION_ACTION_ID,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"unsupported staged-alpha {name}")
        if (
            self.deferred_family != "grow_eprocess"
            or self.cross_query_shrinkage != "defer"
            or self.schema != STAGED_ALPHA_COORDINATE_SCHEMA
        ):
            raise ValueError("staged-alpha owner selection or schema drifted")
        for name in (
            "proof_artifact_hash",
            "owner_decision_record_hash",
            "importance_weight_bound_proof_hash",
            "entropy_source_policy_hash",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonempty(value, name))
        object.__setattr__(self, "metadata", _freeze(self.metadata, "staged-alpha metadata"))

    @property
    def coordinates_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-staged-alpha-coordinates-")

    def attempt_alpha(self, candidate_attempt_index: int) -> float:
        if (
            isinstance(candidate_attempt_index, bool)
            or not isinstance(candidate_attempt_index, Integral)
            or not 0 <= int(candidate_attempt_index) < self.maximum_candidate_count
        ):
            raise ValueError(
                "candidate_attempt_index is outside the registered finite horizon"
            )
        normalization = math.fsum(
            1.0 / float((index + 1) ** 2)
            for index in range(self.maximum_candidate_count)
        )
        result = (
            self.campaign_wide_alpha
            / normalization
            / float((int(candidate_attempt_index) + 1) ** 2)
        )
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError("attempt alpha is outside the float64 representable range")
        return result

    def look_alpha(
        self,
        look_index: int,
        *,
        candidate_attempt_index: int = 0,
    ) -> float:
        if (
            isinstance(look_index, bool)
            or not isinstance(look_index, Integral)
            or not 0 <= int(look_index) < self.maximum_looks_per_step
        ):
            raise ValueError("look_index is outside the registered finite look set")
        result = self.attempt_alpha(candidate_attempt_index) / float(
            self.maximum_looks_per_step
        )
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError("look alpha is outside the float64 representable range")
        return result

    @property
    def proof_attached(self) -> bool:
        return self.proof_artifact_hash is not None

    @property
    def campaign_coordinates_complete(self) -> bool:
        return all(
            value is not None
            for value in (
                self.proof_artifact_hash,
                self.owner_decision_record_hash,
                self.importance_weight_bound_proof_hash,
                self.entropy_source_policy_hash,
                self.campaign_contract_hash,
            )
        )


@dataclass(frozen=True, slots=True)
class V1StagedAlphaStepFreeze:
    """Candidate-specific look schedule and damage threshold frozen pre-draw."""

    registration_hash: str
    campaign_contract_hash: str
    certificate_freeze: V1CertificateFreeze
    coordinates_hash: str
    step_index: int
    candidate_attempt_index: int
    maximum_mean_damage: float
    cumulative_draw_counts: tuple[int, ...]
    schedule_construction_hash: str
    schema: str = STAGED_ALPHA_STEP_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "registration_hash", _nonempty(self.registration_hash, "registration_hash"))
        object.__setattr__(
            self,
            "campaign_contract_hash",
            _nonempty(self.campaign_contract_hash, "campaign_contract_hash"),
        )
        if type(self.certificate_freeze) is not V1CertificateFreeze:
            raise TypeError("staged-alpha step requires an exact certificate freeze")
        if self.certificate_freeze.registration_hash != self.registration_hash:
            raise ValueError("certificate freeze names another registration")
        object.__setattr__(self, "coordinates_hash", _nonempty(self.coordinates_hash, "coordinates_hash"))
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, Integral) or self.step_index < 0:
            raise ValueError("step_index must be a nonnegative integer")
        object.__setattr__(self, "step_index", int(self.step_index))
        if (
            isinstance(self.candidate_attempt_index, bool)
            or not isinstance(self.candidate_attempt_index, Integral)
            or self.candidate_attempt_index < 0
        ):
            raise ValueError("candidate_attempt_index must be a nonnegative integer")
        object.__setattr__(
            self,
            "candidate_attempt_index",
            int(self.candidate_attempt_index),
        )
        threshold = float(self.maximum_mean_damage)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("maximum_mean_damage must be finite and nonnegative")
        object.__setattr__(self, "maximum_mean_damage", threshold)
        counts = tuple(_positive_int(value, "cumulative draw count") for value in self.cumulative_draw_counts)
        if not counts or any(right <= left for left, right in zip(counts, counts[1:])):
            raise ValueError("cumulative draw counts must be strictly increasing")
        object.__setattr__(self, "cumulative_draw_counts", counts)
        object.__setattr__(
            self,
            "schedule_construction_hash",
            _nonempty(self.schedule_construction_hash, "schedule_construction_hash"),
        )
        if self.schema != STAGED_ALPHA_STEP_SCHEMA:
            raise ValueError("unsupported staged-alpha step-freeze schema")

    @property
    def step_freeze_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-staged-alpha-step-freeze-")


def bind_v1_staged_alpha_step(
    *,
    registration: V1OwnerRegistration,
    coordinates: V1StagedAlphaCoordinates,
    certificate_freeze: V1CertificateFreeze,
    step_index: int,
    candidate_attempt_index: int,
    maximum_mean_damage: float,
    cumulative_draw_counts: Sequence[int],
    schedule_construction_hash: str,
) -> V1StagedAlphaStepFreeze:
    """Bind the selected coordinates and exact schedule before any outcome."""

    if type(registration) is not V1OwnerRegistration:
        raise TypeError("staged-alpha binding requires an exact owner registration")
    if type(coordinates) is not V1StagedAlphaCoordinates:
        raise TypeError("staged-alpha binding requires exact coordinates")
    if registration.certificate_family != "staged_alpha":
        raise ValueError("owner registration did not select staged-alpha")
    if registration.cross_query_shrinkage != "defer":
        raise ValueError("initial staged-alpha registration requires deferred cross-query shrinkage")
    if registration.a1_coordinates_hash != coordinates.coordinates_hash:
        raise ValueError("owner registration names different staged-alpha coordinates")
    if registration.campaign_contract_hash != coordinates.campaign_contract_hash:
        raise ValueError("owner registration names a different campaign contract")
    if registration.registration_hash != certificate_freeze.registration_hash:
        raise ValueError("certificate freeze names another owner registration")
    counts = tuple(cumulative_draw_counts)
    if counts != coordinates.registered_cumulative_draw_counts:
        raise ValueError("step schedule differs from the registered campaign schedule")
    if schedule_construction_hash != coordinates.look_schedule_policy_hash:
        raise ValueError("step schedule source differs from the campaign contract")
    coordinates.attempt_alpha(candidate_attempt_index)
    return V1StagedAlphaStepFreeze(
        registration_hash=registration.registration_hash,
        campaign_contract_hash=coordinates.campaign_contract_hash,
        certificate_freeze=certificate_freeze,
        coordinates_hash=coordinates.coordinates_hash,
        step_index=step_index,
        candidate_attempt_index=candidate_attempt_index,
        maximum_mean_damage=maximum_mean_damage,
        cumulative_draw_counts=counts,
        schedule_construction_hash=schedule_construction_hash,
    )


@dataclass(frozen=True, slots=True, init=False)
class V1StagedAlphaPacketEnvelope:
    """Factory-owned importance-weighted damage rows for one certificate look."""

    step_freeze_hash: str
    evidence_packet_hash: str
    request_id: str
    outcome_integrity_hash: str
    look_index: int
    draw_ids: tuple[str, ...]
    source_context_ids: tuple[str, ...]
    weighted_damage_payload: bytes = field(repr=False)
    target_probability_hash: str
    sampling_probability_hash: str
    importance_weight_hash: str
    sampling_plan_hash: str
    provider_contract_hash: str
    envelope_receipt_hash: str
    schema: str = STAGED_ALPHA_PACKET_SCHEMA

    def __init__(self, *, _issuer_token: object, **values: object) -> None:
        if _issuer_token is not _PACKET_TOKEN:
            raise TypeError("staged-alpha packet envelopes are factory-issued only")
        expected = tuple(name for name in self.__dataclass_fields__ if name != "schema")
        if set(values) != set(expected):
            raise TypeError("staged-alpha packet envelope fields differ")
        for name in expected:
            object.__setattr__(self, name, values[name])
        object.__setattr__(self, "schema", STAGED_ALPHA_PACKET_SCHEMA)
        self.__post_init__()

    def __post_init__(self) -> None:
        for name in (
            "step_freeze_hash",
            "evidence_packet_hash",
            "request_id",
            "outcome_integrity_hash",
            "target_probability_hash",
            "sampling_probability_hash",
            "importance_weight_hash",
            "sampling_plan_hash",
            "provider_contract_hash",
            "envelope_receipt_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if isinstance(self.look_index, bool) or not isinstance(self.look_index, Integral) or self.look_index < 0:
            raise ValueError("look_index must be a nonnegative integer")
        object.__setattr__(self, "look_index", int(self.look_index))
        draws = tuple(_nonempty(value, "draw_id") for value in self.draw_ids)
        sources = tuple(_nonempty(value, "source_context_id") for value in self.source_context_ids)
        if not draws or len(set(draws)) != len(draws) or len(draws) != len(sources):
            raise ValueError("staged-alpha draw/source lineage is invalid")
        if type(self.weighted_damage_payload) is not bytes or len(self.weighted_damage_payload) != 8 * len(draws):
            raise ValueError("weighted damage payload is not aligned immutable float64")
        if not np.isfinite(self.weighted_damage).all():
            raise ValueError("weighted damage payload contains nonfinite values")
        object.__setattr__(self, "draw_ids", draws)
        object.__setattr__(self, "source_context_ids", sources)
        if self.schema != STAGED_ALPHA_PACKET_SCHEMA:
            raise ValueError("unsupported staged-alpha packet schema")
        expected = content_hash(
            self._receipt_payload(),
            prefix="partition-free-v1-staged-alpha-packet-envelope-",
        )
        if self.envelope_receipt_hash != expected:
            raise ValueError("staged-alpha packet envelope is not reproducible")

    @property
    def weighted_damage(self) -> np.ndarray:
        return np.frombuffer(self.weighted_damage_payload, dtype="<f8")

    def _receipt_payload(self) -> Mapping[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "envelope_receipt_hash"
        }

    @property
    def envelope_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-staged-alpha-packet-state-")


def issue_v1_staged_alpha_packet(
    *,
    coordinates: V1StagedAlphaCoordinates,
    step_freeze: V1StagedAlphaStepFreeze,
    evidence_packet: V1EvidencePacket,
    look_index: int,
    sampling_plan: V1StagedAlphaSamplingPlan,
    provider: PairedMaskOracleProvider,
    request: MeasurementRequest,
    raw_outcome: MeasurementOutcome,
) -> V1StagedAlphaPacketEnvelope:
    """Recompute one certificate packet from direct candidate-damage rows."""

    if type(coordinates) is not V1StagedAlphaCoordinates or type(step_freeze) is not V1StagedAlphaStepFreeze:
        raise TypeError("staged-alpha issuance requires exact coordinate and step objects")
    if step_freeze.coordinates_hash != coordinates.coordinates_hash:
        raise ValueError("step freeze names different staged-alpha coordinates")
    if step_freeze.campaign_contract_hash != coordinates.campaign_contract_hash:
        raise ValueError("step freeze names a different campaign contract")
    if type(evidence_packet) is not V1EvidencePacket:
        raise TypeError("staged-alpha issuance requires an exact evidence packet")
    assert_packet_matches_certificate_freeze(evidence_packet, step_freeze.certificate_freeze)
    if type(provider) is not PairedMaskOracleProvider:
        raise TypeError("staged-alpha issuance requires the direct paired-mask oracle")
    if type(request) is not MeasurementRequest or request.kind != "hard_mask_oracle":
        raise TypeError("staged-alpha issuance requires a hard-mask-oracle request")
    if type(raw_outcome) is not MeasurementOutcome:
        raise TypeError("staged-alpha issuance requires an exact raw outcome")
    validate_outcome_for_request(request, raw_outcome)
    if raw_outcome.integrity_hash != raw_outcome.compute_integrity_hash():
        raise ValueError("raw staged-alpha outcome changed after content addressing")
    if (
        request.request_id != raw_outcome.request_id
        or request.version != MODERN_V1_WORKFLOW_IDENTITY
        or request.state_hash != step_freeze.certificate_freeze.parent_mask_hash
        or request.candidate_ids != (step_freeze.certificate_freeze.candidate_mask_hash,)
        or tuple(raw_outcome.context_ids) != evidence_packet.draw_ids
        or tuple(request.payload.get("draw_ids", ())) != evidence_packet.draw_ids
        or tuple(request.payload.get("source_context_ids", ())) != evidence_packet.source_context_ids
    ):
        raise ValueError("staged-alpha request/outcome/evidence lineage differs")
    if (
        request.payload.get("parent_mask_hash")
        != step_freeze.certificate_freeze.parent_mask_hash
        or request.payload.get("candidate_mask_hash")
        != step_freeze.certificate_freeze.candidate_mask_hash
        or raw_outcome.metadata.get("paired_common_contexts") is not True
        or raw_outcome.metadata.get("candidate_damage_rows_direct") is not True
        or tuple(raw_outcome.metadata.get("physical_source_context_ids", ()))
        != evidence_packet.source_context_ids
        or raw_outcome.metadata.get("counter_addressed_draw_axis") is not True
    ):
        raise ValueError("staged-alpha outcome is not the frozen measured paired transition")
    frozen_payload = {
        "candidate_definition_hash": request.payload.get("candidate_definition_hash"),
        "null_id": request.payload.get("null_id"),
        "loss_id": request.payload.get("loss_id"),
        "sampling_law_hash": request.payload.get("sampling_law_hash"),
        "importance_weight_hash": request.payload.get("importance_weight_hash"),
        "certificate_factor_hash": request.payload.get("certificate_factor_hash"),
        "pre_draw_filtration_hash": request.payload.get("pre_draw_filtration_hash"),
    }
    for name, observed in frozen_payload.items():
        if observed != getattr(step_freeze.certificate_freeze, name):
            raise ValueError(f"staged-alpha request changed frozen field {name}")
    calibration_freeze_hash = step_freeze.certificate_freeze.metadata.get(
        "candidate_calibration_freeze_hash"
    )
    calibration_freeze_required = (
        step_freeze.certificate_freeze.metadata.get(
            "candidate_calibration_freeze_required",
            False,
        )
        is True
    )
    if calibration_freeze_required and not (
        isinstance(calibration_freeze_hash, str) and calibration_freeze_hash
    ):
        raise ValueError("staged-alpha certificate lacks its required calibration freeze")
    request_declares_calibration = (
        "candidate_calibration_freeze_hash" in request.payload
        or "candidate_calibration_freeze_required" in request.payload
    )
    evidence_declares_calibration = (
        "candidate_calibration_freeze_hash" in evidence_packet.metadata
        or "candidate_calibration_freeze_required" in evidence_packet.metadata
    )
    if calibration_freeze_required or request_declares_calibration or evidence_declares_calibration:
        if (
            request.payload.get("candidate_calibration_freeze_hash")
            != calibration_freeze_hash
            or evidence_packet.metadata.get("candidate_calibration_freeze_hash")
            != calibration_freeze_hash
            or request.payload.get("candidate_calibration_freeze_required")
            is not calibration_freeze_required
            or evidence_packet.metadata.get("candidate_calibration_freeze_required")
            is not calibration_freeze_required
        ):
            raise ValueError(
                "staged-alpha lineage changed the candidate calibration freeze"
            )
    if request.payload.get("stream_role") != evidence_packet.role:
        raise ValueError("staged-alpha request changed its certificate stream role")
    if request.payload.get("draw_reuse_status") != "fresh_candidate_certificate":
        raise ValueError("staged-alpha certificate draws must be declared fresh")
    index = int(look_index)
    coordinates.look_alpha(index)
    if index >= len(step_freeze.cumulative_draw_counts):
        raise ValueError("look index is outside the step schedule")
    if type(sampling_plan) is not V1StagedAlphaSamplingPlan:
        raise TypeError("staged-alpha issuance requires an exact sampling plan")
    sampling_plan.verify()
    birth_binding = dict(sampling_plan.birth_binding)
    if not sampling_plan.runtime_generated or birth_binding != {
        "executor_identity_hash": birth_binding.get("executor_identity_hash"),
        "campaign_contract_hash": step_freeze.campaign_contract_hash,
        "importance_weight_hash": step_freeze.certificate_freeze.importance_weight_hash,
        "pre_draw_filtration_hash": step_freeze.certificate_freeze.pre_draw_filtration_hash,
        "registration_hash": step_freeze.registration_hash,
        "sampling_law_hash": step_freeze.certificate_freeze.sampling_law_hash,
        "step_freeze_hash": step_freeze.step_freeze_hash,
    }:
        raise ValueError(
            "staged-alpha certificate requires a runtime system-entropy plan "
            "bound after the exact pre-draw freeze"
        )
    if (
        sampling_plan.draw_ids != evidence_packet.draw_ids
        or sampling_plan.source_context_ids != evidence_packet.source_context_ids
        or sampling_plan.sampling_law_hash != request.payload.get("sampling_law_hash")
        or sampling_plan.plan_hash != request.payload.get("sampling_plan_hash")
    ):
        raise ValueError("staged-alpha realized draws differ from the sealed sampling plan")

    count = len(evidence_packet.draw_ids)
    population = tuple(request.payload.get("population_context_ids", ()))
    if population != tuple(provider.context_ids):
        raise ValueError("staged-alpha probability law does not cover the exact provider population")
    population_count = len(population)
    target_population = _probabilities(
        request.payload.get("target_population_probabilities"),
        "target population probabilities",
        population_count,
        allow_zero=True,
    )
    sampling_population = _probabilities(
        request.payload.get("sampling_population_probabilities"),
        "sampling population probabilities",
        population_count,
    )
    if sampling_plan.population_context_ids != population or not np.array_equal(
        sampling_population,
        np.asarray(sampling_plan.sampling_population_probabilities, dtype=np.float64),
    ):
        raise ValueError("staged-alpha request probability law differs from its sampling plan")
    probability_sum_tolerance = 64.0 * np.finfo(np.float64).eps
    if (
        not math.isclose(
            float(np.sum(target_population, dtype=np.float64)),
            1.0,
            rel_tol=0.0,
            abs_tol=probability_sum_tolerance,
        )
        or not math.isclose(
            float(np.sum(sampling_population, dtype=np.float64)),
            1.0,
            rel_tol=0.0,
            abs_tol=probability_sum_tolerance,
        )
    ):
        raise ValueError("staged-alpha population probability laws must sum to one")
    probability_identity = content_hash(
        {
            "population_context_ids": population,
            "target_population_probabilities": target_population,
            "sampling_population_probabilities": sampling_population,
        },
        prefix="partition-free-v1-importance-probabilities-",
    )
    if probability_identity != step_freeze.certificate_freeze.importance_weight_hash:
        raise ValueError("staged-alpha probabilities differ from the frozen importance law")
    population_index = {context: index for index, context in enumerate(population)}
    source_indices = np.asarray(
        [population_index[source] for source in evidence_packet.source_context_ids],
        dtype=np.int64,
    )
    weights = target_population[source_indices] / sampling_population[source_indices]
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, coordinates.maximum_importance_weight)
    population_weights = target_population / sampling_population
    if bool(np.any(population_weights > coordinates.maximum_importance_weight + tolerance)):
        raise ValueError("observed importance weight exceeds its registered bound")

    if provider.calibration_state_id != raw_outcome.calibration_state_id:
        raise ValueError("staged-alpha provider/calibration identity differs")
    values = np.asarray(raw_outcome.values, dtype=np.float64)
    if values.shape not in {(count,), (count, 1)}:
        raise ValueError("staged-alpha candidate-damage rows have the wrong shape")
    row_damage = values.reshape(count)
    weighted = np.ascontiguousarray(weights * row_damage, dtype="<f8")
    provider_hash = content_hash(
        {
            "context_ids": tuple(provider.context_ids),
            "calibration_state_id": provider.calibration_state_id,
            "candidate_calibration_freeze_hash": calibration_freeze_hash,
            "evidence_semantics": "direct_candidate_damage_increment_v1",
        },
        prefix="partition-free-v1-staged-alpha-provider-",
    )
    payload = {
        "step_freeze_hash": step_freeze.step_freeze_hash,
        "evidence_packet_hash": evidence_packet.packet_hash,
        "request_id": request.request_id,
        "outcome_integrity_hash": raw_outcome.integrity_hash,
        "look_index": index,
        "draw_ids": evidence_packet.draw_ids,
        "source_context_ids": evidence_packet.source_context_ids,
        "weighted_damage_payload": weighted.tobytes(order="C"),
        "target_probability_hash": content_hash(
            {
                "population_context_ids": population,
                "probabilities": target_population,
            },
            prefix="partition-free-v1-target-probability-",
        ),
        "sampling_probability_hash": content_hash(
            {
                "population_context_ids": population,
                "probabilities": sampling_population,
            },
            prefix="partition-free-v1-sampling-probability-",
        ),
        "importance_weight_hash": probability_identity,
        "sampling_plan_hash": sampling_plan.plan_hash,
        "provider_contract_hash": provider_hash,
    }
    receipt_hash = content_hash(
        {"schema": STAGED_ALPHA_PACKET_SCHEMA, **payload},
        prefix="partition-free-v1-staged-alpha-packet-envelope-",
    )
    return V1StagedAlphaPacketEnvelope(
        _issuer_token=_PACKET_TOKEN,
        envelope_receipt_hash=receipt_hash,
        **payload,
    )


def _reissue_v1_staged_alpha_packet_with_transformed_rows(
    *,
    _issuer_token: object,
    raw_packet: V1StagedAlphaPacketEnvelope,
    transformed_weighted_rows: Sequence[float] | np.ndarray,
    transformation_receipt_hash: str,
    evidence_semantics_id: str,
) -> V1StagedAlphaPacketEnvelope:
    """Factory-reissue one verified packet with a predictable row transform.

    This is the narrow compatibility bridge used by separately versioned
    statistical amendments.  ``raw_packet`` must already have passed the full
    direct-oracle lineage, support, W_max, and source verification performed by
    :func:`issue_v1_staged_alpha_packet`.  The caller supplies only an aligned
    finite row vector plus a content-addressed transformation receipt.  The
    resulting packet binds the raw envelope and transformation into a distinct
    provider-contract identity while preserving the exact draw/request/outcome
    lineage.

    This function does not decide a candidate and does not authorize a mask
    commit.  A transformed-row theorem and amendment-specific freeze remain the
    caller's responsibility.
    """

    if _issuer_token is not _TRANSFORMED_PACKET_TOKEN:
        raise TypeError("transformed staged-alpha packets are internal-factory issued only")
    if type(raw_packet) is not V1StagedAlphaPacketEnvelope:
        raise TypeError("transformed staged-alpha reissue requires an exact raw packet")
    raw_packet.__post_init__()
    receipt = _nonempty(transformation_receipt_hash, "transformation_receipt_hash")
    semantics = _nonempty(evidence_semantics_id, "evidence_semantics_id")
    values = np.asarray(transformed_weighted_rows, dtype=np.float64).reshape(-1)
    if values.shape != (len(raw_packet.draw_ids),) or not np.isfinite(values).all():
        raise ValueError("transformed staged-alpha rows must be finite and draw-aligned")
    weighted = np.ascontiguousarray(values, dtype="<f8")
    provider_hash = content_hash(
        {
            "schema": "partition_free_v1_transformed_packet_provider_binding_v1",
            "raw_packet_envelope_hash": raw_packet.envelope_hash,
            "raw_provider_contract_hash": raw_packet.provider_contract_hash,
            "transformation_receipt_hash": receipt,
            "evidence_semantics_id": semantics,
            "draw_ids": raw_packet.draw_ids,
            "source_context_ids": raw_packet.source_context_ids,
            "request_id": raw_packet.request_id,
            "outcome_integrity_hash": raw_packet.outcome_integrity_hash,
        },
        prefix="partition-free-v1-transformed-packet-provider-",
    )
    payload = {
        "step_freeze_hash": raw_packet.step_freeze_hash,
        "evidence_packet_hash": raw_packet.evidence_packet_hash,
        "request_id": raw_packet.request_id,
        "outcome_integrity_hash": raw_packet.outcome_integrity_hash,
        "look_index": raw_packet.look_index,
        "draw_ids": raw_packet.draw_ids,
        "source_context_ids": raw_packet.source_context_ids,
        "weighted_damage_payload": weighted.tobytes(order="C"),
        "target_probability_hash": raw_packet.target_probability_hash,
        "sampling_probability_hash": raw_packet.sampling_probability_hash,
        "importance_weight_hash": raw_packet.importance_weight_hash,
        "sampling_plan_hash": raw_packet.sampling_plan_hash,
        "provider_contract_hash": provider_hash,
    }
    receipt_hash = content_hash(
        {"schema": STAGED_ALPHA_PACKET_SCHEMA, **payload},
        prefix="partition-free-v1-staged-alpha-packet-envelope-",
    )
    return V1StagedAlphaPacketEnvelope(
        _issuer_token=_PACKET_TOKEN,
        envelope_receipt_hash=receipt_hash,
        **payload,
    )


@dataclass(frozen=True, slots=True)
class V1StagedAlphaLedger:
    """Immutable, consecutive-look ledger for one candidate step."""

    step_freeze: V1StagedAlphaStepFreeze
    packets: tuple[V1StagedAlphaPacketEnvelope, ...] = ()
    schema: str = STAGED_ALPHA_LEDGER_SCHEMA

    def __post_init__(self) -> None:
        if type(self.step_freeze) is not V1StagedAlphaStepFreeze:
            raise TypeError("staged-alpha ledger requires an exact step freeze")
        packets = tuple(self.packets)
        seen_draws: set[str] = set()
        cumulative = 0
        for index, packet in enumerate(packets):
            if type(packet) is not V1StagedAlphaPacketEnvelope:
                raise TypeError("staged-alpha ledger packets must be factory envelopes")
            packet.__post_init__()
            if packet.step_freeze_hash != self.step_freeze.step_freeze_hash or packet.look_index != index:
                raise ValueError("staged-alpha packets are stale, reordered, or nonconsecutive")
            overlap = seen_draws.intersection(packet.draw_ids)
            if overlap:
                raise ValueError("staged-alpha draw IDs were reused across looks")
            seen_draws.update(packet.draw_ids)
            cumulative += len(packet.draw_ids)
            if cumulative != self.step_freeze.cumulative_draw_counts[index]:
                raise ValueError("staged-alpha packet count differs from the frozen look schedule")
        if len(packets) > len(self.step_freeze.cumulative_draw_counts):
            raise ValueError("staged-alpha ledger exceeds its frozen look schedule")
        object.__setattr__(self, "packets", packets)
        if self.schema != STAGED_ALPHA_LEDGER_SCHEMA:
            raise ValueError("unsupported staged-alpha ledger schema")

    @property
    def ledger_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-staged-alpha-ledger-")

    @property
    def weighted_damage(self) -> np.ndarray:
        if not self.packets:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(tuple(packet.weighted_damage for packet in self.packets))


def advance_v1_staged_alpha_ledger(
    ledger: V1StagedAlphaLedger,
    packet: V1StagedAlphaPacketEnvelope,
) -> V1StagedAlphaLedger:
    return V1StagedAlphaLedger(
        step_freeze=ledger.step_freeze,
        packets=ledger.packets + (packet,),
    )


@dataclass(frozen=True, slots=True)
class V1StagedAlphaDecision:
    """Recomputable look decision, still fenced from execution authority."""

    ledger_hash: str
    coordinates_hash: str
    campaign_contract_hash: str
    step_index: int
    look_index: int
    count: int
    look_alpha: float
    sample_mean_damage: float
    lower_mean_damage: float
    upper_mean_damage: float
    maximum_mean_damage: float
    action: Literal[
        "accept_candidate",
        "reject_candidate",
        "continue",
        "refuse_exhausted",
    ]
    reason: str
    decision_receipt_hash: str
    proof_artifact_hash: str | None
    owner_signature_verified: bool = False
    claim_execution_certified: bool = False
    schema: str = STAGED_ALPHA_DECISION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "ledger_hash",
            "coordinates_hash",
            "campaign_contract_hash",
            "reason",
            "decision_receipt_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        for name in (
            "look_alpha",
            "sample_mean_damage",
            "lower_mean_damage",
            "upper_mean_damage",
            "maximum_mean_damage",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        for name in ("step_index", "look_index"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or int(value) < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer")
            object.__setattr__(self, name, int(value))
        object.__setattr__(self, "count", _positive_int(self.count, "count"))
        if not 0.0 < self.look_alpha < 1.0 or self.lower_mean_damage > self.upper_mean_damage:
            raise ValueError("staged-alpha decision bounds are invalid")
        if self.action not in {
            "accept_candidate",
            "reject_candidate",
            "continue",
            "refuse_exhausted",
        }:
            raise ValueError("unsupported staged-alpha decision action")
        if self.proof_artifact_hash is not None:
            object.__setattr__(self, "proof_artifact_hash", _nonempty(self.proof_artifact_hash, "proof_artifact_hash"))
        if (
            self.owner_signature_verified is not False
            or self.claim_execution_certified is not False
            or self.schema != STAGED_ALPHA_DECISION_SCHEMA
        ):
            raise ValueError("staged-alpha decision crossed its current authority firewall")
        expected = content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "decision_receipt_hash"
            },
            prefix="partition-free-v1-staged-alpha-decision-",
        )
        if self.decision_receipt_hash != expected:
            raise ValueError("staged-alpha decision is not reproducible")

    @property
    def can_certify_decision(self) -> bool:
        return False

    @property
    def can_commit_mask(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class V1StagedAlphaTrajectoryLedger:
    """Recomputable accepted-step history with no cross-step confidence claim."""

    step_ledgers: tuple[V1StagedAlphaLedger, ...] = ()
    step_decisions: tuple[V1StagedAlphaDecision, ...] = ()
    schema: str = STAGED_ALPHA_TRAJECTORY_SCHEMA

    def __post_init__(self) -> None:
        ledgers = tuple(self.step_ledgers)
        decisions = tuple(self.step_decisions)
        if len(ledgers) != len(decisions):
            raise ValueError("trajectory ledgers and decisions must align one-to-one")
        seen_draws: set[str] = set()
        registration_hash: str | None = None
        coordinates_hash: str | None = None
        campaign_contract_hash: str | None = None
        previous_candidate: str | None = None
        for expected_step, (ledger, decision) in enumerate(zip(ledgers, decisions)):
            if type(ledger) is not V1StagedAlphaLedger:
                raise TypeError("trajectory entries require exact staged-alpha ledgers")
            if type(decision) is not V1StagedAlphaDecision:
                raise TypeError("trajectory entries require exact staged-alpha decisions")
            ledger.__post_init__()
            decision.__post_init__()
            freeze = ledger.step_freeze
            if (
                freeze.step_index != expected_step
                or decision.step_index != expected_step
                or decision.ledger_hash != ledger.ledger_hash
                or decision.coordinates_hash != freeze.coordinates_hash
                or decision.campaign_contract_hash != freeze.campaign_contract_hash
                or decision.action != "accept_candidate"
            ):
                raise ValueError(
                    "trajectory entries are nonconsecutive, unresolved, or misbound"
                )
            if registration_hash is None:
                registration_hash = freeze.registration_hash
                coordinates_hash = freeze.coordinates_hash
                campaign_contract_hash = freeze.campaign_contract_hash
            elif (
                freeze.registration_hash != registration_hash
                or freeze.coordinates_hash != coordinates_hash
                or freeze.campaign_contract_hash != campaign_contract_hash
            ):
                raise ValueError(
                    "trajectory steps name different registration or campaign coordinates"
                )
            proposal_parent = freeze.certificate_freeze.metadata.get(
                "proposal_parent_mask_hash",
                freeze.certificate_freeze.parent_mask_hash,
            )
            if (
                previous_candidate is not None
                and proposal_parent != previous_candidate
            ):
                raise ValueError("trajectory proposal ancestry is not parent-to-child exact")
            previous_candidate = freeze.certificate_freeze.candidate_mask_hash
            step_draws = {
                draw_id for packet in ledger.packets for draw_id in packet.draw_ids
            }
            if seen_draws.intersection(step_draws):
                raise ValueError("trajectory certificate draw IDs were reused across steps")
            seen_draws.update(step_draws)
        object.__setattr__(self, "step_ledgers", ledgers)
        object.__setattr__(self, "step_decisions", decisions)
        if self.schema != STAGED_ALPHA_TRAJECTORY_SCHEMA:
            raise ValueError("unsupported staged-alpha trajectory-ledger schema")

    @property
    def trajectory_ledger_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-staged-alpha-trajectory-")

    @property
    def diagnostic_sum_of_step_upper_endpoints(self) -> float:
        """Descriptive only; this is not a trajectory confidence bound."""

        return math.fsum(decision.upper_mean_damage for decision in self.step_decisions)

    @property
    def can_certify_trajectory(self) -> bool:
        return False

    @property
    def can_commit_mask(self) -> bool:
        return False


def advance_v1_staged_alpha_trajectory(
    trajectory: V1StagedAlphaTrajectoryLedger,
    *,
    ledger: V1StagedAlphaLedger,
    decision: V1StagedAlphaDecision,
) -> V1StagedAlphaTrajectoryLedger:
    """Append one resolved candidate without creating acceptance authority."""

    if type(trajectory) is not V1StagedAlphaTrajectoryLedger:
        raise TypeError("trajectory advance requires an exact staged-alpha trajectory")
    return V1StagedAlphaTrajectoryLedger(
        step_ledgers=trajectory.step_ledgers + (ledger,),
        step_decisions=trajectory.step_decisions + (decision,),
    )


def evaluate_v1_staged_alpha_look(
    *,
    coordinates: V1StagedAlphaCoordinates,
    ledger: V1StagedAlphaLedger,
) -> V1StagedAlphaDecision:
    """Compute the registered v0-style per-step Student-t decision."""

    if type(coordinates) is not V1StagedAlphaCoordinates or type(ledger) is not V1StagedAlphaLedger:
        raise TypeError("staged-alpha evaluation requires exact coordinates and ledger")
    if ledger.step_freeze.coordinates_hash != coordinates.coordinates_hash or not ledger.packets:
        raise ValueError("staged-alpha evaluation has no aligned observed look")
    ledger.__post_init__()
    values = ledger.weighted_damage
    count = int(values.size)
    look_index = len(ledger.packets) - 1
    alpha = coordinates.look_alpha(
        look_index,
        candidate_attempt_index=ledger.step_freeze.candidate_attempt_index,
    )
    mean = float(np.mean(values, dtype=np.float64))
    if count < 2:
        raise ValueError("per-step Student-t decision requires at least two rows")
    interval = student_t_interval(
        mean,
        float(np.var(values, ddof=1, dtype=np.float64)),
        count,
        alpha=alpha,
    )
    lower = max(0.0, float(interval.lower[()]))
    upper = float(interval.upper[()])
    threshold = ledger.step_freeze.maximum_mean_damage
    if upper <= threshold:
        action: Literal[
            "accept_candidate",
            "reject_candidate",
            "continue",
            "refuse_exhausted",
        ] = "accept_candidate"
        reason = "per-step Student-t upper endpoint lies within the frozen increment target"
    elif lower > threshold:
        action = "reject_candidate"
        reason = "per-step Student-t lower endpoint exceeds the frozen increment target"
    elif len(ledger.packets) == len(ledger.step_freeze.cumulative_draw_counts):
        action = "refuse_exhausted"
        reason = "registered look schedule exhausted without a resolved step decision"
    else:
        action = "continue"
        reason = "increment decision unresolved at this registered look"
    payload = {
        "schema": STAGED_ALPHA_DECISION_SCHEMA,
        "ledger_hash": ledger.ledger_hash,
        "coordinates_hash": coordinates.coordinates_hash,
        "campaign_contract_hash": coordinates.campaign_contract_hash,
        "step_index": ledger.step_freeze.step_index,
        "look_index": look_index,
        "count": count,
        "look_alpha": alpha,
        "sample_mean_damage": mean,
        "lower_mean_damage": lower,
        "upper_mean_damage": upper,
        "maximum_mean_damage": threshold,
        "action": action,
        "reason": reason,
        "proof_artifact_hash": coordinates.proof_artifact_hash,
        "owner_signature_verified": False,
        "claim_execution_certified": False,
    }
    return V1StagedAlphaDecision(
        decision_receipt_hash=content_hash(
            payload,
            prefix="partition-free-v1-staged-alpha-decision-",
        ),
        **{key: value for key, value in payload.items() if key != "schema"},
    )


__all__ = [
    "STAGED_ALPHA_BOUND_ID",
    "STAGED_ALPHA_EXHAUSTION_ACTION_ID",
    "STAGED_ALPHA_LOOK_SPENDING_ID",
    "STAGED_ALPHA_PROOF_ID",
    "STAGED_ALPHA_SAMPLING_ID",
    "STAGED_ALPHA_SAMPLING_MASS_TOTAL",
    "STAGED_ALPHA_STEP_SPENDING_ID",
    "V1StagedAlphaCoordinates",
    "V1StagedAlphaEntropyTape",
    "V1StagedAlphaDecision",
    "V1StagedAlphaLedger",
    "V1StagedAlphaPacketEnvelope",
    "V1StagedAlphaSamplingPlan",
    "V1StagedAlphaStepFreeze",
    "V1StagedAlphaTrajectoryLedger",
    "advance_v1_staged_alpha_ledger",
    "advance_v1_staged_alpha_trajectory",
    "bind_v1_staged_alpha_step",
    "canonicalize_v1_staged_alpha_sampling_probabilities",
    "draw_v1_staged_alpha_sampling_plan_from_tape",
    "evaluate_v1_staged_alpha_look",
    "issue_v1_staged_alpha_entropy_tape",
    "issue_v1_staged_alpha_packet",
    "quantize_v1_staged_alpha_sampling_masses",
]
