"""Workflow-neutral sealed finite-population response-design contracts."""

from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import dataclass, field
from typing import Sequence
import weakref

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.measurement_costs import CostEstimate


SEALED_UNIFORM_PERMUTATION_ALGORITHM_V1 = "sha256_counter_fisher_yates_rejection_v1"
_SEALED_PERMUTATION_RECEIPT_HASH_CACHE: dict[int, str] = {}


def _canonical_population(values: Sequence[str]) -> tuple[str, ...]:
    population = tuple(str(value) for value in values)
    if not population or len(set(population)) != len(population):
        raise ValueError(
            "sealed permutation requires a nonempty population of unique context IDs"
        )
    if population != tuple(sorted(population)):
        raise ValueError(
            "sealed permutation base population must use canonical ascending "
            "context-ID order"
        )
    return population


def _entropy_hex(value: str) -> str:
    entropy = str(value)
    if len(entropy) != 64 or any(
        character not in "0123456789abcdef" for character in entropy
    ):
        raise ValueError(
            "sealed permutation entropy must be a lowercase 256-bit hex value"
        )
    return entropy


def _uniform_index(
    *,
    population_hash: str,
    entropy_hex: str,
    step: int,
    span: int,
) -> int:
    if span <= 0:
        raise ValueError("sealed permutation draw span must be positive")
    range_size = 1 << 256
    acceptance_limit = range_size - (range_size % span)
    counter = 0
    while True:
        block = hashlib.sha256(
            b"\x00".join(
                (
                    SEALED_UNIFORM_PERMUTATION_ALGORITHM_V1.encode("ascii"),
                    population_hash.encode("ascii"),
                    bytes.fromhex(entropy_hex),
                    str(int(step)).encode("ascii"),
                    str(counter).encode("ascii"),
                )
            )
        ).digest()
        value = int.from_bytes(block, byteorder="big", signed=False)
        if value < acceptance_limit:
            return value % span
        counter += 1


def _generate_sealed_uniform_permutation(
    population: tuple[str, ...],
    *,
    entropy_hex: str,
) -> tuple[str, ...]:
    result = list(population)
    population_hash = content_hash(
        {
            "algorithm_id": SEALED_UNIFORM_PERMUTATION_ALGORITHM_V1,
            "canonical_population": population,
        },
        prefix="irg-sealed-permutation-population-",
    )
    for step in range(len(result) - 1, 0, -1):
        swap_index = _uniform_index(
            population_hash=population_hash,
            entropy_hex=entropy_hex,
            step=step,
            span=step + 1,
        )
        result[step], result[swap_index] = result[swap_index], result[step]
    return tuple(result)


def _sealed_uniform_permutation_receipt_hash(
    *,
    canonical_population: tuple[str, ...],
    entropy_hex: str,
    generated_order: tuple[str, ...],
    algorithm_id: str,
    schema: str,
) -> str:
    return content_hash(
        {
            "algorithm_id": algorithm_id,
            "canonical_population": canonical_population,
            "entropy_hex": entropy_hex,
            "generated_order": generated_order,
            "schema": schema,
        },
        prefix="irg-sealed-uniform-permutation-",
    )


@dataclass(frozen=True, slots=True, weakref_slot=True)
class SealedUniformPermutationReceipt:
    canonical_population: tuple[str, ...]
    entropy_hex: str
    generated_order: tuple[str, ...]
    algorithm_id: str = SEALED_UNIFORM_PERMUTATION_ALGORITHM_V1
    schema: str = "irg_sealed_uniform_permutation_receipt_v1"

    def __post_init__(self) -> None:
        population = _canonical_population(self.canonical_population)
        entropy = _entropy_hex(self.entropy_hex)
        order = tuple(str(value) for value in self.generated_order)
        if (
            len(order) != len(population)
            or len(set(order)) != len(order)
            or set(order) != set(population)
        ):
            raise ValueError(
                "sealed permutation generated order must be an exact "
                "population permutation"
            )
        if self.algorithm_id != SEALED_UNIFORM_PERMUTATION_ALGORITHM_V1:
            raise ValueError("unsupported sealed permutation algorithm identity")
        if self.schema != "irg_sealed_uniform_permutation_receipt_v1":
            raise ValueError("unsupported sealed permutation receipt schema")
        object.__setattr__(self, "canonical_population", population)
        object.__setattr__(self, "entropy_hex", entropy)
        object.__setattr__(self, "generated_order", order)
        cache_key = id(self)
        _SEALED_PERMUTATION_RECEIPT_HASH_CACHE[cache_key] = (
            _sealed_uniform_permutation_receipt_hash(
                canonical_population=population,
                entropy_hex=entropy,
                generated_order=order,
                algorithm_id=self.algorithm_id,
                schema=self.schema,
            )
        )
        weakref.finalize(
            self,
            _SEALED_PERMUTATION_RECEIPT_HASH_CACHE.pop,
            cache_key,
            None,
        )

    def __setstate__(self, state: object) -> None:
        # Preserve the dataclass-generated five-field __getstate__ byte law.
        # Restoring establishes construction custody, not permutation validity.
        names = (
            "canonical_population",
            "entropy_hex",
            "generated_order",
            "algorithm_id",
            "schema",
        )
        if type(state) not in (tuple, list) or len(state) != len(names):
            raise ValueError("unsupported sealed permutation pickle state")
        if any(hasattr(self, name) for name in names):
            raise ValueError(
                "cannot restore an already initialized permutation owner"
            )
        population, entropy, order, algorithm, schema = state
        if (
            type(population) is not tuple
            or type(order) is not tuple
            or any(type(value) is not str for value in population)
            or any(type(value) is not str for value in order)
            or any(type(value) is not str for value in (entropy, algorithm, schema))
        ):
            raise TypeError(
                "sealed permutation pickle state requires exact field types"
            )
        for name, value in zip(names, state, strict=True):
            object.__setattr__(self, name, value)
        self.__post_init__()

    @classmethod
    def create(
        cls,
        population: Sequence[str],
        *,
        entropy_hex: str,
    ) -> "SealedUniformPermutationReceipt":
        canonical = _canonical_population(tuple(sorted(map(str, population))))
        entropy = _entropy_hex(entropy_hex)
        return cls(
            canonical_population=canonical,
            entropy_hex=entropy,
            generated_order=_generate_sealed_uniform_permutation(
                canonical,
                entropy_hex=entropy,
            ),
        )

    def regenerate(self) -> tuple[str, ...]:
        return _generate_sealed_uniform_permutation(
            self.canonical_population,
            entropy_hex=self.entropy_hex,
        )

    def verify(self) -> None:
        if self.generated_order != self.regenerate():
            raise ValueError(
                "sealed permutation generated order does not match its frozen entropy"
            )

    @property
    def content_hash(self) -> str:
        cached = _SEALED_PERMUTATION_RECEIPT_HASH_CACHE.get(id(self))
        if cached is not None:
            return cached
        return _sealed_uniform_permutation_receipt_hash(
            canonical_population=self.canonical_population,
            entropy_hex=self.entropy_hex,
            generated_order=self.generated_order,
            algorithm_id=self.algorithm_id,
            schema=self.schema,
        )


@dataclass(frozen=True, slots=True)
class _VerifiedSealedPermutationCustody:
    owner_ref: weakref.ReferenceType[SealedUniformPermutationReceipt]
    receipt_class: type
    canonical_population: tuple[str, ...] = field(repr=False)
    generated_order: tuple[str, ...] = field(repr=False)
    entropy_hex: str
    algorithm_id: str
    schema: str
    verified_digest: str

    def assert_unchanged(self, receipt: SealedUniformPermutationReceipt) -> None:
        # Identity checks are essential: tuple equality and dataclass hashing
        # would walk the full population on every native source integrity call.
        if (
            self.owner_ref() is not receipt
            or type(receipt) is not self.receipt_class
            or receipt.canonical_population is not self.canonical_population
            or receipt.generated_order is not self.generated_order
            or type(receipt.entropy_hex) is not str
            or receipt.entropy_hex != self.entropy_hex
            or type(receipt.algorithm_id) is not str
            or receipt.algorithm_id != self.algorithm_id
            or type(receipt.schema) is not str
            or receipt.schema != self.schema
            or _SEALED_PERMUTATION_RECEIPT_HASH_CACHE.get(id(receipt))
            != self.verified_digest
        ):
            raise ValueError("verified sealed permutation immutable custody changed")


_SEALED_PERMUTATION_VERIFIED_CUSTODY: dict[int, _VerifiedSealedPermutationCustody] = {}
_SEALED_PERMUTATION_VERIFIED_CUSTODY_LOCK = threading.RLock()


def _drop_verified_permutation_custody(
    key: int,
    owner_ref: weakref.ReferenceType[SealedUniformPermutationReceipt],
) -> None:
    with _SEALED_PERMUTATION_VERIFIED_CUSTODY_LOCK:
        entry = _SEALED_PERMUTATION_VERIFIED_CUSTODY.get(key)
        if entry is not None and entry.owner_ref is owner_ref:
            del _SEALED_PERMUTATION_VERIFIED_CUSTODY[key]


def verify_sealed_uniform_permutation_custody(
    receipt: SealedUniformPermutationReceipt,
) -> str:
    """Verify a causal role-stream permutation, then guard its immutable custody.

    Only causal role streams use this memo. Standalone ``receipt.verify()``
    always regenerates. Each new live object must pass full deterministic
    verification plus an independently recomputed hash matching its construction
    seal. A hit checks exact owner/tuple identity and all scalar coordinates;
    altered owners are refused, never resealed. No population tuple is hashed
    or compared on hits. The lock serializes cold admissions and cache mutation;
    it does not authorize concurrent out-of-contract object.__setattr__ writes.
    """
    if type(receipt) is not SealedUniformPermutationReceipt:
        raise TypeError("permutation custody requires the exact receipt class")
    key = id(receipt)
    with _SEALED_PERMUTATION_VERIFIED_CUSTODY_LOCK:
        entry = _SEALED_PERMUTATION_VERIFIED_CUSTODY.get(key)
        if entry is not None:
            if entry.owner_ref() is receipt:
                entry.assert_unchanged(receipt)
                return entry.verified_digest
            # A stale weakref must never authorize a different object at this ID.
            del _SEALED_PERMUTATION_VERIFIED_CUSTODY[key]
        try:
            population = receipt.canonical_population
            order = receipt.generated_order
            entropy = receipt.entropy_hex
            algorithm = receipt.algorithm_id
            schema = receipt.schema
        except AttributeError as error:
            raise TypeError("permutation custody requires initialized fields") from error
        if (
            type(population) is not tuple
            or type(order) is not tuple
            or any(type(value) is not str for value in population)
            or any(type(value) is not str for value in order)
            or any(type(value) is not str for value in (entropy, algorithm, schema))
        ):
            raise TypeError("permutation custody requires exact immutable field types")
        if (
            algorithm != SEALED_UNIFORM_PERMUTATION_ALGORITHM_V1
            or schema != "irg_sealed_uniform_permutation_receipt_v1"
        ):
            raise ValueError("permutation custody has unsupported algorithm or schema")
        _canonical_population(population)
        _entropy_hex(entropy)
        sealed_digest = _SEALED_PERMUTATION_RECEIPT_HASH_CACHE.get(key)
        if type(sealed_digest) is not str:
            raise ValueError("permutation custody lacks its construction seal")
        owner_ref = weakref.ref(
            receipt,
            lambda reference: _drop_verified_permutation_custody(key, reference),
        )
        candidate = _VerifiedSealedPermutationCustody(
            owner_ref=owner_ref,
            receipt_class=SealedUniformPermutationReceipt,
            canonical_population=population,
            generated_order=order,
            entropy_hex=entropy,
            algorithm_id=algorithm,
            schema=schema,
            verified_digest=sealed_digest,
        )
        candidate.assert_unchanged(receipt)
        receipt.verify()
        observed = _sealed_uniform_permutation_receipt_hash(
            canonical_population=population,
            entropy_hex=entropy,
            generated_order=order,
            algorithm_id=algorithm,
            schema=schema,
        )
        candidate.assert_unchanged(receipt)
        if observed != sealed_digest:
            raise ValueError("sealed permutation differs from its construction hash")
        _SEALED_PERMUTATION_VERIFIED_CUSTODY[key] = candidate
        return observed


@dataclass(frozen=True, slots=True)
class ResponseContextStratum:
    stratum_id: str
    context_ids: tuple[str, ...]
    weight: float
    packet_target: int
    maximum_target: int
    expected_cost: CostEstimate = field(
        default_factory=lambda: CostEstimate(wall_seconds=1.0)
    )
    permutation_receipt: SealedUniformPermutationReceipt | None = None

    def __post_init__(self) -> None:
        contexts = _canonical_population(self.context_ids)
        if not self.stratum_id:
            raise ValueError("response context stratum requires a stable ID")
        if not math.isfinite(float(self.weight)) or float(self.weight) <= 0.0:
            raise ValueError(
                "response context stratum weight must be finite and positive"
            )
        if int(self.packet_target) <= 0 or not 0 < int(self.maximum_target) <= len(
            contexts
        ):
            raise ValueError(
                "response context stratum packet/maximum targets are invalid"
            )
        receipt = self.permutation_receipt
        if not isinstance(receipt, SealedUniformPermutationReceipt):
            raise ValueError(
                "response context stratum requires a typed sealed permutation receipt"
            )
        if receipt.canonical_population != contexts:
            raise ValueError(
                "response context stratum and sealed permutation populations differ"
            )
        receipt.verify()
        object.__setattr__(self, "context_ids", contexts)
        object.__setattr__(self, "weight", float(self.weight))
        object.__setattr__(self, "packet_target", int(self.packet_target))
        object.__setattr__(self, "maximum_target", int(self.maximum_target))

    @property
    def sampling_order(self) -> tuple[str, ...]:
        receipt = self.permutation_receipt
        if not isinstance(receipt, SealedUniformPermutationReceipt):
            raise RuntimeError("response stratum lost its sealed permutation receipt")
        return receipt.generated_order


@dataclass(frozen=True, slots=True)
class FrozenResponseContextDesign:
    design_id: str
    strata: tuple[ResponseContextStratum, ...]
    confidence_level: float = 0.95
    stable_looks_required: int = 2
    unresolved_action: str = "full_census"
    schema: str = "irg_frozen_response_context_design_v2"

    def __post_init__(self) -> None:
        strata = tuple(self.strata)
        if not self.design_id or not self.schema:
            raise ValueError("response context design requires stable identity")
        if not strata or len({item.stratum_id for item in strata}) != len(strata):
            raise ValueError("response context design requires unique strata")
        contexts = tuple(value for item in strata for value in item.context_ids)
        if len(set(contexts)) != len(contexts):
            raise ValueError("response context IDs must be unique across strata")
        if int(self.stable_looks_required) <= 0:
            raise ValueError("response context design stable looks must be positive")
        confidence = float(self.confidence_level)
        if not 0.5 < confidence < 1.0:
            raise ValueError(
                "response context design confidence_level must lie in (0.5,1)"
            )
        if self.unresolved_action not in {"full_census", "block"}:
            raise ValueError(
                "response context design unresolved action must be full_census or block"
            )
        object.__setattr__(self, "strata", strata)
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(
            self,
            "stable_looks_required",
            int(self.stable_looks_required),
        )

    @property
    def content_hash(self) -> str:
        return content_hash(self, prefix="irg-response-context-design-")

    @property
    def randomization_commitment(self) -> str:
        return content_hash(
            tuple(
                (item.stratum_id, item.permutation_receipt.content_hash)
                for item in self.strata
                if item.permutation_receipt is not None
            ),
            prefix="irg-response-randomization-",
        )

    def stratum(self, stratum_id: str) -> ResponseContextStratum:
        for item in self.strata:
            if item.stratum_id == str(stratum_id):
                return item
        raise KeyError(f"unknown response context stratum {stratum_id!r}")


__all__ = [
    "FrozenResponseContextDesign",
    "ResponseContextStratum",
    "SEALED_UNIFORM_PERMUTATION_ALGORITHM_V1",
    "SealedUniformPermutationReceipt",
]
