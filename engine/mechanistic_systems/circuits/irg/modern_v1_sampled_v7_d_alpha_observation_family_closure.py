"""Atomically close one sampled-V7 corrected-A/signed-F/literal-J family.

The three observation bindings are individually identity-only.  This module
performs their first durable state transition: exactly the corrected-A,
signed-F, and literal-J cells of one preregistered D look are consumed once as
one nonrecycling family.  The transaction never advances the alpha ledger and
never spends alpha again.

Literal-J population promotion deliberately burns the resident D1 handle.
Consequently the older corrected-A and signed-F receipts cannot use their
ordinary live replay after J exists.  Their sealed payloads and immediate
immutable observation parents are replayed statically, while the literal-J
receipt supplies the transition-aware live replay of the common plan,
registry, durable CAS, measurement owner, chart, and source lineage.

Durable state lives in a separate content-addressed namespace.  One locked,
fsynced record exists per proposal generation.  Byte-identical replay is
idempotent; a different valid observation family for the same generation is a
fork and fails closed.  Symlinks and malformed state are never followed or
repaired heuristically.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
import fcntl
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_sampled_v7_corrected_a_observation_binding import (
    V1SampledV7CorrectedAObservationBindingReceipt,
)
from .modern_v1_sampled_v7_d_alpha_measurement_owner_binding import (
    V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
)
from .modern_v1_sampled_v7_literal_j_observation_binding import (
    V1SampledV7LiteralJObservationBindingReceipt,
)
from .modern_v1_sampled_v7_literal_j_census_observation_binding import (
    V1SampledV7LiteralJCensusObservationBindingReceipt,
)
from .modern_v1_sampled_v7_signed_f_observation_binding import (
    V1SampledV7SignedFObservationBindingReceipt,
)


SAMPLED_V7_D_ALPHA_OBSERVATION_FAMILY_CLOSURE_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_D_alpha_observation_family_closure_v2"
)
SAMPLED_V7_D_ALPHA_OBSERVATION_FAMILY_RECORD_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_D_alpha_observation_family_record_v2"
)
SAMPLED_V7_D_ALPHA_OBSERVATION_FAMILY_CLOSURE_AUTHORITY: Final = (
    "durable_atomic_nonrecycling_consumption_of_one_preregistered_sampled_v7_"
    "corrected_A_signed_F_literal_J_observation_family_only_no_second_alpha_"
    "spend_D_C_K_O_action_commit_terminal_performance_or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-v7-D-alpha-observation-family-"
_A_RECEIPT_PREFIX: Final = (
    "partition-free-v1-sampled-v7-corrected-A-observation-binding-receipt-"
)
_F_RECEIPT_PREFIX: Final = (
    "partition-free-v1-sampled-v7-signed-F-observation-binding-receipt-"
)
_F_ALPHA_RECEIPT_PREFIX: Final = (
    "partition-free-v1-sampled-v7-signed-F-alpha-registration-receipt-"
)
_ROLE_AXIS: Final = (
    "corrected_A_order",
    "signed_F_witness",
    "literal_J_sampling",
)
LITERAL_J_STATISTICAL_REGIME_POSITIVE_ALPHA_PREFIX: Final = (
    "positive_alpha_sampled_prefix"
)
LITERAL_J_STATISTICAL_REGIME_ALPHA_FREE_CENSUS: Final = (
    "alpha_free_exact_census"
)
_LITERAL_J_STATISTICAL_REGIMES: Final = (
    LITERAL_J_STATISTICAL_REGIME_POSITIVE_ALPHA_PREFIX,
    LITERAL_J_STATISTICAL_REGIME_ALPHA_FREE_CENSUS,
)
_STATE_MAX_BYTES: Final = 1 << 20
_LOCK_NAME: Final = "FAMILY.lock"
_ISSUER_TOKEN: Final = object()


class V1SampledV7DAlphaObservationFamilyClosureRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    PARENT_INTEGRITY = "parent_integrity"
    COMMON_FAMILY_MISMATCH = "common_family_mismatch"
    ROLE_MISMATCH = "role_mismatch"
    LOOK_OR_PREFIX_MISMATCH = "look_or_prefix_mismatch"
    ALPHA_REUSE = "alpha_reuse"
    STORE_INVALID = "store_invalid"
    STATE_INVALID = "state_invalid"
    TRANSACTION_CONFLICT = "transaction_conflict"
    DURABILITY_FAILURE = "durability_failure"
    SOURCE_SUBSTITUTION = "source_substitution"
    AUTHORITY_VIOLATION = "authority_violation"
    MUTATED_RECEIPT = "mutated_receipt"


class V1SampledV7DAlphaObservationFamilyClosureRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledV7DAlphaObservationFamilyClosureRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7DAlphaObservationFamilyClosureRefusalCode:
            raise TypeError("observation-family closure refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7DAlphaObservationFamilyClosureRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7DAlphaObservationFamilyClosureRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.STATE_INVALID,
            f"{name} must be one nonempty identity",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or type(value) is not int or value < 0:
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.STATE_INVALID,
            f"{name} must be one nonnegative exact integer",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.STATE_INVALID,
            f"{name} must be positive",
        )
    return result


def _static_hash(
    *,
    payload: dict[str, object],
    cached_hash: str | None,
    prefix: str,
    role: str,
) -> str:
    observed = content_hash(payload, prefix=prefix)
    if cached_hash is None or cached_hash != observed:
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.PARENT_INTEGRITY,
            f"sealed {role} changed before family closure",
        )
    return observed


def _assert_static_a(
    receipt: V1SampledV7CorrectedAObservationBindingReceipt,
) -> tuple[str, V1SampledV7DAlphaMeasurementOwnerBindingReceipt]:
    if type(receipt) is not V1SampledV7CorrectedAObservationBindingReceipt:
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.INVALID_TYPE,
            "family closure requires exactly one corrected-A observation binding",
        )
    try:
        receipt_hash = _static_hash(
            payload=receipt._payload(),
            cached_hash=receipt._receipt_hash,
            prefix=_A_RECEIPT_PREFIX,
            role="corrected-A observation receipt",
        )
        box_hash = receipt._confidence_box.assert_integrity()
    except V1SampledV7DAlphaObservationFamilyClosureRefusal:
        raise
    except Exception as error:
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.PARENT_INTEGRITY,
            f"corrected-A observation failed static replay: {type(error).__name__}: {error}",
        )
    if (
        box_hash != receipt.confidence_box_receipt_hash
        or receipt._confidence_box.state_hash != receipt.confidence_box_state_hash
        or receipt._confidence_box.base_serfling_design_hash
        != receipt.base_serfling_design_hash
        or receipt._confidence_box.design_hash != receipt.hybrid_design_hash
        or receipt._confidence_box.look_index != receipt.registered_look_index
        or receipt._confidence_box.stratum_counts
        != receipt.registered_look_stratum_counts
        or receipt._confidence_box.population_stratum_counts
        != receipt.population_stratum_counts
        or receipt._confidence_box.look_alpha != receipt.corrected_a_cell_alpha
    ):
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.PARENT_INTEGRITY,
            "corrected-A receipt lost its exact confidence-box observation",
        )
    return receipt_hash, receipt._owner_binding


def _assert_static_f(
    receipt: V1SampledV7SignedFObservationBindingReceipt,
) -> tuple[str, V1SampledV7DAlphaMeasurementOwnerBindingReceipt]:
    if type(receipt) is not V1SampledV7SignedFObservationBindingReceipt:
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.INVALID_TYPE,
            "family closure requires exactly one signed-F observation binding",
        )
    alpha = receipt._alpha_registration
    projection = receipt._projection_receipt
    try:
        receipt_hash = _static_hash(
            payload=receipt._payload(),
            cached_hash=receipt._receipt_hash,
            prefix=_F_RECEIPT_PREFIX,
            role="signed-F observation receipt",
        )
        alpha_hash = _static_hash(
            payload=alpha._payload(),
            cached_hash=alpha._receipt_hash,
            prefix=_F_ALPHA_RECEIPT_PREFIX,
            role="signed-F alpha registration",
        )
        projection_hash = projection.assert_integrity()
        registration_hash = alpha._projection_registration.registration_hash
    except V1SampledV7DAlphaObservationFamilyClosureRefusal:
        raise
    except Exception as error:
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.PARENT_INTEGRITY,
            f"signed-F observation failed static replay: {type(error).__name__}: {error}",
        )
    if (
        receipt.signed_f_alpha_registration_receipt_hash != alpha_hash
        or receipt.projection_registration_hash != registration_hash
        or receipt.projection_receipt_hash != projection_hash
        or alpha.projection_registration_hash != registration_hash
        or projection.registration_hash != registration_hash
        or receipt.signed_f_cell_id != alpha.signed_f_cell_id
        or receipt.signed_f_cell_hash != alpha.signed_f_cell_hash
        or receipt.signed_f_cell_scope_hash != alpha.signed_f_cell_scope_hash
        or receipt.signed_f_cell_alpha != alpha.signed_f_cell_alpha
        or receipt.registered_look_index != alpha.registered_look_index
        or receipt.registered_look_stratum_counts
        != alpha.registered_look_stratum_counts
        or receipt.population_stratum_counts != alpha.population_stratum_counts
        or receipt.registered_prefix_axis_hash != alpha.registered_prefix_axis_hash
        or receipt.signed_f_stream_binding_hash
        != alpha.signed_f_stream_binding_hash
    ):
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.PARENT_INTEGRITY,
            "signed-F receipt lost its exact registration or projection observation",
        )
    return receipt_hash, alpha._owner_binding


def _assert_live_j(
    receipt: (
        V1SampledV7LiteralJObservationBindingReceipt
        | V1SampledV7LiteralJCensusObservationBindingReceipt
    ),
) -> tuple[str, V1SampledV7DAlphaMeasurementOwnerBindingReceipt, str]:
    if type(receipt) is V1SampledV7LiteralJObservationBindingReceipt:
        try:
            receipt_hash = receipt.assert_live()
        except Exception as error:
            _refuse(
                V1SampledV7DAlphaObservationFamilyClosureRefusalCode.PARENT_INTEGRITY,
                f"literal-J sampled-prefix replay failed: {type(error).__name__}: {error}",
            )
        return (
            receipt_hash,
            receipt._alpha_registration._owner_binding,
            LITERAL_J_STATISTICAL_REGIME_POSITIVE_ALPHA_PREFIX,
        )
    if type(receipt) is V1SampledV7LiteralJCensusObservationBindingReceipt:
        try:
            receipt_hash = receipt.assert_live()
        except Exception as error:
            _refuse(
                V1SampledV7DAlphaObservationFamilyClosureRefusalCode.PARENT_INTEGRITY,
                f"literal-J alpha-free census replay failed: {type(error).__name__}: {error}",
            )
        return (
            receipt_hash,
            receipt._census_registration._owner_binding,
            LITERAL_J_STATISTICAL_REGIME_ALPHA_FREE_CENSUS,
        )
    _refuse(
        V1SampledV7DAlphaObservationFamilyClosureRefusalCode.INVALID_TYPE,
        "family closure requires exactly one recognized literal-J observation binding",
    )


def _family_coordinates(
    *,
    corrected_a_observation: V1SampledV7CorrectedAObservationBindingReceipt,
    signed_f_observation: V1SampledV7SignedFObservationBindingReceipt,
    literal_j_observation: (
        V1SampledV7LiteralJObservationBindingReceipt
        | V1SampledV7LiteralJCensusObservationBindingReceipt
    ),
) -> dict[str, object]:
    a_hash, a_owner = _assert_static_a(corrected_a_observation)
    f_hash, f_owner = _assert_static_f(signed_f_observation)
    j_hash, j_owner, j_regime = _assert_live_j(literal_j_observation)
    if (
        type(a_owner) is not V1SampledV7DAlphaMeasurementOwnerBindingReceipt
        or type(f_owner) is not V1SampledV7DAlphaMeasurementOwnerBindingReceipt
        or type(j_owner) is not V1SampledV7DAlphaMeasurementOwnerBindingReceipt
    ):
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.INVALID_TYPE,
            "one observation lost its exact alpha-owner binding",
        )
    if a_owner is not j_owner or f_owner is not j_owner:
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.COMMON_FAMILY_MISMATCH,
            "A/F/J observations do not retain the same exact alpha-owner instance",
        )
    owner = j_owner

    if (
        corrected_a_observation.owner_binding_receipt_hash
        != literal_j_observation.owner_binding_receipt_hash
        or signed_f_observation.owner_binding_receipt_hash
        != literal_j_observation.owner_binding_receipt_hash
        or corrected_a_observation.alpha_family_plan_hash
        != owner.alpha_family_plan_hash
        or corrected_a_observation.alpha_family_registry_receipt_hash
        != owner.alpha_family_registry_receipt_hash
        or corrected_a_observation.alpha_head_cas_receipt_hash
        != owner.alpha_head_cas_receipt_hash
        or literal_j_observation.owner_binding_receipt_hash != owner._receipt_hash
    ):
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.COMMON_FAMILY_MISMATCH,
            "A/F/J observations do not share one plan, registry, CAS, and owner join",
        )

    cell_ids = (
        corrected_a_observation.corrected_a_cell_id,
        signed_f_observation.signed_f_cell_id,
        literal_j_observation.literal_j_cell_id,
    )
    cell_hashes = (
        corrected_a_observation.corrected_a_cell_hash,
        signed_f_observation.signed_f_cell_hash,
        literal_j_observation.literal_j_cell_hash,
    )
    cell_scopes = (
        corrected_a_observation.corrected_a_cell_scope_hash,
        signed_f_observation.signed_f_cell_scope_hash,
        literal_j_observation.literal_j_cell_scope_hash,
    )
    cell_alphas = (
        corrected_a_observation.corrected_a_cell_alpha,
        signed_f_observation.signed_f_cell_alpha,
        literal_j_observation.literal_j_cell_alpha,
    )
    if (
        cell_ids
        != (
            owner.corrected_a_cell_id,
            owner.signed_f_cell_id,
            owner.literal_j_cell_id,
        )
        or cell_hashes
        != (
            owner.corrected_a_cell_hash,
            owner.signed_f_cell_hash,
            owner.literal_j_cell_hash,
        )
        or cell_scopes
        != (
            owner.corrected_a_cell_scope_hash,
            owner.signed_f_cell_scope_hash,
            owner.literal_j_cell_scope_hash,
        )
        or cell_alphas
        != (
            owner.corrected_a_cell_alpha,
            owner.signed_f_cell_alpha,
            owner.literal_j_cell_alpha,
        )
        or len(set(cell_ids)) != 3
        or len(set(cell_hashes)) != 3
        or len(set(cell_scopes)) != 3
    ):
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.ROLE_MISMATCH,
            "A/F/J inputs do not contain exactly one distinct registered role cell",
        )

    look = literal_j_observation.registered_look_index
    counts = literal_j_observation.registered_look_stratum_counts
    population = literal_j_observation.population_stratum_counts
    prefix = literal_j_observation.registered_prefix_axis_hash
    if (
        corrected_a_observation.registered_look_index != look
        or signed_f_observation.registered_look_index != look
        or corrected_a_observation.registered_look_stratum_counts != counts
        or signed_f_observation.registered_look_stratum_counts != counts
        or corrected_a_observation.population_stratum_counts != population
        or signed_f_observation.population_stratum_counts != population
        or corrected_a_observation.registered_prefix_axis_hash != prefix
        or signed_f_observation.registered_prefix_axis_hash != prefix
        or owner.registered_look_index != look
        or owner.registered_look_stratum_counts != counts
        or owner.population_stratum_counts != population
        or owner.registered_prefix_axis_hash != prefix
    ):
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.LOOK_OR_PREFIX_MISMATCH,
            "A/F/J observations do not share one registered look, prefix, and population",
        )

    a_box = corrected_a_observation._confidence_box
    f_projection = signed_f_observation._projection_receipt
    if (
        a_box.proposal_generation != owner.proposal_generation
        or f_projection.proposal_generation != owner.proposal_generation
        or literal_j_observation.proposal_generation != owner.proposal_generation
        or a_box.route_axis_hash != owner.ordered_route_axis_hash
        or signed_f_observation.ordered_route_axis_hash
        != owner.ordered_route_axis_hash
        or signed_f_observation.physical_route_axis_hash
        != owner.physical_route_axis_hash
        or literal_j_observation.physical_route_axis_hash
        != owner.physical_route_axis_hash
        or a_box.route_count != owner.route_count
        or signed_f_observation.route_count != owner.route_count
        or literal_j_observation.route_count != owner.route_count
    ):
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.COMMON_FAMILY_MISMATCH,
            "A/F/J observations left their common generation or bridged route axes",
        )

    if (
        corrected_a_observation.source_observation_bound is not True
        or signed_f_observation.source_observation_bound is not True
        or literal_j_observation.source_observation_bound is not True
        or literal_j_observation.literal_j_cell_observation_bound is not True
        or corrected_a_observation.cell_durably_consumed is not False
        or signed_f_observation.cell_durably_consumed is not False
        or literal_j_observation.cell_durably_consumed is not False
        or corrected_a_observation.second_alpha_spend_authority is not False
        or signed_f_observation.second_alpha_spend_authority is not False
        or literal_j_observation.second_alpha_spend_authority is not False
        or literal_j_observation.second_alpha_spend != 0.0
    ):
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.ALPHA_REUSE,
            "one observation was already consumed or claims a second alpha spend",
        )

    if j_regime == LITERAL_J_STATISTICAL_REGIME_ALPHA_FREE_CENSUS:
        if (
            corrected_a_observation.exact_census is not True
            or signed_f_observation.exact_census is not True
            or counts != population
            or any(alpha != 0.0 for alpha in cell_alphas)
        ):
            _refuse(
                V1SampledV7DAlphaObservationFamilyClosureRefusalCode.ALPHA_REUSE,
                "alpha-free census J cannot close with sampled or positive-alpha A/F cells",
            )
    elif j_regime == LITERAL_J_STATISTICAL_REGIME_POSITIVE_ALPHA_PREFIX:
        if (
            corrected_a_observation.exact_census is not False
            or signed_f_observation.exact_census is not False
            or counts == population
            or any(alpha <= 0.0 for alpha in cell_alphas)
        ):
            _refuse(
                V1SampledV7DAlphaObservationFamilyClosureRefusalCode.ALPHA_REUSE,
                "positive-alpha sampled J cannot close with census or zero-alpha A/F cells",
            )
    else:
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.AUTHORITY_VIOLATION,
            "literal-J statistical regime is not registered",
        )

    return {
        "campaign_contract_hash": owner.campaign_contract_hash,
        "controller_run_hash": owner.controller_run_hash,
        "proposal_generation": owner.proposal_generation,
        "owner_binding_receipt_hash": owner._receipt_hash,
        "alpha_family_plan_hash": owner.alpha_family_plan_hash,
        "alpha_family_registry_receipt_hash": (
            owner.alpha_family_registry_receipt_hash
        ),
        "alpha_head_cas_receipt_hash": owner.alpha_head_cas_receipt_hash,
        "published_ledger_head_hash": owner.published_ledger_head_hash,
        "durable_store_identity_hash": (
            owner._alpha_head_cas.durable_store_identity_hash
        ),
        "registered_look_index": look,
        "registered_look_stratum_counts": counts,
        "population_stratum_counts": population,
        "registered_prefix_axis_hash": prefix,
        "cell_role_axis": _ROLE_AXIS,
        "cell_id_axis": cell_ids,
        "cell_hash_axis": cell_hashes,
        "cell_scope_hash_axis": cell_scopes,
        "cell_alpha_axis": cell_alphas,
        "observation_receipt_hash_axis": (a_hash, f_hash, j_hash),
        "literal_j_statistical_regime": j_regime,
    }


def _namespace_hash(
    *,
    campaign_contract_hash: str,
    controller_run_hash: str,
    durable_store_identity_hash: str,
) -> str:
    return content_hash(
        {
            "schema": (
                "partition_free_v1_sampled_v7_D_alpha_observation_family_namespace_v1"
            ),
            "campaign_contract_hash": campaign_contract_hash,
            "controller_run_hash": controller_run_hash,
            "durable_store_identity_hash": durable_store_identity_hash,
        },
        prefix=_PREFIX + "namespace-",
    )


def _assert_no_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        if os.path.lexists(current) and stat.S_ISLNK(os.lstat(current).st_mode):
            _refuse(
                V1SampledV7DAlphaObservationFamilyClosureRefusalCode.STORE_INVALID,
                "durable observation-family path contains a symlink",
            )


def _require_real_directory(path: Path, *, create: bool) -> None:
    _assert_no_symlink_components(path)
    if create:
        try:
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError as error:
            raise V1SampledV7DAlphaObservationFamilyClosureRefusal(
                V1SampledV7DAlphaObservationFamilyClosureRefusalCode.STORE_INVALID,
                f"durable observation-family directory cannot be created: {error}",
            ) from error
    if not os.path.lexists(path):
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.STORE_INVALID,
            "durable observation-family directory is absent",
        )
    mode = os.lstat(path).st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.STORE_INVALID,
            "durable observation-family path must be one real directory",
        )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _owned_directory(
    durable_root: str | os.PathLike[str],
    *,
    namespace_hash: str,
    create: bool,
) -> tuple[Path, Path]:
    root = Path(durable_root)
    if not root.is_absolute():
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.STORE_INVALID,
            "durable observation-family root must be absolute",
        )
    root_existed = os.path.lexists(root)
    _require_real_directory(root, create=create)
    namespace = root / namespace_hash
    namespace_existed = os.path.lexists(namespace)
    _require_real_directory(namespace, create=create)
    if create and not root_existed:
        _fsync_directory(root.parent)
    if create and not namespace_existed:
        _fsync_directory(root)
    return root, namespace


def _canonical_json_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _state_name(proposal_generation: int) -> str:
    return f"generation-{proposal_generation:08d}.json"


def _validate_string_axis(
    value: object,
    *,
    name: str,
    expected_length: int,
    unique: bool,
) -> list[str]:
    if type(value) is not list or len(value) != expected_length:
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.STATE_INVALID,
            f"{name} has the wrong axis shape",
        )
    result = [_identity(item, name=name) for item in value]
    if unique and len(set(result)) != len(result):
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.STATE_INVALID,
            f"{name} contains duplicate role coordinates",
        )
    return result


def _validate_record(record: object) -> dict[str, object]:
    expected_keys = {
        "schema",
        "campaign_contract_hash",
        "controller_run_hash",
        "proposal_generation",
        "owner_binding_receipt_hash",
        "alpha_family_plan_hash",
        "alpha_family_registry_receipt_hash",
        "alpha_head_cas_receipt_hash",
        "published_ledger_head_hash",
        "durable_store_identity_hash",
        "namespace_hash",
        "registered_look_index",
        "registered_look_stratum_counts",
        "population_stratum_counts",
        "registered_prefix_axis_hash",
        "cell_role_axis",
        "cell_id_axis",
        "cell_hash_axis",
        "cell_scope_hash_axis",
        "cell_alpha_axis",
        "observation_receipt_hash_axis",
        "literal_j_statistical_regime",
        "alpha_spent_again",
        "alpha_ledger_advanced",
        "three_cells_consumed_atomically",
    }
    if type(record) is not dict or set(record) != expected_keys:
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.STATE_INVALID,
            "durable observation-family record fields changed",
        )
    if record["schema"] != SAMPLED_V7_D_ALPHA_OBSERVATION_FAMILY_RECORD_SCHEMA:
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.STATE_INVALID,
            "durable observation-family record schema changed",
        )
    for name in (
        "campaign_contract_hash",
        "controller_run_hash",
        "owner_binding_receipt_hash",
        "alpha_family_plan_hash",
        "alpha_family_registry_receipt_hash",
        "alpha_head_cas_receipt_hash",
        "published_ledger_head_hash",
        "durable_store_identity_hash",
        "namespace_hash",
        "registered_prefix_axis_hash",
    ):
        _identity(record[name], name=f"record {name}")
    _nonnegative_int(record["proposal_generation"], name="record generation")
    _positive_int(record["registered_look_index"], name="record look index")
    for name in ("registered_look_stratum_counts", "population_stratum_counts"):
        values = record[name]
        if (
            type(values) is not list
            or len(values) != 2
            or any(isinstance(value, bool) or type(value) is not int or value <= 0 for value in values)
        ):
            _refuse(
                V1SampledV7DAlphaObservationFamilyClosureRefusalCode.STATE_INVALID,
                f"record {name} is malformed",
            )
    roles = _validate_string_axis(
        record["cell_role_axis"],
        name="record role axis",
        expected_length=3,
        unique=True,
    )
    if tuple(roles) != _ROLE_AXIS:
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.STATE_INVALID,
            "durable record role order changed",
        )
    for name in (
        "cell_id_axis",
        "cell_hash_axis",
        "cell_scope_hash_axis",
        "observation_receipt_hash_axis",
    ):
        _validate_string_axis(
            record[name],
            name=f"record {name}",
            expected_length=3,
            unique=True,
        )
    if record["literal_j_statistical_regime"] not in _LITERAL_J_STATISTICAL_REGIMES:
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.STATE_INVALID,
            "durable record literal-J statistical regime is unrecognized",
        )
    alphas = record["cell_alpha_axis"]
    if (
        type(alphas) is not list
        or len(alphas) != 3
        or any(
            isinstance(value, bool)
            or type(value) not in {int, float}
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) < 1.0
            for value in alphas
        )
    ):
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.STATE_INVALID,
            "durable record alpha axis is malformed",
        )
    if (
        record["alpha_spent_again"] != 0.0
        or record["alpha_ledger_advanced"] is not False
        or record["three_cells_consumed_atomically"] is not True
    ):
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.STATE_INVALID,
            "durable record crossed its nonrecycling alpha boundary",
        )
    return record


def _read_state(path: Path) -> dict[str, object] | None:
    if not os.path.lexists(path):
        return None
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.STATE_INVALID,
            "durable observation-family state must be one nonsymlink regular file",
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            raw = stream.read(_STATE_MAX_BYTES + 1)
    except OSError as error:
        raise V1SampledV7DAlphaObservationFamilyClosureRefusal(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.STATE_INVALID,
            f"durable observation-family state is unreadable: {error}",
        ) from error
    if len(raw) > _STATE_MAX_BYTES:
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.STATE_INVALID,
            "durable observation-family state exceeds its bounded schema",
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V1SampledV7DAlphaObservationFamilyClosureRefusal(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.STATE_INVALID,
            f"durable observation-family state is malformed: {error}",
        ) from error
    if type(payload) is not dict or set(payload) != {
        "record",
        "transaction_record_hash",
        "closure_receipt_hash",
    }:
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.STATE_INVALID,
            "durable observation-family state fields changed",
        )
    record = _validate_record(payload["record"])
    expected_record_hash = content_hash(record, prefix=_PREFIX + "record-")
    if payload["transaction_record_hash"] != expected_record_hash:
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.STATE_INVALID,
            "durable observation-family record content address does not replay",
        )
    _identity(payload["closure_receipt_hash"], name="closure receipt hash")
    return payload


def _open_lock(path: Path) -> int:
    if os.path.lexists(path):
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            _refuse(
                V1SampledV7DAlphaObservationFamilyClosureRefusalCode.STORE_INVALID,
                "observation-family lock must be one nonsymlink regular file",
            )
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise V1SampledV7DAlphaObservationFamilyClosureRefusal(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.STORE_INVALID,
            f"observation-family lock cannot be opened safely: {error}",
        ) from error
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.STORE_INVALID,
            "observation-family lock descriptor is not regular",
        )
    return descriptor


def _record_payload(
    coordinates: dict[str, object],
    *,
    durable_store_identity_hash: str,
    namespace_hash: str,
) -> dict[str, object]:
    return {
        "schema": SAMPLED_V7_D_ALPHA_OBSERVATION_FAMILY_RECORD_SCHEMA,
        "campaign_contract_hash": coordinates["campaign_contract_hash"],
        "controller_run_hash": coordinates["controller_run_hash"],
        "proposal_generation": coordinates["proposal_generation"],
        "owner_binding_receipt_hash": coordinates["owner_binding_receipt_hash"],
        "alpha_family_plan_hash": coordinates["alpha_family_plan_hash"],
        "alpha_family_registry_receipt_hash": coordinates[
            "alpha_family_registry_receipt_hash"
        ],
        "alpha_head_cas_receipt_hash": coordinates["alpha_head_cas_receipt_hash"],
        "published_ledger_head_hash": coordinates["published_ledger_head_hash"],
        "durable_store_identity_hash": durable_store_identity_hash,
        "namespace_hash": namespace_hash,
        "registered_look_index": coordinates["registered_look_index"],
        "registered_look_stratum_counts": list(
            coordinates["registered_look_stratum_counts"]  # type: ignore[arg-type]
        ),
        "population_stratum_counts": list(
            coordinates["population_stratum_counts"]  # type: ignore[arg-type]
        ),
        "registered_prefix_axis_hash": coordinates["registered_prefix_axis_hash"],
        "cell_role_axis": list(coordinates["cell_role_axis"]),  # type: ignore[arg-type]
        "cell_id_axis": list(coordinates["cell_id_axis"]),  # type: ignore[arg-type]
        "cell_hash_axis": list(coordinates["cell_hash_axis"]),  # type: ignore[arg-type]
        "cell_scope_hash_axis": list(
            coordinates["cell_scope_hash_axis"]  # type: ignore[arg-type]
        ),
        "cell_alpha_axis": list(coordinates["cell_alpha_axis"]),  # type: ignore[arg-type]
        "observation_receipt_hash_axis": list(
            coordinates["observation_receipt_hash_axis"]  # type: ignore[arg-type]
        ),
        "literal_j_statistical_regime": coordinates[
            "literal_j_statistical_regime"
        ],
        "alpha_spent_again": 0.0,
        "alpha_ledger_advanced": False,
        "three_cells_consumed_atomically": True,
    }


def _receipt_public_payload(
    coordinates: dict[str, object],
    *,
    durable_store_identity_hash: str,
    namespace_hash: str,
    state_name: str,
    transaction_record_hash: str,
) -> dict[str, object]:
    return {
        **coordinates,
        "durable_store_identity_hash": durable_store_identity_hash,
        "namespace_hash": namespace_hash,
        "generation_state_name": state_name,
        "transaction_record_hash": transaction_record_hash,
        "three_cells_durably_consumed": True,
        "nonrecycling_observation_family_closed": True,
        "exact_replay_idempotent": True,
        "separate_content_addressed_namespace": True,
        "alpha_spent_again": 0.0,
        "alpha_ledger_advanced": False,
        "second_alpha_spend_authority": False,
        "d_decision_authority": False,
        "c_measurement_authority": False,
        "calibration_authority": False,
        "oracle_authority": False,
        "action_authority": False,
        "mask_commit_authority": False,
        "terminal_authority": False,
        "performance_authority": False,
        "pareto_authority": False,
        "authority_scope": SAMPLED_V7_D_ALPHA_OBSERVATION_FAMILY_CLOSURE_AUTHORITY,
        "schema": SAMPLED_V7_D_ALPHA_OBSERVATION_FAMILY_CLOSURE_SCHEMA,
    }


@dataclass(frozen=True, slots=True)
class V1SampledV7DAlphaObservationFamilyClosureReceipt:
    campaign_contract_hash: str
    controller_run_hash: str
    proposal_generation: int
    owner_binding_receipt_hash: str
    alpha_family_plan_hash: str
    alpha_family_registry_receipt_hash: str
    alpha_head_cas_receipt_hash: str
    published_ledger_head_hash: str
    registered_look_index: int
    registered_look_stratum_counts: tuple[int, int]
    population_stratum_counts: tuple[int, int]
    registered_prefix_axis_hash: str
    cell_role_axis: tuple[str, str, str]
    cell_id_axis: tuple[str, str, str]
    cell_hash_axis: tuple[str, str, str]
    cell_scope_hash_axis: tuple[str, str, str]
    cell_alpha_axis: tuple[float, float, float]
    observation_receipt_hash_axis: tuple[str, str, str]
    literal_j_statistical_regime: str
    durable_store_identity_hash: str
    namespace_hash: str
    generation_state_name: str
    transaction_record_hash: str
    three_cells_durably_consumed: bool = True
    nonrecycling_observation_family_closed: bool = True
    exact_replay_idempotent: bool = True
    separate_content_addressed_namespace: bool = True
    alpha_spent_again: float = 0.0
    alpha_ledger_advanced: bool = False
    second_alpha_spend_authority: bool = False
    d_decision_authority: bool = False
    c_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    action_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SAMPLED_V7_D_ALPHA_OBSERVATION_FAMILY_CLOSURE_AUTHORITY
    schema: str = SAMPLED_V7_D_ALPHA_OBSERVATION_FAMILY_CLOSURE_SCHEMA
    _corrected_a_observation: V1SampledV7CorrectedAObservationBindingReceipt = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _signed_f_observation: V1SampledV7SignedFObservationBindingReceipt = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _literal_j_observation: (
        V1SampledV7LiteralJObservationBindingReceipt
        | V1SampledV7LiteralJCensusObservationBindingReceipt
    ) = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _durable_root: Path = field(repr=False, compare=False, default=None)  # type: ignore[assignment]
    _issuer_token: InitVar[object | None] = None
    _receipt_hash: str | None = field(init=False, default=None, repr=False)
    _parent_identity_axis: tuple[int, int, int] = field(
        init=False, default=(0, 0, 0), repr=False
    )

    def __post_init__(self, _issuer_token: object | None) -> None:
        if _issuer_token is not _ISSUER_TOKEN:
            raise TypeError("sampled-V7 observation-family closures are issuer-only")
        object.__setattr__(
            self,
            "_parent_identity_axis",
            (
                id(self._corrected_a_observation),
                id(self._signed_f_observation),
                id(self._literal_j_observation),
            ),
        )
        self.assert_live()

    def _payload(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_")
        }

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "closure-receipt-")

    def assert_live(self, expected_hash: str | None = None) -> str:
        if self._parent_identity_axis != (
            id(self._corrected_a_observation),
            id(self._signed_f_observation),
            id(self._literal_j_observation),
        ):
            _refuse(
                V1SampledV7DAlphaObservationFamilyClosureRefusalCode.SOURCE_SUBSTITUTION,
                "one A/F/J observation parent was substituted after closure",
            )
        expected = _family_coordinates(
            corrected_a_observation=self._corrected_a_observation,
            signed_f_observation=self._signed_f_observation,
            literal_j_observation=self._literal_j_observation,
        )
        for name, value in expected.items():
            if getattr(self, name) != value:
                _refuse(
                    V1SampledV7DAlphaObservationFamilyClosureRefusalCode.MUTATED_RECEIPT,
                    f"observation-family closure coordinate {name} changed",
                )
        positive = (
            self.three_cells_durably_consumed,
            self.nonrecycling_observation_family_closed,
            self.exact_replay_idempotent,
            self.separate_content_addressed_namespace,
        )
        closed = (
            self.alpha_ledger_advanced,
            self.second_alpha_spend_authority,
            self.d_decision_authority,
            self.c_measurement_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.action_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.performance_authority,
            self.pareto_authority,
        )
        if (
            any(type(value) is not bool or value is not True for value in positive)
            or any(type(value) is not bool or value is not False for value in closed)
            or self.alpha_spent_again != 0.0
            or self.literal_j_statistical_regime not in _LITERAL_J_STATISTICAL_REGIMES
            or self.authority_scope
            != SAMPLED_V7_D_ALPHA_OBSERVATION_FAMILY_CLOSURE_AUTHORITY
            or self.schema != SAMPLED_V7_D_ALPHA_OBSERVATION_FAMILY_CLOSURE_SCHEMA
        ):
            _refuse(
                V1SampledV7DAlphaObservationFamilyClosureRefusalCode.AUTHORITY_VIOLATION,
                "observation-family closure crossed its nonrecycling authority boundary",
            )
        observed = self._compute_hash()
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1SampledV7DAlphaObservationFamilyClosureRefusalCode.MUTATED_RECEIPT,
                "observation-family closure changed after sealing",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledV7DAlphaObservationFamilyClosureRefusalCode.MUTATED_RECEIPT,
                "observation-family closure hash does not replay",
            )
        _, directory = _owned_directory(
            self._durable_root,
            namespace_hash=self.namespace_hash,
            create=False,
        )
        state = _read_state(directory / self.generation_state_name)
        record = _record_payload(
            expected,
            durable_store_identity_hash=self.durable_store_identity_hash,
            namespace_hash=self.namespace_hash,
        )
        expected_state = {
            "record": record,
            "transaction_record_hash": self.transaction_record_hash,
            "closure_receipt_hash": observed,
        }
        if state != expected_state:
            _refuse(
                V1SampledV7DAlphaObservationFamilyClosureRefusalCode.STATE_INVALID,
                "durable observation-family state no longer equals this closure",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_live()

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live observation-family closures are nonserializable")


def close_v1_sampled_v7_d_alpha_observation_family(
    *,
    durable_root: str | os.PathLike[str],
    corrected_a_observation: V1SampledV7CorrectedAObservationBindingReceipt,
    signed_f_observation: V1SampledV7SignedFObservationBindingReceipt,
    literal_j_observation: (
        V1SampledV7LiteralJObservationBindingReceipt
        | V1SampledV7LiteralJCensusObservationBindingReceipt
    ),
) -> V1SampledV7DAlphaObservationFamilyClosureReceipt:
    """Atomically consume exactly one registered A/F/J observation triplet."""

    coordinates = _family_coordinates(
        corrected_a_observation=corrected_a_observation,
        signed_f_observation=signed_f_observation,
        literal_j_observation=literal_j_observation,
    )
    durable_store_identity_hash = _identity(
        coordinates["durable_store_identity_hash"],
        name="source-owned durable store identity",
    )
    namespace_hash = _namespace_hash(
        campaign_contract_hash=coordinates["campaign_contract_hash"],  # type: ignore[arg-type]
        controller_run_hash=coordinates["controller_run_hash"],  # type: ignore[arg-type]
        durable_store_identity_hash=durable_store_identity_hash,
    )
    _, directory = _owned_directory(
        durable_root,
        namespace_hash=namespace_hash,
        create=True,
    )
    generation = coordinates["proposal_generation"]
    assert type(generation) is int
    state_name = _state_name(generation)
    state_path = directory / state_name
    record = _record_payload(
        coordinates,
        durable_store_identity_hash=durable_store_identity_hash,
        namespace_hash=namespace_hash,
    )
    record_hash = content_hash(record, prefix=_PREFIX + "record-")
    receipt_public = _receipt_public_payload(
        coordinates,
        durable_store_identity_hash=durable_store_identity_hash,
        namespace_hash=namespace_hash,
        state_name=state_name,
        transaction_record_hash=record_hash,
    )
    receipt_hash = content_hash(
        receipt_public,
        prefix=_PREFIX + "closure-receipt-",
    )
    state_payload = {
        "record": record,
        "transaction_record_hash": record_hash,
        "closure_receipt_hash": receipt_hash,
    }

    lock_path = directory / _LOCK_NAME
    lock_fd = _open_lock(lock_path)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        _fsync_directory(directory)
        observed = _read_state(state_path)
        if observed is not None and observed != state_payload:
            _refuse(
                V1SampledV7DAlphaObservationFamilyClosureRefusalCode.TRANSACTION_CONFLICT,
                "another A/F/J observation family already consumed this proposal generation",
            )
        if observed is None:
            encoded = _canonical_json_bytes(state_payload)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{state_name}.",
                suffix=".partial",
                dir=directory,
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb", buffering=0) as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, state_path)
                _fsync_directory(directory)
            finally:
                temporary.unlink(missing_ok=True)
            if _read_state(state_path) != state_payload:
                _refuse(
                    V1SampledV7DAlphaObservationFamilyClosureRefusalCode.DURABILITY_FAILURE,
                    "observation-family state failed exact read-after-write replay",
                )
    except V1SampledV7DAlphaObservationFamilyClosureRefusal:
        raise
    except OSError as error:
        raise V1SampledV7DAlphaObservationFamilyClosureRefusal(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.DURABILITY_FAILURE,
            f"durable observation-family transaction failed: {error}",
        ) from error
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)

    receipt = V1SampledV7DAlphaObservationFamilyClosureReceipt(
        **receipt_public,  # type: ignore[arg-type]
        _corrected_a_observation=corrected_a_observation,
        _signed_f_observation=signed_f_observation,
        _literal_j_observation=literal_j_observation,
        _durable_root=Path(durable_root),
        _issuer_token=_ISSUER_TOKEN,
    )
    if receipt.receipt_hash != receipt_hash:
        _refuse(
            V1SampledV7DAlphaObservationFamilyClosureRefusalCode.DURABILITY_FAILURE,
            "issued closure receipt does not match its durable content address",
        )
    return receipt


__all__ = [
    "SAMPLED_V7_D_ALPHA_OBSERVATION_FAMILY_CLOSURE_AUTHORITY",
    "SAMPLED_V7_D_ALPHA_OBSERVATION_FAMILY_CLOSURE_SCHEMA",
    "SAMPLED_V7_D_ALPHA_OBSERVATION_FAMILY_RECORD_SCHEMA",
    "V1SampledV7DAlphaObservationFamilyClosureReceipt",
    "V1SampledV7DAlphaObservationFamilyClosureRefusal",
    "V1SampledV7DAlphaObservationFamilyClosureRefusalCode",
    "close_v1_sampled_v7_d_alpha_observation_family",
]
