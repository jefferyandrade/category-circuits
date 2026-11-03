"""External persistence for modern-v1 confidence-box v2 receipts.

This module stores the route-scale arrays of
``V1CorrectedAttributionConfidenceBoxV2Receipt`` in the generic immutable
external-state store.  The checkpoint surface is a typed, root-free descriptor;
opening verifies metadata, array identities, current receipt hash, and the
immediate prior running-intersection transition without using pickle and
without constructing a route-scale in-memory receipt object.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import json
import math
import os
from types import MappingProxyType
from typing import Any, Final, NoReturn, cast

import numpy as np

from mechanistic_systems.circuits.irg.modern_v1_external_state import (
    V1ExternalArrayIdentity,
    V1ExternalArrayPolicy,
    V1ExternalStateDescriptor,
    V1ExternalStateLiveAttachment,
    V1ExternalStateView,
    open_external_state,
    open_external_state_with_attachment,
    publish_external_state,
    publish_external_state_with_attachment,
)
from mechanistic_systems.circuits.irg.modern_v1_proposal_statistics import (
    CORRECTED_ATTRIBUTION_CONFIDENCE_BOX_V2_SCHEMA,
    V1CorrectedAttributionConfidenceBoxV2Receipt,
)
from mechanistic_systems.contracts.content_address import content_hash


CONFIDENCE_BOX_V2_EXTERNAL_DESCRIPTOR_SCHEMA: Final = (
    "partition_free_v1_confidence_box_v2_external_descriptor_v1"
)
CONFIDENCE_BOX_V2_EXTERNAL_STATE_SCHEMA: Final = (
    "partition_free_v1_confidence_box_v2_external_state_v1"
)
CONFIDENCE_BOX_V2_EXTERNAL_METADATA_SCHEMA: Final = (
    "partition_free_v1_confidence_box_v2_external_metadata_v1"
)
CONFIDENCE_BOX_V2_EXTERNAL_MEMORY_FORECAST_SCHEMA: Final = (
    "partition_free_v1_confidence_box_v2_external_memory_forecast_v1"
)
CONFIDENCE_BOX_V2_EXTERNAL_NAMESPACE: Final = "modern_v1_confidence_box_v2"
CONFIDENCE_BOX_V2_EXTERNAL_AUTHORITY_SCOPE: Final = (
    "score_order_confidence_box_resume_only_no_fq_bce_or_mask_authority_v1"
)
CONFIDENCE_BOX_V2_EXTERNAL_MEMORY_SCOPE: Final = (
    "route_scale_confidence_box_arrays_memory_mapped_metadata_scale_transition_"
    "verification_no_pickle_v1"
)

_METADATA_NAME: Final = "metadata_utf8"
_BOX_V2_PREFIX: Final = "partition-free-v1-corrected-attribution-confidence-box-v2-"
_COMMIT_PREFIX: Final = "partition-free-v1-confidence-box-v2-external-"
_LE_F64: Final = np.dtype("<f8")
_U8: Final = np.dtype("u1")
_CURRENT_ARRAY_NAMES: Final = (
    "class_balanced_absolute_point",
    "current_class_balanced_absolute_lower",
    "current_class_balanced_absolute_upper",
    "cumulative_class_balanced_absolute_lower",
    "cumulative_class_balanced_absolute_upper",
)
_PRIOR_ARRAY_PREFIX: Final = "prior_"
_TYPED_VALIDATION_CHUNK_ELEMENTS: Final = 128 * 1024
_PUBLISH_BUFFER_ELEMENTS: Final = 1024 * 1024
_VERIFY_BUFFER_BYTES: Final = 8 * 1024 * 1024


class V1ConfidenceBoxExternalStateRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    INVALID_SCHEMA = "invalid_schema"
    TYPED_IDENTITY_MISMATCH = "typed_identity_mismatch"
    PRIOR_CONTINUITY_MISMATCH = "prior_continuity_mismatch"
    CLOSED_VIEW = "closed_view"


class V1ConfidenceBoxExternalStateRefusal(ValueError):
    """Typed fail-closed error for confidence-box external persistence."""

    def __init__(
        self,
        code: V1ConfidenceBoxExternalStateRefusalCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1ConfidenceBoxExternalStateRefusalCode,
    message: str,
) -> NoReturn:
    raise V1ConfidenceBoxExternalStateRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not cast(str, value):
        _refuse(
            V1ConfidenceBoxExternalStateRefusalCode.INVALID_SCHEMA,
            f"{name} must be one nonempty string",
        )
    return cast(str, value)


def _optional_identity(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _identity(value, name=name)


def _exact_nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or cast(int, value) < 0:
        _refuse(
            V1ConfidenceBoxExternalStateRefusalCode.INVALID_SCHEMA,
            f"{name} must be an exact nonnegative integer",
        )
    return cast(int, value)


def _exact_positive_int(value: object, *, name: str) -> int:
    result = _exact_nonnegative_int(value, name=name)
    if result <= 0:
        _refuse(
            V1ConfidenceBoxExternalStateRefusalCode.INVALID_SCHEMA,
            f"{name} must be positive",
        )
    return result


def _exact_mapping(
    value: object,
    *,
    keys: set[str],
    name: str,
) -> dict[str, object]:
    if type(value) is not dict:
        _refuse(
            V1ConfidenceBoxExternalStateRefusalCode.INVALID_SCHEMA,
            f"{name} must be one exact object",
        )
    result = cast(dict[str, object], value)
    if set(result) != keys:
        _refuse(
            V1ConfidenceBoxExternalStateRefusalCode.INVALID_SCHEMA,
            f"{name} fields are not exact",
        )
    return result


def _string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) not in {list, tuple}:
        _refuse(
            V1ConfidenceBoxExternalStateRefusalCode.INVALID_SCHEMA,
            f"{name} must be an ordered string sequence",
        )
    return tuple(
        _identity(item, name=f"{name} item")
        for item in cast(Sequence[object], value)
    )


def _pair_of_strings(value: object, *, name: str) -> tuple[str, str]:
    result = _string_tuple(value, name=name)
    if len(result) != 2:
        _refuse(
            V1ConfidenceBoxExternalStateRefusalCode.INVALID_SCHEMA,
            f"{name} must contain exactly two values",
        )
    return cast(tuple[str, str], result)


def _pair_of_ints(value: object, *, name: str) -> tuple[int, int]:
    if type(value) not in {list, tuple}:
        _refuse(
            V1ConfidenceBoxExternalStateRefusalCode.INVALID_SCHEMA,
            f"{name} must contain exactly two integers",
        )
    result = tuple(
        _exact_nonnegative_int(item, name=f"{name} item")
        for item in cast(Sequence[object], value)
    )
    if len(result) != 2:
        _refuse(
            V1ConfidenceBoxExternalStateRefusalCode.INVALID_SCHEMA,
            f"{name} must contain exactly two integers",
        )
    return cast(tuple[int, int], result)


def _nested_string_pair(
    value: object,
    *,
    name: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if type(value) not in {list, tuple} or len(cast(Sequence[object], value)) != 2:
        _refuse(
            V1ConfidenceBoxExternalStateRefusalCode.INVALID_SCHEMA,
            f"{name} must contain exactly two ledgers",
        )
    rows = cast(Sequence[object], value)
    return (
        _string_tuple(rows[0], name=f"{name} first"),
        _string_tuple(rows[1], name=f"{name} second"),
    )


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _strict_json(raw: bytes) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in items:
            if key in result:
                _refuse(
                    V1ConfidenceBoxExternalStateRefusalCode.INVALID_SCHEMA,
                    f"metadata repeats field {key!r}",
                )
            result[key] = item
        return result

    try:
        return json.loads(raw.decode("ascii"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _refuse(
            V1ConfidenceBoxExternalStateRefusalCode.INVALID_SCHEMA,
            f"metadata JSON is invalid: {error}",
        )


def _strict_metadata(value: np.ndarray) -> dict[str, object]:
    if value.dtype != _U8 or value.ndim != 1 or value.flags.writeable:
        _refuse(
            V1ConfidenceBoxExternalStateRefusalCode.INVALID_SCHEMA,
            "metadata must be a read-only uint8 vector",
        )
    raw = value.tobytes(order="C")
    decoded = _strict_json(raw)
    metadata = _exact_mapping(
        decoded,
        keys={"schema", "current", "prior"},
        name="confidence-box metadata",
    )
    if raw != _canonical_json_bytes(metadata):
        _refuse(
            V1ConfidenceBoxExternalStateRefusalCode.INVALID_SCHEMA,
            "metadata is not canonical ASCII JSON",
        )
    if metadata["schema"] != CONFIDENCE_BOX_V2_EXTERNAL_METADATA_SCHEMA:
        _refuse(
            V1ConfidenceBoxExternalStateRefusalCode.INVALID_SCHEMA,
            "metadata schema is invalid",
        )
    _receipt_metadata_checked(metadata["current"], name="current receipt")
    if metadata["prior"] is not None:
        _receipt_metadata_checked(metadata["prior"], name="prior receipt")
    return metadata


def _receipt_metadata_checked(value: object, *, name: str) -> dict[str, object]:
    array_names = set(_CURRENT_ARRAY_NAMES)
    payload_keys = {
        "schema",
        "state_hash",
        "envelope_certificate_hash",
        "design_hash",
        "base_serfling_design_hash",
        "campaign_contract_hash",
        "model_identity_hash",
        "query_contract_hash",
        "preprocessing_hash",
        "current_mask_hash",
        "raw_logit_stream_source_hash",
        "sampling_law_hash",
        "proposal_generation",
        "route_count",
        "route_axis_hash",
        "stratum_names",
        "stratum_counts",
        "population_stratum_counts",
        "sealed_population_order_hashes",
        "observed_context_ids_by_stratum",
        "observed_draw_ids_by_stratum",
        "class_balanced_stratum_weights",
        "look_index",
        "prior_confidence_box_receipt_hash",
        "prior_statistics_state_hash",
        "trajectory_alpha_ledger_hash",
        "trajectory_alpha_ledger_generation",
        "trajectory_familywise_alpha",
        "trajectory_alpha_committed_before_generation",
        "generation_alpha_allocation",
        "registered_look_stratum_counts",
        "alpha_by_look",
        "total_registered_alpha",
        "look_alpha",
        "cumulative_alpha_spent",
        "remaining_generation_alpha_after_look",
        "unallocated_generation_alpha_reserve",
        "remaining_trajectory_alpha_after_generation_allocation",
        "per_cell_combined_two_sided_alpha",
        "per_cell_hoeffding_serfling_two_sided_alpha",
        "per_cell_empirical_bernstein_two_sided_alpha",
        "empirical_bernstein_theorem_delta",
        "hoeffding_serfling_log_factor",
        "empirical_bernstein_log_factor",
        "hoeffding_serfling_finite_population_corrections",
        "hoeffding_serfling_radius_multipliers_by_stratum",
        "empirical_bernstein_rho_by_stratum",
        "empirical_bernstein_rho_branches_by_stratum",
        "empirical_bernstein_sigma_radius_multipliers_by_stratum",
        "empirical_bernstein_range_radius_multipliers_by_stratum",
        "census_by_stratum",
        "covered_estimand_count",
        "covered_estimands",
        "estimand_weighting",
        "covered_stratum_count",
        "covered_route_count",
        "covered_two_sided_statements_at_look",
        "registered_look_count",
        "covered_two_sided_statement_looks",
        "sampling_law_kind",
        "trajectory_alpha_policy_kind",
        "bound_method_names",
        "bound_method_two_sided_alpha_fractions",
        "empirical_variance_convention",
        "empirical_bernstein_kappa",
        "deterministic_interval_id",
        "outward_rounding_policy",
        "census_override_id",
        "cumulative_intersection_nonempty",
        "inequality_id",
        "evidence_ceiling",
    }
    payload = _exact_mapping(
        value,
        keys={
            "receipt_hash",
            "payload_without_arrays",
            "array_names",
        },
        name=name,
    )
    receipt_hash = _identity(payload["receipt_hash"], name=f"{name} hash")
    no_arrays = _exact_mapping(
        payload["payload_without_arrays"],
        keys=payload_keys,
        name=f"{name} scalar payload",
    )
    if no_arrays["schema"] != CORRECTED_ATTRIBUTION_CONFIDENCE_BOX_V2_SCHEMA:
        _refuse(
            V1ConfidenceBoxExternalStateRefusalCode.INVALID_SCHEMA,
            f"{name} has wrong confidence-box schema",
        )
    route_count = _exact_positive_int(no_arrays["route_count"], name=f"{name} route_count")
    look_index = _exact_positive_int(no_arrays["look_index"], name=f"{name} look_index")
    if _exact_positive_int(no_arrays["covered_route_count"], name=f"{name} covered_route_count") != route_count:
        _refuse(
            V1ConfidenceBoxExternalStateRefusalCode.INVALID_SCHEMA,
            f"{name} route coverage does not match route count",
        )
    _identity(no_arrays["route_axis_hash"], name=f"{name} route axis hash")
    _pair_of_strings(no_arrays["stratum_names"], name=f"{name} stratum names")
    _pair_of_ints(no_arrays["stratum_counts"], name=f"{name} stratum counts")
    _pair_of_ints(
        no_arrays["population_stratum_counts"],
        name=f"{name} population stratum counts",
    )
    _nested_string_pair(
        no_arrays["observed_context_ids_by_stratum"],
        name=f"{name} contexts",
    )
    _nested_string_pair(
        no_arrays["observed_draw_ids_by_stratum"],
        name=f"{name} draws",
    )
    if bool(no_arrays["cumulative_intersection_nonempty"]) is not True:
        _refuse(
            V1ConfidenceBoxExternalStateRefusalCode.INVALID_SCHEMA,
            f"{name} does not certify a nonempty running intersection",
        )
    names = tuple(_string_tuple(payload["array_names"], name=f"{name} array names"))
    if set(names) != array_names or names != tuple(sorted(array_names)):
        _refuse(
            V1ConfidenceBoxExternalStateRefusalCode.INVALID_SCHEMA,
            f"{name} array-name ledger is not canonical",
        )
    if look_index == 1:
        if (
            no_arrays["prior_confidence_box_receipt_hash"] is not None
            or no_arrays["prior_statistics_state_hash"] is not None
        ):
            _refuse(
                V1ConfidenceBoxExternalStateRefusalCode.INVALID_SCHEMA,
                "first confidence-box look cannot name a prior",
            )
    else:
        _optional_identity(
            no_arrays["prior_confidence_box_receipt_hash"],
            name=f"{name} prior hash",
        )
        _optional_identity(
            no_arrays["prior_statistics_state_hash"],
            name=f"{name} prior statistics state hash",
        )
    result = dict(payload)
    result["receipt_hash"] = receipt_hash
    result["payload_without_arrays"] = no_arrays
    result["array_names"] = names
    return result


def _array_hash_token(identity: V1ExternalArrayIdentity) -> dict[str, object]:
    return {
        "dtype": str(np.dtype(identity.dtype)),
        "shape": list(identity.shape),
        "sha256": identity.sha256,
    }


def _array_name(prefix: str | None, name: str) -> str:
    return name if prefix is None else f"{prefix}{name}"


def _receipt_hash_from_identities(
    *,
    metadata: Mapping[str, object],
    identities: Mapping[str, V1ExternalArrayIdentity],
    prefix: str | None,
) -> str:
    checked = _receipt_metadata_checked(metadata, name="receipt metadata")
    payload = dict(cast(Mapping[str, object], checked["payload_without_arrays"]))
    for name in _CURRENT_ARRAY_NAMES:
        identity = identities[_array_name(prefix, name)]
        payload[name] = _array_hash_token(identity)
    return content_hash(payload, prefix=_BOX_V2_PREFIX)


@dataclass(frozen=True, slots=True)
class V1ConfidenceBoxExternalMemoryForecast:
    """Bounded-memory ledger for confidence-box external persistence."""

    current_route_scale_array_count: int
    prior_route_scale_array_count: int
    route_count: int
    route_scale_array_bytes: int
    metadata_payload_bytes: int
    persisted_payload_bytes: int
    largest_route_scale_array_bytes: int
    publication_stream_chunk_bytes_upper_bound: int
    verification_stream_buffer_bytes_upper_bound: int
    typed_open_invariant_chunk_elements: int
    typed_open_scratch_bytes_upper_bound: int
    reopened_numeric_heap_copy_bytes: int = 0
    materialized_current_owned_bytes: int = 0
    materialized_prior_owned_bytes: int = 0
    pickle_codec_used: bool = False
    memory_scope: str = CONFIDENCE_BOX_V2_EXTERNAL_MEMORY_SCOPE
    schema: str = CONFIDENCE_BOX_V2_EXTERNAL_MEMORY_FORECAST_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "current_route_scale_array_count",
            "prior_route_scale_array_count",
            "route_count",
            "route_scale_array_bytes",
            "metadata_payload_bytes",
            "persisted_payload_bytes",
            "largest_route_scale_array_bytes",
            "publication_stream_chunk_bytes_upper_bound",
            "verification_stream_buffer_bytes_upper_bound",
            "typed_open_invariant_chunk_elements",
            "typed_open_scratch_bytes_upper_bound",
            "reopened_numeric_heap_copy_bytes",
            "materialized_current_owned_bytes",
            "materialized_prior_owned_bytes",
        ):
            object.__setattr__(
                self, name, _exact_nonnegative_int(getattr(self, name), name=name)
            )
        if (
            self.current_route_scale_array_count != len(_CURRENT_ARRAY_NAMES)
            or self.route_count <= 0
            or self.route_scale_array_bytes
            != (self.current_route_scale_array_count + self.prior_route_scale_array_count)
            * self.route_count
            * _LE_F64.itemsize
            or self.persisted_payload_bytes
            != self.route_scale_array_bytes + self.metadata_payload_bytes
            or self.reopened_numeric_heap_copy_bytes != 0
            or self.pickle_codec_used is not False
            or self.memory_scope != CONFIDENCE_BOX_V2_EXTERNAL_MEMORY_SCOPE
            or self.schema != CONFIDENCE_BOX_V2_EXTERNAL_MEMORY_FORECAST_SCHEMA
        ):
            _refuse(
                V1ConfidenceBoxExternalStateRefusalCode.INVALID_SCHEMA,
                "confidence-box memory forecast is incoherent",
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "current_route_scale_array_count": self.current_route_scale_array_count,
            "prior_route_scale_array_count": self.prior_route_scale_array_count,
            "route_count": self.route_count,
            "route_scale_array_bytes": self.route_scale_array_bytes,
            "metadata_payload_bytes": self.metadata_payload_bytes,
            "persisted_payload_bytes": self.persisted_payload_bytes,
            "largest_route_scale_array_bytes": self.largest_route_scale_array_bytes,
            "publication_stream_chunk_bytes_upper_bound": (
                self.publication_stream_chunk_bytes_upper_bound
            ),
            "verification_stream_buffer_bytes_upper_bound": (
                self.verification_stream_buffer_bytes_upper_bound
            ),
            "typed_open_invariant_chunk_elements": (
                self.typed_open_invariant_chunk_elements
            ),
            "typed_open_scratch_bytes_upper_bound": (
                self.typed_open_scratch_bytes_upper_bound
            ),
            "reopened_numeric_heap_copy_bytes": self.reopened_numeric_heap_copy_bytes,
            "materialized_current_owned_bytes": self.materialized_current_owned_bytes,
            "materialized_prior_owned_bytes": self.materialized_prior_owned_bytes,
            "pickle_codec_used": self.pickle_codec_used,
            "memory_scope": self.memory_scope,
        }

    @classmethod
    def from_json_dict(cls, value: object) -> "V1ConfidenceBoxExternalMemoryForecast":
        keys = {
            "schema",
            "current_route_scale_array_count",
            "prior_route_scale_array_count",
            "route_count",
            "route_scale_array_bytes",
            "metadata_payload_bytes",
            "persisted_payload_bytes",
            "largest_route_scale_array_bytes",
            "publication_stream_chunk_bytes_upper_bound",
            "verification_stream_buffer_bytes_upper_bound",
            "typed_open_invariant_chunk_elements",
            "typed_open_scratch_bytes_upper_bound",
            "reopened_numeric_heap_copy_bytes",
            "materialized_current_owned_bytes",
            "materialized_prior_owned_bytes",
            "pickle_codec_used",
            "memory_scope",
        }
        return cls(**_exact_mapping(value, keys=keys, name="memory forecast"))  # type: ignore[arg-type]


def _forecast(
    *,
    route_count: int,
    has_prior: bool,
    metadata_bytes: int,
) -> V1ConfidenceBoxExternalMemoryForecast:
    current_count = len(_CURRENT_ARRAY_NAMES)
    prior_count = len(_CURRENT_ARRAY_NAMES) if has_prior else 0
    array_bytes = (current_count + prior_count) * route_count * _LE_F64.itemsize
    largest_array = route_count * _LE_F64.itemsize
    chunk_bytes = min(largest_array, _PUBLISH_BUFFER_ELEMENTS * _LE_F64.itemsize)
    validation_elements = min(route_count, _TYPED_VALIDATION_CHUNK_ELEMENTS)
    # Three float buffers plus two boolean buffers cover current bounds and
    # optional prior-transition replay without route-scale scratch.
    scratch = validation_elements * (3 * _LE_F64.itemsize + 2 * _U8.itemsize)
    return V1ConfidenceBoxExternalMemoryForecast(
        current_route_scale_array_count=current_count,
        prior_route_scale_array_count=prior_count,
        route_count=route_count,
        route_scale_array_bytes=array_bytes,
        metadata_payload_bytes=metadata_bytes,
        persisted_payload_bytes=array_bytes + metadata_bytes,
        largest_route_scale_array_bytes=largest_array,
        publication_stream_chunk_bytes_upper_bound=chunk_bytes,
        verification_stream_buffer_bytes_upper_bound=min(
            max(largest_array, metadata_bytes),
            _VERIFY_BUFFER_BYTES,
        ),
        typed_open_invariant_chunk_elements=validation_elements,
        typed_open_scratch_bytes_upper_bound=scratch,
        materialized_current_owned_bytes=current_count * largest_array,
        materialized_prior_owned_bytes=prior_count * largest_array,
    )


@dataclass(frozen=True, slots=True)
class V1ConfidenceBoxExternalDescriptor:
    """Root-free typed descriptor for one v2 confidence-box continuation."""

    external_state: V1ExternalStateDescriptor
    receipt_hash: str
    prior_receipt_hash: str | None
    prior_statistics_state_hash: str | None
    look_index: int
    route_count: int
    route_axis_hash: str
    memory_forecast: V1ConfidenceBoxExternalMemoryForecast
    commit_hash: str
    authority_scope: str = CONFIDENCE_BOX_V2_EXTERNAL_AUTHORITY_SCOPE
    proposal_discovery_input_authority: bool = True
    fq_certification_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    schema: str = CONFIDENCE_BOX_V2_EXTERNAL_DESCRIPTOR_SCHEMA

    def __post_init__(self) -> None:
        if type(self.external_state) is not V1ExternalStateDescriptor:
            _refuse(
                V1ConfidenceBoxExternalStateRefusalCode.INVALID_TYPE,
                "external_state must use the exact generic descriptor type",
            )
        if (
            self.external_state.namespace != CONFIDENCE_BOX_V2_EXTERNAL_NAMESPACE
            or self.external_state.state_schema
            != CONFIDENCE_BOX_V2_EXTERNAL_STATE_SCHEMA
        ):
            _refuse(
                V1ConfidenceBoxExternalStateRefusalCode.INVALID_SCHEMA,
                "generic external state has the wrong confidence-box namespace/schema",
            )
        object.__setattr__(
            self, "receipt_hash", _identity(self.receipt_hash, name="receipt hash")
        )
        object.__setattr__(
            self,
            "prior_receipt_hash",
            _optional_identity(self.prior_receipt_hash, name="prior receipt hash"),
        )
        object.__setattr__(
            self,
            "prior_statistics_state_hash",
            _optional_identity(
                self.prior_statistics_state_hash,
                name="prior statistics state hash",
            ),
        )
        object.__setattr__(self, "look_index", _exact_positive_int(self.look_index, name="look_index"))
        object.__setattr__(self, "route_count", _exact_positive_int(self.route_count, name="route_count"))
        object.__setattr__(
            self, "route_axis_hash", _identity(self.route_axis_hash, name="route axis hash")
        )
        if type(self.memory_forecast) is not V1ConfidenceBoxExternalMemoryForecast:
            _refuse(
                V1ConfidenceBoxExternalStateRefusalCode.INVALID_TYPE,
                "descriptor requires its exact memory forecast",
            )
        if (
            self.authority_scope != CONFIDENCE_BOX_V2_EXTERNAL_AUTHORITY_SCOPE
            or self.proposal_discovery_input_authority is not True
            or self.fq_certification_authority is not False
            or self.bce_oracle_authority is not False
            or self.mask_commit_authority is not False
            or self.schema != CONFIDENCE_BOX_V2_EXTERNAL_DESCRIPTOR_SCHEMA
        ):
            _refuse(
                V1ConfidenceBoxExternalStateRefusalCode.INVALID_SCHEMA,
                "confidence-box descriptor exceeded score-continuation authority",
            )
        self._validate_array_manifest()
        expected = content_hash(self._commit_payload(), prefix=_COMMIT_PREFIX)
        if self.commit_hash != expected:
            _refuse(
                V1ConfidenceBoxExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
                "confidence-box external commit hash does not replay",
            )

    def _validate_array_manifest(self) -> None:
        by_name = {identity.logical_name: identity for identity in self.external_state.arrays}
        expected = {_METADATA_NAME, *_CURRENT_ARRAY_NAMES}
        if self.prior_receipt_hash is not None:
            expected.update(f"{_PRIOR_ARRAY_PREFIX}{name}" for name in _CURRENT_ARRAY_NAMES)
        if set(by_name) != expected:
            _refuse(
                V1ConfidenceBoxExternalStateRefusalCode.INVALID_SCHEMA,
                "confidence-box external array names are incomplete or unexpected",
            )
        if any(identity.allow_nonfinite for identity in by_name.values()):
            _refuse(
                V1ConfidenceBoxExternalStateRefusalCode.INVALID_SCHEMA,
                "confidence-box arrays cannot waive finite verification",
            )
        metadata = by_name[_METADATA_NAME]
        if metadata.dtype != _U8.str or len(metadata.shape) != 1:
            _refuse(
                V1ConfidenceBoxExternalStateRefusalCode.INVALID_SCHEMA,
                "metadata array identity is invalid",
            )
        for name in expected - {_METADATA_NAME}:
            identity = by_name[name]
            if identity.dtype != _LE_F64.str or identity.shape != (self.route_count,):
                _refuse(
                    V1ConfidenceBoxExternalStateRefusalCode.INVALID_SCHEMA,
                    f"{name} array identity is invalid",
                )
        if sum(identity.byte_count for identity in by_name.values()) != self.memory_forecast.persisted_payload_bytes:
            _refuse(
                V1ConfidenceBoxExternalStateRefusalCode.INVALID_SCHEMA,
                "external array bytes differ from the memory forecast",
            )

    def _commit_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "external_state_descriptor_hash": self.external_state.descriptor_hash,
            "receipt_hash": self.receipt_hash,
            "prior_receipt_hash": self.prior_receipt_hash,
            "prior_statistics_state_hash": self.prior_statistics_state_hash,
            "look_index": self.look_index,
            "route_count": self.route_count,
            "route_axis_hash": self.route_axis_hash,
            "memory_forecast": self.memory_forecast.to_json_dict(),
            "authority_scope": self.authority_scope,
            "proposal_discovery_input_authority": self.proposal_discovery_input_authority,
            "fq_certification_authority": self.fq_certification_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
        }

    def to_json_dict(self) -> dict[str, object]:
        payload = self._commit_payload()
        payload.pop("external_state_descriptor_hash")
        payload["external_state"] = self.external_state.to_json_dict()
        payload["commit_hash"] = self.commit_hash
        return payload

    @property
    def descriptor_hash(self) -> str:
        return content_hash(self.to_json_dict())

    @classmethod
    def from_json_dict(cls, value: object) -> "V1ConfidenceBoxExternalDescriptor":
        keys = {
            "schema",
            "external_state",
            "receipt_hash",
            "prior_receipt_hash",
            "prior_statistics_state_hash",
            "look_index",
            "route_count",
            "route_axis_hash",
            "memory_forecast",
            "authority_scope",
            "proposal_discovery_input_authority",
            "fq_certification_authority",
            "bce_oracle_authority",
            "mask_commit_authority",
            "commit_hash",
        }
        payload = _exact_mapping(value, keys=keys, name="confidence-box descriptor")
        return cls(
            external_state=V1ExternalStateDescriptor.from_json_dict(
                payload["external_state"]
            ),
            receipt_hash=cast(str, payload["receipt_hash"]),
            prior_receipt_hash=cast(str | None, payload["prior_receipt_hash"]),
            prior_statistics_state_hash=cast(str | None, payload["prior_statistics_state_hash"]),
            look_index=cast(int, payload["look_index"]),
            route_count=cast(int, payload["route_count"]),
            route_axis_hash=cast(str, payload["route_axis_hash"]),
            memory_forecast=V1ConfidenceBoxExternalMemoryForecast.from_json_dict(
                payload["memory_forecast"]
            ),
            authority_scope=cast(str, payload["authority_scope"]),
            proposal_discovery_input_authority=cast(bool, payload["proposal_discovery_input_authority"]),
            fq_certification_authority=cast(bool, payload["fq_certification_authority"]),
            bce_oracle_authority=cast(bool, payload["bce_oracle_authority"]),
            mask_commit_authority=cast(bool, payload["mask_commit_authority"]),
            schema=cast(str, payload["schema"]),
            commit_hash=cast(str, payload["commit_hash"]),
        )

    def __reduce__(self) -> tuple[object, tuple[dict[str, object], str]]:
        return _rebuild_confidence_box_external_descriptor, (
            self.to_json_dict(),
            self.descriptor_hash,
        )


def _rebuild_confidence_box_external_descriptor(
    payload: Mapping[str, object],
    expected_hash: str,
) -> V1ConfidenceBoxExternalDescriptor:
    rebuilt = V1ConfidenceBoxExternalDescriptor.from_json_dict(dict(payload))
    if rebuilt.descriptor_hash != _identity(expected_hash, name="expected descriptor hash"):
        _refuse(
            V1ConfidenceBoxExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
            "confidence-box descriptor hash does not replay",
        )
    return rebuilt


def _metadata_from_receipt(
    receipt: V1CorrectedAttributionConfidenceBoxV2Receipt,
) -> dict[str, object]:
    payload = dict(receipt._hash_payload())  # noqa: SLF001 - content-address replay adapter
    for name in _CURRENT_ARRAY_NAMES:
        payload.pop(name)
    return {
        "receipt_hash": receipt.receipt_hash,
        "payload_without_arrays": payload,
        "array_names": tuple(sorted(_CURRENT_ARRAY_NAMES)),
    }


def _array_map_from_receipt(
    receipt: V1CorrectedAttributionConfidenceBoxV2Receipt,
    *,
    prefix: str | None = None,
) -> dict[str, np.ndarray]:
    return {
        _array_name(prefix, name): np.asarray(getattr(receipt, name), dtype=_LE_F64)
        for name in _CURRENT_ARRAY_NAMES
    }


def _validate_receipt_pair(
    receipt: V1CorrectedAttributionConfidenceBoxV2Receipt,
    prior: V1CorrectedAttributionConfidenceBoxV2Receipt | None,
) -> None:
    if type(receipt) is not V1CorrectedAttributionConfidenceBoxV2Receipt:
        _refuse(
            V1ConfidenceBoxExternalStateRefusalCode.INVALID_TYPE,
            "current confidence box must have exact v2 receipt type",
        )
    receipt.assert_integrity()
    if receipt.look_index == 1:
        if prior is not None:
            _refuse(
                V1ConfidenceBoxExternalStateRefusalCode.PRIOR_CONTINUITY_MISMATCH,
                "first confidence-box look cannot publish a prior",
            )
        receipt.assert_prior_transition(None)
    else:
        if type(prior) is not V1CorrectedAttributionConfidenceBoxV2Receipt:
            _refuse(
                V1ConfidenceBoxExternalStateRefusalCode.PRIOR_CONTINUITY_MISMATCH,
                "later confidence-box look requires immediate prior receipt",
            )
        receipt.assert_prior_transition(prior)


def publish_confidence_box_v2_external_state(
    root: str | os.PathLike[str],
    receipt: V1CorrectedAttributionConfidenceBoxV2Receipt,
    *,
    prior: V1CorrectedAttributionConfidenceBoxV2Receipt | None = None,
    issue_live_attachment: bool = False,
) -> (
    V1ConfidenceBoxExternalDescriptor
    | tuple[V1ConfidenceBoxExternalDescriptor, V1ExternalStateLiveAttachment]
):
    """Publish a v2 confidence-box receipt and optional immediate prior."""

    _validate_receipt_pair(receipt, prior)
    current_metadata = _metadata_from_receipt(receipt)
    prior_metadata = None if prior is None else _metadata_from_receipt(prior)
    metadata = {
        "schema": CONFIDENCE_BOX_V2_EXTERNAL_METADATA_SCHEMA,
        "current": current_metadata,
        "prior": prior_metadata,
    }
    metadata_bytes = _canonical_json_bytes(metadata)
    arrays: dict[str, np.ndarray] = {
        _METADATA_NAME: np.frombuffer(metadata_bytes, dtype=_U8),
        **_array_map_from_receipt(receipt),
    }
    if prior is not None:
        arrays.update(_array_map_from_receipt(prior, prefix=_PRIOR_ARRAY_PREFIX))
    forecast = _forecast(
        route_count=receipt.route_count,
        has_prior=prior is not None,
        metadata_bytes=len(metadata_bytes),
    )
    policies = {
        name: V1ExternalArrayPolicy(
            logical_name=name,
            expected_dtype=(_U8.str if name == _METADATA_NAME else _LE_F64.str),
            expected_shape=(
                (len(metadata_bytes),)
                if name == _METADATA_NAME
                else (receipt.route_count,)
            ),
            allow_nonfinite=False,
        )
        for name in arrays
    }
    if issue_live_attachment:
        external, attachment = publish_external_state_with_attachment(
            root,
            namespace=CONFIDENCE_BOX_V2_EXTERNAL_NAMESPACE,
            state_schema=CONFIDENCE_BOX_V2_EXTERNAL_STATE_SCHEMA,
            arrays=arrays,
            policies=policies,
        )
    else:
        external = publish_external_state(
            root,
            namespace=CONFIDENCE_BOX_V2_EXTERNAL_NAMESPACE,
            state_schema=CONFIDENCE_BOX_V2_EXTERNAL_STATE_SCHEMA,
            arrays=arrays,
            policies=policies,
        )
        attachment = None
    commit_hash = content_hash(
        {
            "schema": CONFIDENCE_BOX_V2_EXTERNAL_DESCRIPTOR_SCHEMA,
            "external_state_descriptor_hash": external.descriptor_hash,
            "receipt_hash": receipt.receipt_hash,
            "prior_receipt_hash": receipt.prior_confidence_box_receipt_hash,
            "prior_statistics_state_hash": receipt.prior_statistics_state_hash,
            "look_index": receipt.look_index,
            "route_count": receipt.route_count,
            "route_axis_hash": receipt.route_axis_hash,
            "memory_forecast": forecast.to_json_dict(),
            "authority_scope": CONFIDENCE_BOX_V2_EXTERNAL_AUTHORITY_SCOPE,
            "proposal_discovery_input_authority": True,
            "fq_certification_authority": False,
            "bce_oracle_authority": False,
            "mask_commit_authority": False,
        },
        prefix=_COMMIT_PREFIX,
    )
    descriptor = V1ConfidenceBoxExternalDescriptor(
        external_state=external,
        receipt_hash=receipt.receipt_hash,
        prior_receipt_hash=receipt.prior_confidence_box_receipt_hash,
        prior_statistics_state_hash=receipt.prior_statistics_state_hash,
        look_index=receipt.look_index,
        route_count=receipt.route_count,
        route_axis_hash=receipt.route_axis_hash,
        memory_forecast=forecast,
        commit_hash=commit_hash,
    )
    if attachment is not None:
        return descriptor, attachment
    return descriptor


def _identities_by_name(
    descriptor: V1ExternalStateDescriptor,
) -> dict[str, V1ExternalArrayIdentity]:
    return {identity.logical_name: identity for identity in descriptor.arrays}


def _assert_same_static_coordinates(
    current: Mapping[str, object],
    prior: Mapping[str, object],
) -> None:
    for name in (
        "envelope_certificate_hash",
        "design_hash",
        "base_serfling_design_hash",
        "campaign_contract_hash",
        "model_identity_hash",
        "query_contract_hash",
        "preprocessing_hash",
        "current_mask_hash",
        "raw_logit_stream_source_hash",
        "sampling_law_hash",
        "proposal_generation",
        "route_count",
        "route_axis_hash",
        "stratum_names",
        "population_stratum_counts",
        "sealed_population_order_hashes",
        "trajectory_alpha_ledger_hash",
        "trajectory_alpha_ledger_generation",
    ):
        if current[name] != prior[name]:
            _refuse(
                V1ConfidenceBoxExternalStateRefusalCode.PRIOR_CONTINUITY_MISMATCH,
                f"prior transition disagrees on {name}",
            )


def _validate_mapped_current(
    *,
    arrays: Mapping[str, np.ndarray],
    route_count: int,
    expected_scratch_bytes: int,
) -> None:
    chunk = min(route_count, _TYPED_VALIDATION_CHUNK_ELEMENTS)
    scratch = chunk * (3 * _LE_F64.itemsize + 2 * _U8.itemsize)
    if expected_scratch_bytes != scratch:
        _refuse(
            V1ConfidenceBoxExternalStateRefusalCode.INVALID_SCHEMA,
            "typed-open scratch ledger does not replay",
        )
    left = np.empty(chunk, dtype=_LE_F64)
    right = np.empty(chunk, dtype=_LE_F64)
    temp = np.empty(chunk, dtype=_LE_F64)
    boolean = np.empty(chunk, dtype=np.bool_)
    boolean2 = np.empty(chunk, dtype=np.bool_)
    for name in _CURRENT_ARRAY_NAMES:
        array = arrays[name]
        if array.dtype != _LE_F64 or array.shape != (route_count,) or array.flags.writeable:
            _refuse(
                V1ConfidenceBoxExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
                f"{name} does not preserve read-only canonical shape",
            )
    for start in range(0, route_count, chunk):
        stop = min(route_count, start + chunk)
        length = stop - start
        work_bool = boolean[:length]
        point = arrays["class_balanced_absolute_point"][start:stop]
        current_lower = arrays["current_class_balanced_absolute_lower"][start:stop]
        current_upper = arrays["current_class_balanced_absolute_upper"][start:stop]
        cumulative_lower = arrays["cumulative_class_balanced_absolute_lower"][start:stop]
        cumulative_upper = arrays["cumulative_class_balanced_absolute_upper"][start:stop]
        for array in (point, current_lower, current_upper, cumulative_lower, cumulative_upper):
            np.isfinite(array, out=work_bool)
            if not bool(np.all(work_bool)):
                _refuse(
                    V1ConfidenceBoxExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
                    "confidence-box arrays must be finite",
                )
            np.less(array, 0.0, out=work_bool)
            if bool(np.any(work_bool)):
                _refuse(
                    V1ConfidenceBoxExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
                    "confidence-box arrays must be nonnegative",
                )
        np.less(point, current_lower, out=work_bool)
        np.greater(point, current_upper, out=boolean2[:length])
        if bool(np.any(work_bool)) or bool(np.any(boolean2[:length])):
            _refuse(
                V1ConfidenceBoxExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
                "current box does not contain its point",
            )
        np.greater(current_lower, current_upper, out=work_bool)
        np.greater(cumulative_lower, cumulative_upper, out=boolean2[:length])
        if bool(np.any(work_bool)) or bool(np.any(boolean2[:length])):
            _refuse(
                V1ConfidenceBoxExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
                "confidence-box interval is empty",
            )
        np.subtract(cumulative_lower, current_lower, out=left[:length])
        np.subtract(current_upper, cumulative_upper, out=right[:length])
        np.less(left[:length], 0.0, out=work_bool)
        np.less(right[:length], 0.0, out=boolean2[:length])
        # keep temp live so scratch ledger honestly accounts three float buffers
        np.maximum(current_lower, cumulative_lower, out=temp[:length])
        if bool(np.any(work_bool)) or bool(np.any(boolean2[:length])):
            _refuse(
                V1ConfidenceBoxExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
                "cumulative box is not a subset of current box",
            )


def _validate_prior_transition(
    *,
    arrays: Mapping[str, np.ndarray],
    current_metadata: Mapping[str, object],
    prior_metadata: Mapping[str, object] | None,
    route_count: int,
) -> None:
    current_payload = cast(Mapping[str, object], current_metadata["payload_without_arrays"])
    look_index = int(current_payload["look_index"])
    if look_index == 1:
        if prior_metadata is not None:
            _refuse(
                V1ConfidenceBoxExternalStateRefusalCode.PRIOR_CONTINUITY_MISMATCH,
                "first confidence-box look unexpectedly has prior arrays",
            )
        if not np.array_equal(
            arrays["current_class_balanced_absolute_lower"],
            arrays["cumulative_class_balanced_absolute_lower"],
        ) or not np.array_equal(
            arrays["current_class_balanced_absolute_upper"],
            arrays["cumulative_class_balanced_absolute_upper"],
        ):
            _refuse(
                V1ConfidenceBoxExternalStateRefusalCode.PRIOR_CONTINUITY_MISMATCH,
                "first confidence-box cumulative interval differs from current",
            )
        return
    if prior_metadata is None:
        _refuse(
            V1ConfidenceBoxExternalStateRefusalCode.PRIOR_CONTINUITY_MISMATCH,
            "later confidence-box look lacks prior metadata",
        )
    prior_payload = cast(Mapping[str, object], prior_metadata["payload_without_arrays"])
    if current_payload["prior_confidence_box_receipt_hash"] != prior_metadata["receipt_hash"]:
        _refuse(
            V1ConfidenceBoxExternalStateRefusalCode.PRIOR_CONTINUITY_MISMATCH,
            "current metadata names another prior receipt",
        )
    if current_payload["prior_statistics_state_hash"] != prior_payload["state_hash"]:
        _refuse(
            V1ConfidenceBoxExternalStateRefusalCode.PRIOR_CONTINUITY_MISMATCH,
            "current metadata names another prior statistics state",
        )
    if int(prior_payload["look_index"]) != look_index - 1:
        _refuse(
            V1ConfidenceBoxExternalStateRefusalCode.PRIOR_CONTINUITY_MISMATCH,
            "prior look is not immediate",
        )
    _assert_same_static_coordinates(current_payload, prior_payload)
    current_counts = cast(tuple[int, int], _pair_of_ints(current_payload["stratum_counts"], name="current counts"))
    prior_counts = cast(tuple[int, int], _pair_of_ints(prior_payload["stratum_counts"], name="prior counts"))
    if any(left < right for left, right in zip(current_counts, prior_counts)) or current_counts == prior_counts:
        _refuse(
            V1ConfidenceBoxExternalStateRefusalCode.PRIOR_CONTINUITY_MISMATCH,
            "sample counts are not strictly nested",
        )
    current_contexts = _nested_string_pair(current_payload["observed_context_ids_by_stratum"], name="current contexts")
    prior_contexts = _nested_string_pair(prior_payload["observed_context_ids_by_stratum"], name="prior contexts")
    current_draws = _nested_string_pair(current_payload["observed_draw_ids_by_stratum"], name="current draws")
    prior_draws = _nested_string_pair(prior_payload["observed_draw_ids_by_stratum"], name="prior draws")
    for stratum in (0, 1):
        count = prior_counts[stratum]
        if current_contexts[stratum][:count] != prior_contexts[stratum] or current_draws[stratum][:count] != prior_draws[stratum]:
            _refuse(
                V1ConfidenceBoxExternalStateRefusalCode.PRIOR_CONTINUITY_MISMATCH,
                "sample ledgers are not nested prefixes",
            )
    chunk = min(route_count, _TYPED_VALIDATION_CHUNK_ELEMENTS)
    lower = np.empty(chunk, dtype=_LE_F64)
    upper = np.empty(chunk, dtype=_LE_F64)
    boolean = np.empty(chunk, dtype=np.bool_)
    for start in range(0, route_count, chunk):
        stop = min(route_count, start + chunk)
        length = stop - start
        np.maximum(
            arrays["current_class_balanced_absolute_lower"][start:stop],
            arrays["prior_cumulative_class_balanced_absolute_lower"][start:stop],
            out=lower[:length],
        )
        np.minimum(
            arrays["current_class_balanced_absolute_upper"][start:stop],
            arrays["prior_cumulative_class_balanced_absolute_upper"][start:stop],
            out=upper[:length],
        )
        np.greater(lower[:length], upper[:length], out=boolean[:length])
        if bool(np.any(boolean[:length])):
            _refuse(
                V1ConfidenceBoxExternalStateRefusalCode.PRIOR_CONTINUITY_MISMATCH,
                "running intersection is empty",
            )
        np.not_equal(
            lower[:length],
            arrays["cumulative_class_balanced_absolute_lower"][start:stop],
            out=boolean[:length],
        )
        if bool(np.any(boolean[:length])):
            _refuse(
                V1ConfidenceBoxExternalStateRefusalCode.PRIOR_CONTINUITY_MISMATCH,
                "lower running intersection does not replay",
            )
        np.not_equal(
            upper[:length],
            arrays["cumulative_class_balanced_absolute_upper"][start:stop],
            out=boolean[:length],
        )
        if bool(np.any(boolean[:length])):
            _refuse(
                V1ConfidenceBoxExternalStateRefusalCode.PRIOR_CONTINUITY_MISMATCH,
                "upper running intersection does not replay",
            )


@dataclass(frozen=True, slots=True)
class V1ConfidenceBoxExternalView:
    descriptor: V1ConfidenceBoxExternalDescriptor
    external_view: V1ExternalStateView
    metadata: Mapping[str, object]

    @property
    def arrays(self) -> Mapping[str, np.ndarray]:
        return self.external_view.arrays

    @property
    def current_arrays(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType(
            {name: self.external_view.arrays[name] for name in _CURRENT_ARRAY_NAMES}
        )

    @property
    def prior_arrays(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType(
            {
                name: self.external_view.arrays[f"{_PRIOR_ARRAY_PREFIX}{name}"]
                for name in _CURRENT_ARRAY_NAMES
                if f"{_PRIOR_ARRAY_PREFIX}{name}" in self.external_view.arrays
            }
        )

    def materialize_current(self) -> V1CorrectedAttributionConfidenceBoxV2Receipt:
        current = cast(Mapping[str, object], self.metadata["current"])
        payload = dict(cast(Mapping[str, object], current["payload_without_arrays"]))
        for name in _CURRENT_ARRAY_NAMES:
            payload[name] = np.asarray(self.external_view.arrays[name])
        result = V1CorrectedAttributionConfidenceBoxV2Receipt(**payload)  # type: ignore[arg-type]
        result.assert_integrity(self.descriptor.receipt_hash)
        return result

    def materialize_prior(self) -> V1CorrectedAttributionConfidenceBoxV2Receipt | None:
        prior = self.metadata["prior"]
        if prior is None:
            return None
        payload = dict(cast(Mapping[str, object], cast(Mapping[str, object], prior)["payload_without_arrays"]))
        for name in _CURRENT_ARRAY_NAMES:
            payload[name] = np.asarray(self.external_view.arrays[f"{_PRIOR_ARRAY_PREFIX}{name}"])
        result = V1CorrectedAttributionConfidenceBoxV2Receipt(**payload)  # type: ignore[arg-type]
        result.assert_integrity(self.descriptor.prior_receipt_hash)
        return result

    def close(self) -> None:
        self.external_view.close()

    def __enter__(self) -> "V1ConfidenceBoxExternalView":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _open_checked(
    root: str | os.PathLike[str],
    descriptor: V1ConfidenceBoxExternalDescriptor,
    *,
    issue_live_attachment: bool,
) -> (
    tuple[V1ConfidenceBoxExternalView, V1ExternalStateLiveAttachment]
    | V1ConfidenceBoxExternalView
):
    if type(descriptor) is not V1ConfidenceBoxExternalDescriptor:
        _refuse(
            V1ConfidenceBoxExternalStateRefusalCode.INVALID_TYPE,
            "open requires the typed confidence-box descriptor",
        )
    descriptor = V1ConfidenceBoxExternalDescriptor.from_json_dict(
        descriptor.to_json_dict()
    )
    if issue_live_attachment:
        external_view, attachment = open_external_state_with_attachment(
            root, descriptor.external_state
        )
    else:
        external_view = open_external_state(root, descriptor.external_state)
        attachment = None
    try:
        metadata = _strict_metadata(external_view.arrays[_METADATA_NAME])
        identities = _identities_by_name(descriptor.external_state)
        current_hash = _receipt_hash_from_identities(
            metadata=cast(Mapping[str, object], metadata["current"]),
            identities=identities,
            prefix=None,
        )
        if current_hash != descriptor.receipt_hash:
            _refuse(
                V1ConfidenceBoxExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
                "external arrays do not replay current confidence-box hash",
            )
        prior_metadata = metadata["prior"]
        if prior_metadata is None:
            if descriptor.prior_receipt_hash is not None:
                _refuse(
                    V1ConfidenceBoxExternalStateRefusalCode.PRIOR_CONTINUITY_MISMATCH,
                    "descriptor names a prior absent from metadata",
                )
        else:
            prior_hash = _receipt_hash_from_identities(
                metadata=cast(Mapping[str, object], prior_metadata),
                identities=identities,
                prefix=_PRIOR_ARRAY_PREFIX,
            )
            if prior_hash != descriptor.prior_receipt_hash:
                _refuse(
                    V1ConfidenceBoxExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
                    "external arrays do not replay prior confidence-box hash",
                )
        current_payload = cast(Mapping[str, object], cast(Mapping[str, object], metadata["current"])["payload_without_arrays"])
        if (
            descriptor.look_index != current_payload["look_index"]
            or descriptor.route_count != current_payload["route_count"]
            or descriptor.route_axis_hash != current_payload["route_axis_hash"]
            or descriptor.prior_statistics_state_hash
            != current_payload["prior_statistics_state_hash"]
        ):
            _refuse(
                V1ConfidenceBoxExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
                "descriptor coordinates differ from current metadata",
            )
        _validate_mapped_current(
            arrays=external_view.arrays,
            route_count=descriptor.route_count,
            expected_scratch_bytes=(
                descriptor.memory_forecast.typed_open_scratch_bytes_upper_bound
            ),
        )
        _validate_prior_transition(
            arrays=external_view.arrays,
            current_metadata=cast(Mapping[str, object], metadata["current"]),
            prior_metadata=(
                None
                if prior_metadata is None
                else cast(Mapping[str, object], prior_metadata)
            ),
            route_count=descriptor.route_count,
        )
    except Exception:
        external_view.close()
        raise
    typed_view = V1ConfidenceBoxExternalView(
        descriptor=descriptor,
        external_view=external_view,
        metadata=MappingProxyType(metadata),
    )
    if attachment is not None:
        return typed_view, attachment
    return typed_view


def open_confidence_box_v2_external_state(
    root: str | os.PathLike[str],
    descriptor: V1ConfidenceBoxExternalDescriptor,
) -> V1ConfidenceBoxExternalView:
    """Open and verify a confidence-box v2 external state."""

    result = _open_checked(root, descriptor, issue_live_attachment=False)
    assert type(result) is V1ConfidenceBoxExternalView
    return result


def open_confidence_box_v2_external_state_with_attachment(
    root: str | os.PathLike[str],
    descriptor: V1ConfidenceBoxExternalDescriptor,
) -> tuple[V1ConfidenceBoxExternalView, V1ExternalStateLiveAttachment]:
    """Open and verify a confidence-box state with a live store attachment."""

    result = _open_checked(root, descriptor, issue_live_attachment=True)
    assert type(result) is tuple
    return result


__all__ = [
    "CONFIDENCE_BOX_V2_EXTERNAL_AUTHORITY_SCOPE",
    "CONFIDENCE_BOX_V2_EXTERNAL_DESCRIPTOR_SCHEMA",
    "CONFIDENCE_BOX_V2_EXTERNAL_MEMORY_FORECAST_SCHEMA",
    "CONFIDENCE_BOX_V2_EXTERNAL_NAMESPACE",
    "CONFIDENCE_BOX_V2_EXTERNAL_STATE_SCHEMA",
    "V1ConfidenceBoxExternalDescriptor",
    "V1ConfidenceBoxExternalMemoryForecast",
    "V1ConfidenceBoxExternalStateRefusal",
    "V1ConfidenceBoxExternalStateRefusalCode",
    "V1ConfidenceBoxExternalView",
    "open_confidence_box_v2_external_state",
    "open_confidence_box_v2_external_state_with_attachment",
    "publish_confidence_box_v2_external_state",
]
