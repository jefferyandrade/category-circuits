"""Durable compare-and-swap publication for sampled-V7 D alpha heads.

The alpha-family registry proves a purely functional ledger transition.  That
is not enough to choose the unique authoritative transition when two workers
resume the same controller run.  This module supplies the missing external
state boundary: one small, locked, atomically replaced HEAD per controller.

Publication is content-addressed and idempotent.  Generation zero may replace
only an absent HEAD; every later generation may replace only the exact ledger
head named by its registry as ``ledger_before_hash``.  An exact replay of the
already-published successor is accepted without changing the file.  A fork,
gap, stale predecessor, malformed file, or symlinked owned path fails closed.

The returned receipt closes durable-head publication only.  It has no
measurement-owner, D-decision, C/K/O, commit, terminal, performance, or Pareto
authority.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
import fcntl
import json
import os
from pathlib import Path
import tempfile
from typing import Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_sampled_v7_d_alpha_family_registry import (
    V1SampledV7DAlphaFamilyRegistryReceipt,
)


SAMPLED_V7_D_ALPHA_HEAD_CAS_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_D_alpha_controller_head_CAS_v2"
)
SAMPLED_V7_D_ALPHA_HEAD_RECORD_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_D_alpha_controller_head_record_v2"
)
SAMPLED_V7_D_ALPHA_HEAD_CAS_AUTHORITY: Final = (
    "durable_unique_sampled_v7_D_alpha_controller_head_publication_only_no_"
    "downstream_owner_D_C_K_O_commit_terminal_performance_or_pareto_authority_v1"
)

_RECEIPT_PREFIX: Final = "partition-free-v1-sampled-v7-D-alpha-head-CAS-"
_RECORD_PREFIX: Final = "partition-free-v1-sampled-v7-D-alpha-head-record-"
_NAMESPACE_PREFIX: Final = "partition-free-v1-sampled-v7-D-alpha-head-namespace-"
_HEAD_NAME: Final = "HEAD.json"
_LOCK_NAME: Final = "HEAD.lock"
_ISSUER: Final = object()


class V1SampledV7DAlphaHeadCASRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    REGISTRY_INTEGRITY = "registry_integrity"
    STORE_INVALID = "store_invalid"
    PUBLICATION_CONFLICT = "publication_conflict"
    HEAD_CONTENT_INVALID = "head_content_invalid"
    DURABILITY_FAILURE = "durability_failure"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SampledV7DAlphaHeadCASRefusal(RuntimeError):
    """Typed fail-closed refusal at the durable alpha-head boundary."""

    def __init__(
        self,
        code: V1SampledV7DAlphaHeadCASRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7DAlphaHeadCASRefusalCode:
            raise TypeError("sampled-V7 D alpha-head refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7DAlphaHeadCASRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7DAlphaHeadCASRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7DAlphaHeadCASRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _refuse(
            V1SampledV7DAlphaHeadCASRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative exact integer",
        )
    return value


def _record_payload(
    *,
    campaign_contract_hash: str,
    controller_run_hash: str,
    proposal_generation: int,
    expected_predecessor_ledger_hash: str,
    ledger_head_hash: str,
    registry_receipt_hash: str,
    alpha_family_plan_hash: str,
    master_registration_hash: str,
    durable_store_identity_hash: str,
    namespace_hash: str,
) -> dict[str, object]:
    return {
        "schema": SAMPLED_V7_D_ALPHA_HEAD_RECORD_SCHEMA,
        "campaign_contract_hash": campaign_contract_hash,
        "controller_run_hash": controller_run_hash,
        "proposal_generation": proposal_generation,
        "expected_predecessor_ledger_hash": expected_predecessor_ledger_hash,
        "ledger_head_hash": ledger_head_hash,
        "registry_receipt_hash": registry_receipt_hash,
        "alpha_family_plan_hash": alpha_family_plan_hash,
        "master_registration_hash": master_registration_hash,
        "durable_store_identity_hash": durable_store_identity_hash,
        "namespace_hash": namespace_hash,
    }


@dataclass(frozen=True, slots=True)
class V1SampledV7DAlphaHeadCASReceipt:
    campaign_contract_hash: str
    controller_run_hash: str
    proposal_generation: int
    expected_predecessor_ledger_hash: str
    published_ledger_head_hash: str
    alpha_family_registry_receipt_hash: str
    alpha_family_plan_hash: str
    master_registration_hash: str
    durable_store_identity_hash: str
    namespace_hash: str
    publication_record_hash: str
    durable_controller_head_closed: bool = True
    downstream_owner_binding_closed: bool = False
    d_decision_authority: bool = False
    c_measurement_authority: bool = False
    calibration_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SAMPLED_V7_D_ALPHA_HEAD_CAS_AUTHORITY
    schema: str = SAMPLED_V7_D_ALPHA_HEAD_CAS_SCHEMA
    _issuer_token: InitVar[object | None] = None
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuer_token: object | None) -> None:
        if _issuer_token is not _ISSUER:
            _refuse(
                V1SampledV7DAlphaHeadCASRefusalCode.INVALID_TYPE,
                "sampled-V7 D alpha-head receipts are issuer-only",
            )
        for name in (
            "campaign_contract_hash",
            "controller_run_hash",
            "expected_predecessor_ledger_hash",
            "published_ledger_head_hash",
            "alpha_family_registry_receipt_hash",
            "alpha_family_plan_hash",
            "master_registration_hash",
            "durable_store_identity_hash",
            "namespace_hash",
            "publication_record_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        positive = (self.durable_controller_head_closed,)
        closed = (
            self.downstream_owner_binding_closed,
            self.d_decision_authority,
            self.c_measurement_authority,
            self.calibration_oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.performance_authority,
            self.pareto_authority,
        )
        if (
            any(type(value) is not bool or value is not True for value in positive)
            or any(type(value) is not bool or value is not False for value in closed)
            or self.authority_scope != SAMPLED_V7_D_ALPHA_HEAD_CAS_AUTHORITY
            or self.schema != SAMPLED_V7_D_ALPHA_HEAD_CAS_SCHEMA
        ):
            _refuse(
                V1SampledV7DAlphaHeadCASRefusalCode.AUTHORITY_VIOLATION,
                "durable alpha-head receipt crossed its authority boundary",
            )
        object.__setattr__(self, "_receipt_hash", self._compute_hash())

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_RECEIPT_PREFIX)

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_hash()
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1SampledV7DAlphaHeadCASRefusalCode.HEAD_CONTENT_INVALID,
                "durable alpha-head receipt changed after issuance",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledV7DAlphaHeadCASRefusalCode.HEAD_CONTENT_INVALID,
                "durable alpha-head receipt hash does not replay",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def require_downstream_owner_binding(self) -> NoReturn:
        raise PermissionError("durable alpha-head publication has no owner binding")

    def require_d_decision(self) -> NoReturn:
        raise PermissionError("durable alpha-head publication has no D authority")

    def require_c_measurement(self) -> NoReturn:
        raise PermissionError("durable alpha-head publication has no C authority")


def _namespace_hash(
    *,
    campaign_contract_hash: str,
    controller_run_hash: str,
    durable_store_identity_hash: str,
) -> str:
    return content_hash(
        {
            "schema": SAMPLED_V7_D_ALPHA_HEAD_RECORD_SCHEMA,
            "campaign_contract_hash": campaign_contract_hash,
            "controller_run_hash": controller_run_hash,
            "durable_store_identity_hash": durable_store_identity_hash,
        },
        prefix=_NAMESPACE_PREFIX,
    )


def _owned_directory(
    root: str | os.PathLike[str],
    *,
    namespace_hash: str,
) -> Path:
    path = Path(root)
    if not path.is_absolute():
        _refuse(
            V1SampledV7DAlphaHeadCASRefusalCode.STORE_INVALID,
            "durable alpha-head root must be absolute",
        )
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise V1SampledV7DAlphaHeadCASRefusal(
            V1SampledV7DAlphaHeadCASRefusalCode.STORE_INVALID,
            f"durable alpha-head root cannot be created: {error}",
        ) from error
    if path.is_symlink() or not path.is_dir():
        _refuse(
            V1SampledV7DAlphaHeadCASRefusalCode.STORE_INVALID,
            "durable alpha-head root must be one real directory",
        )
    owned = path / namespace_hash
    try:
        owned.mkdir(mode=0o700, exist_ok=True)
    except OSError as error:
        raise V1SampledV7DAlphaHeadCASRefusal(
            V1SampledV7DAlphaHeadCASRefusalCode.STORE_INVALID,
            f"controller alpha-head namespace cannot be created: {error}",
        ) from error
    if owned.is_symlink() or not owned.is_dir():
        _refuse(
            V1SampledV7DAlphaHeadCASRefusalCode.STORE_INVALID,
            "controller alpha-head namespace must be one real directory",
        )
    return owned


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_json_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _read_head(path: Path) -> dict[str, object] | None:
    if path.is_symlink():
        _refuse(
            V1SampledV7DAlphaHeadCASRefusalCode.HEAD_CONTENT_INVALID,
            "durable alpha HEAD must not be a symlink",
        )
    if not path.exists():
        return None
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V1SampledV7DAlphaHeadCASRefusal(
            V1SampledV7DAlphaHeadCASRefusalCode.HEAD_CONTENT_INVALID,
            f"durable alpha HEAD is unreadable: {error}",
        ) from error
    expected = {
        "record",
        "publication_record_hash",
        "cas_receipt_hash",
    }
    if type(payload) is not dict or set(payload) != expected:
        _refuse(
            V1SampledV7DAlphaHeadCASRefusalCode.HEAD_CONTENT_INVALID,
            "durable alpha HEAD fields changed",
        )
    record = payload.get("record")
    if type(record) is not dict or set(record) != {
        "schema",
        "campaign_contract_hash",
        "controller_run_hash",
        "proposal_generation",
        "expected_predecessor_ledger_hash",
        "ledger_head_hash",
        "registry_receipt_hash",
        "alpha_family_plan_hash",
        "master_registration_hash",
        "durable_store_identity_hash",
        "namespace_hash",
    }:
        _refuse(
            V1SampledV7DAlphaHeadCASRefusalCode.HEAD_CONTENT_INVALID,
            "durable alpha HEAD record fields changed",
        )
    for name in (
        "campaign_contract_hash",
        "controller_run_hash",
        "expected_predecessor_ledger_hash",
        "ledger_head_hash",
        "registry_receipt_hash",
        "alpha_family_plan_hash",
        "master_registration_hash",
        "durable_store_identity_hash",
        "namespace_hash",
    ):
        _identity(record.get(name), name=f"HEAD {name}")
    _nonnegative_int(record.get("proposal_generation"), name="HEAD generation")
    if record.get("schema") != SAMPLED_V7_D_ALPHA_HEAD_RECORD_SCHEMA:
        _refuse(
            V1SampledV7DAlphaHeadCASRefusalCode.HEAD_CONTENT_INVALID,
            "durable alpha HEAD record schema changed",
        )
    expected_record_hash = content_hash(record, prefix=_RECORD_PREFIX)
    expected_receipt = V1SampledV7DAlphaHeadCASReceipt(
        campaign_contract_hash=record["campaign_contract_hash"],
        controller_run_hash=record["controller_run_hash"],
        proposal_generation=record["proposal_generation"],
        expected_predecessor_ledger_hash=record[
            "expected_predecessor_ledger_hash"
        ],
        published_ledger_head_hash=record["ledger_head_hash"],
        alpha_family_registry_receipt_hash=record["registry_receipt_hash"],
        alpha_family_plan_hash=record["alpha_family_plan_hash"],
        master_registration_hash=record["master_registration_hash"],
        durable_store_identity_hash=record["durable_store_identity_hash"],
        namespace_hash=record["namespace_hash"],
        publication_record_hash=expected_record_hash,
        _issuer_token=_ISSUER,
    )
    if (
        payload.get("publication_record_hash") != expected_record_hash
        or payload.get("cas_receipt_hash") != expected_receipt.receipt_hash
    ):
        _refuse(
            V1SampledV7DAlphaHeadCASRefusalCode.HEAD_CONTENT_INVALID,
            "durable alpha HEAD content address does not replay",
        )
    return payload


def _receipt_for_registry(
    registry: V1SampledV7DAlphaFamilyRegistryReceipt,
    *,
    registry_hash: str,
    durable_store_identity_hash: str,
    namespace_hash: str,
) -> tuple[V1SampledV7DAlphaHeadCASReceipt, dict[str, object]]:
    record = _record_payload(
        campaign_contract_hash=registry.campaign_contract_hash,
        controller_run_hash=registry.controller_run_hash,
        proposal_generation=registry.proposal_generation,
        expected_predecessor_ledger_hash=registry.ledger_before_hash,
        ledger_head_hash=registry.ledger_after_hash,
        registry_receipt_hash=registry_hash,
        alpha_family_plan_hash=registry.alpha_family_plan_hash,
        master_registration_hash=registry.master_registration_hash,
        durable_store_identity_hash=durable_store_identity_hash,
        namespace_hash=namespace_hash,
    )
    record_hash = content_hash(record, prefix=_RECORD_PREFIX)
    receipt = V1SampledV7DAlphaHeadCASReceipt(
        campaign_contract_hash=registry.campaign_contract_hash,
        controller_run_hash=registry.controller_run_hash,
        proposal_generation=registry.proposal_generation,
        expected_predecessor_ledger_hash=registry.ledger_before_hash,
        published_ledger_head_hash=registry.ledger_after_hash,
        alpha_family_registry_receipt_hash=registry_hash,
        alpha_family_plan_hash=registry.alpha_family_plan_hash,
        master_registration_hash=registry.master_registration_hash,
        durable_store_identity_hash=durable_store_identity_hash,
        namespace_hash=namespace_hash,
        publication_record_hash=record_hash,
        _issuer_token=_ISSUER,
    )
    return receipt, {
        "record": record,
        "publication_record_hash": record_hash,
        "cas_receipt_hash": receipt.receipt_hash,
    }


def publish_v1_sampled_v7_d_alpha_head_compare_and_swap(
    *,
    durable_root: str | os.PathLike[str],
    durable_store_identity_hash: str,
    registry: V1SampledV7DAlphaFamilyRegistryReceipt,
) -> V1SampledV7DAlphaHeadCASReceipt:
    """Publish the exact registry successor as the unique controller HEAD."""

    if type(registry) is not V1SampledV7DAlphaFamilyRegistryReceipt:
        _refuse(
            V1SampledV7DAlphaHeadCASRefusalCode.INVALID_TYPE,
            "alpha-head CAS requires the exact sampled-V7 registry type",
        )
    try:
        registry_hash = registry.assert_integrity()
    except Exception as error:
        raise V1SampledV7DAlphaHeadCASRefusal(
            V1SampledV7DAlphaHeadCASRefusalCode.REGISTRY_INTEGRITY,
            f"alpha-family registry failed replay: {type(error).__name__}: {error}",
        ) from error
    store_identity = _identity(
        durable_store_identity_hash,
        name="durable store identity hash",
    )
    namespace_hash = _namespace_hash(
        campaign_contract_hash=registry.campaign_contract_hash,
        controller_run_hash=registry.controller_run_hash,
        durable_store_identity_hash=store_identity,
    )
    directory = _owned_directory(durable_root, namespace_hash=namespace_hash)
    receipt, payload = _receipt_for_registry(
        registry,
        registry_hash=registry_hash,
        durable_store_identity_hash=store_identity,
        namespace_hash=namespace_hash,
    )
    head_path = directory / _HEAD_NAME
    lock_path = directory / _LOCK_NAME
    try:
        with lock_path.open("a+b") as lock:
            if lock_path.is_symlink():
                _refuse(
                    V1SampledV7DAlphaHeadCASRefusalCode.STORE_INVALID,
                    "durable alpha-head lock must not be a symlink",
                )
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                observed = _read_head(head_path)
                if observed is not None:
                    if observed == payload:
                        return receipt
                    record = observed["record"]
                    assert type(record) is dict
                    if (
                        record.get("campaign_contract_hash")
                        != registry.campaign_contract_hash
                        or record.get("controller_run_hash")
                        != registry.controller_run_hash
                        or record.get("durable_store_identity_hash") != store_identity
                        or record.get("namespace_hash") != namespace_hash
                    ):
                        _refuse(
                            V1SampledV7DAlphaHeadCASRefusalCode.PUBLICATION_CONFLICT,
                            "controller alpha HEAD belongs to another registered owner",
                        )
                    if registry.proposal_generation == 0:
                        _refuse(
                            V1SampledV7DAlphaHeadCASRefusalCode.PUBLICATION_CONFLICT,
                            "generation-zero alpha publication found a foreign HEAD",
                        )
                    if (
                        record.get("proposal_generation")
                        != registry.proposal_generation - 1
                        or record.get("ledger_head_hash") != registry.ledger_before_hash
                    ):
                        _refuse(
                            V1SampledV7DAlphaHeadCASRefusalCode.PUBLICATION_CONFLICT,
                            "alpha HEAD is not the registry's exact predecessor",
                        )
                elif registry.proposal_generation != 0:
                    _refuse(
                        V1SampledV7DAlphaHeadCASRefusalCode.PUBLICATION_CONFLICT,
                        "nonzero alpha generation has no durable predecessor HEAD",
                    )

                encoded = _canonical_json_bytes(payload)
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=".HEAD.",
                    suffix=".partial",
                    dir=directory,
                )
                temporary = Path(temporary_name)
                try:
                    with os.fdopen(descriptor, "wb", buffering=0) as stream:
                        stream.write(encoded)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, head_path)
                    _fsync_directory(directory)
                finally:
                    temporary.unlink(missing_ok=True)
                if _read_head(head_path) != payload:
                    _refuse(
                        V1SampledV7DAlphaHeadCASRefusalCode.DURABILITY_FAILURE,
                        "published alpha HEAD failed exact read-after-write replay",
                    )
                return receipt
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    except V1SampledV7DAlphaHeadCASRefusal:
        raise
    except OSError as error:
        raise V1SampledV7DAlphaHeadCASRefusal(
            V1SampledV7DAlphaHeadCASRefusalCode.DURABILITY_FAILURE,
            f"durable alpha-head publication failed: {error}",
        ) from error


__all__ = [
    "SAMPLED_V7_D_ALPHA_HEAD_CAS_AUTHORITY",
    "SAMPLED_V7_D_ALPHA_HEAD_CAS_SCHEMA",
    "SAMPLED_V7_D_ALPHA_HEAD_RECORD_SCHEMA",
    "V1SampledV7DAlphaHeadCASReceipt",
    "V1SampledV7DAlphaHeadCASRefusal",
    "V1SampledV7DAlphaHeadCASRefusalCode",
    "publish_v1_sampled_v7_d_alpha_head_compare_and_swap",
]
