"""Typed runtime replicas for registered scientific model sources.

Scientific model identity lives in :class:`StudySpec` and the resulting
``ModelLoadReceipt``.  This module only describes where an executable replica
can be found.  Replica paths are deliberately excluded from ``binding_hash``;
their declared content hashes and the loaded Study identity are not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from contracts.hashing import sha256_file, stable_digest
from mechanistic_systems.contracts import StudySpec

from .models import (
    MODEL_REGISTRY,
    LoadedRegisteredModel,
    load_registered_model_for_study,
    load_registered_model_source,
)


MODEL_SOURCE_REPLICA_SCHEMA = "mechanistic_model_source_replica_v1"
MODEL_SOURCE_KINDS = frozenset({"registered_model", "ssl_fixed_query_readout"})


@dataclass(frozen=True)
class ModelSourceReplica:
    """Runtime locations plus immutable component hashes for one model replica."""

    source_kind: str
    checkpoint_path: str = ""
    target_manifest_path: str = ""
    direction_bank_path: str = ""
    cell_id: str = ""
    target_family: str = ""
    target_layer: str = ""
    source_class_ids: tuple[int, ...] = tuple()
    checkpoint_file_hash: str = ""
    target_manifest_hash: str = ""
    direction_bank_file_hash: str = ""
    direction_bank_hash: str = ""

    def validate(self, *, claim_bearing: bool = True) -> None:
        if self.source_kind not in MODEL_SOURCE_KINDS:
            raise ValueError(f"unsupported model source replica {self.source_kind!r}")
        if self.source_kind == "registered_model":
            forbidden = (
                self.target_manifest_path,
                self.direction_bank_path,
                self.cell_id,
                self.target_family,
                self.target_layer,
                self.source_class_ids,
                self.target_manifest_hash,
                self.direction_bank_file_hash,
                self.direction_bank_hash,
            )
            if any(bool(value) for value in forbidden):
                raise ValueError("registered-model replica contains SSL readout fields")
            if self.checkpoint_file_hash and (
                len(self.checkpoint_file_hash) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in self.checkpoint_file_hash
                )
            ):
                raise ValueError("registered checkpoint file hash is invalid")
            if self.checkpoint_path and not self.checkpoint_file_hash:
                raise ValueError(
                    "registered checkpoint path requires an exact file hash"
                )
            if claim_bearing and self.checkpoint_path and not Path(
                self.checkpoint_path
            ).is_absolute():
                raise ValueError(
                    "claim-bearing registered checkpoint path must be absolute"
                )
            return

        required_paths = {
            "checkpoint_path": self.checkpoint_path,
            "target_manifest_path": self.target_manifest_path,
        }
        missing_paths = sorted(name for name, value in required_paths.items() if not value)
        if missing_paths:
            raise ValueError("SSL model replica is missing " + ",".join(missing_paths))
        if not self.cell_id:
            raise ValueError("SSL model replica requires a stable cell_id")
        if self.source_class_ids and len(set(self.source_class_ids)) != len(
            self.source_class_ids
        ):
            raise ValueError("SSL source class IDs must be unique")
        hashes = {
            "checkpoint_file_hash": self.checkpoint_file_hash,
            "target_manifest_hash": self.target_manifest_hash,
            "direction_bank_file_hash": self.direction_bank_file_hash,
            "direction_bank_hash": self.direction_bank_hash,
        }
        invalid = sorted(name for name, value in hashes.items() if value and len(value) != 64)
        if invalid:
            raise ValueError("model source replica has invalid hashes: " + ",".join(invalid))
        if claim_bearing:
            missing = sorted(name for name, value in hashes.items() if not value)
            if missing:
                raise ValueError(
                    "claim-bearing SSL replicas require component hashes: "
                    + ",".join(missing)
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": MODEL_SOURCE_REPLICA_SCHEMA,
            "source_kind": self.source_kind,
            "checkpoint_path": self.checkpoint_path,
            "target_manifest_path": self.target_manifest_path,
            "direction_bank_path": self.direction_bank_path,
            "cell_id": self.cell_id,
            "target_family": self.target_family,
            "target_layer": self.target_layer,
            "source_class_ids": list(self.source_class_ids),
            "checkpoint_file_hash": self.checkpoint_file_hash,
            "target_manifest_hash": self.target_manifest_hash,
            "direction_bank_file_hash": self.direction_bank_file_hash,
            "direction_bank_hash": self.direction_bank_hash,
        }

    @property
    def binding_hash(self) -> str:
        """Hash scientific source binding while ignoring replica locations."""

        self.validate(claim_bearing=False)
        return stable_digest(
            {
                "schema_name": MODEL_SOURCE_REPLICA_SCHEMA,
                "source_kind": self.source_kind,
                "cell_id": self.cell_id,
                "target_family": self.target_family,
                "target_layer": self.target_layer,
                "source_class_ids": list(self.source_class_ids),
                "checkpoint_file_hash": self.checkpoint_file_hash,
                "target_manifest_hash": self.target_manifest_hash,
                "direction_bank_file_hash": self.direction_bank_file_hash,
                "direction_bank_hash": self.direction_bank_hash,
            }
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ModelSourceReplica":
        if str(payload.get("schema_name", "")) != MODEL_SOURCE_REPLICA_SCHEMA:
            raise ValueError("unsupported model source replica schema")
        result = cls(
            source_kind=str(payload.get("source_kind", "")),
            checkpoint_path=str(payload.get("checkpoint_path", "")),
            target_manifest_path=str(payload.get("target_manifest_path", "")),
            direction_bank_path=str(payload.get("direction_bank_path", "")),
            cell_id=str(payload.get("cell_id", "")),
            target_family=str(payload.get("target_family", "")),
            target_layer=str(payload.get("target_layer", "")),
            source_class_ids=tuple(
                int(value) for value in payload.get("source_class_ids", ()) or ()
            ),
            checkpoint_file_hash=str(payload.get("checkpoint_file_hash", "")),
            target_manifest_hash=str(payload.get("target_manifest_hash", "")),
            direction_bank_file_hash=str(payload.get("direction_bank_file_hash", "")),
            direction_bank_hash=str(payload.get("direction_bank_hash", "")),
        )
        result.validate(claim_bearing=False)
        return result

    @classmethod
    def load(cls, path: str | Path) -> "ModelSourceReplica":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("model source replica must contain a JSON object")
        return cls.from_mapping(payload)

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination


def _validate_ssl_component_identity(
    replica: ModelSourceReplica,
    *,
    bank: Any,
) -> dict[str, str]:
    observed = {
        "checkpoint_file_hash": sha256_file(Path(replica.checkpoint_path)),
        "target_manifest_hash": bank.manifest_hash,
        "direction_bank_file_hash": bank.bank_hash,
        "direction_bank_hash": bank.content_hash,
    }
    expected = {
        "checkpoint_file_hash": replica.checkpoint_file_hash,
        "target_manifest_hash": replica.target_manifest_hash,
        "direction_bank_file_hash": replica.direction_bank_file_hash,
        "direction_bank_hash": replica.direction_bank_hash,
    }
    mismatches = sorted(
        name for name, value in expected.items() if value and value != observed[name]
    )
    if mismatches:
        raise ValueError("SSL model replica differs from its declared identity: " + ",".join(mismatches))
    return observed


def _validate_study_source_binding(
    study: StudySpec,
    replica: ModelSourceReplica,
    *,
    observed: Mapping[str, str],
) -> None:
    mismatches: list[str] = []
    notes = dict(study.notes)
    for name in ("target_manifest_hash", "direction_bank_file_hash", "direction_bank_hash"):
        declared = str(notes.get(name, ""))
        if declared and declared != observed[name]:
            mismatches.append(f"study_{name}")
    source_kind = str(notes.get("model_source_kind", ""))
    if source_kind and source_kind != "ssl_fixed_query_readout":
        mismatches.append("study_model_source_kind")
    binding_hash = str(notes.get("model_source_binding_hash", ""))
    if binding_hash and binding_hash != replica.binding_hash:
        mismatches.append("study_model_source_binding_hash")
    if mismatches:
        raise ValueError("SSL model replica differs from its Study binding: " + ",".join(mismatches))


def load_model_replica(
    model_id: str,
    *,
    source_replica: ModelSourceReplica | None = None,
    checkpoint_path: str | Path | None = None,
    random_seed: int = 0,
    allow_download: bool = False,
    device: Any = "cpu",
    claim_bearing: bool = True,
    model_transform: Any | None = None,
) -> LoadedRegisteredModel:
    """Load one registered model replica before binding it to a StudySpec.

    ``model_transform`` is forwarded to the registered loader,
    which re-derives and verifies the transformed state.  SSL replicas refuse
    it.  When it is ``None`` every call is exactly the untransformed call.
    """

    if model_id in MODEL_REGISTRY:
        checkpoint_file_hash: str | None = None
        if source_replica is not None:
            source_replica.validate(claim_bearing=claim_bearing)
            if source_replica.source_kind != "registered_model":
                raise ValueError("supervised model received an SSL source replica")
            if checkpoint_path is not None and source_replica.checkpoint_path:
                if Path(checkpoint_path).resolve() != Path(
                    source_replica.checkpoint_path
                ).resolve():
                    raise ValueError("conflicting registered checkpoint replicas")
            checkpoint_path = checkpoint_path or source_replica.checkpoint_path or None
            checkpoint_file_hash = source_replica.checkpoint_file_hash or None
            if source_replica.checkpoint_file_hash and checkpoint_path is not None:
                if sha256_file(Path(checkpoint_path)) != source_replica.checkpoint_file_hash:
                    raise ValueError("registered checkpoint replica hash mismatch")
        return load_registered_model_source(
            model_id,
            checkpoint_path=checkpoint_path,
            checkpoint_file_hash=checkpoint_file_hash,
            random_seed=int(random_seed),
            allow_download=allow_download,
            device=device,
            **(
                {}
                if model_transform is None
                else {"model_transform": model_transform}
            ),
        )

    if model_transform is not None:
        raise ValueError("SSL model replicas do not admit a model transform")
    from .ssl_adapter import load_ssl_direction_bank
    from .ssl_models import SSL_BACKBONE_REGISTRY, load_ssl_query_model

    if model_id not in SSL_BACKBONE_REGISTRY:
        raise ValueError(f"unregistered canonical model {model_id!r}")
    if source_replica is None:
        raise ValueError("SSL model execution requires a ModelSourceReplica")
    source_replica.validate(claim_bearing=claim_bearing)
    if source_replica.source_kind != "ssl_fixed_query_readout":
        raise ValueError("SSL model requires an ssl_fixed_query_readout replica")
    if checkpoint_path is not None and Path(checkpoint_path).resolve() != Path(
        source_replica.checkpoint_path
    ).resolve():
        raise ValueError("conflicting SSL checkpoint replicas")
    definition = SSL_BACKBONE_REGISTRY[model_id]
    bank = load_ssl_direction_bank(
        source_replica.target_manifest_path,
        cell_id=source_replica.cell_id,
        architecture_family=definition.architecture_family,
        target_family=source_replica.target_family or None,
        target_layer=source_replica.target_layer or definition.fixed_readout_layer,
        bank_path=source_replica.direction_bank_path or None,
        source_class_ids=source_replica.source_class_ids or None,
    )
    _validate_ssl_component_identity(source_replica, bank=bank)
    return load_ssl_query_model(
        model_id,
        checkpoint_path=source_replica.checkpoint_path,
        direction_bank=bank,
        device=device,
    )


def load_model_for_study(
    study: StudySpec,
    *,
    source_replica: ModelSourceReplica | None = None,
    checkpoint_path: str | Path | None = None,
    allow_download: bool = False,
    device: Any = "cpu",
    claim_bearing: bool = True,
    model_transform: Any | None = None,
) -> LoadedRegisteredModel:
    """Resolve any canonical supervised or fixed-readout SSL model Study."""

    transform_kwargs: dict[str, Any] = (
        {} if model_transform is None else {"model_transform": model_transform}
    )
    if study.model_id in MODEL_REGISTRY:
        # Preserve the existing registered-source restoration path, including
        # seeded-untrained claim checks and official-weight cache semantics.
        if source_replica is None:
            return load_registered_model_for_study(
                study,
                checkpoint_path=checkpoint_path,
                checkpoint_file_hash=None,
                allow_download=allow_download,
                device=device,
                claim_bearing=claim_bearing,
                **transform_kwargs,
            )
    loaded = load_model_replica(
        study.model_id,
        source_replica=source_replica,
        checkpoint_path=checkpoint_path,
        random_seed=int(study.seeds.get("model_init", 0)),
        allow_download=allow_download,
        device=device,
        claim_bearing=claim_bearing,
        **transform_kwargs,
    )
    if source_replica is not None and source_replica.source_kind == "ssl_fixed_query_readout":
        observed = {
            "checkpoint_file_hash": sha256_file(Path(source_replica.checkpoint_path)),
            "target_manifest_hash": source_replica.target_manifest_hash,
            "direction_bank_file_hash": source_replica.direction_bank_file_hash,
            "direction_bank_hash": source_replica.direction_bank_hash,
        }
        _validate_study_source_binding(study, source_replica, observed=observed)
    loaded.validate_against_study(study, claim_bearing=claim_bearing)
    return loaded


__all__ = [
    "MODEL_SOURCE_KINDS",
    "MODEL_SOURCE_REPLICA_SCHEMA",
    "ModelSourceReplica",
    "load_model_replica",
    "load_model_for_study",
]
