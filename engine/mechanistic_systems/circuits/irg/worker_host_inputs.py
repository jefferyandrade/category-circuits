"""CPU launch preparation transferred from a worker helper to its GPU thread.

Only input contracts and training metadata enter this module. It never opens
terminal response data, starts a logical sampling stream, changes the process
environment, or constructs a model/device runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np

from circuitflow.artifacts import ArtifactType
from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.experimental_design.sampling_identity import (
    binary_evidence_role_namespace,
    compile_binary_iid_sampling_design,
)
from .precision import IndependentSampleSizePolicy


@dataclass(frozen=True, slots=True)
class PreparedPilotHostInputs:
    """Owned launch inputs; the preparation thread relinquishes them on return.

    Contract mappings and historical arrays are read-only. Sampling designs
    are freshly compiled per item and retain their ordinary internal stream
    cursor state, which is first used exclusively by the receiving GPU thread.
    """

    query_id: str
    optimization_axis_hash: str
    random_seed: int
    comparison_namespace: str
    partition_family: str
    model_transform: Any | None
    sealed_queries: tuple[Any, ...]
    target_class_id: int
    shared_run_id: str
    decision_namespace: str
    decision_sampling: Any
    decision_oracle_sampling: Any
    decision_oracle_sampling_receipts: Mapping[str, Any]
    historical_mask_payload: Mapping[str, Any] | None
    historical_mask_sha256: str | None
    v0_schedule_replay_schedule: Mapping[str, Any] | None
    v0_schedule_replay_spec_receipt: Mapping[str, Any] | None


class _FrozenDict(dict):
    """Read-only mapping that existing JSON receipt serializers can consume."""

    def _refuse(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("prepared host mappings are read-only")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _refuse


class _FrozenList(list):
    """Preserve list contracts and JSON encodings while refusing mutation."""

    def _refuse(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("prepared host sequences are read-only")

    __setitem__ = __delitem__ = append = extend = insert = pop = remove = reverse = sort = clear = __iadd__ = __imul__ = _refuse


def _freeze_host_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _FrozenDict({key: _freeze_host_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList(_freeze_host_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_host_value(item) for item in value)
    if isinstance(value, np.ndarray):
        # A bytes owner prevents consumers from re-enabling WRITEABLE.
        return np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)
    if hasattr(value, "detach"):
        if value.device.type != "cpu":
            raise ValueError("host mask preparation requires CPU tensors")
        return _freeze_host_value(value.detach().numpy())
    return value


def _freeze_contract(value: Any) -> Any:
    """Copy newly loaded frozen contract mappings without changing their values."""

    if value is None or not is_dataclass(value):
        return value
    result = replace(value)
    for field in fields(value):
        item = getattr(result, field.name)
        if field.init and isinstance(item, (Mapping, tuple, list)):
            object.__setattr__(result, field.name, _freeze_host_value(item))
    return result


def _freeze_sampling_metadata(design: Any) -> Any:
    # The constructors already validated and hashed these immutable values.
    # Preserve that hash and the WOR design's empty, item-owned cursor cache.
    for name in ("source_context_ids_by_stratum", "stratum_weights"):
        object.__setattr__(design, name, _FrozenDict(getattr(design, name)))
    return design


def prepare_pilot_host_inputs(
    kwargs: Mapping[str, Any], environment: Mapping[str, str],
) -> PreparedPilotHostInputs:
    """Prepare an item's contracts, CPU masks and full training sampling plans.

    ``environment`` is the worker's private item mapping. The argument builder
    has already resolved environment-dependent options into ``kwargs``; this
    function reads no process-global environment and performs no CUDA work.
    """

    from mechanistic_systems.circuits.runner import _measurement_queries
    from .registered_resnet18_pilot import (
        _a47_sealed_queries,
        _decision_oracle_sampling_design,
        _load_pilot_model_transform,
        _oracle_sampling_receipt_strings,
        _registered_precision_policy,
        _target_class_id,
    )
    from .workflow_identity import resolve_current_workflow_identity

    version = resolve_current_workflow_identity(str(kwargs["version_name"])).canonical_identity.rsplit("/", 1)[-1]
    study = kwargs["study"]
    model_transform = _freeze_contract(_load_pilot_model_transform(
        kwargs.get("model_transform_path"), study=study,
    ))
    policy = str(kwargs.get("measurement_artifact_policy", "required"))
    queries = _a47_sealed_queries(policy, kwargs.get("measurement_plan_path"))
    if queries is None:
        if policy != "required":
            raise ValueError("measurement_artifact_policy must be 'required' or 'contracts_only'")
        store, plan = kwargs["store"], kwargs["plan"]
        manifest = store.load(plan.measurement_artifact_hash, ArtifactType.MEASUREMENT)
        if manifest.study_hash != study.content_hash:
            raise ValueError("measurement parent and StudySpec differ")
        queries = _measurement_queries(store, manifest.plan_hash)
    sealed_queries = tuple(_freeze_contract(query) for query in queries)
    query_id = str(kwargs["query_id"])
    query_by_id = {query.query_id: query for query in sealed_queries}
    if query_id not in query_by_id:
        raise ValueError("pilot query is outside the measurement artifact")
    target_class_id = _target_class_id(query_by_id[query_id])
    optimization_axis = kwargs["optimization_axis"]
    random_seed = int(kwargs.get("random_seed", 0))
    shared_run_id = content_hash({
        "comparison_namespace": str(kwargs["comparison_namespace"]),
        "query_id": query_id,
        "optimization_axis_hash": optimization_axis.content_hash,
        "random_seed": random_seed,
    }, prefix="irg-matched-pilot-")
    decision_namespace = binary_evidence_role_namespace(
        run_id=shared_run_id, random_seed=random_seed,
        context_axis_hash=optimization_axis.content_hash,
    )
    decision_sampling = _freeze_sampling_metadata(compile_binary_iid_sampling_design(
        optimization_axis, target_class_id=target_class_id,
        randomization_namespace=decision_namespace,
    ))
    precision_policy = _registered_precision_policy(
        version, kwargs.get("sample_size_policy", IndependentSampleSizePolicy()),
        oracle_sampling_law_override=kwargs.get("v0_oracle_sampling_law_override"),
    )
    decision_oracle_sampling = _freeze_sampling_metadata(_decision_oracle_sampling_design(
        optimization_axis, target_class_id=target_class_id,
        randomization_namespace=f"{decision_namespace}:class-resolved-oracle",
        oracle_sampling_law=str(precision_policy.oracle_sampling_law),
    ))

    historical_payload = None
    historical_sha256 = None
    historical_path = kwargs.get("historical_mask_path")
    if historical_path is not None:
        import torch

        # Read once so the supplied source digest binds exactly the decoded
        # input bytes, including when a storage replica is replaced concurrently.
        source_bytes = Path(historical_path).read_bytes()
        historical_payload = torch.load(io.BytesIO(source_bytes), map_location="cpu", weights_only=True)
        if not isinstance(historical_payload, Mapping):
            raise TypeError("historical mask artifact must be a layer mapping")
        historical_payload = _freeze_host_value(historical_payload)
        historical_sha256 = hashlib.sha256(source_bytes).hexdigest()

    replay_schedule = replay_receipt = None
    if kwargs.get("v0_schedule_replay_spec") is not None:
        from .schedule_replay import load_schedule_replay_spec

        replay_schedule, replay_receipt = load_schedule_replay_spec(kwargs["v0_schedule_replay_spec"])
        replay_schedule = _freeze_host_value(replay_schedule)
        replay_receipt = _freeze_host_value(replay_receipt)

    return PreparedPilotHostInputs(
        query_id=query_id, optimization_axis_hash=str(optimization_axis.content_hash),
        random_seed=random_seed, comparison_namespace=str(kwargs["comparison_namespace"]),
        partition_family=str(kwargs["partition_family"]),
        model_transform=model_transform, sealed_queries=sealed_queries,
        target_class_id=target_class_id, shared_run_id=shared_run_id,
        decision_namespace=decision_namespace, decision_sampling=decision_sampling,
        decision_oracle_sampling=decision_oracle_sampling,
        decision_oracle_sampling_receipts=_FrozenDict(_oracle_sampling_receipt_strings(decision_oracle_sampling)),
        historical_mask_payload=historical_payload, historical_mask_sha256=historical_sha256,
        v0_schedule_replay_schedule=replay_schedule,
        v0_schedule_replay_spec_receipt=replay_receipt,
    )
