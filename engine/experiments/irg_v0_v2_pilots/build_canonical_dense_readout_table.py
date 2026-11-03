#!/usr/bin/env python3
"""Build a canonical full-train, full-width dense-readout table."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch


VALUES_FILENAME = "canonical_dense_readout_table_values.npy"
IDS_FILENAME = "canonical_dense_readout_table_ids.npz"
RECEIPT_FILENAME = "canonical_dense_readout_table_receipt.json"
_REQUIRED_IDENTITY_FIELDS = (
    "model_state_hash",
    "route_schema_hash",
    "topology_hash",
    "dense_mask_hash",
    "context_axis_hash",
    "split",
    "forward_tier",
    "torch_version",
    "cuda_version",
    "device_capability",
)


@dataclass(frozen=True)
class CanonicalDenseReadoutTableHandle:
    """Read-only mmap-backed canonical dense-readout table."""

    values: np.ndarray
    source_context_ids: tuple[str, ...]
    labels: np.ndarray
    index_by_context_id: Mapping[str, int]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# --- Parity tolerance provenance --------------------------------------------
#
# The self-check deliberately recomputes rows at a DIFFERENT batch size from
# the build (``_validate_build_inputs`` requires it). That is what lets it
# catch the faults it exists to catch -- row misalignment, stale substitution,
# dtype/layout error -- all of which produce O(1) differences.
#
# But a different batch size means a different panel batching and therefore a
# different fp32 GEMM reduction order, and fp32 accumulation is NOT
# batch-shape invariant. So the check can never legitimately demand bitwise
# agreement, and its threshold must be set from a MEASUREMENT of that
# unavoidable spread rather than from an aspiration. Use
# ``--characterize-reduction-order`` to measure it on samples disjoint from
# the acceptance sample, then pin the constant below and record the basis.
#
# Estimand impact of a residual epsilon on dense logits is bounded and small:
# log-sum-exp is 1-Lipschitz in sup-norm, so the substituted families' offset
# c_l(x) moves by at most epsilon, and p = sigmoid(...) by at most 0.25 *
# epsilon. Additionally, all consumers bind to ONE shared table, so a fixed
# offset from an exact computation is common to every arm and cancels in every
# cross-arm comparison the program makes.
# The registered acceptance POLICY, in place of a magic threshold:
#
#   tolerance = clamp(MARGIN * measured_disjoint_spread, floor=FLOOR)
#   refuse outright if that exceeds CEILING
#
# FLOOR keeps a well-behaved run from tightening below the historical bound.
# MARGIN covers sample-to-sample variation in the spread.
#
# CEILING is the safety catch, and it is DERIVED rather than anchored to an
# observation. fp32 carries a 24-bit mantissa, so one ulp on logits of
# magnitude ~10-30 is ~1-2e-6; recombining a 4096-dim reduction in a different
# order accumulates realistically like sqrt(n)*ulp ~ 64 * 2e-6 ~ 1.3e-4. So
# spreads up to ~1e-4 are consistent with reduction-order variation alone.
# TF32 carries a 10-bit mantissa (~1e-3 relative, ~1e-2 absolute here), an
# order above that. CEILING=1e-3 therefore sits ~8x above the analytic fp32
# bound and ~10x below the TF32 floor -- wide enough not to refuse a healthy
# build, tight enough that an execution-policy fault still refuses. (An
# earlier 2e-4 was 4x a single observation, only 1.5x the analytic bound, and
# would have risked refusing healthy runs.)
#
# Corruption-class faults -- row misalignment, stale substitution, dtype or
# layout error -- shift values by O(1) and remain orders outside CEILING, so
# the check keeps the discriminating power it exists for.
PARITY_TOLERANCE = 1e-5
PARITY_TOLERANCE_BASIS = "historical_unmeasured_1e-5"
PARITY_TOLERANCE_FLOOR = 1e-5
PARITY_TOLERANCE_MARGIN = 4.0
PARITY_TOLERANCE_CEILING = 1e-3


def parity_tolerance_from_spread(
    measured_max_abs_diff: float,
) -> tuple[float, str]:
    """Apply the registered acceptance policy to a measured spread."""

    scaled = float(PARITY_TOLERANCE_MARGIN) * float(measured_max_abs_diff)
    tolerance = max(float(PARITY_TOLERANCE_FLOOR), scaled)
    if tolerance > float(PARITY_TOLERANCE_CEILING):
        raise RuntimeError(
            "measured fp32 reduction-order spread implies a parity tolerance "
            f"of {tolerance}, above the registered ceiling "
            f"{PARITY_TOLERANCE_CEILING}. A spread this large is an "
            "execution-policy fault (check that TF32 is disabled on both "
            "legs), not reduction-order noise; refusing rather than widening."
        )
    return tolerance, (
        "measured_disjoint_spread"
        f"={measured_max_abs_diff:.6g}"
        f"*margin={PARITY_TOLERANCE_MARGIN:g}"
        f",floor={PARITY_TOLERANCE_FLOOR:g}"
        f",ceiling={PARITY_TOLERANCE_CEILING:g}"
    )


def seeded_probe_positions(
    *,
    source_count: int,
    seed_material: str,
    checked: int,
) -> np.ndarray:
    """The deterministic acceptance-probe positions for a given seed."""

    generator = np.random.default_rng(
        int(str(seed_material).rsplit("-", 1)[-1][:16], 16)
    )
    return np.sort(
        generator.choice(
            int(source_count),
            size=int(checked),
            replace=False,
        )
    )


def characterize_reduction_order_spread(
    *,
    source_context_ids: Sequence[str],
    forward_rows: Callable[..., np.ndarray],
    build_batch_size: int,
    probe_batch_size: int,
    seed_material: str,
    sample_count: int,
    rows_per_sample: int = 32,
) -> Mapping[str, Any]:
    """Measure the build-vs-probe fp32 reduction-order spread.

    This is a MEASUREMENT, not an acceptance decision.

    It must reproduce the SHAPE ASYMMETRY the acceptance probe actually faces,
    which is not "the same rows at two nominal batch sizes". Requesting only
    ``rows_per_sample`` rows yields a single small batch through EITHER source,
    so the nominal batch size is not exercised and the legs agree bitwise --
    measuring 0 and telling us nothing. The real asymmetry is that stored rows
    were computed inside full ``build_batch_size`` chunks by the build loop,
    while the probe recomputes a small scattered subset. So the reference leg
    here forwards a whole build-size chunk, exactly as the build does, and the
    probe leg recomputes a subset of that chunk on its own.

    Non-circularity: the reference leg must forward its entire chunk (the build
    works in whole chunks), but only rows OUTSIDE the acceptance probe's seeded
    positions are ever COMPARED. A tolerance pinned from this output is
    therefore not justified by the rows it will later be asked to accept.
    """

    total = len(source_context_ids)
    reserved = set(
        int(position)
        for position in seeded_probe_positions(
            source_count=total,
            seed_material=seed_material,
            checked=min(32, total),
        )
    )
    generator = np.random.default_rng(
        int(str(seed_material).rsplit("-", 1)[-1][:16], 16) ^ 0x5F5F5F5F
    )
    chunk_width = max(1, int(build_batch_size))
    chunk_count = max(1, (total + chunk_width - 1) // chunk_width)
    per_sample: list[float] = []
    attempts = 0
    while len(per_sample) < int(sample_count) and attempts < 8 * int(
        sample_count
    ) + 8:
        attempts += 1
        start = int(generator.integers(chunk_count)) * chunk_width
        stop = min(start + chunk_width, total)
        comparable = [
            index for index in range(start, stop) if index not in reserved
        ]
        if len(comparable) < int(rows_per_sample):
            continue
        chunk_ids = tuple(
            str(source_context_ids[index]) for index in range(start, stop)
        )
        # Reference leg: one whole build-size chunk, as the build loop writes it.
        reference = np.asarray(
            forward_rows(chunk_ids, batch_size=int(build_batch_size)),
            dtype=np.float32,
        )
        subset = np.sort(
            generator.choice(
                np.asarray(comparable, dtype=np.int64),
                size=int(rows_per_sample),
                replace=False,
            )
        )
        subset_ids = tuple(
            str(source_context_ids[int(index)]) for index in subset
        )
        # Probe leg: the same rows recomputed as their own small batch.
        probe = np.asarray(
            forward_rows(subset_ids, batch_size=int(probe_batch_size)),
            dtype=np.float32,
        )
        rows_in_chunk = [int(index) - start for index in subset]
        per_sample.append(
            float(
                np.max(
                    np.abs(reference[rows_in_chunk] - probe),
                    initial=0.0,
                )
            )
        )
    return {
        "schema": "canonical_dense_readout_reduction_order_spread_v2",
        "samples": len(per_sample),
        "rows_per_sample": int(rows_per_sample),
        "build_batch_size": int(build_batch_size),
        "probe_batch_size": int(probe_batch_size),
        "reference_leg": "whole_build_size_chunk",
        "probe_leg": "scattered_subset_own_batch",
        "per_sample_max_abs_diff": per_sample,
        "max_abs_diff": max(per_sample) if per_sample else 0.0,
        "mean_max_abs_diff": (
            float(sum(per_sample) / len(per_sample)) if per_sample else 0.0
        ),
        "acceptance_positions_excluded_from_comparison": len(reserved),
    }


def run_parity_self_check(
    *,
    source_context_ids: Sequence[str],
    values_path: Path,
    forward_rows: Callable[..., np.ndarray],
    probe_batch_size: int,
    seed_material: str,
    tolerance: float,
    tolerance_basis: str,
) -> Mapping[str, Any]:
    """Compare a deterministic sample of stored rows with fresh fp32 rows."""

    checked = min(32, len(source_context_ids))
    if checked == 0:
        return {
            "checked": 0,
            "max_abs_diff": 0.0,
            "accepted": True,
            "probe_batch_size": int(probe_batch_size),
            "tolerance": float(tolerance),
            "tolerance_basis": str(tolerance_basis),
        }
    positions = seeded_probe_positions(
        source_count=len(source_context_ids),
        seed_material=seed_material,
        checked=checked,
    )
    probe_ids = tuple(source_context_ids[int(index)] for index in positions)
    live_values = np.asarray(
        forward_rows(probe_ids, batch_size=int(probe_batch_size)),
        dtype=np.float32,
    )
    values = np.load(values_path, mmap_mode="r")
    if values.ndim != 2:
        raise ValueError("canonical dense-readout values must be rank two")
    table_values = values[positions]
    if live_values.shape != table_values.shape:
        raise ValueError(
            "parity forward rows have shape "
            f"{live_values.shape}, expected {table_values.shape}"
        )
    max_abs_diff = float(
        np.max(np.abs(table_values - live_values), initial=0.0)
    )
    del table_values
    del values
    return {
        "checked": int(checked),
        "max_abs_diff": max_abs_diff,
        "accepted": bool(max_abs_diff <= float(tolerance)),
        "probe_batch_size": int(probe_batch_size),
        "tolerance": float(tolerance),
        "tolerance_basis": str(tolerance_basis),
    }


def _validate_build_inputs(
    *,
    source_context_ids: Sequence[str],
    labels: Sequence[int],
    width: int,
    build_batch_size: int,
    probe_batch_size: int,
    identity_fields: Mapping[str, Any],
) -> tuple[str, ...]:
    source_ids = tuple(str(value) for value in source_context_ids)
    if not source_ids or len(set(source_ids)) != len(source_ids):
        raise ValueError("source_context_ids must be nonempty and unique")
    if len(labels) != len(source_ids):
        raise ValueError("labels must have one value per source context")
    if int(width) <= 0:
        raise ValueError("width must be positive")
    if int(build_batch_size) <= 0:
        raise ValueError("build_batch_size must be positive")
    if int(probe_batch_size) <= 0:
        raise ValueError("probe_batch_size must be positive")
    if int(probe_batch_size) == int(build_batch_size):
        raise ValueError("probe_batch_size must differ from build_batch_size")
    missing = tuple(
        field for field in _REQUIRED_IDENTITY_FIELDS if field not in identity_fields
    )
    if missing:
        raise ValueError("identity_fields missing required keys: " + ", ".join(missing))
    return source_ids


def build_canonical_dense_readout_table(
    *,
    source_context_ids: Sequence[str],
    labels: Sequence[int],
    forward_rows: Callable[..., np.ndarray],
    width: int,
    output_dir: Path,
    build_batch_size: int,
    probe_batch_size: int,
    identity_fields: Mapping[str, Any],
    seed_material: str,
    tf32_scope: bool = True,
    parity_tolerance: float = PARITY_TOLERANCE,
    parity_tolerance_basis: str = PARITY_TOLERANCE_BASIS,
    reduction_order_characterization: Mapping[str, Any] | None = None,
    model_transform: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Stream a no-clobber dense-readout table and certify batch parity.

    ``model_transform`` (the transform's receipt block) is
    recorded under ``receipt["model_transform"]`` only when supplied, so a
    derived table names the registered transform whose derived state its
    ``model_state_hash`` is.  Without it the receipt bytes are unchanged.

    ``seed_material`` is computed by the CLI with the registered IRG
    ``content_hash`` helper. Keeping it injected leaves this core independent
    of runtime resolution and artifact contracts.
    """

    source_ids = _validate_build_inputs(
        source_context_ids=source_context_ids,
        labels=labels,
        width=width,
        build_batch_size=build_batch_size,
        probe_batch_size=probe_batch_size,
        identity_fields=identity_fields,
    )
    destination = Path(output_dir)
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise FileExistsError(
                f"canonical dense-readout output is nonempty: {destination}"
            )

    ambient_matmul_tf32: bool | None = None
    ambient_cudnn_tf32: bool | None = None
    if tf32_scope:
        ambient_matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        ambient_cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    started = time.perf_counter()
    try:
        destination.mkdir(parents=True, exist_ok=True)
        values_path = destination / VALUES_FILENAME
        values = np.lib.format.open_memmap(
            values_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(source_ids), int(width)),
        )
        try:
            for start in range(0, len(source_ids), int(build_batch_size)):
                stop = min(start + int(build_batch_size), len(source_ids))
                chunk_ids = source_ids[start:stop]
                chunk_values = np.asarray(
                    forward_rows(
                        chunk_ids,
                        batch_size=int(build_batch_size),
                    )
                )
                expected_shape = (stop - start, int(width))
                if chunk_values.shape != expected_shape:
                    raise ValueError(
                        "forward_rows returned shape "
                        f"{chunk_values.shape}, expected {expected_shape}"
                    )
                if chunk_values.dtype != np.float32:
                    raise ValueError(
                        "forward_rows must return dtype float32, got "
                        f"{chunk_values.dtype}"
                    )
                values[start:stop] = chunk_values
                del chunk_values
            values.flush()
        finally:
            del values

        ids_path = destination / IDS_FILENAME
        np.savez_compressed(
            ids_path,
            source_context_ids=np.asarray(source_ids),
            labels=np.asarray(labels, dtype=np.int64),
        )
        parity_probe = dict(
            run_parity_self_check(
                source_context_ids=source_ids,
                values_path=values_path,
                forward_rows=forward_rows,
                probe_batch_size=int(probe_batch_size),
                seed_material=str(seed_material),
                tolerance=float(parity_tolerance),
                tolerance_basis=str(parity_tolerance_basis),
            )
        )
        if not parity_probe["accepted"]:
            raise RuntimeError(
                "canonical dense-readout parity self-check failed: "
                f"max_abs_diff={parity_probe['max_abs_diff']} exceeds "
                f"tolerance={parity_probe['tolerance']} "
                f"(basis={parity_probe['tolerance_basis']}). The probe leg "
                "recomputes at a different batch size by design, so some "
                "fp32 reduction-order spread is unavoidable; measure it with "
                "--characterize-reduction-order before changing the tolerance."
            )

        receipt: dict[str, Any] = {
            "schema": "canonical_dense_readout_table_receipt_v1",
            "family": "canonical_dense_readout_table_v1",
            **{
                field: identity_fields[field]
                for field in _REQUIRED_IDENTITY_FIELDS
            },
            "execution_policy": "fp32_tf32_disabled",
            "row_count": len(source_ids),
            "readout_width": int(width),
            "build_batch_size": int(build_batch_size),
            "probe_batch_size": int(probe_batch_size),
            "payload_sha256": {
                VALUES_FILENAME: _sha256_file(values_path),
                IDS_FILENAME: _sha256_file(ids_path),
            },
            "parity_probe": parity_probe,
            "reduction_order_characterization": (
                dict(reduction_order_characterization)
                if reduction_order_characterization is not None
                else None
            ),
            "wall_seconds": float(time.perf_counter() - started),
        }
        if model_transform is not None:
            receipt["model_transform"] = dict(model_transform)
        (destination / RECEIPT_FILENAME).write_text(
            json.dumps(receipt, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        return receipt
    finally:
        if tf32_scope:
            torch.backends.cuda.matmul.allow_tf32 = ambient_matmul_tf32
            torch.backends.cudnn.allow_tf32 = ambient_cudnn_tf32


def open_canonical_dense_readout_table(
    output_dir: str | Path,
) -> CanonicalDenseReadoutTableHandle:
    """Open table values read-only while materializing only IDs and labels."""

    destination = Path(output_dir)
    with np.load(destination / IDS_FILENAME, allow_pickle=False) as sidecar:
        source_context_ids = tuple(
            str(value) for value in sidecar["source_context_ids"]
        )
        labels = np.array(sidecar["labels"], dtype=np.int64, copy=True)
    values = np.load(destination / VALUES_FILENAME, mmap_mode="r")
    if values.ndim != 2 or values.shape[0] != len(source_context_ids):
        raise ValueError("canonical dense-readout payload axes are inconsistent")
    if labels.shape != (len(source_context_ids),):
        raise ValueError("canonical dense-readout labels are inconsistent")
    return CanonicalDenseReadoutTableHandle(
        values=values,
        source_context_ids=source_context_ids,
        labels=labels,
        index_by_context_id={
            context_id: index
            for index, context_id in enumerate(source_context_ids)
        },
    )


def lookup_canonical_dense_readout_rows(
    handle: CanonicalDenseReadoutTableHandle,
    context_ids: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Copy found rows in request order and report missing IDs exactly."""

    found_rows: list[np.ndarray] = []
    missing_ids: list[str] = []
    for raw_context_id in context_ids:
        context_id = str(raw_context_id)
        index = handle.index_by_context_id.get(context_id)
        if index is None:
            missing_ids.append(context_id)
        else:
            found_rows.append(np.array(handle.values[int(index)], copy=True))
    if found_rows:
        rows = np.stack(found_rows, axis=0)
    else:
        rows = np.empty(
            (0, int(handle.values.shape[1])),
            dtype=handle.values.dtype,
        )
    return rows, tuple(missing_ids)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--circuit-plan", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--optimization-axis", required=True)
    parser.add_argument("--evaluation-axis", required=True)
    parser.add_argument("--optimization-image-root", required=True)
    parser.add_argument("--evaluation-image-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--model-source-replica")
    parser.add_argument(
        "--model-transform",
        default=None,
        help=(
            "v5.1: registered model_transform.json. The table is built for "
            "the derived model the sealed loader re-derives and verifies, so "
            "its receipt carries the derived model_state_hash and the "
            "transform binding. Absent keeps the v5 build."
        ),
    )
    parser.add_argument("--optimization-uint8-cache-manifest")
    parser.add_argument("--optimization-uint8-cache-receipt")
    parser.add_argument("--evaluation-uint8-cache-manifest")
    parser.add_argument("--evaluation-uint8-cache-receipt")
    parser.add_argument(
        "--measurement-plan",
        default=None,
        help=(
            "Sealed MeasurementPlan path. REQUIRED under "
            "--measurement-artifact-policy contracts_only: that policy reads "
            "no measurement artifact, so the caller must supply the sealed "
            "query axis (runner.py refuses otherwise). The canonical table "
            "itself is query-INDEPENDENT and full-width; the query axis is "
            "needed only to resolve the runtime."
        ),
    )
    parser.add_argument(
        "--measurement-artifact-policy",
        choices=("required", "contracts_only"),
        default="required",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    # None is resolved after parsing to batch_size + 1, so custom build sizes
    # also receive a genuinely different probe loader unless explicitly set.
    parser.add_argument("--probe-batch-size", type=int, default=None)
    parser.add_argument(
        "--characterize-reduction-order",
        type=int,
        default=0,
        metavar="SAMPLES",
        help=(
            "Measure the build-vs-probe fp32 reduction-order spread on this "
            "many 32-row samples, drawn disjoint from the acceptance probe's "
            "own seeded positions, and set the parity tolerance from it via "
            "the registered floor/margin/ceiling policy. The check varies "
            "batch size by design and fp32 accumulation is not batch-shape "
            "invariant, so the threshold must come from measurement. "
            "Disjointness is what keeps this non-circular."
        ),
    )
    parser.add_argument(
        "--characterization-only",
        action="store_true",
        help=(
            "With --characterize-reduction-order, report the measured spread "
            "and the tolerance it implies, then exit without building."
        ),
    )
    return parser


def _build_runtime_table(args: argparse.Namespace) -> Mapping[str, Any]:
    """Resolve the registered GPU runtime and invoke the hermetic core."""

    from circuitflow.artifacts import ArtifactStore
    from mechanistic_systems.backends.profiling import PhaseProfiler
    from mechanistic_systems.circuits.irg.candidates import HardMask
    from mechanistic_systems.circuits.irg.contracts import content_hash
    from mechanistic_systems.circuits.runner import (
        _resolve_registered_sampled_path_runtime,
    )
    from mechanistic_systems.circuits.statistical_sequential import (
        StatisticalSequentialProfile,
    )
    from mechanistic_systems.contracts import CircuitPlan, StudySpec
    from mechanistic_systems.measurement.contexts import (
        PREPROCESSED_UINT8_BANK_LOADER_CONTRACT,
        ClassIndexedUint8ContextSource,
        ContextAxis,
        PreprocessedUint8BankContextSource,
    )
    from mechanistic_systems.measurement.model_sources import ModelSourceReplica

    study = StudySpec.load(args.study)
    plan = CircuitPlan.load(args.circuit_plan)
    profile = StatisticalSequentialProfile.load(args.profile)
    optimization_axis = ContextAxis.load(args.optimization_axis)
    evaluation_axis = ContextAxis.load(args.evaluation_axis)
    store = ArtifactStore(args.artifact_root)
    profiler = PhaseProfiler()
    model_source_replica = (
        None
        if args.model_source_replica is None
        else ModelSourceReplica.load(args.model_source_replica)
    )
    model_transform = None
    if getattr(args, "model_transform", None) is not None:
        from mechanistic_systems.measurement.model_transforms import (
            load_model_transform,
        )

        model_transform = load_model_transform(args.model_transform)
        if model_transform.base_model_id != str(study.model_id):
            raise ValueError(
                "model transform base model differs from the StudySpec model"
            )
    from mechanistic_systems.circuits.irg.registered_resnet18_pilot import (
        _a47_sealed_queries,
    )
    
    # Reuse the pilot's helper rather than reimplementing the rule:
    # returns None under 'required', and the sealed plan's queries
    # under 'contracts_only'.
    sealed_queries = _a47_sealed_queries(
        args.measurement_artifact_policy, args.measurement_plan
    )
    runtime = _resolve_registered_sampled_path_runtime(
        study=study,
        plan=plan,
        profile=profile,
        promotion_receipt=None,
        optimization_axis=optimization_axis,
        evaluation_axis=evaluation_axis,
        optimization_image_root=args.optimization_image_root,
        evaluation_image_root=args.evaluation_image_root,
        optimization_uint8_cache_manifest=args.optimization_uint8_cache_manifest,
        optimization_uint8_cache_receipt=args.optimization_uint8_cache_receipt,
        evaluation_uint8_cache_manifest=args.evaluation_uint8_cache_manifest,
        evaluation_uint8_cache_receipt=args.evaluation_uint8_cache_receipt,
        store=store,
        profiler=profiler,
        checkpoint_path=args.checkpoint_path,
        allow_download=False,
        device=args.device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=True,
        loaded_model=None,
        model_source_replica=model_source_replica,
        queries=sealed_queries,
        measurement_artifact_policy=args.measurement_artifact_policy,
        **({} if model_transform is None else {"model_transform": model_transform}),
    )
    if model_transform is not None and (
        runtime.loaded.receipt.state_dict_hash
        != model_transform.derived_state_dict_sha256
        or runtime.loaded.receipt.model_transform_hash
        != model_transform.content_hash
    ):
        raise RuntimeError("resolved runtime does not carry the requested model transform")

    probe_batch_size = (
        int(args.batch_size) + 1
        if args.probe_batch_size is None
        else int(args.probe_batch_size)
    )
    if probe_batch_size == int(args.batch_size):
        raise ValueError("--probe-batch-size must differ from --batch-size")

    requires_bank = optimization_axis.selector.get("runtime_loader_contract") == (
        PREPROCESSED_UINT8_BANK_LOADER_CONTRACT
    )
    cache_manifest = args.optimization_uint8_cache_manifest
    cache_receipt = args.optimization_uint8_cache_receipt
    if (cache_manifest is None) != (cache_receipt is None):
        raise ValueError(
            "optimization uint8 cache manifest and receipt must be supplied together"
        )
    if cache_manifest is not None and cache_receipt is not None:
        probe_source = PreprocessedUint8BankContextSource(
            optimization_axis,
            cache_manifest,
            cache_receipt,
            batch_size=probe_batch_size,
            device=args.device,
            preload_to_ram=False,
            pin_memory=True,
            verification_mode="certified_probes",
        )
    elif requires_bank:
        raise ValueError(
            "sampled-path axis requires a certified preprocessed uint8 bank replica"
        )
    else:
        probe_source = ClassIndexedUint8ContextSource(
            optimization_axis,
            args.optimization_image_root,
            batch_size=probe_batch_size,
            device=args.device,
            num_workers=int(args.num_workers),
            pin_memory=True,
        )

    device = torch.device(args.device)
    model = runtime.loaded.model.to(device=device, dtype=torch.float32).eval()

    def forward_rows(
        context_ids: Sequence[str], *, batch_size: int
    ) -> np.ndarray:
        source = (
            probe_source
            if int(batch_size) == probe_batch_size
            else runtime.optimization_source
        )
        requested = tuple(str(value) for value in context_ids)
        row_blocks: list[np.ndarray] = []
        seen_ids: list[str] = []
        with torch.no_grad():
            for batch in source.panel_batches(requested):
                inputs = batch.inputs.to(device=device, dtype=torch.float32)
                logits = model(inputs)
                if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
                    raise ValueError("registered model must return rank-two logits")
                row_blocks.append(
                    logits.detach().to(dtype=torch.float32).cpu().numpy()
                )
                seen_ids.extend(str(value) for value in batch.context_ids)
        if tuple(seen_ids) != requested:
            raise ValueError("context source returned rows outside requested order")
        if not row_blocks:
            return np.empty((0, 0), dtype=np.float32)
        result = np.concatenate(row_blocks, axis=0)
        if result.dtype != np.float32 or result.shape[0] != len(requested):
            raise ValueError("registered model forward violated the row contract")
        return result

    source_context_ids = tuple(str(value) for value in optimization_axis.context_ids)
    width_probe_count = min(len(source_context_ids), int(args.batch_size))
    width_probe = forward_rows(
        source_context_ids[:width_probe_count],
        batch_size=int(args.batch_size),
    )
    if width_probe.ndim != 2 or width_probe.shape[1] <= 0:
        raise ValueError("registered model produced an invalid readout width")
    width = int(width_probe.shape[1])
    del width_probe

    route_axis_hash = str(plan.route_universe_hash)
    route_count = sum(
        int(np.prod(group.shape, dtype=np.int64))
        for group in runtime.route_groups
    )
    dense_mask = HardMask.dense(route_count, route_axis_hash=route_axis_hash)
    device_capability = (
        str(torch.cuda.get_device_capability(device))
        if device.type == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    identity_fields: dict[str, Any] = {
        "model_state_hash": runtime.loaded.receipt.state_dict_hash,
        "route_schema_hash": runtime.evaluator.route_schema_hash,
        "topology_hash": runtime.evaluator.topology_hash,
        "dense_mask_hash": dense_mask.mask_hash,
        "context_axis_hash": optimization_axis.content_hash,
        "split": str(optimization_axis.split),
        "forward_tier": int(
            runtime.evaluator.runtime_execution_profile.forward_tier
        ),
        "torch_version": str(torch.__version__),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_capability": device_capability,
    }
    if model_transform is not None:
        # Seed material of a derived table differs from its base table's.
        identity_fields["model_transform_hash"] = model_transform.content_hash
    labels_by_context_id = {
        str(record.context_id): int(record.label)
        for record in optimization_axis.records
    }
    labels = [labels_by_context_id[context_id] for context_id in source_context_ids]
    seed_material = content_hash(
        dict(identity_fields),
        prefix="irg-canonical-dense-readout-table-",
    )
    characterization: Mapping[str, Any] | None = None
    parity_tolerance = float(PARITY_TOLERANCE)
    parity_tolerance_basis = str(PARITY_TOLERANCE_BASIS)
    sample_count = int(getattr(args, "characterize_reduction_order", 0) or 0)
    if sample_count > 0:
        # Measure the spread the acceptance probe cannot avoid, on rows
        # DISJOINT from the ones it will check, under the same TF32-disabled
        # policy the build itself uses. Then apply the registered policy.
        if hasattr(torch.backends, "cuda") and hasattr(
            torch.backends.cuda, "matmul"
        ):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        characterization = dict(
            characterize_reduction_order_spread(
                source_context_ids=source_context_ids,
                forward_rows=forward_rows,
                build_batch_size=int(args.batch_size),
                probe_batch_size=probe_batch_size,
                seed_material=seed_material,
                sample_count=sample_count,
            )
        )
        parity_tolerance, parity_tolerance_basis = (
            parity_tolerance_from_spread(
                float(characterization["max_abs_diff"])
            )
        )
        if bool(getattr(args, "characterization_only", False)):
            return {
                "schema": "canonical_dense_readout_characterization_v1",
                "identity_fields": dict(identity_fields),
                "reduction_order_spread": dict(characterization),
                "implied_tolerance": float(parity_tolerance),
                "implied_tolerance_basis": str(parity_tolerance_basis),
            }

    return build_canonical_dense_readout_table(
        source_context_ids=source_context_ids,
        labels=labels,
        forward_rows=forward_rows,
        width=width,
        output_dir=Path(args.output_dir),
        build_batch_size=int(args.batch_size),
        probe_batch_size=probe_batch_size,
        identity_fields=identity_fields,
        seed_material=seed_material,
        tf32_scope=True,
        parity_tolerance=parity_tolerance,
        parity_tolerance_basis=parity_tolerance_basis,
        reduction_order_characterization=characterization,
        **(
            {}
            if model_transform is None
            else {"model_transform": model_transform.receipt_block()}
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = _build_runtime_table(args)
    if receipt.get("schema") == "canonical_dense_readout_characterization_v1":
        print(
            json.dumps(
                {
                    "reduction_order_spread": receipt[
                        "reduction_order_spread"
                    ],
                    "implied_tolerance": receipt["implied_tolerance"],
                    "implied_tolerance_basis": receipt[
                        "implied_tolerance_basis"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    print(
        json.dumps(
            {
                "row_count": receipt["row_count"],
                "readout_width": receipt["readout_width"],
                "output_dir": str(args.output_dir),
                "parity_probe": receipt["parity_probe"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
