"""Run SEVERAL registered ResNet18 pilot arms inside ONE process.

The items file names a sequence of category and arm attempts. With
``--persistent-worker on``, the model, bank maps, digest tables, gate graph,
and structural compiled runners remain in process. Statistical providers,
sampling streams, data-plane counters, and item outputs are rebuilt each time.

Isolation contract (what makes packed receipts trustworthy):

* Every item gets its own immutable output directory and per-item provider.
* Read-only contracts and bank maps are shared. On a failed item the retained
  runtime is dropped before the next item.
* Between arms the driver runs gc + CUDA synchronize/empty_cache and
  resets peak-memory statistics, so per-arm HBM peaks stay per-arm honest.
  ``process_peak_rss`` is process-cumulative and therefore monotone across
  a pack; the pack receipt records this caveat explicitly.
* The pack receipt is rewritten after every arm (crash-safe): a killed pack
  keeps the completed arms' receipt hashes and timings.

``--persistent-worker off`` invokes the incumbent per-arm path.

Shared-process worker (``--worker-items``): one process per GPU job runs
the job's (category, arms) items back to back. Each arm is the arm runner's own
launch, taken from the inner launcher's dump of its argument vector and
environment; the worker overlays that environment, builds the pilot call with
the arm runner's ``build_pilot_call`` (one argument builder, no drift), writes
the arm's stdout/stderr to the arm's own logs, and keeps the per-item resets
above. One category memo (``category_memo``) shares the census, the dense bank
and the dense terminal products between that category's arms. Any arm failure
ends the process (exit 1) after its record is written; a recycle rule ends it
cleanly (exit 3) between categories.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import gc
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from mechanistic_systems.core.determinism import configure_deterministic_backend
from circuitflow.artifacts import ArtifactStore
from mechanistic_systems.circuits.irg.precision import IndependentSampleSizePolicy
from mechanistic_systems.circuits.irg.registered_resnet18_pilot import (
    run_registered_resnet18_pilot,
)
from mechanistic_systems.circuits.irg.workflow_identity import (
    resolve_current_workflow_identity,
)
from mechanistic_systems.circuits.native_sampled import (
    load_native_sampled_path_promotion_receipt,
)
from mechanistic_systems.circuits.statistical_sequential import (
    StatisticalSequentialProfile,
)
from mechanistic_systems.contracts import CircuitPlan, StudySpec
from mechanistic_systems.measurement.contexts import ContextAxis
from mechanistic_systems.measurement.model_sources import ModelSourceReplica

PACK_RECEIPT_SCHEMA = "irg_registered_resnet18_pack_receipt_v1"

_PARTITION_FAMILY_CHOICES = (
    "functional_atlas",
    "local_attribution_sign",
    "global_attribution_sign",
    "historical_spatial_weight_sum_sign",
    "randomized_count_matched",
    "single_population",
    # Replay-only aliases for pre-v2 launch receipts.
    "structural_sign",
    "historical_weight_sign",
    "attributional_sign",
    "randomized_2_partition",
)

_ARM_OVERRIDE_FIELDS = (
    "category",
    "random_seed",
    "sensitivity_tau",
    "terminal_namespace_salt",
    "maximum_commits",
    "partition_seed",
    "comparison_namespace",
    "partition_family",
    "query_id",
    "comparison_validation_mode",
)


def _parse_current_version(raw: str) -> str:
    """Validate the qualified identity while preserving legacy bare internals."""

    canonical = resolve_current_workflow_identity(raw).canonical_identity
    return canonical.rsplit("/", 1)[-1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        type=_parse_current_version,
        required=True,
        metavar="irg/v0",
        help=(
            "Current workflow identity. v0 and irg/v0 select modern queue-v0; "
            "v1 is unregistered and v2-v5 are reserved and fail closed."
        ),
    )
    parser.add_argument(
        "--partition-family",
        choices=_PARTITION_FAMILY_CHOICES,
        required=True,
    )
    parser.add_argument("--comparison-namespace", required=True)
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument("--circuit-plan", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--promotion-receipt")
    parser.add_argument("--optimization-axis", required=True)
    parser.add_argument("--evaluation-axis", required=True)
    parser.add_argument("--optimization-image-root", required=True)
    parser.add_argument("--evaluation-image-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="PACK BASE directory; each arm writes to <base>/<attempt_id>.",
    )
    parser.add_argument("--historical-mask-path")
    parser.add_argument(
        "--measurement-artifact-policy",
        choices=("required", "contracts_only"),
        default="required",
    )
    parser.add_argument("--measurement-plan")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--model-source-replica")
    parser.add_argument("--optimization-uint8-cache-manifest")
    parser.add_argument("--optimization-uint8-cache-receipt")
    parser.add_argument("--evaluation-uint8-cache-manifest")
    parser.add_argument("--evaluation-uint8-cache-receipt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--scoring-batch-rows", type=int, default=32)
    parser.add_argument("--calibration-batch-rows", type=int, default=512)
    parser.add_argument("--v0-paired-oracle-batch-rows", type=int, default=256)
    parser.add_argument(
        "--v1-candidate-oracle-batch-rows", type=int, default=512
    )
    parser.add_argument("--terminal-batch-rows", type=int, default=512)
    parser.add_argument(
        "--native-panel-device-cache-bytes", type=int, default=0
    )
    parser.add_argument("--panel-prefetch-depth", type=int, default=2)
    parser.add_argument("--panel-producer-workers", type=int, default=1)
    parser.add_argument("--unit-packing-degree", type=int, default=1)
    parser.add_argument("--query-tile-size", type=int, default=1)
    parser.add_argument(
        "--require-optimized-native-panel", action="store_true"
    )
    parser.add_argument("--maximum-relative-bce-increase", type=float, default=0.20)
    parser.add_argument(
        "--sensitivity-tau",
        type=float,
        help="Shared default tau; an arm entry may override it.",
    )
    parser.add_argument("--terminal-maximum-per-stratum", type=int, default=256)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument(
        "--maximum-prune-order-uncertainty-fraction",
        type=float,
        default=0.001,
    )
    parser.add_argument("--partition-geometry-cache-dir")
    parser.add_argument("--dense-readout-table-cache-dir")
    parser.add_argument(
        "--enable-local-regime-diagnostics", action="store_true"
    )
    parser.add_argument(
        "--v0-calibration-decision-estimand",
        choices=(
            "frozen_extraction_calibration_risk_v1",
            "accepted_state_frozen_candidate_refinable_v1",
            "population_optimal_affine_risk_v1",
        ),
        default="frozen_extraction_calibration_risk_v1",
    )
    parser.add_argument(
        "--v0-accept-replication-gate", action="store_true"
    )
    parser.add_argument(
        "--v0-oracle-minimum-per-stratum", type=int, default=2
    )
    parser.add_argument(
        "--v0-schedule-replay-spec",
        help=(
            "Path to the sealed historical schedule-replay spec JSON; "
            "arms the pre-registered replay probe for every arm in the "
            "pack. Unset keeps the tau path byte-identical."
        ),
    )
    parser.add_argument(
        "--v0-schedule-replay-rank-source",
        choices=("family", "local_attribution"),
        default="family",
        help=(
            "Within-sign-group replay ranks: family activity magnitudes "
            "(primary cell) or modern local-attribution magnitudes "
            "(secondary cell)."
        ),
    )
    parser.add_argument(
        "--v1-ordering-key",
        choices=(
            "abs_raw",
            "k1_expected_damage",
            "k2_damage_energy",
            "k3_whitened_salience",
        ),
        default="abs_raw",
        help=(
            "Registered v1 ordering key shared by every arm in the pack; "
            "abs_raw keeps the incumbent activity order byte-identical."
        ),
    )
    parser.add_argument(
        "--v1-span-budget-fraction",
        type=float,
        default=0.066145,  # registered accept_rule.span_budget_fraction
        help=(
            "v1 ladder acceptance budget as a fraction of the certified "
            "dense-to-empty loss span, shared by every arm in the pack."
        ),
    )
    parser.add_argument(
        "--calibration-deployment-refit-anchor",
        choices=("selected", "unregularized"),
        default="selected",
    )
    parser.add_argument(
        "--pack-arms", "--items",
        required=True,
        help=(
            "JSON list of arm entries, inline or as a file path. Each entry "
            "requires a unique attempt_id and random_seed; optional "
            "overrides: " + ", ".join(_ARM_OVERRIDE_FIELDS) + "."
        ),
    )
    parser.add_argument(
        "--pack-failure-policy",
        choices=("halt", "continue"),
        default="halt",
        help=(
            "halt: first failing arm stops the pack (validation packs); "
            "continue: record the failure and run the remaining arms "
            "(campaign packs)."
        ),
    )
    parser.add_argument(
        "--persistent-worker", choices=("on", "off"), default="off",
        help="Retain model, bank maps, and structural runners between items.",
    )
    return parser


def load_pack_arms(raw: str) -> list[dict[str, Any]]:
    candidate = Path(raw)
    try:
        # An inline JSON list longer than the filesystem's NAME_MAX makes
        # ``exists()`` raise rather than return False.
        raw_is_file = candidate.exists()
    except OSError:
        raw_is_file = False
    text = (
        candidate.read_text(encoding="utf-8")
        if raw_is_file
        else raw
    )
    arms = json.loads(text)
    if not isinstance(arms, list) or not arms:
        raise ValueError("--pack-arms must be a nonempty JSON list")
    seen: set[str] = set()
    for entry in arms:
        if not isinstance(entry, Mapping):
            raise ValueError("each pack arm must be a JSON object")
        attempt = str(entry.get("attempt_id", "")).strip()
        if not attempt:
            raise ValueError("each pack arm requires a nonempty attempt_id")
        if attempt in seen:
            raise ValueError(f"duplicate pack attempt_id: {attempt!r}")
        seen.add(attempt)
        if "random_seed" not in entry:
            raise ValueError(
                f"pack arm {attempt!r} requires an explicit random_seed"
            )
        unknown = set(entry) - {"attempt_id", *_ARM_OVERRIDE_FIELDS}
        if unknown:
            raise ValueError(
                f"pack arm {attempt!r} has unknown fields: {sorted(unknown)!r}"
            )
        if (
            "partition_family" in entry
            and entry["partition_family"] not in _PARTITION_FAMILY_CHOICES
        ):
            raise ValueError(
                f"pack arm {attempt!r} has unsupported partition_family: "
                f"{entry['partition_family']!r}"
            )
        if (
            "query_id" in entry
            and "category" in entry
            and entry["query_id"] != entry["category"]
        ):
            raise ValueError(f"pack arm {attempt!r} has conflicting category and query_id")
        if "query_id" in entry or "category" in entry:
            query_id = entry.get("category", entry.get("query_id"))
            if (
                isinstance(query_id, bool)
                or not isinstance(query_id, int)
                or query_id < 0
            ):
                raise ValueError(
                    f"pack arm {attempt!r} has invalid query_id: {query_id!r}"
                )
    return [dict(entry) for entry in arms]


@contextmanager
def _item_random_state(seed: int):
    """Give each item the same global RNG start regardless of queue order."""

    import torch

    configure_deterministic_backend()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    torch_threads = torch.get_num_threads()
    tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    tf32_cudnn = torch.backends.cudnn.allow_tf32
    matmul_precision = torch.get_float32_matmul_precision()
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.set_rng_state(
            torch.Generator().manual_seed(seed % (2**64)).get_state()
        )
        torch.cuda.manual_seed_all(seed % (2**64))
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.set_num_threads(torch_threads)
        torch.set_float32_matmul_precision(matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = tf32_matmul
        torch.backends.cudnn.allow_tf32 = tf32_cudnn
        configure_deterministic_backend()


def _between_arm_reset() -> None:
    """Free the finished arm's device/host state before the next arm."""

    gc.collect()
    try:
        import torch
    except Exception:  # pragma: no cover - torch is a hard dep in production
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _reset_item_module_state(*, failed: bool) -> None:
    """Retire category-bound module memo entries and completed progress lines."""

    from mechanistic_systems.circuits import v6_records
    from mechanistic_systems.circuits.irg import registered_resnet18_pilot as pilot
    from mechanistic_systems.measurement import native_irg_providers as providers
    from mechanistic_systems import perf_gates
    from mechanistic_systems.measurement import native_panel_data_plane as delivery
    from mechanistic_systems.circuits.irg import census_reference

    pilot._v6_progress_flush()
    perf_gates.disarm_walk_phase()
    # This is a per-item receipt record; compiled callable caches survive.
    providers._COMPILE_COVERAGE_PATHS.clear()
    with pilot._SEALED_TERMINAL_POPULATIONS_MEMO_LOCK:
        pilot._SEALED_TERMINAL_POPULATIONS_MEMO.clear()
    v6_records._FORCED_CACHE.clear()
    with providers._FIT_BUNDLE_HASH_MEMO_LOCK:
        providers._FIT_BUNDLE_HASH_MEMO.clear()
    with providers._TERMINAL_LAYOUT_MEMO_LOCK:
        providers._TERMINAL_LAYOUT_MEMO.clear()
    # The retained evaluator receives a fresh candidate backend in runner.py:
    # activation-prefix rows and their version tokens never cross an item.
    # run_pack closes each plane's pooled iterators and executors first.
    with delivery._WORKER_PINNED_POOLS_LOCK:
        for pool in delivery._WORKER_PINNED_POOLS.values():
            with pool.lock:
                if any(state.get("leased") for state in pool.state.values()):
                    raise RuntimeError("persistent worker item retained a pinned host lease")
        delivery._WORKER_PINNED_POOLS.clear()
    census_reference._FactorBank.read.cache_clear()
    # The async fit executor is a local `with ThreadPoolExecutor` in
    # ensure_calibration; its exit joins the worker before an item returns.
    if failed:
        with providers._PROVENANCE_POOL_LOCK:
            pool, providers._PROVENANCE_POOL = providers._PROVENANCE_POOL, None
        if pool is not None:
            pool.shutdown(wait=True)


def _close_item_planes(retained_runtime_inputs: dict[str, Any]) -> None:
    planes = retained_runtime_inputs.pop("active_planes", ())
    gathers = {}
    for plane in {id(value): value for value in planes}.values():
        plane.close_persistent_producer(wait=True)
        plane._bank_shard_executor.shutdown(wait=True)
        if plane._prefetch_executor is not None:
            plane._prefetch_executor.shutdown(wait=True)
        plane.release_buffer_rings(include_pinned=True)
        plane.release_device_cache()
        # The bank's device registration is process-wide and stays; the
        # plane's look-ahead pages (up to 2 GiB) are released now.
        gather = getattr(plane, "_device_gather", None)
        if gather is not None:
            gather.close()
            gathers[id(gather)] = gather
    # Calibration and validation rows belong to this item. Their registered
    # mmap survives, but its resident HBM must not overlap the next census.
    # First close every plane: two planes can share the same registration.
    for gather in gathers.values():
        gather.release_resident_populations()


def _release_item_runtime(retained_runtime_inputs: dict[str, Any]) -> None:
    """Keep the loaded model and executable structures, retire item payloads."""

    evaluator = retained_runtime_inputs.get("evaluator")
    if evaluator is not None:
        evaluator.release_item_resources()
    for role in ("optimization_source", "evaluation_source"):
        source = retained_runtime_inputs.get(role)
        if source is not None:
            source._bundle_cache.clear()
            if hasattr(source, "exit_raw_panel_delivery"):
                source.exit_raw_panel_delivery()


def run_pack(
    args: argparse.Namespace,
    *,
    pilot_runner: Callable[..., Mapping[str, Any]] = run_registered_resnet18_pilot,
    between_arm_reset: Callable[[], None] = _between_arm_reset,
) -> dict[str, Any]:
    process_started = time.perf_counter()
    persistent = (
        getattr(args, "persistent_worker", "off") == "on"
        and (
            pilot_runner is run_registered_resnet18_pilot
            or getattr(pilot_runner, "supports_retained_runtime_inputs", False)
        )
    )
    retained_runtime_inputs: dict[str, Any] = {}
    arms = load_pack_arms(args.pack_arms)
    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)
    receipt_path = base / "pack_receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"pack receipt already exists: {receipt_path}")
    for entry in arms:
        arm_dir = base / str(entry["attempt_id"])
        if arm_dir.exists():
            raise FileExistsError(f"pack arm output exists: {arm_dir}")

    runtime_environment = {
        "CIRCUIT_SCORING_BATCH_ROWS": args.scoring_batch_rows,
        "CIRCUIT_CALIBRATION_BATCH_ROWS": args.calibration_batch_rows,
        "CIRCUIT_V0_PAIRED_ORACLE_BATCH_ROWS": (
            args.v0_paired_oracle_batch_rows
        ),
        "CIRCUIT_V1_CANDIDATE_ORACLE_BATCH_ROWS": (
            args.v1_candidate_oracle_batch_rows
        ),
        "CIRCUIT_TERMINAL_BATCH_ROWS": args.terminal_batch_rows,
        "CIRCUIT_NATIVE_PANEL_DEVICE_CACHE_BYTES": (
            args.native_panel_device_cache_bytes
        ),
        "CIRCUIT_SAMPLED_PREFETCH_DEPTH": args.panel_prefetch_depth,
        "CIRCUIT_PRODUCER_WORKERS": args.panel_producer_workers,
        "CIRCUIT_UNIT_PACKING_DEGREE": args.unit_packing_degree,
        "CIRCUIT_QUERY_TILE_SIZE": args.query_tile_size,
        "CIRCUIT_REQUIRE_OPTIMIZED_NATIVE_PANEL": int(
            args.require_optimized_native_panel
        ),
        "CIRCUIT_MPS_ENABLED": 0,
    }
    for name, value in runtime_environment.items():
        os.environ[name] = str(value)

    shared = {
        "study": StudySpec.load(args.study),
        "plan": CircuitPlan.load(args.circuit_plan),
        "profile": StatisticalSequentialProfile.load(args.profile),
        "promotion_receipt": (
            None
            if args.promotion_receipt is None
            else load_native_sampled_path_promotion_receipt(
                args.promotion_receipt
            )
        ),
        "optimization_axis": ContextAxis.load(args.optimization_axis),
        "evaluation_axis": ContextAxis.load(args.evaluation_axis),
        "store": ArtifactStore(args.artifact_root),
        "model_source_replica": (
            None
            if args.model_source_replica is None
            else ModelSourceReplica.load(args.model_source_replica)
        ),
        "policy": IndependentSampleSizePolicy(
            confidence_level=args.confidence_level,
            maximum_prune_order_uncertainty_fraction=(
                args.maximum_prune_order_uncertainty_fraction
            ),
            precision_only_iid=args.version in {"v0", "v1"},
            oracle_maximum_contexts_per_candidate=(
                None if args.version in {"v0", "v1"} else 4096
            ),
        ),
    }
    shared_startup_seconds = time.perf_counter() - process_started

    pack: dict[str, Any] = {
        "schema_name": PACK_RECEIPT_SCHEMA,
        "version": args.version,
        "partition_family": args.partition_family,
        "pack_failure_policy": args.pack_failure_policy,
        "shared_startup_seconds": float(shared_startup_seconds),
        "process_peak_rss_note": (
            "per-arm process_peak_rss is process-cumulative under packing "
            "and monotone across arms; per-arm HBM peaks are reset between "
            "arms and stay per-arm honest"
        ),
        "arms": [],
    }

    def flush() -> None:
        receipt_path.write_text(
            json.dumps(pack, indent=1, sort_keys=True), encoding="utf-8"
        )

    flush()
    for index, entry in enumerate(arms):
        attempt = str(entry["attempt_id"])
        arm_partition_family = str(
            entry.get("partition_family", args.partition_family)
        )
        arm_query_id = str(entry.get("category", entry.get("query_id", args.query_id)))
        arm_dir = base / attempt
        arm_started = time.perf_counter()
        record: dict[str, Any] = {
            "attempt_id": attempt,
            "index": index,
            "random_seed": int(entry["random_seed"]),
            "partition_family": arm_partition_family,
            "query_id": arm_query_id,
            "comparison_validation_mode": str(
                entry.get("comparison_validation_mode", "v0_marginal")
            ),
        }
        random_context = (
            _item_random_state(int(entry["random_seed"]))
            if persistent
            else nullcontext()
        )
        random_context.__enter__()
        try:
            result = pilot_runner(
                version_name=args.version,
                comparison_validation_mode=str(
                    entry.get("comparison_validation_mode", "v0_marginal")
                ),
                partition_family=arm_partition_family,
                partition_random_seed=entry.get("partition_seed"),
                comparison_namespace=str(
                    entry.get("comparison_namespace", args.comparison_namespace)
                ),
                query_id=arm_query_id,
                study=shared["study"],
                plan=shared["plan"],
                profile=shared["profile"],
                promotion_receipt=shared["promotion_receipt"],
                optimization_axis=shared["optimization_axis"],
                evaluation_axis=shared["evaluation_axis"],
                optimization_image_root=args.optimization_image_root,
                evaluation_image_root=args.evaluation_image_root,
                store=shared["store"],
                output_dir=str(arm_dir),
                historical_mask_path=args.historical_mask_path,
                measurement_artifact_policy=args.measurement_artifact_policy,
                measurement_plan_path=args.measurement_plan,
                checkpoint_path=args.checkpoint_path,
                model_source_replica=shared["model_source_replica"],
                optimization_uint8_cache_manifest=(
                    args.optimization_uint8_cache_manifest
                ),
                optimization_uint8_cache_receipt=(
                    args.optimization_uint8_cache_receipt
                ),
                evaluation_uint8_cache_manifest=(
                    args.evaluation_uint8_cache_manifest
                ),
                evaluation_uint8_cache_receipt=(
                    args.evaluation_uint8_cache_receipt
                ),
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                random_seed=int(entry["random_seed"]),
                maximum_commits=entry.get("maximum_commits"),
                maximum_relative_bce_increase=(
                    args.maximum_relative_bce_increase
                ),
                sensitivity_tau=(
                    float(entry["sensitivity_tau"])
                    if entry.get("sensitivity_tau") is not None
                    else args.sensitivity_tau
                ),
                terminal_maximum_per_stratum=(
                    args.terminal_maximum_per_stratum
                ),
                sample_size_policy=shared["policy"],
                enable_local_regime_diagnostics=(
                    args.enable_local_regime_diagnostics
                ),
                partition_geometry_cache_dir=(
                    args.partition_geometry_cache_dir
                ),
                dense_readout_table_cache_dir=(
                    args.dense_readout_table_cache_dir
                ),
                calibration_deployment_refit_anchor=(
                    args.calibration_deployment_refit_anchor
                ),
                terminal_namespace_salt=entry.get("terminal_namespace_salt"),
                v0_calibration_decision_estimand=(
                    args.v0_calibration_decision_estimand
                ),
                v0_oracle_minimum_per_stratum=(
                    args.v0_oracle_minimum_per_stratum
                ),
                v0_accept_replication_gate=args.v0_accept_replication_gate,
                v0_schedule_replay_spec=args.v0_schedule_replay_spec,
                v0_schedule_replay_rank_source=(
                    args.v0_schedule_replay_rank_source
                ),
                v1_ordering_key=args.v1_ordering_key,
                v1_span_budget_fraction=args.v1_span_budget_fraction,
                **(
                    {"retained_runtime_inputs": retained_runtime_inputs}
                    if persistent
                    else {}
                ),
            )
        except Exception as exc:
            record.update(
                status="failed",
                error=f"{type(exc).__name__}: {exc}"[:400],
                wall_seconds=float(time.perf_counter() - arm_started),
            )
            pack["arms"].append(record)
            flush()
            if args.pack_failure_policy == "halt":
                raise
        else:
            record.update(
                status="completed",
                wall_seconds=float(time.perf_counter() - arm_started),
                receipt_hash=result["receipt_hash"],
                retained_fraction=result["retained_fraction"],
                terminal_final_bce=result["terminal_final_bce"][
                    "per_image_binary_bce" if result.get("loss_weighting") == "per_image"
                    else "class_balanced_binary_bce"
                ],
                **({"loss_weighting": "per_image"} if result.get("loss_weighting") == "per_image" else {}),
                output_dir=result["output_dir"],
            )
            pack["arms"].append(record)
            flush()
            print(json.dumps({k: record[k] for k in (
                "attempt_id", "status", "wall_seconds", "receipt_hash",
                "retained_fraction", "terminal_final_bce",
                *( ("loss_weighting",) if record.get("loss_weighting") == "per_image" else () ),
            )}, sort_keys=True), flush=True)
        finally:
            _close_item_planes(retained_runtime_inputs)
            _release_item_runtime(retained_runtime_inputs)
            if record.get("status") == "failed":
                retained_runtime_inputs.clear()
            if persistent:
                _reset_item_module_state(failed=record.get("status") == "failed")
            random_context.__exit__(None, None, None)
            between_arm_reset()
    pack["total_wall_seconds"] = float(time.perf_counter() - process_started)
    flush()
    return pack


# --------------------------------------------------------------------------
# Shared-process worker
# --------------------------------------------------------------------------

WORKER_ITEMS_SCHEMA = "irg_shared_worker_items_v1"
WORKER_RECORD_SCHEMA = "irg_shared_worker_record_v1"
ARM_LAUNCH_DUMP_SCHEMA = "irg_arm_launch_dump_v1"
EXIT_COMPLETED, EXIT_ARM_FAILED, EXIT_REFUSED, EXIT_RECYCLE = 0, 1, 2, 3

# Settings read once per process (at import, at CUDA or allocator start, or by
# the compiler caches). Every arm a worker runs must carry the worker's own
# values; an arm that differs is refused before anything runs for it.
PROCESS_LEVEL_ENV = (
    "IRG_CENSUS_CACHE_FILES",
    "IRG_COMPILED_MASK_CACHE_CAPACITY",
    "MECHANISTIC_SYSTEMS_TRITON_IMPORT_ENABLED",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_FX_GRAPH_CACHE",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MALLOC_MMAP_THRESHOLD_",
    "MALLOC_TRIM_THRESHOLD_",
    "MALLOC_MMAP_MAX_",
    "NUMPY_MADVISE_HUGEPAGE",
    "PYTHONPATH",
    "PYTHONHASHSEED",
    "CUDA_VISIBLE_DEVICES",
    "LD_PRELOAD",
)


class WorkerRefusal(RuntimeError):
    """An arm this process cannot run as it would run alone."""


def load_worker_items(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != WORKER_ITEMS_SCHEMA:
        raise ValueError("worker items file has an unknown schema")
    categories = payload.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError("worker items need at least one category")
    seen: set[tuple[str, str]] = set()
    for item in categories:
        arms = item.get("arms")
        if not isinstance(arms, list) or not arms:
            raise ValueError("each worker category needs at least one arm")
        for arm in arms:
            for name in ("arm", "dump", "stdout", "stderr"):
                if not str(arm.get(name, "")).strip():
                    raise ValueError(f"worker arm entry lacks {name!r}")
            key = (str(item["category"]), str(arm["arm"]))
            if key in seen:
                raise ValueError(f"duplicate worker item {key!r}")
            seen.add(key)
    return [dict(item) for item in categories]


def load_arm_dump(path: str | Path) -> dict[str, Any]:
    dump = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        dump.get("schema") != ARM_LAUNCH_DUMP_SCHEMA
        or not isinstance(dump.get("argv"), list)
        or not isinstance(dump.get("env"), dict)
    ):
        raise ValueError(f"arm launch dump {path} is invalid")
    return dump


@contextmanager
def _arm_environment(environment: Mapping[str, str]):
    """The arm sees exactly the environment its launcher would have exec'd."""

    saved = dict(os.environ)
    os.environ.clear()
    os.environ.update({str(k): str(v) for k, v in environment.items()})
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


@contextmanager
def _arm_logs(stdout_path: str, stderr_path: str):
    """Point fds 1 and 2 at the arm's own log files for the arm's duration."""

    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    streams = (sys.stdout, sys.stderr)
    out = open(stdout_path, "ab", buffering=0)
    err = out if stderr_path == stdout_path else open(stderr_path, "ab", buffering=0)
    try:
        os.dup2(out.fileno(), 1)
        os.dup2(err.fileno(), 2)
        # Python-level writers follow the descriptors too, whatever sys.stdout was.
        sys.stdout = open(1, "w", buffering=1, closefd=False)
        sys.stderr = open(2, "w", buffering=1, closefd=False)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = streams
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])
        out.close()
        if err is not out:
            err.close()


def _host_rss_bytes() -> int | None:
    try:
        with open("/proc/self/statm", encoding="ascii") as stream:
            return int(stream.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def _device_allocated_bytes() -> int | None:
    try:
        import torch
    except Exception:  # pragma: no cover - torch is a hard dependency
        return None
    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return None
    return int(torch.cuda.memory_allocated())


def _between_category_cleanup() -> None:
    """Return host and device memory after a category, before measuring it."""

    _between_arm_reset()
    try:
        import torch

        host_empty = getattr(torch._C, "_host_emptyCache", None)
        if callable(host_empty):
            host_empty()
    except Exception:  # pragma: no cover - optional allocator surface
        pass
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:  # not glibc (e.g. macOS tests)
        pass


def _write_record(path: Path, record: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(record, indent=1, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)


def run_worker(
    items_path: str | Path,
    record_path: str | Path,
    *,
    recycle_categories: int = 25,
    recycle_rss_growth_bytes: int = 40 << 30,
    recycle_hbm_growth_bytes: int = 4 << 30,
    pilot_runner: Callable[..., Mapping[str, Any]] = run_registered_resnet18_pilot,
    build_call: Callable[..., tuple[dict[str, Any], argparse.Namespace]] | None = None,
    category_cleanup: Callable[[], None] = _between_category_cleanup,
    between_arm_reset: Callable[[], None] = _between_arm_reset,
) -> int:
    """Run items with one host look-ahead and ordered product completion.

    GPU state is released at each item boundary. Host completion snapshots
    can outlive that boundary; the record marks completion only after their
    receipt is published. A failed write blocks every later receipt.
    """
    import socket
    import sys
    import traceback
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime, timezone

    from experiments.irg_v0_v2_pilots import run_registered_resnet18_arm as arm_cli
    from mechanistic_systems.circuits.irg import category_memo
    from mechanistic_systems.circuits.irg.registered_resnet18_pilot import _PhaseTiming
    from mechanistic_systems.circuits.irg.worker_host_inputs import prepare_pilot_host_inputs
    from mechanistic_systems.circuits.irg.worker_overlap import DeferredItemProducts, OrderedItemWriter
    from mechanistic_systems.measurement.memory_probe import probe_from_environment, worker_memory_boundary

    if build_call is None:
        build_call = arm_cli.build_pilot_call
        probe_path = arm_cli.memory_probe_path
    else:
        probe_path = None
    native_builder = build_call is arm_cli.build_pilot_call
    overlap = pilot_runner is run_registered_resnet18_pilot or getattr(pilot_runner, "supports_worker_overlap", False)
    categories = load_worker_items(items_path)
    entries = [entry for category in categories for entry in category["arms"]]
    record_path = Path(record_path)
    started = time.time()
    identity = {
        "worker_id": f"{socket.gethostname()}:{os.getpid()}:{int(started)}",
        "host": socket.gethostname(), "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "items_file": str(items_path),
        "items_sha256": hashlib.sha256(Path(items_path).read_bytes()).hexdigest(),
    }
    base_environment = dict(os.environ)
    worker_cwd = os.getcwd()
    record: dict[str, Any] = {
        "schema": WORKER_RECORD_SCHEMA, "worker": identity, "status": "running",
        "recycle_rule": {"categories": int(recycle_categories),
                         "rss_growth_bytes": int(recycle_rss_growth_bytes),
                         "hbm_growth_bytes": int(recycle_hbm_growth_bytes)},
        "items": [], "categories": [],
    }
    _write_record(record_path, record)
    contracts: dict[tuple[str, str], Any] = {}

    def load(kind: str, path: str) -> Any:
        key = (kind, os.path.realpath(path))
        if key not in contracts:
            contracts[key] = arm_cli.load_contract(kind, path)
        return contracts[key]

    def prepare(entry):
        preparation_started = time.perf_counter()
        try:
            dump = load_arm_dump(entry["dump"])
        except (OSError, ValueError) as error:
            raise WorkerRefusal(f"invalid arm launch dump: {error}") from error
        differing = sorted(key for key in PROCESS_LEVEL_ENV
                           if dump["env"].get(key) != base_environment.get(key))
        if differing or dump.get("cwd", worker_cwd) != worker_cwd:
            raise WorkerRefusal("process-level settings differ from the worker's: "
                                + ", ".join(differing or ["cwd"]))
        environment = dict(dump["env"])
        built = host = None
        if native_builder:
            built = build_call(dump["argv"], load=load, environment=environment)
            if overlap and not (Path(built[0]["output_dir"]) / "receipt.json").exists():
                host = prepare_pilot_host_inputs(built[0], environment)
        return dump, environment, built, host, time.perf_counter() - preparation_started

    retained: dict[str, Any] = {}
    previous_items: list[list[str]] = []
    pending: list[tuple[dict[str, Any], DeferredItemProducts, float]] = []
    failures: list[dict[str, Any]] = []
    baseline = None
    refused = ran_categories = entry_position = 0

    def fail(item, error):
        item.update(status="failed", error=f"{type(error).__name__}: {error}"[:400],
                    traceback=traceback.format_exc()[-4000:])
        failures.append(item)

    def completed(item, result, end, arm_started):
        item.update(status="completed", receipt_hash=result["receipt_hash"],
                    wall_seconds=float(end - arm_started),
                    completed_unix=item["started_unix"] + float(end - arm_started))
        with open(item["stdout"], "a", encoding="utf-8") as output:
            print(json.dumps({"receipt_hash": result["receipt_hash"],
                              "retained_fraction": result.get("retained_fraction"),
                              "output_dir": result.get("output_dir", item.get("output_dir")),
                              "shared_worker_position": item["position"]}, sort_keys=True), file=output, flush=True)

    def retire(*, wait):
        while pending and (wait or pending[0][1].future.done()):
            item, deferred, arm_started = pending.pop(0)
            try:
                result, ended = deferred.future.result()
                completed(item, result, ended, arm_started)
            except BaseException as error:
                fail(item, error)
                item["wall_seconds"] = float(time.perf_counter() - arm_started)
            _write_record(record_path, record)
            if not wait:
                continue

    writer = OrderedItemWriter()
    prep = ThreadPoolExecutor(max_workers=1, thread_name_prefix="item-host-setup")
    prepared = prep.submit(prepare, entries[0])
    try:
        for category_index, category in enumerate(categories):
            name = str(category["category"])
            category_started = time.perf_counter()
            ran_arms = 0
            with category_memo.category(name) as memo:
                for arm_index, entry in enumerate(category["arms"]):
                    retire(wait=False)
                    if failures:
                        break
                    arm = str(entry["arm"])
                    item = {"category": name, "arm": arm, "position": len(previous_items), "status": "running",
                            "stdout": str(entry["stdout"]), "stderr": str(entry["stderr"])}
                    record["items"].append(item)
                    _write_record(record_path, record)
                    arm_started = time.perf_counter()
                    item["started_unix"] = time.time()
                    ran = failed = False
                    try:
                        # Only the next item's private mapping is touched on
                        # the helper. Main owns environment/RNG/model state.
                        try:
                            dump, environment, built, host, setup_seconds = prepared.result()
                        finally:
                            entry_position += 1
                            prepared = (prep.submit(prepare, entries[entry_position])
                                        if entry_position < len(entries) else None)
                        item["host_prepare_seconds"] = float(setup_seconds)
                        with _arm_environment(environment), _arm_logs(entry["stdout"], entry["stderr"]):
                            kwargs, args = built if built is not None else build_call(dump["argv"], load=load)
                            output = Path(kwargs["output_dir"])
                            item["output_dir"] = str(output)
                            if (output / "receipt.json").exists():
                                item["status"] = "skipped_receipt_present"
                                continue
                            ran = True
                            ran_arms += 1
                            item["memory_start"] = worker_memory_boundary(retained, label="item_start")
                            retained.pop("worker_setup_memory", None)
                            _write_record(record_path, record)
                            phase_timing = (_PhaseTiming(enabled=True)
                                            if os.environ.get("IRG_V6_RUNTIME_ITEMS", "0") == "1" else None)
                            if phase_timing is not None:
                                phase_timing.host_output_path = str(entry["stderr"])
                                phase_timing.mark("process_start", utc_time=datetime.now(timezone.utc).isoformat(), shared_worker=True)
                                phase_timing.mark("imports_end")
                            probe = None if probe_path is None else probe_from_environment(probe_path(args))
                            context = {"schema": "irg_shared_worker_process_context_v1", "worker": identity,
                                       "position": len(previous_items), "category_position": category_index,
                                       "arm_position_in_category": arm_index,
                                       "previous_items": [list(pair) for pair in previous_items]}
                            if probe is not None:
                                probe.start()
                            try:
                                with _item_random_state(int(args.random_seed)), category_memo.arm_context(context):
                                    result = pilot_runner(
                                        **kwargs,
                                        **({} if phase_timing is None else {"phase_timing": phase_timing}),
                                        **({"prepared_host_inputs": host, "output_submit": writer.submit} if overlap else {}),
                                        retained_runtime_inputs=retained,
                                    )
                            finally:
                                if probe is not None:
                                    probe.stop()
                        # Bound the completion backlog to the preceding item
                        # plus this snapshot, even if storage is slower than GPU.
                        retire(wait=True)
                        if isinstance(result, DeferredItemProducts):
                            item["status"] = "writing_products"
                            pending.append((item, result, arm_started))
                        else:
                            completed(item, result, time.perf_counter(), arm_started)
                    except WorkerRefusal as error:
                        item.update(status="refused", error=f"{type(error).__name__}: {error}"[:400])
                        refused += 1
                    except (Exception, SystemExit) as error:
                        failed = True
                        fail(item, error)
                    finally:
                        if ran or failed:
                            if item["status"] != "completed":
                                item["wall_seconds"] = float(time.perf_counter() - arm_started)
                            previous_items.append([name, arm])
                            item["memory_end"] = worker_memory_boundary(retained, label="item_end")
                            setup_memory = retained.pop("worker_setup_memory", None)
                            if setup_memory is not None:
                                item["memory_setup"] = setup_memory
                            _close_item_planes(retained)
                            _release_item_runtime(retained)
                            if failed or failures:
                                retained.clear()
                            _reset_item_module_state(failed=failed or bool(failures))
                            between_arm_reset()
                            item["memory_post_cleanup"] = worker_memory_boundary(retained, label="item_post_cleanup")
                        _write_record(record_path, record)
                    if failures:
                        break
                retire(wait=True)
                events = list(memo.events)
            if failures:
                record["status"] = "arm_failed"
                record["categories"].append({"category": name, "memo_events": events})
                _write_record(record_path, record)
                first = failures[0]
                print(f"[worker] {first['category']}/{first['arm']} failed: {first['error']}", file=sys.stderr, flush=True)
                return EXIT_ARM_FAILED
            if not ran_arms:
                record["categories"].append({"category": name, "skipped": True})
                _write_record(record_path, record)
                continue
            ran_categories += 1
            category_cleanup()
            rss, hbm = _host_rss_bytes(), _device_allocated_bytes()
            if baseline is None:
                baseline = {"rss_bytes": rss, "hbm_allocated_bytes": hbm}
                record["post_cleanup_baseline"] = baseline
            record["categories"].append({
                "category": name, "wall_seconds": float(time.perf_counter() - category_started),
                "post_cleanup_rss_bytes": rss, "post_cleanup_hbm_allocated_bytes": hbm,
                "memory_post_cleanup": worker_memory_boundary(retained, label="category_post_cleanup"),
                "memo_events": events,
            })
            reasons = []
            if ran_categories >= int(recycle_categories):
                reasons.append("category_count")
            if rss is not None and baseline["rss_bytes"] is not None and rss - baseline["rss_bytes"] > recycle_rss_growth_bytes:
                reasons.append("host_rss_growth")
            if hbm is not None and baseline["hbm_allocated_bytes"] is not None and hbm - baseline["hbm_allocated_bytes"] > recycle_hbm_growth_bytes:
                reasons.append("hbm_growth")
            if reasons and category_index + 1 < len(categories):
                record["status"] = "recycle"
                record["recycle_reasons"] = reasons
                _write_record(record_path, record)
                return EXIT_RECYCLE
            _write_record(record_path, record)
        record["status"] = "completed" if not refused else "completed_with_refusals"
        _write_record(record_path, record)
        return EXIT_COMPLETED if not refused else EXIT_REFUSED
    finally:
        # A look-ahead failure is observed only if its item actually runs.
        # Joining all helpers makes every exit a clean worker boundary.
        prep.shutdown(wait=True, cancel_futures=True)
        writer.close()

def build_worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v6.3 shared-process worker")
    parser.add_argument("--worker-items", required=True)
    parser.add_argument("--worker-record", required=True)
    parser.add_argument("--recycle-categories", type=int, default=25)
    parser.add_argument("--recycle-rss-growth-gb", type=float, default=40.0)
    parser.add_argument("--recycle-hbm-growth-gb", type=float, default=4.0)
    return parser


def main() -> int:
    import sys

    from experiments.irg_v0_v2_pilots import paper_settings

    paper_settings.apply(os.environ)
    if "--worker-items" in sys.argv[1:]:
        worker_args = build_worker_parser().parse_args()
        if worker_args.recycle_categories < 1:
            raise SystemExit("--recycle-categories must be at least 1")
        return run_worker(
            worker_args.worker_items,
            worker_args.worker_record,
            recycle_categories=worker_args.recycle_categories,
            recycle_rss_growth_bytes=int(worker_args.recycle_rss_growth_gb * (1 << 30)),
            recycle_hbm_growth_bytes=int(worker_args.recycle_hbm_growth_gb * (1 << 30)),
        )
    args = build_parser().parse_args()
    pack = run_pack(args)
    completed = [a for a in pack["arms"] if a["status"] == "completed"]
    print(
        json.dumps(
            {
                "schema_name": PACK_RECEIPT_SCHEMA,
                "arms_completed": len(completed),
                "arms_total": len(pack["arms"]),
                "shared_startup_seconds": pack["shared_startup_seconds"],
                "total_wall_seconds": pack["total_wall_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0 if len(completed) == len(pack["arms"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
