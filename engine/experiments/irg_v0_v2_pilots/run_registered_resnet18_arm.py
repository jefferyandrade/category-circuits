#!/usr/bin/env python3
"""Run one authoritative modern queue-v0 registered pilot arm."""

from __future__ import annotations

import time as _phase_clock
_PROCESS_START_MONOTONIC = _phase_clock.monotonic()
_PROCESS_START_UNIX = _phase_clock.time()

import argparse
import atexit
from datetime import datetime, timezone
import json
import os
import sys
from typing import Any, Callable, MutableMapping, Sequence

from circuitflow.artifacts import ArtifactStore
from experiments.irg_v0_v2_pilots import paper_settings
from mechanistic_systems.circuits.irg.modern_v0_runtime_profile import (
    modern_v0_runtime_profile,
    resolve_modern_v0_sensitivity_tau,
)
from mechanistic_systems.circuits.irg.precision import IndependentSampleSizePolicy
from mechanistic_systems.circuits.irg.registered_resnet18_pilot import (
    _PhaseTiming,
    V0_DEFAULT_ATTRIBUTION_ORDER,
    V0_TRAIN_ONLY_MASK_LOCK_RECEIPT_SCHEMA,
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
from mechanistic_systems.inference.sequential_bce import (
    IID_WITH_REPLACEMENT_SAMPLING_ID,
    ORACLE_SAMPLING_LAWS,
    SEALED_WOR_CLASS_STRATIFIED_SAMPLING_ID,
)
from mechanistic_systems.measurement.contexts import ContextAxis
from mechanistic_systems.measurement.memory_probe import probe_from_environment
from mechanistic_systems.measurement.model_sources import ModelSourceReplica


def _parse_current_version(raw: str) -> str:
    """Validate the qualified identity while preserving legacy bare internals."""

    canonical = resolve_current_workflow_identity(raw).canonical_identity
    return canonical.rsplit("/", 1)[-1]


def _result_summary(result: dict) -> dict:
    """Keep the ordinary runner output stable while admitting mask-lock mode."""

    if result.get("schema_name") == V0_TRAIN_ONLY_MASK_LOCK_RECEIPT_SCHEMA:
        return {
            "receipt_hash": result["receipt_hash"],
            "version_name": result["version_name"],
            "retained_fraction": result["retained_fraction"],
            "retained_route_count": result["retained_route_count"],
            "terminal_validation_opened": False,
            **({"loss_weighting": "per_image"} if result.get("loss_weighting") == "per_image" else {}),
            "output_dir": result["output_dir"],
        }
    return {
        "receipt_hash": result["receipt_hash"],
        "version_name": result["version_name"],
        "partition_family": result["partition_family"],
        "retained_fraction": result["retained_fraction"],
        "terminal_final_bce": result["terminal_final_bce"][
            "per_image_binary_bce" if result.get("loss_weighting") == "per_image"
            else "class_balanced_binary_bce"],
        **({"loss_weighting": "per_image"} if result.get("loss_weighting") == "per_image" else {}),
        "output_dir": result["output_dir"],
    }


def build_parser() -> argparse.ArgumentParser:
    """The arm CLI. The shared-process worker parses each arm's argv with it."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        type=_parse_current_version,
        default="v0",
        metavar="irg/v0",
        help=(
            "Workflow version: v0 (also written irg/v0), the paper's method and the default. "
            "Later versions are not supported and fail closed."
        ),
    )
    parser.add_argument(
        "--partition-family",
        choices=(
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
        ),
        default="local_attribution_sign",
    )
    parser.add_argument("--partition-seed", type=int)
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--train-only-mask-lock",
        action="store_true",
        help=(
            "Stop canonical irg/v0 at the exact final-mask lock, export its "
            "typed receipt, and do not perform terminal preparation or "
            "validation. This mode also disables terminal and step matrices."
        ),
    )
    parser.add_argument("--historical-mask-path")
    parser.add_argument(
        "--measurement-artifact-policy",
        choices=("required", "contracts_only"),
        default="contracts_only",
    )
    parser.add_argument("--measurement-plan")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--model-source-replica")
    parser.add_argument(
        "--model-transform",
        default=None,
        help=(
            "v5.1: path to a registered model_transform.json (R03 "
            "zero_routes_v1 or GAUGE positive_unit_rescale_v1). The sealed "
            "loader re-derives and verifies the transformed model on the "
            "unchanged base Study; the transform is hash-bound into the model "
            "identity, cache keys and receipts. Absent keeps the v5 path."
        ),
    )
    parser.add_argument("--optimization-uint8-cache-manifest")
    parser.add_argument("--optimization-uint8-cache-receipt")
    parser.add_argument("--evaluation-uint8-cache-manifest")
    parser.add_argument("--evaluation-uint8-cache-receipt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--scoring-batch-rows", type=int, default=None)
    parser.add_argument("--calibration-batch-rows", type=int, default=None)
    parser.add_argument("--v0-paired-oracle-batch-rows", type=int, default=None)
    parser.add_argument("--v1-candidate-oracle-batch-rows", type=int, default=None)
    parser.add_argument("--terminal-batch-rows", type=int, default=None)
    parser.add_argument(
        "--native-panel-device-cache-bytes",
        type=int,
        default=0,
        help=(
            "Byte cap for the optional physical-image GPU LRU. Modern IID "
            "v0 defaults it off because measured first-look uniqueness is "
            "above 95%%; projected-logit slabs retain the useful reuse."
        ),
    )
    parser.add_argument("--panel-prefetch-depth", type=int, default=2)
    parser.add_argument("--panel-producer-workers", type=int, default=1)
    parser.add_argument("--unit-packing-degree", type=int, default=1)
    parser.add_argument("--query-tile-size", type=int, default=1)
    parser.add_argument(
        "--require-optimized-native-panel", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--mps-enabled",
        action="store_true",
        help=(
            "Record that this process is one unit inside a launcher-owned "
            "NVIDIA MPS packing allocation."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--maximum-commits",
        type=int,
        default=None,
        help=(
            "Explicit canary/debug trajectory cap. Omit for the canonical "
            "full precision-resolved pruning traversal."
        ),
    )
    parser.add_argument("--maximum-relative-bce-increase", type=float, default=0.20)
    parser.add_argument(
        "--sensitivity-tau",
        type=float,
        help=(
            "Modern-v0 fractional dense-to-empty BCE span consumed per "
            "ablated circuit connection. Required here or through "
            "IRG_SENSITIVITY_TAU."
        ),
    )
    parser.add_argument("--terminal-maximum-per-stratum", type=int, default=256)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument(
        "--maximum-prune-order-uncertainty-fraction",
        type=float,
        default=0.001,
        help=(
            "Maximum metric-global activity mass whose retained/pruned "
            "membership may change at an audited executable score boundary. This "
            "is an order-equivalence tolerance, not an error probability."
        ),
    )
    parser.add_argument(
        "--memory-probe-path",
        default=None,
        help=(
            "Where IRG_MEMORY_PROBE writes its trace. MUST NOT be inside "
            "--output-dir: creating a file there creates the directory, and the "
            "pilot refuses an output directory that already exists, so the "
            "probe would kill every arm it was armed on. Defaults to a sibling "
            "of the arm directory."
        ),
    )
    parser.add_argument(
        "--prune-order-consequence-gated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Gate the prune order on the CONSEQUENCE PRICE against its own "
            "derived floor instead of the activity-mass fraction above. The "
            "mass bar's smallest nonzero step in a scope is one route's share, "
            "and at coarse granularity features.stage1.conv's largest route "
            "carries 142%% of the whole 0.001 budget -- so a single boundary "
            "swap cannot respect it and the bar is unsatisfiable by arithmetic. "
            "On by default: this changes what the census certifies "
            "about, and is recorded per-arm by acquisition_policy_hash."
        ),
    )
    parser.add_argument(
        "--response-acquisition-envelope-seconds",
        type=float,
        default=None,
        help=(
            "Wall-clock envelope for ONE hard-partition response-field "
            "acquisition. Operational fail-safe only -- it is not a sample-size "
            "target and enters no confidence statement. Unset reproduces prior "
            "behaviour exactly, in which an unresolvable field runs to the job "
            "wall and writes NOTHING: no receipt, no partial field, no "
            "diagnosis. Arming it converts that into a disclosed unresolved "
            "certificate naming the bar that bound."
        ),
    )
    parser.add_argument(
        "--oracle-acquisition-envelope-seconds",
        type=float,
        default=None,
        help=(
            "Wall-clock envelope for ONE proposal's paired-oracle acquisition "
            "(sibling of --response-acquisition-envelope-seconds). Operational "
            "fail-safe only: never refuses the first look, enters no confidence "
            "statement, and unset reproduces prior behaviour exactly. When the "
            "median per-row cost of completed looks predicts the next look "
            "cannot fit, the proposal returns a typed unresolved decision with "
            "an oracle_acquisition_envelope_receipt_v1 disclosure."
        ),
    )
    parser.add_argument(
        "--v0-oracle-sampling-law",
        choices=ORACLE_SAMPLING_LAWS,
        default=None,
        help=(
            "Proposal-adjudication oracle sampling law for modern v0. Default "
            f"for v0 is {SEALED_WOR_CLASS_STRATIFIED_SAMPLING_ID!r}: each class "
            "stratum is a sealed uniform permutation of its finite training "
            "population revealed as a nested prefix, so the paired oracle "
            "terminates at census with the exact finite-population "
            f"correction. {IID_WITH_REPLACEMENT_SAMPLING_ID!r} replays the "
            "historical unbounded with-replacement law byte for byte. "
            "Non-v0 versions accept only the with-replacement id."
        ),
    )
    parser.add_argument("--partition-geometry-cache-dir")
    parser.add_argument("--dense-readout-table-cache-dir")
    parser.add_argument(
        "--enable-local-regime-diagnostics",
        action="store_true",
        help="Opt into expensive gate/regime diagnostics; off for ordinary pilots.",
    )
    parser.add_argument(
        "--v0-calibration-decision-estimand",
        choices=(
            "frozen_extraction_calibration_risk_v1",
            "accepted_state_frozen_candidate_refinable_v1",
            "population_optimal_affine_risk_v1",
        ),
        default="frozen_extraction_calibration_risk_v1",
        help=(
            "Declared v0 decision estimand. The candidate-refinable policy "
            "legalizes candidate-calibration purchases before acceptance and "
            "carries the candidate improvement bound in the decision "
            "envelope; changing this changes the declared scientific "
            "estimand and requires a fresh comparison namespace."
        ),
    )
    parser.add_argument(
        "--v0-accept-replication-gate",
        action="store_true",
        help=(
            "Require every provisional accept to replicate on a disjoint "
            "confirm stream before it commits (one capped confirm query per "
            "proposal, accepts only). Filters winner's-curse accepts at the "
            "threshold."
        ),
    )
    parser.add_argument(
        "--v0-oracle-minimum-per-stratum",
        type=int,
        default=2,
        help=(
            "Per-stratum minimum for the class-resolved paired-oracle "
            "design; raises the per-decision evidence floor (2 gives the "
            "historical ~2304-row first look, 8 gives ~4x). Config-only "
            "panel-power lever for the seed-variance experiment."
        ),
    )
    parser.add_argument(
        "--terminal-namespace-salt",
        help=(
            "Rotate the sealed terminal validation panel identity for this "
            "run so earlier import-only readouts under the shared run "
            "identity carry no cumulative exposure."
        ),
    )
    parser.add_argument(
        "--v0-schedule-replay-spec",
        help=(
            "Path to the sealed historical schedule-replay spec JSON. When "
            "set, the v0 arm replaces the tau traversal with the recorded "
            "group schedule (pre-registered probe); unset keeps the tau "
            "path byte-identical."
        ),
    )
    parser.add_argument(
        "--v0-schedule-replay-rank-source",
        choices=("family", "local_attribution"),
        default="family",
        help=(
            "Within-sign-group replay ranks: 'family' uses the family "
            "arm's own activity magnitudes (A3 primary cell); "
            "'local_attribution' uses modern local-attribution magnitudes "
            "within the same sign groups (secondary cell)."
        ),
    )
    parser.add_argument(
        "--v0-attribution-order",
        choices=(
            "plain_global_attribution",
            "corrected_aumann_shapley_global_attribution",
        ),
        default=V0_DEFAULT_ATTRIBUTION_ORDER,
        help=(
            "Within the v0 sign partitions, rank connections by the plain "
            "global attribution (the paper's score and the default) or by the "
            "path-integrated Aumann-Shapley attribution, available for AlexNet "
            "only. Neither mode changes partition, calibration, oracle, or "
            "walker semantics."
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
            "Registered v1 ordering key (partition commensurability theory "
            "Section 3, c_r == 1): 'abs_raw' keeps the incumbent activity "
            "order byte-identical; K1/K2/K3 order the ladder by expected "
            "damage, damage energy, or whitened salience.  v1-only; keys "
            "never carry commit authority."
        ),
    )
    parser.add_argument(
        "--v1-span-budget-fraction",
        type=float,
        default=0.066145,  # registered accept_rule.span_budget_fraction
        help=(
            "v1 ladder acceptance budget as a fraction of the certified "
            "dense-to-empty loss span (matching v0's span-normalized tau "
            "convention).  The historical dense-relative budget form is "
            "retired from the v1 path; --maximum-relative-bce-increase "
            "keeps its v0/tau and per-decision-bound roles untouched."
        ),
    )
    parser.add_argument(
        "--terminal-category-matrix",
        choices=("on", "off"),
        default="on",
        help=(
            "A53: capture the terminal (context-stratum x output-logit) "
            "category-structure matrices for the dense reference and final "
            "masks as sidecar artifacts referenced from the receipt. 'on' "
            "is the registered default; 'off' records only the "
            "{enabled: false} receipt block."
        ),
    )
    parser.add_argument(
        "--step-category-matrix",
        choices=("on", "off"),
        default="off",
        help=(
            "'on' runs the v0 paired-oracle decision lanes with the "
            "full-width readout projection (the same estimand on a different "
            "floating-point path) and records the per-step category-structure "
            "stack (raw paired-delta matrices, frozen-dense calibrated frames "
            "and the anchored three-level hierarchical tracker) as one sidecar "
            "of the step_category_structure_v1 receipt block. 'off', the "
            "setting of the paper's circuits and the default, records only the "
            "{enabled: false} block. Versions other than v0 always run it off."
        ),
    )
    parser.add_argument(
        "--calibration-deployment-refit-anchor",
        choices=("selected", "unregularized"),
        default="selected",
        help=(
            "Deployment-refit objective for extraction calibration fits: "
            "'selected' keeps the crossfit-selected identity-anchor strength "
            "in the deployed objective (reference behavior); 'unregularized' "
            "adds one lambda=0 refit so the raw BCE gradient vanishes at "
            "interior deployed optima and the attainable-improvement "
            "certificate loses its anchored first-order floor."
        ),
    )
    return parser


def load_contract(kind: str, path: str) -> Any:
    """Load one launch input. The worker memoizes these across its arms."""

    if kind == "study":
        return StudySpec.load(path)
    if kind == "circuit_plan":
        return CircuitPlan.load(path)
    if kind == "profile":
        return StatisticalSequentialProfile.load(path)
    if kind == "promotion_receipt":
        return load_native_sampled_path_promotion_receipt(path)
    if kind == "context_axis":
        return ContextAxis.load(path)
    if kind == "artifact_store":
        return ArtifactStore(path)
    if kind == "model_source_replica":
        return ModelSourceReplica.load(path)
    raise ValueError(f"unknown launch input kind {kind!r}")


def build_pilot_call(
    argv: Sequence[str] | None = None,
    *,
    load: Callable[[str, str], Any] = load_contract,
    environment: MutableMapping[str, str] | None = None,
) -> tuple[dict[str, Any], argparse.Namespace]:
    """Parse ``argv`` as the arm CLI and return the pilot's keyword arguments.

    This is the one argument builder: the CLI and the shared-process worker
    both call it, so no option can differ between them. It also applies the
    environment the CLI sets: the CIRCUIT_* rows and, for v0, the settings of
    the paper's circuits (``paper_settings``). The worker supplies its private next-item environment so host
    preparation can run while the current arm uses the process environment.
    """

    preparing_in_worker = environment is not None
    environment = os.environ if environment is None else environment
    parser = build_parser()
    if preparing_in_worker:
        # argparse normally writes errors to process stderr. A preparation
        # thread must report its own failure through the future instead of
        # writing into the currently running arm's redirected logs.
        def preparation_error(message: str) -> None:
            raise ValueError(f"invalid prepared arm arguments: {message}")

        parser.error = preparation_error
    args = parser.parse_args(argv)
    study = load("study", args.study)
    if args.version == "v0":
        # The settings of the paper's circuits fill in whatever the environment leaves unset.
        paper_settings.apply(environment, study.model_id)
        try:
            args.sensitivity_tau = resolve_modern_v0_sensitivity_tau(
                args.sensitivity_tau, environ=environment
            )
        except ValueError as error:
            parser.error(str(error))

    modern_v0_profile = modern_v0_runtime_profile(study.model_id)
    runtime_defaults = {
        "batch_size": modern_v0_profile.execution_batch_rows,
        "scoring_batch_rows": modern_v0_profile.scoring_batch_rows,
        "calibration_batch_rows": modern_v0_profile.calibration_batch_rows,
        "v0_paired_oracle_batch_rows": (modern_v0_profile.v0_paired_oracle_batch_rows),
        "v1_candidate_oracle_batch_rows": (
            modern_v0_profile.v0_paired_oracle_batch_rows
        ),
        "terminal_batch_rows": modern_v0_profile.terminal_batch_rows,
    }
    for name, value in runtime_defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, int(value))

    runtime_environment = {
        "CIRCUIT_SCORING_BATCH_ROWS": args.scoring_batch_rows,
        "CIRCUIT_CALIBRATION_BATCH_ROWS": args.calibration_batch_rows,
        "CIRCUIT_V0_PAIRED_ORACLE_BATCH_ROWS": (args.v0_paired_oracle_batch_rows),
        "CIRCUIT_V1_CANDIDATE_ORACLE_BATCH_ROWS": (args.v1_candidate_oracle_batch_rows),
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
        "CIRCUIT_MPS_ENABLED": int(args.mps_enabled),
    }
    for name, value in runtime_environment.items():
        environment[name] = str(value)
    if args.mps_enabled and not environment.get("CUDA_MPS_PIPE_DIRECTORY"):
        parser.error("--mps-enabled requires CUDA_MPS_PIPE_DIRECTORY")
    if args.v0_oracle_sampling_law is None:
        args.v0_oracle_sampling_law = (
            SEALED_WOR_CLASS_STRATIFIED_SAMPLING_ID
            if args.version == "v0"
            else IID_WITH_REPLACEMENT_SAMPLING_ID
        )
    elif (
        args.version != "v0"
        and args.v0_oracle_sampling_law != IID_WITH_REPLACEMENT_SAMPLING_ID
    ):
        parser.error(
            "--v0-oracle-sampling-law other than "
            f"{IID_WITH_REPLACEMENT_SAMPLING_ID!r} is admitted only for v0"
        )

    policy = IndependentSampleSizePolicy(
        confidence_level=args.confidence_level,
        maximum_prune_order_uncertainty_fraction=(
            args.maximum_prune_order_uncertainty_fraction
        ),
        precision_only_iid=args.version in {"v0", "v1"},
        oracle_maximum_contexts_per_candidate=(
            None if args.version in {"v0", "v1"} else 4096
        ),
        prune_order_consequence_gated=bool(args.prune_order_consequence_gated),
        response_acquisition_envelope_seconds=(
            args.response_acquisition_envelope_seconds
        ),
        oracle_sampling_law=str(args.v0_oracle_sampling_law),
        oracle_acquisition_envelope_seconds=(
            args.oracle_acquisition_envelope_seconds
        ),
        # Coverage is already carried by the simultaneous confidence family;
        # this is operational hysteresis only.  The registered fast profile
        # uses one resolved look, while the default remains two.
        response_stable_looks=int(environment.get("IRG_RESPONSE_STABLE_LOOKS", "2")),
    )
    kwargs: dict[str, Any] = dict(
        version_name=args.version,
        partition_family=args.partition_family,
        partition_random_seed=args.partition_seed,
        comparison_namespace=args.comparison_namespace,
        query_id=args.query_id,
        study=study,
        plan=load("circuit_plan", args.circuit_plan),
        profile=load("profile", args.profile),
        promotion_receipt=(
            None
            if args.promotion_receipt is None
            else load("promotion_receipt", args.promotion_receipt)
        ),
        optimization_axis=load("context_axis", args.optimization_axis),
        evaluation_axis=load("context_axis", args.evaluation_axis),
        optimization_image_root=args.optimization_image_root,
        evaluation_image_root=args.evaluation_image_root,
        store=load("artifact_store", args.artifact_root),
        output_dir=args.output_dir,
        historical_mask_path=args.historical_mask_path,
        measurement_artifact_policy=args.measurement_artifact_policy,
        measurement_plan_path=args.measurement_plan,
        checkpoint_path=args.checkpoint_path,
        model_source_replica=(
            None
            if args.model_source_replica is None
            else load("model_source_replica", args.model_source_replica)
        ),
        optimization_uint8_cache_manifest=(args.optimization_uint8_cache_manifest),
        optimization_uint8_cache_receipt=args.optimization_uint8_cache_receipt,
        evaluation_uint8_cache_manifest=args.evaluation_uint8_cache_manifest,
        evaluation_uint8_cache_receipt=args.evaluation_uint8_cache_receipt,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_seed=args.random_seed,
        maximum_commits=args.maximum_commits,
        maximum_relative_bce_increase=args.maximum_relative_bce_increase,
        sensitivity_tau=args.sensitivity_tau,
        terminal_maximum_per_stratum=args.terminal_maximum_per_stratum,
        sample_size_policy=policy,
        enable_local_regime_diagnostics=(args.enable_local_regime_diagnostics),
        partition_geometry_cache_dir=args.partition_geometry_cache_dir,
        dense_readout_table_cache_dir=args.dense_readout_table_cache_dir,
        calibration_deployment_refit_anchor=(args.calibration_deployment_refit_anchor),
        terminal_namespace_salt=args.terminal_namespace_salt,
        v0_calibration_decision_estimand=(args.v0_calibration_decision_estimand),
        v0_oracle_minimum_per_stratum=(args.v0_oracle_minimum_per_stratum),
        v0_accept_replication_gate=args.v0_accept_replication_gate,
        # The registered v0 identity OWNS the adjudication sampling law (as it
        # owns precision-only stopping); the CLI reaches it only through the
        # replay override, which the mask lock receipts.
        v0_oracle_sampling_law_override=str(args.v0_oracle_sampling_law),
        v0_schedule_replay_spec=args.v0_schedule_replay_spec,
        v0_schedule_replay_rank_source=(args.v0_schedule_replay_rank_source),
        v0_attribution_order=args.v0_attribution_order,
        v1_ordering_key=args.v1_ordering_key,
        v1_span_budget_fraction=args.v1_span_budget_fraction,
        terminal_category_matrix=(
            False
            if args.train_only_mask_lock
            else args.terminal_category_matrix == "on"
        ),
        step_category_matrix=(
            False if args.train_only_mask_lock else args.step_category_matrix == "on"
        ),
        train_only_mask_lock=bool(args.train_only_mask_lock),
        **(
            {}
            if args.model_transform is None
            else {"model_transform_path": args.model_transform}
        ),
    )
    return kwargs, args


def memory_probe_path(args: argparse.Namespace) -> str:
    # Armed by IRG_MEMORY_PROBE; None (and no file created) when disarmed.
    # Flushed PER SAMPLE, which is what makes it survive the failure it exists
    # to diagnose: a cgroup OOM kill is SIGKILL, so there is no unwinding and no
    # atexit, and anything still buffered would be lost. Explicit stop() is
    # registered for the normal-exit path only. This is the only reliable in-run
    # view -- mid-run sstat was measured reporting 8.3 GiB for a job whose
    # completed sacct MaxRSS was 358.7 GiB.
    #
    # THE TRACE MUST NOT LAND IN --output-dir. Opening a file there CREATES the
    # directory, and the pilot then refuses at `destination.exists()` with
    # `immutable pilot output exists` -- so arming the probe guaranteed the arm
    # died at startup. MEASURED across the whole results tree: 198 arms carry a
    # receipt and NOT ONE carries a probe trace, because the two are mutually
    # exclusive by construction. The instrument had never run.
    #
    # Default is therefore a sibling of the arm directory, not a child of it.
    # --memory-probe-path overrides, and the sbatch points it at the job-id
    # keyed logs directory where the rest of this job's diagnostics already live.
    return args.memory_probe_path or os.path.join(
        os.path.dirname(os.path.abspath(args.output_dir)),
        f"{os.path.basename(os.path.normpath(args.output_dir))}.memory_probe.jsonl",
    )


def main(argv: Sequence[str] | None = None) -> int:
    paper_settings.apply(os.environ)
    phase_timing = (
        _PhaseTiming(enabled=True)
        if os.environ.get("IRG_V6_RUNTIME_ITEMS", "0") == "1"
        else None
    )
    if phase_timing is not None:
        phase_timing.mark(
            "process_start",
            monotonic_seconds=_PROCESS_START_MONOTONIC,
            utc_time=datetime.fromtimestamp(_PROCESS_START_UNIX, timezone.utc).isoformat(),
        )
        phase_timing.mark("imports_end")
    axes_loaded = 0

    def timed_load(kind: str, path: str) -> Any:
        nonlocal axes_loaded
        if kind == "context_axis" and phase_timing is not None and axes_loaded == 0:
            phase_timing.mark("context_axis_load_start")
        value = load_contract(kind, path)
        if kind == "context_axis":
            axes_loaded += 1
            if phase_timing is not None and axes_loaded == 2:
                phase_timing.mark("context_axis_load_end")
        return value

    kwargs, args = build_pilot_call(argv, load=timed_load)
    memory_probe = probe_from_environment(memory_probe_path(args))
    if memory_probe is not None:
        memory_probe.start()
        atexit.register(memory_probe.stop)
    result = run_registered_resnet18_pilot(
        **kwargs,
        **({} if phase_timing is None else {"phase_timing": phase_timing}),
    )
    print(json.dumps(_result_summary(dict(result)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
