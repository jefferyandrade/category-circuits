"""
Shared, architecture-agnostic defaults for the scoring workflow.

Import this module in each concrete config to inherit a coherent baseline, then
override anything model-specific.  Every option is documented inline so the
intent is clear without spelunking through the pipeline.

Important: each concrete config must expose `model` as a fully constructed
`torch.nn.Module` (not a dataclass wrapper) so the scoring pipelines can call
`named_modules()` without additional indirection.
"""

from typing import Dict

# ---------------------------------------------------------------------------
# Clustering defaults
# ---------------------------------------------------------------------------
# Parameters used by the optional frontier clustering pass (diagnostics/pruning).
# They do not affect the core actgrad scoring, but we keep them here for a
# consistent, config-wide baseline.

# Which clustering heuristic to apply (`info_submodular`, `kmeans`, etc.).
cluster_mode = None #"info_submodular"
# Rank of the low-rank sketch when approximating covariance matrices.
cluster_low_rank_k = 64
# Seed for the sketch random projection.
cluster_low_rank_seed = 0
# Hard cap on cluster size.
cluster_max_members = 2048
# Hard cap on the number of clusters to keep.
cluster_max_clusters = 4

# Weight of the information-theoretic objective term.
cluster_info_weight = 1.0
# Optional rank constraint for the information term (`None` = full rank).
cluster_info_rank = None
# Penalty discouraging redundant clusters.
cluster_info_redundancy_penalty = 0.1
# Minimum information gain required to accept a new cluster.
cluster_info_min_gain = 1e-4
# Additional curvature penalty (usually zero).
cluster_info_curvature_penalty = 0.0

# Equivalent knobs for curvature-aware clustering.
cluster_curv_rank = 64
cluster_curv_ridge = 1e-6
cluster_curv_weight = 0.5
cluster_curv_redundancy_penalty = 0.1

# Legacy heuristic tuning (kept for completeness)
cluster_selectivity = 0.1
cluster_magnitude = 0.05
cluster_positive_index = 0.5
cluster_negative_index = 0.5
cluster_boundary_gain = 0.05
cluster_gate_fraction = 0.3
cluster_suppressor = 0.05
cluster_gate_open = 0.05
cluster_kmeans_k = 0
cluster_kmeans_iters = 10
cluster_coupling_comm = 0.5
cluster_coupling_align = 0.3
cluster_coupling_synergy = 0.3

# ---------------------------------------------------------------------------
# Hessian / HVP defaults (quality-first)
# ---------------------------------------------------------------------------
# These govern the optional layer-wise Hessian-vector product sketches.  Leave
# them disabled (hvp_enabled=False) for pure streaming runs; enable them only
# when you explicitly need curvature information.

# Master switch for HVP computation.
hvp_enabled = False
# How frequently to compute HVPs (every N scoring batches).
hvp_every = 1
# Number of probe vectors injected per layer.
hvp_probes = 16
# Number of principal components kept from the probe responses.
hvp_top_k = 8
# Power iterations applied to refine the subspace estimate.
hvp_power_steps = 2
# Fraction of the batch used when sampling activations (`1.0` = full batch).
hvp_batch_frac = 1.0
# Random seed for probe generation.
hvp_seed = 0
# Whether to force deterministic behaviour (True) or allow faster stochastic runs.
hvp_deterministic = True
# Reuse the previous sketch as a warm start (saves compute, may drift slightly).
hvp_reuse_subspace = True

# ---------------------------------------------------------------------------
# Scoring behaviour
# ---------------------------------------------------------------------------
# Core knobs that control how actgrad scoring behaves.

# Per-sample objective used during backward; controls which signal the
# salience scores approximate. Options include:
#   "imagewise_activation_changes" (logit itself),
#   "unit_component_total_loss"   (cross-entropy component),
#   "log_prob"                    (log-softmax of the target unit).
target_type = "imagewise_activation_changes"
# Compute first-order forward gradients in addition to actgrad (rarely needed).
compute_forward_grads = False
# Flip sign of off-target contributions (useful for certain objectives).
invert_off_target = False
# Aggregate scores in absolute value (True) or signed form (False).
absolute = True
# Keep negative scores (True) or clamp them to zero (False).
score_neg = True
# Include convolutional layers in the scoring pass.
conv = True
# Include linear layers in the scoring pass.
linear = True
# Persist per-category activation data (turn off to save memory if unused).
save_target_category_data = False
# L2-normalise feature maps per image before scoring.
normalize_features_per_image = False
# Minimum activation threshold (0.0 → no thresholding).
output_threshold = 0.0
# Score post-activation tensors instead of pre-activations.
post_activation = False
# Compute neuron-level statistics alongside route/connection scores. This is
# part of the canonical production delivery package; use
# `--disable-neuron-metrics` only for lean reducer debugging.
compute_neuron_metrics = True
# Optional streaming statistics / HVP collectors; disable to skip their overhead.
collect_stats = False
collect_hvp = False
collect_bie = True
# Enable streamed reductions (chunked scoring).  Set False to fall back to the
# legacy full-batch path (discouraged).
streaming_reductions = False
# Enable batching for simple frontier layers (conv/linear without extra collectors).
scoring_fast_batch = False
# Fuse multiply/abs/reduction kernels when the CUDA extension is available.
scoring_use_fused = True
# Prefer torch.compile fallback when the fused kernel is unavailable.
scoring_use_compile = True
# Enable aggregated per-layer correlation logging (concatenate chunks then log once per layer).
log_layer_objective_correlation = True

# Boundary-integrated estimator (dataset-level inference) defaults.

bie_enabled = True  # Enable local boundary-integral corrections by default.
bie_correction_enabled = True  # Optional overshoot correction (off for speed unless needed).
bie_dataset_inference_enabled = False  # When true, accumulate dataset-level BIE summaries.
bie_dataset_samples_per_layer = 0  # Max sampled weights per layer (0 => use all elements).
bie_dataset_log_interval = 50  # Emit `[bie-dataset]` summaries every N updates (0 => suppress logs).
bie_dataset_track_variance = True  # Track second moments to report std estimates.
bie_dataset_clip = None  # Optional absolute clip applied before sampling (None => no clipping).
bie_dataset_sampling_strategy = "topk"  # "topk" (deterministic smallest-|z|) or "kernel" (stochastic).
bie_dataset_kernel = "gaussian"  # Kernel used when sampling_strategy == "kernel".
bie_dataset_kernel_width = 1.0  # Characteristic τ that defines how aggressively we focus near-boundary units.
bie_dataset_adaptive_tau = False  # If true, derive per-unit τ from |z|/|g|.
bie_dataset_tau_scale = 1.0  # Global multiplier applied to adaptive τ estimates.


# ---------------------------------------------------------------------------
# Layer gating / selection options
# (Legacy knobs retained for backward compatibility.)
# Minimum selectivity required for a score to count.
scoring_selectivity_threshold = 1e-6
# Which layers to score (`["all_layers"]` lets architecture configs provide their own lists).
scoring_layers_spec = ["all_layers"]
# Frequency (in batches) for GPU stats/telemetry; 0 disables periodic logging.
scoring_stats_every = 1
# Optional overrides for streamed chunk sizing.
streaming_chunk_default = None  # Global chunk override (int or None)
streaming_chunk_overrides: Dict[str, int] = {}

# ---------------------------------------------------------------------------
# Pruning defaults (quality-focused)
# ---------------------------------------------------------------------------
# Defaults consumed by downstream pruning scripts; actgrad itself doesn’t touch
# most of these knobs but we keep them here for consistency across configs.

score_set = "train"
eval_set = None
primary_metric = "binary_classification_loss"
norm = "minmaxnorm"
images_vs_scoring = "different_images_from_scoring"
correct_global_shift = "optimized_output_bias"
readjust = "no_readjust"
readout_layer_prune = "preserve_readout_layer"
pruning_score_type = "all_abs_scores"
num_eval_images = 10
prune_approach = "equalized_derivs"
peak_threshold = 1e-7
sequentially_pinned_layers = None
prune_type = "reverse_hierarchical"
final_sequential_layer = None
pruning_metric = "rank"

# ---------------------------------------------------------------------------
# Data loading / runtime (set architecture-specific paths in individual configs)
# ---------------------------------------------------------------------------
# Paths and loader-level knobs.  Model configs typically override these.

ffcv_cache_dir = None
ffcv_num_workers = None
ffcv_scan_workers = None
ffcv_scan_progress = None
ffcv_max_rows_per_class = None
FFCV = True
num_workers = 5
device = "cuda:0"
batch_size_dataloader = 1000
batch_size_inference = None
checkpoint_length = 20
# Let the configuration builder derive sensible tiered cache defaults; override with
# positive integers in model-specific configs if a different policy is required.
tiered_gpu_capacity = None
tiered_pinned_capacity = None

# Lightweight toggles for additional diagnostics.

# Mixed-precision execution (per forward chunk).
use_autocast = True
# Auto: prefer bf16 when supported, else fp16. Accepts "auto", "float16", "bfloat16".
autocast_dtype = "auto"
# Keep captured activations in fp32 for downstream numerics (disable to save memory).
keep_activations_fp32 = True


# ---------------------------------------------------------------------------
# Debugging knobs
# ---------------------------------------------------------------------------
# Emit verbose scoring logs.
scoring_debug = False
# Measure copy/compute overlap for streamed chunks (see docs for format).
stream_timing_debug = False
# Record peak memory per layer during scoring.
memory_debug = False
# Print per-batch runtime and classifier abs-score diagnostics. Keep disabled
# for production; it is useful when chasing a specific slow batch, but noisy.
debug_classifier_abs_scores = False
debug_classifier_layer = "classifier.6"
debug_batch_timing = False

# Profiling / GPU telemetry knobs
profile_memory = False
scoring_profile = False

# ---------------------------------------------------------------------------
# Weight clustering defaults (captured features + post hoc PCA/K-means)
# ---------------------------------------------------------------------------
weight_clustering_enabled = False  # Master switch for capturing features + running PCA/K-means.
weight_clustering_features = ()  # Explicit feature list; empty tuple falls back to schema order.
weight_clustering_embedding_dim = 16  # Target dimensionality for the PCA embedding.
weight_clustering_importance_feature = None  # Optional importance-weight feature name for PCA stats.
weight_clustering_device = "cuda"  # Device used for PCA + clustering computations.
weight_clustering_sample_size = 1_000_000  # Hard cap on the number of weights used during clustering.
weight_clustering_sample_fraction = None  # Fractional sampling rate (None disables fractional sampling).
weight_clustering_min_sample_size = 100_000  # Minimum weights retained after fractional sampling.
weight_clustering_max_iter = 50  # K-means iterations.
weight_clustering_num_clusters = 16  # Number of clusters to produce per layer.
weight_clustering_backend = "auto"  # Clustering backend: "auto", "torch", or "faiss".
weight_clustering_chunk_format = "pt"  # On-disk format for captured feature chunks ("pt" or "parquet").
weight_clustering_max_buffer_rows = 262_144  # Buffered rows before flushing features to disk.
weight_clustering_reuse_stats = True  # Reuse previously computed PCA statistics when available.
weight_clustering_pca_method = "auto"  # PCA solver strategy ("auto", "eigh", "randomized", "cusolver").
weight_clustering_pca_oversample = 4  # Oversampling factor when using randomized PCA.
weight_clustering_pca_power_iter = 2  # Power iterations for randomized PCA refinement.
weight_clustering_generate_masks = True  # Emit mask artifacts after clustering.
weight_clustering_layers = ()  # Restrict clustering to selected layers (empty tuple → all scored layers).
weight_clustering_sample_seed = 0  # Random seed applied to sampling.
weight_clustering_assignment_batch_size = None  # Optional override for batch size during centroid assignment.
weight_clustering_prefetch_chunks = None  # Optional override for number of embedding chunks to prefetch.
weight_clustering_artifact_subdir = "weight_clustering"  # Subdirectory (under the run base dir) for artifacts.

# ---------------------------------------------------------------------------
# Frontier / streaming integration defaults
# ---------------------------------------------------------------------------
# These flags drive the low-level frontier modules via environment variables.
# They live here so config authors have a single place to override them before
# job submission.  The frontier utilities consume the exported env vars but
# fall back to the values below when not present.

frontier_chunk_target_elements = 262_144
frontier_chunk_max_samples = 1000
frontier_compute_loss_delta = False
frontier_compute_bitpack = False
frontier_enable_attention_wrapper = False
frontier_enable_swiglu_wrapper = False
frontier_debug = False
frontier_validate = False

# ---------------------------------------------------------------------------
# Async writer / runtime toggles (config-controlled defaults)
# ---------------------------------------------------------------------------
# These mirror the SCORING_* environment variables consumed by
# packages.salience.pipeline.scoring. They are exported by the config builder so you can
# control them from configs instead of env.

async_cpu_writer = True                 # SCORING_ASYNC_CPU_WRITER
async_writer_compress = False           # SCORING_ASYNC_WRITER_COMPRESS
async_writer_compression_format = "gzip"  # SCORING_ASYNC_WRITER_COMPRESSION
async_writer_output_dir = None          # SCORING_ASYNC_WRITER_DIR
