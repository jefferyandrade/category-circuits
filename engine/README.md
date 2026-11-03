# The extraction engine

The code that extracted the paper's circuits, at the version that made them: every file its entry points import,
without one optional speed path (fused convolution kernels and the numerical package they use). It is not maintained
here.

Running an extraction needs a GPU, the ImageNet training set, the torchvision checkpoints and each network's census
inputs: the dense network's logits on every training image, class by class, with their row hashes. It also reads
each network's dense-baseline receipts, which `experiments/irg_v0_v2_pilots/modern_v0_cross_architecture_default100_20260813/dense_baseline_registry.json`
lists under `$AUTHORS_STORE`; set `AUTHORS_STORE` to the folder that holds your copies (their hashes are checked). The scripts that
staged those inputs and submitted the jobs are not included.

## Patches
The engine as shipped made the AlexNet, VGG-11 and ResNet-18 circuits. Two patches give the versions that made the
rest; apply one from the repository's root with `git apply --directory=engine engine/patches/<name>.diff`, or
`patch -p1 -d engine < engine/patches/<name>.diff` outside a git checkout.
- `patches/firth_iteration_ceiling.diff` lets Firth calibration fits run up to 2,000 iterations instead of 64; fits
  that converge within 64 iterations are unchanged. It made the ResNet-50 circuits, the last ResNet-18 circuits and
  the E/I-pooled circuits of VGG-11 and ResNet-18.
- `patches/device_mask_holder.diff` fixes how a shared mask holder hands its device copy between consumers
  (`mechanistic_systems/circuits/device_mask_plane.py`). It was applied for one category (464) of the E/I-pooled
  comparison at a second threshold (τ = 1.25e-7, Fig. 3A) and changes no results.

## The Methods in code

| Methods step | Where it is computed |
|---|---|
| Score every connection: the class-balanced mean of \|w ∂ℓ<sub>k</sub>/∂w\| over the training images | the ranking statistic is `loss_activity`, formed in `mechanistic_systems/circuits/irg/registered_resnet18_pilot.py` from each census look's snapshot (`combine_stratified_route_decision_snapshots` in `mechanistic_systems/measurement/device_route_moments.py`), which weights the target and non-target class strata one half each (`_binary_loss_weights`). The order is the default `--v0-attribution-order plain_global_attribution` (see Settings) |
| Calibrate the target logit's readout: an affine fit, Firth-penalized for the paper's circuits | `mechanistic_systems/backends/device_calibration_fit.py`, `fit_cells` |
| The step test's interval: a Student-t interval with sampling and calibration uncertainty and a χ² second-order allowance | `mechanistic_systems/inference/device_bce.py`, `look_statistics`; the host version is `mechanistic_systems/inference/sequential_bce.py`, `scalar_v0_decision_interval` |
| Keep, refuse or leave unresolved a pruning step; an interval still straddling the threshold at an exact census counts as a refusal | `mechanistic_systems/circuits/evidence_bce_precision.py`, `_decide_bce_precision_step` |
| Excitatory and inhibitory queues take turns, and an unresolved step freezes its sign's queue for the layer | `mechanistic_systems/circuits/evidence_bce_precision.py`, `_frontier_step_to_next_proposal` |
| Corrected accuracy on held-out images: the hit rate minus the false-alarm rate | `mechanistic_systems/measurement/terminal_statistics.py`, `terminal_statistics_v1` |
| Run a list of categories and pruning methods on one GPU | `experiments/irg_v0_v2_pilots/run_registered_resnet18_pack.py` (`--worker-items`; despite its name it serves every network), which runs each method with `run_registered_resnet18_arm.py` |

## Settings
By default the arm runner (`experiments/irg_v0_v2_pilots/run_registered_resnet18_arm.py`), and the pack runner's
`--worker-items` path that runs each method through it, run with the settings of the paper's circuits.
`experiments/irg_v0_v2_pilots/paper_settings.py` lists them: the same for every network, plus each network's loss
threshold and census batch size. The options' defaults select the paper's method. A setting given in the environment
or on the command line takes precedence, and every circuit's receipt in the circuit library records the settings it
ran with. The ones the Methods describe:

| Setting | Default (the paper's) | Meaning |
|---|---|---|
| `IRG_SENSITIVITY_TAU` | AlexNet 1.5e-7, VGG-11 1.768e-7, ResNet-18 5e-7, ResNet-50 4.033e-7 | the loss threshold (Table 1) |
| `IRG_CALIBRATION_PENALTY` | `firth` | Firth-penalized calibration fits |
| `IRG_LOSS_WEIGHTING` | `balanced` | the class-balanced loss |
| `IRG_CENSUS_PRECISION_EPS` | `0` | an exact census from the start |
| `IRG_CENSUS_SAMPLE_DESIGN` | `importance_batches_v1` | the importance-batch census |
| `IRG_CENSUS_BATCH_SIZE`, `IRG_CENSUS_CAPTURE_ROWS` | 1024 AlexNet, 512 VGG-11 and ResNet-18, 256 ResNet-50 | census batch size |
| `IRG_ANALYTIC_SIGN_PARTITION` | `1` | the census's analytic sign partition |
| `IRG_SUBSTITUTED_LOSS_WEIGHTING` | `per_image` | per-image loss weighting |
| `IRG_TERMINAL_CALIBRATION_RULE` | `walk_resolution` | final calibration after mask selection |
| `IRG_LAYER_EXECUTION_ORDER` | `forward` | prune layers deep to shallow (AlexNet: fc7 first, conv1 last); `reverse` goes from the input side |
| `--partition-family` | `local_attribution_sign` | E/I-separate pruning (other values give the comparison methods) |
| `--v0-attribution-order` | `plain_global_attribution` | rank connections by the first-order score in the Methods; a path-integrated (Aumann–Shapley) order is available for AlexNet |

Speed switches, such as `IRG_FUSED_ROUTE_MOMENTS` and `IRG_PREFIX_REUSE`, do not change the circuits and keep the
engine's own defaults.
