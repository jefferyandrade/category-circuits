# Analysis code

| Folder | What |
|---|---|
| `stats/` | the statistics behind most of the paper's numbers: the method comparison, Table 1 and Figs. 3, 4B, 4D and 6 |
| `scripts/` | the frontier comparison (Fig. 3A), the loss thresholds (Table 1) and section 2.3 |
| `readout_control/` | the readout-weight control (section 2.4) |
| `input_divergence/` | input divergence and its references (Fig. 4C) |
| `animacy/` | the animacy of the kept connections (Fig. 5) |
| `circuit_activations/` | circuit activations against the category prototypes and the wiring (Fig. 7, section 2.5) |
| `common/` | modules several folders share: the shared categories and the category subgroups, the networks' layer layouts, the weight strata and the ResNets' block means |

Scripts run from the repository root as `python analysis/<folder>/<script>.py`, with no `PYTHONPATH`: each finds the
modules it imports in this repository. Each module's docstring says what it computes, for which result, and how to run
it. Inputs outside the repository (the circuit library and the intermediate results) are named by arguments or by the
variables in the top-level README. Scripts never overwrite a result: each refuses an existing output before doing any
work, and the restartable ones skip the records they already finished.

## `stats/`: the statistics

| Module | Computes | Paper |
|---|---|---|
| `category_resampling.py`, `pair_stats.py` | the category bootstrap, sign-flip tests, label permutations, the delete-one-category jackknife of category-pair statistics, and the Holm adjustment | throughout |
| `arms.py`, `arms_config.json` | where each network's circuits are, by pruning method ("arm") and threshold, and which receipt fields the statistics read | throughout |
| `circuit_masks.py`, `mask_gram.py` | the networks' weight layout and the checked mask reader; the exact count of kept weights each pair of circuits shares | throughout |
| `method_contrasts.py` | the pruning methods' means and paired contrasts on AlexNet's 1,000 categories | 2.1, Fig. 3A |
| `resnet50_pooled.py` | ResNet-50, E/I-separate against E/I-pooled pruning on the 99 categories where both circuits are complete: a per-category table read from the receipts, then the figure's rows | Fig. 3B |
| `table1_stats.py` | each network's kept fraction and corrected-accuracy drop on the shared categories | Table 1 |
| `sign_density.py`, `depth_trends.py` | the kept fraction of weights by sign, for each AlexNet layer and pruning method, and its trend over depth | Figs. 3C, 4A |
| `sign_magnitude.py` | weight counts and sizes by sign in AlexNet's checkpoint | 2.1 |
| `overlap_nulls.py` | the circuits' overlap at one layer against the size-, sign- and weight-size-matched references | Figs. 4B, 4D |
| `tree_share.py`, `tree_infer.py`, `tree_contrast.py` | the share of category-pair similarity explained by the category tree, for the wiring and for the mean category representations, plain and each beyond the other, with jackknife intervals and permutation tests; with `--wiring-minus-representations`, the paired difference W − A at each level; with `--exclude-superordinate`, the same without one superordinate group's categories, difference included | Fig. 6, 2.3 |
| `cross_network_agreement.py`, `cross_network_adjusted.py` | the agreement of AlexNet's wiring with each other network's, its ceiling from repeat extractions, and the agreement beyond the category tree and the representations | Fig. 6E |
| `make_logit_geometry.py`, `confusion_wiring.py` | the output-geometry control: the dense network's output geometry in place of the representations | 2.4 |
| `size_adapter.py`, `size_stats.py` | real-world size (big against small objects) beyond the category tree | 2.4 |

`arms_config.json` names the circuit library's folders with the placeholder `$CAMPAIGN`; point it at your copy of the
library. Its circuits are keyed by `price`, the threshold set: `v66` selects the paper's circuits and is the default
`--price`; `v64` and `v65` are earlier extractions that some comparisons read.

## `scripts/`: the frontier comparison, the thresholds and section 2.3

| Script | Computes | Paper |
|---|---|---|
| `frontier_analysis.py`, `matched_size.py`, `frontier_settings.json` | E/I-separate circuits extracted again at τ = 2.5e-7 (smaller circuits), compared category by category with each method's circuits at the paper's threshold on all 1,000 categories; the settings file fixes the categories, the bootstrap and the methods, and its sha256 is checked | Fig. 3A |
| `price_match.py` | each network's loss threshold τ, matched to AlexNet's from the threshold pilots (`data/thresholds/`) | Table 1, Methods |
| `parting_tree.py`, `parting_tree_stats.py`, `parting_controls.py` | where circuits part (a layer's kept fraction) against the category-tree structure of its wiring, across the layers of the four networks, with the controls for layer size and reliability | 2.3 |

## `readout_control/`: the readout-weight control (section 2.4)
At each network's last layer, a connection's pruning score is multiplied by its unit's readout weight to the target
category, so the late wiring could restate how similar the categories' readout weights are. This control measures the
wiring's category-tree structure beyond that similarity.

| File | What |
|---|---|
| `make_readout_geometry.py` | the categories' readout-weight similarity, signed and absolute, from the torchvision checkpoints |
| `run_readout_control.sh` | runs `stats/tree_contrast.py` with the readout geometry in place of the representations, for each network's last two layers (three for AlexNet: conv5, fc6 and fc7), with a shuffled-readout check (Slurm or local; its inputs are the variables in its header) |
| `summarize.py` | the summary table, `data/readout_control/SUMMARY_estimates.tsv`, from the per-task outputs |

## `input_divergence/`: input divergence (Fig. 4C)
When two circuits share a unit, input divergence measures how differently they choose the inputs into it. At a unit
with n inputs, two circuits that keep c1 and c2 of them, I in common, diverge by min(c1, c2) − I out of a possible
min(c1, c2) − max(0, c1 + c2 − n); a pair's input divergence is the summed divergence over the units where each circuit
keeps some but not all inputs, divided by the summed maximum. In convolutional layers a unit's inputs are its input
channels, kept if any of their kernel weights is kept. The references (Fig. 4C, dashes) replace I by its expectation
when each circuit chooses its inputs at random, among all of a unit's inputs (uniform) or within each decile of input
weight size (|w|, or a kernel's summed |w|). A layer's value is the mean over the pairs where divergence is defined,
with a delete-one-category jackknife interval, point ± t(0.975, Q − 1) × SE.

| File | Computes | Paper |
|---|---|---|
| `census.py` | AlexNet, all 1,000 categories: every pair's integer divergence sums at one layer, the exact mean over the eligible categories, the means within and across subgroups, and the extraction-noise floor from the repeat extractions (P-redraw) of the shared categories | Fig. 4C, AlexNet points |
| `alexnet_intervals.py` | AlexNet's points with their jackknife intervals, from the census's pair sums | Fig. 4C, AlexNet intervals |
| `alexnet_references.py` | AlexNet's uniform and weight-size-matched references and the observed divergence minus each, with paired jackknife intervals, from the census, the circuits' masks and the checkpoint | Fig. 4C, AlexNet references |
| `run_four_networks.py`, `network_sources.py` | every pruned layer of the four networks on the shared categories: the observed divergence with its interval, both references and the differences, from the circuits' masks and the checkpoints; layers where fewer than half of the pairs are defined are flagged and left out of the figure | Fig. 4C, VGG-11 and the ResNets |
| `divergence_math.py` | the estimators: exact pair sums over batches of units, the analytic references, the jackknife | Fig. 4C |
| `publish.py` | moves a finished output folder into place without replacing an existing one | |

```bash
# AlexNet: one census per layer (features.stage3.conv, stage4 and stage5 at unit_kernel_grain;
# readout.hidden1 and readout.hidden2 at unit_fanin), then the intervals and the references
python analysis/input_divergence/census.py --network alexnet --layer readout.hidden1 --grain unit_fanin \
    --masks-root LIBRARY_RESULTS --out CENSUS/readout.hidden1_unit_fanin
CC_CENSUS=CENSUS python analysis/input_divergence/alexnet_intervals.py --out INTERVALS
CC_CENSUS=CENSUS python analysis/input_divergence/alexnet_references.py --out REFERENCES
# The four networks on the shared categories: a table of sources, then every layer
CC_CENSUS=CENSUS python analysis/input_divergence/run_four_networks.py --mode catalog --workspace SOURCES --catalog CATALOG.json
python analysis/input_divergence/run_four_networks.py --mode full --catalog CATALOG.json --out FOUR_NETWORKS
# The ResNets' block means, from the figure table
python analysis/common/block_means.py --values data/figures/values_fig4_sharing.csv --panel C
```

LIBRARY_RESULTS is the circuit library's results folder. SOURCES holds the overlap runs' source table (`u_xnet.csv`)
and the kept fractions by sign (`xnet_sign_density.tsv`), against which every mask read is checked. Fig. 4C's AlexNet
rows join the references' table with the intervals; the other networks' rows are the four-network run's rows that are
not flagged, with the ResNets' block means. The census and the four-network run need the circuit library; the
references and the four-network run also need torch and the torchvision checkpoints.

## `animacy/`: animacy (Fig. 5)
Fig. 5 asks whether the circuits' kept connections prefer animate or inanimate categories. A kept connection of a layer
is kept by k_A of the n_A animate and k_I of the n_I inanimate categories' circuits; with p_A = k_A/n_A and
p_I = k_I/n_I, its specificity is s = (p_A − p_I)/(p_A + p_I) and its participation q = (p_A + p_I)/2. The layer test
compares the layer's mean |s| with a label-permutation null, which shuffles the animate and inanimate labels over the
categories, each layer seeded by its network and name. AlexNet is tested on all 1,000 categories (401 animate, 599
inanimate), with 5,000 permutations and 10,000 at the classifier layers; the cross-network panel uses the 100
categories extracted in every network (40 animate, 60 inanimate), with 10,000 permutations at every layer.

| File | Computes | Paper |
|---|---|---|
| `run_alexnet_animacy.py` | AlexNet, all 1,000 categories: the statistics from the circuits' incidence library (step `statistics`), then the plotted tables and the layer test (step `tables`) | Fig. 5A–B |
| `library_statistics.py`, `incidence_library.py` | each kept connection's specificity and participation, the permutation null, the participation curves with their max-T envelope and the joint count grids H[k_A, k_I], from an incidence library (one bit per category and connection, with each connection's sign); `build_library` writes a library from any mask reader | Fig. 5A–B |
| `specificity_plane.py` | the participation-specificity plane, its null (hypergeometric for each connection's number of categories) and the chance contours at 1%, 10% and 50% of the null's peak | Fig. 5A |
| `panel_tables.py`, `animacy_statistics.py` | the plotted tables; the permutation p values Holm-adjusted over the seven layers, the observed-minus-null mean \|s\| and the ResNets' block means | Fig. 5 |
| `run_shared_animacy.py`, `shared_specificity.py`, `shared_overlap.py`, `cell_histograms.py`, `incidence_package.py`, `pair_statistics.py` | each network's statistics on the shared categories, from its incidence package (the categories that keep each connection, as packed signatures): the specificity with its label-permutation null and jackknife, and the category-pair overlap against its size-matched reference | Fig. 5C |
| `read_shared_layers.py`, `collect_shared_layers.py` | the cross-network panel's rows from the stored per-layer statistics: observed and null mean \|s\|, their difference and the p values Holm-adjusted within each network | Fig. 5C |

```bash
# AlexNet, all 1,000 categories (40 permutation streams, as in the paper)
python analysis/animacy/run_alexnet_animacy.py statistics --library LIBRARY --out STATISTICS
python analysis/animacy/run_alexnet_animacy.py tables --statistics STATISTICS --out TABLES
# The four networks on the shared categories
python analysis/animacy/run_shared_animacy.py --incidence-package SHARED/handoff --out SHARED/finalization_pkg \
    --models alexnet vgg11 resnet18 resnet50
CC_SHARED_RUN=SHARED python analysis/animacy/read_shared_layers.py > LAYERS.json
python analysis/animacy/collect_shared_layers.py --layers LAYERS.json --density DENSITY.tsv --out ROWS.csv --checks CHECKS.json
```

In the run folder, `handoff/` holds the incidence package and `finalization_pkg/` the statistics, the layout
`read_shared_layers.py` reads. DENSITY.tsv holds each network's kept fractions by sign per layer, against which the
collector checks the packages. The permutation draws depend on the number of streams (`--nproc`, 40 in the paper). `cell_histograms.py` compiles a
small C++ loop into `analysis/animacy/scratch/` on first use; without a compiler a NumPy version gives the same result.
AlexNet's joint count grids are in `data/animacy/`, from which the plane, its null and the participation curves can be
recomputed without the circuit library. The parquet tables need pyarrow.

## `circuit_activations/`: circuit activations (Fig. 7, section 2.5)
Each circuit was run on every training image, with its mask applied at one layer and the layers below it dense, and its
activations at that layer were averaged per category; the dense network's category means at the same layer are the
category prototypes. The code here compares the category-tree structure of the circuits' activations with that of the
prototypes and of the wiring, from those saved category means.

| File | Computes | Paper |
|---|---|---|
| `layer_activations.py`, `layer_statistics.py` | at every pruned layer of a network: merges the per-shard sums into category means (`merge`), then the tree shares of the circuits' activations and of the prototypes and their difference, with a delete-one-category jackknife interval and a paired category-swap test (`aggregate`, `aggregate-all`) | Fig. 7D |
| `collect_layer_table.py` | one row per network and layer with both tree shares and their difference | Fig. 7D |
| `fc_all_images.py`, `fc_statistics.py` | AlexNet fc6 and fc7: the same comparison from every training image, with a jackknife interval (single categories at the shared 100, 100 random groups of ten at all 1,000) and a category-swap test; with `--population held_out`, on the 900 categories outside the shared 100 (90 random groups of ten) | Fig. 7B; Fig. 7D, AlexNet fc6 and fc7; 2.5, held-out check |
| `partial_tree_shares.py` | the category-tree structure each of the circuits' activations, the prototypes and the wiring carries beyond another, at fc6 and fc7 on all 1,000 categories | Fig. 7C |
| `wiring_minus_prototypes.py` | at one layer, the wiring's tree share minus the prototypes', with a joint delete-one-category jackknife interval | Fig. 7D, crossovers; Fig. 6, `C-test` rows |
| `whole_circuits.py` | the exploratory comparison with every layer's mask applied at once, at fc6 and fc7 | 2.5 |
| `layer_type_tests.py` | whether layer width, spatial pooling, the residual skip path or the ReLU explains why the difference appears only at fully connected layers: VGG-11's fc6 at 512 of its 4,096 units against conv5_2, conv5_2 position by position, ResNet-18's last block without the skip path, and fc6 before the ReLU in AlexNet and VGG-11 | 2.5 |
| `fig7_values.py` | the plotted rows of Figs. 7C and 7D from the outputs above: each layer's point, the ResNets' block means and the wiring crossovers (the shallowest layer whose wiring-minus-prototypes interval lies above zero) | Figs. 7C, 7D |

```bash
# Fig. 7D, per network: merge the image shards, then every layer's statistics
python analysis/circuit_activations/layer_activations.py merge --network NET --out RUN --prototypes PROTOTYPES
python analysis/circuit_activations/layer_activations.py aggregate-all --network NET --out RUN \
    --taxonomy taxonomy/inputs/category_taxonomy.csv
(cd AGGREGATION && python REPO/analysis/circuit_activations/collect_layer_table.py > site_tree_shares.tsv)
CC_FIGURE_INPUTS=FIGURE_INPUTS python analysis/circuit_activations/fig7_values.py D --out values_D.csv
# Fig. 7B and the AlexNet fc6 and fc7 points: every training image
python analysis/circuit_activations/fc_all_images.py --weight ALL_IMAGES --taxonomy taxonomy/inputs/category_taxonomy.csv --out FC.json
# Fig. 7C
python analysis/circuit_activations/partial_tree_shares.py --run ALL_IMAGES --wiring WIRING --taxonomy taxonomy/inputs/category_taxonomy.csv --out PARTIAL.json
python analysis/circuit_activations/fig7_values.py C --partial-shares PARTIAL.json --out values_C.csv
# Section 2.5: the held-out categories, and the layer-type tests
python analysis/circuit_activations/fc_all_images.py --weight ALL_IMAGES --taxonomy taxonomy/inputs/category_taxonomy.csv \
    --population held_out --out populations.tsv
python analysis/circuit_activations/layer_type_tests.py width --activations RUN/run/vgg11 --out width.json
python analysis/circuit_activations/layer_type_tests.py pooling --capture CAPTURE_64 --out pooling.json
python analysis/circuit_activations/layer_type_tests.py skip --capture CAPTURE_16 --out skip.json
python analysis/circuit_activations/layer_type_tests.py relu-alexnet --pool5 POOL5.npz --activations ALEXNET_SHARED --out relu_alexnet.json
python analysis/circuit_activations/layer_type_tests.py relu-vgg11 --capture CAPTURE_64 --out relu_vgg11.json
python analysis/circuit_activations/layer_type_tests.py table width.json pooling.json skip.json relu_alexnet.json \
    relu_vgg11.json --out LAYER_TYPES
```

AGGREGATION is the folder where `aggregate-all` wrote its per-layer results, and REPO this repository's folder.
FIGURE_INPUTS holds the three tables `fig7_values.py D` reads: the per-layer table, the layers' depths
(`data/sign_sharing/sign_sharing_four_networks_holm.tsv`) and the wiring-minus-prototypes table. The defaults are the
paper's settings: 64 image shards per network, 1,000 category swaps with seed 0, 50 label permutations with seed 0 for
every tree share, and random groups of ten categories (seed 0) for the jackknife at all 1,000 categories and on the
held-out categories. `layer_type_tests.py`'s docstring describes each reading and its inputs; CAPTURE_64 and
CAPTURE_16 are the GPU passes described below, POOL5 the dense AlexNet's category means of fc6's input.

The activations were captured on a GPU: torchvision AlexNet, VGG-11, ResNet-18 and ResNet-50 (IMAGENET1K_V1 weights) in eval
mode, every image of the ImageNet training set resized to 256 and center-cropped to 224, in 64 shards (each category's
images whose position modulo 64 is the shard's number) and batches of up to 256 images of one category. Each image
went once through the dense network; at each layer's site (its output after the ReLU, averaged over space in
convolutional layers; for a residual block's last convolution, the block output after the skip path is added), each
circuit was applied to the site's dense input with only that layer's (or block's) masks. Each image of a category
carried weight 1/(n_k K), so every category counts equally. Each shard saved, per site, the float64 sums of every
circuit's responses and of the dense network's, with the image counts and the even- and odd-image halves; a separate
dense pass saved the prototypes. For the layer-type tests, further passes read each category's first 64 (or 16)
images in a fixed hashed order and saved, per position: VGG-11's conv5_2 and fc6 before and after their ReLU, with
the circuit's mask on that layer and the dense input below it, and ResNet-18's last residual branch without the skip
path, with both of the block's convolutions masked; a hook on AlexNet's fc6 input saved the dense category means of
pool5 over every training image. The captures ran on the authors' evaluation code around the extraction engine,
which is not in this repository.

## `common/`: shared modules

| Module | What |
|---|---|
| `populations.py` | the 100 shared categories and each category's subgroup (a WordNet block of at most 50 categories), read from `taxonomy/inputs/CATEGORY_BLOCKS_cap50.tsv` |
| `architectures.py`, `route_layout.py` | each network's pruned layers on the connection ("route") axis of its circuit masks |
| `strata.py` | the weight strata of the sign- and weight-size-matched overlap reference, from the checkpoint's weights before batch-norm folding |
| `block_means.py` | the ResNets' block means: the mean of a per-layer value over each residual block's main-path convolutions |
| `atomic_io.py` | write-once outputs: each file is written beside its destination and linked into place without replacing anything; a script refuses an existing output, or an output that is also an input, before doing any work |
