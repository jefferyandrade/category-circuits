# category-circuits

[![tests](https://github.com/jefferyandrade/category-circuits/actions/workflows/tests.yml/badge.svg)](https://github.com/jefferyandrade/category-circuits/actions/workflows/tests.yml)

Companion code and summary data for

> Jeffery W. Andrade, Parisa A. Vaziri, Talia Konkle and George A. Alvarez (2026).
> **Beyond representations: the wiring of visual untangling in deep networks carries taxonomic category structure.**
> Manuscript in preparation for *Minds, Machines, and Brains*.

For every ImageNet category, the paper extracts a category circuit from a pretrained network: a mask over single
weights, never retrained, that preserves the network's recognition of that category, tested on held-out images.
Excitatory and inhibitory connections are pruned separately, in two queues that take turns, and a pruning step is kept
only if a sequential test shows that its cost stays below a loss threshold. Circuits were extracted for all 1,000
categories of AlexNet and for 100 shared categories of VGG-11 and ResNet-18 (99 of them in ResNet-50). The analyses
then compare the circuits' wiring, and the activations they produce, with the networks' mean category representations
against the WordNet category tree.

## Quick start

Python 3.11 (the analyses ran in 3.11.10). `requirements.txt` holds what the tests and the statistics need;
`requirements-full.txt` adds torch, torchvision and matplotlib for the engine and the scripts that read checkpoints.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests
```

The tests need no GPU, network access or circuit library, and take under a minute. They check 87 of the 174 numbers
the paper prints, and every number in Table 1, against the tables in `data/`. Method means and contrasts, accuracies
and accuracy drops with their intervals, and the loss thresholds are recomputed from per-category results with the
statistics code that produced the paper's numbers, and Fig. 3B's table byte for byte from per-category values read
from the circuits' receipts; sign densities are recomputed from per-category results, Holm adjustments from the saved
p values, and the layer-type tests' intervals from their category deletions; the rest, such as the kept fractions of
VGG-11 and the ResNets, the frontier intervals and the tree-share estimates, are read from the result tables. The tests
also rebuild the category tree and the readout-weight control's summary table byte for byte. For Figs. 4C, 5 and 7 they
recompute the derived values of the figure tables (block means, Holm adjustments, differences from references, chance
contours from AlexNet's joint counts), and they run the analysis code on small planted inputs.

## Layout

```
engine/      the circuit-extraction engine that made the paper's circuits, with a map from each Methods step to its code
analysis/    the statistics and the analyses behind every figure and printed number
taxonomy/    the WordNet category tree and the script that builds it
data/        summary tables behind the printed numbers, Table 1 and the figures
tests/       tests that recompute the paper's numbers from data/
```

## Names used in the code and data

| Name | Meaning |
|---|---|
| P | E/I-separate pruning, layer by layer: the paper's method |
| P-redraw | an independent repeat extraction of P with a new random seed, on the shared categories |
| SA-H (also SAH) | E/I-pooled pruning, layer by layer |
| SG, SA-G | E/I-separate and E/I-pooled pruning over all layers at once |
| RP1 | the random split: each layer divided at random into two groups of the true excitatory and inhibitory sizes |
| SF | E/I-separate pruning with the layers taken from input to output, run with the method comparison and not reported in the paper |
| arm | one pruning method |
| receipt | the record each extraction writes: its settings, its circuit's fidelity and its pruning log |
| census | the pass over every training image that scores connections and tests each pruning step |
| τ | a network's loss threshold for a pruning step (Table 1) |
| price | the loss threshold τ a circuit was extracted at; in the configuration and the per-category tables, the threshold set (`v66`: the paper's thresholds) |
| shared categories | the 100 categories extracted in every network (99 in ResNet-50) |

## Where each result lives

| Paper | Code | Data | Tested |
|---|---|---|---|
| Category tree (32 superordinate groups, 131 subgroups) | `taxonomy/build_taxonomy.py` | `taxonomy/inputs/` | rebuilt byte for byte |
| Circuit extraction (Methods, Fig. 2) | `engine/` | the circuit library (to be deposited) | |
| 2.1, Fig. 3A: connections kept and accuracy lost under five pruning methods | `analysis/stats/method_contrasts.py` | `data/method_comparison/` | recomputed |
| Fig. 3A: the frontier comparison | `analysis/scripts/frontier_analysis.py` | `data/frontier/` | re-read |
| Fig. 3B: ResNet-50, separate against pooled signs | `analysis/stats/resnet50_pooled.py` | `data/resnet50_pooled/` | recomputed |
| Figs. 3C, 4A: connections kept by sign and layer | `analysis/stats/sign_density.py` | `data/sign_density/` | recomputed |
| 2.1: weight counts and sizes by sign | `analysis/stats/sign_magnitude.py` | AlexNet's checkpoint | |
| Fig. 4B: overlap beyond its references | `analysis/stats/overlap_nulls.py` | `data/figures/values_fig4_sharing.csv` | |
| Fig. 4C: input divergence | `analysis/input_divergence/` | `data/figures/values_fig4_sharing.csv` | block means and differences from the references recomputed |
| Fig. 4D: sharing by sign | `analysis/stats/overlap_nulls.py` | `data/sign_sharing/` | Holm adjustment and counts recomputed |
| Fig. 5: animacy | `analysis/animacy/` | `data/figures/values_fig5_animacy.csv`, `data/animacy/` | Holm adjustments, excesses, permutation counts, envelope exceedances, block means and the cross-network rows recomputed; the plane, its null and the observed participation curves recomputed from the joint counts |
| Fig. 6: category-tree structure of wiring and representations | `analysis/stats/tree_share.py`, `tree_contrast.py`; Fig. 6E `cross_network_agreement.py`, `cross_network_adjusted.py`; the `C-test` rows `analysis/circuit_activations/wiring_minus_prototypes.py` | `data/tree_share/`, `data/figures/values_fig6_tree.csv` | AlexNet fc6 and fc7 re-read |
| 2.3: sparsity and category-tree structure | `analysis/scripts/parting_*.py` | `data/parting/` | recomputed |
| 2.4: the output-geometry control | `analysis/stats/make_logit_geometry.py`, `tree_contrast.py` | `data/readout_control/output_geometry/` | |
| 2.4: real-world size beyond the category tree | `analysis/stats/size_adapter.py`, `size_stats.py` | the printed values in `data/printed_numbers.tsv` (the size analysis's per-layer exports are not included) | |
| 2.4: the readout-weight control | `analysis/readout_control/` | `data/readout_control/` | summary rebuilt byte for byte |
| Fig. 7, 2.5: circuit activations | `analysis/circuit_activations/` | `data/figures/values_fig7_running.csv` | panel D's points, block means and crossovers, the crossover intervals and panel B's differences recomputed |
| 2.5: the comparison on the 900 held-out categories | `analysis/circuit_activations/fc_all_images.py --population held_out` | `data/robustness/` | printed values re-read; the categories and deletion groups recomputed |
| 2.3: the wiring minus the mean representations on all 1,000 categories | `analysis/stats/tree_contrast.py --wiring-minus-representations` | `data/tree_share/wiring_minus_representations.tsv` | the printed differences and intervals re-read, each interval recomputed; the W and A rows equal Fig. 6's tables |
| 2.3: the wiring against the mean representations without the mammals | `analysis/stats/tree_contrast.py --exclude-superordinate mammal.n.01` | `data/robustness/wiring_without_mammals.tsv` | the count and both intervals re-read; each difference and interval recomputed |
| 2.5: layer width, pooling, skip path and ReLU | `analysis/circuit_activations/layer_type_tests.py` | `data/layer_types/` | intervals recomputed from the category deletions; differences re-read |
| Table 1 | `analysis/stats/table1_stats.py`, `analysis/scripts/price_match.py` | `data/table1/`, `data/thresholds/` | every number |
| Every number the paper prints | | `data/printed_numbers.tsv` | 87 recomputed or re-read |
| The plotted values of Figs. 3–7 | | `data/figures/` (one table per figure) | |

In the Tested column, *recomputed* means a test recomputes the value from the shipped per-category or per-layer
results, and *re-read* means a test reads the stored result and checks it against the printed value.

`analysis/README.md` describes each analysis and how to run it, `data/README.md` each table, and `engine/README.md`
where each Methods step is computed and the settings the circuits were extracted with.

## Running against the circuit library

The analyses ran on a Slurm cluster over the circuit library and intermediate results (pair-similarity matrices,
activation captures) that are not in this repository. Scripts take their locations from arguments, or from these
variables:

| Variable | Meaning |
|---|---|
| `CC_STATS_OUT` | the statistics outputs: the pair matrices and per-network inputs the statistics read |
| `CC_AN_ROOT` | the layer-capture analyses holding VGG-11's and the ResNets' representation Gram folders, for the readout-weight control |
| `CC_CENSUS` | AlexNet's input-divergence census, one folder per layer and grain (`census.py --out`) |
| `CC_SHARED_RUN` | the shared-category animacy statistics that `analysis/animacy/read_shared_layers.py` reads |
| `CC_FIGURE_INPUTS` | the folder holding the three tables `analysis/circuit_activations/fig7_values.py` reads (default: the current folder) |
| `CC_CHECKPOINTS` | the torchvision checkpoints (default: `$TORCH_HOME/hub/checkpoints`) |
| `CC_TAXONOMY` | the category tree (default: `taxonomy/inputs/category_taxonomy.csv`) |
| `CC_PYTHON`, `CC_REPO` | the Python to run and this repository's folder, for the readout-weight control's Slurm script |

`analysis/stats/arms_config.json` names the circuit library's folders with the placeholder `$CAMPAIGN`, and the
engine's registry of dense-network baselines names its files under `$AUTHORS_STORE` (`engine/README.md`); point them at
your copies.

## Not included

- **The circuit library**: every circuit's weight masks, receipt (settings and fidelity) and pruning log, 84.3 GB.
  It is to be deposited in a public data archive under CC BY 4.0, and this README will give its DOI.
- **Intermediate results**: the pair-similarity matrices and activation captures the statistics read. `data/` holds
  the summary tables computed from them.
- **ImageNet** images and the **torchvision checkpoints**: download them from their sources.
- **Figure-drawing code.** Every plotted value is in `data/figures/`; the published figures' layout used a PDF library
  whose licence (AGPL) is incompatible with this repository's.
- **The readers that built Fig. 5's inputs** (the incidence library and package) from the circuits' masks, and **the
  GPU capture of circuit activations** for Fig. 7 and section 2.5: both ran on the authors' wider analysis code.
  `analysis/animacy/incidence_library.py` (`build_library`) writes an incidence library from any mask reader, and
  `analysis/README.md` describes the capture; everything from their outputs onwards is here.

## Code and the Methods

Where the code makes a choice that the paper's Methods leave implicit:

- **Weight strata use the checkpoint's weights before batch-norm folding.** The sign- and weight-size-matched overlap
  reference (Fig. 4B2, the within-sign comparison of Fig. 4D, and the wiring similarity of Fig. 6) places circuits at
  random within strata of sign and weight-size decile. Those strata come from the convolution weights as stored in the
  torchvision checkpoint, before batch-norm folding (`analysis/stats/overlap_nulls.py`). Extraction itself prunes by
  the folded signs. The difference matters only in the ResNets, since AlexNet and VGG-11 have no batch norm.
- **Two critical values for jackknife intervals.** The comparisons of AlexNet's fc6 and fc7 from every training image
  (Fig. 7B, AlexNet's fc6 and fc7 points in Fig. 7D, the held-out check of section 2.5), the partial tree shares of
  Fig. 7C, and the width and AlexNet ReLU tests of section 2.5 take the estimate ± 1.96 jackknife standard errors;
  the other jackknife intervals take t(0.975, n − 1) standard errors, except those of the five untrained networks
  (Fig. 6): their means (U, A|U and W|U) and the trained-minus-untrained difference A − U take ± 1.96 standard
  errors of a variance that adds the between-network variance over five to the jackknife variance of the mean or
  of the paired difference.
- **The pooling test's similarity** (section 2.5) is the mean, over conv5_2's 14 × 14 positions, of the correlation of
  two categories' 512 channels at each position.
- **ResNet-18's last block is read with both of its convolutions masked**, in the skip-path test of section 2.5 as at
  that block's point in Fig. 7D: a residual block's masks act together.
- **The defaults are the paper's settings.** The extraction's arm runner, and the pack runner's `--worker-items`
  path that runs each method through it, start every run with the settings of the paper's circuits
  (`engine/README.md`, Settings), such as Firth-penalized calibration, an exact census and each network's loss
  threshold; the analysis scripts' defaults are the paper's runs. Every circuit's receipt records the settings it
  ran with.

## Licences and citation

- Code: MIT (`LICENSE`).
- Data in `data/` and `taxonomy/inputs/`, and the circuit library when deposited: CC BY 4.0 (`LICENSE-DATA.md`).
- To cite this repository or the paper, see `CITATION.cff`.
