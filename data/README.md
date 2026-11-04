# Data (CC BY 4.0)

Summary tables behind the paper's printed numbers, Table 1 and figures. Categories are ImageNet class indices (0–999);
`taxonomy/inputs/category_taxonomy.csv` names them. Intervals are 95%.

| Folder or file | What | Paper |
|---|---|---|
| `printed_numbers.tsv` | every number the paper prints (173), by the name of its LaTeX macro, with its value, sections and meaning | all |
| `method_comparison/` | AlexNet, all 1,000 categories: each circuit's kept fraction and corrected-accuracy drop under five pruning methods (`per_category_endpoints.csv`), their means with category-bootstrap intervals (`absolute_means.csv`) and paired contrasts (`method_contrasts.csv`, `direct_global_sign.json`) | 2.1, Fig. 3A |
| `table1/` | per-network kept fraction and accuracy drop with intervals (`table1.csv`), per-category dense and circuit accuracy (`fidelity_*.csv`) and prunable weights per layer (`layer_manifest.csv`); `table1.csv` also holds the rows of ResNet-50's IMAGENET1K_V2 checkpoint and the paired V1 − V2 contrast, the supplement's | Table 1 |
| `thresholds/` | the threshold pilots, in two rounds: per-category loss shares of AlexNet at its threshold and of the other networks at candidate thresholds (ResNet-50's on its IMAGENET1K_V2 checkpoint), the inputs of `analysis/scripts/price_match.py`; and `resnet50v1_check.csv`, the same quantities for ResNet-50's IMAGENET1K_V1 circuits on the same 21 pilot categories at that threshold, checked after the extraction | Table 1, Methods |
| `frontier/` | the frontier comparison (`pairing.csv`): E/I-separate circuits extracted at τ = 2.5e-7, paired category by category with each method's at the paper's threshold on all 1,000 categories, in connections kept, accuracy, hits, correct rejections and loss share (`verdict_family`: whether the family-wise intervals show the smaller circuits dominating) | Fig. 3A |
| `resnet50_pooled/` | ResNet-50, E/I-separate against E/I-pooled pruning on the 99 categories where both circuits are complete (category 454's E/I-pooled extraction stopped on a non-finite value): each circuit's kept fraction and the class-balanced binary accuracy of the dense network and of the circuit on held-out images, read from the receipts (`per_category.csv`), and the figure's rows (`values_fig3B_resnet50.csv`) | Fig. 3B |
| `sign_density/` | per-category kept fractions by sign and layer, AlexNet | Figs. 3C, 4A |
| `sign_sharing/` | inhibitory minus excitatory excess overlap per layer, four networks, with Holm-adjusted p values | Fig. 4D |
| `tree_share/` | category-tree structure of AlexNet's wiring, mean category representations and untrained networks, plain and partial, at fc6 and fc7 (`analysis/stats/tree_contrast.py` outputs); and the wiring minus the mean representations (`wiring_minus_representations.tsv`): at fc6 and fc7, the superordinate rows A, W and W−A that `tree_contrast.py --wiring-minus-representations` writes on all 1,000 categories, in that script's columns | Fig. 6A–B, 2.3 |
| `parting/` | per layer of every network, the kept fraction and the category-tree share of the wiring (`parting_tree.csv`), and their rank correlation with controls for layer size, reliability and depth (`parting_controls.csv`) | 2.3 |
| `readout_control/` | the readout-weight control: per-task `tree_contrast.py` outputs (`readout/`), the output-geometry control (`output_geometry/`) and the summary table | 2.4 |
| `robustness/` | Fig. 7B's comparison on the 900 categories outside the shared 100, at fc6 and fc7 (`populations.tsv`: `population`, `n_categories`, `layer`, the circuits' and the prototypes' superordinate tree shares, their `difference` with its jackknife `se` and the interval ± 1.96 SE, the category-swap `p`, and the numbers of deletion groups and swaps); and section 2.3's check without the mammals (`wiring_without_mammals.tsv`): at fc6 and fc7, the superordinate rows A, W and W−A that `analysis/stats/tree_contrast.py --exclude-superordinate mammal.n.01` writes on the 782 categories outside the mammals' group, in that script's columns | 2.3, 2.5 |
| `layer_types/` | the layer-type tests: one row per reading (`layer_type_tests.tsv`: `test`, `network`, `layer`, `reading`, `images_per_category`, the circuits' and the prototypes' tree shares, their `difference`, its jackknife `se`, the `critical_value` and the interval), and each reading's difference with each category deleted (`category_deletions.tsv`) | 2.5 |
| `animacy/` | AlexNet's joint count grids H[k_A, k_I], per layer: the number of kept connections that k_A of the 401 animate and k_I of the 599 inanimate categories' circuits keep (`domain_grids_alexnet1000.npz`, one 402 × 600 integer array per layer, keyed `<layer>\|H` with the circuit library's layer names, `features.0` to `classifier.4`) | Fig. 5 |
| `figures/` | the plotted values of each figure, one table per figure | Figs. 3–7 |

## Figure tables
Each row is one plotted value: its panel, network and layer, the quantity, the value and, where the figure shows one,
its interval. In the Fig. 6 and Fig. 7 tables, `lens` names what a category-tree share measures: `W` the wiring, `A`
the mean category representations, `U` the untrained networks' representations, `X|Y` the share of X beyond Y (`O` is
the outputs), and `W-A` their difference. Panels ending in `-test` hold the tests behind the figure's significance
marks. In every table, `arm` is the pruning method (the names table in the top-level README lists the codes), `price` is
its loss threshold τ, and `prototypes` are the mean category representations; `asterisk`, `colour_class` and
`display_depth` record how the figure draws a value.

In the Fig. 4 table, panel C's AlexNet rows (all 1,000 categories) carry the columns that
`analysis/input_divergence/alexnet_references.py` writes (`uniform_reference`, `weight_size_matched_reference`,
`observed_minus_uniform`, ...) and the other networks' rows those of `run_four_networks.py` (`ref_uniform`, `ref_weight`,
`obs_minus_uniform`, ...); `reference` is the weight-size-matched reference in both, and `block_mean` rows are the
ResNets' block means. In the Fig. 7 table, panel D's per-layer points keep the three decimals of the per-layer table
they were read from; the ResNet `block_mean` rows are the means of those values within each residual block. In panels
B and C, `weight` (or `computation`) is the circuits' activations and `image` the prototypes. Panel B's `key` is the
value's path in a summary of `fc_all_images.py`'s output; in the output itself it sits under
`differences/<layer>/weight_minus_image/metrics/tree_superordinate` (`left`, `right`, `difference`, and the
jackknife's `lo` and `hi`). Panel C's `key` is the value's path in `partial_tree_shares.py`'s output.
