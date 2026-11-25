# Benchmark sources and splits

The learner receives only `x_train` and `y_train`. The benchmark returns separate core and external evaluation arrays. Metadata with target membership, labels by class and source-row identifiers is for evaluation and audit. All benchmarks use the identity representation without pretraining. A learner-side scaler is fitted on the allowed core rows.

## Synthetic benchmarks

| Task | Observed core | External evaluation |
| --- | --- | --- |
| `rings_wedge` | Three ring arcs with angles within ±0.34π | Complementary angular region for the same rings |
| `curved_wedge` | Middle 46% of each of two curved paths | Both held-out ends |
| `anisotropic_narrow`, `anisotropic_wide` | Elliptical neighborhoods of component centers with respective latent radius limits 0.55 and 0.90 | Outside the selected limits, up to latent radius 2.20 |
| `disconnected` | Two of three components per class | Third component for each class. Its position competes with different observed classes |
| `paired_a`, `paired_b` | Identical observations and class labels on both tracks | Identical inputs, with every label complemented in `paired_b` |

Latent positions determine membership before observation noise. Measured features can cross latent cutoffs. Rings have both radial and Cartesian noise. The anisotropic tasks include six independent nuisance dimensions. The curves, anisotropic tasks and disconnected components use unequal training counts across classes and equal external counts. Synthetic targets come from the latent geometry. No generator force or parameters enter model fitting or tuning.

Each final seed generates new draws of training, in-core evaluation and external entities. The two paired worlds share the seed, inputs and sample identifiers by construction. For binary resolved predictions, the two paired balanced accuracies sum to one at every seed. With numerical failures counted as errors, the sum equals the mean resolved fraction across the two classes. The two anisotropic coverage variants share the same random draws transformed under different radius cutoffs.

## Optical Recognition of Handwritten Digits

The UCI authors' training collection has 3,823 rows from 30 writers. The independently held-out source has 1,797 rows from 13 different writers. The 64 columns count ink pixels over the source's 8×8 block representation. `optdigits.tra.gz` contains the original `optdigits.tra` bytes with gzip compression, without feature or label transformations. The held-out set is the scikit-learn `load_digits` copy of the UCI test file. The exact rows have zero duplicates within or across the two sources. Individual writer IDs and near-duplicate family identifiers are unavailable in these two data payloads. [UCI dataset and access terms](https://doi.org/10.24432/C50P49) · [scikit-learn loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)

The development evaluator operates only inside the UCI training source. Final evaluation partitions the author's locked test collection into five disjoint class-stratified row folds using a fixed permutation and the order of seeds in `gdm.data.FINAL_SEEDS`. Writers can recur across those five row folds, so repetition cannot quantify uncertainty over independently recruited writers. The two coverage levels for a seed share its evaluation fold and training candidate draw.

Attribution for the compressed copy goes to E. Alpaydin and C. Kaynak (1998), *Optical Recognition of Handwritten Digits*, UCI Machine Learning Repository. The original data are provided under CC BY 4.0. [Original training text](https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra) · [License](https://creativecommons.org/licenses/by/4.0/)

## Wine

The UCI Wine dataset contains 178 unique rows with 13 measured constituents and three cultivars. Source data come from the scikit-learn `load_wine` copy. The curator fixes a class-stratified row split with about 60% of rows for training by using seed 52420. Each final seed changes the allowed training core within this pool. All final seeds use the same approximately 40% holdout; the core/periphery membership of a row is allowed to change with the training core. Development evaluation uses a further split of the training pool. The source supplies no vintage, winery or related-entity grouping. The small repeated holdout limits generalization and underlies descriptive comparisons over seed draws. [UCI Wine dataset and CC BY 4.0 terms](https://doi.org/10.24432/C5PC7J) · [scikit-learn loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html)

## Public core selection

For each true class in the training pool, the curator samples up to 88 Digits or 24 Wine rows without replacement. The first draw is the class anchor, which will be admitted. Distances are Euclidean in raw pixels for Digits and in the elementwise natural log of the raw constituents for Wine. No scaler or full-class center is involved in these curator coordinates. The admission quota is 27% or 50% of sampled candidates, rounded to the nearest integer, with at least four rows per class. The radius is the distance of the quota-th closest same-class candidate. The radius includes all ties. Only admitted candidate rows are passed to the learner. A separate evaluation row is in-core when its distance from its true class's anchor falls within that radius, and is external otherwise. Labels are used by benchmark curation and scoring. Curator anchor identity, radius, class-specific metadata and evaluation row membership are withheld from fitting.

The evaluation subset depends on the curator's geometric definition and class labels. The two coverages can score different external rows from the same evaluation fold. The studies estimate extrapolation under those declared selections. They support no assumption about performance away from arbitrary definitions of distributional support.

## Snapshot verification

| Input verified before use | SHA-256 |
| --- | --- |
| Decompressed `optdigits.tra.gz` | `e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd` |
| scikit-learn packaged `digits.csv.gz` | `09f66e6debdee2cd2b5ae59e0d6abbb73fc2b0e0185d2e1957e9ebb51e23aa22` |
| scikit-learn packaged `wine_data.csv` | `a5f9961298dc86deafae12b9ee88a283508cefa03cd1f719e9dfe581ec898777` |

`gdm.data.make_task` provides stable IDs for the original public rows in `metadata['train_ids']`, `metadata['core_ids']` and `metadata['external_ids']`. The same row keeps its ID across seeds, split roles and coverage variants. Per-class counts and source digests remain in the metadata. The synthetic IDs describe draws in their generated split rather than an external population.
