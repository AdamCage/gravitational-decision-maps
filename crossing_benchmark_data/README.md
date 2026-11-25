# Crossing benchmark sources

The synthetic cases and public transfer cases serve different comparisons. Calls to model fitting should pass only `x_train` and `y_train`. The labeled target arrays and curator metadata are for evaluation. Both sets use the source coordinates without pretraining and permit a common training-only scaler.

## Held-out region on generated shapes

The synthetic suite draws 64 training examples and 120 evaluation examples for each of three known classes. The generator draws an invertible affine map and shape variation from each family and seed. Evaluation occupies a fixed latent interval from −0.34 to +0.34. Each class supplies 32 training examples on each side in bands of width 0.95 that start 0.48, 0.80 or 1.12 units from the origin. These choices give nominal separations of 0.14, 0.46 and 0.78 units from the fixed test support. The units are radians for the shell families and generator coordinates for the other families. The same seed reuses geometry, test features and labels, training draws and observation noise across the three band placements. Moving the bands also changes the shape and scaled distance of the measured training cores.

| Family | Target geometry | Features |
| --- | --- | --- |
| `affine_shells` | Three elliptical shells under a common affine map | 2 |
| `wavy_shells` | Three closed shells distorted by three-, five- and two-cycle harmonics | 2 |
| `open_chains` | Three nearby curved tracks | 2 signal coordinates and 4 independent nuisance coordinates |
| `local_lobes` | Three separated sets around class-specific centers | 2 |

Noise is added after latent membership is assigned. A latent gap therefore does not specify a distance in observed feature space. The test class is the generator's known class throughout. No generator coefficients or source forces are supplied to fitting.

## Pen-Based Recognition of Handwritten Digits

The UCI authors collected normalized pen trajectories from 44 writers, representing each digit by eight coordinate pairs. Their original training-side file contains 7,494 examples from 30 writers. The test file contains 3,498 examples from the other 14 writers. Our development evaluator stays inside the 30-writer source. Final evaluation uses the other writer group. An optional fixed final seed roster partitions the writer-held-out test source by class into disjoint row folds. Writers can recur across row folds because individual writer IDs are unavailable in these two payloads. The files have a different representation and source from the block-pixel Digits collection used in the earlier pilot. [UCI dataset](https://doi.org/10.24432/C5MG6K)

For each digit, the curator draws up to 80 training-side rows. The first draw is the anchor, and the core radius is the distance to the 28th nearest candidate, including the anchor. The rule admits all boundary ties. Distances use Euclidean geometry on the author's normalized coordinates. The evaluation pool uses its known labels to keep rows beyond the corresponding class radius. A balanced sample contains at most 36 eligible rows per class. Exact duplicates in any source retain their first original row ID. Exact feature vectors shared with the original training-side source are removed from the writer-held-out evaluation pool before fold selection.

Attribution goes to E. Alpaydin and F. Alimoglu (1996), *Pen-Based Recognition of Handwritten Digits*, UCI Machine Learning Repository. Both source files are unchanged except for gzip packaging. [Training text](https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra) · [Held-out text](https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Seeds

Seeds contains 210 rows with seven measured geometric features and three known wheat varieties. The curator uses a fixed class-stratified training and held-out row assignment, with approximately 58% for training. Development evaluation uses only the training-side pool. Field and batch identifiers are unavailable, limiting the inference to this declared row split. Final seeds reuse the held-out pool and vary choices inside the training pool. [UCI dataset](https://doi.org/10.24432/C5H30K)

For each class, the curator draws up to 25 training-side candidates. The first draw is the anchor and the core radius admits at least the ten closest candidates, including any boundary ties. Distances use the natural log of the positive source measurements only during curation to compare proportional differences. The fit receives the unlogged source features. A balanced sample contains up to 20 eligible external evaluation rows per class. Exact duplicate feature vectors retain their first original row ID.

Attribution goes to M. Charytanowicz, J. Niewczas, P. Kulczycki, P. Kowalski and S. Lukasik (2010), *Seeds*, UCI Machine Learning Repository. The source file is unchanged except for gzip packaging. [Source text](https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Snapshot hashes

The loader checks SHA-256 of the original decompressed bytes before parsing them.

| Source file | SHA-256 |
| --- | --- |
| `pendigits.tra` | `e2b9eb9f0d0467e2b64a4816a3420edf2b8043447576f4b84337aba44a9f97d3` |
| `pendigits.tes` | `8bd03229c5c5291fefe43e45465dd948d2645bf23328b9d993e0b777666b2015` |
| `seeds_dataset.txt` | `1f3f83c0d8485ae9148061389d19628607e3f5660e3d6f40ec9102fb398bb12f` |
