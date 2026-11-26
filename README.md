# Gravitational Decision Maps

**Attractor-based classification beyond observed training cores.**

This project investigates whether a dynamical classifier can generalize from densely sampled class regions to previously unobserved regions of the same classes. An input is treated as an initial state; its predicted class is determined by the attractor reached by the resulting trajectory.

The follow-up studies preservation of a fixed affine classifier under Gaussian prototype transport. Preservation at one positive horizon forces preservation at every horizon and can be tested by finite facet balances, including for component-specific Gaussian precisions. The short-time switch mass has a facet-flux formula with a quadratic remainder. A same-precision comparison measures when facet certificates accelerate validated integration. Strict margin checks also preserve a frozen activity prediction during transfer between paired PAMAP2 sensor channels. Supervised calibration lowers feature error on this source, while the Gaussian flow raises it. [Focused article](output/pdf/article.pdf) · [Technical manuscript](output/pdf/manuscript.pdf) · [LaTeX source](paper/manuscript.tex) · [Current independent audit](crossing/Q1-EXTERNAL-REVIEW.md) · [Prior review](crossing/Q1-REVISION-REVIEW.md)

The technical manuscript includes a matched comparison of static prediction, classical inertia, hard velocity clipping, and relativistic inertia. All four methods use the same scores and evaluation roster. The two capped methods share a ceiling, and the results also report how far their trajectories travel. [Inertia protocol](relativistic/study.json) · [Frozen inertia evidence](relativistic_results/frozen.json)

The repository contains the completed falsification pilot, its frozen protocol, both benchmark suites and the recorded results. The report gives the aggregate outcome and the evidence for each hypothesis. [Research report](RESEARCH.md)

The CIFAR‑100 experiment compares a CNN trained on native RGB images, Optuna-tuned CatBoost, linear heads, and Gaussian transport. On the official test, transporting the CNN features before a frozen 960-tree CatBoost head improves top‑1 from 66.09% to 70.00% and lowers top‑5 from 88.32% to 85.83%. The original CNN achieves 71.20% top‑1. [Experiment report and replay commands](crossing/CIFAR100-RESULTS.md) · [Benchmark figure](output/cifar100-benchmark.png)

## Research question

> When does learned attractor dynamics improve classification outside observed training cores compared with static classifiers given the same data, representations, and tuning budget?

The motivating picture is a collection of broad class basins around well-observed cores, with less familiar regions farther away. The project studies whether the geometry learned near those cores can support a useful continuation of class decisions.

The primary task is **classification of new examples from known classes**. Out-of-distribution detection and abstention are secondary questions and cannot substitute for demonstrating correct external classification.

## Study design

1. Estimate local class cores and one or more prototypes per class using only the allowed training data.
2. Construct or learn a vector field with stable class-associated attractors.
3. Classify a new input by its terminal attractor, while recording non-convergent trajectories explicitly.
4. Compare against strong static methods and existing dynamical classifiers on controlled training-core-to-periphery splits.
5. Identify which assumptions make the continuation useful and where it fails.

“Gravitational” describes the motivating analogy. The implemented classifiers use a softened mixture potential, gradient and damped inertial flows, and a feature-space adaptation of SA-nODE.

## What the comparisons test

- Reproducible improvement on held-out external regions across different geometries and real datasets.
- A benefit that remains when the representation, local metric, prototype count, and access to data are controlled.
- Ablations showing whether trajectories contribute beyond static scores or a single update.
- Reported computational cost, sensitivity to noise, convergence failures, and negative results.

The paired-world check uses identical core observations and complementary periphery labels to quantify the identification limit.

## Verify or reproduce

Python 3.13 and `uv` are needed for execution. The archived metrics can be checked using only the Python standard library.

```powershell
python tools/evidence_bundle.py
uv sync --frozen
uv run --frozen python -m unittest discover -s tests -v
```

For a new execution, use EpistemeOS commit `7f5a1a0e43d7c35196cc8b3c9c74a618aa2559c3` in a sibling checkout or pass `--episteme` to each command. The workflow registers both stages, runs them under locked dependency closure, and reanalyses the same per-example predictions with a separate implementation. A rerun yields new evidence and machine-specific timings.

The historical evidence tests use that same commit. Set `GDM_EPISTEME` to the historical checkout when the sibling checkout is on the newer inertia-study commit.

```powershell
uv run --frozen python -m gdm.evidence prepare --online --root .research/replay
uv run --frozen python -m gdm.evidence freeze --workers 1 --root .research/replay
uv run --frozen python -m gdm.evidence run --stage synthetic --root .research/replay --results .research/replay-export
uv run --frozen python -m gdm.evidence run --stage real --root .research/replay --results .research/replay-export
python tools/evidence_bundle.py --export-dir .research/replay-export --results .research/replay-bundle
uv run tools/plot_results.py --results .research/replay-bundle --output .research/replay-bundle/figures
```

## Documents

| File | Contents |
| --- | --- |
| [README.md](README.md) | Project overview and current status |
| [RESEARCH.md](RESEARCH.md) | Results, mathematical setup, and assessment of H1–H4 |
| [protocol.json](protocol.json) | Locked tasks, configuration, contrasts, and decision criteria |
| [Benchmark sources](benchmark_data/README.md) | Data snapshots, licenses, core assignments, and grouping |
| [Frozen manifest](results/frozen.json) | EpistemeOS registrations and exact scientific input hashes |

The follow-up evidence has a separate source lock. Verify the archived run and regenerate the manuscript figures with:

```powershell
python -m crossing.evidence verify
python tools/plot_crossing.py
```

Use `python -m crossing.evidence freeze --help` and `run --help` for a fresh run against the pinned EpistemeOS checkout. To typeset the source with Tectonic, run `tectonic --outdir ../output/pdf manuscript.tex` from `paper/`.

### Affine decision study

The affine studies use the locked Python runtime. The public certificate replay treats the train-standardized inputs, fitted centroids and bandwidths as exact binary64 encodings. It evaluates a whole-path stability certificate and an endpoint-switch certificate without a solver. Sampled DOP853 paths are independent numerical comparisons. The three-source drivers verify source SHA-256 hashes and cache downloads under `.research/flat-boundary-data/`. Keep latency reruns in scratch paths because clock measurements vary across hosts and runs.

```powershell
uv sync --frozen
uv run --frozen python -m tools.affine_certificate_transfer --output .research/affine-certificate-replay.json
uv run --frozen python -m tools.affine_interval_comparison --output .research/affine-interval-replay.json
uv run --frozen python -m tools.affine_score_comparison --output .research/affine-score-comparison-replay.json
uv run --frozen python -m tools.switching_mass_validation --output .research/switching-mass-replay.json
uv run --frozen python -m tools.affine_span_replay --repeats 15 --output .research/affine-span-replay.json
uv run --frozen python -m tools.decision_safe_restoration --output .research/restoration-replay.json
uv run --frozen python -m tools.restoration_robustness --output .research/restoration-robustness-replay.json
```

[Verified public decisions](crossing_results/affine_certificate_transfer.json) · [Same-precision validation and timings](crossing_results/affine_score_comparison.json) · [Decimal comparison](crossing_results/affine_interval_comparison.json) · [Switch-mass calibration](crossing_results/switching_mass_validation.json) · [Endpoint span replay](crossing_results/affine_span_replay.json) · [Decision-preserving restoration](crossing_results/decision_safe_restoration.json) · [Sensitivity to noise](crossing_results/restoration_robustness.json)

The PAMAP2 experiment refits an affine activity head and paired calibrators in eight leave-one-subject-out folds. The data loader caches the 688 MB source archive under `.research/pamap2/` and checks its SHA-256. Pass `--archive PATH` to use an existing verified archive. No EpistemeOS setup is required for this experiment or the same-precision comparison. The public dataset is CC BY 4.0. [Source and citation](https://doi.org/10.24432/C5NW2H) · [Locally fixed protocol](crossing/pamap2_protocol.json) · [Results and code digests](crossing_results/pamap2_restoration.json)

```powershell
uv run --frozen python -m tools.pamap2_restoration --output .research/pamap2-replay.json
uv run --no-project --script tools/plot_pamap2.py
```

The earlier sensitivity check integrates a perturbed two-dimensional mixture with a fixed classifier. The interval reference uses outward-rounded decimal arithmetic on exact two-dimensional fixtures. The original external evaluation and scaling study document the coverage and latency of the floating-point guard.

```powershell
uv sync --frozen
uv run --frozen python -m tools.flat_boundary_sensitivity
uv run --frozen python -m tools.flat_boundary_interval
uv run --frozen python -m tools.flat_boundary_transfer
uv run --frozen python -m tools.flat_boundary_transfer --timings-output .research/flat-boundary-timings-replay.json
uv run --frozen python -m tools.flat_boundary_scaling --output .research/flat-boundary-scaling-replay.json
uv run --no-project --script tools/plot_flat_boundary.py
```

[Sensitivity observations](crossing_results/flat_boundary_sensitivity.json) · [Interval comparison](crossing_results/flat_boundary_interval.json) · [External evaluation and data provenance](crossing_results/flat_boundary_transfer.json) · [External timing](crossing_results/flat_boundary_timings.json) · [Synthetic scaling](crossing_results/flat_boundary_scaling.json)

The [focused article](output/pdf/article.pdf) contains the current theory and application. The [technical manuscript](output/pdf/manuscript.pdf) adds the controlled, core-only and inertia studies. The [short companion](output/pdf/letter.pdf) summarizes the original isotropic result and its direct tests. Build each PDF with `tectonic --outdir ../output/pdf` followed by `article.tex`, `manuscript.tex` or `letter.tex` from `paper/`. OpenAI Codex assisted with candidate proofs, independent agent checks, implementation, analysis and drafting. Author names and the declaration of responsibility require author confirmation before submission. [SIMODS author instructions](https://epubs.siam.org/journal/simods/instructions-for-authors) · [SIAM AI policy](https://epubs.siam.org/artificial-intelligence)

### Inertial comparison

Verify the archived observations, independently recompute the headline comparisons, and check the manuscript tables with:

```powershell
python -m relativistic.evidence verify
python tools/inertia_tables.py --check
```

For a fresh run, the inertia study uses EpistemeOS commit `98bfdde732a2122796313b62c6d8ecbeb532a92d`. Use a sibling checkout or pass `--episteme` to `prepare`, `freeze`, and `run`. Run the two stages serially for latency measurement. A fresh run records machine-specific timings.

```powershell
uv sync --frozen
$inertiaStore = '.research/relativistic-replay-run'
$inertiaExport = '.research/relativistic-replay-export'
uv run --frozen python -m relativistic.evidence prepare --online --root $inertiaStore
uv run --frozen python -m relativistic.evidence freeze --root $inertiaStore
uv run --frozen python -m relativistic.evidence run --stage relativistic_synthetic --root $inertiaStore --export $inertiaExport
uv run --frozen python -m relativistic.evidence run --stage relativistic_public --root $inertiaStore --export $inertiaExport
uv run --frozen python -m relativistic.evidence bundle --export $inertiaExport --results .research/relativistic-replay
python tools/inertia_tables.py --results .research/relativistic-replay --output .research/relativistic-replay/inertia-tables.tex
```

## Video

The Russian explainer *Can a Flow Change an Algorithm's Answer?* visualizes the decision-invariance theorem, its counterexample, and the external validation. The release includes the upload master, captions, thumbnail, and replayable sound and animation sources. [Video and manuscript release](https://github.com/AdamCage/gravitational-decision-maps/releases/tag/v0.2.0) · [Video source and evidence](output/decision-invariance-video/README.md)

## Starting points in the literature

- [SA-nODE: Stable Attractors for Neural Networks Classification via Ordinary Differential Equations](https://arxiv.org/abs/2311.10387) — classification through learned basins of planted stable attractors.
- [SODEF: Stable Neural ODE with Lyapunov-Stable Equilibrium Points for Defending Against Adversarial Attacks](https://proceedings.neurips.cc/paper_files/paper/2021/hash/7d5430cf85f78c4b7aa09813b14bce0d-Abstract.html) — equilibrium stability and robustness to perturbations.
- [How Neural Networks Extrapolate: From Feedforward to Graph Neural Networks](https://arxiv.org/abs/2009.11848) — assumptions governing generalization outside training support.

The research plan positions the project against these approaches. The proposed experimental regime is a question to investigate, not a verified claim of first-of-its-kind novelty.
