# Gravitational Decision Maps

**Attractor-based classification beyond observed training cores.**

This project investigates whether a dynamical classifier can generalize from densely sampled class regions to previously unobserved regions of the same classes. An input is treated as an initial state; its predicted class is determined by the attractor reached by the resulting trajectory.

The repository contains the completed falsification pilot, its frozen protocol, both benchmark suites and the recorded results. The report gives the aggregate outcome and the evidence for each hypothesis. [Research report](RESEARCH.md)

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

## Starting points in the literature

- [SA-nODE: Stable Attractors for Neural Networks Classification via Ordinary Differential Equations](https://arxiv.org/abs/2311.10387) — classification through learned basins of planted stable attractors.
- [SODEF: Stable Neural ODE with Lyapunov-Stable Equilibrium Points for Defending Against Adversarial Attacks](https://proceedings.neurips.cc/paper_files/paper/2021/hash/7d5430cf85f78c4b7aa09813b14bce0d-Abstract.html) — equilibrium stability and robustness to perturbations.
- [How Neural Networks Extrapolate: From Feedforward to Graph Neural Networks](https://arxiv.org/abs/2009.11848) — assumptions governing generalization outside training support.

The research plan positions the project against these approaches. The proposed experimental regime is a question to investigate, not a verified claim of first-of-its-kind novelty.
