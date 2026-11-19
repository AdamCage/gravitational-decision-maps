# Gravitational Decision Maps

**Attractor-based classification beyond observed training cores.**

This project investigates whether a dynamical classifier can generalize from densely sampled class regions to previously unobserved regions of the same classes. An input is treated as an initial state; its predicted class is determined by the attractor reached by the resulting trajectory.

**Status:** research proposal. The repository currently contains documentation only. No implementation, experiments, or empirical results are available, and no performance advantage or novel algorithm is claimed.

## Research question

> When does learned attractor dynamics improve classification outside observed training cores compared with static classifiers given the same data, representations, and tuning budget?

The motivating picture is a collection of broad class basins around well-observed cores, with less familiar regions farther away. The project studies whether the geometry learned near those cores can support a useful continuation of class decisions.

The primary task is **classification of new examples from known classes**. Out-of-distribution detection and abstention are secondary questions and cannot substitute for demonstrating correct external classification.

## Proposed approach

1. Estimate local class cores and one or more prototypes per class using only the allowed training data.
2. Construct or learn a vector field with stable class-associated attractors.
3. Classify a new input by its terminal attractor, while recording non-convergent trajectories explicitly.
4. Compare against strong static methods and existing dynamical classifiers on controlled training-core-to-periphery splits.
5. Identify which assumptions make the continuation useful and where it fails.

“Gravitational” describes the initial physical intuition. The study will compare explicit attraction fields, potential flows, and learned dynamics; reproducing Newtonian gravity or a fractal-looking boundary is not an objective by itself.

## What would count as evidence

- Reproducible improvement on held-out external regions across different geometries and real datasets.
- A benefit that remains when the representation, local metric, prototype count, and access to data are controlled.
- Ablations showing whether trajectories contribute beyond static scores or a single update.
- Reported computational cost, sensitivity to noise, convergence failures, and negative results.

Class cores do not uniquely determine their distant decision boundaries. A successful result must therefore explain which additional assumptions about the field and task make extrapolation possible. Local stability or agreement between perturbed trajectories alone does not establish label correctness.

## Documents

| File | Contents |
| --- | --- |
| [README.md](README.md) | Project overview and current status |
| [RESEARCH.md](RESEARCH.md) | Research questions, mathematical setup, hypotheses, evaluation protocol, limitations, and references |

## Starting points in the literature

- [SA-nODE: Stable Attractors for Neural Networks Classification via Ordinary Differential Equations](https://arxiv.org/abs/2311.10387) — classification through learned basins of planted stable attractors.
- [SODEF: Stable Neural ODE with Lyapunov-Stable Equilibrium Points for Defending Against Adversarial Attacks](https://proceedings.neurips.cc/paper_files/paper/2021/hash/7d5430cf85f78c4b7aa09813b14bce0d-Abstract.html) — equilibrium stability and robustness to perturbations.
- [How Neural Networks Extrapolate: From Feedforward to Graph Neural Networks](https://arxiv.org/abs/2009.11848) — assumptions governing generalization outside training support.

The research plan positions the project against these approaches. The proposed experimental regime is a question to investigate, not a verified claim of first-of-its-kind novelty.
