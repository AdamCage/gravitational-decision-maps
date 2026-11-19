# Research Proposal: Gravitational Decision Maps

**Status: proposal only. No implementation, experiments, or results exist in this repository yet.**

This document specifies a falsifiable research plan. Architectural choices, datasets, and numerical budgets remain provisional. It makes no claim of a novel classification principle, demonstrated improvement, or publication readiness.

## 1. Motivation and scope

A classifier assigns labels beyond the regions represented by its training examples. This project asks whether learned attraction dynamics can provide a useful rule for extending known classes from observed class cores into unobserved regions.

The motivating picture contains several broad attraction basins near stable centers and complicated boundaries farther away. An input initializes a trajectory; its destination determines its class. The scientific question is whether this construction improves prediction, rather than merely producing elaborate decision maps.

The target is **known-class extrapolation**: training labels are available only in restricted cores, while evaluation includes examples of the same classes outside those cores. Detecting unknown classes or refusing uncertain predictions is secondary and cannot substitute for successful classification of these external examples. Here, “outside” refers to a declared core-selection rule; it does not necessarily mean outside the convex hull or every notion of distributional support.

Attractor-based classification already exists. In particular, [SA-nODE](https://arxiv.org/abs/2311.10387) constructs stable attractors and learns their basins. A possible contribution must therefore concern the extrapolation setting, a justified restriction on dynamics, or evidence about when dynamics outperform static alternatives.

## 2. Research question and hypotheses

**Main question:** Under equal information and comparable computational budgets, can learned class-attraction dynamics improve classification outside observed class cores relative to static classifiers using the same representation?

| Hypothesis | Corresponding null or failure condition |
| --- | --- |
| H1: Dynamics improve balanced accuracy on external examples. | Strong static methods match or outperform them. |
| H2: Integrating trajectories adds value beyond the learned metric, prototypes, and parameters. | Removing integration preserves the gain. |
| H3: Local class geometry helps determine useful external basins. | Gains require external labels, generator-specific information, or arbitrary tuning. |
| H4: Stability constraints improve the accuracy–cost–sensitivity tradeoff. | They merely suppress useful boundaries or increase cost. |

These are empirical hypotheses, not expected findings. Any theoretical claim would need explicit assumptions on the target function, observed geometry, and admissible dynamics. [Xu et al.](https://arxiv.org/abs/2009.11848) provide relevant background on how architectural assumptions affect extrapolation.

## 3. Formal setting

Let inputs be mapped to features by

$$
z=h_\phi(x)\in\mathbb{R}^d,\qquad y\in\{1,\ldots,C\}.
$$

The permitted labeled training set is $D_{\mathrm{core}}$. A documented construction defines observed core sets $K_c$ from permitted observations. External evaluation examples lie outside the corresponding core region while retaining labels in the same vocabulary.

A continuous-time classifier uses

$$
\dot z(t)=F_\theta(z(t)),\qquad z(0)=h_\phi(x).
$$

Each class can have multiple attractors $A_c=\{a_{c1},\ldots,a_{cM_c}\}$. Multiple prototypes can help represent disconnected or multimodal class regions and avoid placing a single class-average prototype in an empty region.

For point attractors, require $F_\theta(a_{cj})=0$ and establish local asymptotic stability through the chosen construction or stated sufficient conditions. For a continuously differentiable field, strictly negative real parts of all Jacobian eigenvalues at an equilibrium are one local sufficient condition. Such a condition does not establish global convergence.

The ideal class basin is

$$
\mathcal B_c=\left\{z_0:\lim_{t\to\infty}\operatorname{dist}(\Phi_t(z_0),A_c)=0\right\}.
$$

Here $\Phi_t$ denotes the flow where it exists. A prediction is the destination class. Numerical implementations must define finite-time tolerances, convergence checks, and unresolved outcomes explicitly; finite-time proximity alone is not proof of asymptotic convergence.

## 4. Distinct dynamical models

### Gradient flow

$$
\dot z=-\nabla U_\theta(z).
$$

The potential decreases along nonstationary trajectories. Simple smooth multiwell potentials do not automatically reproduce the intricate boundaries motivating this project. Their simpler geometry is a useful baseline, not a defect to conceal.

### Damped inertial dynamics

$$
\dot z=v,\qquad \dot v=-\gamma v-\nabla U_\theta(z),\qquad \gamma>0.
$$

The full state is $(z,v)$. Fix and report the initialization, initially $v_0=0$. Changing initial velocity, damping, or integration horizon can change the predicted class. These choices are part of the model and its tuning budget.

### Absorbing-core classification

A separate variant assigns the class of the first core entered and terminates the trajectory. First arrival and eventual convergence are different rules: an inertial trajectory may cross one core before settling elsewhere. Results must label this variant separately.

Centroids used as force sources are not automatically equilibria. Forces from other sources can displace a minimum or eliminate it. Either construct genuine stable equilibria, identify and validate the resulting equilibria, or explicitly use absorbing sets. Do not describe source coordinates as proven attractors.

## 5. Initial model family

Start with softened attraction fields parameterized by a small number of prototypes, positive source weights, local scales, and optional regularized anisotropic metrics. Compare gradient and inertial variants before adding a flexible neural residual field.

Fit all parameters using permitted core observations. A candidate objective combines endpoint classification with penalties or structural constraints enforcing core preservation, local stability, and controlled sensitivity. Its exact form must be fixed during development and ablated. Penalizing a Jacobian is not, by itself, a global guarantee.

Class priors and source masses must be explicit. Raw sample counts can make the largest class dominate remote regions; balancing classes is also an assumption, not a universally correct correction. Prototype counts, bandwidths, and metric capacity must be matched in static comparisons.

The project studies an inductive bias. Core labels alone generally cannot identify a unique external field.

## 6. Information regimes and leakage prevention

### Primary regime: inductive, core-only

The learner receives labeled core examples and no target features or labels. Preprocessing, representation training, prototype fitting, covariance estimates, and early stopping use only permitted development information.

### Optional regime: unlabeled target access

The learner additionally receives external features without labels. This is a different experiment because target geometry supplies new information. Report it separately and compare with suitable graph-based or semi-supervised methods under identical access. Do not combine its scores with core-only results.

### Split and validation protocol

1. Separate development and final evaluation entities before fitting statistics. Respect subject, source, duplicate, or temporal grouping where relevant.
2. Construct cores using a declared rule. Keep excluded observations unavailable to the learner. A benchmark curator may use labels to construct controlled subsets, but must disclose that selection and release its definition.
3. Estimate every prototype, scaler, feature transform, and density threshold from the allowed subset. Do not compute class centers from full classes and then claim core-only learning.
4. Select hyperparameters using core validation or separate development tasks. External validation labels, if permitted, define an additional information regime and must be disclosed.
5. Freeze the configuration and evaluation rules before accessing final external test labels.

Core-only validation may be weak at selecting an extrapolation rule. Report that limitation rather than silently choosing the best external test configuration. Development across separate tasks is permissible, but the resulting cross-task knowledge must be documented.

Use a common frozen encoder for the primary comparison. Record its training provenance, possible dataset overlap, and external supervision. A pretrained representation already contains knowledge beyond the supplied cores; successful downstream extrapolation cannot automatically be attributed to dynamics. Runs with known overlap between encoder pretraining data and final evaluation examples must be reported separately and excluded from claims about a clean inductive core-only setting. Unknown overlap remains an explicit limitation. Evaluate jointly trained encoders only as a separate ablation with equal data access.

## 7. Data plan

### Synthetic tasks

Use rings, curved regions, anisotropic mixtures, disconnected class components, and competing local extensions. Vary retained core coverage, noise, class imbalance, dimensionality, and distance from observed cores. Include withheld wedges and whole components, not only random missing points.

At least one suite must use target rules unrelated to the model's potential. A physical system with known dynamics can supply an illustrative positive control, but feeding its true force law to the classifier is an oracle comparison, not learned extrapolation.

Include paired worlds with identical observed cores and different external labels or dynamics. For example, alter a potential only outside core neighborhoods. These expose nonidentifiability rather than expecting an impossible universal solution.

### Public real-data candidates

Candidates include image classification with frozen features and public tabular classification tasks. Fashion-MNIST or CIFAR-10 and suitable UCI/OpenML datasets are possible starting points; the final selection is not fixed. Check access terms, provenance, duplicates, class coverage, and whether a defensible core/periphery split can be specified before committing to a benchmark.

No private organizational data are part of this proposal. Two-dimensional visualizations will be diagnostic only; decisions and distances will be evaluated in the actual feature space.

## 8. Baselines and fair budgets

Required static baselines are kNN, nearest centroid, multiple prototypes or medoids, regularized Mahalanobis classification, RBF/kernel classifiers, Gaussian mixtures, and an MLP. Include gradient-boosted trees for tabular tasks. Tune them seriously rather than treating default settings as competitive controls.

Include SA-nODE as a directly relevant dynamical baseline, subject to a faithful implementation and compatible evaluation. [SODEF](https://proceedings.neurips.cc/paper_files/paper/2021/hash/7d5430cf85f78c4b7aa09813b14bce0d-Abstract.html) informs stability constraints and robustness comparisons; its adversarial-defense objective is not identical to extrapolation from class cores.

[DUQ](https://proceedings.mlr.press/v119/van-amersfoort20a.html) and [DDU](https://arxiv.org/abs/2102.11582) provide relevant uncertainty and representation comparisons. Their uncertainty metrics may supplement the study, but cannot replace the main classification comparison.

Match information, encoder, prototype budget, tuning trials, and relevant parameter capacity. Report both a parameter-matched comparison and a compute-aware comparison where exact matching is impossible. Count all solver evaluations, not just nominal integration steps. Include feature extraction consistently when reporting end-to-end latency.

## 9. Essential ablations

- Static scoring with the same learned prototypes, metric, and source weights; no trajectory integration.
- Zero, one, and multiple integration steps, followed by the same readout.
- Gradient versus inertial dynamics, and eventual-destination versus first-entry labeling.
- One versus multiple prototypes per class; isotropic versus anisotropic geometry.
- Fixed versus learned fields; with and without each stability or sensitivity constraint.
- Frozen versus jointly learned features, with encoder parity maintained.
- Matched-capacity and matched-compute feedforward controls.

If the improvement survives removal of integration, attribute it to the remaining representation or geometry. If a larger ordinary classifier reproduces the gain at equal cost, avoid claiming that continuous dynamics were essential.

## 10. Metrics and statistical reporting

The primary metric is **balanced accuracy on the external known-class test set**. Report per-class results, ordinary accuracy, core-test accuracy, and performance stratified by a train-defined distance or support measure. Define the primary task aggregate before inspecting outcomes; retain dataset-level results rather than hiding heterogeneous effects.

Count numerically unresolved predictions as errors in the primary all-example score. Additionally report resolved coverage, accuracy conditional on resolution, and the performance of any predetermined fallback. This prevents gains obtained by silently discarding difficult trajectories.

Report training cost, tuning cost, inference latency, memory, function evaluations, integration tolerances, and convergence failures. Test sensitivity to modest changes in step size and solver tolerance. Numerical artifacts must not be mistaken for meaningful basin geometry.

Use repeated independent splits and initialization seeds, initially at least five where feasible. Report paired differences and 95% confidence intervals with the resampling unit stated. Account for shared splits and grouped observations; thousands of points on one synthetic grid are not thousands of independent experiments. Do not choose the best seed.

Measure perturbation sensitivity separately from correctness. A stable prediction can be consistently wrong. Basin complexity or [basin entropy](https://doi.org/10.1038/srep31416) can be diagnostic, but increased complexity is not an optimization target or evidence of generalization.

## 11. Staged pilot and decisions

**Stage A — specification:** finalize information access, core construction, model variants, solver rules, baseline budgets, and a locked evaluation plan. Complete a closer reading of the cited methods before claiming methodological differences.

**Stage B — falsification pilot:** run small synthetic suites with mismatched target rules and static controls. Continue if dynamic effects are reproducible and not explained entirely by prototype count, metric learning, or additional compute. Stop enlarging the model if simple baselines consistently match it.

**Stage C — external validity:** evaluate frozen-feature real tasks with locked settings and several coverage levels. Continue only if benefits persist beyond the motivating synthetic geometry, with tolerable cost and no hidden target access.

**Stage D — contribution assessment:** determine whether evidence supports a method, an empirical study of extrapolation biases, or a useful negative result. A publication case requires reproducible insight beyond a physical analogy; no venue outcome is promised.

Before each stage, specify the smallest practically meaningful gain and acceptable cost for its tasks. Do not invent a universal accuracy threshold after seeing results.

## 12. Limits and expected failure cases

Identical local cores can admit incompatible external class assignments. No method using only those cores can distinguish every such case. A known physical law supplies additional structure; an arbitrary gravitational metaphor does not.

Representation collapse can place semantically different inputs at the same feature location. Local covariance can be unreliable in high dimensions. Minority classes can lose basin volume. Disconnected classes can require many prototypes. Highly interwoven boundaries can amplify measurement noise, and stable equilibria do not ensure that every trajectory reaches one.

The desired outcome is a bounded, testable account of when attraction dynamics help classify unobserved regions. Negative and mixed findings must be retained. All sections above describe planned work; none report completed experiments.

## References

1. Marino et al. **Stable Attractors for Neural networks classification via Ordinary Differential Equations (SA-nODE).** [arXiv:2311.10387](https://arxiv.org/abs/2311.10387).
2. Kang et al. **Stable Neural ODE with Lyapunov-Stable Equilibrium Points for Defending Against Adversarial Attacks.** [NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/7d5430cf85f78c4b7aa09813b14bce0d-Abstract.html).
3. Xu et al. **How Neural Networks Extrapolate: From Feedforward to Graph Neural Networks.** [arXiv:2009.11848](https://arxiv.org/abs/2009.11848).
4. van Amersfoort et al. **Uncertainty Estimation Using a Single Deep Deterministic Neural Network.** [ICML 2020](https://proceedings.mlr.press/v119/van-amersfoort20a.html).
5. **Deep Deterministic Uncertainty (DDU).** [arXiv:2102.11582](https://arxiv.org/abs/2102.11582).
6. Daza et al. **Basin entropy: a new tool to analyze uncertainty in dynamical systems.** [Scientific Reports, 2016](https://doi.org/10.1038/srep31416).
