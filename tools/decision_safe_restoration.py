"""Exploratory decision-preserving restoration on writer-held-out digits.

Run with `python -m tools.decision_safe_restoration --output PATH`. Train
only on the official training files. Generate additive Gaussian perturbations
of the official holdouts using three fixed noise seeds at three levels. The
clean feature vectors and labels are used only for evaluation. The noise
level is supplied to all fitted denoisers.
"""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.integrate import solve_ivp
import sklearn
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from crossing.flow import MixtureFlow
from crossing.public import _load, _overlap_filter
from gdm.data import _load_digits_train, _load_packaged_data


FIT_SEED = 5421
NOISE_SEEDS = (2627, 10457, 60209)
NOISE_LEVELS = (0.1, 0.2, 0.35)
PROTOTYPES_PER_CLASS = 8
CELL_RESERVE = 1e-4
FLOW_STEPS = 16
GUARD_BATCH_SIZE = 256
GUARD_BISECTIONS = 12
REFERENCE_CHECKPOINTS = 101


def datasets():
    x_train, y_train = _load_digits_train()
    x_test, y_test = _load_packaged_data("digits")
    yield "optdigits", x_train / 16, y_train, x_test / 16, y_test
    x_train, y_train, train = _load("pendigits.tra")
    x_test, y_test, test = _load("pendigits.tes")
    test = _overlap_filter(x_train, train, x_test, test)
    yield "pendigits", x_train[train] / 100, y_train[train], x_test[test] / 100, y_test[test]


def fit_prior(features, labels):
    centers, classes, counts = [], [], []
    squared_error = 0.0
    for label in np.unique(labels):
        points = features[labels == label]
        model = KMeans(n_clusters=PROTOTYPES_PER_CLASS, n_init=4, random_state=FIT_SEED).fit(points)
        centers.extend(model.cluster_centers_)
        classes.extend([label] * PROTOTYPES_PER_CLASS)
        counts.extend(np.bincount(model.labels_, minlength=PROTOTYPES_PER_CLASS))
        squared_error += model.inertia_
    return centers, classes, np.asarray(counts) / len(features), squared_error / features.size


class DecisionCell:
    """Affine score margins for the class picked at each corrupted input."""

    def __init__(self, classifier, noisy, normals, bias):
        self.leader = classifier.decision_function(noisy).argmax(axis=1)
        self.normals = normals[self.leader]
        self.bias = bias[self.leader]
        self.start = self.margins(noisy)
        self.required = CELL_RESERVE * self.start

    def margins(self, points):
        return np.einsum("ncd,nd->nc", self.normals, points) + self.bias

    def clip_segment(self, noisy, candidate):
        end = self.margins(candidate)
        needs_clip = end < self.required
        safe_fraction = np.divide(self.start - self.required, self.start - end,
                                  out=np.ones_like(end), where=needs_clip)
        return noisy + safe_fraction.min(axis=1, keepdims=True) * (candidate - noisy)

    def project(self, noisy, candidate):
        """Dykstra projection onto the shrunken cell, with a feasible fallback."""
        selected = (self.margins(candidate) < self.required).any(axis=1)
        projected = candidate[selected].copy()
        normals, bias, required = self.normals[selected], self.bias[selected], self.required[selected]
        dual = np.zeros_like(required)
        for _ in range(200):
            previous = projected.copy()
            for rival in range(normals.shape[1]):
                normal = normals[:, rival]
                corrected = projected + dual[:, rival, None] * normal
                margin = (corrected * normal).sum(axis=1) + bias[:, rival]
                deficit = np.maximum(0, required[:, rival] - margin)
                projected = corrected + deficit[:, None] * normal
                dual[:, rival] = -deficit
            if np.max(np.abs(projected - previous), initial=0) <= 1e-12:
                break
        result = candidate.copy()
        result[selected] = projected
        fallback = (self.margins(result) < self.required - 1e-10).any(axis=1)
        result[fallback] = self.clip_segment(noisy, result)[fallback]
        # The halfspace corrections are nonpositive. Their negations give
        # dual-feasible multipliers and a lower bound on the QP objective.
        objective = 0.5 * np.square(result[selected] - candidate[selected]).sum(axis=1)
        weighted_normals = np.einsum("nc,ncd->nd", -dual, normals)
        dual_lower = (dual * (self.margins(candidate)[selected] - required)).sum(axis=1)
        dual_lower -= 0.5 * np.square(weighted_normals).sum(axis=1)
        return result, {
            "fallback_count": int(fallback.sum()),
            "largest_primal_minus_dual": float(np.max(objective - dual_lower, initial=0)),
        }

def affine_pairs(classifier):
    weight_differences = classifier.coef_[:, None] - classifier.coef_[None, :]
    bias_differences = classifier.intercept_[:, None] - classifier.intercept_[None, :]
    length = np.linalg.norm(weight_differences, axis=2)
    np.fill_diagonal(length, 1.0)
    return weight_differences / length[..., None], bias_differences / length


class FlowGuard:
    """Certify the original flow up to a query-dependent time in exact arithmetic."""

    def __init__(self, classifier, model):
        self.classifier = classifier
        self.model = model
        self.normals, self.bias = affine_pairs(classifier)
        self.normal_coordinates = self.normals @ model.centers.T + self.bias[..., None]
        slopes = self.normals @ model.centers_t / model.beta
        self.slopes = slopes
        center_logits = model._rel_logits(model.centered)
        projected_logits = center_logits - self.normal_coordinates[..., :, None] * slopes[..., None, :]
        self.lower = projected_logits.min(axis=2)
        self.upper = projected_logits.max(axis=2)
        self.minimum = self.normal_coordinates.min(axis=2)

    def global_time(self, cell, horizon):
        minimum = self.minimum[cell.leader]
        exit_time = np.full_like(cell.start, np.inf)
        deficit = cell.required - minimum
        adverse = deficit > 0
        exit_time[adverse] = np.log1p((1 - CELL_RESERVE) * cell.start[adverse] / deficit[adverse])
        return np.minimum(horizon, exit_time.min(axis=1))

    def local_time(self, cell, noisy, global_time, horizon):
        result = global_time.copy()
        unresolved = np.flatnonzero(global_time < horizon)
        for offset in range(0, len(unresolved), GUARD_BATCH_SIZE):
            selected = unresolved[offset:offset + GUARD_BATCH_SIZE]
            leader, margin = cell.leader[selected], cell.start[selected]
            base = self.model._rel_logits(noisy[selected] - self.model.origin)
            projected = base[:, None, :] - margin[..., None] * self.slopes[leader]
            lower = np.minimum(0, self.lower[leader] - projected)
            upper = np.maximum(0, self.upper[leader] - projected)
            coordinates = self.normal_coordinates[leader]

            def stable(time):
                margin_budget = (1 - CELL_RESERVE) * margin / np.expm1(time[:, None]) - cell.required[selected]
                direction = coordinates + margin_budget[..., None]
                expansion = -np.expm1(-time[:, None, None])
                adversary = projected + expansion * np.where(direction >= 0, lower, upper)
                adversary -= adversary.max(axis=2, keepdims=True)
                terms = direction * np.exp(adversary)
                valid = terms.sum(axis=2) > 1e-10 * (1 + np.abs(terms).sum(axis=2))
                valid[np.arange(len(leader)), leader] = True
                return valid.all(axis=1)

            low, high = global_time[selected], np.full(len(selected), horizon)
            low = np.where(stable(high), high, low)
            for _ in range(GUARD_BISECTIONS):
                midpoint = (low + high) / 2
                valid = stable(midpoint)
                low = np.where(valid, midpoint, low)
                high = np.where(valid, high, midpoint)
            result[selected] = low
        return result


def flow_endpoint(model, start, horizon, cell=None):
    step = np.broadcast_to(horizon, (len(start),))[:, None] / FLOW_STEPS
    point = start.copy()
    departed = np.zeros(len(start), dtype=bool)
    for _ in range(FLOW_STEPS):
        first = model.field(point)
        second = model.field(point + step * first / 2)
        third = model.field(point + step * second / 2)
        fourth = model.field(point + step * third)
        point += step / 6 * (first + 2 * second + 2 * third + fourth)
        if cell is not None:
            departed |= (cell.margins(point) < 0).any(axis=1)
    return point, int(departed.sum())


def reference_diagnostic(model, noisy, cell, times, endpoints, seed):
    row = np.arange(len(noisy))
    margins = cell.start.copy()
    margins[row, cell.leader] = np.inf
    limiting = np.argsort(margins.min(axis=1), kind="stable")[:8]
    remaining = np.setdiff1d(row, limiting)
    random = np.random.default_rng(seed).choice(remaining, size=8, replace=False)
    selected = np.concatenate((limiting, random))
    smallest_margin = np.inf
    worst_reserve_slack = np.inf
    largest_error = 0.0
    for index in selected:
        time = times[index]
        solution = solve_ivp(lambda _, point: model.field(point[None])[0], (0, time), noisy[index],
                             method="DOP853", rtol=2e-12, atol=2e-14,
                             t_eval=np.linspace(0, time, REFERENCE_CHECKPOINTS))
        if not solution.success:
            raise RuntimeError(solution.message)
        path_margins = cell.normals[index] @ solution.y + cell.bias[index, :, None]
        path_margins[cell.leader[index]] = np.inf
        smallest_margin = min(smallest_margin, float(path_margins.min()))
        margin_slack = path_margins - cell.required[index, :, None]
        worst_reserve_slack = min(worst_reserve_slack, float(margin_slack.min()))
        error = np.linalg.norm(solution.y[:, -1] - endpoints[index])
        largest_error = max(largest_error, float(error))
    return {"paths": len(selected), "checkpoints_per_path": REFERENCE_CHECKPOINTS,
            "smallest_rival_margin": smallest_margin, "smallest_reserve_slack": worst_reserve_slack,
            "largest_rk4_endpoint_difference": largest_error}


def measure(guard, noisy, clean, labels, seed, sigma, horizon, wiener):
    classifier, flow = guard.classifier, guard.model
    cell = DecisionCell(classifier, noisy, guard.normals, guard.bias)
    posterior_mean = noisy + (-np.expm1(-horizon)) * flow.field(noisy)
    full_flow, full_departures = flow_endpoint(flow, noisy, horizon, cell)
    global_time = guard.global_time(cell, horizon)
    local_time = guard.local_time(cell, noisy, global_time, horizon)
    global_flow, global_departures = flow_endpoint(flow, noisy, global_time, cell)
    local_flow, local_departures = flow_endpoint(flow, noisy, local_time, cell)
    projected_wiener, projection_wiener = cell.project(noisy, wiener)
    projected_mean, projection_mean = cell.project(noisy, posterior_mean)
    projected_flow, projection_flow = cell.project(noisy, full_flow)
    initial = classifier.classes_[cell.leader]

    predictions = {
        "identity": noisy,
        "wiener": wiener,
        "wiener_cell_projection": projected_wiener,
        "posterior_mean": posterior_mean,
        "posterior_mean_cell_clip": cell.clip_segment(noisy, posterior_mean),
        "posterior_mean_cell_projection": projected_mean,
        "prototype_flow": full_flow,
        "prototype_flow_cell_projection": projected_flow,
        "prototype_flow_global_certified_time": global_flow,
        "prototype_flow_local_certified_time": local_flow,
    }
    metrics = {}
    for method, restored in predictions.items():
        output_labels = classifier.predict(restored)
        error = np.square(restored - clean).mean(axis=1)
        metrics[method] = {
            "mean_squared_error": float(error.mean()),
            "class_changes": int(np.count_nonzero(output_labels != initial)),
            "correct": int(np.count_nonzero(output_labels == labels)),
        }
    return {
        "noise_seed": seed, "noise_standard_deviation": sigma, "rows": len(clean),
        "horizon": horizon,
        "average_global_horizon_fraction": float(global_time.mean() / horizon),
        "average_local_horizon_fraction": float(local_time.mean() / horizon),
        "global_full_horizon_certified": int(np.count_nonzero(global_time == horizon)),
        "local_full_horizon_certified": int(np.count_nonzero(local_time == horizon)),
        "sampled_path_departures": {"full": full_departures, "global": global_departures, "local": local_departures},
        "dop853_local_guard_check": reference_diagnostic(flow, noisy, cell, local_time, local_flow, seed),
        "projection": {"wiener": projection_wiener, "posterior_mean": projection_mean,
                       "prototype_flow": projection_flow},
        "methods": metrics,
    }


def run():
    results = {}
    for name, x_train, y_train, x_test, y_test in datasets():
        classifier = LogisticRegression(max_iter=1500, random_state=FIT_SEED).fit(x_train, y_train)
        centers, classes, masses, prior_variance = fit_prior(x_train, y_train)
        x_mean = x_train.mean(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(x_train, rowvar=False))
        eigenvalues = np.maximum(0, eigenvalues)
        rows = []
        for sigma in NOISE_LEVELS:
            flow = MixtureFlow(centers, classes, masses, prior_variance + sigma**2)
            horizon = np.log1p(sigma**2 / prior_variance)
            guard = FlowGuard(classifier, flow)
            gains = eigenvalues / (eigenvalues + sigma**2)
            for seed in NOISE_SEEDS:
                generator = np.random.default_rng(seed)
                noisy = x_test + sigma * generator.normal(size=x_test.shape)
                wiener = x_mean + ((noisy - x_mean) @ eigenvectors * gains) @ eigenvectors.T
                rows.append(measure(guard, noisy, x_test, y_test, seed, sigma, horizon, wiener))
        results[name] = {
            "train_rows": len(x_train), "holdout_rows": len(x_test), "dimensions": x_train.shape[1],
            "prior_component_variance": prior_variance, "rows": rows,
        }
    return {
        "study": "exploratory",
        "interpretation": "Writer-held-out source features with simulated iid Gaussian corruption.",
        "fit_seed": FIT_SEED, "prototypes_per_class": PROTOTYPES_PER_CLASS, "noise_seeds": NOISE_SEEDS,
        "noise_standard_deviations": NOISE_LEVELS, "flow_steps": FLOW_STEPS,
        "guard_bisections": GUARD_BISECTIONS, "decision_margin_reserve": CELL_RESERVE,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "datasets": results,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = run()
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
