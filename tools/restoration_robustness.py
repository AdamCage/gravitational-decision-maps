"""Paired sensitivity check for the decision-safe restoration experiment."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from crossing.flow import MixtureFlow
from tools.decision_safe_restoration import (
    FIT_SEED, NOISE_SEEDS, DecisionCell, FlowGuard, datasets, fit_prior, flow_endpoint,
)


TRUE_SIGMA = 0.20
CONTAMINATION = 0.05
BACKGROUND_SCALE = 0.25
OUTLIER_SCALE = np.sqrt((1 - (1 - CONTAMINATION) * BACKGROUND_SCALE**2) / CONTAMINATION)
SCENARIOS = (
    ("laplace", 0.20),
    ("contaminated", 0.20),
    ("contaminated", 0.10),
)


def corrupt(clean, law, seed):
    rng = np.random.default_rng(seed)
    if law == "laplace":
        return clean + rng.laplace(0, TRUE_SIGMA / np.sqrt(2), clean.shape)
    scale = np.where(rng.random(clean.shape) < CONTAMINATION, OUTLIER_SCALE, BACKGROUND_SCALE)
    return clean + TRUE_SIGMA * scale * rng.normal(size=clean.shape)


def run():
    results = {}
    for name, x_train, y_train, clean, _ in datasets():
        classifier = LogisticRegression(max_iter=1500, random_state=FIT_SEED).fit(x_train, y_train)
        centers, classes, masses, variance = fit_prior(x_train, y_train)
        center = x_train.mean(axis=0)
        eig, vec = np.linalg.eigh(np.cov(x_train, rowvar=False))
        eig = np.maximum(eig, 0)
        rows = []
        for law, supplied_sigma in SCENARIOS:
            model = MixtureFlow(centers, classes, masses, variance + supplied_sigma**2)
            horizon = np.log1p(supplied_sigma**2 / variance)
            guard = FlowGuard(classifier, model)
            gain = eig / (eig + supplied_sigma**2)
            for seed in NOISE_SEEDS:
                noisy = corrupt(clean, law, seed)
                cell = DecisionCell(classifier, noisy, guard.normals, guard.bias)
                mean = noisy - np.expm1(-horizon) * model.field(noisy)
                wiener = center + ((noisy - center) @ vec * gain) @ vec.T
                projected_mean, mean_solver = cell.project(noisy, mean)
                projected_wiener, wiener_solver = cell.project(noisy, wiener)
                global_time = guard.global_time(cell, horizon)
                local_time = guard.local_time(cell, noisy, global_time, horizon)
                local, departures = flow_endpoint(model, noisy, local_time, cell)
                methods = {"input": noisy, "projected_wiener": projected_wiener,
                           "clipped_mean": cell.clip_segment(noisy, mean),
                           "projected_mean": projected_mean, "local_flow": local}
                initial = classifier.classes_[cell.leader]
                rows.append({
                    "law": law, "supplied_sigma": supplied_sigma, "noise_seed": seed, "rows": len(clean),
                    "mean_mse": {key: float(np.mean(np.square(value - clean))) for key, value in methods.items()},
                    "class_switches": {key: int(np.sum(classifier.predict(value) != initial))
                                       for key, value in methods.items()},
                    "rk4_sampled_departures": departures,
                    "allowed_horizon_fraction": float(local_time.mean() / horizon),
                    "projection_fallbacks": {
                        "mean": mean_solver["fallback_count"], "wiener": wiener_solver["fallback_count"],
                    },
                })
        results[name] = rows
    return {"nominal_true_sigma": TRUE_SIGMA, "contamination_fraction": CONTAMINATION,
            "mixture_scale_background": BACKGROUND_SCALE, "mixture_scale_outlier": float(OUTLIER_SCALE),
            "rows": results}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = run()
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
