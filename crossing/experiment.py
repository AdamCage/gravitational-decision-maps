"""Execute frozen crossing, bound, and integration-avoidance comparisons."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from statistics import median
from time import perf_counter
import os

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .flow import MixtureFlow, hybrid, reference_path, rk4, triage


SPEC_PATH = Path(__file__).with_name("study.json")


def _boundary_check(model, x, rng, config):
    initial = model.scores(x).argmax(axis=1)
    requested = config["boundary_pairs"]
    pair = rng.integers(len(x), size=(requested * 4, 2))
    pair = pair[initial[pair[:, 0]] != initial[pair[:, 1]]][:requested]
    if not len(pair):
        return {"candidates": 0, "valid": 0, "wrong": 0, "min_gap": None, "mean_derivative_absolute_error": None}
    left, right = x[pair[:, 0]].copy(), x[pair[:, 1]].copy()
    a, b = initial[pair[:, 0]], initial[pair[:, 1]]
    rows = np.arange(len(a))
    for _ in range(38):
        middle = (left + right) * 0.5
        scores = model.scores(middle)
        left_side = scores[rows, a] > scores[rows, b]
        left[left_side], right[~left_side] = middle[left_side], middle[~left_side]
    x0 = (left + right) * 0.5
    score, mean, velocity, _, _ = model.moments(x0)
    tie_minimum = np.minimum(score[rows, a], score[rows, b])
    tie_error = score[rows, b] - score[rows, a]
    score[rows, a], score[rows, b] = -np.inf, -np.inf
    gap = tie_minimum - score.max(axis=1)
    derivative = np.einsum("nd,nd->n", mean[rows, b] - mean[rows, a], velocity)
    valid = (gap > config["minimum_top_gap"]) & (np.abs(tie_error) <= 1e-8) & (
        np.abs(derivative) > config["minimum_normal_derivative"])
    if not valid.any():
        return {"candidates": len(pair), "valid": 0, "wrong": 0, "min_gap": None,
                "mean_derivative_absolute_error": None}
    tau = config["boundary_probe"]
    after = model.scores(reference_path(model, x0[valid], [tau])[0][0])
    expected = np.where(derivative[valid] > 0, b[valid], a[valid])
    sel = np.arange(len(after))
    fd = (after[sel, b[valid]] - after[sel, a[valid]] - tie_error[valid]) / tau
    min_gap = np.min(gap[valid])
    return {"candidates": len(pair), "valid": int(valid.sum()), "wrong": int(np.sum(after.argmax(axis=1) != expected)),
            "min_gap": float(min_gap) if np.isfinite(min_gap) else None,
            "mean_derivative_absolute_error": float(np.mean(np.abs(fd - derivative[valid])))}


def _checkpoints(horizons, resolution):
    pieces, start = [], 0
    for end in horizons:
        pieces.append(np.linspace(start, end, resolution + 1)[1:])
        start = end
    return np.concatenate(pieces)


def _timed_solvers(model, targets, horizon, step, repeats, buffer):
    methods = ("full", "hybrid", "exponential", "triage")
    results, timings = {}, {name: [] for name in methods}
    for rep in range(repeats):
        order = methods[rep % len(methods):] + methods[:rep % len(methods)]
        for label in order:
            start = perf_counter()
            if label == "full":
                z, nfe = rk4(model, targets, horizon, step)
                result = model.predict(z), nfe * len(targets)
            elif label == "hybrid":
                result = hybrid(model, targets, horizon, step)
            elif label == "exponential":
                _, predicted_scores = model.exponential_scores(targets, horizon)
                result = model.classes[predicted_scores.argmax(axis=1)], len(targets)
            else:
                result = triage(model, targets, horizon, step, buffer)
            timings[label].append(perf_counter() - start)
            results[label] = result
    return {name: (*results[name], median(timings[name])) for name in methods}


def _trial(config, kind, family, gap, seed, phase):
    if kind == "synthetic":
        from .data import make_case

        data = make_case(family, gap, seed)
    else:
        from .public import make_public

        data = make_public(family, seed, split=phase, fold_seeds=tuple(config["final_seeds"]) if phase == "final" else ())

    with threadpool_limits(limits=1):
        scaler = StandardScaler().fit(data.x_train)
        training = scaler.transform(data.x_train)
        external = scaler.transform(data.x_test)
        times = _checkpoints(config["horizons"], config["checkpoints_per_horizon"])
        records = []
        for ratio in config["bandwidth_ratios"]:
            model = MixtureFlow.fit(training, data.y_train, prototypes=config["prototypes"], bandwidth_ratio=ratio, seed=seed)
            initial = model.predict(external)
            path, reference_nfe = reference_path(model, external, times)
            directions = _boundary_check(model, external, np.random.default_rng(seed + 2048), config)
            for j, horizon in enumerate(config["horizons"]):
                group = config["checkpoints_per_horizon"]
                ref_path = np.stack([model.predict(row) for row in path[:group * (j + 1)]])
                end = ref_path[-1]
                changes = end != initial
                sampled_change = np.any(ref_path != initial, axis=0)
                certificate = model.certify(external, horizon)
                r = config["rk4_max_step"]
                reps = config["timing_repeats"]
                solver = _timed_solvers(model, external, horizon, r, reps, config["triage_log_margin"])
                full, nfe, latency = solver["full"]
                accelerated, nfe_hybrid, hybrid_latency = solver["hybrid"]
                _, exp_scores = model.exponential_scores(external, horizon)
                origin_index = np.searchsorted(model.classes, initial)
                point = np.arange(len(initial))
                exp_winning = exp_scores[point, origin_index].copy()
                exp_scores[point, origin_index] = -np.inf
                exp_strength = exp_scores.max(axis=1) - exp_winning
                euler = model.predict(external + horizon * model.field(external))
                speed_mismatch = int(np.sum(accelerated != end))
                certified = certificate.endpoint >= 0
                false_endpoint = int(np.sum(certified & (certificate.endpoint != end)))
                false_crossing = int(np.sum(certificate.forced_crossing & ~changes))
                false_stable = int(np.sum(certificate.stable & sampled_change))
                false_ball = int(np.sum(certificate.ball_stable & sampled_change))
                if changes.any() and (~changes).any():
                    ap_linear = float(average_precision_score(changes, certificate.linear_strength))
                    ap_drag = float(average_precision_score(changes, certificate.drag_strength))
                    ap_margin = float(average_precision_score(changes, -certificate.static_margin))
                    ap_exp = float(average_precision_score(changes, exp_strength))
                else:
                    ap_linear, ap_drag, ap_margin, ap_exp = None, None, None, None
                primary = ratio == config["primary_bandwidth_ratio"] and horizon == config["primary_horizon"]
                ranking = ({"drag": certificate.drag_strength.tolist(), "static": (-certificate.static_margin).tolist(),
                            "exponential": exp_strength.tolist()} if primary else None)
                records.append({
                    "bandwidth_ratio": ratio, "horizon": horizon, "reference_nfe_path": reference_nfe,
                    "tie_probe": directions, "endpoint": end.tolist(), "initial": initial.tolist(),
                    "sampled_crossing": sampled_change.tolist(), "certified_endpoint": certificate.endpoint.tolist(),
                    "certified_stable": certificate.stable.tolist(), "certified_crossing": certificate.forced_crossing.tolist(),
                    "taylor_endpoint": certificate.taylor_endpoint.tolist(), "drag_endpoint": certificate.drag_endpoint.tolist(),
                    "state_endpoint": certificate.state_endpoint.tolist(),
                    "ball_stable": certificate.ball_stable.tolist(), "euler": euler.tolist(),
                    "linear": certificate.linear_label.tolist(), "full": full.tolist(), "hybrid": accelerated.tolist(),
                    "exponential": solver["exponential"][0].tolist(), "triage": solver["triage"][0].tolist(),
                    "nfe_triage": solver["triage"][1], "latency_s_triage": solver["triage"][2],
                    "latency_s_exponential": solver["exponential"][2], "ranking": ranking,
                    "average_precision_linear": ap_linear, "average_precision_drag": ap_drag, "average_precision_margin": ap_margin,
                    "average_precision_exponential": ap_exp,
                    "nfe_full": nfe, "nfe_hybrid": nfe_hybrid, "latency_s_full": latency, "latency_s_hybrid": hybrid_latency,
                    "contradictions": {"endpoint": false_endpoint, "stable": false_stable, "crossing": false_crossing,
                                       "ball": false_ball},
                    "reference_disagreement": {"full": int(np.sum(full != end)), "hybrid": speed_mismatch},
                })
    # Model fit, boundary probes, and certificate evaluation above use no test labels.
    return {"kind": kind, "family": family, "gap": gap, "seed": seed, "metadata": data.metadata,
            "test_input_sha256": sha256(np.ascontiguousarray(data.x_test, dtype="<f8").tobytes()).hexdigest(),
            "y_test": data.y_test.tolist(), "results": records}


def run(config, stage, phase, workers=1, family=None):
    seeds = config["final_seeds" if phase == "final" else "development_seeds"]
    roster = [(stage, f, g, s) for f in config["families" if stage == "synthetic" else "public_families"]
              if family is None or family == f for g in (config["gaps"] if stage == "synthetic" else ["fixed"]) for s in seeds]
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    records = {}
    if workers == 1:
        for job in roster:
            records[job] = _trial(config, *job, phase)
            print(f"completed {' '.join(map(str, job))}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_trial, config, *job, phase): job for job in roster}
            for future in as_completed(futures):
                job = futures[future]
                records[job] = future.result()
                print(f"completed {' '.join(map(str, job))}", flush=True)
    return [records[job] for job in roster]
