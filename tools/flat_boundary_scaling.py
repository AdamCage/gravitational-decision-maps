"""Measure fixed-dimensional guard coverage and latency as class count grows.

The fixed design uses 256 near-balanced queries at each of three seeds for
each (feature dimension, class count) setting. Draw Gaussian centers, fit
the coordinate scaler only to those centers, and center each query at its
assigned class. The two regimes share centers, class assignments and noise
directions. Coordinate noise standard deviations in scaled coordinates
are 0.08 and 0.45 times the diameter of the centers. The width ratio is 0.35
and the horizon is 0.5. Results describe this implementation and these
synthetic cases. Run as tools.flat_boundary_scaling.
"""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import platform

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from crossing.flow import MixtureFlow, reference_path, rk4
from tools import flat_boundary_transfer as transfer


DIMENSIONS = (16, 128)
CLASS_COUNTS = (3, 6, 12, 26, 40)
SEEDS = (2601, 2609, 2683)
REGIMES = {"separated": 0.08, "overlapping": 0.45}
N_QUERIES = 256
CHECKPOINTS = 12
TIMING = {"horizon": 0.5, "ratio": 0.35, "max_step": 0.1, "repeats": 7}


def cases(dimension, class_count, seed):
    rng = np.random.default_rng(np.random.SeedSequence((seed, dimension, class_count)))
    centers = rng.normal(size=(class_count, dimension))
    labels = np.arange(class_count)
    scaler = StandardScaler().fit(centers)
    scaled_centers = scaler.transform(centers)
    diameter = cdist(scaled_centers, scaled_centers).max()
    assigned = rng.permutation(np.arange(N_QUERIES) % class_count)
    noise = rng.normal(size=(N_QUERIES, dimension))
    for regime, width in REGIMES.items():
        query = centers[assigned] + width * diameter * (noise * scaler.scale_)
        yield regime, (centers, labels, query, assigned, {})


def check_coverage(case, horizon, ratio, max_step):
    centers, labels, query, assigned, _ = case
    scaler = StandardScaler().fit(centers)
    model = MixtureFlow.fit(scaler.transform(centers), labels, prototypes=1, bandwidth_ratio=ratio, seed=transfer.SEED)
    query = scaler.transform(query)
    distances, inward = transfer.bisector_offsets(model.centered)
    guard = transfer.LocalPlaneGuard(model, distances)
    checks = np.linspace(0, horizon, CHECKPOINTS + 1)[1:]
    counts = dict.fromkeys((
        "n", "static_matches_assignment", "global_stable", "local_stable", "covariance_stable",
        "endpoint_switches", "sampled_departures", "local_sampled_departures",
        "rk4_reference_disagreements",
    ), 0)
    for left in range(0, len(query), transfer.REFERENCE_BATCH_SIZE):
        right = left + transfer.REFERENCE_BATCH_SIZE
        batch = query[left:right]
        scores = model.scores(batch)
        initial = scores.argmax(axis=1)
        global_mask = transfer.bisector_stable(scores, distances, inward, horizon)
        local_mask = guard.stable(guard.prepare(scores), horizon)
        covariance_mask = model.certify(batch, horizon).stable
        path, _ = reference_path(model, batch, checks)
        path_scores = model.scores(path.reshape(-1, batch.shape[1])).reshape(len(path), len(batch), -1)
        endpoint = path_scores[-1].argmax(axis=1)
        initial_scores = path_scores[:, np.arange(len(batch)), initial].copy()
        path_scores[:, np.arange(len(batch)), initial] = -np.inf
        departed = (path_scores.max(axis=2) >= initial_scores).any(axis=0)
        fallback, _ = rk4(model, batch, horizon, max_step)
        values = {
            "n": len(batch), "static_matches_assignment": (initial == assigned[left:right]).sum(),
            "global_stable": global_mask.sum(), "local_stable": local_mask.sum(),
            "covariance_stable": covariance_mask.sum(), "endpoint_switches": (endpoint != initial).sum(),
            "sampled_departures": departed.sum(), "local_sampled_departures": (departed & local_mask).sum(),
            "rk4_reference_disagreements": (model.predict(fallback) != endpoint).sum(),
        }
        for name, value in values.items():
            counts[name] += int(value)
    return counts


def summarize(rows, dimension, class_count, regime):
    counts = {key: sum(row["counts"][key] for row in rows) for key in rows[0]["counts"]}
    clocks = [row["timing"]["median_ms_one_fold"] for row in rows]
    return {
        "dimension": dimension, "classes": class_count, "regime": regime, "counts": counts,
        "fraction": {key: value / counts["n"] for key, value in counts.items() if key != "n"},
        "median_ms_per_256": {key: float(np.median([clock[key] for clock in clocks])) for key in clocks[0]},
        "median_hybrid_to_rk4_ratio": float(np.median([row["timing"]["hybrid_to_rk4_ratio"] for row in rows])),
        "hybrid_to_rk4_label_disagreements": sum(row["timing"]["hybrid_to_rk4_label_disagreements"] for row in rows),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("crossing_results/flat_boundary_scaling.json"))
    args = parser.parse_args()
    rows, overall = [], []
    with threadpool_limits(limits=1):
        for dimension in DIMENSIONS:
            for class_count in CLASS_COUNTS:
                results = {regime: [] for regime in REGIMES}
                for seed in SEEDS:
                    for regime, case in cases(dimension, class_count, seed):
                        counts = check_coverage(case, **{key: TIMING[key] for key in ("horizon", "ratio", "max_step")})
                        latency = transfer.measure_latency(case, **TIMING)
                        row = {"dimension": dimension, "classes": class_count, "seed": seed,
                               "regime": regime, "counts": counts, "timing": latency}
                        rows.append(row)
                        results[regime].append(row)
                overall.extend(summarize(subset, dimension, class_count, regime) for regime, subset in results.items())
                print(f"d={dimension}, K={class_count}: complete", flush=True)
    transfer.write_result(args.output, {
        "status": "synthetic implementation scaling diagnostic",
        "design": {
            "dimensions": DIMENSIONS, "class_counts": CLASS_COUNTS, "seeds": SEEDS, "regime_noise_scale": REGIMES,
            "queries_per_seed": N_QUERIES, "reference_checkpoints": CHECKPOINTS,
            "timing": TIMING | {
                "threads": 1, "batch_limit": transfer.REFERENCE_BATCH_SIZE, "setup_timed_separately": True,
                "host": platform.processor(), "python": platform.python_version(),
            },
        },
        "sha256": {"script": sha256(Path(__file__).read_bytes()).hexdigest(),
                   "transfer_driver": sha256(Path(transfer.__file__).read_bytes()).hexdigest()},
        "overall": overall, "per_seed": rows,
    })


if __name__ == "__main__":
    main()
