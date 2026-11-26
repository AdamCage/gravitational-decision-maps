"""Held-out matched binary64 comparison for the facet and score-interval methods."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import platform
from time import perf_counter_ns
from typing import NamedTuple

import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from crossing.affine_certificate import VoronoiCertificate
from crossing.affine_score_interval import ScoreInterval
from crossing.flow import MixtureFlow, reference_path
from tools import flat_boundary_transfer as transfer
from tools.flat_boundary_interval import FIXTURES


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".research" / "flat-boundary-data"
OUTPUT = ROOT / "crossing_results" / "affine_score_comparison.json"
RATIO = 0.35
CAP = 96
LADDER = (1, 4, 16, 64)
REPEATS = 3
METHODS = ("facet", "interval", "facet_then_interval")


class Outcome(NamedTuple):
    initial: np.ndarray
    stable: np.ndarray
    switched: np.ndarray
    row_steps: int


def cases_with_model(cases):
    for fold, (training, labels, test, _, _) in enumerate(cases, start=1):
        scaler = StandardScaler().fit(training)
        model = MixtureFlow.fit(scaler.transform(training), labels, prototypes=1,
                                bandwidth_ratio=RATIO, seed=transfer.SEED)
        indices = np.arange(min(CAP, len(test))) * len(test) // min(CAP, len(test))
        yield fold, model, scaler.transform(test[indices]), indices


def agreement(first, second):
    both_initial = (first.initial >= 0) & (second.initial >= 0)
    if (np.any(both_initial & (first.initial != second.initial))
            or np.any(first.stable & second.switched) or np.any(first.switched & second.stable)):
        raise ArithmeticError("Two rigorous decisions conflict")


def run_method(method, facet, interval, queries, horizon):
    pieces = []
    for start in range(0, len(queries), transfer.REFERENCE_BATCH_SIZE):
        batch = queries[start:start + transfer.REFERENCE_BATCH_SIZE]
        if method == "interval":
            pieces.append(interval.classify(batch, horizon, LADDER))
            continue
        initial, stable, changed, slab = facet.classify(batch, horizon)
        stable |= slab
        row_steps = 0
        if method == "facet_then_interval":
            remaining = ~(stable | changed)
            if np.any(remaining):
                residual = interval.classify(batch[remaining], horizon, LADDER)
                initial[remaining] = np.maximum(initial[remaining], residual.initial)
                stable[remaining] = residual.stable
                changed[remaining] = residual.switched
                row_steps = residual.row_steps
        pieces.append(Outcome(initial, stable, changed, row_steps))
    return Outcome(*(np.concatenate([part[index] for part in pieces]) for index in range(3)),
                   sum(part.row_steps for part in pieces))


def numerical_paths(model, queries):
    labels = []
    for start in range(0, len(queries), transfer.REFERENCE_BATCH_SIZE):
        batch = queries[start:start + transfer.REFERENCE_BATCH_SIZE]
        path, _ = reference_path(model, batch, transfer.CHECKPOINTS)
        scores = model.scores(path.reshape(-1, batch.shape[1]))
        labels.append(scores.argmax(axis=-1).reshape(len(path), len(batch)))
    return model.scores(queries).argmax(axis=-1), np.concatenate(labels, axis=-1)


def summarize(outcome):
    return {"n_strict_initial": int((outcome.initial >= 0).sum()),
            "stable": int(outcome.stable.sum()), "terminal_loss": int(outcome.switched.sum()),
            "unresolved": int((~(outcome.stable | outcome.switched)).sum()), "row_steps": outcome.row_steps}


def evaluate_case(model, queries, timing):
    start = perf_counter_ns()
    facet = VoronoiCertificate(model.centers, model.beta)
    facet_setup = perf_counter_ns() - start
    start = perf_counter_ns()
    interval = ScoreInterval(model.centers, model.beta)
    interval_setup = perf_counter_ns() - start
    initial, labels = numerical_paths(model, queries)
    result = {"rows": len(queries), "feature_dimension": queries.shape[1], "sites": len(model.centers),
              "comparisons": []}
    if timing:
        result["preparation_ns"] = {"facet": facet_setup, "interval": interval_setup}
    for i, horizon in enumerate(transfer.HORIZONS):
        outcomes = {method: run_method(method, facet, interval, queries, horizon) for method in METHODS}
        agreement(outcomes["facet"], outcomes["interval"])
        agreement(outcomes["facet_then_interval"], outcomes["interval"])
        stop = (i + 1) * transfer.CHECKPOINTS_PER_INCREMENT
        departure = (labels[:stop] != initial).any(axis=0)
        endpoint_loss = labels[stop - 1] != initial
        for outcome in outcomes.values():
            if (np.any((outcome.initial >= 0) & (outcome.initial != initial))
                    or np.any(outcome.stable & departure) or np.any(outcome.switched & ~endpoint_loss)):
                raise ArithmeticError(f"DOP853 cross-check at T={horizon} contradicted a rigorous verdict")
        row = {"horizon": horizon, "numeric_sampled_departure": int(departure.sum()),
               "numeric_terminal_loss": int(endpoint_loss.sum()),
               "outcomes": {method: summarize(outcome) for method, outcome in outcomes.items()}}
        if timing:
            times = {method: [] for method in METHODS}
            for repeat in range(REPEATS):
                order = METHODS[repeat:] + METHODS[:repeat]
                for method in order:
                    start = perf_counter_ns()
                    run_method(method, facet, interval, queries, horizon)
                    times[method].append(perf_counter_ns() - start)
            row["timing_ns"] = {method: {"median": int(np.median(samples)), "samples": samples}
                                for method, samples in times.items()}
        result["comparisons"].append(row)
    return result


def fixtures(timing):
    results = {}
    for name, fixture in FIXTURES.items():
        sites = np.array(fixture["sites"], dtype=float)
        model = MixtureFlow(sites, np.arange(len(sites)), np.ones(len(sites)), float(fixture["beta"]))
        results[name] = evaluate_case(model, np.array(fixture["queries"], dtype=float), timing)
    return results


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--timing", action="store_true", help="Record alternating warmed whole-pass timings")
    args = parser.parse_args()
    sources = {"iris": transfer.iris_cases(), "letter": transfer.letter_cases(args.cache),
               "har": transfer.har_cases(args.cache)}
    with threadpool_limits(limits=1):
        simulated = fixtures(args.timing)
        public = {}
        for name, cases in sources.items():
            public[name] = []
            for fold, model, queries, indices in cases_with_model(cases):
                comparison = evaluate_case(model, queries, args.timing)
                public[name].append({"fold": fold, "sample_indices": indices.tolist(), **comparison})
    paths = ("crossing/affine_certificate.py", "crossing/affine_score_interval.py", "tools/affine_score_comparison.py")
    design = {
        "study": "Matched same-precision comparison of verified Voronoi decisions",
        "design": "Prespecified descriptive comparison on existing transfer splits",
        "max_rows_per_fold": CAP,
        "selection": "Retain floor(j*N/min(N,cap)) for j=0,...,min(N,cap)-1, in existing fold order",
        "horizons": transfer.HORIZONS,
        "bandwidth_ratio": RATIO,
        "max_batch_rows": transfer.REFERENCE_BATCH_SIZE,
        "score_interval_max_steps_per_pass": LADDER,
        "timing_configuration": "One thread, one untimed pass, cyclic order over three repetitions; setup separate",
        "sources": transfer.SOURCES,
        "code_sha256": {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in paths},
        "results": public,
        "fixtures": simulated,
    }
    if args.timing:
        design["timing_environment"] = {"host": platform.platform(), "processor": platform.processor(),
                                        "python": platform.python_version(), "numpy": np.__version__}
    transfer.write_result(args.output, design)


if __name__ == "__main__":
    main()
