"""Independently count accuracy and check measured latency on frozen observations."""

from argparse import ArgumentParser
from collections import Counter
from fractions import Fraction
import json
from math import isfinite
from pathlib import Path
from statistics import mean, median


METHODS = ("static", "inertial", "clipped_inertial", "relativistic")
ROOT = Path(__file__).resolve().parent


def _validate_methods(result, labels, timing_repeats):
    methods = result["methods"]
    if methods.keys() != set(METHODS):
        raise ValueError("Wrong set of methods in inertial observation")
    for name, method in methods.items():
        for field in ("predictions", "refined_predictions"):
            if len(method[field]) != len(labels) or any(type(value) != int for value in method[field]):
                raise ValueError(f"Invalid {field} for {name}")
        times = method["latencies_s"]
        if len(times) != timing_repeats or any(not isfinite(value) or value <= 0 for value in times):
            raise ValueError(f"Invalid latency repeats for {name}")
        nfe = method["nfe"]
        if type(nfe) != int or (nfe != 0 if name == "static" else nfe < 1):
            raise ValueError(f"Invalid field-evaluation count for {name}")
    if len({methods[name]["nfe"] for name in METHODS[1:]}) != 1:
        raise ValueError("The inertial comparisons must use the same field-evaluation budget")
    static = methods["static"]
    if static["predictions"] != static["refined_predictions"]:
        raise ValueError("Refinement changed a static prediction")


def _mismatches(first, second):
    return sum(a != b for a, b in zip(first, second, strict=True))


def _balanced_accuracy(labels, predictions, support):
    hits = Counter(actual for actual, predicted in zip(labels, predictions, strict=True) if actual == predicted)
    return mean(Fraction(hits[label], count) for label, count in support.items())


def recompute(raw, base_spec, spec):
    stage = raw["stage"]
    if stage not in {"synthetic", "public"} or raw["phase"] not in {"development", "final"}:
        raise ValueError("Unknown phase or stage")
    if raw["workers"] != 1:
        raise ValueError("Timing observations require a single worker")
    synthetic = stage == "synthetic"
    families = base_spec["families" if synthetic else "public_families"]
    gaps = base_spec["gaps"] if synthetic else ["fixed"]
    seeds = base_spec["final_seeds" if raw["phase"] == "final" else "development_seeds"]
    expected = {(family, gap, seed) for family in families for gap in gaps for seed in seeds}
    observed = [(unit["family"], unit["gap"], unit["seed"]) for unit in raw["units"]]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("The independent reanalysis requires the complete registered roster")
    bandwidth = base_spec["primary_bandwidth_ratio"]
    horizon = base_spec["primary_horizon"]
    cap = spec["primary_speed_limit_ratio"]
    expected_settings = {
        (bandwidth_ratio, stop, cap)
        for bandwidth_ratio in base_spec["bandwidth_ratios"] for stop in base_spec["horizons"]
    } | {(bandwidth, horizon, alternative) for alternative in spec["sensitivity_speed_limit_ratios"]}

    accuracy = {name: [] for name in METHODS}
    per_query = {name: [] for name in METHODS}
    latency_ratios = {name: [] for name in METHODS}
    refinement = Counter()
    flips = Counter()
    relative_vs_clipped = queries = 0
    test_data = {}
    for unit in raw["units"]:
        labels = unit["y_test"]
        if not labels or any(type(label) != int for label in labels):
            raise ValueError("Invalid external labels")
        test_key = (unit["family"], unit["seed"])
        test_observation = (unit["test_input_sha256"], labels)
        if test_data.setdefault(test_key, test_observation) != test_observation:
            raise ValueError("Paired gaps must use the same external observations")
        settings = {}
        for result in unit["results"]:
            key = (result["bandwidth_ratio"], result["horizon"], result["speed_limit_ratio"])
            if key in settings or not isfinite(result["speed_limit"]) or result["speed_limit"] <= 0:
                raise ValueError("Duplicate setting or invalid speed limit")
            _validate_methods(result, labels, spec["timing_repeats"])
            settings[key] = result
        if settings.keys() != expected_settings:
            raise ValueError("Observed solver grid does not match the registered settings")

        methods = settings[bandwidth, horizon, cap]["methods"]
        static = methods["static"]
        static_median = median(static["latencies_s"])
        support = Counter(labels)
        queries += len(labels)
        for name in METHODS:
            method = methods[name]
            prediction = method["predictions"]
            accuracy[name].append(_balanced_accuracy(labels, prediction, support))
            median_time = median(method["latencies_s"])
            per_query[name].append(median_time * 1e6 / len(labels))
            latency_ratios[name].append(median_time / static_median)
            refinement[name] += _mismatches(prediction, method["refined_predictions"])
            flips[name] += _mismatches(prediction, static["predictions"])
        relative_vs_clipped += _mismatches(methods["relativistic"]["predictions"],
                                           methods["clipped_inertial"]["predictions"])

    accuracies = {name: mean(values) for name, values in accuracy.items()}
    metrics = dict(primary_gain=float(accuracies["relativistic"] - accuracies["clipped_inertial"]),
                   units=len(expected), queries=queries, relativistic_clipped_disagreements=relative_vs_clipped)
    for name in METHODS:
        metrics[f"{name}_balanced_accuracy"] = float(accuracies[name])
        metrics[f"{name}_median_us_per_query"] = median(per_query[name])
        metrics[f"{name}_median_latency_ratio"] = median(latency_ratios[name])
        metrics[f"{name}_refinement_disagreements"] = refinement[name]
        metrics[f"{name}_static_flips"] = flips[name]
    return metrics


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    content = args.input.read_bytes()
    base_spec = json.loads((ROOT / "crossing" / "study.json").read_bytes())
    spec = json.loads((ROOT / "relativistic" / "study.json").read_bytes())
    if args.seed != base_spec["final_seeds"][0]:
        parser.error("Reanalysis requires the registered execution seed")
    metrics = recompute(json.loads(content), base_spec, spec)
    Path("raw.json").write_bytes(content)
    Path("metrics.json").write_bytes((json.dumps(metrics, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    main()
