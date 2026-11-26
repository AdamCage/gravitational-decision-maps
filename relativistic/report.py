"""Summarize paired accuracy, empirical label stability, and timings."""

from statistics import mean, median

import numpy as np


def _balanced(y, prediction):
    return mean(float(np.mean(prediction[y == label] == label)) for label in np.unique(y))


def _profile(rows, methods):
    return {method: {
        "balanced_accuracy": mean(row[method]["balanced_accuracy"] for row in rows),
        "median_us_per_query": median(row[method]["us_per_query"] for row in rows),
        "median_latency_ratio": median(row[method]["latency_ratio"] for row in rows),
        "median_batch_ms": median(row[method]["batch_ms"] for row in rows),
        "mean_relative_displacement": mean(row[method]["mean_relative_displacement"] for row in rows),
        "static_flips": sum(row[method]["static_flips"] for row in rows),
        "correct_to_wrong": sum(row[method]["correct_to_wrong"] for row in rows),
        "wrong_to_correct": sum(row[method]["wrong_to_correct"] for row in rows),
        "refinement_disagreements": sum(row[method]["refinement_disagreements"] for row in rows),
        "nfe_per_query": sorted(set(row[method]["nfe"] for row in rows)),
    } for method in methods}


def _score_unit(unit, result, methods):
    y = np.asarray(unit["y_test"])
    static = np.asarray(result["methods"]["static"]["predictions"])
    static_time = median(result["methods"]["static"]["latencies_s"])
    row = {key: unit[key] for key in ("family", "gap", "seed")}
    row["queries"] = len(y)
    for method in methods:
        raw = result["methods"][method]
        pred = np.asarray(raw["predictions"])
        fine = np.asarray(raw["refined_predictions"])
        if not len(y) or y.shape != pred.shape or y.shape != fine.shape:
            raise ValueError("Mismatched outcome arrays")
        elapsed = median(raw["latencies_s"])
        row[method] = {
            "balanced_accuracy": _balanced(y, pred), "us_per_query": elapsed * 1e6 / len(y),
            "batch_ms": elapsed * 1e3, "latency_ratio": elapsed / static_time,
            "mean_relative_displacement": raw["mean_relative_displacement"],
            "static_flips": int(np.count_nonzero(pred != static)),
            "correct_to_wrong": int(np.count_nonzero((static == y) & (pred != y))),
            "wrong_to_correct": int(np.count_nonzero((static != y) & (pred == y))),
            "refinement_disagreements": int(np.count_nonzero(pred != fine)), "nfe": raw["nfe"],
        }
    row["relativistic_clipped_disagreements"] = sum(
        a != b for a, b in zip(result["methods"]["relativistic"]["predictions"],
                               result["methods"]["clipped_inertial"]["predictions"], strict=True))
    row["relativistic_clipped_latency_ratio"] = row["relativistic"]["batch_ms"] / row["clipped_inertial"]["batch_ms"]
    return row


def _paired_intervals(rows, seeds, methods, protocol):
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    indices = rng.integers(len(seeds), size=(protocol["bootstrap_draws"], len(seeds)))
    result = {}
    for baseline in methods:
        if baseline == "relativistic":
            continue
        values = np.asarray([mean(row["relativistic"]["balanced_accuracy"] - row[baseline]["balanced_accuracy"]
                                  for row in rows if row["seed"] == seed) for seed in seeds])
        interval = np.quantile(values[indices].mean(axis=1), (0.025, 0.975)).tolist()
        result[baseline] = {"mean": float(values.mean()), "interval_95": interval, "by_seed": values.tolist()}
    return result


def build(raw, protocol, study):
    from .experiment import settings

    methods = study["methods"]
    seeds = protocol[f"{raw['phase']}_seeds"]
    primary = (protocol["primary_bandwidth_ratio"], protocol["primary_horizon"], study["primary_speed_limit_ratio"])
    by_setting = {setting: [] for setting in settings(protocol, study)}
    for unit in raw["units"]:
        results = unit["results"]
        keys = [(r["bandwidth_ratio"], r["horizon"], r["speed_limit_ratio"]) for r in results]
        if keys != list(by_setting):
            raise ValueError("Incomplete or out-of-order comparison settings")
        for key, result in zip(keys, results, strict=True):
            by_setting[key].append(_score_unit(unit, result, methods))
    primary_rows = by_setting[primary]
    intervals = _paired_intervals(primary_rows, seeds, methods, protocol)
    primary_profile = _profile(primary_rows, methods)
    sensitivity = []
    for (ratio, horizon, cap), rows in by_setting.items():
        sensitivity.append({"bandwidth_ratio": ratio, "horizon": horizon, "speed_limit_ratio": cap,
                            "methods": _profile(rows, methods),
                            "relativistic_clipped_disagreements": sum(r["relativistic_clipped_disagreements"] for r in rows)})
    n = sum(row["queries"] for row in primary_rows)
    speed = median(row["relativistic_clipped_latency_ratio"] for row in primary_rows)
    families = {family: _profile([row for row in primary_rows if row["family"] == family], methods)
                for family in dict.fromkeys(row["family"] for row in primary_rows)}
    disagreement = max(s["methods"][method]["refinement_disagreements"] / n for s in sensitivity for method in methods)
    return {"stage": raw["stage"], "phase": raw["phase"], "primary": primary_profile, "intervals": intervals,
            "queries": n, "units": len(primary_rows), "timing_environment": raw["timing_environment"],
            "runtime_seconds": raw["runtime_seconds"], "families": families, "settings": sensitivity,
            "relativistic_clipped_disagreements": sum(row["relativistic_clipped_disagreements"] for row in primary_rows),
            "median_relativistic_clipped_latency_ratio": speed, "max_refinement_disagreement_rate": disagreement,
            "decision": {
                "gain_over_clipped": intervals["clipped_inertial"]["mean"] >= study["minimum_mean_gain"]
                and intervals["clipped_inertial"]["interval_95"][0] > 0,
                "gain_over_static": intervals["static"]["interval_95"][0] > 0,
                "latency": speed <= study["maximum_median_latency_ratio_to_clipped"],
                "refinement": disagreement <= study["maximum_refinement_disagreement"],
            }, "rows": primary_rows}


def headline(summary):
    names = ("balanced_accuracy", "median_us_per_query", "refinement_disagreements", "static_flips", "median_latency_ratio")
    fields = {f"{method}_{name}": values[name] for method, values in summary["primary"].items() for name in names}
    return fields | {"primary_gain": summary["intervals"]["clipped_inertial"]["mean"],
                     "units": summary["units"], "queries": summary["queries"],
                     "relativistic_clipped_disagreements": summary["relativistic_clipped_disagreements"]}
