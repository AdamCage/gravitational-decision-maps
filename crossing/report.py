"""Scoring for class crossing, skip coverage and the paired gap intervention."""

from collections import defaultdict
from statistics import mean, median

import numpy as np


def _balanced(y, pred):
    return mean(float(np.mean(pred[y == c] == c)) for c in np.unique(y))


def _record(row, result):
    y = np.asarray(row["y_test"], dtype=int)
    start, end, cert, tay, drag, state, full, hybrid, linear, euler, exponential, triaged = [np.asarray(result[key], dtype=int) for key in (
        "initial", "endpoint", "certified_endpoint", "taylor_endpoint", "drag_endpoint", "state_endpoint",
        "full", "hybrid", "linear", "euler", "exponential", "triage")]
    stable, cross, ball, sampled = [np.asarray(result[key], dtype=bool) for key in (
        "certified_stable", "certified_crossing", "ball_stable", "sampled_crossing")]
    arrays = (start, end, cert, tay, drag, state, full, hybrid, linear, euler, exponential, triaged, stable, cross, ball, sampled)
    if any(arr.shape != y.shape for arr in arrays):
        raise ValueError("Predictions and evaluator labels must have identical shapes")
    flip = start != end
    equal = (cert != -1) & (cert == end)
    lost = (start == y) & (end != y)
    gained = (start != y) & (end == y)
    interval_wrong = {"endpoint": int(np.sum((cert != -1) & ~equal)), "stable": int(np.sum(stable & sampled)),
                      "crossing": int(np.sum(cross & ~flip)), "ball": int(np.sum(ball & sampled)),
                      "taylor": int(np.sum((tay != -1) & (tay != end))), "drag": int(np.sum((drag != -1) & (drag != end))),
                      "state": int(np.sum((state != -1) & (state != end)))}
    if any(result["contradictions"][k] != interval_wrong[k] for k in result["contradictions"]):
        raise ValueError("Certificate indicator counts disagree with the saved rows")
    pred_crossings = defaultdict(lambda: [0, 0])
    for a, b, is_cert in zip(start[flip], end[flip], cross[flip], strict=True):
        direction = pred_crossings[f"{a}->{b}"]
        direction[0] += 1
        direction[1] += int(is_cert)
    return {
        "family": row["family"], "gap": row["gap"], "seed": row["seed"], "n": len(y),
        "horizon": result["horizon"], "bandwidth_ratio": result["bandwidth_ratio"],
        "endpoint_coverage": float(np.mean(cert != -1)), "stable_coverage": float(stable.mean()),
        "taylor_coverage": float(np.mean(tay != -1)), "drag_coverage": float(np.mean(drag != -1)),
        "state_coverage": float(np.mean(state != -1)),
        "ball_coverage": float(ball.mean()), "stable_gain_over_ball": float(stable.mean() - ball.mean()),
        "interval_vs_taylor": float(np.mean(cert != -1) - np.mean(tay != -1)),
        "flips": int(flip.sum()), "certified_crossings": int(cross.sum()),
        "certified_flip_endpoints": int(np.sum(flip & (cert != -1))),
        "sampled_crossings": int(sampled.sum()), "sampled_return": int(np.sum(sampled & ~flip)),
        "directions": dict(pred_crossings), "lost": int(lost.sum()), "gained": int(gained.sum()),
        "net_harm": float((lost.sum() - gained.sum()) / len(y)),
        "ba_start": _balanced(y, start), "ba_endpoint": _balanced(y, end),
        "linear_agreement": float(np.mean(linear == end)), "euler_agreement": float(np.mean(euler == end)),
        "full_mismatch": int(np.sum(full != end)), "hybrid_mismatch": int(np.sum(hybrid != end)),
        "exponential_mismatch": int(np.sum(exponential != end)), "triage_mismatch": int(np.sum(triaged != end)),
        "hybrid_full_mismatch": int(np.sum(hybrid != full)), "certificate_contradictions": interval_wrong,
        "nfe_reduction": 1 - result["nfe_hybrid"] / result["nfe_full"],
        "triage_nfe_reduction": 1 - result["nfe_triage"] / result["nfe_full"],
        "latency_ratio": result["latency_s_hybrid"] / result["latency_s_full"],
        "triage_latency_ratio": result["latency_s_triage"] / result["latency_s_full"],
        "exponential_latency_ratio": result["latency_s_exponential"] / result["latency_s_full"],
        "latency_us_full": result["latency_s_full"] * 1e6 / len(y),
        "latency_us_hybrid": result["latency_s_hybrid"] * 1e6 / len(y),
        "ap_linear": result["average_precision_linear"], "ap_drag": result["average_precision_drag"],
        "ap_margin": result["average_precision_margin"], "ap_exponential": result["average_precision_exponential"],
        "tie_probe": result["tie_probe"],
    }


def _mean_stats(rows):
    mean_keys = ("endpoint_coverage", "stable_coverage", "taylor_coverage", "drag_coverage", "state_coverage", "ball_coverage",
                 "stable_gain_over_ball", "interval_vs_taylor", "nfe_reduction", "triage_nfe_reduction", "ba_start", "ba_endpoint", "net_harm",
                 "linear_agreement", "euler_agreement")
    sum_keys = ("n", "flips", "certified_crossings", "certified_flip_endpoints", "sampled_crossings", "sampled_return",
                "lost", "gained", "full_mismatch", "hybrid_mismatch", "hybrid_full_mismatch", "exponential_mismatch", "triage_mismatch")
    profile = {key: mean(r[key] for r in rows) for key in mean_keys}
    profile.update({key: sum(r[key] for r in rows) for key in sum_keys})
    for key in ("latency_ratio", "latency_us_full", "latency_us_hybrid", "triage_latency_ratio", "exponential_latency_ratio"):
        profile[f"median_{key}"] = median(r[key] for r in rows)
    profile["contradictions"] = {key: sum(r["certificate_contradictions"][key] for r in rows)
                                 for key in rows[0]["certificate_contradictions"]}
    directions = defaultdict(lambda: [0, 0])
    for row in rows:
        for key, (count, caught) in row["directions"].items():
            directions[key][0] += count
            directions[key][1] += caught
    profile["directions"] = dict(directions)
    mixed = [r for r in rows if r["ap_margin"] is not None]
    profile["ap_mixed_units"] = len(mixed)
    for key in ("ap_linear", "ap_drag", "ap_margin", "ap_exponential"):
        profile[key] = mean(r[key] for r in mixed) if mixed else None
    profile["ap_drag_gain"] = mean(r["ap_drag"] - r["ap_margin"] for r in mixed) if mixed else None
    profile["ap_mixed_prevalence"] = mean(r["flips"] / r["n"] for r in mixed) if mixed else None
    profile["tie_probes"] = {k: sum(r["tie_probe"][k] for r in rows) for k in ("candidates", "valid", "wrong")}
    return profile


def _bootstrap(values, config):
    v = np.asarray(values)
    rng = np.random.default_rng(config["bootstrap_seed"])
    ix = rng.integers(len(v), size=(config["bootstrap_draws"], len(v)))
    return {"mean": float(v.mean()), "interval_95": np.quantile(v[ix].mean(axis=1), (0.025, 0.975)).tolist(),
            "by_seed": v.tolist()}


def build(raw, config, partial=False):
    stage, phase = raw["stage"], raw["phase"]
    units = raw["units"]
    families = config["families" if stage == "synthetic" else "public_families"]
    gaps = config["gaps"] if stage == "synthetic" else ["fixed"]
    seeds = config["final_seeds" if phase == "final" else "development_seeds"]
    keys = [(u["family"], u["gap"], u["seed"]) for u in units]
    roster = [(f, g, s) for f in families for g in gaps for s in seeds]
    if (keys != roster and not partial) or len(set(keys)) != len(keys):
        raise ValueError("Incomplete, unordered or duplicated roster")
    rows = []
    seen = {}
    for u in units:
        expected = [(r, t) for r in config["bandwidth_ratios"] for t in config["horizons"]]
        actual = [(r["bandwidth_ratio"], r["horizon"]) for r in u["results"]]
        if expected != actual:
            raise ValueError("Missing horizon or bandwidth pair")
        key = (u["family"], u["seed"])
        test_data = (u["test_input_sha256"], u["y_test"])
        if key in seen and seen[key] != test_data:
            raise ValueError("Unpaired test data across gap levels")
        seen[key] = test_data
        rows.extend(_record(u, r) for r in u["results"])

    primary = [r for r in rows if r["bandwidth_ratio"] == config["primary_bandwidth_ratio"] and r["horizon"] == config["primary_horizon"]]
    profile = _mean_stats(primary)
    aggregate = {"stage": stage, "phase": phase, "units": len(units), "runtime_seconds": raw["runtime_seconds"],
                 "primary": profile, "rows": rows, "settings": {}, "families": {}, "gap_net_harm": {}, "intervals": {}}
    for r in config["bandwidth_ratios"]:
        for t in config["horizons"]:
            matched = [u for u in rows if u["bandwidth_ratio"] == r and u["horizon"] == t]
            aggregate["settings"][f"{r}/{t}"] = _mean_stats(matched)
    for f in families:
        selected = [r for r in primary if r["family"] == f]
        if not selected:
            continue
        aggregate["families"][f] = _mean_stats(selected)
        if stage == "synthetic":
            by_gap = {g: {r["seed"]: r["net_harm"] for r in selected if r["gap"] == g} for g in gaps}
            if all(set(g) == set(seeds) for g in by_gap.values()):
                dif = [by_gap[gaps[-1]][s] - by_gap[gaps[0]][s] for s in seeds]
                aggregate["gap_net_harm"][f] = {"largest_minus_smallest": _bootstrap(dif, config),
                                                   "gap_means": {g: mean(s.values()) for g, s in by_gap.items()}}
    if stage == "synthetic" and all(f in aggregate["gap_net_harm"] for f in config["shell_families"]):
        shells = [aggregate["gap_net_harm"][f]["largest_minus_smallest"]["by_seed"] for f in config["shell_families"]]
        aggregate["shell_gap_net_harm"] = _bootstrap(np.asarray(shells).mean(axis=0), config)
    for metric in ("stable_gain_over_ball", "interval_vs_taylor", "endpoint_coverage", "nfe_reduction", "net_harm"):
        aggregate["intervals"][metric] = _bootstrap([mean(r[metric] for r in primary if r["seed"] == s) for s in seeds], config)
    for other in ("ap_margin", "ap_exponential"):
        groups = [[r["ap_drag"] - r[other] for r in primary if r["seed"] == s and r["ap_drag"] is not None] for s in seeds]
        aggregate["intervals"][f"ap_drag_minus_{other}"] = _bootstrap([mean(g) for g in groups if g], config) if any(groups) else None

    rules = config["decision"]
    aggregate["decision"] = {
        "no_certificate_contradictions": sum(profile["contradictions"].values()) <= rules["max_certificate_contradictions"],
        "reference_agreement": max(profile["full_mismatch"], profile["hybrid_mismatch"]) / profile["n"] <= rules["max_reference_label_disagreement"],
        "endpoint_coverage": profile["endpoint_coverage"] >= rules["minimum_mean_endpoint_certificate_coverage"],
        "nfe_reduction": profile["nfe_reduction"] >= rules["minimum_mean_relative_nfe_reduction"],
        "stable_gain_over_ball": profile["stable_gain_over_ball"] >= rules["minimum_local_gain_over_ball_coverage"],
        "latency": profile["median_latency_ratio"] <= rules["max_median_hybrid_to_full_latency"],
        "projected_crossing_rank": profile["ap_drag_gain"] is not None and profile["ap_drag_gain"] >= rules["minimum_mean_projected_ap_gain_over_static_margin"],
    }
    if "known_law_oracle" in raw:
        oracle = raw["known_law_oracle"]
        aggregate["known_law_oracle"] = oracle
        errors = []
        for f in oracle["fixtures"]:
            shortest = f["observations"][0]
            for target, measured in ((f["predicted_coefficient"], shortest["ratio_to_horizon_squared"]),
                                     (f["predicted_flip_coefficient"], shortest["ratio_to_horizon"])):
                if target == 0 and measured != 0:
                    raise ValueError("Stationary oracle has nonzero measured risk or mass")
                errors.append(abs(measured / target - 1) if target else 0)
        summary_error = max(errors)
        aggregate["max_oracle_relative_error"] = summary_error
        aggregate["decision"]["oracle_coefficient_check"] = summary_error <= config["oracle"]["max_relative_coefficient_error_at_shortest_horizon"]
    return aggregate


def headline(summary):
    primary = summary["primary"]
    result = {"primary_gain": primary["stable_gain_over_ball"], "endpoint_coverage": primary["endpoint_coverage"],
            "interval_vs_taylor": primary["interval_vs_taylor"], "nfe_reduction": primary["nfe_reduction"],
            "median_latency_ratio": primary["median_latency_ratio"], "certified_crossings": primary["certified_crossings"],
            "endpoint_flips": primary["flips"], "contradictions": sum(primary["contradictions"].values()),
            "ap_drag_gain": primary["ap_drag_gain"] if primary["ap_drag_gain"] is not None else 0,
            "ap_mixed_units": primary["ap_mixed_units"], "triage_mismatch": primary["triage_mismatch"],
            "units": summary["units"]}
    if "max_oracle_relative_error" in summary:
        result["max_oracle_relative_error"] = summary["max_oracle_relative_error"]
    return result
