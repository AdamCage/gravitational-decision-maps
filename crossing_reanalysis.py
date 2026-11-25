"""Recalculate crossing headlines using independent counts and rational arithmetic."""

from argparse import ArgumentParser
from fractions import Fraction
from itertools import groupby
import json
from math import erf, fsum, isclose, sqrt
from pathlib import Path
from statistics import mean, median


def _average_precision(event, rank):
    total_positive = sum(event)
    positive = total = 0
    area = Fraction(0)
    data = sorted(zip(rank, event, strict=True), reverse=True)
    for _, group in groupby(data, key=lambda pair: pair[0]):
        group_labels = [v for _, v in group]
        mass = sum(group_labels)
        total += len(group_labels)
        positive += mass
        area += Fraction(mass * positive, total_positive * total)
    return area


def _oracle_check(oracle):
    errors = []
    for case in oracle["fixtures"]:
        faces = case["boundaries"]
        risk = fsum(face["density_slope_right_minus_left"] * face["velocity"]**2 / 2 for face in faces)
        flips = fsum(face["feature_density"] * abs(face["velocity"]) for face in faces)
        if not isclose(risk, case["predicted_coefficient"], abs_tol=1e-14) or not isclose(flips, case["predicted_flip_coefficient"], abs_tol=1e-14):
            raise ValueError("Oracle boundary terms disagree with the analytic coefficient")
        short = case["observations"][0]
        h = short["horizon"]
        scale = sqrt(2 * case["beta"])
        flip_mass = fsum(w * (erf((max(a["root"], b) - u) / scale) - erf((min(a["root"], b) - u) / scale)) / 2
                        for a, b in zip(faces, short["preimages"], strict=True)
                        for w, u in zip(case["masses"], case["centers"], strict=True))
        observed = fsum(short["excess_per_boundary"])
        if not isclose(observed, short["excess_cdf"], rel_tol=0, abs_tol=2e-13) or not isclose(flip_mass, short["flip_mass"], rel_tol=0, abs_tol=2e-13):
            raise ValueError("Oracle quadrature or CDF check failed")
        for predicted, measured in ((risk, observed / h**2), (flips, flip_mass / h)):
            if predicted == 0 and abs(measured) > 1e-13:
                raise ValueError("Stationary oracle discrepancy")
            errors.append(abs(measured / predicted - 1) if predicted else 0)
    return max(errors)


def recompute(raw, protocol):
    stage = raw["stage"]
    families = protocol["families" if stage == "synthetic" else "public_families"]
    gaps = protocol["gaps"] if stage == "synthetic" else ["fixed"]
    seeds = protocol["final_seeds" if raw["phase"] == "final" else "development_seeds"]
    expected = [(f, g, s) for f in families for g in gaps for s in seeds]
    observed = [(u["family"], u["gap"], u["seed"]) for u in raw["units"]]
    if observed != expected:
        raise ValueError("The independent reanalysis requires the complete pre-registered roster")
    measurements, latency, pr_gains = [], [], []
    flips = crosses = contradictions = triage_errors = 0
    test_keys = {}
    for unit in raw["units"]:
        key = (unit["family"], unit["seed"])
        frozen_test = (unit["test_input_sha256"], unit["y_test"])
        if test_keys.setdefault(key, frozen_test) != frozen_test:
            raise ValueError("Paired rows do not use common external observations")
        picked, = [r for r in unit["results"] if (r["bandwidth_ratio"], r["horizon"]) ==
                   (protocol["primary_bandwidth_ratio"], protocol["primary_horizon"])]
        fields = ("initial", "endpoint", "certified_endpoint", "taylor_endpoint", "drag_endpoint", "state_endpoint",
                  "certified_stable", "certified_crossing", "ball_stable", "sampled_crossing")
        initial = [(n, picked[n]) for n in fields]
        n = len(unit["y_test"])
        if len({len(values) for _, values in initial} | {n}) != 1 or not n:
            raise ValueError("Malformed unit")
        stable = ball = cert = taylor = 0
        for a, b, e, t, d, h, s, c, q, touched in zip(*(v for _, v in initial), strict=True):
            stable += int(s)
            ball += int(q)
            cert += int(e != -1)
            taylor += int(t != -1)
            crosses += int(c)
            flips += int(a != b)
            contradictions += sum((e not in (-1, b), t not in (-1, b), d not in (-1, b), h not in (-1, b),
                                   s and touched, q and touched, c and a == b))
        measurements.append((Fraction(stable - ball, n), Fraction(cert, n), Fraction(cert - taylor, n),
                             1 - Fraction(picked["nfe_hybrid"], picked["nfe_full"])))
        latency.append(picked["latency_s_hybrid"] / picked["latency_s_full"])
        event = [a != b for a, b in zip(picked["initial"], picked["endpoint"], strict=True)]
        if any(event) and not all(event):
            rank = picked["ranking"]
            pr_gains.append(_average_precision(event, rank["drag"]) - _average_precision(event, rank["static"]))
        triage_errors += sum(a != b for a, b in zip(picked["triage"], picked["endpoint"], strict=True))
    gain, coverage, increment, nfe = map(mean, zip(*measurements, strict=True))
    summary = {"primary_gain": float(gain), "endpoint_coverage": float(coverage), "interval_vs_taylor": float(increment),
            "nfe_reduction": float(nfe), "median_latency_ratio": median(latency), "certified_crossings": crosses,
            "endpoint_flips": flips, "contradictions": contradictions,
            "ap_drag_gain": float(mean(pr_gains)) if pr_gains else 0,
            "ap_mixed_units": len(pr_gains), "triage_mismatch": triage_errors, "units": len(expected)}
    if stage == "synthetic":
        summary["max_oracle_relative_error"] = _oracle_check(raw["known_law_oracle"])
    return summary


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    content = args.input.read_bytes()
    protocol = json.loads((Path(__file__).parent / "crossing" / "study.json").read_bytes())
    metrics = recompute(json.loads(content), protocol)
    Path("raw.json").write_bytes(content)
    Path("metrics.json").write_bytes((json.dumps(metrics, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    main()
