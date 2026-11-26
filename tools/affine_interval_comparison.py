"""Compare two verified approaches on identical encoded Voronoi queries."""

from argparse import ArgumentParser
from decimal import Decimal
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from crossing.affine_certificate import VoronoiCertificate
from crossing.flow import MixtureFlow
from tools import flat_boundary_transfer as transfer
from tools.flat_boundary_interval import FIXTURES, Flow, STEP, point


HORIZONS = (0.125, 0.5)


def interval_verdicts(flow, queries):
    shape = (len(HORIZONS), len(queries))
    stable = np.zeros(shape, dtype=bool)
    switched = np.zeros(shape, dtype=bool)
    initial = np.full(len(queries), -1)
    observation_steps = [int(Decimal.from_float(horizon) / STEP) for horizon in HORIZONS]
    for source, query in enumerate(queries):
        box = tuple(point(Decimal.from_float(float(value))) for value in query)
        leader = flow.winner(box)
        if leader is None:
            continue
        initial[source] = leader
        path_is_stable = True
        for step in range(1, observation_steps[-1] + 1):
            tube, box = flow.step(box)
            path_is_stable = path_is_stable and flow.certifies(leader, tube)
            if step in observation_steps:
                row = observation_steps.index(step)
                endpoint = flow.winner(box)
                stable[row, source] = path_is_stable
                switched[row, source] = endpoint is not None and endpoint != leader
    return initial, stable, switched


def compare(name, sites, beta, queries, timing):
    encoded = {"sites": [[Decimal.from_float(float(value)) for value in row] for row in sites],
               "beta": Decimal.from_float(float(beta))}
    interval = Flow(encoded)
    projected = VoronoiCertificate(sites, beta)
    start = perf_counter_ns()
    local = [projected.classify(queries, horizon) for horizon in HORIZONS]
    local_ns = perf_counter_ns() - start
    start = perf_counter_ns()
    initial, stable, switched = interval_verdicts(interval, queries)
    interval_ns = perf_counter_ns() - start
    result = {"case": name, "rows": len(queries), "horizons": []}
    if timing:
        result["path_evaluation_ns"] = {"local": local_ns, "interval": interval_ns}
    for index, (horizon, verdict) in enumerate(zip(HORIZONS, local, strict=True)):
        both_initial = (verdict.initial >= 0) & (initial >= 0)
        if (np.any(both_initial & (verdict.initial != initial))
                or np.any(verdict.stable & switched[index]) or np.any(verdict.switched & stable[index])):
            raise ArithmeticError(f"Certified results conflict on {name}, horizon {horizon}")
        result["horizons"].append({
            "horizon": horizon,
            "slab_stable": int(verdict.slab_stable.sum()),
            "local_stable": int(verdict.stable.sum()), "local_switch": int(verdict.switched.sum()),
            "interval_stable": int(stable[index].sum()), "interval_switch": int(switched[index].sum()),
            "both_resolved": int(((verdict.stable | verdict.switched) & (stable[index] | switched[index])).sum()),
        })
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing", action="store_true", help="Record descriptive wall times including query arithmetic")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        rows = [compare(name, np.array(case["sites"], dtype=float), float(case["beta"]),
                        np.array(case["queries"], dtype=float), args.timing) for name, case in FIXTURES.items()]
        for number, (train, labels, queries, _, _) in enumerate(transfer.iris_cases(), start=1):
            scaler = StandardScaler().fit(train)
            model = MixtureFlow.fit(scaler.transform(train), labels, prototypes=1,
                                    bandwidth_ratio=0.35, seed=transfer.SEED)
            rows.append(compare(f"iris_fold_{number}", model.centers, model.beta, scaler.transform(queries), args.timing))
    transfer.write_result(args.output, {
        "study": "Same binary64 encodings for the local binary64 certificate and 38-digit Decimal interval solver",
        "interval_step": str(STEP), "timing_includes_fit_or_model_setup": False,
        "comparison": "Number certified for continuous-time stability or strict terminal disagreement",
        "cases": rows,
    })


if __name__ == "__main__":
    main()
