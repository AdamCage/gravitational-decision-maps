"""Compare full and span-restricted endpoint predictions on the public affine roster."""

from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from crossing.flow import MixtureFlow, rk4
from tools import flat_boundary_transfer as transfer


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".research" / "flat-boundary-data"
HORIZON = 0.5
STEP = 0.025
RATIO = 0.35


def evaluate(name, cases, repeats):
    results = []
    for fold, (train, labels, queries, _, audit) in enumerate(cases, start=1):
        scaler = StandardScaler().fit(train)
        x = scaler.transform(queries)
        model = MixtureFlow.fit(scaler.transform(train), labels, prototypes=1, bandwidth_ratio=RATIO, seed=transfer.SEED)
        setup_start = perf_counter_ns()
        center = model.centers[0]
        sites = model.centers - center
        basis, _ = np.linalg.qr(sites[1:].T, mode="reduced")
        reduced = MixtureFlow(sites @ basis, model.source_labels, np.ones(len(sites)), model.beta)
        setup_ns = perf_counter_ns() - setup_start

        def full():
            endpoint, _ = rk4(model, x, HORIZON, STEP)
            return model.scores(endpoint)

        def projected():
            start = (x - center) @ basis
            endpoint, _ = rk4(reduced, start, HORIZON, STEP)
            return reduced.scores(endpoint)

        scores, reduced_scores = full(), projected()
        shift = scores - scores[:, :1]
        reduced_shift = reduced_scores - reduced_scores[:, :1]
        row = {
            "dataset": name, "fold": fold, "rows": len(x), "dimension": x.shape[1],
            "retained_dimension": basis.shape[1], "disagreements": int(np.count_nonzero(
                scores.argmax(-1) != reduced_scores.argmax(-1))),
            "max_absolute_margin_difference": float(np.max(np.abs(shift - reduced_shift))),
            **audit,
        }
        if repeats:
            samples = {"full": [], "projected": []}
            routines = (("full", full), ("projected", projected))
            for index in range(repeats):
                for key, routine in (routines if index % 2 == 0 else reversed(routines)):
                    start = perf_counter_ns()
                    routine()
                    samples[key].append(perf_counter_ns() - start)
            row["timing"] = {"basis_setup_ns": setup_ns, "full_median_ns": int(np.median(samples["full"])),
                             "projected_median_ns": int(np.median(samples["projected"])), "samples_ns": samples}
        results.append(row)
    return results


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=0, help="Also collect warmed full-batch timing samples")
    args = parser.parse_args()
    cases = {"iris": transfer.iris_cases(), "letter": transfer.letter_cases(args.cache), "har": transfer.har_cases(args.cache)}
    with threadpool_limits(limits=1):
        observations = [row for name, source in cases.items() for row in evaluate(name, source, args.repeats)]
    data = {"horizon": HORIZON, "max_rk4_step": STEP, "bandwidth_ratio": RATIO, "threads": 1,
            "includes_query_projection_in_timing": True, "sources": transfer.SOURCES, "observations": observations}
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
