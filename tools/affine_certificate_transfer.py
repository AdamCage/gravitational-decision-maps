"""Replay outward-rounded certificates on the full affine transfer roster.

Counts describe the same fixed sources and splits as flat_boundary_transfer.
The DOP853 checkpoints are a cross-check, independent of the certificates.
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from crossing.affine_certificate import VoronoiCertificate
from crossing.flow import MixtureFlow, reference_path
from tools import flat_boundary_transfer as transfer


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "crossing_results" / "affine_certificate_transfer.json"
CACHE = ROOT / ".research" / "flat-boundary-data"
FIELDS = ("rows", "strict_initial", "slab_stable", "stable", "switch", "numeric_sampled_departure", "numeric_endpoint_switch")


def evaluate(cases):
    folds = []
    for fold, (x_fit, y_fit, x_test, _, audit) in enumerate(cases, start=1):
        scaler = StandardScaler().fit(x_fit)
        training, queries = scaler.transform(x_fit), scaler.transform(x_test)
        for ratio in transfer.RATIOS:
            model = MixtureFlow.fit(training, y_fit, prototypes=1, bandwidth_ratio=ratio, seed=transfer.SEED)
            cert = VoronoiCertificate(model.centers, model.beta)
            counts = [dict.fromkeys(FIELDS, 0) for _ in transfer.HORIZONS]
            for start in range(0, len(queries), transfer.REFERENCE_BATCH_SIZE):
                batch = queries[start:start + transfer.REFERENCE_BATCH_SIZE]
                path, _ = reference_path(model, batch, transfer.CHECKPOINTS)
                labels = model.scores(path.reshape(-1, batch.shape[1])).argmax(axis=-1).reshape(len(path), len(batch))
                initial = model.scores(batch).argmax(axis=-1)
                for i, horizon in enumerate(transfer.HORIZONS):
                    validated, stable, switch, slab = cert.classify(batch, horizon)
                    stop = (i + 1) * transfer.CHECKPOINTS_PER_INCREMENT
                    departure = (labels[:stop] != initial).any(axis=0)
                    endpoint_switch = labels[stop - 1] != initial
                    strict = validated >= 0
                    if (np.any(strict & (validated != initial)) or np.any((stable | slab) & departure)
                            or np.any(switch & ~endpoint_switch)):
                        raise ArithmeticError(f"Numerical cross-check disagrees in fold {fold} at horizon {horizon}")
                    values = (len(batch), strict.sum(), slab.sum(), stable.sum(), switch.sum(),
                              departure.sum(), endpoint_switch.sum())
                    for field, value in zip(FIELDS, values, strict=True):
                        counts[i][field] += int(value)
            folds.extend({"fold": fold, "ratio": ratio, "horizon": horizon, "counts": count, **audit}
                         for horizon, count in zip(transfer.HORIZONS, counts, strict=True))
    overall = []
    for ratio in transfer.RATIOS:
        for horizon in transfer.HORIZONS:
            subset = [row["counts"] for row in folds if row["ratio"] == ratio and row["horizon"] == horizon]
            counts = {field: sum(row[field] for row in subset) for field in FIELDS}
            n = counts["rows"]
            overall.append({
                "ratio": ratio, "horizon": horizon, "counts": counts,
                "switch_rate_interval": [counts["switch"] / n, 1 - counts["stable"] / n],
            })
    return {"overall": overall, "per_fold": folds}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    cases = {"iris": transfer.iris_cases(), "letter": transfer.letter_cases(args.cache), "har": transfer.har_cases(args.cache)}
    with threadpool_limits(limits=1):
        payload = {
            "study": "Outward-rounded whole-path stability and endpoint-switch certificates",
            "inputs": "Exact float64 values of the train-standardized queries, centroids and fitted bandwidths",
            "design": "Post hoc replay of the frozen transfer roster, without adaptive choices or data-set inference",
            "sources": transfer.SOURCES,
            "results": {name: evaluate(source) for name, source in cases.items()},
        }
    transfer.write_result(args.output, payload)


if __name__ == "__main__":
    main()
