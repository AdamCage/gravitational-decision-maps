"""Run one frozen EpistemeOS workload, or run a development-only smoke test."""

from argparse import ArgumentParser
import json
from pathlib import Path
import platform
from time import perf_counter

from gdm.experiment import run_stage
from gdm.score import build_report, write_json


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--development", choices=["synthetic", "real"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--task", action="append")
    parser.add_argument("--output", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.development is not None:
        if args.manifest is not None or args.seed is not None:
            parser.error("Development and frozen execution use distinct entry points")
        stage, split, workers = args.development, "development", args.workers
    else:
        if args.manifest is None or args.seed is None or args.task is not None:
            parser.error("Frozen execution requires the EpistemeOS manifest and run seed")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if manifest["phase"] != "final":
            parser.error("The EpistemeOS manifest must select the final regime")
        stage, split, workers = manifest["stage"], "final", manifest["workers"]

    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text(encoding="utf-8"))
    if not 1 <= workers <= 16:
        parser.error("Worker count must be between 1 and 16")
    args.output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    rows = run_stage(protocol, stage, split, workers, tasks=args.task)
    raw = {"stage": stage, "split": split, "workers": workers, "rows": rows,
           "runtime_seconds": perf_counter() - start, "python": platform.python_version()}
    write_json(args.output / "raw.json", raw)
    summary = build_report(raw, protocol, partial=args.task is not None)
    write_json(args.output / "summary.json", summary)
    metrics = {"row_count": len(rows)}
    if args.task is None:
        primary_gain = summary["contrasts"]["gradient vs static_champion"]["mean_delta"]
        metrics.update({"primary_gain": primary_gain, "delta_vs_static": primary_gain,
                        "delta_vs_matched": summary["contrasts"]["gradient vs matched_static"]["mean_delta"],
                        "gradient_external_balanced_accuracy": summary["aggregate_ba"]["gradient"],
                        "median_latency_ratio": summary["median_latency_ratio"]})
    write_json(args.output / "metrics.json", metrics)
    print(f"scored {stage} {split}: {len(rows)} task-seed units", flush=True)


if __name__ == "__main__":
    main()
