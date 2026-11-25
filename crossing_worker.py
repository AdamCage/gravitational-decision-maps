"""Run the new boundary crossing study or development fixtures."""

from argparse import ArgumentParser
import json
from pathlib import Path
import platform
from time import perf_counter

from crossing.experiment import SPEC_PATH, run
from crossing.report import build, headline


def write_json(path, data):
    path.write_bytes((json.dumps(data, allow_nan=False, separators=(",", ":")) + "\n").encode())


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--development", choices=("synthetic", "public"))
    parser.add_argument("--family")
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--output", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.development:
        if args.input or args.seed is not None:
            parser.error("Development runs accept no frozen input")
        phase, stage, workers = "development", args.development, args.workers
    else:
        if args.input is None or args.seed is None or args.family is not None:
            parser.error("Final evaluation needs its registered EpistemeOS input and job seed")
        deployment = json.loads(args.input.read_bytes())
        if deployment["phase"] != "final" or deployment["execution_seed"] != args.seed:
            parser.error("EpistemeOS input and execution seed do not agree")
        phase, stage, workers = "final", deployment["stage"], deployment["workers"]
    spec = json.loads(SPEC_PATH.read_bytes())
    start = perf_counter()
    units = run(spec, stage, phase, workers=workers, family=args.family)
    raw = {"stage": stage, "phase": phase, "workers": workers, "python": platform.python_version(),
           "runtime_seconds": perf_counter() - start, "units": units}
    if stage == "synthetic":
        from crossing.risk_oracle import oracle_experiments

        raw["known_law_oracle"] = oracle_experiments()
    report = build(raw, spec, partial=args.family is not None)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "raw.json", raw)
    write_json(args.output / "summary.json", report)
    write_json(args.output / "metrics.json", headline(report))


if __name__ == "__main__":
    main()
