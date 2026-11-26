"""Run the locked finite-horizon inertia comparison or its development smoke test."""

from argparse import ArgumentParser
import json
from pathlib import Path

from relativistic.experiment import SPEC_PATH, STUDY_PATH, run
from relativistic.report import build, headline


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--development", choices=("synthetic", "public"))
    parser.add_argument("--family")
    parser.add_argument("--output", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.development:
        if args.input or args.seed is not None:
            parser.error("Development runs accept no frozen input")
        stage, phase = args.development, "development"
    else:
        if args.input is None or args.seed is None or args.family is not None:
            parser.error("Final evaluation requires a frozen input and execution seed")
        deployment = json.loads(args.input.read_bytes())
        if (deployment["phase"] != "final" or deployment["execution_seed"] != args.seed
                or deployment["workers"] != 1 or deployment["stage"] not in {"synthetic", "public"}):
            parser.error("The registration does not match this single-worker final execution")
        stage, phase = deployment["stage"], "final"
    protocol = json.loads(SPEC_PATH.read_bytes())
    study = json.loads(STUDY_PATH.read_bytes())
    raw = run(protocol, study, stage, phase, args.family)
    summary = build(raw, protocol, study)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in (("raw", raw), ("summary", summary), ("metrics", headline(summary))):
        content = json.dumps(data, allow_nan=False, separators=(",", ":")) + "\n"
        (args.output / f"{name}.json").write_bytes(content.encode())


if __name__ == "__main__":
    main()
