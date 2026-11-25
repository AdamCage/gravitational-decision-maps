"""Compact or verify the portable portion of the EpistemeOS export."""

from argparse import ArgumentParser
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from study_reanalysis import reanalyse


def require_digest(content, expected, name):
    if sha256(content).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch: {name}")


def require_metrics(recorded, recalculated):
    if recorded.keys() != recalculated.keys() or not all(
        math.isclose(recorded[key], value, rel_tol=0, abs_tol=1e-12) for key, value in recalculated.items()
    ):
        raise ValueError("Recomputed and recorded metrics disagree")


def bundle(destination, source=None, *, stage=None):
    active = source if source is not None else destination
    manifest_bytes = (active / "frozen.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    for name, digest in manifest["files"].items():
        require_digest((PROJECT / name).read_bytes(), digest, name)
    protocol = json.loads((PROJECT / "protocol.json").read_bytes())
    for name, registration in manifest["stages"].items():
        if stage is not None and name != stage:
            continue
        directory = active / name
        files = {key: (directory / key).read_bytes() for key in (
            "evidence.json", "summary.json", "metrics.json", "reanalysis-metrics.json", "graph.json")}
        evidence = json.loads(files["evidence.json"])
        primary, repeat = evidence["primary"], evidence["reanalysis"]
        if evidence["registration"] != registration or primary["status"] != "completed" or repeat["status"] != "completed":
            raise ValueError(f"Incomplete frozen evidence for {name}")
        require_digest(manifest_bytes, evidence["frozen_manifest_sha256"], "frozen.json")
        require_digest(files["graph.json"], evidence["graph_sha256"], "graph.json")
        for filename, key, run in [("metrics.json", "metrics", primary), ("summary.json", "summary", primary),
                                    ("reanalysis-metrics.json", "metrics", repeat)]:
            require_digest(files[filename], run["outputs"][key], filename)
        raw_path = directory / ("raw.json" if source is not None else "raw.json.gz")
        raw = raw_path.read_bytes() if source is not None else gzip.decompress(raw_path.read_bytes())
        for run in (primary, repeat):
            require_digest(raw, run["outputs"]["raw_data"], "per-example raw predictions")
        independently = reanalyse(json.loads(raw), protocol)
        for key in ("metrics.json", "reanalysis-metrics.json"):
            require_metrics(json.loads(files[key]), independently)

        if source is not None:
            target = destination / name
            target.mkdir(parents=True, exist_ok=True)
            files["raw.json.gz"] = gzip.compress(raw, compresslevel=9, mtime=0)
            for key, content in files.items():
                (target / key).write_bytes(content)
        print(f"verified {name}: {independently['row_count']} task-seed units")
    if source is not None:
        (destination / "frozen.json").write_bytes(manifest_bytes)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=PROJECT / "results")
    parser.add_argument("--export-dir", type=Path, help="Collect portable files from this existing EpistemeOS export")
    parser.add_argument("--stage", choices=["synthetic", "real"])
    args = parser.parse_args()
    bundle(args.results, args.export_dir, stage=args.stage)


if __name__ == "__main__":
    main()
