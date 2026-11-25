"""Register, run and check the follow-up under the pinned EpistemeOS runner."""

from argparse import ArgumentParser
from datetime import UTC, datetime
import gzip
from hashlib import sha256
import json
from math import isclose
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from gdm.evidence import EPISTEME_COMMIT, StudySources, _advance, _episteme, _export, _json_bytes, _write


ROOT = Path(__file__).resolve().parent.parent
STAGES = ("crossing_synthetic", "crossing_public")


def _sources():
    from episteme.execution_locked import project_files

    paths = [*(ROOT / "crossing").glob("*.py"), *(ROOT / "crossing_benchmark_data").glob("*.gz"),
             ROOT / "crossing_worker.py", ROOT / "crossing" / "study.json"]
    primary = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in paths}
    reanalysis = {"crossing_reanalysis.py": (ROOT / "crossing_reanalysis.py").read_bytes(),
                  "crossing/study.json": primary["crossing/study.json"]}
    return StudySources(primary, reanalysis, project_files(ROOT))


def _declare(store, sources, stage, workers):
    from episteme.execution_locked import freeze_closure, freeze_source
    from episteme.kernel import Actor, Kernel
    from episteme.planning import Planning

    config = json.loads(sources.primary["crossing/study.json"])
    subset = stage.removeprefix("crossing_")
    roster = config["families" if subset == "synthetic" else "public_families"]
    scope = {"population": ", ".join(roster), "stage": subset, "inference": "fixed core-only adversarial test of the certificate"}
    planner = Actor("crossing-planner", "planner")
    planning, kernel = Planning(store, planner), Kernel(store, planner)
    question = planning.question(study_id=f"gdm-{stage}", statement=config["question"],
                                 objective="Measure correctness and practical coverage for a label-free short-horizon certificate",
                                 scope=scope, constraints=["Frozen class scores and exact-autonomous-flow claim.",
                                                          "Reference tolerance and floating-point limitations must remain visible.",
                                                          "Only observed cores determine prototypes and scaling."],
                                 stopping_criteria=["One frozen final roster and one same-observation independent arithmetic check."])
    hypotheses = [kernel.hypothesis(a, b, c, scope) for a, b, c in (
        ("Directional certification has useful coverage", "Label certificates agree with the reference and avoid enough integration.",
         "Any contradictions or insufficient coverage/latency gains."),
        ("Enclosures are correct but too conservative", "Bounds agree on covered cases but often abstain or cost more than integration.",
         "A practically useful coverage and latency improvement on the frozen roster."),
        ("Net harm depends on geometry", "Wider shell gaps increase harmful endpoint changes on paired test rows.",
         "A reversal or negligible paired gap effect on held-out shell families."))]
    set_id = planning.explanation_set(question=question, hypotheses=hypotheses, comparison_plan=(
        "Compare analytic tie direction and interval bounds to DOP853; compare ball, Taylor, drag and state enclosures; "
        "paired no-step/endpoint classification on fixed test draws; time matched RK4 with and without the endpoint guard."))
    worker = freeze_source(store, sources.primary, entry_point="crossing_worker.py", input_name="study.json")
    reanalysis = freeze_source(store, sources.reanalysis, entry_point="crossing_reanalysis.py", input_name="observations.json")
    env = freeze_closure(store, sources.closure)
    seed = config["final_seeds"][0]
    details = store.put_json({"stage": subset, "workers": workers, "phase": "final", "execution_seed": seed})
    digest = sha256(sources.primary["crossing/study.json"]).hexdigest()
    protocol_id = kernel.preregister_for_set(
        explanation_set=set_id, design=f"Frozen config SHA-256 {digest}, reanalysis source {reanalysis}, roster {config['final_seeds']}.",
        metric="primary_gain", analysis_plan="Primary mean whole-path-stability coverage increment over the displacement-ball baseline.",
        stopping_rule="Run the complete roster before outcome-led follow-up. Report all contradictory or negative observations.",
        seeds=[seed], run_limit=2, implementation=worker, environment=env, data=details, replication_tolerance=1e-12)
    return {"question": question, "explanation_set": set_id, "protocol": protocol_id, "worker": worker,
            "reanalysis": reanalysis, "environment": env, "data": details, "execution_seed": seed}


def _digests(sources):
    return sources.digests() | {"gdm/evidence.py": sha256((ROOT / "gdm" / "evidence.py").read_bytes()).hexdigest()}


def _read(root):
    content = (root / "frozen.json").read_bytes()
    frozen = json.loads(content)
    if frozen["episteme_commit"] != EPISTEME_COMMIT or frozen["files"] != _digests(_sources()):
        raise ValueError("Source or runner drift since crossing registration")
    return content, frozen


def freeze(root, workers):
    from episteme.store import Store

    if not 1 <= workers <= 8:
        raise ValueError("Worker count must lie within 1..8")
    if root.exists():
        frozen = _read(root)[1]
        if frozen["workers"] != workers:
            raise ValueError("Cannot change a registered concurrency setting")
        return frozen
    root.parent.mkdir(parents=True, exist_ok=True)
    sources = _sources()
    with TemporaryDirectory(prefix="cross-register-", dir=root.parent) as temp:
        pending = Path(temp) / "store"
        stages = {}
        for stage in STAGES:
            with Store(pending / stage) as store:
                stages[stage] = _declare(store, sources, stage, workers)
        frozen = {"schema_version": 1, "frozen_utc": datetime.now(UTC).isoformat(), "episteme_commit": EPISTEME_COMMIT,
                  "execution_profile": "uv_locked_python_v2", "workers": workers, "files": _digests(sources), "stages": stages}
        _write(pending / "frozen.json", _json_bytes(frozen))
        pending.rename(root)
    return frozen


def execute(root, stage, export_dir):
    from episteme.store import Store

    content, frozen = _read(root)
    lock = frozen["stages"][stage]
    with Store(root / stage) as store:
        primary = _advance(store, stage, lock)
        if primary["status"] == "completed":
            _advance(store, stage, lock, primary["run"])
        result = _export(store, stage, lock, export_dir, content)
    agreement = result["reanalysis_agreement"]
    if agreement is None or not agreement["agrees"] or not agreement["identical_observations"]:
        raise RuntimeError(f"Incomplete or disagreeing results preserved in {export_dir / stage}")
    return result


def bundle(destination, export=None):
    from crossing_reanalysis import recompute

    base = destination if export is None else export
    content = (base / "frozen.json").read_bytes()
    frozen = json.loads(content)
    for name, digest in frozen["files"].items():
        if sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen source has changed: {name}")
    config = json.loads((ROOT / "crossing" / "study.json").read_bytes())
    for stage, registration in frozen["stages"].items():
        directory = base / stage
        files = {name: (directory / name).read_bytes() for name in (
            "metrics.json", "summary.json", "reanalysis-metrics.json", "evidence.json", "graph.json")}
        raw = ((directory / "raw.json").read_bytes() if export is not None
               else gzip.decompress((directory / "raw.json.gz").read_bytes()))
        ev = json.loads(files["evidence.json"])
        if ev["registration"] != registration or sha256(content).hexdigest() != ev["frozen_manifest_sha256"]:
            raise ValueError(f"Registration does not match evidence for {stage}")
        one, two = ev["primary"]["outputs"], ev["reanalysis"]["outputs"]
        checks = [(raw, one["raw_data"]), (raw, two["raw_data"]), (files["metrics.json"], one["metrics"]),
                  (files["summary.json"], one["summary"]), (files["reanalysis-metrics.json"], two["metrics"]),
                  (files["graph.json"], ev["graph_sha256"])]
        if any(sha256(data).hexdigest() != digest for data, digest in checks):
            raise ValueError(f"Artifact hashes do not match for {stage}")
        metrics = recompute(json.loads(raw), config)
        for name in ("metrics.json", "reanalysis-metrics.json"):
            values = json.loads(files[name])
            if values.keys() != metrics.keys() or any(not isclose(v, values[k], abs_tol=1e-12, rel_tol=0) for k, v in metrics.items()):
                raise ValueError(f"Independent numerical check failed for {stage}")
        if export is not None:
            files["raw.json.gz"] = gzip.compress(raw, mtime=0)
            for name, data in files.items():
                _write(destination / stage / name, data)
        print(f"verified {stage}: {metrics['units']} task-seed units", flush=True)
    if export is not None:
        _write(destination / "frozen.json", content)


def main():
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("freeze", "run", "bundle", "verify"):
        cmd = sub.add_parser(action)
        cmd.add_argument("--root", type=Path, default=ROOT / ".research" / "crossing-final")
        cmd.add_argument("--export", type=Path, default=ROOT / ".research" / "crossing-export")
        cmd.add_argument("--results", type=Path, default=ROOT / "crossing_results")
        if action in {"freeze", "run"}:
            cmd.add_argument("--episteme", type=Path, default=Path(os.environ.get("GDM_EPISTEME", ROOT.parent / "EpistemeOS")))
        if action == "run":
            cmd.add_argument("--stage", choices=STAGES, required=True)
        if action == "freeze":
            cmd.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.action in {"bundle", "verify"}:
        bundle(args.results.resolve(), args.export.resolve() if args.action == "bundle" else None)
        return
    _episteme(args.episteme)
    root = args.root.resolve()
    value = freeze(root, args.workers) if args.action == "freeze" else execute(root, args.stage, args.export.resolve())
    print(_json_bytes(value).decode(), end="", flush=True)


if __name__ == "__main__":
    main()
