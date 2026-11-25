"""Run both frozen GDM stages through EpistemeOS's locked execution profile."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile
from uuid import uuid4
import json
import math
import os
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EPISTEME_COMMIT = "7f5a1a0e43d7c35196cc8b3c9c74a618aa2559c3"
STAGES = ("synthetic", "real")
METRIC = "primary_gain"
WALL_SECONDS = 1800
REANALYSIS_SECONDS = 300
MAX_OUTPUT_BYTES = 64 * 1024 ** 2
PRIMARY_OUTPUTS = {"raw_data": "raw.json", "metrics": "metrics.json", "summary": "summary.json"}
REANALYSIS_OUTPUTS = {"raw_data": "raw.json", "metrics": "metrics.json"}
AGREEMENT_TOLERANCE = 1e-12


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _digest(data):
    return sha256(data).hexdigest()


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        pending = Path(temp.name)
        temp.write(data)
    try:
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def _episteme(path):
    checkout = path.expanduser().resolve()
    found = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    changed = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--", "src"], text=True,
    ).strip()
    if found != EPISTEME_COMMIT or changed:
        raise ValueError(f"Use the EpistemeOS source checkout at {EPISTEME_COMMIT} with a clean src directory")
    source_path = checkout / "src"
    sys.path.insert(0, str(source_path))
    import episteme.store

    if not Path(episteme.store.__file__).resolve().is_relative_to(source_path):
        raise ValueError("An EpistemeOS package outside the pinned checkout is already imported")


@dataclass(frozen=True)
class StudySources:
    primary: dict[str, bytes]
    reanalysis: dict[str, bytes]
    closure: dict[str, bytes]

    def digests(self):
        return {name: _digest(data) for name, data in sorted({**self.primary, **self.reanalysis, **self.closure}.items())}


def _sources(project=PROJECT_ROOT):
    from episteme.execution_locked import project_files

    paths = sorted((project / "gdm").glob("*.py")) + sorted((project / "benchmark_data").glob("*.gz"))
    paths += [project / "study_worker.py", project / "protocol.json"]
    primary = {path.relative_to(project).as_posix(): path.read_bytes() for path in paths}
    reanalysis = {"study_reanalysis.py": (project / "study_reanalysis.py").read_bytes(),
                  "protocol.json": primary["protocol.json"]}
    return StudySources(primary, reanalysis, project_files(project))


def _declare(stage, store, sources, protocol, workers):
    from episteme.execution_locked import freeze_closure, freeze_source
    from episteme.kernel import Actor, Kernel
    from episteme.planning import Planning

    study_id = f"gdm-{stage}"
    planner = Actor("gdm-study-planner", "planner")
    planning = Planning(store, planner)
    kernel = Kernel(store, planner)
    settings = protocol["stages"][stage]
    stopping = protocol["stopping"]
    scope = {"benchmark_stage": stage, "information": "inductive, core only",
             "population": ", ".join(settings["tasks"]), "inference": "bounded falsification pilot"}
    question = planning.question(
        study_id=study_id, statement=protocol["question"],
        objective="Estimate the paired external balanced-accuracy increment of learned attraction dynamics.",
        scope=scope,
        constraints=["Only labeled core examples for preprocessing, model selection and fitting.",
                     "Predictions unresolved at the fixed solver horizon count as errors.",
                     "Benchmark rules and seed roster are frozen before evaluating external labels."],
        stopping_criteria=[stopping, "One locked primary run and one implementation-independent same-data check."],
    )
    hypotheses = [
        kernel.hypothesis("A distinct integration effect", "Trajectories improve on both static controls under the locked criteria.",
                          "The static champion or the same-geometry static readout matches the trajectories.", scope),
        kernel.hypothesis("Geometry explains the observed effect", "A shared representation, prototype or metric explains any apparent gain.",
                          "Trajectories gain over the matched geometry as well as over the static champion.", scope),
        kernel.hypothesis("Finite solver behavior dominates", "Readout sensitivity or numerical non-resolution offsets any extrapolation gain.",
                          "Solver sensitivity is modest and all-example gains survive cost and stability checks.", scope),
    ]
    explanations = planning.explanation_set(
        question=question, hypotheses=hypotheses,
        comparison_plan="Use paired scores across fixed seeds, a validation-selected static champion, and identical frozen geometry without integration.",
    )
    primary = freeze_source(store, sources.primary, entry_point="study_worker.py", input_name="study.json")
    reanalysis = freeze_source(store, sources.reanalysis, entry_point="study_reanalysis.py", input_name="observations.json")
    environment = freeze_closure(store, sources.closure)
    data = store.put_json({"stage": stage, "workers": workers, "phase": "final"})
    run_seed = protocol["final_seeds"][0]
    plan = kernel.preregister_for_set(
        explanation_set=explanations,
        design=(f"Protocol SHA-256 {_digest(sources.primary['protocol.json'])}; "
                f"separate reanalysis source tree {reanalysis}; frozen internal seed roster {protocol['final_seeds']}. "
                f"The mechanical execution seed {run_seed} identifies a single job over the whole roster."),
        metric=METRIC,
        analysis_plan=("Mean across frozen task groups and paired split seeds. "
                       f"A practical gain is at least {settings['minimum_gain']} balanced accuracy points; "
                       f"the latency ratio limit is {settings['maximum_latency_ratio']}."),
        stopping_rule=stopping,
        seeds=[run_seed], run_limit=2,
        implementation=primary, environment=environment, data=data, replication_tolerance=AGREEMENT_TOLERANCE,
    )
    return dict(question=question, explanation_set=explanations, protocol=plan, worker=primary,
                reanalysis=reanalysis, environment=environment, data=data, execution_seed=run_seed)


def freeze(root, workers, project=PROJECT_ROOT):
    from episteme.store import Store

    if not 1 <= workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    if root.exists():
        frozen = _locked(root, project)
        if workers != frozen["workers"]:
            raise ValueError("Worker count already locked for this study")
        return frozen
    sources = _sources(project)
    protocol = json.loads(sources.primary["protocol.json"])
    root.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="gdm-freeze-", dir=root.parent) as directory:
        pending = Path(directory) / "study"
        pending.mkdir()
        stage_locks = {}
        for stage in STAGES:
            with Store(pending / stage) as store:
                stage_locks[stage] = _declare(stage, store, sources, protocol, workers)
        frozen = dict(schema_version=1, frozen_utc=datetime.now(UTC).isoformat(),
                      episteme_commit=EPISTEME_COMMIT, execution_profile="uv_locked_python_v2",
                      workers=workers, files=sources.digests(), stages=stage_locks)
        _write(pending / "frozen.json", _json_bytes(frozen))
        pending.rename(root)
    return frozen


def _locked(root, project=PROJECT_ROOT):
    frozen = json.loads((root / "frozen.json").read_bytes())
    if frozen.get("schema_version") != 1 or frozen.get("episteme_commit") != EPISTEME_COMMIT:
        raise ValueError("Unrecognized freeze manifest")
    current = _sources(project).digests()
    if current != frozen["files"]:
        diff = sorted(name for name in current.keys() | frozen["files"].keys()
                      if current.get(name) != frozen["files"].get(name))
        raise ValueError(f"Changes since study freeze: {', '.join(diff)}")
    return frozen


def _find_job(store, plan, replicate_of):
    history = store.events()
    runs = [event["id"] for event in history if event["kind"] == "run"
            and event["payload"]["protocol"] == plan and event["payload"]["replicate_of"] == replicate_of]
    jobs = [event["id"] for event in history if event["kind"] == "execution_job" and event["payload"]["run"] in runs]
    if len(runs) != len(jobs) or len(jobs) > 1:
        raise ValueError("Ambiguous EpistemeOS job history for this pre-registered run")
    return jobs[0] if jobs else None


def _advance(store, stage, lock, replicate_of=None):
    from episteme.commands import CommandService
    from episteme.execution import work_job

    job = _find_job(store, lock["protocol"], replicate_of)
    if job is None:
        reanalysis = replicate_of is not None
        role = "replicator" if reanalysis else "executor"
        options = {"implementation": lock["reanalysis"], "replicate_of": replicate_of} if reanalysis else {}
        context = dict(command_id=uuid4().hex, expected_revision=len(store.events()),
                       actor=f"gdm-{role}", role=role, study_id=f"gdm-{stage}",
                       correlation_id=f"gdm-{stage}-locked", causation_id=replicate_of or lock["protocol"])
        payload = dict(protocol=lock["protocol"], seed=lock["execution_seed"],
                       outputs=REANALYSIS_OUTPUTS if reanalysis else PRIMARY_OUTPUTS,
                       wall_seconds=REANALYSIS_SECONDS if reanalysis else WALL_SECONDS,
                       max_output_bytes=MAX_OUTPUT_BYTES, required_capabilities=["locked_environment"], **options)
        request = dict(version=1, action="execution.enqueue", payload=payload)
        job = CommandService(store).execute(dict(context=context, request=request))
    return work_job(store, job)


def _result(store, state):
    if state["result"] is None:
        return None
    return next(event["payload"] for event in store.events() if event["id"] == state["result"])


def _execution_details(store, state, outputs, directory, prefix):
    result = _result(store, state)
    if result is None:
        return {**state, "outputs": {}}
    digests = result["outputs"]
    for label, name in outputs.items():
        if label in digests and (prefix == "" or label != "raw_data"):
            _write(directory / (prefix + name), store.read(digests[label]))
    if "environment_record" in digests:
        _write(directory / (prefix + "environment.json"), store.read(digests["environment_record"]))
    if "execution_manifest" in digests:
        manifest = store.read(digests["execution_manifest"])
        _write(directory / (prefix + "execution.json"), manifest)
        record = json.loads(manifest)
        for stream in ("stdout", "stderr"):
            _write(directory / (prefix + stream + ".txt"), store.read(record[stream]["sha256"]))
    return {**state, "outputs": digests, "reason": result["reason"]}


def _compare_metrics(measured, recalculated):
    details = {}
    for name in sorted(measured.keys() | recalculated.keys()):
        primary, repeated = measured.get(name), recalculated.get(name)
        agrees = (type(primary) in (int, float) and type(repeated) in (int, float)
                  and math.isfinite(primary) and math.isfinite(repeated)
                  and math.isclose(primary, repeated, abs_tol=AGREEMENT_TOLERANCE, rel_tol=0))
        details[name] = dict(primary=primary, reanalysis=repeated, agrees=agrees)
    return details


def _export(store, stage, lock, output, frozen_bytes):
    from episteme.execution import job_state
    from episteme.graph import ResearchGraph
    from episteme.reporting import export_store

    job = _find_job(store, lock["protocol"], None)
    if job is None:
        raise ValueError("Study has no primary execution job yet")
    directory = output / stage
    graph = ResearchGraph.from_store(store)
    graph_bytes = graph.to_json().encode("utf-8")
    exports = {name: Path(path).read_bytes() for name, path in export_store(store).items()}
    if json.loads(exports["review-bundle.json"])["summary"]["last_event_hash"] != graph.snapshot_hash:
        raise ValueError("EpistemeOS store changed during export. Run export again.")
    _write(directory / "graph.json", graph_bytes)
    for name, data in exports.items():
        _write(directory / name, data)
    _write(output / "frozen.json", frozen_bytes)
    state = job_state(store, job)
    primary = _execution_details(store, state, PRIMARY_OUTPUTS, directory, "")
    replica = _find_job(store, lock["protocol"], state["run"])
    repeated = (_execution_details(store, job_state(store, replica), REANALYSIS_OUTPUTS, directory, "reanalysis-")
                if replica is not None else None)
    agreement = None
    if primary["status"] == "completed" and repeated and repeated["status"] == "completed":
        measured = json.loads(store.read(primary["outputs"]["metrics"]))
        recalculated = json.loads(store.read(repeated["outputs"]["metrics"]))
        details = _compare_metrics(measured, recalculated)
        agreement = dict(metric=METRIC, primary=measured[METRIC], reanalysis=recalculated[METRIC],
                         tolerance=AGREEMENT_TOLERANCE,
                         agrees=all(item["agrees"] for item in details.values()), metrics=details,
                         identical_observations=primary["outputs"]["raw_data"] == repeated["outputs"]["raw_data"])
    if store.events()[-1]["hash"] != graph.snapshot_hash:
        raise ValueError("EpistemeOS store changed during export. Run export again.")
    evidence = dict(stage=stage, episteme_commit=EPISTEME_COMMIT, registration=lock,
                    frozen_manifest_sha256=_digest(frozen_bytes), graph_sha256=_digest(graph_bytes),
                    graph_snapshot=graph.snapshot_hash, graph_revision=graph.revision, primary=primary,
                    reanalysis=repeated, reanalysis_agreement=agreement, scientific_validity="not_assessed")
    _write(directory / "evidence.json", _json_bytes(evidence))
    return evidence


def prepare(root, *, online=False):
    from episteme.execution_locked import freeze_closure, prepare as prepare_closure, project_files
    from episteme.store import Store

    if not root.exists():
        with TemporaryDirectory(prefix="gdm-prepare-") as directory:
            with Store(Path(directory)) as store:
                closure = freeze_closure(store, project_files(PROJECT_ROOT))
                return prepare_closure(store, closure, online=online)
    frozen = _locked(root)
    lock = frozen["stages"]["synthetic"]
    with Store(root / "synthetic", read_only=True) as store:
        return prepare_closure(store, lock["environment"], online=online)


def run_or_export(root, output, stage, *, execute, workers=None, project=PROJECT_ROOT):
    from episteme.store import Store

    frozen = _locked(root, project)
    if workers is not None and workers != frozen["workers"]:
        raise ValueError("Requested worker count differs from the frozen count")
    lock = frozen["stages"][stage]
    with Store(root / stage) as store:
        if execute:
            primary = _advance(store, stage, lock)
            if primary["status"] == "completed":
                _advance(store, stage, lock, primary["run"])
        return _export(store, stage, lock, output, (root / "frozen.json").read_bytes())


def main(argv=None):
    options = argparse.ArgumentParser(description=__doc__)
    sub = options.add_subparsers(dest="operation", required=True)
    for action in ("freeze", "prepare", "run", "export"):
        command = sub.add_parser(action)
        command.add_argument("--root", type=Path, default=PROJECT_ROOT / ".research" / "final")
        command.add_argument("--episteme", type=Path,
                             default=Path(os.environ.get("GDM_EPISTEME", PROJECT_ROOT.parent / "EpistemeOS")))
        if action in {"run", "export"}:
            command.add_argument("--stage", choices=STAGES, required=True)
            command.add_argument("--results", type=Path, default=PROJECT_ROOT / "results")
        if action in {"run", "freeze"}:
            command.add_argument("--workers", type=int, default=1 if action == "freeze" else None)
        if action == "prepare":
            command.add_argument("--online", action="store_true")
    args = options.parse_args(argv)
    _episteme(args.episteme)
    root = args.root.resolve()
    if args.operation == "freeze":
        value = freeze(root, args.workers)
    elif args.operation == "prepare":
        value = prepare(root, online=args.online)
    else:
        value = run_or_export(root, args.results.resolve(), args.stage,
                              execute=args.operation == "run", workers=getattr(args, "workers", None))
    print(_json_bytes(value).decode("utf-8"), end="")
    if args.operation == "run":
        agreement = value["reanalysis_agreement"]
        return 0 if agreement and agreement["agrees"] and agreement["identical_observations"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
