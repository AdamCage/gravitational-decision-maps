"""Register and verify the inertial comparisons using EpistemeOS 0.3.0."""

from argparse import ArgumentParser
from datetime import UTC, datetime
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from gdm.evidence import (
    AGREEMENT_TOLERANCE,
    PRIMARY_OUTPUTS,
    REANALYSIS_OUTPUTS,
    StudySources,
    _advance,
    _compare_metrics,
    _digest,
    _execution_details,
    _find_job,
    _json_bytes,
    _write,
)


ROOT = Path(__file__).resolve().parent.parent
EPISTEME_COMMIT = "98bfdde732a2122796313b62c6d8ecbeb532a92d"
STAGES = ("relativistic_synthetic", "relativistic_public")
BASE_SPEC = "crossing/study.json"
STUDY_SPEC = "relativistic/study.json"


def _use_episteme(path):
    checkout = path.expanduser().resolve()
    git = ["git", "-C", str(checkout)]
    commit = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    changed = subprocess.check_output([*git, "status", "--porcelain", "--", "src"], text=True)
    if commit != EPISTEME_COMMIT or changed:
        raise ValueError(f"Use the EpistemeOS checkout at {EPISTEME_COMMIT} with clean src")
    source = checkout / "src"
    sys.path.insert(0, str(source))
    from episteme import store

    if not Path(store.__file__).resolve().is_relative_to(source):
        raise ValueError("An EpistemeOS package outside the pinned checkout is already imported")


def _source_files():
    paths = [
        *(ROOT / "relativistic").glob("*.py"),
        *(ROOT / "crossing_benchmark_data").glob("*.gz"),
        *(ROOT / "crossing" / name for name in ("__init__.py", "data.py", "flow.py", "public.py")),
        ROOT / BASE_SPEC,
        ROOT / STUDY_SPEC,
        ROOT / "relativistic_worker.py",
    ]
    return {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in paths
            if path.name != "evidence.py"}


def _sources():
    from episteme.execution_locked import project_files

    primary = _source_files()
    check = {"relativistic_reanalysis.py": (ROOT / "relativistic_reanalysis.py").read_bytes(),
             BASE_SPEC: primary[BASE_SPEC], STUDY_SPEC: primary[STUDY_SPEC]}
    return StudySources(primary, check, project_files(ROOT))


def _digests(sources):
    return sources.digests() | {
        "gdm/evidence.py": _digest((ROOT / "gdm/evidence.py").read_bytes()),
        "relativistic/evidence.py": _digest(Path(__file__).read_bytes()),
    }


def _declare(store, sources, stage, workers):
    from episteme.execution_locked import freeze_closure, freeze_source
    from episteme.kernel import Actor, Kernel
    from episteme.planning import Planning

    base_spec = json.loads(sources.primary[BASE_SPEC])
    subset = stage.removeprefix("relativistic_")
    families = base_spec["families" if subset == "synthetic" else "public_families"]
    scope = {"population": ", ".join(families), "stage": subset,
             "inference": "matched, core-only finite-horizon classification"}
    planner = Actor("relativistic-planner", "planner")
    planning, kernel = Planning(store, planner), Kernel(store, planner)
    question = planning.question(
        study_id=f"gdm-{stage}",
        statement="Does relativistic inertia improve finite-horizon classification over hard speed clipping?",
        objective="Estimate paired external balanced accuracy and measured inference latency.",
        scope=scope,
        constraints=[
            "Use the crossing study's split roster, feature scaling, prototypes, bandwidths and horizons.",
            "Compare the same class scores at rest and after three matched inertial flows.",
            "The soft and hard speed limits share the same per-fit value.",
            "External outcomes may not select speed limits or solver settings.",
        ],
        stopping_criteria=["Complete the registered roster and one same-observation reanalysis."],
    )
    hypotheses = [kernel.hypothesis(title, prediction, challenge, scope) for title, prediction, challenge in (
        ("The relativistic speed law helps", "Relativistic inertia improves accuracy over hard clipping.",
         "The matched hard limit equals or exceeds relativistic accuracy."),
        ("Bounded travel explains any gain", "Both bounded flows have similar effects relative to classical inertia.",
         "The two speed laws have meaningfully different effects at the same limit."),
        ("Motion has little practical value", "Integration produces little accuracy gain or increases inference cost.",
         "A dynamic method gives a meaningful accuracy gain with acceptable latency."),
    )]
    set_id = planning.explanation_set(
        question=question,
        hypotheses=hypotheses,
        comparison_plan="Run static, classical, hard-capped and relativistic readouts on each paired external set.",
    )
    worker = freeze_source(store, sources.primary, entry_point="relativistic_worker.py", input_name="study.json")
    reanalysis = freeze_source(
        store, sources.reanalysis, entry_point="relativistic_reanalysis.py", input_name="observations.json",
    )
    closure = freeze_closure(store, sources.closure)
    seeds = base_spec["final_seeds"]
    execution_seed = seeds[0]
    data = store.put_json({"stage": subset, "workers": workers, "phase": "final",
                           "execution_seed": execution_seed})
    protocol = kernel.preregister_for_set(
        explanation_set=set_id,
        design=(f"Roster {seeds}; crossing config SHA-256 {_digest(sources.primary[BASE_SPEC])}; "
                f"inertial addendum SHA-256 {_digest(sources.primary[STUDY_SPEC])}; "
                f"reanalysis source {reanalysis}."),
        metric="primary_gain",
        analysis_plan=("Paired relative balanced accuracy of relativistic inertia minus hard clipping, "
                       "measured as a fraction on the fixed primary setting; report accuracy and "
                       "latency for all four methods on the complete roster."),
        stopping_rule="Complete the frozen roster and report all outcomes, including adverse or negligible contrasts.",
        seeds=[execution_seed], run_limit=2, implementation=worker, environment=closure, data=data,
        replication_tolerance=AGREEMENT_TOLERANCE,
    )
    return dict(question=question, explanation_set=set_id, protocol=protocol, worker=worker,
                reanalysis=reanalysis, environment=closure, data=data, execution_seed=execution_seed)


def _read_frozen(root, sources):
    content = (root / "frozen.json").read_bytes()
    frozen = json.loads(content)
    if frozen["episteme_commit"] != EPISTEME_COMMIT or frozen["files"] != _digests(sources):
        raise ValueError("Source or runner drift since inertial registration")
    return content, frozen


def freeze(root, workers):
    from episteme.store import Store

    if workers != 1:
        raise ValueError("The timing comparison requires sequential evaluation (--workers 1)")
    sources = _sources()
    if root.exists():
        return _read_frozen(root, sources)[1]
    root.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="inertia-register-", dir=root.parent) as temp:
        pending = Path(temp) / "store"
        locks = {}
        for stage in STAGES:
            with Store(pending / stage) as store:
                locks[stage] = _declare(store, sources, stage, workers)
        frozen = dict(schema_version=1, frozen_utc=datetime.now(UTC).isoformat(),
                      episteme_commit=EPISTEME_COMMIT, execution_profile="uv_locked_python_v2",
                      workers=workers, files=_digests(sources), stages=locks)
        _write(pending / "frozen.json", _json_bytes(frozen))
        pending.rename(root)
    return frozen


def prepare(root, *, online=False):
    from episteme.execution_locked import freeze_closure, prepare as materialize, project_files
    from episteme.store import Store

    if root.exists():
        _, frozen = _read_frozen(root, _sources())
        with Store(root / STAGES[0], read_only=True) as store:
            return materialize(store, frozen["stages"][STAGES[0]]["environment"], online=online)
    with TemporaryDirectory(prefix="inertia-prepare-") as dirname:
        with Store(Path(dirname)) as store:
            closure = freeze_closure(store, project_files(ROOT))
            return materialize(store, closure, online=online)


def _export(store, stage, lock, destination, frozen_content):
    from episteme.execution import job_state
    from episteme.graph import ResearchGraph
    from episteme.reporting import export_store

    directory = destination / stage
    graph = ResearchGraph.from_store(store)
    graph_content = graph.to_json().encode("utf-8")
    exports = {name: Path(path).read_bytes() for name, path in export_store(store).items()}
    if json.loads(exports["review-bundle.json"])["summary"]["last_event_hash"] != graph.snapshot_hash:
        raise ValueError("EpistemeOS store changed during export. Run export again.")
    _write(directory / "graph.json", graph_content)
    for name, data in exports.items():
        _write(directory / name, data)
    _write(destination / "frozen.json", frozen_content)
    state = job_state(store, _find_job(store, lock["protocol"], None))
    primary = _execution_details(store, state, PRIMARY_OUTPUTS, directory, "")
    reanalysis_job = _find_job(store, lock["protocol"], state["run"])
    reanalysis = (_execution_details(store, job_state(store, reanalysis_job), REANALYSIS_OUTPUTS,
                                     directory, "reanalysis-") if reanalysis_job else None)
    agreement = None
    if primary["status"] == "completed" and reanalysis and reanalysis["status"] == "completed":
        measured = json.loads(store.read(primary["outputs"]["metrics"]))
        checked = json.loads(store.read(reanalysis["outputs"]["metrics"]))
        details = _compare_metrics(measured, checked)
        agreement = dict(metric="primary_gain", primary=measured["primary_gain"],
                         reanalysis=checked["primary_gain"], tolerance=AGREEMENT_TOLERANCE,
                         agrees=all(item["agrees"] for item in details.values()), metrics=details,
                         identical_observations=primary["outputs"]["raw_data"] == reanalysis["outputs"]["raw_data"])
    if store.events()[-1]["hash"] != graph.snapshot_hash:
        raise ValueError("EpistemeOS store changed during export. Run export again.")
    evidence = dict(stage=stage, episteme_commit=EPISTEME_COMMIT, registration=lock,
                    frozen_manifest_sha256=_digest(frozen_content), graph_sha256=_digest(graph_content),
                    graph_snapshot=graph.snapshot_hash, graph_revision=graph.revision, primary=primary,
                    reanalysis=reanalysis, reanalysis_agreement=agreement, scientific_validity="not_assessed")
    _write(directory / "evidence.json", _json_bytes(evidence))
    return evidence


def execute(root, stage, destination):
    from episteme.store import Store

    content, frozen = _read_frozen(root, _sources())
    lock = frozen["stages"][stage]
    with Store(root / stage) as store:
        primary = _advance(store, stage, lock)
        if primary["status"] == "completed":
            _advance(store, stage, lock, primary["run"])
        evidence = _export(store, stage, lock, destination, content)
    agreement = evidence["reanalysis_agreement"]
    if not agreement or not agreement["agrees"] or not agreement["identical_observations"]:
        raise RuntimeError(f"Incomplete or disagreeing results preserved in {destination / stage}")
    return evidence


def bundle(destination, exported=None):
    from relativistic_reanalysis import recompute

    base = destination if exported is None else exported
    content = (base / "frozen.json").read_bytes()
    frozen = json.loads(content)
    if frozen["episteme_commit"] != EPISTEME_COMMIT or set(frozen["stages"]) != set(STAGES):
        raise ValueError("Unexpected runner or stage roster in bundle")
    for name, digest in frozen["files"].items():
        if _digest((ROOT / name).read_bytes()) != digest:
            raise ValueError(f"Frozen source has changed: {name}")
    base_spec = json.loads((ROOT / BASE_SPEC).read_bytes())
    spec = json.loads((ROOT / STUDY_SPEC).read_bytes())
    for stage, registration in frozen["stages"].items():
        directory = base / stage
        files = {name: (directory / name).read_bytes() for name in (
            "metrics.json", "summary.json", "reanalysis-metrics.json", "evidence.json", "graph.json",
        )}
        raw = ((directory / "raw.json").read_bytes() if exported
               else gzip.decompress((directory / "raw.json.gz").read_bytes()))
        evidence = json.loads(files["evidence.json"])
        if (evidence["stage"] != stage or evidence["episteme_commit"] != EPISTEME_COMMIT
                or evidence["registration"] != registration or evidence["frozen_manifest_sha256"] != _digest(content)):
            raise ValueError(f"Registration does not match evidence for {stage}")
        first, second = evidence["primary"], evidence["reanalysis"]
        if first["status"] != "completed" or second is None or second["status"] != "completed":
            raise ValueError(f"An execution did not complete for {stage}")
        one, two = first["outputs"], second["outputs"]
        checks = {
            "metrics.json": one["metrics"], "summary.json": one["summary"],
            "reanalysis-metrics.json": two["metrics"], "graph.json": evidence["graph_sha256"],
        }
        if _digest(raw) != one["raw_data"] or one["raw_data"] != two["raw_data"] or any(
                _digest(files[name]) != digest for name, digest in checks.items()):
            raise ValueError(f"Artifact hashes do not match for {stage}")
        observations = json.loads(raw)
        if observations["phase"] != "final" or "relativistic_" + observations["stage"] != stage:
            raise ValueError(f"Unexpected raw observation scope for {stage}")
        metrics = recompute(observations, base_spec, spec)
        for name in ("metrics.json", "reanalysis-metrics.json"):
            details = _compare_metrics(metrics, json.loads(files[name]))
            if not all(item["agrees"] for item in details.values()):
                raise ValueError(f"Independent numerical check failed for {stage} / {name}")
        if exported:
            files["raw.json.gz"] = gzip.compress(raw, mtime=0)
            for name, data in files.items():
                _write(destination / stage / name, data)
        print(f"verified {stage}: {len(observations['units'])} task-seed units", flush=True)
    if exported:
        _write(destination / "frozen.json", content)


def main():
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "freeze", "run", "bundle", "verify"):
        command = sub.add_parser(action)
        command.add_argument("--root", type=Path, default=ROOT / ".research" / "relativistic-final")
        command.add_argument("--export", type=Path, default=ROOT / ".research" / "relativistic-export")
        command.add_argument("--results", type=Path, default=ROOT / "relativistic_results")
        if action in ("prepare", "freeze", "run"):
            command.add_argument("--episteme", type=Path,
                                 default=Path(os.environ.get("GDM_EPISTEME", ROOT.parent / "EpistemeOS")))
        if action == "prepare":
            command.add_argument("--online", action="store_true")
        if action == "freeze":
            command.add_argument("--workers", type=int, default=1)
        if action == "run":
            command.add_argument("--stage", choices=STAGES, required=True)
    args = parser.parse_args()
    if args.action in ("bundle", "verify"):
        bundle(args.results.resolve(), args.export.resolve() if args.action == "bundle" else None)
        return
    _use_episteme(args.episteme)
    root = args.root.resolve()
    if args.action == "prepare":
        result = prepare(root, online=args.online)
    elif args.action == "freeze":
        result = freeze(root, args.workers)
    else:
        result = execute(root, args.stage, args.export.resolve())
    print(_json_bytes(result).decode("utf-8"), end="", flush=True)


if __name__ == "__main__":
    main()
