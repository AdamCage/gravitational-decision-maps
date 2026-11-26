"""Run the research-video RouterAI adapter with punctuation-only timing recovery."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parent


def load_skill(script):
    spec = importlib.util.spec_from_file_location("research_video_provider", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def punctuation_adapter(original):
    def post(stage, data, output, headers, duration):
        if stage != "asr":
            return original(stage, data, output, headers, duration)
        payload = json.loads(data)
        words = payload.get("words")
        ignored = []
        if isinstance(words, list) and duration is not None:
            for index, word in enumerate(words):
                start, end, token = word["start"], word["end"], word["word"]
                if (isinstance(token, str) and token.strip()
                        and not any(character.isalnum() for character in token)
                        and math.isfinite(start) and start == end and 0 <= start <= duration + .25):
                    ignored.append(index)
        if not ignored:
            return original(stage, data, output, headers, duration)

        # Preserve the provider bytes. Run the unmodified skill's strict timing
        # validation on the lexical timeline before promoting the saved response.
        partial = output.with_name(output.stem + ".part" + output.suffix)
        partial.parent.mkdir(parents=True, exist_ok=True)
        if not partial.is_file() or partial.read_bytes() != data:
            partial.write_bytes(data)
        checked = output.with_name(output.stem + ".validated" + output.suffix)
        ignored_indexes = set(ignored)
        lexical = {**payload, "words": [w for i, w in enumerate(words) if i not in ignored_indexes]}
        extra = original(stage, json.dumps(lexical).encode("utf-8"), checked, headers, duration)
        checked.unlink()
        partial.replace(output)
        return {**extra, "zero_duration_punctuation_indices": ignored}

    return post


def reconcile(skill, project, sid):
    cfg = skill.load(project)
    pending = project / "record" / "pending" / "asr" / f"{sid}.json"
    output = project / "record" / "routerai" / "transcripts" / f"{sid}.json"
    partial = output.with_name(f"{sid}.part.json")
    receipt_path = output.with_suffix(".receipt.json")
    completed = pending.with_suffix(".completed.json")
    if any(path.exists() for path in (completed, output, receipt_path)):
        raise ValueError(f"Cannot reconcile ASR {sid}; an output already exists")
    attempt = skill.read(pending)
    audio, duration = skill.asr_length(project, cfg, sid, None)
    source_hash = skill.sha(audio)
    body = skill.transcription_body(cfg, {"sha256": source_hash})
    budget = skill.budget_paths(project, cfg)
    with budget.ledger.open(encoding="utf-8") as ledger:
        charges = [json.loads(row) for row in ledger if row.strip()]
    if (attempt["scene_id"] != sid or body != attempt["body"]
            or skill.request_hash(body) != attempt["request_sha256"]
            or not any(row.get("attempt_id") == attempt["attempt_id"]
                       and row.get("cost_rub") == attempt["cost_rub"] for row in charges)):
        raise ValueError(f"ASR {sid} does not match the arrival and the booked request")
    extra = skill.posted_output("asr", partial.read_bytes(), output, attempt["headers"], duration)
    receipt = {
        "scene_id": sid, "attempt_id": attempt["attempt_id"], "request": body,
        "request_sha256": attempt["request_sha256"], "response_sha256": skill.sha(output),
        "headers": attempt["headers"], "catalog_estimate_rub": attempt["cost_rub"],
        "billing_status": "catalog_estimate_not_provider_invoice", "source_sha256": source_hash,
        "model": cfg["transcription"]["model"], "reconciled_from_saved_response": True, **extra,
    }
    if budget.currency == "USD":
        receipt["catalog_estimate_usd"] = attempt["cost_usd"]
    skill.write(receipt_path, receipt)
    skill.append(project / "record" / "events.jsonl", {"event": "asr_completed", **receipt})
    pending.rename(completed)
    print(f"{sid}: saved ASR response recovered without a second request")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("asr", "reconcile"))
    parser.add_argument("--skill", type=Path, required=True, help="Updated research-video scripts/video.py")
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--scene")
    args = parser.parse_args()
    if args.command == "reconcile" and not args.scene:
        parser.error("reconcile requires --scene")
    skill = load_skill(args.skill.resolve())
    skill.posted_output = punctuation_adapter(skill.posted_output)
    if args.command == "reconcile":
        reconcile(skill, args.project.resolve(), args.scene)
    else:
        skill.generate_audio(args.project.resolve(), "asr", args.scene)


if __name__ == "__main__":
    main()
