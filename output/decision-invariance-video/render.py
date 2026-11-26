"""Build the Russian research film from measured speech cues and animated scenes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

from PIL import Image


PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[1]
OUT = PROJECT / "output"
RECORD = PROJECT / "record"
TIMELINE = OUT / "timeline.json"
SCORE = OUT / "mix.wav"
MASTER = OUT / "video.mp4"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def digest(path):
    with path.open("rb") as stream:
        return sha256_stream(stream)


def sha256_stream(stream):
    state = sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        state.update(block)
    return state.hexdigest()


def run(*args, **options):
    return subprocess.run(list(map(str, args)), check=True, **options)


def ffmpeg(*args, **options):
    return run("ffmpeg", "-nostdin", "-hide_banner", "-y", *args, **options)


def prepare():
    from router_voice import prepare as prepare_router_voice

    prepare_router_voice(PROJECT, None)


def read_timeline():
    plan = read(TIMELINE)
    for name in ("episode", "evidence"):
        if plan[f"{name}_sha256"] != digest(PROJECT / f"{name}.json"):
            raise ValueError(f"Timeline no longer matches {name}.json; prepare it again")
    return plan


def scene_frame(scene, local, size):
    from data_visuals import render_data_scene
    from visuals import render_scene

    dur = scene["duration"]
    clock = [0] + [cue["start"] - scene["start"] for cue in scene["cues"][1:]] + [dur]
    for beat in range(len(clock) - 1):
        if local <= clock[beat + 1]:
            t = dur * (beat + (local - clock[beat]) / (clock[beat + 1] - clock[beat])) / (len(clock) - 1)
            break
    else:
        t = dur
    renderer = render_data_scene if 7 <= int(scene["id"]) <= 10 else render_scene
    frame = renderer(scene["id"], t, dur, size)
    if frame.size != size:
        raise ValueError(f"Unexpected frame dimensions {frame.size}")
    fade = None
    if local < 0.23:
        fade = min(1, max(0, local / 0.23))
    elif dur - local < 0.28:
        fade = min(1, max(0, (dur - local) / 0.28))
    rgb = frame.convert("RGB")
    return Image.blend(Image.new("RGB", size, "#08131F"), rgb, fade) if fade is not None else rgb


def previews(plan, ids):
    output = OUT / "preview"
    output.mkdir(parents=True, exist_ok=True)
    size = plan["width"], plan["height"]
    for scene in plan["scenes"]:
        if ids and scene["id"] not in ids:
            continue
        for index, cue in enumerate(scene["cues"]):
            local = cue["start"] - scene["start"] + cue["duration"] * .65
            image = scene_frame(scene, local, size)
            image.thumbnail((960, 540), Image.Resampling.LANCZOS)
            image.save(output / f"{scene['id']}-{index}.jpg", quality=91)


def encoder(mode):
    if mode in ("auto", "gpu"):
        probe = subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
             "-i", "color=black:s=1920x1080:d=0.08", "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True,
        )
        if probe.returncode == 0:
            return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-b:v", "0", "-cq", "18"]
        if mode == "gpu":
            raise RuntimeError(probe.stderr.decode(errors="replace"))
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-threads", "3"]


def clip(scene, plan, video_encoder):
    fps = plan["fps"]
    size = plan["width"], plan["height"]
    directory = OUT / "clips"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{scene['id']}.mp4"
    temporary = target.with_name(f"{scene['id']}.partial.mp4")
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-filter_threads", "2",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", f"{size[0]}x{size[1]}",
            "-framerate", str(fps), "-i", "pipe:0", "-an",
            "-vf", "scale=out_color_matrix=bt709,format=yuv420p", *video_encoder,
            "-profile:v", "high", "-g", str(2 * fps), "-bf", "2", "-color_range", "tv",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", str(temporary)]
    print(f"{scene['id']} | {scene['duration']:.2f} s | {scene['chapter']}", flush=True)
    with target.with_suffix(".ffmpeg.log").open("wb") as errors:
        child = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=errors)
        try:
            for number in range(scene["frames"]):
                child.stdin.write(scene_frame(scene, number / fps, size).tobytes())
            child.stdin.close()
            if child.wait():
                raise RuntimeError(f"Encoding failed for {scene['id']}; see {errors.name}")
        except BaseException:
            if child.poll() is None:
                child.kill()
            child.wait()
            raise
    temporary.replace(target)
    print(f"{scene['id']} | encoded {target.stat().st_size / 1e6:.1f} MB", flush=True)
    return target


def escaped(text):
    return re.sub(r"([\\;#=])", r"\\\1", text).replace("\n", "\\\n")


def chapter_metadata(plan):
    metadata = [";FFMETADATA1", f"title={escaped(plan['title'])}"]
    listing = []
    for scene in plan["scenes"]:
        start, end = scene["start"], scene["start"] + scene["duration"]
        metadata += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={round(start * 1000)}",
                     f"END={round(end * 1000)}", f"title={escaped(scene['chapter'])}"]
        minutes, seconds = divmod(int(start), 60)
        listing.append(f"{minutes:02d}:{seconds:02d} {scene['chapter']}")
    path = OUT / "chapters.ffmetadata"
    path.write_text("\n".join(metadata) + "\n", encoding="utf-8", newline="\n")
    (OUT / "chapters.txt").write_text("\n".join(listing) + "\n", encoding="utf-8", newline="\n")
    description = (PROJECT / "youtube-description.md").read_text(encoding="utf-8").strip()
    (OUT / "youtube.txt").write_text(
        f"{plan['title']}\n\n{description}\n\nГлавы\n" + "\n".join(listing) + "\n",
        encoding="utf-8", newline="\n",
    )
    return path


def loudness(path, goal):
    filt = f"loudnorm=I={goal['lufs']}:TP={goal['true_peak']}:LRA={goal['lra']}:print_format=json"
    proc = ffmpeg("-i", path, "-vn", "-af", filt, "-f", "null", "-", capture_output=True,
                  encoding="utf-8", errors="replace")
    matches = re.findall(r"\{\s*\"input_i\"[\s\S]*?\}", proc.stderr)
    if not matches:
        raise RuntimeError("FFmpeg did not return loudness measurements")
    return json.loads(matches[-1])


def master(plan):
    directory = OUT / "clips"
    clips = [directory / f"{scene['id']}.mp4" for scene in plan["scenes"]]
    if not all(path.is_file() for path in clips):
        raise FileNotFoundError("Complete the visual render before mastering")
    concat = directory / "concat.txt"
    concat.write_text("".join(f"file '{path.name}'\n" for path in clips), encoding="ascii")
    meta = chapter_metadata(plan)
    goal = plan["mastering"]
    measured = loudness(SCORE, goal)
    volume = f"loudnorm=I={goal['lufs']}:TP={goal['true_peak']}:LRA={goal['lra']}"
    keys = {"I": "input_i", "TP": "input_tp", "LRA": "input_lra", "thresh": "input_thresh"}
    volume += "".join(f":measured_{key}={measured[value]}" for key, value in keys.items())
    volume += f":offset={measured['target_offset']}:linear=true:print_format=summary"
    ffmpeg("-loglevel", "error", "-f", "concat", "-safe", "1", "-i", concat, "-i", SCORE,
           "-i", OUT / "subtitles.ru.srt", "-i", meta,
           "-map", "0:v:0", "-map", "1:a:0", "-map", "2:s:0", "-map_metadata", "3",
           "-map_chapters", "3", "-c:v", "copy", "-af", volume, "-c:a", "aac",
           "-b:a", "256k", "-ar", "48000", "-ac", "2", "-c:s", "mov_text",
           "-metadata:s:s:0", "language=rus", "-metadata:s:a:0", "language=rus",
           "-disposition:s:0", "0", "-movflags", "+faststart", MASTER)
    print(MASTER, flush=True)


def audit(plan):
    probe = run("ffprobe", "-v", "error", "-show_format", "-show_streams", "-show_chapters",
                "-of", "json", MASTER, capture_output=True, text=True, encoding="utf-8")
    details = json.loads(probe.stdout)
    video, track, subtitle = [next(st for st in details["streams"] if st["codec_type"] == kind)
                              for kind in ("video", "audio", "subtitle")]
    if (video["codec_name"], video["pix_fmt"], video["width"], video["height"],
            video["avg_frame_rate"], int(video["nb_frames"])) != (
                "h264", "yuv420p", plan["width"], plan["height"], f"{plan['fps']}/1", plan["frames"]):
        raise ValueError("Master video layout does not match the planned timeline")
    if (track["codec_name"], int(track["sample_rate"]), track["channels"], subtitle["tags"]["language"]) != (
            "aac", 48000, 2, plan["subtitle_language"]):
        raise ValueError("Audio or subtitle streams do not match their requested format")
    if len(details["chapters"]) != len(plan["scenes"]):
        raise ValueError("Unexpected chapter count")
    if abs(float(details["format"]["duration"]) - plan["duration"]) > 0.11:
        raise ValueError("Audio and visual timelines have diverged")
    ffmpeg("-v", "error", "-xerror", "-i", MASTER, "-map", "0:v:0", "-map", "0:a:0",
           "-f", "null", "-")
    level = loudness(MASTER, plan["mastering"])
    if abs(float(level["input_i"]) - plan["mastering"]["lufs"]) > 1.0 or float(level["input_tp"]) > -1.0:
        raise ValueError(f"Encoded audio outside mastering tolerance: {level}")
    checks = {"video_sha256": digest(MASTER), "timeline_sha256": digest(TIMELINE),
              "episode_sha256": plan["episode_sha256"], "evidence_sha256": plan["evidence_sha256"],
              "duration_seconds": plan["duration"], "frames": plan["frames"],
              "decoded_entire_video_and_audio": True, "video_codec": video["codec_name"],
              "video_size": [video["width"], video["height"]], "fps": plan["fps"],
              "audio_codec": track["codec_name"], "audio_hz": int(track["sample_rate"]),
              "chapters": len(details["chapters"]), "subtitles": "rus", "loudness": level,
              "ffmpeg": run("ffmpeg", "-version", capture_output=True, text=True).stdout.splitlines()[0]}
    write(RECORD / "master.json", checks)
    print(json.dumps(checks, ensure_ascii=False, indent=2))


def bundle_visuals():
    files = {
        "03": "shared-equilibrium.png",
        "04": "facet-balance.png",
        "08": "certificate-coverage.png",
        "09": "interval-checks.png",
        "10": "runtime-comparison.png",
    }
    destination = OUT / "visual-stills"
    destination.mkdir(parents=True, exist_ok=True)
    times = {}
    for scene in read_timeline()["scenes"]:
        if scene["id"] in files:
            name = files[scene["id"]]
            time = round(scene["start"] + .91 * scene["duration"], 3)
            ffmpeg("-loglevel", "error", "-ss", time, "-i", MASTER,
                   "-frames:v", "1", "-threads", "1", destination / name)
            times[name] = {"scene": scene["id"], "seconds": time}
    write(destination / "from-video.json", times)
    target = OUT / "visuals.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for name in [*files.values(), "from-video.json"]:
            archive.write(destination / name, "schemes-and-graphs/" + name)
        for file in sorted((PROJECT / "media/visuals").glob("*.png")):
            archive.write(file, "illustrations/" + file.name)
        archive.write(RECORD / "artwork.json", "illustrations/provenance.json")
        archive.write(PROJECT / "evidence.json", "evidence.json")
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise IOError("Visual archive failed its CRC test")
    return target


def bundle():
    extras = [path for path in PROJECT.iterdir() if path.is_file() and not path.name.startswith(".")]
    extras += sorted((PROJECT / "media" / "voice").glob("*.mp3"))
    extras += sorted((PROJECT / "media" / "visuals").glob("*.png"))
    extras += [
        path for folder in ("record", "record/voice", "record/routerai/transcripts")
        for path in (PROJECT / folder).glob("*") if path.is_file()
    ]
    extras += [OUT / filename for filename in ("subtitles.ru.srt", "subtitles.ru.vtt", "chapters.txt", "timeline.json",
                                               "youtube.txt", "thumbnail.jpg")]
    references = {ROOT / source["path"] for source in read(PROJECT / "evidence.json")["sources"]}
    extras += sorted(references)
    papers = {
        ROOT / "output/pdf/letter.pdf": OUT / "decision-invariance-letter.pdf",
        ROOT / "output/pdf/manuscript.pdf": OUT / "decision-invariance-technical.pdf",
    }
    for source, output in papers.items():
        shutil.copyfile(source, output)
    extras += list(papers)
    target = OUT / "source-project.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(extras)):
            archive.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise IOError("Source archive failed its CRC test")
    files = [MASTER, target, OUT / "subtitles.ru.srt", OUT / "subtitles.ru.vtt", OUT / "youtube.txt",
             OUT / "thumbnail.jpg", OUT / "chapters.txt", *papers.values(), bundle_visuals()]
    (OUT / "SHA256SUMS").write_text("".join(f"{digest(path)}  {path.name}\n" for path in files),
                                     encoding="ascii", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preview", "clips", "master", "audit", "bundle"))
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--encoder", choices=("auto", "gpu", "cpu"), default="auto")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
        return
    plan = read_timeline()
    if args.command == "preview":
        previews(plan, args.only)
    elif args.command == "clips":
        settings = encoder(args.encoder)
        chosen = [scene for scene in plan["scenes"] if not args.only or scene["id"] in args.only]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda scene: clip(scene, plan, settings), chosen))
    else:
        {"master": lambda: master(plan), "audit": lambda: audit(plan), "bundle": bundle}[args.command]()


if __name__ == "__main__":
    main()
