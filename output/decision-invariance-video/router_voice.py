"""Build a frame timeline and Russian captions from archived RouterAI takes."""

from argparse import ArgumentParser
from difflib import SequenceMatcher
from hashlib import sha256
import itertools
import json
import math
from pathlib import Path
import re
import subprocess
import textwrap
import unicodedata


PROJECT = Path(__file__).resolve().parent
AUDIO_RATE = 48_000
MINIMUM_ANCHOR_FRACTION = 0.8
ASR_TAIL_TOLERANCE = 0.30
SENTENCE_END = re.compile(r'[.!?…][»”"’)]*$')


def digest(path):
    with path.open("rb") as stream:
        state = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            state.update(block)
        return state.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_text(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def request_digest(request):
    packed = json.dumps(request, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return sha256(packed.encode("utf-8")).hexdigest()


def decoded_samples(path):
    data = subprocess.check_output(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path), "-map", "0:a:0",
         "-ar", str(AUDIO_RATE), "-ac", "1", "-f", "s16le", "pipe:1"],
    )
    if not data or len(data) % 2:
        raise ValueError(f"Empty decoded chapter {path.name}")
    return len(data) // 2


def read_take(project, scene, speech):
    sid = scene["id"]
    audio_relative = Path("media") / "voice" / f"{sid}.mp3"
    audio = project / audio_relative
    audio_hash = digest(audio)
    tts_path = project / "record" / "voice" / f"{sid}.json"
    tts = read(tts_path)
    request = tts["request"]
    if tts["request_sha256"] != request_digest(request):
        raise ValueError(f"{sid}: TTS request digest mismatch")
    if (
        tts.get("audio_sha256", tts.get("response_sha256")) != audio_hash
        or tts.get("response_sha256", audio_hash) != audio_hash
        or request.get("input") != scene.get("speech_text", scene["narration"])
        or request.get("model") != speech["model"]
        or request.get("voice") != speech["voice"]
    ):
        raise ValueError(f"{sid}: TTS receipt does not bind the selected text, model, voice and audio")
    transcript_dir = project / "record" / "routerai" / "transcripts"
    transcript = transcript_dir / f"{sid}.json"
    transcript_receipt = transcript_dir / f"{sid}.receipt.json"
    asr = read(transcript_receipt)
    transcript_hash = digest(transcript)
    if asr.get("source_sha256") != audio_hash or asr.get("response_sha256") != transcript_hash:
        raise ValueError(f"{sid}: ASR receipt does not bind the transcript and selected MP3")
    samples = decoded_samples(audio)
    return {
        "path": audio_relative.as_posix(),
        "sha256": audio_hash,
        "decoded_samples_48000": samples,
        "duration": samples / AUDIO_RATE,
        "tts_model": request["model"],
        "tts_voice": request["voice"],
        "tts_request_sha256": tts["request_sha256"],
        "tts_receipt_sha256": digest(tts_path),
        "asr_model": asr.get("model"),
        "asr_response_sha256": transcript_hash,
        "asr_receipt_sha256": digest(transcript_receipt),
    }, read(transcript)["words"]


def token(word):
    return "".join(ch for ch in word.casefold().replace("ё", "е") if ch.isalnum())


def verified_words(raw, duration, sid):
    words = []
    last_start = 0.0
    for item in raw:
        text, start, end = (item[name] for name in ("word", "start", "end"))
        if not (
            isinstance(text, str) and all(type(value) in (int, float) and math.isfinite(value) for value in (start, end))
            and last_start <= start <= end <= duration + ASR_TAIL_TOLERANCE
        ):
            raise ValueError(f"{sid}: invalid or unordered ASR word timestamp")
        if start == end:
            if text.strip() and all(
                char.isspace() or unicodedata.category(char).startswith("P") for char in text
            ):
                continue
            raise ValueError(f"{sid}: zero-duration ASR token outside standalone punctuation")
        last_start = start
        if token(text) and start < min(end, duration):
            words.append((token(text), start, min(end, duration)))
    for index, (left, right) in enumerate(itertools.pairwise(words)):
        words[index] = (left[0], left[1], min(left[2], right[1]))
    return [item for item in words if item[1] < item[2]]


def word_timing(authored, recognised, duration):
    audible = [(index, token(word)) for index, word in enumerate(authored) if token(word)]
    matcher = SequenceMatcher(None, [word for _, word in audible], [item[0] for item in recognised], autojunk=False)
    anchors = {}
    for block in matcher.get_matching_blocks():
        for delta in range(block.size):
            index = audible[block.a + delta][0]
            _, start, end = recognised[block.b + delta]
            anchors[index] = (start, end)
    times = dict(anchors)
    bounds = [(-1, (0.0, 0.0)), *sorted(anchors.items()), (len(authored), (duration, duration))]
    for (left, (_, past)), (right, (future, _)) in itertools.pairwise(bounds):
        weights = [len(token(word)) for word in authored[left + 1:right]]
        weight = sum(weights)
        accumulated = 0
        for index, size in zip(range(left + 1, right), weights, strict=True):
            begin = past + (future - past) * accumulated / weight if weight else past
            accumulated += size
            end = past + (future - past) * accumulated / weight if weight else future
            times[index] = (begin, end)
    return times, anchors, len(anchors) / max(1, len(audible))


def cut_points(segments, words, times, anchors, duration):
    indexes = list(itertools.accumulate([0] + [len(segment.split()) for segment in segments]))
    cuts, checks = [0.0], []
    for boundary in indexes[1:-1]:
        index = next(i for i in range(boundary, len(words)) if token(words[i]))
        previous = next(i for i in range(boundary - 1, -1, -1) if token(words[i]))
        left = max((i for i in anchors if i < boundary), default=-1)
        right = min((i for i in anchors if i >= boundary), default=len(words))
        previous_end, first_start = times[previous][1], times[index][0]
        # Move the picture during the inter-sentence pause if present. The
        # narration remains one uninterrupted MP3 decode through the cut.
        point = min(first_start, max(previous_end, first_start - 0.10))
        cuts.append(point)
        checks.append({
            "opening": " ".join(words[index:index + 4]),
            "offset_seconds": round(point, 6),
            "direct_first_word_anchor": index in anchors,
            "preceding_anchor_distance_words": boundary - left - 1,
            "following_anchor_distance_words": right - boundary,
            "opening_time_seconds": round(first_start, 6),
            "preceding_word_end_seconds": round(previous_end, 6),
        })
    cuts.append(duration)
    if any(last - first < 0.25 for first, last in itertools.pairwise(cuts)):
        raise ValueError("Invalid editorial beat order")
    return indexes, cuts, checks


def phrase_lines(words):
    return textwrap.wrap(" ".join(words), width=41, break_long_words=False, break_on_hyphens=False)


def chapter_captions(words, indexes, times, audio_start, duration):
    captions = []
    for first, boundary in itertools.pairwise(indexes):
        while first < boundary:
            stop = first + 1
            while stop < boundary and not SENTENCE_END.search(words[stop - 1]):
                candidate = words[first:stop + 1]
                if len(" ".join(candidate)) > 72 or len(phrase_lines(candidate)) > 2:
                    break
                stop += 1
            if stop < boundary and not any(token(word) for word in words[stop:boundary]):
                stop = boundary
            begin, end = times[first][0], times[stop - 1][1]
            captions.append({
                "text": "\n".join(phrase_lines(words[first:stop])),
                "startMs": round((audio_start + max(0, begin - 0.06)) * 1000),
                "endMs": round((audio_start + min(duration, end + 0.17)) * 1000),
                "timestampMs": None,
                "confidence": None,
            })
            first = stop
    for left, right in itertools.pairwise(captions):
        left["endMs"] = min(left["endMs"], right["startMs"])
    if any(caption["endMs"] <= caption["startMs"] for caption in captions):
        raise ValueError("Nonpositive caption interval; inspect unanchored script span")
    return captions


def timestamp(ms, separator):
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, decimals = divmod(rest, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02}{separator}{decimals:03}"


def write_captions(out, captions):
    out.mkdir(parents=True, exist_ok=True)
    for extension, separator in (("srt", ","), ("vtt", ".")):
        parts = [] if extension == "srt" else ["WEBVTT\n"]
        for index, caption in enumerate(captions, 1):
            begin = timestamp(caption["startMs"], separator)
            end = timestamp(caption["endMs"], separator)
            number = f"{index}\n" if extension == "srt" else ""
            parts.append(f"{number}{begin} --> {end}\n{caption['text']}\n")
        atomic_text(out / f"subtitles.ru.{extension}", "\n".join(parts))


def prepare(project, inspect_scene):
    episode = project / "episode.json"
    cfg = read(episode)
    fps = cfg["fps"]
    lead = round(cfg["scene_lead_seconds"] * fps)
    tail = round(cfg["scene_tail_seconds"] * fps)
    scenes = cfg["scenes"]
    if inspect_scene:
        scenes = [scene for scene in scenes if scene["id"] == inspect_scene]
    if not scenes:
        raise ValueError("No narration scenes selected")
    timeline = {field: cfg[field] for field in ("fps", "width", "height", "title", "subtitle_language", "mastering")}
    timeline["narration_mode"] = "continuous_routerai_mp3"
    timeline["scenes"] = []
    captions, report, voice_beats = [], [], {}
    position = 0
    for order, scene in enumerate(scenes):
        sid = scene["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", sid) or sid in voice_beats:
            raise ValueError(f"Unsafe or repeated chapter ID {sid}")
        segments = scene["segments"]
        if len(segments) != 4 or " ".join(segments) != scene["narration"]:
            raise ValueError(f"{sid}: four consecutive editorial segments must reproduce the narration")
        audio, raw_words = read_take(project, scene, cfg["speech"])
        duration = audio["duration"]
        words = scene["narration"].split()
        recognised = verified_words(raw_words, duration, sid)
        times, anchors, fraction = word_timing(words, recognised, duration)
        if fraction < 0.55:
            raise ValueError(f"{sid}: only {fraction:.1%} of authored words matched the provider transcript")
        indexes, cuts, beat_checks = cut_points(segments, words, times, anchors, duration)
        voice_beats[sid] = [round(cut, 6) for cut in cuts[:-1]]
        audio_start = (position + lead) / fps
        chapter_subtitles = chapter_captions(words, indexes, times, audio_start, duration)
        captions.extend(chapter_subtitles)
        cues = [{
            "path": audio["path"],
            "sha256": audio["sha256"],
            "text": text,
            "start": audio_start + begin,
            "duration": end - begin,
        } for text, begin, end in zip(segments, cuts[:-1], cuts[1:], strict=True)]
        number_frames = lead + math.ceil(duration * fps) + tail + (fps if order == len(scenes) - 1 else 0)
        timeline["scenes"].append({
            "id": sid, "chapter": scene["chapter"], "start": position / fps,
            "frames": number_frames, "duration": number_frames / fps, "cues": cues,
            "audio": {**audio, "start": audio_start},
        })
        position += number_frames
        suspect = fraction < MINIMUM_ANCHOR_FRACTION or any(
            max(beat["preceding_anchor_distance_words"], beat["following_anchor_distance_words"]) > 3
            for beat in beat_checks
        )
        max_caption_speed = max(
            len(item["text"].replace("\n", " ")) / ((item["endMs"] - item["startMs"]) / 1000)
            for item in chapter_subtitles
        )
        report.append({
            **audio, "scene": sid, "anchor_fraction": round(fraction, 6), "caption_count": len(chapter_subtitles),
            "max_caption_characters_per_second": round(max_caption_speed, 2),
            "beat_boundaries": beat_checks,
            "status": "needs_review" if suspect else "timing_anchors_available",
        })
        print(f"{sid}: {duration:.2f}s, anchors {fraction:.1%}, four editorial beats", flush=True)
    if inspect_scene:
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return
    if any(left["endMs"] > right["startMs"] for left, right in itertools.pairwise(captions)):
        raise ValueError("Overlapping captions across chapters")
    authored = " ".join(" ".join(scene["narration"].split()) for scene in scenes)
    subtitle = " ".join(" ".join(item["text"].split()) for item in captions)
    if subtitle != authored:
        raise ValueError("Subtitle text differs from the authored narration")
    timeline.update({
        "frames": position,
        "duration": position / fps,
        "episode_sha256": digest(episode),
        "evidence_sha256": digest(project / "evidence.json"),
    })
    record = project / "record"
    out = project / "output"
    write(record / "voice-beats.json", voice_beats)
    write(record / "captions.json", captions)
    write(record / "alignment.json", {
        "method": "archived RouterAI word-anchor matching with interpolation for unanchored authored words",
        "beat_review_anchor_fraction": MINIMUM_ANCHOR_FRACTION,
        "scenes": report,
    })
    write_captions(out, captions)
    write(out / "timeline.json", timeline)
    print(f"{timeline['duration']:.2f}s, {position} frames, {len(captions)} captions", flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--inspect-scene", help="Audit one archived chapter and print results without writing")
    args = parser.parse_args()
    prepare(args.project.resolve(), args.inspect_scene)


if __name__ == "__main__":
    main()
