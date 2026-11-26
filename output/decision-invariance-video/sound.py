"""Mix archived narration with the episode's original synth-pop score."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from math import ceil, isfinite, sqrt
from pathlib import Path
import subprocess
import wave

import numpy as np


PROJECT = Path(__file__).resolve().parent
RATE = 48_000
TEMPO = 112
SCORE_SEED = 261126
VOICE_RMS = 0.118
PROGRESSION = (
    (40, (52, 55, 59, 66)),
    (36, (52, 55, 59, 62)),
    (43, (50, 55, 59, 64)),
    (38, (50, 54, 57, 64)),
    (45, (52, 55, 59, 60)),
    (36, (48, 55, 59, 62)),
    (40, (52, 55, 59, 66)),
    (38, (50, 54, 57, 69)),
)


def digest(path):
    with path.open("rb") as stream:
        hasher = sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    temp.replace(path)


def rhythm(seconds):
    return np.arange(round(seconds * RATE), dtype=np.float32) / RATE


def gate(time, length, attack, release):
    return np.clip(np.minimum(time / attack, (length - time) / release), 0, 1)


def frequency(note):
    return 440.0 * 2 ** ((note - 69) / 12)


def place(stereo, offset, mono, gain, pan=0.0):
    first = round(offset * RATE)
    available = min(len(mono), len(stereo) - first)
    if available <= 0:
        return
    left = sqrt(1 - pan)
    right = sqrt(1 + pan)
    mono = mono[:available]
    stereo[first:first + available, 0] += mono * np.float32(gain * left)
    stereo[first:first + available, 1] += mono * np.float32(gain * right)


def instrument(note, length, sound):
    t = rhythm(length)
    phase = 2 * np.pi * frequency(note) * t
    if sound == "bass":
        wave_data = np.sin(phase) + 0.20 * np.sin(phase * 2) + 0.10 * np.sin(phase * 3)
        return wave_data * gate(t, length, 0.011, 0.07) * np.exp(-2 * t)
    if sound == "pad":
        wave_data = 0.65 * np.sin(phase) + 0.30 * np.sin(phase * 1.003) + 0.10 * np.sin(phase * 2)
        return wave_data * gate(t, length, 0.23, 0.49)
    wave_data = np.sin(phase) + 0.24 * np.sin(phase * 2) + 0.11 * np.sin(phase * 3)
    return wave_data * gate(t, length, 0.008, 0.10) * np.exp(-6 * t)


def percussion():
    rng = np.random.default_rng(SCORE_SEED)
    t = rhythm(0.31)
    phase = 2 * np.pi * (44 * t + 94 * 0.024 * (1 - np.exp(-t / 0.024)))
    kick = 0.9 * np.sin(phase) * np.exp(-14 * t) * gate(t, 0.31, 0.003, 0.045)
    t = rhythm(0.18)
    white = rng.normal(size=len(t)).astype(np.float32)
    high = np.diff(white, prepend=0) * 0.2
    snare = (0.65 * high + 0.23 * np.sin(2 * np.pi * 169 * t)) * np.exp(-27 * t)
    snare *= gate(t, 0.18, 0.004, 0.04)
    t = rhythm(0.10)
    white = rng.normal(size=len(t)).astype(np.float32)
    hat = 0.25 * np.diff(white, prepend=0) * np.exp(-39 * t) * gate(t, 0.10, 0.005, 0.025)
    return kick, snare, hat


def score(samples, scenes):
    """Original arrangement with a dynamic rhythm section and scene stings."""
    bed = np.zeros((samples, 2), dtype=np.float32)
    beat = 60 / TEMPO
    bar = beat * 4
    kick, snare, hat = percussion()
    for index in range(ceil(samples / RATE / bar)):
        origin = index * bar
        bass, chord = PROGRESSION[index % len(PROGRESSION)]
        phase = index % 16
        quiet = phase in (8, 9) and index > 0
        for k, note in enumerate(chord):
            place(bed, origin, instrument(note, bar + 0.14, "pad"), 0.012, (-0.56, 0.42, -0.26, 0.67)[k])
        if index > 0 and not quiet:
            for step in (0, 2, 2.75) if phase % 4 == 3 else (0, 2):
                place(bed, origin + beat * step, kick, 0.092)
            for step in (1, 3):
                place(bed, origin + beat * step, snare, 0.070, 0.12)
            for step in range(8):
                place(bed, origin + beat * step / 2, hat, 0.048 if step % 2 else 0.023, (-1) ** step * 0.32)
            for step, interval in ((0, 0), (1.5, 0), (2.5, 12), (3.25, 7)):
                place(bed, origin + beat * step, instrument(bass + interval, beat * 0.72, "bass"), 0.054)
        else:
            place(bed, origin, instrument(bass, 1.2, "bass"), 0.034)
        if phase in (0, 1, 4, 5, 8, 9, 12, 13, 15):
            arpeggio = (2, 0, 3, 1) if phase < 8 else (1, 3, 0, 2)
            for k, (step, degree) in enumerate(zip((0.45, 1.25, 2, 3.15), arpeggio, strict=True)):
                place(bed, origin + beat * step, instrument(chord[degree] + 12, 0.36, "pluck"),
                      0.029 if k != 2 else 0.020, (-1) ** k * 0.42)
    for k, scene in enumerate(scenes[1:], start=1):
        note = (71, 74, 76, 69)[k % 4]
        start = float(scene["start"])
        place(bed, start, instrument(note, 0.30, "pluck"), 0.043, -0.21)
        place(bed, start + 0.085, instrument(note + 7, 0.23, "pluck"), 0.027, 0.32)
    return bed


def audio_file(project, source):
    path = (project / source).resolve()
    if not path.is_relative_to(project) or not path.is_file():
        raise ValueError(f"Missing narration audio in the project: {source}")
    return path


def read_voice(path):
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(path), "-map", "0:a:0",
         "-af", "highpass=f=72,lowpass=f=9500", "-ac", "1", "-ar", str(RATE), "-f", "f32le", "-"],
        check=True, capture_output=True,
    )
    signal = np.frombuffer(result.stdout, dtype="<f4").copy()
    if not len(signal) or not np.isfinite(signal).all():
        raise ValueError(f"Empty or invalid WAV: {path.name}")
    active = signal[np.abs(signal) > 0.018]
    if not len(active):
        raise ValueError(f"Narration is too quiet to normalize: {path.name}")
    peak = np.abs(signal).max()
    rms = np.sqrt(np.mean(active**2))
    signal *= min(VOICE_RMS / rms, 0.73 / peak, 3.5)
    fade = min(round(RATE * 0.007), len(signal) // 2)
    if fade:
        window = np.linspace(0, 1, fade, dtype=np.float32)
        signal[:fade] *= window
        signal[-fade:] *= window[::-1]
    return signal


def duck(bed, start, end):
    front = max(0, start - round(RATE * 0.11))
    back = min(len(bed), end + round(RATE * 0.23))
    gain = 0.29
    if front < start:
        ramp = np.linspace(1, gain, start - front, dtype=np.float32)
        bed[front:start] *= ramp[:, None]
    bed[start:end] *= gain
    if end < back:
        ramp = np.linspace(gain, 1, back - end, dtype=np.float32)
        bed[end:back] *= ramp[:, None]


def speech_windows(signal):
    """Merge short voice gaps so the music returns smoothly in real pauses."""
    block = RATE // 50
    full_blocks = len(signal) // block
    if not full_blocks:
        return [(0, len(signal))]
    levels = np.sqrt(np.mean(signal[:full_blocks * block].reshape(-1, block)**2, axis=1))
    windows = []
    for index in np.flatnonzero(levels > 0.006):
        start = int(index) * block
        end = min(len(signal), start + block)
        if windows and start - windows[-1][1] < 0.46 * RATE:
            windows[-1][1] = end
        else:
            windows.append([start, end])
    return windows or [(0, len(signal))]


def apply_score_fades(bed):
    roll_in = min(round(RATE * 0.35), len(bed) // 2)
    roll_out = min(round(RATE * 2.35), len(bed) // 2)
    if roll_in:
        bed[:roll_in] *= np.linspace(0, 1, roll_in, dtype=np.float32)[:, None]
    if roll_out:
        bed[-roll_out:] *= np.linspace(1, 0, roll_out, dtype=np.float32)[:, None]


def export_wav(path, stereo):
    peak = max(float(stereo.max()), -float(stereo.min()))
    if peak > 0.89:
        stereo *= 0.89 / peak
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.partial{path.suffix}")
    with wave.open(str(temporary), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        for start in range(0, len(stereo), RATE * 3):
            chunk = np.rint(stereo[start:start + RATE * 3] * 32767.0).astype("<i2")
            stream.writeframesraw(chunk.tobytes())
    temporary.replace(path)
    return min(peak, 0.89)


def render_mix(project, timeline, output):
    spec = read_json(timeline)
    total = float(spec["duration"])
    if not isfinite(total) or total <= 0:
        raise ValueError("Mix duration must be finite and positive")
    samples = round(total * RATE)
    bed = score(samples, spec["scenes"])
    apply_score_fades(bed)
    recordings = []
    sources = [clip for scene in spec["scenes"] for clip in (
        [scene["audio"]] if "audio" in scene else scene["cues"]
    )]
    for cue in sources:
        path = audio_file(project, cue["path"])
        if "sha256" in cue and digest(path) != cue["sha256"]:
            raise ValueError(f"Voice file changed after timeline preparation: {cue['path']}")
        signal = read_voice(path)
        start = round(float(cue["start"]) * RATE)
        end = start + len(signal)
        if start < 0 or end > samples:
            raise ValueError(f"Narration lies outside the timeline: {cue['path']}")
        if abs(len(signal) / RATE - float(cue["duration"])) > 0.035:
            raise ValueError(f"Timeline duration does not match this narration: {cue['path']}")
        recordings.append((start, end, signal))
    if not recordings:
        raise ValueError("No speech cues in the timeline")
    recordings.sort(key=lambda item: item[0])
    for earlier, later in zip(recordings, recordings[1:]):
        if earlier[1] > later[0]:
            raise ValueError(f"Overlapping narration at {later[0] / RATE:.2f}s")
    duck_intervals = []
    for offset, _, signal in recordings:
        for first, last in speech_windows(signal):
            start, end = offset + first, offset + last
            if duck_intervals and start - duck_intervals[-1][1] < 0.34 * RATE:
                duck_intervals[-1][1] = end
            else:
                duck_intervals.append([start, end])
    for start, end in duck_intervals:
        duck(bed, start, end)
    for start, end, signal in recordings:
        bed[start:end] += signal[:, None]
    peak = export_wav(output, bed)
    write_json(project / "record" / "sound.json", {
        "score": "Original procedural synth-pop, composed in sound.py",
        "tempo_bpm": TEMPO,
        "seed": SCORE_SEED,
        "sample_rate": RATE,
        "channels": 2,
        "duration": samples / RATE,
        "narration_cues": len(recordings),
        "editorial_beats": sum(len(scene["cues"]) for scene in spec["scenes"]),
        "narration_mode": spec.get("narration_mode", "separate_cues"),
        "timeline_sha256": digest(timeline),
        "mix_sha256": digest(output),
        "sample_peak": round(peak, 6),
    })
    print(f"{output} | {samples / RATE:.3f}s | {len(recordings)} narration cues | peak {peak:.3f}", flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("command", choices=("mix",))
    parser.add_argument("--timeline", type=Path, default=Path("output/timeline.json"))
    parser.add_argument("--output", type=Path, default=Path("output/mix.wav"))
    args = parser.parse_args()
    project = args.project.resolve()
    render_mix(project, project / args.timeline, project / args.output)


if __name__ == "__main__":
    main()
