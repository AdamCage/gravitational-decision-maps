"""Read paired low- and high-range acceleration windows from PAMAP2.

Each published row contains the two channels at all three body sites and
one shared timestamp. Both ranges record the same activity but the wider
range is itself a measurement. No interpolation or synthetic clipping is
performed here.
"""

from dataclasses import dataclass
from hashlib import file_digest
from http.client import HTTPException
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np


DOI = "https://doi.org/10.24432/C5NW2H"
SOURCE_URL = "https://archive.ics.uci.edu/static/public/231/pamap2%2Bphysical%2Bactivity%2Bmonitoring.zip"
MIRROR_URL = (
    "https://huggingface.co/datasets/AI-MED-AGH/Recruitment-Task-2B/resolve/main/"
    "original/pamap2%2Bphysical%2Bactivity%2Bmonitoring.zip"
)
SOURCE_SHA256 = "76b3580bd5a804121f507717cb498c66a312ef5c774f4a78ac470e6558add352"
CACHE = Path(__file__).resolve().parents[1] / ".research" / "pamap2" / "source.zip"
PROTOCOL_SUBJECTS = tuple(range(101, 109))
SAMPLE_RATE = 100
STANDARD_GRAVITY = 9.80665
NEAR_RANGE_OBSERVED = 0.95 * 6 * STANDARD_GRAVITY
NEAR_RANGE_REFERENCE = 0.95 * 16 * STANDARD_GRAVITY

# Zero-based file columns, grouped by hand, chest and ankle.
REFERENCE_COLUMNS = (4, 5, 6, 21, 22, 23, 38, 39, 40)
OBSERVED_COLUMNS = (7, 8, 9, 24, 25, 26, 41, 42, 43)
AXES = tuple(f"{site}_{axis}" for site in ("hand", "chest", "ankle") for axis in "xyz")
SUMMARIES = ("mean", "sd", "rms", "max_abs", "mean_abs_step")
FEATURE_NAMES = tuple(f"{summary}_{axis}" for summary in SUMMARIES for axis in AXES)


@dataclass(frozen=True)
class PairedWindows:
    observed: np.ndarray
    reference: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    start_times: np.ndarray
    peak_observed: np.ndarray
    peak_reference: np.ndarray
    metadata: dict


def _verify_source(archive):
    with archive.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != SOURCE_SHA256:
            raise ValueError(f"PAMAP2 archive SHA-256 mismatch: {archive}")


def _verified_archive(archive):
    if archive is not None:
        archive = Path(archive)
        _verify_source(archive)
        return archive
    if CACHE.is_file():
        _verify_source(CACHE)
        return CACHE
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=CACHE.parent, suffix=".zip.part", delete=False) as staging:
        staged = Path(staging.name)
    try:
        for url in (SOURCE_URL, MIRROR_URL):
            try:
                with urlopen(url, timeout=45) as response, staged.open("wb") as stream:
                    copyfileobj(response, stream, length=1 << 20)
                _verify_source(staged)
                break
            except (HTTPException, URLError, TimeoutError, ValueError):
                if url == MIRROR_URL:
                    raise
        staged.replace(CACHE)
    finally:
        staged.unlink(missing_ok=True)
    return CACHE


def read_subjects(archive=None, *, subjects=PROTOCOL_SUBJECTS):
    """Yield (subject, timestamps, labels, observed, reference) from the protocol."""
    subjects = tuple(subjects)
    if not subjects or len(set(subjects)) != len(subjects) or set(subjects) - set(range(101, 110)):
        raise ValueError("Use a nonempty roster of distinct protocol subject IDs from 101 to 109")
    archive = _verified_archive(archive)
    columns = (0, 1, *OBSERVED_COLUMNS, *REFERENCE_COLUMNS)
    with ZipFile(archive) as outer, outer.open("PAMAP2_Dataset.zip") as inner, ZipFile(inner) as data:
        for subject in subjects:
            member = f"PAMAP2_Dataset/Protocol/subject{subject}.dat"
            with data.open(member) as stream:
                rows = np.loadtxt(stream, usecols=columns, ndmin=2)
            activity = rows[:, 1]
            if not np.isfinite(activity).all() or not np.equal(activity, np.floor(activity)).all():
                raise ValueError(f"Noninteger activity label in {member}")
            yield subject, rows[:, 0], activity.astype(int), rows[:, 2:11], rows[:, 11:20]


def _window_starts(time, labels, observed, reference, size, maximum_gap):
    eligible = (labels > 0) & np.isfinite(time)
    eligible &= np.isfinite(observed).all(axis=1) & np.isfinite(reference).all(axis=1)
    delta = np.diff(time)
    connected = eligible[:-1] & eligible[1:] & (labels[:-1] == labels[1:])
    breaks = np.flatnonzero(~connected | (delta <= 0) | (delta > maximum_gap)) + 1
    limits = np.r_[0, breaks, len(time)]
    runs = [(left, right) for left, right in zip(limits[:-1], limits[1:], strict=True)
            if right - left >= size and eligible[left]]
    starts = np.concatenate([np.arange(left, right - size + 1, size) for left, right in runs]) if runs else np.empty(0, dtype=int)
    audit = {
        "file_rows": len(time),
        "named_activity_rows": int(np.count_nonzero(labels > 0)),
        "eligible_paired_rows": int(eligible.sum()),
        "gaps_between_eligible_same_label_rows": int(np.count_nonzero(connected & (delta > maximum_gap))),
        "nonpositive_steps_between_eligible_same_label_rows": int(np.count_nonzero(connected & (delta <= 0))),
        "windows": len(starts),
    }
    return starts, audit


def _summarize(windows):
    magnitude = np.abs(windows).max(axis=1)
    features = np.concatenate((
        windows.mean(axis=1),
        windows.std(axis=1),
        np.sqrt(np.square(windows).mean(axis=1)),
        magnitude,
        np.abs(np.diff(windows, axis=1)).mean(axis=1),
    ), axis=1)
    return features, magnitude.max(axis=1)


def load_pamap2_pairs(archive=None, *, window_samples=200, maximum_gap=0.025, subjects=PROTOCOL_SUBJECTS):
    """Summarize consecutive nonoverlapping paired windows before modeling.

    Split the stream at every activity transition, missing paired sample,
    nonincreasing timestamp or timestamp gap exceeding ``maximum_gap``.
    Use complete windows aligned to the start of each uninterrupted run.
    Subject 109 and the optional activity files are outside the default
    cohort.
    """
    if not isinstance(window_samples, int) or window_samples < 2:
        raise ValueError("window_samples must be an integer of at least two")
    if not np.isfinite(maximum_gap) or maximum_gap <= 0:
        raise ValueError("maximum_gap must be positive and finite")
    observed_features, reference_features = [], []
    labels, ids, times, observed_peaks, reference_peaks = [], [], [], [], []
    audits = {}
    for subject, time, activity, observed, reference in read_subjects(archive, subjects=subjects):
        starts, audit = _window_starts(time, activity, observed, reference, window_samples, maximum_gap)
        indices = starts[:, None] + np.arange(window_samples)
        observed_summary, observed_peak = _summarize(observed[indices])
        reference_summary, reference_peak = _summarize(reference[indices])
        observed_features.append(observed_summary)
        reference_features.append(reference_summary)
        observed_peaks.append(observed_peak)
        reference_peaks.append(reference_peak)
        labels.append(activity[starts])
        ids.append(np.full(len(starts), subject, dtype=int))
        times.append(time[starts])
        unique, counts = np.unique(activity[starts], return_counts=True)
        audit["windows_by_activity"] = dict(zip(map(str, unique), map(int, counts), strict=True))
        audit["observed_near_range_windows"] = int(np.count_nonzero(observed_peak >= NEAR_RANGE_OBSERVED))
        audit["reference_near_range_windows"] = int(np.count_nonzero(reference_peak >= NEAR_RANGE_REFERENCE))
        audits[str(subject)] = audit
    if not any(len(part) for part in labels):
        raise ValueError("No complete labeled paired windows under this windowing rule")
    return PairedWindows(
        observed=np.concatenate(observed_features),
        reference=np.concatenate(reference_features),
        labels=np.concatenate(labels),
        subjects=np.concatenate(ids),
        start_times=np.concatenate(times),
        peak_observed=np.concatenate(observed_peaks),
        peak_reference=np.concatenate(reference_peaks),
        metadata={
            "source_url": SOURCE_URL,
            "source_mirror_url": MIRROR_URL,
            "source_doi": DOI,
            "source_license": "CC BY 4.0",
            "source_sha256": SOURCE_SHA256,
            "sample_rate_hz_nominal": SAMPLE_RATE,
            "window_samples": window_samples,
            "maximum_gap_seconds": maximum_gap,
            "feature_names": FEATURE_NAMES,
            "acceleration_unit": "m/s^2",
            "observed_range_g": 6,
            "reference_range_g": 16,
            "observed_columns_zero_based": OBSERVED_COLUMNS,
            "reference_columns_zero_based": REFERENCE_COLUMNS,
            "observed_near_range_m_per_s2": NEAR_RANGE_OBSERVED,
            "reference_near_range_m_per_s2": NEAR_RANGE_REFERENCE,
            "subjects": audits,
        },
    )
