"""Prepare the official CIFAR-100 splits and the study's paired blur arrays."""

from argparse import ArgumentParser
from contextlib import contextmanager
from hashlib import file_digest, md5
from http.client import HTTPException
import json
from pathlib import Path
import pickle
from shutil import copyfileobj
import tarfile
from tempfile import NamedTemporaryFile
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
from scipy import __version__ as scipy_version
from scipy.ndimage import convolve1d


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / ".research" / "cifar100"
ARCHIVE_NAME = "cifar-100-python.tar.gz"
AUTHOR_PAGE = "https://cave.cs.toronto.edu/kriz/cifar.html"
AUTHOR_URL = "https://cave.cs.toronto.edu/kriz/cifar-100-python.tar.gz"
MIRROR_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/cifar-100-python.tar.gz"
EXPECTED_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
EXPECTED_SHA256 = "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7"
MEMBER_MD5 = {
    "train": "16019d7e3df5f24257cddd939b257f8d",
    "test": "f0ef6b0ae62326f3e7ffdfab6717acfc",
    "meta": "7973b15100ade9c7d40fb424638fde48",
}
VALIDATION_SEED = 20261127
VALIDATION_COUNT = 5000
SIGMA = 1.25
COORDINATES = np.arange(-2, 3, dtype=np.float64)
KERNEL = np.exp(-0.5 * np.square(COORDINATES / SIGMA))
KERNEL /= KERNEL.sum()


@contextmanager
def staged_path(destination):
    with NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as stream:
        staged = Path(stream.name)
    try:
        yield staged
        staged.replace(destination)
    finally:
        staged.unlink(missing_ok=True)


def digest(path, algorithm="sha256"):
    with path.open("rb") as stream:
        return file_digest(stream, algorithm).hexdigest()


def verify_archive(archive):
    sha = digest(archive)
    legacy = digest(archive, lambda: md5(usedforsecurity=False))
    if sha != EXPECTED_SHA256 or legacy != EXPECTED_MD5:
        raise ValueError(f"CIFAR-100 archive hash mismatch at {archive}: MD5={legacy} SHA-256={sha}")


def acquire_archive(output, supplied, supplied_origin):
    if supplied is not None:
        verify_archive(supplied)
        return supplied, supplied_origin

    archive = output / ARCHIVE_NAME
    if archive.is_file():
        verify_archive(archive)
        existing_record = output / "metadata.json"
        if existing_record.is_file():
            existing = json.loads(existing_record.read_text(encoding="utf-8"))
            if existing.get("archive_sha256") == EXPECTED_SHA256:
                return archive, existing.get("acquisition_url")
        return archive, None

    failures = []
    for url in (AUTHOR_URL, MIRROR_URL):
        try:
            with staged_path(archive) as staged:
                with urlopen(url, timeout=40) as response, staged.open("wb") as stream:
                    copyfileobj(response, stream, length=1 << 20)
                verify_archive(staged)
            return archive, url
        except (HTTPException, URLError, TimeoutError, ValueError) as error:
            failures.append(f"{url}: {error}")
    raise RuntimeError("Unable to acquire the verified CIFAR-100 archive\n" + "\n".join(failures))


def read_member(archive, name):
    member = archive.getmember(f"cifar-100-python/{name}")
    if not member.isfile():
        raise ValueError(f"Expected archive member {member.name} to be a regular file")
    with archive.extractfile(member) as stream:
        payload = stream.read()
    if md5(payload, usedforsecurity=False).hexdigest() != MEMBER_MD5[name]:
        raise ValueError(f"Archive member hash mismatch for {member.name}")
    # The CLI verifies both published archive hashes before reading any pickle.
    return pickle.loads(payload, encoding="latin1")


def write_array(path, array):
    with staged_path(path) as staged, staged.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    return {"shape": list(array.shape), "dtype": array.dtype.name, "sha256": digest(path)}


def write_metadata(path, metadata):
    with staged_path(path) as staged:
        staged.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")


def export_splits(archive, output, source_url):
    files = {}
    with tarfile.open(archive, mode="r:gz") as dataset:
        fine_names = read_member(dataset, "meta")["fine_label_names"]
        if len(fine_names) != 100 or len(set(fine_names)) != 100:
            raise ValueError("Expected exactly 100 unique fine label names")
        for split, count in (("train", 50000), ("test", 10000)):
            source = read_member(dataset, split)
            pixels = source["data"]
            if pixels.shape != (count, 3072) or pixels.dtype != np.uint8:
                raise ValueError(f"Invalid image data in {split}")
            labels = np.asarray(source["fine_labels"], dtype=np.int64)
            if labels.shape != (count,) or labels.min() != 0 or labels.max() != 99:
                raise ValueError(f"Invalid fine labels in {split}")
            if not np.all(np.bincount(labels, minlength=100) == count // 100):
                raise ValueError(f"Invalid class counts in {split}")
            images = np.ascontiguousarray(pixels.reshape(count, 3, 32, 32).transpose(0, 2, 3, 1))
            for prefix, array in (("x", images), ("y", labels)):
                name = f"{prefix}_{split}.npy"
                files[name] = write_array(output / name, array)
    metadata = {
        "dataset": "CIFAR-100",
        "authors": ["Alex Krizhevsky", "Vinod Nair", "Geoffrey Hinton"],
        "source_page": AUTHOR_PAGE,
        "source_url": AUTHOR_URL,
        "mirror_url": MIRROR_URL,
        "acquisition_url": source_url,
        "archive_bytes": archive.stat().st_size,
        "archive_md5": EXPECTED_MD5,
        "archive_sha256": EXPECTED_SHA256,
        "archive_md5_reference": AUTHOR_PAGE,
        "archive_sha256_reference": "https://github.com/keras-team/keras/blob/master/keras/src/datasets/cifar100.py",
        "archive_member_md5": MEMBER_MD5,
        "official_split": True,
        "ordering": "Unchanged source row order within each split",
        "pixels": "Unchanged uint8 values; unpacked from channel-plane to NHWC RGB layout",
        "label_space": "Fine labels 0–99 in original integer encoding",
        "fine_label_names": fine_names,
        "split_rows": {"train": 50000, "test": 10000},
        "images_per_class": {"train": 500, "test": 100},
        "array_files": files,
    }
    write_metadata(output / "metadata.json", metadata)
    return metadata


def blur_split(path, images, rows=None, batch_size=256):
    size = len(rows) if rows is not None else len(images)
    stored = np.empty((size, 32, 32, 3), dtype=np.uint8)
    for first in range(0, size, batch_size):
        last = min(first + batch_size, size)
        samples = images[first:last] if rows is None else images[rows[first:last]]
        vertical = convolve1d(samples, KERNEL, axis=1, output=np.float64, mode="mirror")
        filtered = convolve1d(vertical, KERNEL, axis=2, mode="mirror")
        stored[first:last] = np.rint(filtered).astype(np.uint8)
    return write_array(path, stored)


def export_blur(output, source):
    train_path, test_path = (output / f"x_{part}.npy" for part in ("train", "test"))
    train = np.load(train_path, allow_pickle=False, mmap_mode="r")
    test = np.load(test_path, allow_pickle=False, mmap_mode="r")
    validation_rows = np.random.default_rng(VALIDATION_SEED).permutation(len(train))[:VALIDATION_COUNT]
    index_file = "val_indices.npy"
    index_record = write_array(output / index_file, validation_rows)
    outputs = {
        "x_val_blur.npy": blur_split(output / "x_val_blur.npy", train, rows=validation_rows),
        "x_test_blur.npy": blur_split(output / "x_test_blur.npy", test),
    }
    metadata = {
        "name": "Local paired Gaussian blur on CIFAR-100",
        "source_archive_sha256": source["archive_sha256"],
        "source_array_sha256": {p.name: source["array_files"][p.name]["sha256"] for p in (train_path, test_path)},
        "kernel_shape": [5, 5],
        "sigma_pixels": [SIGMA, SIGMA],
        "separable_kernel": KERNEL.tolist(),
        "normalized_kernel": "exp(-0.5 * (coordinates / sigma) ** 2) normalized to sum to one",
        "boundary": "SciPy mirror (whole-sample symmetric, duplicating no edge pixel; OpenCV REFLECT_101 semantics)",
        "filter_order": ["height", "width"],
        "computation": "Binary64 scalar channels, then np.rint ties-to-even output to uint8",
        "validation": {
            "source_split": "Official train",
            "selection": "First 5000 indices in np.random.default_rng(20261127).permutation(50000) order",
            "indices_file": index_file,
            "indices_sha256": index_record["sha256"],
        },
        "test": {"source_split": "Official test", "selection": "All rows unchanged source order"},
        "numpy": np.__version__,
        "scipy": scipy_version,
        "outputs": outputs,
    }
    write_metadata(output / "gaussian_blur.json", metadata)
    return metadata


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Array and provenance directory")
    parser.add_argument("--archive", type=Path, help="Use a local copy of the official archive")
    parser.add_argument("--archive-origin", help="Record the known source URL of the supplied archive")
    arguments = parser.parse_args()
    if arguments.archive_origin is not None and arguments.archive is None:
        parser.error("--archive-origin requires --archive")
    arguments.output.mkdir(parents=True, exist_ok=True)
    archive, origin = acquire_archive(arguments.output, arguments.archive, arguments.archive_origin)
    source = export_splits(archive, arguments.output, origin)
    blur = export_blur(arguments.output, source)
    print(json.dumps({"archive_sha256": source["archive_sha256"],
                      "arrays": {**source["array_files"], **blur["outputs"]}, "output": str(arguments.output)}))


if __name__ == "__main__":
    main()
