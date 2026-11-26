"""Train a validation-selected CIFAR-100 neural baseline and export affine-head inputs.

Preferred run uses CUDA and learns a compact residual network at native size.
The CPU fallback uses frozen official ImageNet ResNet-18 features at 64 px.
Install PyTorch and NumPy in a separate environment, then use
python -m tools.cifar100_neural --help from the repository root.
"""

from argparse import ArgumentParser
from hashlib import sha256
import json
from math import cos, pi
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SPLIT_SEED = 20261127
MODEL_SEED = 93117
WEIGHT_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
CIFAR_CLASS_COUNT = 100


def convolution(input_channels, output_channels, kernel=3, stride=1):
    return nn.Conv2d(input_channels, output_channels, kernel, stride=stride,
                     padding=kernel // 2, bias=False)


def conv_norm(input_channels, output_channels):
    return nn.Sequential(convolution(input_channels, output_channels), nn.BatchNorm2d(output_channels))


class CifarStage(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.project = conv_norm(input_channels, output_channels)
        self.body = nn.Sequential(
            conv_norm(output_channels, output_channels), nn.GELU(),
            conv_norm(output_channels, output_channels),
        )

    def forward(self, batch):
        base = F.gelu(self.project(F.max_pool2d(batch, 2)))
        return F.gelu(base + self.body(base))


class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(conv_norm(3, 64), nn.GELU())
        self.stages = nn.Sequential(CifarStage(64, 96), CifarStage(96, 192), CifarStage(192, 384))
        self.head = nn.Linear(384, CIFAR_CLASS_COUNT)

    def features(self, batch):
        return self.stages(self.stem(batch)).amax(dim=(2, 3))

    def forward(self, batch):
        return self.head(self.features(batch))


class ResNetBlock(nn.Module):
    """ResNet-18 BasicBlock with the official weight bundle's module names."""

    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__()
        self.conv1 = convolution(input_channels, output_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = convolution(output_channels, output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.downsample = None
        if stride != 1:
            self.downsample = nn.Sequential(convolution(input_channels, output_channels, kernel=1, stride=stride),
                                            nn.BatchNorm2d(output_channels))

    def forward(self, batch):
        skip = batch if self.downsample is None else self.downsample(batch)
        hidden = F.relu(self.bn1(self.conv1(batch)))
        return F.relu(self.bn2(self.conv2(hidden)) + skip)


class FrozenResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = convolution(3, 64, kernel=7, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = nn.Sequential(ResNetBlock(64, 64), ResNetBlock(64, 64))
        self.layer2 = nn.Sequential(ResNetBlock(64, 128, 2), ResNetBlock(128, 128))
        self.layer3 = nn.Sequential(ResNetBlock(128, 256, 2), ResNetBlock(256, 256))
        self.layer4 = nn.Sequential(ResNetBlock(256, 512, 2), ResNetBlock(512, 512))
        self.fc = nn.Linear(512, 1000)

    def features(self, batch):
        base = F.max_pool2d(F.relu(self.bn1(self.conv1(batch))), kernel_size=3, stride=2, padding=1)
        return self.layer4(self.layer3(self.layer2(self.layer1(base)))).mean(dim=(2, 3))


def atomic_json(path, value):
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def save_array(folder, name, array):
    pending = folder / f"{name}.pending.npy"
    np.save(pending, array, allow_pickle=False)
    pending.replace(folder / f"{name}.npy")


def load_data(folder):
    arrays = tuple(np.load(folder / f"{name}.npy", mmap_mode="r") for name in (
        "x_train", "y_train", "x_test", "y_test",
    ))
    x_train, y_train, x_test, y_test = arrays
    if (x_train.shape, x_test.shape) != ((50_000, 32, 32, 3), (10_000, 32, 32, 3)):
        raise ValueError("Expected complete official CIFAR-100 train/test NHWC RGB arrays")
    if x_train.dtype != np.uint8 or x_test.dtype != np.uint8:
        raise ValueError("Expected uint8 images")
    for label, count in ((y_train, 50_000), (y_test, 10_000)):
        if label.shape != (count,) or label.dtype != np.int64 or label.min() != 0 or label.max() != 99:
            raise ValueError("Expected int64 CIFAR-100 fine labels")
    return arrays


def augment(images, pixel, generator):
    size = len(images)
    padded = F.pad(images.permute(0, 3, 1, 2).float(), (4, 4, 4, 4), mode="reflect").permute(0, 2, 3, 1)
    offset = torch.randint(9, (2, size, 1), device=images.device, generator=generator)
    flip = torch.rand((size, 1), device=images.device, generator=generator) < 0.5
    rows = pixel + offset[0]
    cols = torch.where(flip, 31 - pixel, pixel) + offset[1]
    return padded[torch.arange(size, device=images.device)[:, None, None], rows[:, :, None], cols[:, None, :]]


def cifar_pixels(images):
    return images.permute(0, 3, 1, 2).float().mul_(4 / 255).sub_(2)


@torch.inference_mode()
def device_validation(model, images, labels, batch_size):
    model.eval()
    accuracy_count, total_loss = 0, 0.0
    for batch, truth in zip(images.split(batch_size), labels.split(batch_size), strict=True):
        logits = model(cifar_pixels(batch))
        accuracy_count += int((logits.argmax(dim=1) == truth).sum())
        total_loss += float(F.cross_entropy(logits, truth, reduction="sum"))
    return accuracy_count / len(labels), total_loss / len(labels)


@torch.inference_mode()
def gpu_embeddings(model, images, batch_size):
    model.eval()
    return torch.cat([model.features(cifar_pixels(batch)).cpu() for batch in images.split(batch_size)]).numpy()


def training_log(path, row):
    print(json.dumps(row), flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")


def fit_cuda(args, x_train, y_train, x_test, fit, valid):
    device = torch.device("cuda")
    name = torch.cuda.get_device_name()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("highest")
    train_images = torch.from_numpy(x_train.copy()).to(device)
    train_labels = torch.from_numpy(np.array(y_train)).to(device)
    fit_ids, valid_ids = torch.from_numpy(fit.copy()).to(device), torch.from_numpy(valid.copy()).to(device)
    validation_images, validation_labels = train_images[valid_ids], train_labels[valid_ids]
    model = CifarNet().to(device=device, memory_format=torch.channels_last)
    optim = torch.optim.SGD(model.parameters(), lr=0.18, momentum=0.9, nesterov=True,
                            weight_decay=5e-4, foreach=True)
    generator = torch.Generator(device=device).manual_seed(MODEL_SEED)
    pixel = torch.arange(32, device=device)[None]
    history = []
    best_accuracy, best_epoch = -1.0, 0
    check_path = args.output / "validation.jsonl"
    check_path.write_text("", encoding="utf-8")
    model_path = args.output / "encoder.pt"
    start = perf_counter()
    print(f"training on {name}; {args.epochs} requested epochs", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        rate = 0.18 * (epoch / 2 if epoch <= 2 else 0.05 + 0.95 * (1 + cos(pi * (epoch - 2) / max(1, args.epochs - 2))) / 2)
        optim.param_groups[0]["lr"] = rate
        order = fit_ids[torch.randperm(len(fit_ids), device=device, generator=generator)]
        loss_sum = torch.zeros((), device=device)
        for indices in order.split(args.batch_size):
            batch = cifar_pixels(augment(train_images[indices], pixel, generator))
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(batch)
                loss = F.cross_entropy(logits, train_labels[indices], label_smoothing=0.05)
            loss.backward()
            optim.step()
            loss_sum += loss.detach() * len(indices)
        accuracy, validation_loss = device_validation(model, validation_images, validation_labels, args.batch_size)
        if accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch
            torch.save(model.state_dict(), model_path)
        elapsed = perf_counter() - start
        row = {"epoch": epoch, "learning_rate": rate, "train_objective": float(loss_sum) / len(fit_ids),
               "validation_accuracy": accuracy, "validation_log_loss": validation_loss,
               "seconds_elapsed": elapsed, "best_epoch": best_epoch}
        history.append(row)
        training_log(check_path, row)
        if elapsed + elapsed / epoch > args.max_seconds:
            break
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    encoding_start = perf_counter()
    z_train = gpu_embeddings(model, train_images, args.batch_size)
    z_test = gpu_embeddings(model, torch.from_numpy(x_test.copy()).to(device), args.batch_size)
    coefficients = model.head.weight.detach().cpu().numpy().copy()
    intercept = model.head.bias.detach().cpu().numpy().copy()
    plan = {
        "architecture": "compact native 32px residual CNN with three max-pool residual stages",
        "stage_channels": [64, 96, 192, 384],
        "device": name, "mode": "native_cuda", "input_side": 32, "initialization": "random",
        "augmentation": "per-image 4px reflection-padded random crop and horizontal flip",
        "input_transform": "4*x/255-2", "precision_training": "bfloat16 autocast",
        "precision_embeddings": "float32 with CUDA TF32 convolution enabled",
        "optimizer": "SGD Nesterov(momentum=.9, weight_decay=.0005), 2 epochs warmup then cosine",
        "loss": "cross-entropy with smoothing .05", "early_time_budget_seconds": args.max_seconds,
        "training_seconds": history[-1]["seconds_elapsed"], "encoding_seconds": perf_counter() - encoding_start,
        "validation_checkpoints": history, "selected_epoch": best_epoch,
        "encoder_sha256": sha256(model_path.read_bytes()).hexdigest(),
    }
    return z_train, z_test, coefficients, intercept, plan


@torch.inference_mode()
def cpu_embeddings(model, images, size):
    average = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    scale = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    rows = []
    for offset in range(0, len(images), size):
        batch = torch.from_numpy(images[offset:offset + size].copy()).permute(0, 3, 1, 2).float().div_(255)
        batch = F.interpolate(batch, size=(64, 64), mode="bilinear", align_corners=False)
        rows.append(model.features((batch - average) / scale))
        if (offset // size + 1) % 24 == 0:
            print(f"encoded {min(offset + size, len(images))}/{len(images)} on CPU", flush=True)
    return torch.cat(rows).numpy()


def fit_cpu(args, x_train, y_train, x_test, fit, valid):
    source = args.imagenet_weights
    digest = sha256(source.read_bytes()).hexdigest()
    if not digest.startswith("f37072fd"):
        raise ValueError("ResNet18 weights do not match the official f37072fd bundle")
    model = FrozenResNet18().eval().to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(source, map_location="cpu", weights_only=True))
    start_encoding = perf_counter()
    z_train = cpu_embeddings(model, x_train, args.batch_size)
    train_encoding_seconds = perf_counter() - start_encoding
    tensor = torch.from_numpy(z_train)
    fit_tensor = tensor[fit]
    center, scale = fit_tensor.mean(dim=0), fit_tensor.std(dim=0).clamp_min(1e-4)
    features = (tensor - center) / scale
    truth = torch.from_numpy(y_train.copy())
    head = nn.Linear(features.shape[1], CIFAR_CLASS_COUNT)
    optim = torch.optim.AdamW(head.parameters(), lr=0.003, weight_decay=0.02)
    generator = torch.Generator().manual_seed(MODEL_SEED)
    check_path = args.output / "validation.jsonl"
    check_path.write_text("", encoding="utf-8")
    history = []
    best_accuracy, best_epoch = -1.0, 0
    start = perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = torch.from_numpy(fit)[torch.randperm(len(fit), generator=generator)]
        for indices in order.split(1024):
            optim.zero_grad(set_to_none=True)
            F.cross_entropy(head(features[indices]), truth[indices]).backward()
            optim.step()
        with torch.no_grad():
            logits = head(features[valid])
            accuracy = float((logits.argmax(dim=1) == truth[valid]).float().mean())
            loss = float(F.cross_entropy(logits, truth[valid]))
        if accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch
            coefficients = (head.weight.detach() / scale).clone()
            intercept = (head.bias.detach() - coefficients @ center).clone()
        elapsed = perf_counter() - start
        row = {"epoch": epoch, "validation_accuracy": accuracy, "validation_log_loss": loss,
               "seconds_elapsed": elapsed, "best_epoch": best_epoch}
        history.append(row)
        training_log(check_path, row)
        if elapsed + elapsed / epoch > args.max_seconds:
            break
    start_encoding = perf_counter()
    z_test = cpu_embeddings(model, x_test, args.batch_size)
    plan = {
        "architecture": "ResNet-18 with a frozen pretrained encoder and fitted 100-class linear head",
        "device": platform.processor(), "mode": "pretrained_cpu", "input_side": 64,
        "initialization": "official ImageNet-1K V1", "weights_url": WEIGHT_URL,
        "encoder_sha256": digest, "python_threads": torch.get_num_threads(),
        "input_transform": "bilinear resize to 64px with ImageNet channel normalization",
        "precision_embeddings": "float32", "optimizer": "AdamW(lr=.003, weight_decay=.02)",
        "loss": "cross-entropy", "early_time_budget_seconds": args.max_seconds,
        "training_seconds": history[-1]["seconds_elapsed"],
        "encoding_seconds": train_encoding_seconds + perf_counter() - start_encoding,
        "validation_checkpoints": history, "selected_epoch": best_epoch,
    }
    return z_train, z_test, coefficients.numpy(), intercept.numpy(), plan


def classification_metrics(logits, labels):
    truth = torch.from_numpy(np.array(labels))
    scores = torch.from_numpy(logits)
    rank = scores.topk(5, dim=1).indices
    return {
        "top1_accuracy": float((rank[:, 0] == truth).double().mean()),
        "top5_accuracy": float((rank == truth[:, None]).any(dim=1).double().mean()),
        "log_loss": float(F.cross_entropy(scores, truth)),
        "n": len(truth),
    }


def export_blur(args, labels, valid, coefficients, intercept):
    started = perf_counter()
    if args.backend == "cuda":
        model = CifarNet().to(device="cuda", memory_format=torch.channels_last)
        model.load_state_dict(torch.load(args.output / "encoder.pt", map_location="cuda", weights_only=True))

        def encode(images):
            return gpu_embeddings(model, torch.from_numpy(images.copy()).to("cuda"), args.batch_size)
    else:
        model = FrozenResNet18().eval().to(memory_format=torch.channels_last)
        model.load_state_dict(torch.load(args.imagenet_weights, map_location="cpu", weights_only=True))

        def encode(images):
            return cpu_embeddings(model, images, args.batch_size)
    metrics = {}
    for kind, truth in zip(("val", "test"), (labels[0][valid], labels[1]), strict=True):
        images = np.load(args.data / f"x_{kind}_blur.npy", mmap_mode="r")
        if images.shape != (len(truth), 32, 32, 3) or images.dtype != np.uint8:
            raise ValueError(f"Invalid {kind} blur image array")
        features = encode(images)
        logits = features @ coefficients.T + intercept
        save_array(args.output, f"e_{kind}_blur", features)
        save_array(args.output, f"{kind}_blur_logits", logits)
        metrics[kind] = classification_metrics(logits, truth)
    metrics["seconds"] = perf_counter() - started
    atomic_json(args.output / "blur_metrics.json", metrics)
    print(json.dumps({"blur": metrics}), flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(".research/cifar100"))
    parser.add_argument("--output", type=Path, default=Path(".research/cifar100/neural"))
    parser.add_argument("--backend", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-seconds", type=float, default=530)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--blur", action="store_true", help="Also encode paired blur after the clean benchmark")
    parser.add_argument("--imagenet-weights", type=Path,
                        default=Path(".research/cifar100/neural/resnet18-f37072fd.pth"))
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.max_seconds, args.threads) <= 0:
        parser.error("Epochs, batch size, time limit, and thread count must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(MODEL_SEED)
    started = perf_counter()
    x_train, y_train, x_test, y_test = load_data(args.data)
    order = np.random.default_rng(SPLIT_SEED).permutation(len(x_train))
    valid, fit = order[:5000], order[5000:]
    runner = fit_cuda if args.backend == "cuda" else fit_cpu
    z_train, z_test, coef, bias, plan = runner(args, x_train, y_train, x_test, fit, valid)
    val_logits = np.asarray(z_train[valid] @ coef.T + bias, dtype=np.float32)
    test_logits = np.asarray(z_test @ coef.T + bias, dtype=np.float32)
    items = {
        "e_train": z_train, "e_test": z_test,
        "classifier_coef": coef, "classifier_intercept": bias,
        "fit_indices": fit, "validation_indices": valid,
        "validation_logits": val_logits, "test_logits": test_logits,
    }
    for name, array in items.items():
        save_array(args.output, name, array)
    np.savez(args.output / "head.pending.npz", coef=coef, intercept=bias, class_ids=np.arange(CIFAR_CLASS_COUNT))
    (args.output / "head.pending.npz").replace(args.output / "head.npz")
    manifest = {
        "dataset": "CIFAR-100 official train and test split, fine classes",
        "split_seed": SPLIT_SEED, "model_seed": MODEL_SEED, "fit_rows": len(fit),
        "selection_rule": "highest validation top-1 accuracy, earliest checkpoint on ties",
        "requested_epochs": args.epochs, "input_batch_size": args.batch_size,
        "validation": classification_metrics(val_logits, y_train[valid]),
        "official_test": classification_metrics(test_logits, y_test),
        "array_shapes": {name: list(value.shape) for name, value in items.items()},
        "array_dtypes": {name: str(value.dtype) for name, value in items.items()},
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(), "pytorch": torch.__version__, "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "run_seconds": perf_counter() - started, **plan,
    }
    atomic_json(args.output / "metrics.json", manifest)
    print(json.dumps({"validation": manifest["validation"], "official_test": manifest["official_test"],
                      "seconds": manifest["run_seconds"], "output": str(args.output)}), flush=True)
    if args.blur:
        export_blur(args, (y_train, y_test), valid, coef, bias)


if __name__ == "__main__":
    main()
