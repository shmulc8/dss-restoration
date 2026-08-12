#!/usr/bin/env python3
"""Modern binary PixelCNN pilot for the released Qumran letter images."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from experiments.qumran_vision.download_data import ASSETS


SIZE = 32


def fit_image(path: Path, *, mask: bool = False) -> torch.Tensor:
    """Match the original 32x32, aspect-preserving, black-padded preprocessing."""
    with Image.open(path) as source:
        image = source.convert("L")
        scale = min(1.0, SIZE / image.height, SIZE / image.width)
        if scale < 1.0:
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.Resampling.BICUBIC,
            )
        canvas = Image.new("L", (SIZE, SIZE), 0)
        canvas.paste(image, ((SIZE - image.width) // 2, (SIZE - image.height) // 2))
    values = np.asarray(canvas, dtype=np.float32)
    values = (values == 255).astype(np.float32) if mask else values / 255.0
    return torch.from_numpy(values).unsqueeze(0)


def image_files(root: Path) -> list[Path]:
    result = sorted(path for path in root.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not result:
        raise RuntimeError(f"no images under {root}")
    return result


def pairs(root: Path) -> list[tuple[Path, Path]]:
    masks = {path.stem: path for path in image_files(root / "masks")}
    result = [(letter, masks[letter.stem]) for letter in image_files(root / "letters") if letter.stem in masks]
    if not result:
        raise RuntimeError(f"no letter/mask pairs under {root}")
    return result


def transform(value: torch.Tensor, code: int) -> torch.Tensor:
    if code >= 4:
        value = value.flip(-1)
    return torch.rot90(value, code % 4, (-2, -1))


def inverse_transform(value: torch.Tensor, code: int) -> torch.Tensor:
    value = torch.rot90(value, -(code % 4), (-2, -1))
    return value.flip(-1) if code >= 4 else value


def best_orientation(mask: torch.Tensor) -> int:
    """Choose the D4 orientation exposing the most known pixels earliest."""
    weights = torch.arange(SIZE * SIZE, 0, -1, dtype=mask.dtype).reshape(1, SIZE, SIZE)
    return max(range(8), key=lambda code: float((transform(mask, code) * weights).sum()))


class WholeLetterDataset(Dataset):
    def __init__(self, root: Path, *, augment: bool = False):
        self.samples = image_files(root / "letters")
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        letter = fit_image(self.samples[index])
        return transform(letter, random.randrange(8)) if self.augment else letter


class BrokenLetterDataset(Dataset):
    def __init__(self, root: Path):
        self.samples = pairs(root)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        letter_path, mask_path = self.samples[index]
        letter = fit_image(letter_path)
        mask = fit_image(mask_path, mask=True)
        code = best_orientation(mask)
        return transform(letter * mask, code), transform(mask, code), code


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mask = torch.ones_like(self.weight)
        center_y, center_x = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        mask[:, :, center_y + 1 :, :] = 0
        mask[:, :, center_y, center_x + (mask_type == "B") :] = 0
        self.register_buffer("mask", mask)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return nn.functional.conv2d(
            value,
            self.weight * self.mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class BinaryPixelCNN(nn.Module):
    """A current-PyTorch analogue, not an exact PixelCNN++ reimplementation."""

    def __init__(self, channels: int = 48, layers: int = 7):
        super().__init__()
        body: list[nn.Module] = [MaskedConv2d("A", 1, channels, 7, padding=3), nn.ReLU()]
        for _ in range(layers):
            body.extend((MaskedConv2d("B", channels, channels, 3, padding=1), nn.ReLU()))
        body.extend((nn.Conv2d(channels, channels, 1), nn.ReLU(), nn.Conv2d(channels, 1, 1)))
        self.body = nn.Sequential(*body)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.body(value)


def device_for(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def teacher_forced_bits(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = total_pixels = 0.0
    for target in loader:
        target = target.to(device)
        total_loss += nn.functional.binary_cross_entropy_with_logits(model(target), target, reduction="sum").item()
        total_pixels += target.numel()
    return total_loss / total_pixels / math.log(2.0)


@torch.no_grad()
def complete(model: nn.Module, observed: torch.Tensor, mask: torch.Tensor, *, sample: bool) -> torch.Tensor:
    canvas = observed.clone()
    for y in range(SIZE):
        for x in range(SIZE):
            probabilities = torch.sigmoid(model(canvas)[:, :, y, x])
            values = torch.bernoulli(probabilities) if sample else (probabilities >= 0.5).float()
            missing = 1.0 - mask[:, :, y, x]
            canvas[:, :, y, x] = canvas[:, :, y, x] + values * missing
    return canvas


@torch.no_grad()
def evaluate_completion(
    model: nn.Module,
    dataset: BrokenLetterDataset,
    device: torch.device,
    samples: int,
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=len(dataset))
    observed, mask, _ = next(iter(loader))
    observed, mask = observed.to(device), mask.to(device)
    target = torch.stack(
        [transform(fit_image(letter), best_orientation(fit_image(mask_path, mask=True))) for letter, mask_path in dataset.samples]
    ).to(device)
    full_psnr: list[float] = []
    missing_psnr: list[float] = []
    for _ in range(samples):
        prediction = complete(model, observed, mask, sample=True)
        error = (prediction - target).square()
        full_mse = error.mean().item()
        missing_mse = (error * (1.0 - mask)).sum().item() / max((1.0 - mask).sum().item(), 1.0)
        full_psnr.append(-10.0 * math.log10(max(full_mse, 1e-12)))
        missing_psnr.append(-10.0 * math.log10(max(missing_mse, 1e-12)))
    return {
        "mean_psnr_full_db": float(np.mean(full_psnr)),
        "sd_psnr_full_db": float(np.std(full_psnr)),
        "mean_psnr_missing_db": float(np.mean(missing_psnr)),
        "sd_psnr_missing_db": float(np.std(missing_psnr)),
        "sampling_runs": samples,
    }


@torch.no_grad()
def render_real_cases(model: nn.Module, dataset: BrokenLetterDataset, output: Path, device: torch.device) -> None:
    observed, mask, codes = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    prediction = complete(model, observed.to(device), mask.to(device), sample=True).cpu()
    cells: list[Image.Image] = []
    for index in range(len(dataset)):
        panels = [
            inverse_transform(tensor, int(codes[index]))
            for tensor in (observed[index], mask[index], prediction[index])
        ]
        cell = Image.new("L", (288, 114), 255)
        for panel_index, tensor in enumerate(panels):
            panel = Image.fromarray((tensor[0].clamp(0, 1).numpy() * 255).astype(np.uint8)).resize((96, 96))
            cell.paste(panel, (panel_index * 96, 18))
        ImageDraw.Draw(cell).text((3, 3), dataset.samples[index][0].stem, fill=0)
        cells.append(cell)
    sheet = Image.new("L", (576, math.ceil(len(cells) / 2) * 114), 255)
    for index, cell in enumerate(cells):
        sheet.paste(cell, ((index % 2) * 288, (index // 2) * 114))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("scratch/qumran_vision"))
    parser.add_argument("--output", type=Path, default=Path("output/qumran_vision/pixelcnn_pilot.json"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = device_for(args.device)
    data_root = args.data_root.resolve()
    released_train = WholeLetterDataset(data_root / "qumran_dataset_v1/train", augment=True)
    heldout = BrokenLetterDataset(data_root / "qumran_dataset_v1/val")
    order = np.random.default_rng(args.seed).permutation(len(released_train)).tolist()
    dev_size = max(1, len(order) // 10)
    train = Subset(released_train, order[dev_size:])
    dev = Subset(WholeLetterDataset(data_root / "qumran_dataset_v1/train"), order[:dev_size])
    train_loader = DataLoader(
        train,
        args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    dev_loader = DataLoader(dev, args.batch_size)
    model = BinaryPixelCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for target in train_loader:
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(target), target)
            loss.backward()
            optimizer.step()
            running += loss.item()
        if epoch in {0, args.epochs - 1} or (epoch + 1) % 10 == 0:
            print(f"epoch={epoch + 1} mean_batch_loss={running / len(train_loader):.5f}", flush=True)

    result = {
        "status": "exploratory_modern_vision_pilot_not_a_paper_result",
        "source": "Uzan, Dershowitz, and Wolf (ICDAR 2017)",
        "source_repository_commit": "319d19621236cd2dc228194b907dbc5b6b508829",
        "upstream_archive_sha256": {
            name: expected_hash for name, (_, expected_hash) in ASSETS.items()
        },
        "model": "binary PixelCNN with D4 augmentation; not an exact PixelCNN++ reproduction",
        "torch_version": torch.__version__,
        "seed": args.seed,
        "epochs": args.epochs,
        "device": str(device),
        "released_train": len(released_train),
        "internal_train": len(train),
        "internal_dev": len(dev),
        "fragment_heldout_test": len(heldout),
        "dev_teacher_forced_bits_per_pixel": teacher_forced_bits(model, dev_loader, device),
        "fragment_heldout_completion": evaluate_completion(model, heldout, device, args.samples),
        "paper_reference_psnr_db": 15.54,
        "elapsed_seconds": time.time() - started,
        "limitations": [
            "the released training and held-out letters all come from 11Q5",
            "the 19 real cases lack machine-readable letter labels",
            "the release does not provide transcription-to-image registration",
            "dataset redistribution terms are not stated separately from the MIT code license",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    real_cases = BrokenLetterDataset(data_root / "qumran_test_letters")
    render_real_cases(model, real_cases, args.output.with_name("pixelcnn_real_cases.png"), device)
    torch.save({"state_dict": model.state_dict(), "result": result}, args.output.with_name("pixelcnn_model.pt"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
