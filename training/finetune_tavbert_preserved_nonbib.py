"""Fine-tune TavBERT on contiguous spans of preserved non-biblical DSS text.

Modern reconstructions and natural gap contents are never used. Training
examples are preserved text segments separated by ``<GAP>``. Each example masks
one contiguous span of one to three complete words, including internal spaces,
so the character model learns both content and word boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoModelForMaskedLM, AutoTokenizer, logging as tlog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.preserved_corpus import GAP_TOKEN, load_chunks

tlog.set_verbosity_error()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="tau/tavbert-he")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def corpus_sha256() -> str:
    source = ROOT / "data" / "derived" / "preserved_nonbib_chunks.jsonl"
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preserved_segments(split: str) -> list[str]:
    segments = []
    for row in load_chunks(split):
        for segment in row["text"].split(GAP_TOKEN):
            normalized = " ".join(segment.split())
            if len(normalized.split()) >= 5:
                segments.append(normalized)
    return segments


def choose_word_span(
    text: str,
    rng: np.random.Generator,
) -> tuple[int, int, int]:
    words = text.split()
    max_words = min(3, len(words))
    probabilities = np.array([0.5, 0.3, 0.2][:max_words], dtype=float)
    probabilities /= probabilities.sum()
    word_count = int(rng.choice(np.arange(1, max_words + 1), p=probabilities))
    start_word = int(rng.integers(0, len(words) - word_count + 1))
    prefix = " ".join(words[:start_word])
    target = " ".join(words[start_word:start_word + word_count])
    start_character = len(prefix) + (1 if prefix else 0)
    return start_character, start_character + len(target), word_count


def make_batch(
    texts: list[str],
    *,
    tokenizer: Any,
    rng: np.random.Generator,
    max_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    encodings = tokenizer(
        texts,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    input_ids = encodings["input_ids"].clone()
    labels = torch.full_like(input_ids, -100)
    masked_word_counts = []
    for row_index, text in enumerate(texts):
        start, end, word_count = choose_word_span(text, rng)
        offsets = encodings["offset_mapping"][row_index].tolist()
        positions = [
            position
            for position, (token_start, token_end) in enumerate(offsets)
            if token_end > token_start
            and token_start >= start
            and token_end <= end
        ]
        if not positions:
            continue
        labels[row_index, positions] = input_ids[row_index, positions]
        input_ids[row_index, positions] = tokenizer.mask_token_id
        masked_word_counts.append(word_count)
    return (
        input_ids.to(device),
        encodings["attention_mask"].to(device),
        labels.to(device),
        masked_word_counts,
    )


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    if args.output_dir.resolve() == ROOT:
        raise ValueError("output directory cannot be the repository root")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    segments = preserved_segments("train")
    if not segments:
        raise RuntimeError("no preserved training segments")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if not tokenizer.is_fast:
        raise ValueError("TavBERT training requires offset mappings")
    model = AutoModelForMaskedLM.from_pretrained(
        args.base_model,
        local_files_only=args.local_files_only,
    ).to(device).train()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    steps_per_epoch = math.ceil(len(segments) / args.batch_size)
    epoch_losses = []
    masked_word_histogram = {"1": 0, "2": 0, "3": 0}

    print(
        f"base={args.base_model} device={device} segments={len(segments)} "
        f"epochs={args.epochs} batch={args.batch_size} seed={args.seed}"
    )
    for epoch in range(args.epochs):
        order = rng.permutation(len(segments))
        total_loss = 0.0
        completed_steps = 0
        for step in range(steps_per_epoch):
            indices = order[
                step * args.batch_size:(step + 1) * args.batch_size
            ]
            batch = [segments[index] for index in indices]
            input_ids, attention, labels, word_counts = make_batch(
                batch,
                tokenizer=tokenizer,
                rng=rng,
                max_length=args.max_length,
                device=device,
            )
            if not word_counts:
                continue
            for word_count in word_counts:
                masked_word_histogram[str(word_count)] += 1
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                labels=labels,
            )
            output.loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += float(output.loss.item())
            completed_steps += 1
        epoch_loss = total_loss / max(1, completed_steps)
        epoch_losses.append(epoch_loss)
        print(
            f"epoch {epoch + 1}/{args.epochs} loss={epoch_loss:.4f}",
            flush=True,
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metadata = {
        "base_model": args.base_model,
        "training_split": "preserved_nonbib train",
        "training_segments": len(segments),
        "objective": "one contiguous preserved span of 1-3 complete words",
        "natural_gap_contents_used": False,
        "modern_reconstructions_used": False,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "seed": args.seed,
        "device": device,
        "epoch_losses": epoch_losses,
        "masked_word_histogram": masked_word_histogram,
        "corpus_sha256": corpus_sha256(),
    }
    (output_dir / "preserved_char_training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved {output_dir}")


if __name__ == "__main__":
    main()
