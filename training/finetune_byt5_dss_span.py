"""Fine-tune ByT5 on contiguous spans of preserved non-biblical DSS text.

This script implements a tokenization-free sequence generator (google/byt5-small).
Modern reconstructions and editorial completions are never used as labels.
Training examples mask a contiguous multiword span (1 to 3 words) of physically
preserved DSS text with a special byte mask/gap token and require the model to
generate the exact complete sequence (including letters and internal spaces).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, logging as tlog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.preserved_corpus import GAP_TOKEN, load_chunks

tlog.set_verbosity_error()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="google/byt5-small")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "ft_byt5_span_preserved_nonbib_seed41"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def corpus_sha256() -> str:
    source = ROOT / "data" / "derived" / "preserved_nonbib_chunks.jsonl"
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preserved_segments(split: str) -> list[tuple[str, str]]:
    segments = []
    for row in load_chunks(split):
        scroll = row.get("scroll", "")
        for segment in row["text"].split(GAP_TOKEN):
            normalized = " ".join(segment.split())
            if len(normalized.split()) >= 5:
                segments.append((scroll, normalized))
    return segments


def choose_word_span(
    scroll: str,
    text: str,
    rng: np.random.Generator,
) -> tuple[str, str]:
    words = text.split()
    max_words = min(3, len(words))
    probabilities = np.array([0.5, 0.3, 0.2][:max_words], dtype=float)
    probabilities /= probabilities.sum()
    word_count = int(rng.choice(np.arange(1, max_words + 1), p=probabilities))
    start_word = int(rng.integers(0, len(words) - word_count + 1))

    prefix = " ".join(words[:start_word])
    target = " ".join(words[start_word : start_word + word_count])
    suffix = " ".join(words[start_word + word_count :])

    scroll_tag = f"[{scroll}] " if scroll else ""
    context_text = f"restoration: {scroll_tag}{prefix} <extra_id_0> {suffix}".strip()
    return context_text, f"<extra_id_0> {target}"


def train() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Loading ByT5 tokenization-free model ({args.base_model}) on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=args.local_files_only
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.base_model, local_files_only=args.local_files_only
    )
    model.to(device)

    train_texts = preserved_segments("train")
    dev_texts = preserved_segments("dev")
    print(
        f"Loaded {len(train_texts)} train segments and {len(dev_texts)} dev segments."
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "base_model": args.base_model,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "corpus_sha256": corpus_sha256(),
    }
    (args.output_dir / "byt5_training_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        shuffled = train_texts.copy()
        random.shuffle(shuffled)

        batch_inputs = []
        batch_targets = []
        step_count = 0

        for scroll, text in shuffled:
            ctx, target = choose_word_span(scroll, text, rng)
            batch_inputs.append(ctx)
            batch_targets.append(target)

            if len(batch_inputs) >= args.batch_size:
                model_inputs = tokenizer(
                    batch_inputs,
                    max_length=args.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                labels = tokenizer(
                    text_target=batch_targets,
                    max_length=128,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )["input_ids"].to(device)
                labels[labels == tokenizer.pad_token_id] = -100

                optimizer.zero_grad()
                outputs = model(**model_inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                step_count += 1
                batch_inputs, batch_targets = [], []

        avg_loss = total_loss / max(1, step_count)
        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")

    print(f"Saving fine-tuned ByT5 model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete.")


if __name__ == "__main__":
    train()
