"""Fine-tune MsBERT on preserved DSS words with deterministic span masking."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoModelForMaskedLM, AutoTokenizer, logging as tlog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from curation.preserved_corpus import CHUNKS_PATH, GAP_TOKEN, load_chunks

tlog.set_verbosity_error()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="dicta-il/MsBERT")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help="optional JSON with train_scrolls; used for composition exclusion",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_preserved_words(
    words: list[str], rng: np.random.Generator
) -> set[int]:
    eligible = [index for index, word in enumerate(words) if word != GAP_TOKEN]
    target = max(1, round(len(eligible) * 0.15))
    chosen: set[int] = set()
    for _ in range(100):
        if len(chosen) >= target:
            break
        start = int(rng.choice(eligible))
        span_length = min(int(rng.geometric(0.3)), 10)
        for word_index in range(start, min(start + span_length, len(words))):
            if words[word_index] == GAP_TOKEN:
                break
            chosen.add(word_index)
            if len(chosen) >= target:
                break
    return chosen


def make_batch(
    batch_words: list[list[str]],
    *,
    tokenizer,
    rng: np.random.Generator,
    max_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model_words = [
        [tokenizer.mask_token if word == GAP_TOKEN else word for word in words]
        for words in batch_words
    ]
    encodings = [
        tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
        )
        for words in model_words
    ]
    longest = max(len(encoding["input_ids"]) for encoding in encodings)
    input_ids = torch.full(
        (len(encodings), longest), tokenizer.pad_token_id, dtype=torch.long
    )
    attention = torch.zeros((len(encodings), longest), dtype=torch.long)
    labels = torch.full((len(encodings), longest), -100, dtype=torch.long)
    for batch_index, (words, encoding) in enumerate(zip(batch_words, encodings)):
        ids = encoding["input_ids"]
        input_ids[batch_index, : len(ids)] = torch.tensor(ids)
        attention[batch_index, : len(ids)] = 1
        groups: dict[int, list[int]] = {}
        for position, word_id in enumerate(encoding.word_ids()):
            if word_id is not None:
                groups.setdefault(word_id, []).append(position)
        for word_index in choose_preserved_words(words, rng):
            for position in groups.get(word_index, []):
                labels[batch_index, position] = ids[position]
                draw = rng.random()
                if draw < 0.8:
                    input_ids[batch_index, position] = tokenizer.mask_token_id
                elif draw < 0.9:
                    input_ids[batch_index, position] = int(
                        rng.integers(tokenizer.vocab_size)
                    )
    return input_ids.to(device), attention.to(device), labels.to(device)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or not 1 <= args.batch_size <= 128:
        raise ValueError("epochs and batch size must be positive and batch <= 128")
    output_dir = args.output_dir.resolve()
    if output_dir == ROOT or ROOT not in output_dir.parents:
        raise ValueError("output directory must be inside the repository")

    allowed_scrolls = None
    split_manifest_sha256 = None
    if args.split_manifest:
        split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
        allowed_scrolls = set(split_manifest["train_scrolls"])
        split_manifest_sha256 = sha256(args.split_manifest)

    rows = load_chunks("train")
    if allowed_scrolls is not None:
        rows = [row for row in rows if row["scroll"] in allowed_scrolls]
    training_words = [row["text"].split() for row in rows]
    if not training_words:
        raise RuntimeError("no training chunks remain")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if not tokenizer.is_fast:
        raise ValueError("training requires a fast tokenizer with word alignment")
    model = AutoModelForMaskedLM.from_pretrained(
        args.base_model,
        local_files_only=args.local_files_only,
    ).to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    steps_per_epoch = math.ceil(len(training_words) / args.batch_size)
    losses: list[float] = []

    print(
        f"base={args.base_model} device={device} chunks={len(training_words)} "
        f"epochs={args.epochs} batch={args.batch_size} seed={args.seed}",
        flush=True,
    )
    for epoch in range(args.epochs):
        order = rng.permutation(len(training_words))
        total_loss = 0.0
        for step in range(steps_per_epoch):
            indices = order[step * args.batch_size : (step + 1) * args.batch_size]
            batch = [training_words[index] for index in indices]
            input_ids, attention, labels = make_batch(
                batch,
                tokenizer=tokenizer,
                rng=rng,
                max_length=args.max_length,
                device=device,
            )
            output = model(input_ids=input_ids, attention_mask=attention, labels=labels)
            output.loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += float(output.loss.item())
        loss = total_loss / steps_per_epoch
        losses.append(loss)
        print(f"epoch {epoch + 1}/{args.epochs} loss={loss:.4f}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metadata = {
        "base_model": args.base_model,
        "training_split": "preserved_nonbib train",
        "chunks": len(training_words),
        "scrolls": len({row["scroll"] for row in rows}),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "seed": args.seed,
        "real_gap_input": tokenizer.mask_token,
        "real_gap_label": -100,
        "modern_reconstructions_used": False,
        "corpus_sha256": sha256(CHUNKS_PATH),
        "split_manifest_sha256": split_manifest_sha256,
        "epoch_losses": losses,
    }
    (output_dir / "preserved_training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved -> {output_dir}")


if __name__ == "__main__":
    main()
