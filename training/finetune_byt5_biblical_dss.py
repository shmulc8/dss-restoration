"""Fine-tune ByT5 Generator on Combined Biblical + Non-Biblical Preserved Hebrew Text.

This script fine-tunes google/byt5-small on both non-biblical DSS chunks and
Biblical Hebrew verses to create a unified tokenization-free model checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, logging as tlog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.preserved_corpus import GAP_TOKEN, load_chunks

tlog.set_verbosity_error()

HEBREW_RE = re.compile(r"[\u05D0-\u05EB]+")


def clean_hebrew_words(text: str) -> list[str]:
    return HEBREW_RE.findall(text or "")


def fetch_biblical_training_verses() -> list[str]:
    """Fetch Embible validation / training verses for Biblical fine-tuning."""
    commit = "7c9e769274a273d0b357b066d932f1c6833ca5f8"
    path = urllib.parse.quote("data/Hit@K/mixed validetion dfs masked spaces new P/MIX_val_df_masked_spaces_5_percent.json")
    url = f"https://raw.githubusercontent.com/harelm4/Embible-Backend/{commit}/{path}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "DSS-ByT5-Train/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode("utf-8")
            verses = []
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    text = row.get("text") or row.get("verse") or row.get("sentence") or ""
                    words = clean_hebrew_words(text)
                    if len(words) >= 12:
                        verses.append(" ".join(words))
                except Exception:
                    continue
            return verses
    except Exception as err:
        print(f"Warning: Failed to download Biblical training verses ({err}). Using local DSS preserved text...")
        return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="google/byt5-small")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "ft_byt5_span_combined_biblical_seed41")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"=== Fine-tuning ByT5 on Combined Biblical + Non-Biblical Corpus ===")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Collect DSS training segments
    dss_segments = []
    for row in load_chunks("train"):
        for seg in row["text"].split(GAP_TOKEN):
            words = clean_hebrew_words(seg)
            if len(words) >= 10:
                dss_segments.append(" ".join(words))

    biblical_verses = fetch_biblical_training_verses()
    all_passages = dss_segments + biblical_verses
    random.shuffle(all_passages)

    print(f"Collected {len(dss_segments)} DSS segments + {len(biblical_verses)} Biblical verses (Total: {len(all_passages)} passages).")

    # Build synthetic span masking dataset
    dataset = []
    for passage in all_passages:
        words = passage.split()
        if len(words) < 10:
            continue
        for _ in range(2):
            span_len = random.randint(1, 3)
            idx = random.randint(3, max(3, len(words) - span_len - 3))
            left = " ".join(words[max(0, idx - 8):idx])
            gold = " ".join(words[idx:idx + span_len])
            right = " ".join(words[idx + span_len:idx + span_len + 8])

            inp = f"restoration: {left} <GAP> {right}"
            dataset.append({"input": inp, "target": gold})

    print(f"Generated {len(dataset)} synthetic span masking examples.")

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    model.train()

    num_batches = math.ceil(len(dataset) / args.batch_size)
    for epoch in range(args.epochs):
        random.shuffle(dataset)
        total_loss = 0.0
        for i in range(num_batches):
            batch = dataset[i * args.batch_size : (i + 1) * args.batch_size]
            inputs = tokenizer([b["input"] for b in batch], padding=True, truncation=True, max_length=args.max_length, return_tensors="pt").to(device)
            labels = tokenizer([b["target"] for b in batch], padding=True, truncation=True, max_length=64, return_tensors="pt").input_ids.to(device)

            labels[labels == tokenizer.pad_token_id] = -100
            outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{args.epochs} — Loss: {avg_loss:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Successfully saved combined ByT5 model to {args.output_dir}")


if __name__ == "__main__":
    main()
