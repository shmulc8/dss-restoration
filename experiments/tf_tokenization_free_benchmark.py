"""Leakage-safe exact-span evaluation for the preserved-only ByT5 checkpoint.

The primary condition hides one to three contiguous, physically preserved DSS
words. The generator sees eight words on either side and is not told the gold
word count, character count, or word boundaries. Modern reconstructions are
never used as inputs or labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, logging as tlog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tf_embible_dss_benchmark import Item, sample_items, sample_sha256

tlog.set_verbosity_error()

MAX_WORDS = 3
MAX_CHARS = 18
CONTEXT_WORDS = 8
DEFAULT_PER_LENGTH = {"dev": 20, "heldout": 100}
DEFAULT_SEEDS = {"dev": 71, "heldout": 73}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "ft_byt5_span_preserved_nonbib_seed41",
    )
    parser.add_argument("--split", choices=("dev", "heldout"), default="dev")
    parser.add_argument("--per-length", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_candidate(text: str) -> str:
    return " ".join(text.replace("<extra_id_0>", " ").split())


def generate_candidates(
    items: Sequence[Item],
    *,
    tokenizer: Any,
    model: Any,
    device: str,
    beam_width: int,
    batch_size: int,
    max_new_tokens: int,
) -> list[list[str]]:
    if beam_width < 10:
        raise ValueError("beam width must be at least 10 for a Top-10 benchmark")
    if batch_size < 1:
        raise ValueError("batch size must be positive")

    all_candidates: list[list[str]] = []
    for offset in range(0, len(items), batch_size):
        batch = items[offset : offset + batch_size]
        inputs = [
            f"{' '.join(item.left)} <extra_id_0> {' '.join(item.right)}"
            for item in batch
        ]
        encoding = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.inference_mode():
            outputs = model.generate(
                **encoding,
                max_new_tokens=max_new_tokens,
                num_beams=beam_width,
                num_return_sequences=10,
                early_stopping=True,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for row_start in range(0, len(decoded), 10):
            seen = set()
            candidates = []
            for raw in decoded[row_start : row_start + 10]:
                candidate = normalize_candidate(raw)
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    candidates.append(candidate)
            all_candidates.append(candidates)
        print(
            f"generated {min(offset + len(batch), len(items))}/{len(items)}",
            flush=True,
        )
    if len(all_candidates) != len(items):
        raise RuntimeError("generation output count does not match input count")
    return all_candidates


def summarize(
    items: Sequence[Item],
    candidates: Sequence[Sequence[str]],
) -> dict[str, Any]:
    cases = []
    ranks = []
    by_length: dict[int, list[int]] = defaultdict(list)
    for item, rows in zip(items, candidates):
        rank = next(
            (
                index
                for index, candidate in enumerate(rows)
                if candidate == item.gold_text
            ),
            999,
        )
        ranks.append(rank)
        by_length[len(item.gold)].append(rank)
        cases.append(
            {
                "item_id": item.item_id,
                "scroll": item.scroll,
                "word_count": len(item.gold),
                "gold": item.gold_text,
                "rank": None if rank == 999 else rank,
                "top10": list(rows[:10]),
            }
        )

    def metrics(local: Sequence[int]) -> dict[str, Any]:
        return {
            "n": len(local),
            "exact_top1": 100 * sum(rank == 0 for rank in local) / len(local),
            "exact_top5": 100 * sum(rank < 5 for rank in local) / len(local),
            "exact_top10": 100 * sum(rank < 10 for rank in local) / len(local),
            "mean_reciprocal_rank": sum(
                1 / (rank + 1) if rank != 999 else 0 for rank in local
            )
            / len(local),
        }

    return {
        **metrics(ranks),
        "by_word_count": {
            str(length): metrics(local) for length, local in sorted(by_length.items())
        },
        "cases": cases,
    }


def evaluate() -> None:
    args = parse_args()
    if args.per_length is not None and args.per_length < 1:
        raise ValueError("per-length must be positive")
    model_dir = args.model_dir.resolve()
    weights = model_dir / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(
            f"complete model weights not found at {weights}; train the checkpoint first"
        )

    per_length = args.per_length or DEFAULT_PER_LENGTH[args.split]
    seed = args.seed if args.seed is not None else DEFAULT_SEEDS[args.split]
    items, eligible = sample_items(
        args.split,
        per_length=per_length,
        max_words=MAX_WORDS,
        max_chars=MAX_CHARS,
        context_words=CONTEXT_WORDS,
        seed=seed,
    )
    expected = per_length * MAX_WORDS
    if len(items) != expected:
        raise ValueError(f"expected {expected} benchmark items, found {len(items)}")

    device = choose_device(args.device)
    print(f"loading {model_dir} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=args.local_files_only,
    )
    model = (
        AutoModelForSeq2SeqLM.from_pretrained(
            model_dir,
            local_files_only=args.local_files_only,
        )
        .to(device)
        .eval()
    )
    candidates = generate_candidates(
        items,
        tokenizer=tokenizer,
        model=model,
        device=device,
        beam_width=args.beam_width,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    results = summarize(items, candidates)
    report = {
        "status": "single_checkpoint_development"
        if args.split == "dev"
        else "single_checkpoint_heldout",
        "protocol": {
            "split": args.split,
            "sample_seed": seed,
            "per_word_length": per_length,
            "sample_sha256": sample_sha256(items),
            "eligible_pool_by_word_count": eligible,
            "context_words_each_side": CONTEXT_WORDS,
            "candidate_word_counts": [1, 2, 3],
            "gold_length_or_boundaries_given": False,
            "modern_reconstructions_used": False,
            "model_dir": str(model_dir),
            "model_sha256": file_sha256(weights),
            "beam_width": args.beam_width,
            "returned_sequences": 10,
        },
        "results": results,
    }
    print(json.dumps({**report["protocol"], **results}, ensure_ascii=False, indent=2))
    if args.output_json:
        output = args.output_json.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    evaluate()
