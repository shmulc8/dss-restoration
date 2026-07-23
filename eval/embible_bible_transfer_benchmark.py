"""Run the DSS completion systems on Embible's held-out Biblical verses.

This is a domain-transfer diagnostic. It reconstructs the known original text
from Embible's public JSONL validation/test files, then creates the same
contiguous one-, two-, and three-word synthetic gaps used by our DSS benchmark.
Biblical text is never used for model training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, logging as tlog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.tf_embible_dss_benchmark import (
    Item,
    cached_huggingface_revision,
    evaluate_records,
    file_sha256,
    fit_ensemble_weight,
    fit_penalty,
    generate_records,
    is_hebrew_token,
    is_hebrew_word,
    sample_sha256,
)


BACKEND_COMMIT = "7c9e769274a273d0b357b066d932f1c6833ca5f8"
RAW_ROOT = (
    "https://raw.githubusercontent.com/harelm4/Embible-Backend/"
    f"{BACKEND_COMMIT}/"
)
SOURCES = {
    "dev": {
        "path": (
            "data/Hit@K/mixed validetion dfs masked spaces new P/"
            "MIX_val_df_masked_spaces_5_percent.json"
        ),
        "sha256": "3646d0d7e39e85006c4fa4b1531a1f8a847d7ea56e0a3d8cd66ebaa825ecf96f",
        "rows": 535,
    },
    "test": {
        "path": (
            "data/Hit@K/mixed test dfs masked spaces new P/"
            "MIX_test_df_masked_spaces_5_percent.json"
        ),
        "sha256": "c3a707c1d8ef6d934df9f55467abadd9b7e6e42fae9c534a18fc517a0277e7ea",
        "rows": 536,
    },
}
MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024

tlog.set_verbosity_error()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--word-model",
        type=Path,
        default=ROOT / "ft_msbert_span_preserved_nonbib",
    )
    parser.add_argument("--char-model", default="tau/tavbert-he")
    parser.add_argument("--dev-per-length", type=int, default=20)
    parser.add_argument("--test-per-length", type=int, default=40)
    parser.add_argument("--max-words", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=18)
    parser.add_argument("--context-words", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--top-k-per-step", type=int, default=24)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "analysis" / "reports" / "embible_bible_transfer.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=ROOT / "analysis" / "reports" / "EMBIBLE_BIBLE_TRANSFER.md",
    )
    return parser.parse_args()


def source_url(relative_path: str) -> str:
    encoded = urllib.parse.quote(relative_path, safe="/")
    url = urllib.parse.urljoin(RAW_ROOT, encoded)
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "raw.githubusercontent.com":
        raise ValueError("unexpected Embible source URL")
    return url


def fetch_source(split: str) -> tuple[list[dict[str, Any]], str]:
    metadata = SOURCES[split]
    request = urllib.request.Request(
        source_url(str(metadata["path"])),
        headers={"User-Agent": "dss-restoration-research/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        final_url = urllib.parse.urlparse(response.geturl())
        if (
            final_url.scheme != "https"
            or final_url.hostname != "raw.githubusercontent.com"
        ):
            raise ValueError("unexpected redirect for Embible source")
        payload = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(payload) > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"{split} source exceeds download limit")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != metadata["sha256"]:
        raise ValueError(f"{split} source hash mismatch: {digest}")
    rows = [
        json.loads(line)
        for line in payload.decode("utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != metadata["rows"]:
        raise ValueError(f"{split} row count mismatch: {len(rows)}")
    return rows, digest


def fetch_json(relative_path: str) -> tuple[Any, str]:
    request = urllib.request.Request(
        source_url(relative_path),
        headers={"User-Agent": "dss-restoration-research/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        final_url = urllib.parse.urlparse(response.geturl())
        if (
            final_url.scheme != "https"
            or final_url.hostname != "raw.githubusercontent.com"
        ):
            raise ValueError("unexpected redirect for Embible source")
        payload = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(payload) > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"source exceeds download limit: {relative_path}")
    return json.loads(payload.decode("utf-8")), hashlib.sha256(payload).hexdigest()


def normalize_bible_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value)
    without_marks = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )
    without_marks = without_marks.replace("־", " ")
    filtered = "".join(
        character
        if ("א" <= character <= "ת" or character in {"?", " "})
        else ""
        for character in without_marks
    )
    return " ".join(filtered.split())


def resolve_canonical_verses(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, str]]:
    books: dict[str, list[str]] = {}
    book_hashes: dict[str, str] = {}
    for book in sorted({str(row["name"]) for row in rows}):
        verses, digest = fetch_json(
            f"data/bible_books_jsons/{book}.txt.json"
        )
        if not isinstance(verses, list):
            raise ValueError(f"unexpected Bible book payload: {book}")
        books[book] = [normalize_bible_text(str(verse)) for verse in verses]
        book_hashes[book] = digest

    resolved = []
    resolution = {"unique": 0, "ambiguous": 0, "missing": 0}
    for row in rows:
        masked = normalize_bible_text(str(row["verse"]))
        pattern = re.compile(
            "^" + re.escape(masked).replace(r"\?", ".") + "$"
        )
        matches = [
            verse
            for verse in books[str(row["name"])]
            if pattern.fullmatch(verse)
        ]
        if len(matches) == 1:
            resolved.append({**row, "resolved_verse": matches[0]})
            resolution["unique"] += 1
        elif matches:
            resolution["ambiguous"] += 1
        else:
            resolution["missing"] += 1
    return resolved, resolution, book_hashes


def restore_verse(row: dict[str, Any]) -> str:
    verse = str(row["verse"])
    restored = list(verse)
    missing = row.get("missing_dictionary", {})
    for raw_index, raw_character in missing.items():
        index = int(raw_index)
        character = str(raw_character)
        if index < 0 or index >= len(restored):
            raise ValueError(f"missing index outside verse: {index}")
        if restored[index] != "?":
            raise ValueError(f"expected '?' at index {index}")
        if len(character) != 1:
            raise ValueError("replacement must be exactly one character")
        restored[index] = character
    result = "".join(restored)
    if "?" in result:
        raise ValueError("source row contains an unrestored placeholder")
    return " ".join(result.split())


def candidate_spans(
    rows: list[dict[str, Any]],
    split: str,
    *,
    max_words: int,
    max_chars: int,
    context_words: int,
) -> dict[int, list[Item]]:
    pools: dict[int, list[Item]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        words = str(row["resolved_verse"]).split()
        for length in range(1, max_words + 1):
            if len(words) < context_words * 2 + length:
                continue
            for start in range(
                context_words,
                len(words) - context_words - length + 1,
            ):
                gold = tuple(words[start:start + length])
                left = tuple(words[start - context_words:start])
                right = tuple(words[start + length:start + length + context_words])
                if not all(is_hebrew_word(word) for word in gold):
                    continue
                if len(" ".join(gold)) > max_chars:
                    continue
                if not all(is_hebrew_token(word) for word in (*left, *right)):
                    continue
                pools[length].append(
                    Item(
                        item_id=(
                            f"embible-bible:{split}:{row['name']}:"
                            f"{row['verse_idx']}:{start}:{length}"
                        ),
                        scroll=f"{row['name']}:{row_index}",
                        left=left,
                        gold=gold,
                        right=right,
                    )
                )
    return pools


def sample_items(
    rows: list[dict[str, Any]],
    split: str,
    *,
    per_length: int,
    max_words: int,
    max_chars: int,
    context_words: int,
    seed: int,
) -> tuple[list[Item], dict[str, int]]:
    pools = candidate_spans(
        rows,
        split,
        max_words=max_words,
        max_chars=max_chars,
        context_words=context_words,
    )
    rng = random.Random(seed)
    selected: list[Item] = []
    used_verses: set[str] = set()
    eligible = {str(length): len(pools.get(length, [])) for length in pools}
    # Allocate the rarest, longest targets first so shorter-span pools do not
    # consume their verses.
    for length in range(max_words, 0, -1):
        pool = list(pools.get(length, []))
        rng.shuffle(pool)
        for item in pool:
            if item.scroll in used_verses:
                continue
            selected.append(item)
            used_verses.add(item.scroll)
            if sum(len(row.gold) == length for row in selected) == per_length:
                break
    return selected, eligible


def compact_metrics(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "n",
        "top1",
        "top5",
        "top10",
        "word_hit1",
        "word_hit5",
        "mean_top1_cer",
        "mean_boundary_f1",
        "mean_word_count_error",
        "decoder_failure_rate",
    )
    return {key: result[key] for key in keys}


def render_markdown(report: dict[str, Any]) -> str:
    rows = []
    for word_count in ("1", "2", "3"):
        results = report["results"]["by_word_count"][word_count]
        for system in (
            "uwc_word",
            "char_unknown",
            "embible_overlap_ensemble",
            "rank_ensemble",
            "cwc_word_oracle",
        ):
            metrics = results[system]
            rows.append(
                f"| {word_count} | {system} | {metrics['top1']:.1f}% | "
                f"{metrics['top5']:.1f}% | {metrics['top10']:.1f}% | "
                f"{metrics['word_hit1']:.1f}% | {metrics['word_hit5']:.1f}% |"
            )
    return f"""# Bible-to-DSS domain-transfer diagnostic

The models and decoder are our DSS systems. Embible's Biblical validation and
test verses are evaluation-only: their masked characters are restored, then the
same contiguous 1/2/3-word synthetic damage used in our DSS benchmark is
applied. This is an apples-to-apples domain diagnostic, not a reproduction of
Embible's random character/word masking or published metrics.

| Hidden words | System | Exact Top-1 | Exact Top-5 | Exact Top-10 | Seq WordHit@1 | Seq WordHit@5 |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Development tuning used {report['protocol']['dev_items']} Biblical spans and the
held-out evaluation used {report['protocol']['test_items']} spans, with at most
one target per verse in each split. No Biblical text was used for training.
"""


def main() -> None:
    args = parse_args()
    if args.dev_per_length < 1 or args.test_per_length < 1:
        raise ValueError("sample sizes must be positive")
    raw_dev_rows, dev_source_hash = fetch_source("dev")
    raw_test_rows, test_source_hash = fetch_source("test")
    dev_rows, dev_resolution, dev_book_hashes = resolve_canonical_verses(
        raw_dev_rows
    )
    test_rows, test_resolution, test_book_hashes = resolve_canonical_verses(
        raw_test_rows
    )
    dev_items, dev_eligible = sample_items(
        dev_rows,
        "dev",
        per_length=args.dev_per_length,
        max_words=args.max_words,
        max_chars=args.max_chars,
        context_words=args.context_words,
        seed=171,
    )
    test_items, test_eligible = sample_items(
        test_rows,
        "test",
        per_length=args.test_per_length,
        max_words=args.max_words,
        max_chars=args.max_chars,
        context_words=args.context_words,
        seed=173,
    )
    expected_dev = args.dev_per_length * args.max_words
    expected_test = args.test_per_length * args.max_words
    if len(dev_items) != expected_dev or len(test_items) != expected_test:
        raise RuntimeError(
            f"insufficient eligible spans: dev={len(dev_items)}/{expected_dev}, "
            f"test={len(test_items)}/{expected_test}"
        )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    word_source = args.word_model.resolve()
    if not word_source.is_dir():
        raise FileNotFoundError(f"word checkpoint not found: {word_source}")
    word_tokenizer = AutoTokenizer.from_pretrained(
        str(word_source), use_fast=True, local_files_only=True
    )
    word_model = AutoModelForMaskedLM.from_pretrained(
        str(word_source), local_files_only=True
    ).to(device).eval()
    char_tokenizer = AutoTokenizer.from_pretrained(
        args.char_model,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    char_model = AutoModelForMaskedLM.from_pretrained(
        args.char_model,
        local_files_only=args.local_files_only,
    ).to(device).eval()

    print(f"device={device}; bible dev={len(dev_items)}; test={len(test_items)}")
    dev_records = generate_records(
        dev_items,
        word_tokenizer=word_tokenizer,
        word_model=word_model,
        char_tokenizer=char_tokenizer,
        char_model=char_model,
        device=device,
        args=args,
    )
    word_penalty = fit_penalty(dev_records, "word")
    char_penalty = fit_penalty(dev_records, "char")
    ensemble_weight = fit_ensemble_weight(
        dev_records,
        word_penalty=word_penalty,
        char_penalty=char_penalty,
    )
    print(
        f"dev fit: word_penalty={word_penalty}, char_penalty={char_penalty}, "
        f"ensemble_word_weight={ensemble_weight}"
    )
    test_records = generate_records(
        test_items,
        word_tokenizer=word_tokenizer,
        word_model=word_model,
        char_tokenizer=char_tokenizer,
        char_model=char_model,
        device=device,
        args=args,
    )
    evaluation = evaluate_records(
        test_records,
        word_penalty=word_penalty,
        char_penalty=char_penalty,
        ensemble_weight=ensemble_weight,
    )
    word_checkpoint = word_source / "model.safetensors"
    report = {
        "protocol": {
            "purpose": "domain transfer with the DSS task and decoder held fixed",
            "bible_used_for_training": False,
            "source_repository": "https://github.com/harelm4/Embible-Backend",
            "source_commit": BACKEND_COMMIT,
            "source_sha256": {
                "dev": dev_source_hash,
                "test": test_source_hash,
            },
            "canonical_book_sha256": {
                "dev": dev_book_hashes,
                "test": test_book_hashes,
            },
            "source_resolution": {
                "dev": dev_resolution,
                "test": test_resolution,
            },
            "damage": "contiguous one-, two-, and three-word synthetic gaps",
            "not_embible_metric_replication": True,
            "word_model": word_source.name,
            "word_model_sha256": (
                file_sha256(word_checkpoint) if word_checkpoint.is_file() else None
            ),
            "char_model": args.char_model,
            "char_model_revision": cached_huggingface_revision(args.char_model),
            "dev_items": len(dev_items),
            "test_items": len(test_items),
            "dev_per_length": args.dev_per_length,
            "test_per_length": args.test_per_length,
            "dev_seed": 171,
            "test_seed": 173,
            "dev_sample_sha256": sample_sha256(dev_items),
            "test_sample_sha256": sample_sha256(test_items),
            "eligible_dev_by_words": dev_eligible,
            "eligible_test_by_words": test_eligible,
            "dev_fit": {
                "word_length_penalty": word_penalty,
                "char_length_penalty": char_penalty,
                "ensemble_word_weight": ensemble_weight,
            },
        },
        "results": {
            key: value for key, value in evaluation.items() if key != "cases"
        },
        "cases": evaluation["cases"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(report)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"saved {args.output_json}")
    print(f"saved {args.output_markdown}")


if __name__ == "__main__":
    main()
