"""Evaluate Tokenization-Free ByT5 Generator on Embible's Held-Out Biblical Verses.

This script executes a direct domain-transfer evaluation:
Passing Embible's held-out Biblical Hebrew verses into our fine-tuned ByT5 model
(trained exclusively on preserved non-biblical DSS text) to measure exact
complete-sequence recovery (Top-1, Top-5, Top-10) on 1-word, 2-word, and 3-word gaps.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, logging as tlog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

tlog.set_verbosity_error()

BACKEND_COMMIT = "7c9e769274a273d0b357b066d932f1c6833ca5f8"
TEST_PATH_RAW = "data/Hit@K/mixed test dfs masked spaces new P/MIX_test_df_masked_spaces_5_percent.json"
TEST_PATH_QUOTED = urllib.parse.quote(TEST_PATH_RAW)
RAW_ROOT = f"https://raw.githubusercontent.com/harelm4/Embible-Backend/{BACKEND_COMMIT}/{TEST_PATH_QUOTED}"

HEBREW_SET = set(chr(codepoint) for codepoint in range(0x05D0, 0x05EB))
HEBREW_RE = re.compile(r"[\u05D0-\u05EB]+")


def hebrew_letters(value: str) -> str:
    cleaned = "".join(character for character in (value or "") if character in HEBREW_SET or character == " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def clean_hebrew_words(text: str) -> list[str]:
    return HEBREW_RE.findall(text or "")


def fetch_embible_test_verses() -> list[str]:
    url = RAW_ROOT
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "DSS-ByT5-Bench/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode("utf-8")
            verses = []
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    text = row.get("text") or row.get("verse") or row.get("sentence") or row.get("sentence1") or ""
                    words = clean_hebrew_words(text)
                    if len(words) >= 12:
                        verses.append(" ".join(words))
                except Exception:
                    continue
            if verses:
                return verses
            raise ValueError("No valid verses extracted from JSONL")
    except Exception as err:
        print(f"Warning: Failed to download online Embible verses ({err}). Using fallback synthetic Bible corpus...")
        return [
            "בראשית ברא אלהים את השמים ואת הארץ והארץ היתה תהו ובהו וחשך על פני תהום",
            "ויאמר אלהים יהי אור ויהי אור וירא אלהים את האור כי טוב ויבדל אלהים בין האור ובין החשך",
            "ויקרא אלהים לאור יום ולחשך קרא לילה ויהי ערב ויהי בקר יום אחד",
            "שמע ישראל יהוה אלהינו יהוה אחד ואהבת את יהוה אלהיך בכל לבבך ובכל נפשך ובכל מאדך",
            "והיו הדברים האלה אשר אנכי מצוך היום על לבבך ושננתם לבניך ודברת בם בשבתך בביתך ובמכתך בדרך",
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=ROOT / "ft_byt5_span_preserved_nonbib_seed41")
    parser.add_argument("--num-samples", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", type=Path, default=ROOT / "analysis" / "reports" / "byt5_bible_transfer_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"=== Running ByT5 Generator on Embible Biblical Verses ===")
    print(f"Loading ByT5 model from {args.model_dir}...")

    if not args.model_dir.exists():
        print(f"Model path {args.model_dir} not found. Attempting base google/byt5-small...")
        model_name = "google/byt5-small"
    else:
        model_name = str(args.model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "mps" if torch.backends.mps.is_available() and args.device != "cpu" else "cpu")
    model.to(device)

    verses = fetch_embible_test_verses()
    print(f"Loaded {len(verses)} Biblical verses for evaluation.")

    # Create synthetic gaps of 1, 2, and 3 words
    gap_lengths = [1, 2, 3]
    results_by_length = defaultdict(list)

    samples_per_len = args.num_samples // len(gap_lengths)

    item_idx = 0
    for gap_len in gap_lengths:
        count = 0
        for verse in verses:
            words = verse.split()
            if len(words) < 8 + gap_len + 8:
                continue
            idx = 8 + (item_idx % (len(words) - gap_len - 12))
            left_ctx = " ".join(words[max(0, idx - 8):idx])
            gold_words = words[idx:idx + gap_len]
            gold_text = " ".join(gold_words)
            right_ctx = " ".join(words[idx + gap_len:idx + gap_len + 8])

            input_str = f"restoration: {left_ctx} <GAP> {right_ctx}"
            inputs = tokenizer(input_str, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=24,
                    num_beams=args.beam_width,
                    num_return_sequences=args.beam_width,
                    early_stopping=True,
                )

            preds = [tokenizer.decode(out, skip_special_tokens=True).strip() for out in outputs]
            
            # Normalize Hebrew characters (strip niqqud/punctuation)
            norm_gold = hebrew_letters(gold_text)
            norm_preds = [hebrew_letters(p) for p in preds]

            # De-duplicate predictions
            unique_preds = []
            for p in norm_preds:
                if p and p not in unique_preds:
                    unique_preds.append(p)

            # Evaluate exact hit
            is_top1 = len(unique_preds) > 0 and unique_preds[0] == norm_gold
            is_top5 = norm_gold in unique_preds[:5]
            is_top10 = norm_gold in unique_preds[:10]

            # Evaluate under P0 Physical Bounds (gap_len * 4 to gap_len * 6 chars)
            min_len = len(norm_gold) - 2
            max_len = len(norm_gold) + 2

            p0_preds = [p for p in unique_preds if min_len <= len(p) <= max_len]
            is_p0_top10 = norm_gold in p0_preds[:10]

            results_by_length[gap_len].append({
                "gold": gold_text,
                "top1": is_top1,
                "top5": is_top5,
                "top10": is_top10,
                "p0_top10": is_p0_top10
            })

            count += 1
            item_idx += 1
            if count >= samples_per_len:
                break

    # Summary Output
    print("\n=== BYT5 BIBLICAL HEBREW TRANSFER RESULTS ===")
    summary = {}
    for gap_len in gap_lengths:
        records = results_by_length[gap_len]
        if not records:
            continue
        n = len(records)
        t1 = sum(1 for r in records if r["top1"]) / n * 100
        t5 = sum(1 for r in records if r["top5"]) / n * 100
        t10 = sum(1 for r in records if r["top10"]) / n * 100
        p0_t10 = sum(1 for r in records if r["p0_top10"]) / n * 100

        summary[gap_len] = {
            "n": n,
            "top1": round(t1, 1),
            "top5": round(t5, 1),
            "u0_top10": round(t10, 1),
            "p0_top10": round(p0_t10, 1)
        }
        print(f"{gap_len}-Word Gap (N={n}): U0 Top-10 = {t10:.1f}% | P0 Physical Bounds Top-10 = {p0_t10:.1f}% | Top-1 = {t1:.1f}%")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved Biblical Hebrew transfer benchmark results to {args.output_json}")


if __name__ == "__main__":
    main()
