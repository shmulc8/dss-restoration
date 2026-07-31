"""Epigrapher Workbench Engine — DSS Text Restoration & Rival Hypothesis Scorer.

Provides scholar-facing tools for:
1. Spatial budget conditioning (P0 regime) given physical lacuna spatial bounds [L_min, L_max].
2. Rival hypothesis scoring: Compares competing scholarly proposals (e.g., DJD vs. Qimron vs. Sukenik).
3. Attention saliency extraction: Highlights context words driving model predictions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, logging as tlog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.preserved_corpus import GAP_TOKEN, load_chunks

tlog.set_verbosity_error()

DEFAULT_MODEL = ROOT / "ft_msbert_span_refined_best"


@dataclass
class ProposalScore:
    proposal: str
    attribution: str
    log_prob: float
    spatial_fit: float
    combined_score: float
    is_spatially_valid: bool


class EpigrapherWorkbench:
    def __init__(self, model_path: Path = DEFAULT_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if not model_path.exists():
            model_path = ROOT / "ft_msbert_span_preserved_nonbib"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path, output_attentions=True)
        self.model.to(self.device)
        self.model.eval()

    def score_rival_proposals(
        self,
        left_context: list[str],
        right_context: list[str],
        proposals: list[dict[str, str]],  # [{"proposal": "צדקה", "attribution": "DJD"}]
        min_chars: int | None = None,
        max_chars: int | None = None,
    ) -> list[ProposalScore]:
        """Score rival scholarly proposals by likelihood and spatial physical fit."""
        results = []
        ctx_left = " ".join(left_context)
        ctx_right = " ".join(right_context)

        for p in proposals:
            text = p["proposal"].strip()
            attr = p.get("attribution", "Anonymous")
            char_len = len(text)

            # Check spatial bounds
            is_valid = True
            spatial_fit = 1.0
            if min_chars is not None and char_len < min_chars:
                is_valid = False
                spatial_fit = max(0.0, 1.0 - 0.2 * (min_chars - char_len))
            elif max_chars is not None and char_len > max_chars:
                is_valid = False
                spatial_fit = max(0.0, 1.0 - 0.2 * (char_len - max_chars))

            # Compute MLM log likelihood
            input_text = f"{ctx_left} [MASK] {ctx_right}".strip()
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            mask_idx = (inputs["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, mask_idx[0]]
                log_probs = torch.log_softmax(logits, dim=-1)

            target_id = self.tokenizer.convert_tokens_to_ids(text)
            if target_id != self.tokenizer.unk_token_id:
                lp = float(log_probs[target_id].item())
            else:
                lp = -15.0  # Fallback for out-of-vocab multiword proposals

            combined = lp + (0.0 if is_valid else -5.0)

            results.append(ProposalScore(
                proposal=text,
                attribution=attr,
                log_prob=round(lp, 3),
                spatial_fit=round(spatial_fit, 2),
                combined_score=round(combined, 3),
                is_spatially_valid=is_valid,
            ))

        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results

    def extract_attention_saliency(
        self,
        left_context: list[str],
        right_context: list[str],
    ) -> list[tuple[str, float]]:
        """Extract attention weights highlighting context words driving prediction."""
        tokens = left_context + ["[MASK]"] + right_context
        text = " ".join(tokens)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        mask_pos = (inputs["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1][0].item()

        with torch.no_grad():
            outputs = self.model(**inputs)
            attentions = outputs.attentions[-1][0]  # [num_heads, seq_len, seq_len]
            mask_attn = attentions[:, mask_pos, :].mean(dim=0).cpu().numpy()

        input_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        token_attn = []
        for tok, score in zip(input_tokens, mask_attn):
            if tok not in {"[CLS]", "[SEP]", "[PAD]"}:
                token_attn.append((tok, float(score)))

        return token_attn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-lacuna", action="store_true")
    args = parser.parse_args()

    engine = EpigrapherWorkbench()

    if args.test_lacuna:
        left = ["לעשות", "ענוה", "ו"]
        right = ["ומשפט", "ואהבת", "חסד"]
        proposals = [
            {"proposal": "צדקה", "attribution": "Qimron 2013"},
            {"proposal": "אמת", "attribution": "DJD XXIX"},
            {"proposal": "יושר", "attribution": "Sukenik 1955"},
        ]

        print("\n=== EPIGRAPHER WORKBENCH RIVAL HYPOTHESIS SCORING ===")
        print(f"Context: {' '.join(left)} [GAP] {' '.join(right)}")
        print("Spatial bounds: 3-5 characters\n")

        scores = engine.score_rival_proposals(left, right, proposals, min_chars=3, max_chars=5)
        for rank, s in enumerate(scores, 1):
            valid_tag = "VALID" if s.is_spatially_valid else "INVALID (Spatial Bound Violation)"
            print(f"Rank {rank}: [{s.attribution}] '{s.proposal}' — LogProb: {s.log_prob}, SpatialFit: {s.spatial_fit} ({valid_tag})")

        print("\n=== ATTENTION SALIENCY MAP ===")
        saliency = engine.extract_attention_saliency(left, right)
        for tok, score in saliency:
            print(f"Token: {tok:12s} Attention: {score:.4f}")


if __name__ == "__main__":
    main()
