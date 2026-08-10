"""ByT5 adapter for the external eval protocol (decision #4).

Reuses the exact sentence/span structure from the merged tavbert-base run
(masking is tokenizer-independent, so word_indices and golds are identical),
generates top-10 ByT5 candidates per span, and writes predictions.jsonl in the
external schema so score_run_dir scores it bit-identically.

Candidate "tokens" are the candidate's whitespace words (no "##" prefixes), so
the external split_candidate_words aligns multi-word spans correctly.
"""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
MODEL_DIR = REPO / "models" / "ft_byt5_span_preserved_nonbib_seed41"
SRC = HERE / "merged_results" / "tavbert-base" / "predictions.jsonl"
OUT_DIR = HERE / "merged_results" / "byt5-preserved"
TOP_K = 10

device = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(device).eval()

sentences, spans_by_uid = {}, {}
for line in SRC.read_text(encoding="utf-8").splitlines():
    rec = json.loads(line)
    if rec.get("record") == "sentence":
        sentences[rec["sentence_uid"]] = rec
    else:
        spans_by_uid.setdefault(rec["sentence_uid"], []).append(rec)

OUT_DIR.mkdir(parents=True, exist_ok=True)
out = open(OUT_DIR / "predictions.jsonl", "w", encoding="utf-8")

n_done = 0
for uid, sent in sentences.items():
    words = sent["sentence"].split()
    out.write(json.dumps(sent, ensure_ascii=False) + "\n")
    for span in sorted(spans_by_uid.get(uid, []), key=lambda s: s["span_index"]):
        idxs = span["word_indices"]
        prefix = " ".join(words[: idxs[0]])
        suffix = " ".join(words[idxs[-1] + 1 :])
        scroll_tag = f"[{sent['scroll']}] " if sent.get("scroll") else ""
        ctx = f"restoration: {scroll_tag}{prefix} <extra_id_0> {suffix}".strip()
        enc = tok(ctx, return_tensors="pt", truncation=True, max_length=512).to(device)
        gold_len = len(span["gold_text"])
        with torch.no_grad():
            gen = model.generate(
                **enc,
                num_beams=TOP_K,
                num_return_sequences=TOP_K,
                max_new_tokens=max(24, gold_len + 16),
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=True,
            )
        cands, seen = [], set()
        for seq, score in zip(gen.sequences, gen.sequences_scores.tolist()):
            text = tok.decode(seq, skip_special_tokens=True).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            cands.append(
                {
                    "text": text,
                    "tokens": text.split(),
                    "token_ids": [],
                    "score": score,
                }
            )
        rec = dict(span)
        rec["candidates"] = cands
        rec["mode"] = "byt5_beam"
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n_done += 1
    if n_done % 10 == 0:
        print(f"{n_done}/{len(sentences)} sentences", flush=True)

out.close()
manifest = json.loads(
    (HERE / "merged_results" / "tavbert-base" / "manifest.json").read_text()
)
manifest["model"] = {
    "id": str(MODEL_DIR),
    "family": "byt5-seq2seq",
    "adapter": "byt5_predictor.py",
}
(OUT_DIR / "manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False)
)
print("DONE, predictions at", OUT_DIR)
