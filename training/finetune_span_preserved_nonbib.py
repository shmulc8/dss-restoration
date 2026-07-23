"""Fine-tune MsBERT only on preserved non-biblical DSS text.

Real lacunae are present as ``<GAP>`` in the derived corpus and become
unlabelled ``[MASK]`` inputs. Artificial span masking is applied only to
preserved words, so modern editorial reconstructions never become targets.
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoModelForMaskedLM, AutoTokenizer, logging as tlog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.paths import repo_path
from utils.preserved_corpus import GAP_TOKEN, load_chunks

tlog.set_verbosity_error()

BASE_REPO = os.environ.get("BASE_REPO", "dicta-il/MsBERT")
OUTDIR = repo_path(
    os.environ.get("OUTDIR_NAME", "ft_msbert_span_preserved_nonbib")
)
MAX_LEN = int(os.environ.get("MAX_LEN", "160"))
EPOCHS = int(os.environ.get("EPOCHS", "2"))
BATCH = int(os.environ.get("BATCH", "16"))
LR = float(os.environ.get("LR", "3e-5"))
MASK_FRAC = 0.15
SPAN_P = 0.3
SPAN_MAX = 10
SEED = 42
rng = np.random.default_rng(SEED)
device = "mps" if torch.backends.mps.is_available() else "cpu"


def choose_preserved_words(words):
    eligible = [index for index, word in enumerate(words) if word != GAP_TOKEN]
    target = max(1, round(len(eligible) * MASK_FRAC))
    chosen = set()
    tries = 0
    while len(chosen) < target and tries < 100:
        tries += 1
        start = int(rng.choice(eligible))
        span_length = min(int(rng.geometric(SPAN_P)), SPAN_MAX)
        for word_index in range(start, min(start + span_length, len(words))):
            if words[word_index] == GAP_TOKEN:
                break
            chosen.add(word_index)
            if len(chosen) >= target:
                break
    return chosen


def make_batch(batch_words, tokenizer, vocab_size):
    model_words = [
        [tokenizer.mask_token if word == GAP_TOKEN else word for word in words]
        for words in batch_words
    ]
    encodings = [
        tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LEN,
        )
        for words in model_words
    ]
    max_length = max(len(encoding["input_ids"]) for encoding in encodings)
    input_ids = torch.full(
        (len(encodings), max_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
    )
    attention = torch.zeros((len(encodings), max_length), dtype=torch.long)
    labels = torch.full((len(encodings), max_length), -100, dtype=torch.long)

    for batch_index, (words, encoding) in enumerate(zip(batch_words, encodings)):
        ids = encoding["input_ids"]
        input_ids[batch_index, :len(ids)] = torch.tensor(ids)
        attention[batch_index, :len(ids)] = 1
        groups = {}
        for position, word_id in enumerate(encoding.word_ids()):
            if word_id is not None:
                groups.setdefault(word_id, []).append(position)
        chosen = choose_preserved_words(words)
        assert not any(words[word_index] == GAP_TOKEN for word_index in chosen)
        for word_index in chosen:
            for position in groups.get(word_index, []):
                labels[batch_index, position] = ids[position]
                draw = rng.random()
                if draw < 0.8:
                    input_ids[batch_index, position] = tokenizer.mask_token_id
                elif draw < 0.9:
                    input_ids[batch_index, position] = int(
                        rng.integers(vocab_size)
                    )

    return input_ids.to(device), attention.to(device), labels.to(device)


def main():
    rows = load_chunks("train")
    training_words = [row["text"].split() for row in rows]
    tokenizer = AutoTokenizer.from_pretrained(BASE_REPO, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(BASE_REPO).to(device).train()
    optimizer = AdamW(model.parameters(), lr=LR)
    steps_per_epoch = math.ceil(len(training_words) / BATCH)

    print(
        f"base={BASE_REPO} | device={device} | chunks={len(training_words)} "
        f"| epochs={EPOCHS} | batch={BATCH}"
    )
    losses = []
    for epoch in range(EPOCHS):
        order = rng.permutation(len(training_words))
        total_loss = 0.0
        for step in range(steps_per_epoch):
            indices = order[step * BATCH:(step + 1) * BATCH]
            batch_words = [training_words[index] for index in indices]
            input_ids, attention, labels = make_batch(
                batch_words,
                tokenizer,
                model.config.vocab_size,
            )
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                labels=labels,
            )
            output.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += output.loss.item()
        epoch_loss = total_loss / steps_per_epoch
        losses.append(epoch_loss)
        print(f"epoch {epoch + 1}/{EPOCHS} loss={epoch_loss:.3f}", flush=True)

    model.save_pretrained(str(OUTDIR))
    tokenizer.save_pretrained(str(OUTDIR))
    metadata = {
        "base_model": BASE_REPO,
        "training_split": "train",
        "chunks": len(training_words),
        "epochs": EPOCHS,
        "batch": BATCH,
        "learning_rate": LR,
        "seed": SEED,
        "real_gap_input": tokenizer.mask_token,
        "real_gap_label": -100,
        "reconstructed_text_used": False,
        "epoch_losses": losses,
    }
    (OUTDIR / "preserved_training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"saved -> {OUTDIR}")


if __name__ == "__main__":
    main()
