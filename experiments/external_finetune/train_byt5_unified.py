"""Fine-tune google/byt5-small on Itay's ppp_nonbib dataset, eval-matched task.

Training examples replicate exactly what the ByT5 adapter sees at eval time:
spans drawn by the external PercentageContentMaskingPolicy (mask_ratio 0.3,
span_concentration 0.5, per-sentence deterministic seeding), input
"restoration: [scroll] prefix <extra_id_0> suffix" with all non-span words
visible, target "<extra_id_0> span". Multiple mask draws per sentence (seed
offsets) give ~3x data. Early stopping on val loss; best checkpoint saved to
models/byt5_unified_ppp_nonbib.
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
)

HERE = Path(__file__).resolve().parent
DATA = (
    HERE.parent
    / "external_impl"
    / "new_dead_sea_scrolls"
    / "data_preparation"
    / "dss_sentences_min7_splits_ppp_nonbib.xlsx"
)
OUT = HERE / "models" / "byt5_unified_ppp_nonbib"
sys.path.insert(0, str(TUNING))
from tuning.eval_utils import PercentageContentMaskingPolicy
from transformers import AutoTokenizer as AT

SEED = 42
MAX_EPOCHS = 8
PATIENCE = 2
LR = 5e-4
MICRO_BATCH = 4
GRAD_ACCUM = 8
MASK_DRAWS = 3  # mask-policy draws per sentence (different uids -> different spans)

torch.manual_seed(SEED)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# The policy only uses its tokenizer to build [MASK] strings; word selection is
# tokenizer-independent. TavBERT's tokenizer is the cheapest valid choice.
policy_tok = AT.from_pretrained("tau/tavbert-he")
policy = PercentageContentMaskingPolicy(
    policy_tok, mask_ratio=0.3, span_concentration=0.5, seed=SEED
)

tok = AutoTokenizer.from_pretrained("google/byt5-small")
model = T5ForConditionalGeneration.from_pretrained("google/byt5-small").to(device)


def spans_of(indices):
    groups, cur = [], [indices[0]]
    for i in indices[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    return groups


def build_examples(df, draws):
    examples = []
    for _, r in df.iterrows():
        words = str(r["sentence"]).split()
        for d in range(draws):
            ex = policy.generate(
                str(r["sentence"]),
                uid=f"{r['scroll']}|{r['fragment']}|{r['line_start']}|{r['line_end']}|draw{d}",
            )
            if not ex.masked_word_indices:
                continue
            for grp in spans_of(ex.masked_word_indices):
                prefix = " ".join(words[: grp[0]])
                suffix = " ".join(words[grp[-1] + 1 :])
                target = " ".join(words[grp[0] : grp[-1] + 1])
                ctx = f"restoration: [{r['scroll']}] {prefix} <extra_id_0> {suffix}".strip()
                examples.append((ctx, f"<extra_id_0> {target}"))
    return examples


df = pd.read_excel(DATA).drop(columns=["Unnamed: 0"], errors="ignore")
train_ex = build_examples(df[df["split"] == "train"], MASK_DRAWS)
val_ex = build_examples(df[df["split"] == "val"], 1)
print(f"train examples: {len(train_ex)}, val examples: {len(val_ex)}", flush=True)


class SpanDS(Dataset):
    def __init__(self, ex):
        self.ex = ex

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


def collate(batch):
    ctxs, tgts = zip(*batch)
    enc = tok(
        list(ctxs), return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    lab = tok(
        list(tgts), return_tensors="pt", padding=True, truncation=True, max_length=64
    )
    labels = lab["input_ids"].masked_fill(lab["input_ids"] == tok.pad_token_id, -100)
    return enc["input_ids"], enc["attention_mask"], labels


train_dl = DataLoader(
    SpanDS(train_ex), batch_size=MICRO_BATCH, shuffle=True, collate_fn=collate
)
val_dl = DataLoader(
    SpanDS(val_ex), batch_size=MICRO_BATCH, shuffle=False, collate_fn=collate
)

steps = (len(train_dl) // GRAD_ACCUM) * MAX_EPOCHS
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)


@torch.no_grad()
def val_loss():
    model.eval()
    tot, n = 0.0, 0
    for ids, am, labels in val_dl:
        out = model(
            input_ids=ids.to(device),
            attention_mask=am.to(device),
            labels=labels.to(device),
        )
        tot += out.loss.item() * ids.size(0)
        n += ids.size(0)
    model.train()
    return tot / n


best, bad = float("inf"), 0
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    running, seen = 0.0, 0
    opt.zero_grad()
    for step, (ids, am, labels) in enumerate(train_dl, 1):
        out = model(
            input_ids=ids.to(device),
            attention_mask=am.to(device),
            labels=labels.to(device),
        )
        (out.loss / GRAD_ACCUM).backward()
        running += out.loss.item()
        seen += 1
        if step % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
    vl = val_loss()
    print(
        f"epoch {epoch} | train_loss {running / seen:.4f} | val_loss {vl:.4f}",
        flush=True,
    )
    if vl < best - 0.01:
        best, bad = vl, 0
        OUT.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(OUT)
        tok.save_pretrained(OUT)
        print(f"saved best (val_loss {vl:.4f})", flush=True)
    else:
        bad += 1
        if bad >= PATIENCE:
            print(f"early stopping at epoch {epoch}", flush=True)
            break

print("TRAINING DONE:", OUT, flush=True)
