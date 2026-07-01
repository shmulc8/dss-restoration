"""Why are restoration metrics low? Check on real single-word lacunae:
  1. OOV gold rate — words the model's vocab literally cannot emit (auto-miss).
  2. exact vs 'loose' match — Qumran plene spelling (waw/yod matres, final forms)
     makes exact-match undercount scholar-acceptable predictions.
  3. eyeball concrete gold vs top-5.
"""
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, logging as tlog
from tf.app import use
tlog.set_verbosity_error()

WINDOW, MIN_PRESERVED, MAX_ITEMS = 20, 8, 300
TOPN, BEAM, K = 20, 25, 10
HEB = set(chr(c) for c in range(0x05D0, 0x05EB))
FINAL = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
rng = np.random.default_rng(0)


DIVINE = {"יי", "ייי", "ה'", "יהו", "יהוה", "אדני"}
FUNCTION = {"אשר", "כי", "כיא", "את", "אל", "על", "אם", "לא", "לוא", "כל", "כול",
            "מן", "הוא", "היא", "אני", "אתה", "הם", "זה", "זאת", "לו", "בו", "עד",
            "גם", "או", "כן", "אך", "רק", "יש", "אין", "מה", "מי", "ולא", "ואת", "וכל", "כה"}


def norm(w):                                   # scholar-lenient: finals, matres, divine name
    w = "".join(FINAL.get(c, c) for c in w)
    if w in DIVINE:
        return "יהוה"
    if w == "כיא":
        return "כי"
    if len(w) > 2:
        inner = w[1:-1].replace("ו", "").replace("י", "")
        return w[0] + inner + w[-1]
    return w


def is_content(w):
    return len(w) >= 3 and w not in FUNCTION


A = use("etcbc/dss", silent="deep")
F, L = A.api.F, A.api.L


def winfo(w):
    signs = L.d(w, "sign")
    g = "".join(F.glyph.v(s) or "" for s in signs)
    recs = [F.rec.v(s) for s in signs]
    return g, (bool(signs) and all(r == 1 for r in recs)), (bool(signs) and all(r != 1 for r in recs))


scrolls = {}
for w in F.otype.s("word"):
    if F.biblical.v(w):
        continue
    sc = L.u(w, "scroll")
    if sc:
        scrolls.setdefault(sc[0], []).append(winfo(w))

items = []
for ws in scrolls.values():
    for i, (g, fr, pr) in enumerate(ws):
        if not fr or len(g) < 2 or any(c not in HEB for c in g):
            continue
        lo, hi = max(0, i - WINDOW), min(len(ws), i + WINDOW + 1)
        ctx, tpos, preserved = [], None, 0
        for k in range(lo, hi):
            gg, ffr, ppr = ws[k]
            if len(gg) >= 1 and all(c in HEB for c in gg):
                if k == i:
                    tpos = len(ctx)
                ctx.append(gg)
                if ppr and k != i:
                    preserved += 1
        if tpos is not None and preserved >= MIN_PRESERVED:
            items.append((ctx, tpos, ctx[tpos]))
sel = rng.choice(len(items), size=min(MAX_ITEMS, len(items)), replace=False)
items = [items[i] for i in sel]
print(f"items: {len(items)}\n")


def beam_words(logits, ps, tok):
    beams = [(0.0, [])]
    for p in ps:
        lp = torch.log_softmax(logits[p], -1)
        top = torch.topk(lp, TOPN)
        beams = sorted([(s + v, seq + [i]) for s, seq in beams
                        for i, v in zip(top.indices.tolist(), top.values.tolist())],
                       key=lambda x: -x[0])[:BEAM]
    out = []
    for _, seq in beams:
        w = tok.decode(seq).replace(" ", "").replace("##", "")
        if w not in out:
            out.append(w)
        if len(out) >= K:
            break
    return out


for repo, nice in [("dicta-il/MsBERT", "MsBERT base"), ("dicta-il/BEREL", "BEREL base")]:
    tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(repo).eval()
    UNK = tok.unk_token_id
    oov = single = subtok_sum = 0
    ex1 = ex10 = lo1 = lo10 = n = 0
    cn = c_ex1 = c_ex10 = c_no1 = c_no10 = 0     # content-word only
    cf_ex1 = cf_ex10 = cf_no1 = cf_no10 = 0     # content-word filtered by len >= 3
    examples = []
    for ctx, tpos, gold in items:
        gt = tok(gold, add_special_tokens=False)["input_ids"]
        subtok_sum += len(gt)
        if UNK in gt:
            oov += 1
        if len(gt) == 1:
            single += 1
        enc = tok(ctx, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
        wmap = {}
        for pos, wid in enumerate(enc.word_ids(0)):
            if wid is not None:
                wmap.setdefault(wid, []).append(pos)
        ps = wmap.get(tpos)
        if not ps:
            continue
        ids = enc["input_ids"][0].clone()
        for p in ps:
            ids[p] = tok.mask_token_id
        with torch.no_grad():
            logits = model(ids.unsqueeze(0)).logits[0]
        ranked = beam_words(logits, ps, tok)
        n += 1
        e1 = bool(ranked) and gold == ranked[0]
        e10 = gold in ranked
        ng = norm(gold)
        n1 = bool(ranked) and norm(ranked[0]) == ng
        n10 = any(norm(r) == ng for r in ranked)
        ex1 += e1; ex10 += e10; lo1 += n1; lo10 += n10
        if is_content(gold):
            cn += 1; c_ex1 += e1; c_ex10 += e10; c_no1 += n1; c_no10 += n10
            # filtered by length >= 3
            ranked_filt = [w for w in ranked if len(w) >= 3]
            f_e1 = bool(ranked_filt) and gold == ranked_filt[0]
            f_e10 = gold in ranked_filt
            f_n1 = bool(ranked_filt) and norm(ranked_filt[0]) == ng
            f_n10 = any(norm(r) == ng for r in ranked_filt)
            cf_ex1 += f_e1; cf_ex10 += f_e10; cf_no1 += f_n1; cf_no10 += f_n10
        if len(examples) < 12:
            examples.append((gold, ranked[:5]))
    print(f"### {nice}")
    print(f"  gold OOV(→UNK): {oov/len(items)*100:.0f}%   single-token gold: {single/len(items)*100:.0f}%"
          f"   avg subtokens/gold: {subtok_sum/len(items):.2f}")
    print(f"  ALL words     EXACT top1={ex1/n*100:.1f} top10={ex10/n*100:.1f} | "
          f"NORM top1={lo1/n*100:.1f} top10={lo10/n*100:.1f}   (n={n})")
    print(f"  CONTENT (raw) EXACT top1={c_ex1/cn*100:.1f} top10={c_ex10/cn*100:.1f} | "
          f"NORM top1={c_no1/cn*100:.1f} top10={c_no10/cn*100:.1f}   (n={cn})")
    print(f"  CONTENT (filt)EXACT top1={cf_ex1/cn*100:.1f} top10={cf_ex10/cn*100:.1f} | "
          f"NORM top1={cf_no1/cn*100:.1f} top10={cf_no10/cn*100:.1f}   (n={cn})")
    print("  examples (gold | top-5):")
    for gold, top5 in examples:
        hit = "EXACT" if top5 and gold == top5[0] else ("in5" if gold in top5 else
              ("~norm" if any(norm(t) == norm(gold) for t in top5) else ""))
        print(f"    {gold:12s} | {'  '.join(top5)}   {hit}")
    print()
