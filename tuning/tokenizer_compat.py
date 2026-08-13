"""Tokenizer-shape helpers shared by training (finetune_msbert.ipynb) and
evaluation (baseline.ipynb / eval_utils.py).

MsBERT is WordPiece: whitespace is discarded, a word is several adjacent tokens,
continuation pieces carry "##". TavBERT is character-level: whitespace is a real
vocab token, and every character is its own token. Everything in this module
exists to keep the rest of the pipeline from caring which it is talking to.
"""

from transformers import AutoTokenizer, PreTrainedTokenizerFast


# Ordinary consonantal Hebrew, the only script this corpus contains. Any
# tokenizer this project can use must encode it without falling back to [UNK].
_FIDELITY_PROBE = "ויאמר אלוהים אל משה לאמר בני ישראל"


def _unk_fraction(tokenizer):
    """Share of [UNK]s when encoding plain Hebrew; 1.0 if it cannot encode at all.

    The signal that a tokenizer was assembled from the wrong files -- see
    load_tokenizer. Deliberately behavioural, like every other probe here.
    """
    try:
        ids = tokenizer(_FIDELITY_PROBE, add_special_tokens=False)["input_ids"]
    except Exception:
        return 1.0
    if not ids:
        return 1.0
    unk_id = tokenizer.unk_token_id
    if unk_id is None:
        return 0.0
    return sum(1 for i in ids if i == unk_id) / len(ids)


def load_tokenizer(model_name):
    """AutoTokenizer, falling back to PreTrainedTokenizerFast.

    Artifacts exported by transformers 5.x record "tokenizer_class":
    "TokenizersBackend", which does not exist in 4.48.0, so AutoTokenizer's class
    lookup raises. tokenizer.json itself is version-stable, so naming the fast
    class directly loads the identical tokenizer.

    The same fallback also repairs a *silent* failure that cost a full training
    run: dicta-il/dictabert-large-char's tokenizer_config.json declares
    "tokenizer_class": "BertTokenizer", and under some transformers versions
    AutoTokenizer honours that by building plain WordPiece from vocab.txt --
    discarding the Split pre-tokenizer and normalizer that live only in
    tokenizer.json. The vocab holds single characters and no "##" continuations,
    so every Hebrew word becomes one [UNK]. Nothing raises: masking, training and
    validation all proceed on a corpus of [UNK]s, where "predict the masked
    token" has a single answer, so val loss collapses to ~0 and probe accuracy to
    ~1. Loading tokenizer.json directly restores the intended tokenizer, and the
    closing assert refuses to hand back one that still mangles Hebrew rather than
    let a run produce meaningless numbers.
    """
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
    except ValueError as e:
        if "does not exist or is not currently imported" not in str(e):
            raise
        tok = None

    if tok is None or not tok.is_fast or _unk_fraction(tok) > 0:
        fallback = PreTrainedTokenizerFast.from_pretrained(model_name)
        if tok is not None:
            print(f"tokenizer_compat: AutoTokenizer returned {type(tok).__name__} "
                  f"({_unk_fraction(tok):.0%} [UNK] on Hebrew); "
                  "reloading from tokenizer.json instead")
        tok = fallback

    assert tok.is_fast, "the pipeline relies on offset mappings, which need a fast tokenizer"
    unk_share = _unk_fraction(tok)
    assert unk_share == 0, (
        f"{model_name}: tokenizer encodes {unk_share:.0%} of plain Hebrew as [UNK]. "
        "Training on this would silently optimise a one-class task. "
        "Check that the repo ships a tokenizer.json describing the real tokenizer."
    )
    return tok


def whitespace_token_id(tokenizer):
    """Id of the token that IS the bare space character, or None.

    None means a tokenizer that discards whitespace (MsBERT). A non-None id
    (TavBERT: 104) is the separator FixedWindowMaskingCollator must re-insert
    between words, because precompute_word_tokens drops whitespace tokens from
    its records. The offsets must equal (1, 2) exactly -- a token that merely
    *covers* the space (e.g. a metaspace piece spanning (1, 3)) is not a
    standalone separator, and such a tokenizer is better served by the
    WordPiece-shaped path.
    """
    enc = tokenizer("א ב", add_special_tokens=False, return_offsets_mapping=True)
    for token_id, (start, end) in zip(enc["input_ids"], enc["offset_mapping"]):
        if (start, end) == (1, 2):
            return token_id
    return None


def emits_whitespace_tokens(tokenizer):
    """True if the tokenizer emits the space as its own token.

    Behavioural probe, deliberately not a class-name or config check: it is the
    property the rest of this module actually depends on.
    """
    return whitespace_token_id(tokenizer) is not None


def word_char_spans(text):
    """[(start, end)] of whitespace-delimited words in `text`, in order."""
    spans, cursor = [], 0
    for chunk in text.split(" "):
        if chunk:
            spans.append((cursor, cursor + len(chunk)))
        cursor += len(chunk) + 1
    return spans


def offset_word_ids(text, offset_mapping):
    """Word index per token, or None for special/whitespace tokens.

    Replaces the fast tokenizer's native word_ids(). For MsBERT the two agree; for
    a character-level tokenizer, native word_ids() returns one id per *character*
    (spaces included), which turns whole-word masking into per-character masking.
    """
    spans = word_char_spans(text)
    out = []
    for start, end in offset_mapping:
        if start == end:                       # special token
            out.append(None)
            continue
        hit = next((i for i, (s, e) in enumerate(spans) if s <= start < e), None)
        out.append(hit)                        # None for a pure-whitespace token
    return out


def make_detokenizer(tokenizer):
    """Return tokens -> str, correct for the given tokenizer's shape.

    Not tokenizer.convert_tokens_to_string(): TavBERT's tokenizer.json carries no
    decoder, so that returns 'ו י א מ ר'.
    """
    if emits_whitespace_tokens(tokenizer):
        return lambda tokens: "".join(tokens)          # spaces are already tokens

    def wordpiece_detok(tokens):
        out = ""
        for token in tokens:
            if token.startswith("##"):
                out += token[2:]
            else:
                out += (" " if out else "") + token
        return out

    return wordpiece_detok
