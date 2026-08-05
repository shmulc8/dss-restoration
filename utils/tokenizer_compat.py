"""Tokenizer-shape helpers shared by training and evaluation.

Implements tokenizer-shape handling for WordPiece (MsBERT) and character-level
(TavBERT / DictaBERT-char) models.
"""

from transformers import AutoTokenizer, PreTrainedTokenizerFast


def load_tokenizer(model_name):
    """AutoTokenizer, falling back to PreTrainedTokenizerFast.

    Handles tokenizer_class compatibility across transformers versions.
    """
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
    except ValueError as e:
        if "does not exist or is not currently imported" not in str(e):
            raise
        tok = PreTrainedTokenizerFast.from_pretrained(model_name)
    assert tok.is_fast, "the pipeline relies on offset mappings, which need a fast tokenizer"
    return tok


def whitespace_token_id(tokenizer):
    """Id of the token that IS the bare space character, or None."""
    enc = tokenizer("א ב", add_special_tokens=False, return_offsets_mapping=True)
    for token_id, (start, end) in zip(enc["input_ids"], enc["offset_mapping"]):
        if (start, end) == (1, 2):
            return token_id
    return None


def emits_whitespace_tokens(tokenizer):
    """True if the tokenizer emits space as its own token."""
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
    """Word index per token, or None for special/whitespace tokens."""
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
    """Return tokens -> str, correct for the given tokenizer's shape."""
    if emits_whitespace_tokens(tokenizer):
        return lambda tokens: "".join(tokens)          # spaces are already tokens

    def wordpiece_detok(tokens):
        out = ""
        for token in tokens:
            if token.startswith("##"):
                out += token[2:]
            else:
                if out:
                    out += " "
                out += token
        return out

    return wordpiece_detok
