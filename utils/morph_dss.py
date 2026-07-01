"""Hebrew lemmatization helper using dicta-il/dictabert-lex."""
import os
import threading
from transformers import AutoModel, AutoTokenizer

_LEX_ID = "dicta-il/dictabert-lex"
_lock = threading.Lock()
_tok = _model = None
_BATCH = 512

def _load():
    global _tok, _model
    if _model is None:
        with _lock:
            if _model is None:
                _tok = AutoTokenizer.from_pretrained(_LEX_ID)
                _model = AutoModel.from_pretrained(_LEX_ID, trust_remote_code=True).eval()
    return _tok, _model

_cache = {}

def lemmas(words) -> list[str]:
    words = list(words)
    if not words:
        return []
    tok, model = _load()
    out = []
    # Find uncached words
    uncached = [w for w in words if w not in _cache]
    if uncached:
        for i in range(0, len(uncached), _BATCH):
            batch = uncached[i:i + _BATCH]
            preds = model.predict(batch, tok)
            for w, pred in zip(batch, preds):
                lem = None
                if pred:
                    first = pred[0]
                    lem = first[1] if isinstance(first, (list, tuple)) and len(first) > 1 else None
                _cache[w] = lem if lem and lem != "[BLANK]" else w
    
    return [_cache[w] for w in words]

def lemma(word: str) -> str:
    return lemmas([word])[0]

if __name__ == "__main__":
    print("Testing lemmatization...")
    test_words = ["מילים", "האש", "התוכנית", "לבית", "ובניהם"]
    res = lemmas(test_words)
    for w, l in zip(test_words, res):
        print(f"  {w} -> {l}")
