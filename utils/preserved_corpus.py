"""Helpers for a reconstruction-free, non-biblical DSS corpus."""

import json
from pathlib import Path

from utils.paths import repo_path

GAP_TOKEN = "<GAP>"
HEBREW = set(chr(codepoint) for codepoint in range(0x05D0, 0x05EB))
DERIVED_DIR = repo_path("data", "derived")
CHUNKS_PATH = DERIVED_DIR / "preserved_nonbib_chunks.jsonl"
LACUNAE_PATH = DERIVED_DIR / "nonbib_lacunae.jsonl"
MANIFEST_PATH = DERIVED_DIR / "preserved_nonbib_manifest.json"


def hebrew_letters(value: str) -> str:
    return "".join(character for character in (value or "") if character in HEBREW)


def classify_word(F, L, word_node):
    """Return a preserved word, a redacted gap, or a structural separator."""
    signs = L.d(word_node, "sign")
    full = F.full.v(word_node) or ""
    preserved_letters = []
    visible_pattern = []
    reconstructed_chars = 0
    has_reconstruction = False
    has_modern_removal = False

    for sign in signs:
        sign_letters = hebrew_letters(F.glyph.v(sign) or "")
        if F.rec.v(sign) == 1:
            has_reconstruction = True
            reconstructed_chars += len(sign_letters)
            visible_pattern.extend("?" for _ in sign_letters)
        else:
            preserved_letters.extend(sign_letters)
            visible_pattern.extend(sign_letters)
        if F.rem.v(sign) == 1:
            has_modern_removal = True

    hash_unknowns = full.count("#")
    is_gap = has_reconstruction or has_modern_removal or hash_unknowns > 0
    clean_word = "".join(preserved_letters)
    if is_gap:
        visible_pattern.extend("?" for _ in range(hash_unknowns))
        return {
            "kind": "gap",
            "node": word_node,
            "pattern": "".join(visible_pattern) or "?",
            "missing_chars_estimate": reconstructed_chars + hash_unknowns or None,
            "basis": (
                "modern-reconstruction-signs"
                if has_reconstruction
                else "modern-removal-or-unknown-marker"
            ),
        }
    if clean_word:
        return {"kind": "word", "node": word_node, "token": clean_word}
    return {"kind": "separator", "node": word_node}


def load_jsonl(path: Path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_manifest(path: Path = MANIFEST_PATH):
    return json.loads(path.read_text())


def load_chunks(split: str, path: Path = CHUNKS_PATH):
    normalized = split.strip().lower()
    if normalized not in {"train", "dev", "heldout"}:
        raise ValueError(f"Unknown split {split!r}; expected train|dev|heldout")
    return [row for row in load_jsonl(path) if row["split"] == normalized]


def split_scrolls(split: str, path: Path = MANIFEST_PATH):
    manifest = load_manifest(path)
    normalized = split.strip().lower()
    try:
        return set(manifest["scroll_splits"][normalized])
    except KeyError as error:
        raise ValueError(f"Unknown split {split!r}; expected train|dev|heldout") from error
