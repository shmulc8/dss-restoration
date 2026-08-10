"""Validate the derived DSS corpus, optionally against source Text-Fabric."""

import argparse
import os
import sys
from pathlib import Path

from tf.fabric import Fabric

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from curation.preserved_corpus import (
    GAP_TOKEN,
    HEBREW,
    LACUNAE_PATH,
    MANIFEST_PATH,
    classify_word,
    load_chunks,
    load_jsonl,
    load_manifest,
)

TF_DIR = Path(
    os.environ.get(
        "DSS_TF_DIR",
        "/Users/shmulc/text-fabric-data/github/ETCBC/dss/tf/2.0",
    )
)


def main(*, derived_only: bool = False):
    manifest = load_manifest(MANIFEST_PATH)
    split_sets = {
        split: set(scrolls) for split, scrolls in manifest["scroll_splits"].items()
    }
    assert not (split_sets["train"] & split_sets["dev"])
    assert not (split_sets["train"] & split_sets["heldout"])
    assert not (split_sets["dev"] & split_sets["heldout"])

    chunks = [
        row for split in ("train", "dev", "heldout") for row in load_chunks(split)
    ]
    for row in chunks:
        assert row["scroll"] in split_sets[row["split"]]
        tokens = row["text"].split()
        assert all(
            token == GAP_TOKEN
            or (token and all(character in HEBREW for character in token))
            for token in tokens
        )
        assert not any(marker in row["text"] for marker in ("[", "]", "#", "?"))
        assert row["n_gap_slots"] == tokens.count(GAP_TOKEN)
        assert row["n_preserved_words"] == len(tokens) - tokens.count(GAP_TOKEN)

    lacunae = load_jsonl(LACUNAE_PATH)
    for row in lacunae:
        assert row["scroll"] in split_sets[row["split"]]
        assert row["gap_word_count_estimate"] == len(row["source_word_nodes"])
        assert len(row["visible_patterns"]) == len(row["source_word_nodes"])
        for stored_pattern in row["visible_patterns"]:
            assert all(
                character in HEBREW or character == "?" for character in stored_pattern
            )

    if not derived_only:
        tf = Fabric(locations=str(TF_DIR), silent="deep")
        api = tf.load(
            "otype glyph full rec rem biblical scroll",
            silent="deep",
        )
        if api is False:
            raise RuntimeError(
                f"Could not load DSS Text-Fabric corpus from {TF_DIR}. "
                "Set DSS_TF_DIR or use --derived-only."
            )
        F, L = api.F, api.L
        for row in lacunae:
            for node, stored_pattern in zip(
                row["source_word_nodes"],
                row["visible_patterns"],
            ):
                assert not F.biblical.v(node)
                source_event = classify_word(F, L, node)
                assert source_event["kind"] == "gap"
                assert source_event["pattern"] == stored_pattern

    print("SUCCESS: train/dev/heldout scroll sets are disjoint")
    print("SUCCESS: all training tokens are preserved Hebrew or <GAP>")
    print(
        "SUCCESS: no brackets, unknown markers, or reconstructed letters "
        "occur in training text"
    )
    if derived_only:
        print("SUCCESS: every lacuna record passes checked-in schema invariants")
    else:
        print("SUCCESS: every lacuna record maps to non-biblical Text-Fabric gap nodes")
    print(
        f"validated {len(chunks):,} chunks and {len(lacunae):,} lacunae "
        f"across {sum(len(value) for value in split_sets.values()):,} scrolls"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--derived-only",
        action="store_true",
        help="validate checked-in artifacts without loading source Text-Fabric",
    )
    main(derived_only=parser.parse_args().derived_only)
