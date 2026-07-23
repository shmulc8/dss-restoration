"""Build a non-biblical DSS corpus without modern editorial reconstructions.

Output has two complementary views:

1. ``preserved_nonbib_chunks.jsonl`` contains scroll-order training chunks.
   Reconstructed or otherwise missing words are represented only by ``<GAP>``.
2. ``nonbib_lacunae.jsonl`` retains lacuna-size estimates and visible-letter
   patterns, but never emits the modern editor's reconstructed letters.

The Text-Fabric ``rec`` feature is authoritative: a sign with ``rec == 1`` was
reconstructed by a modern editor. Modern removals and explicit ``#`` unknown
material are also excluded from preserved training text.
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from tf.fabric import Fabric

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.paths import repo_path
from utils.preserved_corpus import (
    CHUNKS_PATH,
    GAP_TOKEN,
    LACUNAE_PATH,
    MANIFEST_PATH,
    classify_word,
)

DEFAULT_TF_DIR = Path(
    os.environ.get(
        "DSS_TF_DIR",
        "/Users/shmulc/text-fabric-data/github/ETCBC/dss/tf/2.0",
    )
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf-dir", type=Path, default=DEFAULT_TF_DIR)
    parser.add_argument("--chunks-out", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--lacunae-out", type=Path, default=LACUNAE_PATH)
    parser.add_argument("--manifest-out", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--max-units", type=int, default=128)
    parser.add_argument("--min-preserved", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def split_names(scroll_names, seed):
    names = sorted(scroll_names)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(names))
    heldout_count = max(1, int(len(names) * 0.30))
    heldout = {names[index] for index in permutation[:heldout_count]}
    fit = [names[index] for index in permutation[heldout_count:]]
    dev_count = max(1, int(len(fit) * 0.20))
    dev = set(fit[:dev_count])
    train = set(fit[dev_count:])
    return {"train": train, "dev": dev, "heldout": heldout}


def chunk_events(events, split, scroll, max_units, min_preserved):
    chunks = []
    units = [
        GAP_TOKEN if event["kind"] == "gap" else event["token"]
        for event in events
        if event["kind"] in {"word", "gap"}
    ]
    for chunk_index, start in enumerate(range(0, len(units), max_units)):
        tokens = units[start:start + max_units]
        preserved_count = sum(token != GAP_TOKEN for token in tokens)
        if preserved_count < min_preserved:
            continue
        chunks.append(
            {
                "scroll": scroll,
                "split": split,
                "chunk_index": chunk_index,
                "text": " ".join(tokens),
                "n_units": len(tokens),
                "n_preserved_words": preserved_count,
                "n_gap_slots": sum(token == GAP_TOKEN for token in tokens),
            }
        )
    return chunks


def lacuna_records(events, split, scroll, context_size=20):
    records = []
    significant = [event for event in events if event["kind"] in {"word", "gap"}]
    index = 0
    while index < len(significant):
        if significant[index]["kind"] != "gap":
            index += 1
            continue
        end = index + 1
        while end < len(significant) and significant[end]["kind"] == "gap":
            end += 1
        run = significant[index:end]
        left = [
            event["token"]
            for event in significant[max(0, index - context_size):index]
            if event["kind"] == "word"
        ]
        right = [
            event["token"]
            for event in significant[end:end + context_size]
            if event["kind"] == "word"
        ]
        missing_char_values = [
            event["missing_chars_estimate"]
            for event in run
            if event["missing_chars_estimate"] is not None
        ]
        records.append(
            {
                "scroll": scroll,
                "split": split,
                "gap_index": len(records),
                "gap_word_count_estimate": len(run),
                "gap_word_count_basis": "source-word-segmentation-without-editorial-text",
                "missing_char_count_estimate": (
                    sum(missing_char_values) if missing_char_values else None
                ),
                "visible_patterns": [event["pattern"] for event in run],
                "left_context": left,
                "right_context": right,
                "source_word_nodes": [event["node"] for event in run],
                "estimate_sources": sorted({event["basis"] for event in run}),
            }
        )
        index = end
    return records


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    tf = Fabric(locations=str(args.tf_dir), silent="deep")
    api = tf.load(
        "otype glyph full rec rem biblical scroll",
        silent="deep",
    )
    if api is False:
        raise RuntimeError(f"Could not load DSS Text-Fabric corpus from {args.tf_dir}")
    F, L = api.F, api.L

    scroll_nodes = {}
    for scroll_node in F.otype.s("scroll"):
        nonbib_words = [
            word_node
            for word_node in L.d(scroll_node, "word")
            if not F.biblical.v(word_node)
        ]
        if nonbib_words:
            scroll_nodes[F.scroll.v(scroll_node)] = nonbib_words

    scroll_splits = split_names(scroll_nodes, args.seed)
    split_by_scroll = {
        scroll: split
        for split, scrolls in scroll_splits.items()
        for scroll in scrolls
    }

    chunks = []
    lacunae = []
    source_counts = Counter()
    for scroll, word_nodes in sorted(scroll_nodes.items()):
        split = split_by_scroll[scroll]
        events = []
        for word_node in word_nodes:
            event = classify_word(F, L, word_node)
            events.append(event)
            source_counts[event["kind"]] += 1
        chunks.extend(
            chunk_events(
                events,
                split,
                scroll,
                args.max_units,
                args.min_preserved,
            )
        )
        lacunae.extend(lacuna_records(events, split, scroll))

    # Hard validation: emitted training text contains no editorial syntax and
    # every corpus record is non-biblical by construction.
    forbidden = ("[", "]", "#", "?")
    for row in chunks:
        if any(marker in row["text"] for marker in forbidden):
            raise AssertionError(f"Editorial material leaked into chunk: {row}")
        if not set(row["text"].split()) <= ({GAP_TOKEN} | {
            token
            for token in row["text"].split()
            if token and all(character in set(chr(c) for c in range(0x05D0, 0x05EB)) for character in token)
        }):
            raise AssertionError(f"Non-Hebrew training token found: {row}")

    write_jsonl(args.chunks_out, chunks)
    write_jsonl(args.lacunae_out, lacunae)

    manifest = {
        "schema_version": 1,
        "source": {
            "corpus": "ETCBC/dss",
            "text_fabric_version": "2.0",
            "transcription": "Martin Abegg data files",
            "feature_creators": [
                "Martin G. Abegg Jr.",
                "James E. Bowley",
                "Edward M. Cook",
            ],
            "license": "CC BY-NC 4.0",
            "local_path_config": "DSS_TF_DIR",
        },
        "rules": {
            "non_biblical_only": True,
            "modern_reconstruction_text_emitted": False,
            "modern_reconstruction_feature": "rec == 1 on signs",
            "gap_placeholder": GAP_TOKEN,
            "gap_size_metadata_retained": True,
        },
        "parameters": {
            "seed": args.seed,
            "max_units": args.max_units,
            "min_preserved_words_per_chunk": args.min_preserved,
        },
        "scroll_splits": {
            split: sorted(scrolls) for split, scrolls in scroll_splits.items()
        },
        "counts": {
            "scrolls": len(scroll_nodes),
            "chunks": len(chunks),
            "lacunae": len(lacunae),
            "source_words": dict(source_counts),
            "chunks_by_split": dict(Counter(row["split"] for row in chunks)),
            "lacunae_by_split": dict(Counter(row["split"] for row in lacunae)),
            "preserved_words_by_split": dict(
                Counter(
                    {
                        split: sum(
                            row["n_preserved_words"]
                            for row in chunks
                            if row["split"] == split
                        )
                        for split in scroll_splits
                    }
                )
            ),
        },
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    manifest["sha256"] = {
        "chunks": sha256(args.chunks_out),
        "lacunae": sha256(args.lacunae_out),
    }
    args.manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print(f"wrote {args.chunks_out}")
    print(f"wrote {args.lacunae_out}")
    print(f"wrote {args.manifest_out}")


if __name__ == "__main__":
    main()
