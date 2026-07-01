"""Compatibility wrapper for the shared DSS split utilities."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.dss_split import SECT, load_split


if __name__ == "__main__":
    tr, te, bc = load_split()

    def sect_n(rows):
        return sum(1 for row in rows if SECT.get(row["section"]) == "sect")

    print(f"train nonbib chunks: {len(tr)} ({sect_n(tr)} sectarian)")
    print(f"test  nonbib chunks: {len(te)} ({sect_n(te)} sectarian)")
    print(f"bib contrast chunks: {len(bc)}")
    print(f"train words ~ {sum(len(row['text'].split()) for row in tr):,}")
