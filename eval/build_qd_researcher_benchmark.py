"""Build an attributed, held-out Qumran Digital reading-variant benchmark.

The Qumran Digital (QD) transcription pages expose a deliberately selected
subset of especially useful alternative readings inline.  For those words,
the public API returns the full bibliographic attribution for readings found
in editions and other scholarly publications.

This importer is intentionally conservative:

* only non-biblical DSS manuscripts are considered;
* only scrolls in our pre-existing held-out split are considered;
* only explicit readings are retained (never API-inferred/assumed readings);
* surrounding words containing square-bracket reconstructions are redacted;
* every row keeps the QD snapshot date, endpoint, bibliography, and page.

The result is a real but selected comparison corpus.  It is not a complete
catalogue of all QD variants, because QD exposes the complete list only
through one API request per word.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
API_BASE = "https://lexicon.qumran-digital.org/v1/qumran"
ALLOWED_HOST = "lexicon.qumran-digital.org"
DEFAULT_OUTPUT = ROOT / "data" / "derived" / "qd_researcher_variants.jsonl"
DEFAULT_MANIFEST = (
    ROOT / "data" / "derived" / "qd_researcher_variants_manifest.json"
)
SPLIT_MANIFEST = ROOT / "data" / "derived" / "preserved_nonbib_manifest.json"
HEBREW_RE = re.compile(r"[\u05d0-\u05ea]")
SQUARE_RECONSTRUCTION_RE = re.compile(r"\[[^\]]*[\u05d0-\u05ea][^\]]*\]")


def canonical_siglum(value: str) -> str:
    """Normalize harmless typography differences for split membership only."""
    return re.sub(r"[^0-9A-Za-z]", "", value).lower()


def hebrew_letters(value: str) -> str:
    return "".join(character for character in value if HEBREW_RE.fullmatch(character))


def safe_get_json(
    session: requests.Session,
    path: str,
    *,
    params: dict[str, str] | None = None,
    retries: int = 4,
) -> Any:
    """GET JSON from the fixed QD origin with bounded retries and response size."""
    if not path.startswith("/") or ".." in path:
        raise ValueError(f"Unsafe API path: {path!r}")
    url = f"{API_BASE}{path}"
    if urlparse(url).hostname != ALLOWED_HOST:
        raise ValueError("Refusing a request outside the fixed QD API host")
    for attempt in range(retries):
        try:
            response = session.get(url, params=params, timeout=(10, 45))
            response.raise_for_status()
            if len(response.content) > 10_000_000:
                raise RuntimeError(f"Oversized response from {url}")
            return response.json()
        except (requests.RequestException, ValueError) as error:
            if attempt + 1 == retries:
                raise RuntimeError(f"QD request failed: {url}") from error
            time.sleep(0.5 * (2**attempt))
    raise AssertionError("unreachable")


def safe_get_text(
    session: requests.Session,
    path: str,
    *,
    params: dict[str, str] | None = None,
    retries: int = 4,
) -> str:
    if not path.startswith("/") or ".." in path:
        raise ValueError(f"Unsafe API path: {path!r}")
    url = f"{API_BASE}{path}"
    if urlparse(url).hostname != ALLOWED_HOST:
        raise ValueError("Refusing a request outside the fixed QD API host")
    for attempt in range(retries):
        try:
            response = session.get(url, params=params, timeout=(10, 45))
            response.raise_for_status()
            if len(response.content) > 10_000_000:
                raise RuntimeError(f"Oversized response from {url}")
            return response.text
        except requests.RequestException as error:
            if attempt + 1 == retries:
                raise RuntimeError(f"QD request failed: {url}") from error
            time.sleep(0.5 * (2**attempt))
    raise AssertionError("unreachable")


class TranscriptionIndexParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_nonbib_item = False
        self.sigla: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "li":
            classes = set((values.get("class") or "").split())
            self.in_nonbib_item = "dss" in classes and "dss-biblical" not in classes
        elif tag == "a" and self.in_nonbib_item:
            href = unquote(values.get("href") or "")
            match = re.match(r"^/transcriptions/([^/]+)/", href)
            if match:
                self.sigla.append(match.group(1))

    def handle_endtag(self, tag: str) -> None:
        if tag == "li":
            self.in_nonbib_item = False


class ColumnParser(HTMLParser):
    """Extract primary line words and inline-alternative target IDs."""

    def __init__(self) -> None:
        super().__init__()
        self.in_primary_line = False
        self.line_depth = 0
        self.current_line_name = ""
        self.current_words: list[dict[str, Any]] = []
        self.lines: list[dict[str, Any]] = []
        self.alt_depth = 0
        self.word_depth = 0
        self.current_word: dict[str, Any] | None = None
        self.target_ids: set[int] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = set((values.get("class") or "").split())
        if tag == "tr" and "line-verse" in classes:
            self.in_primary_line = True
            self.line_depth = 1
            self.current_line_name = values.get("data-name") or ""
            self.current_words = []
            return
        if not self.in_primary_line:
            return
        if tag == "tr":
            self.line_depth += 1
        if tag == "span" and "alternative-reading" in classes:
            self.alt_depth = 1
            return
        if tag == "span" and self.alt_depth:
            self.alt_depth += 1
        if tag == "span" and "word" in classes:
            raw_id = values.get("data-word-id") or ""
            if raw_id.isdigit() and int(raw_id) > 0:
                word_id = int(raw_id)
                if self.alt_depth:
                    self.target_ids.add(word_id)
                else:
                    self.current_word = {"word_id": word_id, "raw": ""}
                    self.word_depth = 1
            return
        if tag == "span" and self.current_word is not None:
            self.word_depth += 1

    def handle_data(self, data: str) -> None:
        if self.current_word is not None:
            self.current_word["raw"] += data

    def handle_endtag(self, tag: str) -> None:
        if not self.in_primary_line:
            return
        if tag == "span" and self.current_word is not None:
            self.word_depth -= 1
            if self.word_depth == 0:
                self.current_words.append(self.current_word)
                self.current_word = None
        if tag == "span" and self.alt_depth:
            self.alt_depth -= 1
        if tag == "tr":
            self.line_depth -= 1
            if self.line_depth == 0:
                self.lines.append(
                    {
                        "line": self.current_line_name,
                        "words": self.current_words,
                    }
                )
                self.in_primary_line = False


def redact_context_word(raw: str) -> str:
    """Remove researcher restorations while retaining visibly preserved letters."""
    if (
        SQUARE_RECONSTRUCTION_RE.search(raw)
        or "--" in raw
        or raw.count("[") != raw.count("]")
    ):
        return "<GAP>"
    clean = hebrew_letters(raw)
    return clean or "<GAP>"


def fetch_scroll_targets(
    siglum: str,
    version: str,
    *,
    user_agent: str,
) -> list[dict[str, Any]]:
    session = requests.Session()
    session.headers["User-Agent"] = user_agent
    encoded = quote(siglum, safe="")
    columns = safe_get_json(
        session,
        f"/transcriptions/{encoded}/columns",
        params={"version": version},
    )
    parsed_columns: list[dict[str, Any]] = []
    for column in columns.get("columns", []):
        column_id = str(column.get("id", ""))
        if not re.fullmatch(r"c[0-9]+", column_id):
            continue
        html = safe_get_text(
            session,
            f"/transcriptions/{encoded}/column/{column_id}",
            params={"version": version},
        )
        parser = ColumnParser()
        parser.feed(html)
        flat_words = [
            {
                **word,
                "line": line["line"],
                "column": column.get("name", ""),
            }
            for line in parser.lines
            for word in line["words"]
        ]
        parsed_columns.append(
            {
                "column": column.get("name", ""),
                "words": flat_words,
                "target_ids": parser.target_ids,
            }
        )

    targets: list[dict[str, Any]] = []
    for column in parsed_columns:
        words = column["words"]
        index_by_id = {
            word["word_id"]: index for index, word in enumerate(words)
        }
        for word_id in sorted(column["target_ids"]):
            if word_id not in index_by_id:
                continue
            target_index = index_by_id[word_id]
            start = max(0, target_index - 40)
            end = min(len(words), target_index + 41)
            window = words[start:end]
            context = [redact_context_word(word["raw"]) for word in window]
            context[target_index - start] = "<TARGET>"
            target = words[target_index]
            targets.append(
                {
                    "siglum": siglum,
                    "column": target["column"],
                    "line": target["line"],
                    "word_id": word_id,
                    "qd_display_reading": target["raw"],
                    "context_words": context,
                    "target_index": target_index - start,
                }
            )
    return targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Contact Qumran Digital and replace the stored snapshot. "
            "Without this flag, existing outputs are reused without network I/O."
        ),
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("--workers must be between 1 and 8")
    if args.output.is_file() and args.manifest.is_file() and not args.refresh:
        stored = json.loads(args.manifest.read_text(encoding="utf-8"))
        print(
            "Reusing stored Qumran Digital snapshot "
            f"{stored['source']['snapshot']} "
            f"({stored['counts']['attributed_reading_rows']} rows)."
        )
        print("Pass --refresh explicitly to perform network collection again.")
        return

    user_agent = (
        "dss-restoration-research-benchmark/1.0 "
        "(academic reproducibility; contact via repository)"
    )
    session = requests.Session()
    session.headers["User-Agent"] = user_agent
    versions = safe_get_json(session, "/data-version-names")["versionDates"]
    version = max(versions)
    index_html = safe_get_text(
        session,
        "/transcription-index",
        params={"version": version},
    )
    index_parser = TranscriptionIndexParser()
    index_parser.feed(index_html)

    split_data = json.loads(SPLIT_MANIFEST.read_text())
    heldout_by_key = {
        canonical_siglum(siglum): siglum
        for siglum in split_data["scroll_splits"]["heldout"]
    }
    selected = [
        siglum
        for siglum in index_parser.sigla
        if canonical_siglum(siglum) in heldout_by_key
    ]
    print(
        f"QD snapshot={version} | non-biblical held-out scrolls={len(selected)}",
        flush=True,
    )

    all_targets: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                fetch_scroll_targets,
                siglum,
                version,
                user_agent=user_agent,
            ): siglum
            for siglum in selected
        }
        completed = 0
        for future in as_completed(futures):
            siglum = futures[future]
            all_targets.extend(future.result())
            completed += 1
            if completed % 20 == 0 or completed == len(futures):
                print(
                    f"columns: {completed}/{len(futures)} scrolls; "
                    f"inline targets={len(all_targets)}",
                    flush=True,
                )

    rows: list[dict[str, Any]] = []
    bibliography: dict[int, dict[str, Any]] = {}
    for index, target in enumerate(sorted(
        all_targets, key=lambda item: (item["siglum"], item["word_id"])
    )):
        payload = safe_get_json(
            session,
            f"/word-variants/{target['word_id']}",
            params={"database-date": version},
        )
        explicit_variants = [
            variant
            for variant in payload.get("variants", [])
            if not variant.get("readingIsAssumed", False)
            and variant.get("variantType") == "variant"
            and variant.get("bibliography")
            and hebrew_letters(str(variant.get("reading", "")))
        ]
        for variant in explicit_variants:
            source = variant["bibliography"]
            bibliography[int(source["bibliographyId"])] = source
            rows.append(
                {
                    **target,
                    "qd_snapshot": version,
                    "qd_initial_reading": payload.get("initialReading", ""),
                    "reading": variant["reading"],
                    "reading_hebrew": hebrew_letters(variant["reading"]),
                    "bibliography_id": source["bibliographyId"],
                    "bibliography_abbreviation": source[
                        "bibliographicAbbreviation"
                    ],
                    "bibliography_formatted": source[
                        "formattedBibliographicString"
                    ],
                    "page": variant.get("page"),
                    "comment": variant.get("comment"),
                    "reading_is_assumed": False,
                    "variant_type": variant["variantType"],
                    "provenance_endpoint": (
                        f"{API_BASE}/word-variants/{target['word_id']}"
                        f"?database-date={version}"
                    ),
                }
            )
        if (index + 1) % 50 == 0 or index + 1 == len(all_targets):
            print(
                f"variants: {index + 1}/{len(all_targets)} targets; "
                f"attributed rows={len(rows)}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": 1,
        "source": {
            "name": "Qumran Digital: Text und Lexikon",
            "api_base": API_BASE,
            "snapshot": version,
            "license": "CC BY-SA 4.0",
            "faq": "https://lexicon.qumran-digital.org/faq/v1/en/index.html",
        },
        "scope": {
            "non_biblical_only": True,
            "local_split": "heldout",
            "target_selection": "variants embedded inline by Qumran Digital",
            "explicit_readings_only": True,
            "assumed_readings_included": False,
            "surrounding_square_bracket_reconstructions": "redacted as <GAP>",
            "warning": (
                "QD describes its variant collection as working data that has "
                "not yet been checked extensively. This is a selected "
                "literature-agreement benchmark, not verified ground truth."
            ),
        },
        "counts": {
            "matched_heldout_scrolls": len(selected),
            "inline_variant_targets": len(all_targets),
            "attributed_reading_rows": len(rows),
            "bibliographic_sources": len(bibliography),
        },
        "bibliography": sorted(
            bibliography.values(),
            key=lambda item: item["bibliographicAbbreviation"],
        ),
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved -> {args.output}")
    print(f"saved -> {args.manifest}")


if __name__ == "__main__":
    main()
