#!/usr/bin/env python3
"""Download and safely unpack the public Uzan et al. Qumran image data."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath


ASSETS = {
    "qumran_dataset_v1": (
        "1gQlAP77kFxZFZnKYfXAWtJCdhv4ALUlm",
        "79296087623dc94e795437dc7cbb6069fbdc0f8cd1809a6b344b5bf67f3704ef",
    ),
    "qumran_test_letters": (
        "14F-XuI5i4Sq9LytotwqxEsNMs-hACBhl",
        "398e8940e17e6f486ee0a430fa665195e51d6489f6282d3b134300c326c678c2",
    ),
}
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract_images(archive: Path, destination: Path, expected_root: str) -> int:
    """Extract image files only; deliberately ignore the release's pickle caches."""
    extracted = 0
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            if not member.isfile():
                continue
            relative = PurePosixPath(member.name)
            if (
                not relative.parts
                or relative.parts[0] != expected_root
                or ".." in relative.parts
                or relative.suffix.lower() not in ALLOWED_SUFFIXES
            ):
                continue
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(f"could not read {member.name}")
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as handle:
                handle.write(source.read())
            extracted += 1
    if not extracted:
        raise RuntimeError(f"no images extracted from {archive}")
    return extracted


def download(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name, (file_id, expected_hash) in ASSETS.items():
        archive = destination / f"{name}.tar.gz"
        if not archive.exists() or sha256(archive) != expected_hash:
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            request = urllib.request.Request(url, headers={"User-Agent": "dss-restoration/1"})
            partial = archive.with_suffix(archive.suffix + ".partial")
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
                while block := response.read(1024 * 1024):
                    handle.write(block)
            if sha256(partial) != expected_hash:
                partial.unlink(missing_ok=True)
                raise RuntimeError(f"SHA-256 mismatch for {name}")
            partial.replace(archive)
        count = safe_extract_images(archive, destination, name)
        print(f"{name}: verified {expected_hash}; extracted {count} images")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("scratch/qumran_vision"),
    )
    download(parser.parse_args().destination.resolve())


if __name__ == "__main__":
    main()
