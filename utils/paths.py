"""Path helpers anchored at the repository root."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)

