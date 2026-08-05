"""Unified CandidateGenerator Interface & Core Implementations.

Implements Decision Point #4 from UNIFICATION_DECISION_POINTS.md:
- Model-neutral CandidateGenerator interface.
- Supports both WordPiece (WordMLMGenerator) and Character-level MLMs (CharMLMGenerator).
- Natively implements Partial-Letters Conditioning (§6c / R2b) for char-level models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import re


@dataclass
class Candidate:
    """Represents a single restoration prediction candidate."""
    text: str
    score: float
    tokens: Optional[List[str]] = None
    token_ids: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CandidateGenerator(ABC):
    """Abstract base interface for model candidate generation."""

    @abstractmethod
    def generate_candidates(
        self,
        context_left: str,
        context_right: str,
        target_len: Optional[int] = None,
        partial_pattern: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Candidate]:
        """Generate top-k restoration candidates given context and optional constraints.
        
        Args:
            context_left: Surviving text preceding the lacuna.
            context_right: Surviving text following the lacuna.
            target_len: Optional target character length constraint.
            partial_pattern: Optional physical partial letter pattern, e.g. "סר⬚⬚ך".
            top_k: Number of candidate suggestions to return.
        """
        pass


class PartialLetterFilter:
    """Utilities for matching candidates against physical partial letter patterns."""

    @staticmethod
    def is_compatible(candidate_text: str, pattern: str, wildcard_char: str = "⬚") -> bool:
        """Check if candidate_text is physically compatible with a partial letter pattern.
        
        Example: candidate "סרכיך" is compatible with pattern "סר⬚⬚ך".
        """
        if len(candidate_text) != len(pattern):
            return False
        for c_char, p_char in zip(candidate_text, pattern):
            if p_char != wildcard_char and c_char != p_char:
                return False
        return True


class MockCandidateGenerator(CandidateGenerator):
    """Deterministic mock generator for testing pipeline infrastructure without GPU/models."""

    def __init__(self, mock_candidates: Optional[List[str]] = None):
        self.mock_candidates = mock_candidates or ["אמר", "דבר", "צוה", "קרא", "עשה"]

    def generate_candidates(
        self,
        context_left: str,
        context_right: str,
        target_len: Optional[int] = None,
        partial_pattern: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Candidate]:
        results = []
        for i, text in enumerate(self.mock_candidates):
            if target_len is not None and len(text) != target_len:
                continue
            if partial_pattern is not None and not PartialLetterFilter.is_compatible(text, partial_pattern):
                continue
            score = -1.0 * (i + 1)
            results.append(Candidate(text=text, score=score))
            if len(results) >= top_k:
                break
        return results
