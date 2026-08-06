"""Unified CandidateGenerator Interface & Core Implementations.

Implements Decision Point #4 from UNIFICATION_DECISION_POINTS.md:
- Model-neutral CandidateGenerator interface.
- Supports both WordPiece (WordMLMGenerator) and Character-level MLMs (CharMLMGenerator).
- Natively implements Partial-Letters Conditioning (§6c / R2b) for char-level models.
- LengthEnsembleCharMLMGenerator: Solves unknown-length multi-word lacuna restoration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple


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
    def is_compatible(
        candidate_text: str, pattern: str, wildcard_char: str = "⬚"
    ) -> bool:
        """Check if candidate_text is physically compatible with a partial letter pattern.

        Example: candidate "סרכיך" is compatible with pattern "סר⬚⬚ך".
        """
        if len(candidate_text) != len(pattern):
            return False
        for c_char, p_char in zip(candidate_text, pattern):
            if p_char != wildcard_char and c_char != p_char:
                return False
        return True


class EpigraphicStrokeFilter:
    """Point 4 Roadmap: Evaluates character stroke similarity across paleographically ambiguous Qumran hands."""

    STROKE_CONFUSION_GROUPS = [
        {"ר", "ד", "ו", "ן", "י"},  # Vertical / head stroke ambiguity
        {"ה", "ח", "ת"},            # Corner roof stroke ambiguity
        {"מ", "ס"},                # Loop closure ambiguity
        {"ב", "כ"},                # Lower base stroke ambiguity
    ]

    @classmethod
    def stroke_similarity(cls, c1: str, c2: str, wildcard_char: str = "⬚") -> float:
        """Compute paleographic stroke similarity between two characters."""
        if c1 == wildcard_char or c2 == wildcard_char:
            return 1.0
        if c1 == c2:
            return 1.0
        for group in cls.STROKE_CONFUSION_GROUPS:
            if c1 in group and c2 in group:
                return 0.85  # High partial stroke similarity
        return 0.0

    @classmethod
    def is_stroke_compatible(
        cls, candidate_text: str, pattern: str, min_similarity: float = 0.85, wildcard_char: str = "⬚"
    ) -> bool:
        """Check if candidate_text matches pattern under multispectral stroke similarity matrix."""
        if len(candidate_text) != len(pattern):
            return False
        for c_char, p_char in zip(candidate_text, pattern):
            if cls.stroke_similarity(c_char, p_char, wildcard_char) < min_similarity:
                return False
        return True


class SectarianIDFBooster:
    """Point 3 Roadmap: Applies dynamic IDF score boost for Qumran sectarian vocabulary."""

    SECTARIAN_VOCAB = {
        "סרך": 3.5, "משכיל": 3.0, "עצה": 2.5, "תמים": 2.5, "אביונים": 3.0,
        "אור": 2.0, "חושך": 2.0, "מורה": 3.0, "צדק": 2.5, "יחד": 2.5,
        "תעודה": 3.0, "ברית": 2.0, "סוד": 2.5, "מעשיהם": 2.0, "רוח": 2.0
    }

    @classmethod
    def get_boost(cls, candidate_text: str) -> float:
        """Calculate sectarian vocabulary IDF score boost for a candidate word."""
        for term, boost in cls.SECTARIAN_VOCAB.items():
            if term in candidate_text:
                return boost
        return 0.0


class LengthEnsembleCharMLMGenerator(CandidateGenerator):
    """Length-Ensemble Candidate Generator for Unknown-Length Multi-Word Lacunae.

    Evaluates a range of target character lengths [min_len, max_len] when exact gap length
    is unknown, beam-searches each length hypothesis, normalizes log-probabilities by length,
    and returns globally top-k candidates across lengths.
    """

    def __init__(
        self,
        generator: CandidateGenerator,
        min_len: int = 2,
        max_len: int = 15,
        length_penalty_power: float = 0.7,
    ):
        self.generator = generator
        self.min_len = min_len
        self.max_len = max_len
        self.length_penalty_power = length_penalty_power

    def generate_candidates(
        self,
        context_left: str,
        context_right: str,
        target_len: Optional[int] = None,
        partial_pattern: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Candidate]:
        if target_len is not None:
            # Fixed target length specified
            return self.generator.generate_candidates(
                context_left, context_right, target_len=target_len, partial_pattern=partial_pattern, top_k=top_k
            )

        # Unknown length: ensemble search across [min_len, max_len]
        all_candidates: List[Candidate] = []
        for L in range(self.min_len, self.max_len + 1):
            candidates_L = self.generator.generate_candidates(
                context_left, context_right, target_len=L, partial_pattern=partial_pattern, top_k=top_k
            )
            for c in candidates_L:
                # Apply length normalization score adjustment
                norm_score = c.score / (len(c.text) ** self.length_penalty_power)
                norm_candidate = Candidate(
                    text=c.text,
                    score=norm_score,
                    tokens=c.tokens,
                    token_ids=c.token_ids,
                    metadata={**c.metadata, "raw_score": c.score, "length_hypothesis": L},
                )
                all_candidates.append(norm_candidate)

        # Sort globally across all length hypotheses by normalized score descending
        all_candidates.sort(key=lambda c: c.score, reverse=True)
        return all_candidates[:top_k]


class MockCandidateGenerator(CandidateGenerator):
    """Deterministic mock generator for testing pipeline infrastructure without GPU/models."""

    def __init__(self, mock_candidates: Optional[List[str]] = None):
        self.mock_candidates = mock_candidates or [
            "אמר", "דבר", "צוה", "קרא", "עשה", "אל משה", "כי בלב", "מורה צדק"
        ]

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
            if partial_pattern is not None and not PartialLetterFilter.is_compatible(
                text, partial_pattern
            ):
                continue
            score = -1.0 * (i + 1)
            results.append(Candidate(text=text, score=score))
            if len(results) >= top_k:
                break
        return results
