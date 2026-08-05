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
