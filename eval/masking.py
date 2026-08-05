"""
Verbatim copy of the eval classes from experiments/baseline.ipynb:
MaskedExample, MaskingPolicy, RandomWordMaskingPolicy,
PercentageContentMaskingPolicy, SingleWordMaskingPolicy, PredictionPolicy,
SingleTokenPredictionPolicy, MultiSpanPredictionPolicy, Evaluator.

Do not edit the logic here independently of the baseline notebook — it is
the source of truth. This module exists so tuning/finetune_msbert.ipynb can
reuse the exact same masking/prediction/evaluation machinery.

tokenizer_compat.py is shared by both this module and baseline.ipynb, but is
not part of the verbatim-copy contract above -- it is imported by both, not
duplicated.
"""

from dataclasses import dataclass
from abc import ABC, abstractmethod

import hashlib
import random
import torch

from utils.tokenizer_compat import make_detokenizer, emits_whitespace_tokens


@dataclass
class MaskedExample:
    original_sentence: str
    masked_sentence: str

    masked_word_indices: list   # word positions in the original sentence, sorted
    gold_words: list            # original words, aligned with masked_word_indices

    gold_word_tokens: list      # per masked word: its tokenizer pieces (raw, "##" kept when the tokenizer uses it)
    gold_word_token_ids: list   # per masked word: the matching token ids

class MaskingPolicy(ABC):

    @abstractmethod
    def generate(self, sentence: str, uid=None) -> MaskedExample:
        pass

    def _rng_for(self, uid):
        """Deterministic per-sentence RNG.

        The instance-level self.rng is consumed in dataset order, so the words masked in a
        sentence depend on how many sentences preceded it -- two runs over different dataset
        variants mask different words in the same sentence, which makes runs unpaired and
        paired statistics invalid. Seeding from sentence identity makes masking a pure
        function of (seed, sentence).
        """
        if uid is None:
            return self.rng                      # legacy path, order-dependent
        digest = hashlib.sha1(f"{self.seed}|{uid}".encode()).hexdigest()
        return random.Random(int(digest[:16], 16))

    def _build_example(self, sentence, words, chosen):
        """
        Replace each chosen word with one [MASK] per gold token, so the
        model can gauge the length of the missing word. Spaces between
        words are preserved.
        """
        chosen = sorted(chosen)

        masked_words = words.copy()

        gold = []
        gold_tokens = []
        gold_token_ids = []

        for idx in chosen:
            word = words[idx]

            pieces = self.tokenizer.tokenize(word)

            if not pieces:
                pieces = [self.tokenizer.unk_token]

            piece_ids = self.tokenizer.convert_tokens_to_ids(pieces)

            gold.append(word)
            gold_tokens.append(pieces)
            gold_token_ids.append(piece_ids)

            # One [MASK] per gold token, joined with NO separator. Under WordPiece
            # the separator is irrelevant (whitespace is discarded); under a
            # character-level tokenizer a space is a real token and would split
            # every span, so _group_mask_spans would only ever see length-1 spans.
            masked_words[idx] = "".join(
                [self.tokenizer.mask_token] * len(pieces)
            )

        return MaskedExample(
            original_sentence=sentence,
            masked_sentence=" ".join(masked_words),
            masked_word_indices=chosen,
            gold_words=gold,
            gold_word_tokens=gold_tokens,
            gold_word_token_ids=gold_token_ids
        )

class RandomWordMaskingPolicy(MaskingPolicy):

    def __init__(
        self,
        tokenizer,
        mask_ratio=0.15,
        seed=42
    ):
        self.tokenizer = tokenizer

        # These two policies pair with SingleTokenPredictionPolicy, which can only
        # predict one token, so they keep only single-token words. Under a
        # character-level tokenizer that means single-*character* words, which
        # Hebrew sentences essentially never have -- generate() would return None
        # for every sentence and the caller would silently produce no rows.
        if emits_whitespace_tokens(tokenizer):
            raise ValueError(
                f"{type(self).__name__} requires a subword tokenizer; "
                f"{tokenizer.name_or_path} is character-level, where 'words that "
                "tokenize to a single token' means one-character words. Use "
                "PercentageContentMaskingPolicy with MultiSpanPredictionPolicy instead."
            )

        self.mask_ratio = mask_ratio
        self.seed = seed
        self.rng = random.Random(seed)

    def generate(self, sentence, uid=None):

        rng = self._rng_for(uid)

        words = sentence.split()

        eligible = []

        # only words represented by a single token
        for i, word in enumerate(words):

            pieces = self.tokenizer.tokenize(word)

            if len(pieces) == 1:
                eligible.append(i)

        if len(eligible) == 0:
            return None

        n_mask = max(
            1,
            round(len(eligible) * self.mask_ratio)
        )

        chosen = rng.sample(
            eligible,
            min(n_mask, len(eligible))
        )

        return self._build_example(sentence, words, chosen)

class PercentageContentMaskingPolicy(MaskingPolicy):
    def __init__(self, tokenizer, mask_ratio=0.15, span_concentration=0.5, seed=42):
        self.tokenizer = tokenizer
        self.mask_ratio = mask_ratio
        self.span_concentration = span_concentration
        self.seed = seed
        self.rng = random.Random(seed)

    def generate(self, sentence, uid=None):
        rng = self._rng_for(uid)

        words = sentence.split()

        if len(words) == 0:
            return None

        n = len(words)

        # Mask X% of words, at least 1 word.
        n_mask = max(1, round(n * self.mask_ratio))
        n_mask = min(n_mask, n)

        if self.span_concentration <= 0.0:
            chosen = rng.sample(range(n), n_mask)
        else:
            chosen = self._grow_spans(n, n_mask, rng)

        return self._build_example(sentence, words, chosen)

    def _grow_spans(self, n, n_mask, rng):
        """
        Pick n_mask masked positions one at a time. Each step extends an
        existing span (probability span_concentration) or starts a new one
        at a random unmasked word (probability 1 - span_concentration).
        Higher span_concentration -> fewer, longer spans.
        """
        masked = set()
        spans = []  # list of [start, end] inclusive, kept non-touching

        def add(idx):
            masked.add(idx)

            new_span = [idx, idx]
            merged = []

            for span in spans:
                if span[1] + 1 == new_span[0]:
                    new_span[0] = span[0]
                elif new_span[1] + 1 == span[0]:
                    new_span[1] = span[1]
                else:
                    merged.append(span)

            merged.append(new_span)
            spans[:] = merged

        def extend(span):
            left, right = span[0] - 1, span[1] + 1
            options = []

            if left >= 0 and left not in masked:
                options.append(left)
            if right < n and right not in masked:
                options.append(right)

            return rng.choice(options) if options else None

        for _ in range(n_mask):

            idx = None

            if spans and rng.random() < self.span_concentration:
                idx = extend(rng.choice(spans))

            if idx is None:
                candidates = [i for i in range(n) if i not in masked]
                idx = rng.choice(candidates)

            add(idx)

        return masked

class SingleWordMaskingPolicy(MaskingPolicy):

    def __init__(
        self,
        tokenizer,
        seed=42
    ):
        self.tokenizer = tokenizer

        # These two policies pair with SingleTokenPredictionPolicy, which can only
        # predict one token, so they keep only single-token words. Under a
        # character-level tokenizer that means single-*character* words, which
        # Hebrew sentences essentially never have -- generate() would return None
        # for every sentence and the caller would silently produce no rows.
        if emits_whitespace_tokens(tokenizer):
            raise ValueError(
                f"{type(self).__name__} requires a subword tokenizer; "
                f"{tokenizer.name_or_path} is character-level, where 'words that "
                "tokenize to a single token' means one-character words. Use "
                "PercentageContentMaskingPolicy with MultiSpanPredictionPolicy instead."
            )

        self.seed = seed
        self.rng = random.Random(seed)

    def generate(self, sentence, uid=None):

        rng = self._rng_for(uid)

        words = sentence.split()

        eligible = []

        # Only keep words that tokenize into a single token
        for i, word in enumerate(words):

            pieces = self.tokenizer.tokenize(word)

            if len(pieces) == 1:
                eligible.append(i)

        if len(eligible) == 0:
            return None

        # Choose exactly one word
        idx = rng.choice(eligible)

        return self._build_example(sentence, words, [idx])


class PredictionPolicy(ABC):

    def __init__(
        self,
        model,
        tokenizer,
        top_k=10,
        device="cpu"
    ):
        self.model = model.to(device)
        self.model.eval()

        self.tokenizer = tokenizer
        self.top_k = top_k
        self.device = device
        self.detokenize = make_detokenizer(tokenizer)

    @torch.no_grad()
    def _encode(self, text):
        """
        Tokenize a sentence and move it to the correct device.
        """
        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True
        ).to(self.device)

    @torch.no_grad()
    def _forward(self, encoded):
        """
        Run a forward pass through the model.
        """
        return self.model(**encoded).logits

    def _mask_positions(self, encoded):
        """
        Return the indices of every [MASK] token.
        """
        return (
            encoded.input_ids[0]
            == self.tokenizer.mask_token_id
        ).nonzero(as_tuple=True)[0]

    @torch.no_grad()
    def _topk_single_token_candidates(self, logits):
        """
        Top-k candidates for one [MASK] position, in the shared candidate format.
        """
        log_probs = torch.log_softmax(logits, dim=-1)

        top = torch.topk(
            log_probs,
            self.top_k
        )

        candidates = []

        for token_id, score in zip(top.indices.tolist(), top.values.tolist()):
            token = self.tokenizer.convert_ids_to_tokens([token_id])[0]

            candidates.append({
                "text": self.detokenize([token]),
                "tokens": [token],
                "token_ids": [token_id],
                "score": float(score)
            })

        return candidates

    @abstractmethod
    def predict(self, masked_example):
        """
        Return a list of span dicts:
        {span_index, mask_positions, span_length, mode, topk_phrases}
        """
        pass


class SingleTokenPredictionPolicy(PredictionPolicy):

    @torch.no_grad()
    def predict(self, masked_example):

        encoded = self._encode(
            masked_example.masked_sentence
        )

        logits = self._forward(encoded)

        results = []

        for span_index, pos in enumerate(self._mask_positions(encoded)):

            candidates = self._topk_single_token_candidates(
                logits[0, pos]
            )

            results.append({
                "span_index": span_index,
                "mask_positions": [int(pos.item())],
                "span_length": 1,
                "mode": "single_token",
                "topk_phrases": candidates
            })

        return results


class MultiSpanPredictionPolicy(PredictionPolicy):

    def __init__(
        self,
        model,
        tokenizer,
        top_k=10,
        beam_width=3,
        beam_depth=5,
        device="cpu"
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            top_k=top_k,
            device=device
        )

        self.beam_width = beam_width
        self.beam_depth = beam_depth

    def _group_mask_spans(self, mask_positions):

        spans = []

        if len(mask_positions) == 0:
            return spans

        current = [int(mask_positions[0].item())]

        for pos in mask_positions[1:]:
            pos_int = int(pos.item())

            if pos_int == current[-1] + 1:
                current.append(pos_int)
            else:
                spans.append(current)
                current = [pos_int]

        spans.append(current)

        return spans

    @torch.no_grad()
    def _decode_span_with_beam(
        self,
        encoded,
        span_positions
    ):
        # Beams live as a (n_beams, seq_len) tensor; one forward per position
        # covers every beam. Mathematically identical to expanding beams one
        # at a time (up to float associativity and topk tie order).
        beam_ids = encoded.input_ids.clone()                       # (1, L)
        beam_scores = torch.zeros(1, device=beam_ids.device)
        chosen_ids = [[]]                                          # python ids per beam
        attn = encoded.attention_mask                              # (1, L)

        search_steps = min(len(span_positions), self.beam_depth)

        for pos in span_positions[:search_steps]:
            n = beam_ids.size(0)
            logits = self.model(
                input_ids=beam_ids,
                attention_mask=attn.expand(n, -1),
            ).logits[:, pos]                                       # (n, V)
            log_probs = torch.log_softmax(logits, dim=-1)
            top = torch.topk(log_probs, self.beam_width)           # (n, W)

            total = beam_scores.unsqueeze(1) + top.values          # (n, W)
            best = torch.topk(total.flatten(), self.beam_width)
            parent = best.indices // self.beam_width               # (W,)
            token = top.indices.flatten()[best.indices]            # (W,)

            beam_ids = beam_ids[parent].clone()
            beam_ids[:, pos] = token
            beam_scores = best.values
            chosen_ids = [chosen_ids[p] + [t] for p, t in
                          zip(parent.tolist(), token.tolist())]

        # Span longer than beam_depth: finish remaining positions greedily,
        # still one batched forward per position across all beams.
        for pos in span_positions[search_steps:]:
            n = beam_ids.size(0)
            logits = self.model(
                input_ids=beam_ids,
                attention_mask=attn.expand(n, -1),
            ).logits[:, pos]
            log_probs = torch.log_softmax(logits, dim=-1)
            token_scores, token_ids = log_probs.max(dim=-1)        # (n,)
            beam_ids[:, pos] = token_ids
            beam_scores = beam_scores + token_scores
            chosen_ids = [c + [t] for c, t in zip(chosen_ids, token_ids.tolist())]

        phrase_candidates = []
        for ids, score in zip(chosen_ids, beam_scores.tolist()):
            tokens = self.tokenizer.convert_ids_to_tokens(ids)
            phrase_candidates.append({
                "text": self.detokenize(tokens),
                "tokens": tokens,
                "token_ids": ids,
                "score": float(score),
            })

        phrase_candidates.sort(key=lambda item: item["score"], reverse=True)
        return phrase_candidates[:self.top_k]

    @torch.no_grad()
    def predict(self, masked_example):

        encoded = self._encode(
            masked_example.masked_sentence
        )

        logits = self._forward(encoded)

        mask_positions = self._mask_positions(encoded)
        spans = self._group_mask_spans(mask_positions)

        results = []

        for span_index, span_positions in enumerate(spans):

            if len(span_positions) == 1:
                pos = span_positions[0]

                candidates = self._topk_single_token_candidates(
                    logits[0, pos]
                )

                mode = "single_token"
            else:
                candidates = self._decode_span_with_beam(
                    encoded,
                    span_positions
                )

                mode = "beam"

            results.append({
                "span_index": span_index,
                "mask_positions": span_positions,
                "span_length": len(span_positions),
                "mode": mode,
                "topk_phrases": candidates
            })

        return results


class Evaluator:

    def _gold_spans(self, masked_example, prediction_spans):
        """
        Align gold token slots with prediction spans: each masked word
        contributes one slot per gold token (in position order), and each
        prediction span consumes span_length slots.
        """
        slot_word_indices = []
        slot_tokens = []
        slot_token_ids = []

        for word_idx, word, pieces, piece_ids in zip(
            masked_example.masked_word_indices,
            masked_example.gold_words,
            masked_example.gold_word_tokens,
            masked_example.gold_word_token_ids
        ):
            for piece, piece_id in zip(pieces, piece_ids):
                slot_word_indices.append(word_idx)
                slot_tokens.append(piece)
                slot_token_ids.append(piece_id)

        spans = []
        cursor = 0

        for span in prediction_spans:
            take = span["span_length"]

            word_indices = []

            for idx in slot_word_indices[cursor:cursor + take]:
                if not word_indices or word_indices[-1] != idx:
                    word_indices.append(idx)

            gold_words = [
                masked_example.gold_words[
                    masked_example.masked_word_indices.index(idx)
                ]
                for idx in word_indices
            ]

            spans.append({
                "word_indices": word_indices,
                "gold_words": gold_words,
                "gold_text": " ".join(gold_words),
                "gold_tokens": slot_tokens[cursor:cursor + take],
                "gold_token_ids": slot_token_ids[cursor:cursor + take]
            })

            cursor += take

        return spans

    def _positional_matches(self, predicted_ids, gold_ids):
        """
        Number of slots where the predicted token id equals the gold id.
        """
        return sum(
            1
            for predicted, gold in zip(predicted_ids, gold_ids)
            if predicted == gold
        )

    def _levenshtein_similarity(self, predicted_ids, gold_ids):
        """
        1 - normalized token-level edit distance. Credits correct tokens
        even when they appear at shifted positions.
        """
        n, m = len(predicted_ids), len(gold_ids)

        if n == 0 and m == 0:
            return 1.0

        previous = list(range(m + 1))

        for i in range(1, n + 1):
            current = [i] + [0] * m

            for j in range(1, m + 1):
                cost = 0 if predicted_ids[i - 1] == gold_ids[j - 1] else 1

                current[j] = min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost
                )

            previous = current

        return 1.0 - previous[m] / max(n, m)

    def evaluate(
        self,
        masked_example,
        predictions
    ):

        rows = []

        gold_spans = self._gold_spans(masked_example, predictions)

        for gold, span in zip(gold_spans, predictions):

            candidates = [
                phrase["text"]
                for phrase in span["topk_phrases"]
            ]

            # Best partial score over the top-k candidates.
            n_tokens_correct = max(
                (
                    self._positional_matches(
                        phrase["token_ids"],
                        gold["gold_token_ids"]
                    )
                    for phrase in span["topk_phrases"]
                ),
                default=0
            )

            levenshtein_similarity = max(
                (
                    self._levenshtein_similarity(
                        phrase["token_ids"],
                        gold["gold_token_ids"]
                    )
                    for phrase in span["topk_phrases"]
                ),
                default=0.0
            )

            rows.append({

                "sentence":
                    masked_example.original_sentence,

                "masked_sentence":
                    masked_example.masked_sentence,

                "word_indices":
                    gold["word_indices"],

                "gold":
                    gold["gold_text"],

                "predictions":
                    candidates,

                "mode":
                    span["mode"],

                "span_length":
                    span["span_length"],

                "n_tokens_correct":
                    n_tokens_correct,

                "token_accuracy":
                    n_tokens_correct / span["span_length"],

                "levenshtein_similarity":
                    levenshtein_similarity,

                "correct":
                    n_tokens_correct == span["span_length"]

            })

        return rows
