import unittest
from collections import Counter

from eval.score_qd_researcher_benchmark import (
    PhysicalConstraint,
    _visible_segments,
    build_constraint,
    contiguous_context,
    join_clitics,
    parse_attributed_reading,
    rag_context_keys,
    rag_score,
)
from utils.preserved_corpus import GAP_TOKEN


class QDEditorialParsingTests(unittest.TestCase):
    def test_prefix_inside_lacuna_and_visible_suffix(self):
        segments, left, right = _visible_segments("מ]אדם")
        self.assertEqual(segments, ("אדם",))
        self.assertFalse(left)
        self.assertTrue(right)

    def test_visible_prefix_and_open_lacuna(self):
        segments, left, right = _visible_segments("מבין[")
        self.assertEqual(segments, ("מבין",))
        self.assertTrue(left)
        self.assertFalse(right)

    def test_unknown_visible_slots_split_middle_segment(self):
        segments, left, right = _visible_segments("]○בוכ○○[")
        self.assertEqual(segments, ("בוכ",))
        self.assertFalse(left)
        self.assertFalse(right)

    def test_constraint_uses_initial_reading_only_for_length(self):
        constraint, reason = build_constraint(
            {
                "qd_display_reading": "]להרוג",
                "qd_initial_reading": "ו]להרוג",
            }
        )
        self.assertEqual(reason, "eligible")
        self.assertEqual(constraint.estimated_length, 6)
        self.assertEqual(constraint.visible_segments, ("להרוג",))
        self.assertTrue(constraint.matches("ולהרוג", 1))
        self.assertTrue(constraint.matches("להרוג", 1))
        self.assertFalse(constraint.matches("להריג", 1))

    def test_reading_parser_rejects_concatenation_cases(self):
        constraint = PhysicalConstraint(("אדם",), False, True, 4, 4, 4)
        base = {"reading": "האדם"}
        self.assertEqual(
            parse_attributed_reading(base, constraint, 1),
            ("האדם", "eligible"),
        )
        for reading in ("כי רוח", "אדם/איש", "{ו}אדם", "אד○ם"):
            normalized, reason = parse_attributed_reading(
                {"reading": reading}, constraint, 1
            )
            self.assertIsNone(normalized, reason)


class PreservedRAGTests(unittest.TestCase):
    def test_clitics_join_without_crossing_gap(self):
        self.assertEqual(
            join_clitics(["ו", "ב", "בית", GAP_TOKEN, "ה", "עם"]),
            ["ובבית", GAP_TOKEN, "העם"],
        )

    def test_context_stops_at_gap(self):
        words = ["דבר", GAP_TOKEN, "אמת", "וטוב"]
        self.assertEqual(
            contiguous_context(words, 2),
            ((), ("וטוב",)),
        )

    def test_context_keys_require_one_visible_neighbor(self):
        self.assertEqual(rag_context_keys((), ()), [])
        self.assertEqual(rag_context_keys(("דבר",), ()), [(("דבר",), ())])
        self.assertEqual(
            rag_context_keys(("דבר",), ("טוב",)),
            [
                ((), ("טוב",)),
                (("דבר",), ()),
                (("דבר",), ("טוב",)),
            ],
        )

    def test_exact_context_retrieval_supports_matching_candidate(self):
        key = (("דבר",), ("טוב",))
        index = {key: Counter({"אמת": 3})}
        score, span, hits = rag_score(index, *key, "אמת")
        self.assertGreater(score, 0)
        self.assertEqual(span, 3)
        self.assertEqual(hits, 3)
        self.assertEqual(rag_score(index, *key, "שקר"), (0.0, 0, 0))


if __name__ == "__main__":
    unittest.main()
