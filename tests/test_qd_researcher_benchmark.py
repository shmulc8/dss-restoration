import unittest
from collections import Counter

from experiments.run_qd_benchmark import (
    PhysicalConstraint,
    _visible_segments,
    build_constraint,
    bootstrap_top10_delta_ci,
    bootstrap_top10_ci,
    contiguous_context,
    context_window,
    join_clitics,
    parse_attributed_reading,
    rag_context_keys,
    rag_score,
    simulate_constraint,
    trace_penalty,
)
from curation.preserved_corpus import GAP_TOKEN


class QDEditorialParsingTests(unittest.TestCase):
    def test_soft_trace_cost_preserves_exact_matches_without_discarding(self):
        constraint = PhysicalConstraint(("אב",), True, False, 4, 4, 4)
        self.assertEqual(trace_penalty("אבגד", constraint), 0)
        self.assertGreater(trace_penalty("אדגד", constraint), 0)

    def test_simulated_trace_uses_shape_but_development_gold_letters(self):
        template = PhysicalConstraint(("אב",), False, True, 5, 5, 5)
        simulated = simulate_constraint("שלומ", template, 0)
        self.assertEqual(simulated.visible_segments, ("ומ",))
        self.assertTrue(simulated.anchored_right)

    def test_context_window_reindexes_target(self):
        item = {"context_words": ["א", "ב", "<TARGET>", "ג", "ד"], "target_index": 2}
        self.assertEqual(context_window(item, 1), (["ב", "<TARGET>", "ג"], 1))

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

    def test_constraint_labels_initial_reading_as_editor_length(self):
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
            parse_attributed_reading(base),
            ("האדם", "eligible"),
        )
        for reading in ("כי רוח", "אדם/איש", "{ו}אדם", "אד○ם"):
            normalized, reason = parse_attributed_reading({"reading": reading})
            self.assertIsNone(normalized, reason)

    def test_reading_parser_does_not_filter_against_test_condition(self):
        normalized, reason = parse_attributed_reading({"reading": "שלום"})
        self.assertEqual((normalized, reason), ("שלום", "eligible"))


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


class QDStatisticsTests(unittest.TestCase):
    def test_bootstrap_resamples_scroll_clusters(self):
        records = [
            {"siglum": "A", "rank": 0},
            {"siglum": "A", "rank": 0},
            {"siglum": "B", "rank": None},
        ]
        low, high = bootstrap_top10_ci(records, "rank", seed=7, samples=500)
        self.assertEqual(low, 0.0)
        self.assertEqual(high, 100.0)

    def test_paired_bootstrap_delta_resamples_scrolls(self):
        records = [
            {"siglum": "A", "left": None, "right": 0},
            {"siglum": "A", "left": None, "right": 0},
            {"siglum": "B", "left": None, "right": None},
        ]
        low, high = bootstrap_top10_delta_ci(
            records, "left", "right", seed=7, samples=500
        )
        self.assertEqual(low, 0.0)
        self.assertEqual(high, 100.0)


if __name__ == "__main__":
    unittest.main()
