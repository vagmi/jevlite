#!/usr/bin/env python3
"""
test_primitives.py — tests for the module everything else renders through.

primitives.py is the contract: build_data.py, teacher.py, jev_lite.py and both
servers all format questions here. A change that shifts a prompt by one
character silently invalidates a trained adapter, and a change to `answer()`
breaks the wire protocol. Both failures are quiet, so they get tests.

Stdlib only, so it runs without a venv:

    python test_primitives.py          # or: pytest test_primitives.py
"""
import unittest

import primitives as P


class KindOf(unittest.TestCase):
    """Which primitive is this row?"""

    def test_explicit_type_wins(self):
        row = {"type": "choice", "options": ["yes", "no"]}
        self.assertEqual(P.kind_of(row), "choice")

    def test_ordered_means_score(self):
        self.assertEqual(P.kind_of({"ordered": True, "options": ["a", "b", "c"]}), "score")

    def test_yes_no_is_noul_in_either_order(self):
        self.assertEqual(P.kind_of({"options": ["yes", "no"]}), "noul")
        self.assertEqual(P.kind_of({"options": ["No", "Yes"]}), "noul")
        self.assertEqual(P.kind_of({"options": ["true", "false"]}), "noul")

    def test_two_options_that_are_not_yes_no_are_a_choice(self):
        self.assertEqual(P.kind_of({"options": ["negative", "positive"]}), "choice")

    def test_unknown_type_falls_back_to_shape(self):
        self.assertEqual(P.kind_of({"type": "bogus", "options": ["yes", "no"]}), "noul")


class Normalize(unittest.TestCase):
    """noul rows must come out true-first, meaning intact."""

    def test_reorders_no_yes_and_moves_the_answer_with_it(self):
        row = P.normalize({"options": ["No", "Yes"], "answer": 1,
                           "state": "s", "question": "q"})
        self.assertEqual(row["type"], "noul")
        self.assertEqual(row["options"], ["true", "false"])
        # answer pointed at "Yes"; "true" is now index 0
        self.assertEqual(row["answer"], 0)

    def test_permutes_soft_labels_too(self):
        row = P.normalize({"options": ["no", "yes"], "label": [0.3, 0.7],
                           "state": "s", "question": "q"})
        self.assertEqual(row["options"], ["true", "false"])
        self.assertEqual(row["label"], [0.7, 0.3])

    def test_remaps_criteria_keys_to_true_false(self):
        row = P.normalize({"options": ["No", "Yes"], "answer": 0,
                           "criteria": {"Yes": "it is", "No": "it is not"},
                           "state": "s", "question": "q"})
        self.assertEqual(row["criteria"], {"true": "it is", "false": "it is not"})

    def test_score_gets_ordered_stamped(self):
        row = P.normalize({"type": "score", "options": ["lo", "hi"],
                           "state": "s", "question": "q"})
        self.assertTrue(row["ordered"])

    def test_is_idempotent(self):
        once = P.normalize({"options": ["No", "Yes"], "answer": 1,
                            "state": "s", "question": "q"})
        self.assertEqual(P.normalize(once), once)

    def test_does_not_mutate_its_input(self):
        original = {"options": ["No", "Yes"], "answer": 1, "state": "s", "question": "q"}
        P.normalize(original)
        self.assertEqual(original["options"], ["No", "Yes"])

    def test_a_pair_that_only_looks_yes_no_stays_a_choice(self):
        row = P.normalize({"options": ["yes", "affirmative"],
                           "state": "s", "question": "q"})
        self.assertEqual(row["type"], "choice")
        self.assertEqual(row["options"], ["yes", "affirmative"])


class Rendering(unittest.TestCase):
    """The prompt text IS the contract with the trained weights."""

    CHOICE = {"type": "choice", "state": "Ticket #1", "question": "Which team?",
              "options": ["billing", "technical"],
              "criteria": {"billing": "Payments", "technical": "Bugs"}}

    def test_criteria_are_rendered_after_an_em_dash(self):
        self.assertEqual(P.option_lines(self.CHOICE),
                         "A. billing — Payments\nB. technical — Bugs")

    def test_options_without_criteria_render_bare(self):
        row = {"type": "choice", "options": ["a", "b"], "question": "q", "state": "s"}
        self.assertEqual(P.option_lines(row), "A. a\nB. b")

    def test_partial_criteria_describe_only_what_was_given(self):
        row = dict(self.CHOICE, criteria={"billing": "Payments"})
        self.assertEqual(P.option_lines(row), "A. billing — Payments\nB. technical")

    def test_explicit_option_order_overrides_the_row(self):
        # this is how jev_lite shuffles options without rewriting the row
        self.assertEqual(P.option_lines(self.CHOICE, ["technical", "billing"]),
                         "A. technical — Bugs\nB. billing — Payments")

    def test_score_header_announces_the_ordering(self):
        row = {"type": "score", "ordered": True, "state": "s", "question": "How bad?",
               "options": ["Calm", "Angry"]}
        self.assertIn("Levels, lowest to highest:", P.question_block(row))
        self.assertNotIn("Options:", P.question_block(row))

    def test_choice_header_is_options(self):
        self.assertIn("Options:", P.question_block(self.CHOICE))

    def test_score_criteria_as_a_parallel_list(self):
        row = {"type": "score", "ordered": True, "state": "s", "question": "q",
               "options": ["0", "1"], "criteria": ["none", "lots"]}
        self.assertEqual(P.option_lines(row), "A. 0 — none\nB. 1 — lots")

    def test_more_options_than_letters_is_refused(self):
        row = {"type": "choice", "state": "s", "question": "q",
               "options": [f"o{i}" for i in range(27)]}
        with self.assertRaises(ValueError):
            P.option_lines(row)

    def test_full_prompt_structure(self):
        prompt = P.build_prompt(self.CHOICE)
        self.assertTrue(prompt.startswith(P.PREFIX))
        self.assertIn("Ticket #1\n</state>", prompt)
        self.assertIn("Question: Which team?", prompt)
        self.assertTrue(prompt.endswith("Answer:"))


class Confidence(unittest.TestCase):
    """Confidence is P(the answer returned is right) — nothing else."""

    def test_choice_confidence_is_the_selected_probability(self):
        self.assertAlmostEqual(P.confidence("choice", [0.08, 0.85, 0.07]), 0.85)

    def test_score_confidence_is_the_mass_that_rounds_to_the_score(self):
        probs = [0.05, 0.30, 0.65]
        expected = sum(i * p for i, p in enumerate(probs))   # 1.6 -> rounds to level 2
        self.assertAlmostEqual(P.confidence("score", probs, expected), 0.65)

    def test_score_confidence_collects_both_levels_when_the_score_sits_between(self):
        probs = [0.0, 0.5, 0.5]                              # expected exactly 1.5
        got = P.confidence("score", probs, 1.5)
        self.assertAlmostEqual(got, 1.0)

    def test_a_uniform_choice_is_not_confident(self):
        self.assertAlmostEqual(P.confidence("choice", [0.25] * 4), 0.25)


class Answers(unittest.TestCase):
    """The wire shapes, checked against docs.typesafe.ai/api.md."""

    def test_noul_reports_the_probability_of_true_and_no_confidence(self):
        row = P.normalize({"type": "noul", "state": "s", "question": "q",
                           "options": ["true", "false"]})
        got = P.answer(row, [0.92, 0.08])
        self.assertEqual(got, {"type": "noul", "noul": 0.92})

    def test_choice_shape(self):
        row = {"type": "choice", "state": "s", "question": "q",
               "options": ["billing", "technical", "sales"]}
        got = P.answer(row, [0.08, 0.85, 0.07])
        self.assertEqual(got["type"], "choice")
        self.assertEqual(got["choice"], "technical")
        self.assertEqual(got["probabilities"],
                         {"billing": 0.08, "technical": 0.85, "sales": 0.07})
        self.assertAlmostEqual(got["confidence"], 0.85)

    def test_score_is_the_expected_level_with_a_positional_legend(self):
        # the worked example from the API reference: 0.05/0.30/0.65 -> 1.6
        row = {"type": "score", "ordered": True, "state": "s", "question": "q",
               "options": ["Calm", "Frustrated", "Very angry"]}
        got = P.answer(row, [0.05, 0.30, 0.65])
        self.assertEqual(got["type"], "score")
        self.assertAlmostEqual(got["score"], 1.6)
        self.assertEqual(got["legend"],
                         {"0": "Calm", "1": "Frustrated", "2": "Very angry"})
        self.assertEqual(got["probabilities"], {"0": 0.05, "1": 0.3, "2": 0.65})

    def test_score_and_probabilities_keys_agree(self):
        row = {"type": "score", "ordered": True, "state": "s", "question": "q",
               "options": ["a", "b", "c"]}
        got = P.answer(row, [0.2, 0.3, 0.5])
        self.assertEqual(set(got["legend"]), set(got["probabilities"]))
        self.assertAlmostEqual(got["score"],
                               sum(int(k) * v for k, v in got["probabilities"].items()),
                               places=3)

    def test_choice_probabilities_sum_to_one(self):
        row = {"type": "choice", "state": "s", "question": "q", "options": list("abcd")}
        got = P.answer(row, [0.4, 0.3, 0.2, 0.1])
        self.assertAlmostEqual(sum(got["probabilities"].values()), 1.0, places=3)


class Temper(unittest.TestCase):
    """Serving-time calibration must not change what the answer is."""

    PROBS = [0.7, 0.2, 0.1]

    def test_identity_at_one(self):
        self.assertEqual(P.temper(self.PROBS, 1.0), self.PROBS)

    def test_still_a_distribution(self):
        self.assertAlmostEqual(sum(P.temper(self.PROBS, 1.4)), 1.0, places=9)

    def test_above_one_flattens(self):
        self.assertLess(max(P.temper(self.PROBS, 1.4)), max(self.PROBS))

    def test_below_one_sharpens(self):
        self.assertGreater(max(P.temper(self.PROBS, 0.7)), max(self.PROBS))

    def test_ranking_is_preserved(self):
        for T in (0.5, 1.4, 3.0):
            out = P.temper(self.PROBS, T)
            self.assertEqual(sorted(range(3), key=out.__getitem__),
                             sorted(range(3), key=self.PROBS.__getitem__))

    def test_survives_a_zero_probability(self):
        out = P.temper([1.0, 0.0], 1.4)
        self.assertAlmostEqual(sum(out), 1.0, places=9)


class Entropy(unittest.TestCase):

    def test_certain_is_zero(self):
        self.assertAlmostEqual(P.normalized_entropy([1.0, 0.0, 0.0]), 0.0)

    def test_uniform_is_one(self):
        self.assertAlmostEqual(P.normalized_entropy([0.25] * 4), 1.0, places=9)

    def test_normalized_across_option_counts(self):
        # the point of dividing by log(n): uniform is 1.0 whatever n is
        self.assertAlmostEqual(P.normalized_entropy([0.5, 0.5]), 1.0, places=9)


class RoundTrip(unittest.TestCase):
    """A normalized row still means what it meant, all the way to an answer."""

    def test_a_yes_no_row_answers_true_when_yes_was_the_gold(self):
        row = P.normalize({"options": ["No", "Yes"], "answer": 1,
                           "state": "s", "question": "Is it?"})
        probs = [0.0] * len(row["options"])
        probs[row["answer"]] = 1.0
        self.assertEqual(P.answer(row, probs), {"type": "noul", "noul": 1.0})

    def test_every_option_appears_in_the_rendered_prompt(self):
        row = P.normalize({"type": "choice", "state": "s", "question": "q",
                           "options": ["alpha", "beta", "gamma"],
                           "criteria": {"alpha": "first", "beta": None}})
        prompt = P.build_prompt(row)
        for option in row["options"]:
            self.assertIn(option, prompt)
        self.assertIn("first", prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
