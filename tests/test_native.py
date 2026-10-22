import io
import json
import os
import random
import tempfile
import unittest
from contextlib import redirect_stdout

from engine import claims, native

FANO = [[1, 2, 3], [1, 4, 5], [1, 6, 7], [2, 4, 6], [2, 5, 7], [3, 4, 7], [3, 5, 6]]


def row(name, record, lb=1):
    v, k, t = (int(x) for x in name[2:-1].split(","))
    return {"name": name, "v": v, "k": k, "t": t, "record": record, "lower_bound": lb, "holder": "test",
            "method": "test", "date": "2026-01-01"}


class NativeSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native.build()

    def test_finds_the_fano_plane_and_the_checkers_agree(self):
        unc, moves, blocks = native.anneal(7, 3, 2, 7, 5, 1)
        self.assertEqual(unc, 0)
        self.assertEqual(native.check(7, 3, 2, blocks), 0)
        self.assertTrue(claims.independent_check(blocks, 7, 3, 2))

    def test_weighted_mode_counts_match_the_checker_when_it_cannot_cover(self):
        unc, moves, blocks = native.anneal(7, 3, 2, 6, 1, 2, weighted=True)  # C(7,3,2) = 7, so 6 blocks must miss
        self.assertGreater(unc, 0)
        self.assertEqual(native.check(7, 3, 2, blocks), unc)


class NativeLoop(unittest.TestCase):
    def test_a_covering_below_the_record_becomes_a_packet_and_the_target_drops(self):
        def fake(v, k, t, b, seconds, seed, weighted=False):
            return (0, 1, [list(x) for x in FANO]) if b == 7 else (3, 1, [list(x) for x in FANO[:b]])

        with tempfile.TemporaryDirectory() as d:
            loop = native.Loop([row("C(7,3,2)", 8)], "test", root=os.path.join(d, "native"),
                               claims_root=os.path.join(d, "claims"), seconds=1, rng=random.Random(1),
                               log=lambda m: None, run=fake)
            rec = loop.attempt(loop.pick())
            self.assertIn("found", rec)
            self.assertEqual(loop.state["C(7,3,2)"]["ours"], 7)
            self.assertEqual(loop.state["C(7,3,2)"]["target"], 6)
            self.assertIsNotNone(claims.best_claimed(os.path.join(d, "claims"), "cover_7_3_2"))
            rec2 = loop.attempt(loop.pick())  # next attempt aims one block lower and fails honestly
            self.assertNotIn("found", rec2)
            self.assertEqual(rec2["exact_uncovered"], native.check(7, 3, 2, [list(x) for x in FANO[:6]]))
            with open(os.path.join(d, "native", "attempts.jsonl")) as f:
                self.assertEqual(len(f.readlines()), 2)
            out = io.StringIO()
            with redirect_stdout(out):
                native.progress(os.path.join(d, "native"), os.path.join(d, "claims"))
            self.assertIn("2", out.getvalue())

    def test_close_misses_are_retried_first_then_untried_targets_then_far_misses(self):
        with tempfile.TemporaryDirectory() as d:
            loop = native.Loop([row("C(7,3,2)", 8), row("C(8,3,2)", 12), row("C(9,3,2)", 14)], "test", root=d,
                               rng=random.Random(0), log=lambda m: None)
            loop.state["C(7,3,2)"].update(attempts=1, best_uncovered=1)
            loop.state["C(8,3,2)"].update(attempts=1, best_uncovered=40)
            self.assertEqual(loop.pick(), "C(7,3,2)")  # tried once, one subset short: retry soon
            self.assertEqual(loop.pick(), "C(9,3,2)")  # never tried
            self.assertEqual(loop.pick(), "C(8,3,2)")  # tried once, 40 short
            self.assertIsNone(loop.pick())  # all busy


if __name__ == "__main__":
    unittest.main()
