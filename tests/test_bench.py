import os
import unittest

from engine import bench

HAVE_DATA = os.path.exists(os.path.join(os.path.dirname(bench.__file__), "data", "coverdata.json"))

FANO = "def construct():\n    return [[1,2,3],[1,4,5],[1,6,7],[2,4,6],[2,5,7],[3,4,7],[3,5,6]]\n"


class Bench(unittest.TestCase):
    def test_a_valid_covering_within_the_target_scores_one_and_a_broken_one_scores_zero(self):
        task = {"id": "t", "tier": "easy", "v": 7, "k": 3, "t": 2, "target": 7, "record": 7}
        self.assertEqual(bench.attempt(task, FANO, 20)[0], 1)
        self.assertEqual(bench.attempt(dict(task, target=6), FANO, 20)[0], 0)  # valid but over the target
        broken = FANO.replace("[3,5,6]", "[3,5,7]")
        score, note, _ = bench.attempt(task, broken, 20)
        self.assertEqual(score, 0); self.assertIn("not covered", note)

    @unittest.skipUnless(HAVE_DATA, "needs engine/data/coverdata.json (scripts/ljcr_targets.py downloads it)")
    def test_tasks_are_deterministic_and_tiered(self):
        a, b = bench.tasks(3), bench.tasks(3)
        self.assertEqual([t["id"] for t in a], [t["id"] for t in b])
        self.assertTrue({"easy", "medium"} <= {t["tier"] for t in a})
        for t in a:
            if t["tier"] == "hard":
                self.assertEqual(t["target"], t["record"] - 1 if t["reference_blocks"] == t["record"] - 1 else t["target"])


if __name__ == "__main__":
    unittest.main()
