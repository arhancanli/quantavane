import unittest

from engine import lotto


class LotterySearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lotto.build()

    def test_more_than_eight_points_per_subset(self):
        # two 10-point sets inside 11 points share at least 9, so one block covers every 10-subset with t = 9
        unc, _, blocks = lotto.anneal(11, 10, 9, 10, 1, 1, 1)
        self.assertEqual(unc, 0)
        self.assertEqual(lotto.check(11, 10, 9, 10, blocks), 0)
        self.assertTrue(lotto.independent_check(blocks, 11, 10, 9, 10))

    def test_a_failed_run_says_why(self):
        with self.assertRaisesRegex(RuntimeError, "bad sizes"):
            lotto.anneal(5, 6, 2, 3, 1, 1, 1)


if __name__ == "__main__":
    unittest.main()
