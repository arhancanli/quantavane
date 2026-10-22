import json
import os
import random
import tempfile
import unittest

from engine import claims, problems, training
from engine.archive import Archive
from engine.loop import evaluate
from engine.problems import CoveringDesign, Record, Verdict, retarget, seed_cover

FANO = [[1, 2, 3], [1, 4, 5], [1, 6, 7], [2, 4, 6], [2, 5, 7], [3, 4, 7], [3, 5, 6]]


def design(v, k, t, record, start=None):
    return CoveringDesign(v, k, t, Record(record, "test", "test"), 1, start)


class Covering(unittest.TestCase):
    def test_fano_plane_covers_every_pair(self):
        v = design(7, 3, 2, 7).check(FANO)
        self.assertTrue(v.valid, v.note); self.assertEqual(v.score, 7)

    def test_missing_pair_is_rejected_with_an_example(self):
        v = design(7, 3, 2, 7).check(FANO[:-1])
        self.assertFalse(v.valid); self.assertIn("not covered", v.note)

    def test_rejects_bad_blocks(self):
        p = design(7, 3, 2, 7)
        for bad in [FANO + [[1, 2]], FANO + [[1, 1, 2]], FANO + [[0, 1, 2]], FANO + [[1, 2, 8]],
                    FANO + [[1, 2, 3.0]], FANO + [[True, 2, 3]], "x", []]:
            self.assertFalse(p.check(bad).valid, bad)

    def test_checker_agrees_with_independent_check(self):
        rng = random.Random(5)
        for _ in range(40):
            v, k, t = rng.choice([(8, 4, 2), (9, 5, 3), (10, 5, 2), (11, 6, 3)])
            blocks = [sorted(rng.sample(range(1, v + 1), k)) for _ in range(rng.randint(3, 25))]
            self.assertEqual(design(v, k, t, 99).check(blocks).valid, claims.independent_check(blocks, v, k, t))

    def test_record_coverings_from_the_repository_are_valid(self):
        targets = {**problems.PROBLEMS, **problems.HIDDEN}
        names = [n for n in targets if n.startswith("cover_")]
        if not names:
            self.skipTest("covering data not downloaded")
        for name in names:
            p = targets[name][0]
            with open(p.start) as f:
                blocks = json.load(f)
            v = p.check(blocks)
            self.assertTrue(v.valid, f"{name}: {v.note}"); self.assertEqual(v.score, p.record.value)
            self.assertFalse(p.check(blocks[1:]).valid, f"{name} minus a block should not be a covering")

    def test_retarget_rewrites_instance_and_budget(self):
        p = design(9, 5, 3, 12)
        out = retarget(seed_cover(design(31, 19, 5, 40)), p, limit=40)
        self.assertIn("V, K, T = 9, 5, 3", out); self.assertIn("LIMIT = 40", out)
        self.assertIsNone(retarget("def construct():\n    return []\n", p))

    def test_both_seed_searches_return_valid_coverings_no_larger_than_their_start(self):
        p = design(11, 5, 3, 1000)
        p.timeout = 30
        start = training.weak_start(p, seed=1)
        self.assertTrue(p.check(start).valid)
        for template in (problems.SEED_COVER, problems.SEED_ORBITS):
            verdict, output, _ = evaluate(p, seed_cover(p, template), {"start.json": json.dumps(start)})
            self.assertTrue(verdict.valid, verdict.note)
            self.assertLess(verdict.score, len(start))


class ArchiveDirection(unittest.TestCase):
    def test_minimisation_keeps_the_smallest(self):
        with tempfile.TemporaryDirectory() as d:
            a = Archive(os.path.join(d, "a.db"), "min")
            for s in (5, 3, 4):
                a.add(island=0, program="p", verdict=Verdict(True, s), output=[s], proposer="t")
            self.assertEqual(a.best()["score"], 3); self.assertEqual(a.best_solution()["score"], 3)
            self.assertEqual(a.stats()["best"], 3)


class Training(unittest.TestCase):
    def registry(self, d):
        return training.Registry(os.path.join(d, "lab.db"), defaults=[("champ", "ollama", "m", 0.8, "champion"),
                                                                       ("chal", "ollama", "m", 1.1, "challenger")])

    def test_router_sends_traffic_to_the_winner(self):
        with tempfile.TemporaryDirectory() as d:
            reg = self.registry(d)
            for _ in range(60):
                reg.log_use("champ", "covering", "x", True, 0.05)
                reg.log_use("chal", "covering", "x", True, 1.0)
            router = training.Router(reg, random.Random(1), explore=0.0)
            picks = [router.choose("covering")["name"] for _ in range(50)]
            self.assertGreater(picks.count("chal"), 45)

    def test_promotion_needs_hidden_and_live_evidence_and_is_logged(self):
        with tempfile.TemporaryDirectory() as d:
            reg = self.registry(d)
            self.assertIsNone(training.review(reg, "covering", ["h"], log=lambda m: None))
            for name, gain, live in (("champ", 0.0, 0.1), ("chal", 0.05, 0.3)):
                for _ in range(8):
                    reg.db.execute("INSERT INTO hidden (at, model, problem, parent_score, score, valid, gain) VALUES (0,?,?,10,10,1,?)",
                                   (name, "h", gain))
                for _ in range(50):
                    reg.log_use(name, "covering", "x", True, live)
            self.assertEqual(training.review(reg, "covering", ["h"], log=lambda m: None), "chal")
            self.assertEqual(reg.champion()["name"], "chal")
            decisions = [r["decision"] for r in reg.db.execute("SELECT decision FROM decisions ORDER BY id")]
            self.assertEqual(decisions, ["not enough evidence", "promoted"])


class LabEndToEnd(unittest.TestCase):
    def test_lab_seeds_reruns_takes_proposals_and_files_a_claim(self):
        from engine.lab import Lab
        with tempfile.TemporaryDirectory() as d:
            p = design(11, 5, 3, 25, start=os.path.join(d, "start.json"))
            p.timeout = 20
            with open(p.start, "w") as f:
                json.dump(training.weak_start(p, seed=2), f)
            p.extra_seeds = [seed_cover(p, problems.SEED_ORBITS)]

            class Echo:  # stands in for a model: returns the seed program unchanged
                name = "echo"
                def propose(self, prompt):
                    return seed_cover(p), 0.0

            lab = Lab({p.name: (p, seed_cover(p))}, random.Random(3), root=os.path.join(d, "runs"),
                      claims_root=os.path.join(d, "claims"), log=lambda m: None)
            lab.proposer_for = lambda m: Echo()
            lab.run(0.012, cpu_workers=1, llm_workers=1)
            archive = lab.archive(p.name)
            proposers = {r["proposer"] for r in archive.db.execute("SELECT proposer FROM attempts")}
            self.assertTrue({"seed", "rerun"} <= proposers, proposers)
            self.assertEqual({r["island"] for r in archive.db.execute("SELECT island FROM attempts")}, {0, 1})
            self.assertLess(archive.best()["score"], 25)
            self.assertIsNotNone(claims.best_claimed(os.path.join(d, "claims"), p.name))
            self.assertGreater(lab.registry.db.execute("SELECT COUNT(*) FROM uses").fetchone()[0], 0)


class MemoryGuard(unittest.TestCase):
    def test_free_memory_is_a_percentage_or_none(self):
        from engine.lab import free_percent
        free = free_percent()
        self.assertTrue(free is None or 0 <= free <= 100, free)

    def test_lab_waits_while_memory_is_low_and_logs_once(self):
        from engine.lab import Lab
        import time
        with tempfile.TemporaryDirectory() as d:
            logs = []
            lab = Lab({}, random.Random(0), root=d, claims_root=d, log=logs.append)
            readings = iter([5.0, 5.0, 40.0])
            lab.memory = lambda: next(readings)
            original_sleep, time.sleep = time.sleep, lambda s: None
            try:
                lab.wait_for_memory(10, "re-runs", time.time() + 60)
            finally:
                time.sleep = original_sleep
            self.assertEqual(len(logs), 2, logs)
            self.assertIn("waiting", logs[0]); self.assertIn("resuming", logs[1])


class Claims(unittest.TestCase):
    def test_a_better_covering_becomes_a_packet_that_is_not_sent(self):
        with tempfile.TemporaryDirectory() as d:
            p = design(7, 3, 2, 8)
            folder, ok = claims.file_claim(p, 7, FANO, "def construct(): ...", "test", root=d)
            self.assertTrue(ok)
            with open(os.path.join(folder, "covering.txt")) as f:
                self.assertEqual(f.read().count("\n"), 7)
            with open(os.path.join(folder, "CLAIM.md")) as f:
                self.assertIn("draft, not sent", f.read())
            self.assertEqual(claims.best_claimed(d, p.name), 7)


if __name__ == "__main__":
    unittest.main()
