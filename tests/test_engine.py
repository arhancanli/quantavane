import os
import tempfile
import unittest

from engine import sandbox
from engine.archive import Archive
from engine.loop import evaluate, evolve
from engine.problems import PROBLEMS, CirclePacking, Record, Verdict
from engine.proposers import BudgetExceeded, Ledger, extract_program

CP = CirclePacking(2, Record(0.5, "test", "test"))


class Sandbox(unittest.TestCase):
    def test_returns_json_output(self):
        r = sandbox.run("import numpy as np\ndef construct():\n    return np.arange(3)\n", timeout=20)
        self.assertTrue(r.ok, r.error); self.assertEqual(r.output, [0, 1, 2])

    def test_blocks_network(self):
        r = sandbox.run("import socket\ndef construct():\n    socket.create_connection(('1.1.1.1', 80), timeout=3)\n    return 1\n", timeout=20)
        self.assertFalse(r.ok)

    def test_blocks_writes_outside_its_folder(self):
        target = os.path.join(tempfile.gettempdir(), "..", "qv-escape-test")
        r = sandbox.run(f"def construct():\n    open({target!r}, 'w').write('x')\n    return 1\n", timeout=20)
        self.assertFalse(r.ok); self.assertFalse(os.path.exists(target))

    def test_cannot_read_private_files(self):
        secret = os.path.expanduser("~/.ssh/known_hosts")
        self.assertTrue(os.path.exists(secret))
        r = sandbox.run(f"def construct():\n    return open({secret!r}).read()\n", timeout=20)
        self.assertFalse(r.ok)
        self.assertIn("ermission", r.error)

    def test_network_failure_comes_from_the_sandbox(self):
        r = sandbox.run("import socket\ndef construct():\n    socket.create_connection(('1.1.1.1', 80), timeout=3)\n    return 1\n", timeout=20)
        self.assertIn("rror", r.error); self.assertNotIn("timed out", r.error)

    def test_cannot_start_other_programs_or_fork(self):
        for body in ["import subprocess\n    subprocess.run(['/bin/echo', 'x'])", "import os\n    os.fork()"]:
            r = sandbox.run(f"def construct():\n    {body}\n    return 1\n", timeout=20)
            self.assertFalse(r.ok, body)

    def test_output_flood_is_capped(self):
        r = sandbox.run("import sys\ndef construct():\n    sys.stdout.write('x' * 50_000_000)\n    return 1\n", timeout=20)
        self.assertFalse(r.ok)

    def test_cannot_read_other_system_files(self):
        r = sandbox.run("def construct():\n    return open('/etc/hosts').read()\n", timeout=20)
        self.assertFalse(r.ok)

    def test_kills_runaway_programs(self):
        r = sandbox.run("def construct():\n    while True: pass\n", timeout=2)
        self.assertFalse(r.ok); self.assertLess(r.seconds, 10)

    def test_cannot_forge_result_by_printing(self):
        r = sandbox.run(f"print('{sandbox.MARKER}[1]')\ndef construct():\n    raise SystemExit(3)\n", timeout=20)
        self.assertFalse(r.ok)


class Checker(unittest.TestCase):
    def test_accepts_touching_circles(self):
        self.assertEqual(CP.check([[0.25, 0.5, 0.25], [0.75, 0.5, 0.25]]).score, 0.5)

    def test_rejects_tiny_overlap(self):
        self.assertFalse(CP.check([[0.25, 0.5, 0.25], [0.75, 0.5, 0.25 + 1e-15]]).valid)

    def test_rejects_outside_square(self):
        self.assertFalse(CP.check([[0.2, 0.5, 0.25], [0.75, 0.5, 0.2]]).valid)

    def test_rejects_wrong_shape_nan_and_zero_radius(self):
        for bad in [[[0.25, 0.5, 0.25]], [[0.25, 0.5, float("nan")], [0.75, 0.5, 0.2]], [[0.25, 0.5, 0], [0.75, 0.5, 0.2]], "x", [[0.25, 0.5, 0.25], [0.75, 0.5, True]]]:
            self.assertFalse(CP.check(bad).valid, bad)


class Ledgers(unittest.TestCase):
    def test_refuses_spend_over_cap(self):
        l = Ledger(1.0); l.reserve(0.6); l.charge(0.6)
        with self.assertRaises(BudgetExceeded):
            l.reserve(0.5)
        Ledger(0).reserve(0.0)


class Extraction(unittest.TestCase):
    def test_takes_last_block_with_construct(self):
        text = "```python\nx=1\n```\nthen\n```python\ndef construct():\n    return 1\n```"
        self.assertIn("def construct", extract_program(text))
        self.assertIsNone(extract_program("no code"))


class FakeProposer:
    name = "fake"
    def __init__(self, programs): self.programs = list(programs)
    def propose(self, prompt):
        self.prompts = getattr(self, "prompts", []) + [prompt]
        return (self.programs.pop(0) if self.programs else None), 0.0


class Loop(unittest.TestCase):
    def test_seed_reaches_a_valid_baseline(self):
        problem, seed = PROBLEMS["circle_packing_26"]
        v, _, _ = evaluate(problem, seed)
        self.assertTrue(v.valid, v.note); self.assertGreater(v.score, 2.0)

    def test_keeps_improvements_and_logs_failures(self):
        seed = "def construct():\n    return [[0.25, 0.5, 0.2], [0.75, 0.5, 0.2]]\n"
        better = "def construct():\n    return [[0.25, 0.5, 0.25], [0.75, 0.5, 0.25]]\n"
        cheat = "def construct():\n    return [[0.25, 0.5, 0.3], [0.75, 0.5, 0.3]]\n"
        with tempfile.TemporaryDirectory() as d:
            a = Archive(os.path.join(d, "a.db"))
            best = evolve(CP, a, FakeProposer([cheat, better]), seed_program=seed, iterations=3, islands=1, workers=1, log=lambda m: None)
            self.assertEqual(best["score"], 0.5)
            s = a.stats(); self.assertEqual(s["n"], 4); self.assertEqual(s["ok"], 2)

    def test_migrated_copies_keep_their_solution(self):
        seed = "def construct():\n    return [[0.25, 0.5, 0.25], [0.75, 0.5, 0.25]]\n"
        with tempfile.TemporaryDirectory() as d:
            a = Archive(os.path.join(d, "a.db"))
            evolve(CP, a, FakeProposer([]), seed_program=seed, iterations=3, islands=2, workers=1, migrate_every=1, log=lambda m: None)
            rows = a.db.execute("SELECT output FROM attempts WHERE proposer = 'migration'").fetchall()
            self.assertTrue(rows)
            for r in rows:
                self.assertTrue(CP.check(__import__("json").loads(r["output"])).valid)


if __name__ == "__main__":
    unittest.main()
