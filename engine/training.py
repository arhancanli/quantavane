"""Training system v1: which model writes the next program, and which model is the champion.

Every model call is logged with what it achieved. The router sends each problem family to whichever model is winning
right now (Thompson sampling over recent results, with some traffic always kept for challengers). Hidden tests use
instances the evolution never sees: a model edits the family's best program, the edit runs on a hidden instance from a
deliberately loose start, and it is scored against the unedited program on the same start with the same CPU budget.
A challenger replaces the champion only after beating it on hidden tests and holding up in live use, and the same rule
brings the old champion back if it later wins. Everything is logged in runs/lab.db."""
import json
import math
import sqlite3
import time

import numpy as np

from .loop import evaluate
from .problems import better, binomials, retarget, unrank
from .proposers import build_prompt

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (name TEXT PRIMARY KEY, backend TEXT NOT NULL, model TEXT NOT NULL,
  temperature REAL NOT NULL, role TEXT NOT NULL, added_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS uses (id INTEGER PRIMARY KEY, at REAL NOT NULL, model TEXT NOT NULL, family TEXT NOT NULL,
  problem TEXT NOT NULL, valid INTEGER NOT NULL, reward REAL NOT NULL);
CREATE TABLE IF NOT EXISTS hidden (id INTEGER PRIMARY KEY, at REAL NOT NULL, model TEXT NOT NULL, problem TEXT NOT NULL,
  parent_score REAL, score REAL, valid INTEGER NOT NULL, gain REAL NOT NULL);
CREATE TABLE IF NOT EXISTS decisions (id INTEGER PRIMARY KEY, at REAL NOT NULL, family TEXT NOT NULL, champion TEXT,
  challenger TEXT, decision TEXT NOT NULL, evidence TEXT NOT NULL);
"""

# Free local configurations today; API models join the registry as the owner's credits arrive.
DEFAULT_MODELS = [("gemma4-12b@0.8", "ollama", "gemma4-12b-32k", 0.8, "champion"),
                  ("gemma4-12b@1.1", "ollama", "gemma4-12b-32k", 1.1, "challenger"),
                  ("gemma4-12b@0.4", "ollama", "gemma4-12b-32k", 0.4, "challenger")]


def reward(problem, verdict, parent_score):
    if not verdict.valid:
        return 0.0
    if better(problem, verdict.score, parent_score):
        return 1.0
    return 0.25 if verdict.score == parent_score else 0.05


class Registry:
    def __init__(self, path, defaults=DEFAULT_MODELS):
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        if not self.db.execute("SELECT 1 FROM models").fetchone():
            for m in defaults:
                self.add(*m)

    def add(self, name, backend, model, temperature, role="challenger"):
        self.db.execute("INSERT OR IGNORE INTO models VALUES (?,?,?,?,?,?)",
                        (name, backend, model, temperature, role, time.time()))

    def active(self):
        return self.db.execute("SELECT * FROM models WHERE role != 'retired' ORDER BY role = 'champion' DESC, name").fetchall()

    def champion(self):
        return self.db.execute("SELECT * FROM models WHERE role = 'champion'").fetchone()

    def log_use(self, model, family, problem, valid, value):
        self.db.execute("INSERT INTO uses (at, model, family, problem, valid, reward) VALUES (?,?,?,?,?,?)",
                        (time.time(), model, family, problem, int(valid), value))

    def live(self, model, family, window=300):
        r = self.db.execute("SELECT COUNT(*) n, COALESCE(SUM(reward), 0) w FROM (SELECT reward FROM uses WHERE model = ?"
                            " AND family = ? ORDER BY id DESC LIMIT ?)", (model, family, window)).fetchone()
        return r["n"], r["w"]

    def hidden_mean(self, model, family_problems, window=50):
        marks = ",".join("?" * len(family_problems)) or "''"
        r = self.db.execute(f"SELECT COUNT(*) n, AVG(gain) g FROM (SELECT gain FROM hidden WHERE model = ? AND problem IN"
                            f" ({marks}) ORDER BY id DESC LIMIT ?)", (model, *family_problems, window)).fetchone()
        return r["n"], r["g"] or 0.0


class Router:
    """Thompson sampling over each model's recent rewards in a family; challengers always get some traffic."""

    def __init__(self, registry, rng, explore=0.15):
        self.registry, self.rng, self.explore = registry, rng, explore

    def choose(self, family):
        models = self.registry.active()
        challengers = [m for m in models if m["role"] == "challenger"]
        if challengers and self.rng.random() < self.explore:
            return self.rng.choice(challengers)

        def draw(m):
            n, w = self.registry.live(m["name"], family)
            return self.rng.betavariate(1 + w, 1 + n - w)
        return max(models, key=draw)


def weak_start(p, seed=0):
    """A deliberately loose covering (greedy with random choices), so a better search shows up as fewer blocks."""
    rng = np.random.default_rng(seed)
    binom = binomials(p.v, p.t)
    pat = np.array(list(__import__("itertools").combinations(range(p.k), p.t)), dtype=np.int64)
    cols = np.arange(1, p.t + 1)
    covered, blocks = np.zeros(math.comb(p.v, p.t), dtype=bool), []
    while not covered.all():
        S = np.array(unrank(int(rng.choice(np.flatnonzero(~covered))), p.t, binom), dtype=np.int64)
        best = None
        for _ in range(6):
            blk = np.sort(np.concatenate([S, rng.permutation(np.setdiff1d(np.arange(p.v), S))[:p.k - p.t]]))
            r = binom[blk[pat], cols].sum(axis=1)
            gain = int((~covered[r]).sum())
            if best is None or gain > best[0]:
                best = (gain, blk, r)
        covered[best[2]] = True
        blocks.append([int(x) + 1 for x in best[1]])
    return blocks


def hidden_session(registry, family, parent_program, hidden, proposer_for, *, per_model=2, budget=120, rng, log=print):
    """Each active model edits parent_program once on each of per_model sampled hidden instances. The edit and the
    unedited program run on the same loose start with the same CPU budget; gain = (parent blocks - edit blocks) /
    parent blocks, and an invalid edit scores -0.1."""
    import copy
    for m in registry.active():
        for name in rng.sample(sorted(hidden), min(per_model, len(hidden))):
            p = copy.copy(hidden[name][0])
            p.timeout = budget
            start = {"start.json": json.dumps(weak_start(p, seed=rng.randrange(1 << 30)))}
            parent = retarget(parent_program, p, limit=budget - 15)
            pv, _, _ = evaluate(p, parent, start)
            if not pv.valid:
                log(f"hidden {name}: parent program invalid here ({pv.note[:80]}); skipped"); continue
            program, _ = proposer_for(m).propose(build_prompt(p, {"program": parent, "score": pv.score}, [], []))
            program = retarget(program, p, limit=budget - 15) if program else None
            cv = evaluate(p, program, start)[0] if program else None
            ok = bool(cv and cv.valid)
            gain = (pv.score - cv.score) / pv.score if ok else -0.1
            registry.db.execute("INSERT INTO hidden (at, model, problem, parent_score, score, valid, gain) VALUES (?,?,?,?,?,?,?)",
                                (time.time(), m["name"], name, pv.score, cv.score if ok else None, int(ok), gain))
            log(f"hidden {name} {m['name']}: parent {pv.score:g}, edit {cv.score if ok else 'invalid'} (gain {gain:+.3f})")


def review(registry, family, family_problems, *, min_hidden=6, min_live=40, margin=0.01, log=print):
    """Promote a challenger that beats the champion on hidden tests and is at least as good in live use."""
    champ = registry.champion()
    ch_h, ch_g = registry.hidden_mean(champ["name"], family_problems)
    ch_n, ch_w = registry.live(champ["name"], family)
    for m in registry.active():
        if m["role"] != "challenger":
            continue
        h, g = registry.hidden_mean(m["name"], family_problems)
        n, w = registry.live(m["name"], family)
        evidence = json.dumps({"hidden": [h, g], "live": [n, w], "champion_hidden": [ch_h, ch_g], "champion_live": [ch_n, ch_w]})
        if h < min_hidden or n < min_live or ch_h < min_hidden or ch_n < min_live:
            decision = "not enough evidence"
        elif g > ch_g + margin and w / n >= ch_w / ch_n:
            registry.db.execute("UPDATE models SET role = 'challenger' WHERE name = ?", (champ["name"],))
            registry.db.execute("UPDATE models SET role = 'champion' WHERE name = ?", (m["name"],))
            decision = "promoted"
        else:
            decision = "champion kept"
        registry.db.execute("INSERT INTO decisions (at, family, champion, challenger, decision, evidence) VALUES (?,?,?,?,?,?)",
                            (time.time(), family, champ["name"], m["name"], decision, evidence))
        log(f"review {family}: {m['name']} vs {champ['name']}: {decision}")
        if decision == "promoted":
            return m["name"]
    return None
