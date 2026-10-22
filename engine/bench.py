"""Open Problems Benchmark v0 (internal): can a model write a program that builds a verified construction?

Each task is a covering design C(v,k,t) with a block target. The model writes a Python program with construct(); the
program runs in the sandbox under a CPU limit; the exact checker decides. Three tiers:
  easy    target = ceil(1.25 x best known)   (a sensible search gets there)
  medium  target = best known                (match the public record)
  hard    target = best known - 1            (beat the public record; only entries where Quantavane holds a covering
                                              with that many blocks, so every hard task is known to be solvable)
The hidden part is the reference covering for each task, kept out of the prompt. A task scores 1 when the checker
accepts a covering within the target. Results go to runs/bench/results.jsonl, one line per attempt.

    python -m engine bench --model gemma4-12b-32k --per-tier 10 --seconds 60
    python -m engine bench --report"""
import datetime
import glob
import json
import math
import os
import random

from .loop import evaluate
from .problems import CoveringDesign, Record
from .proposers import Ledger, OllamaProposer, extract_program

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

PROMPT = """Write a Python program that defines construct() and returns a ({v},{k},{t}) covering design with at most {target}
blocks: a list of blocks, each a list of {k} distinct integers from 1 to {v}, such that every {t}-element subset of
{{1..{v}}} lies inside at least one block. The program runs once, without network access, with numpy and scipy available,
and must return within {limit} seconds of CPU time (use time.process_time()). Return ONE complete program in a single
```python block."""


def tasks(per_tier=10, seed=0, max_subsets=60_000):
    """Deterministic task list. Reference coverings for the hard tier come from Quantavane's own verified finds."""
    rng = random.Random(seed)
    with open(os.path.join(HERE, "data", "coverdata.json")) as f:
        idx = json.load(f)
    ours = {}
    for path in glob.glob(os.path.join(ROOT, "runs", "native", "found", "C(*)_*.json")) + \
            glob.glob(os.path.join(ROOT, "runs", "triage", "found", "C(*)_*.json")):
        name = os.path.basename(path).split("_")[0]
        with open(path) as f:
            blocks = json.load(f)
        if name not in ours or len(blocks) < len(ours[name]):
            ours[name] = blocks
    pool = []
    for name, e in idx.items():
        v, k, t = (int(x) for x in name[2:-1].split(","))
        if math.comb(v, t) <= max_subsets and int(e["size"]) > int(e["low_bd"]) and k >= 4:
            pool.append((name, v, k, t, int(e["size"])))
    pool.sort()
    rng.shuffle(pool)
    out = []
    for name, v, k, t, rec in pool[:per_tier]:
        out.append({"id": f"easy-{name}", "tier": "easy", "v": v, "k": k, "t": t, "target": math.ceil(1.25 * rec), "record": rec})
    for name, v, k, t, rec in pool[per_tier:2 * per_tier]:
        out.append({"id": f"medium-{name}", "tier": "medium", "v": v, "k": k, "t": t, "target": rec, "record": rec})
    hard = sorted((n, b) for n, b in ours.items() if n in idx and len(b) < int(idx[n]["size"]))
    for name, blocks in hard[:per_tier]:
        v, k, t = (int(x) for x in name[2:-1].split(","))
        out.append({"id": f"hard-{name}", "tier": "hard", "v": v, "k": k, "t": t, "target": len(blocks),
                    "record": int(idx[name]["size"]), "reference_blocks": len(blocks)})
    return out


def problem_for(task, seconds):
    p = CoveringDesign(task["v"], task["k"], task["t"], Record(task["target"], "benchmark target", "Quantavane bench v0"),
                       0, None)
    p.timeout = int(seconds)
    return p


def attempt(task, program, seconds):
    """Run a program on a task: (score, note). Score 1 when the checker accepts a covering within the target."""
    p = problem_for(task, seconds)
    verdict, output, secs = evaluate(p, program, {})
    if not verdict.valid:
        return 0, verdict.note[:200], secs
    blocks = int(verdict.score)
    return (1 if blocks <= task["target"] else 0), f"valid covering with {blocks} blocks (target {task['target']})", secs


def run(model, per_tier=10, seconds=60, temperature=0.4, log=print):
    proposer = OllamaProposer(model, Ledger(0.0), temperature=temperature, name=f"{model}@{temperature}")
    os.makedirs(os.path.join(ROOT, "runs", "bench"), exist_ok=True)
    path = os.path.join(ROOT, "runs", "bench", "results.jsonl")
    for task in tasks(per_tier):
        prompt = PROMPT.format(v=task["v"], k=task["k"], t=task["t"], target=task["target"], limit=int(seconds) - 10)
        program, _ = proposer.propose(prompt)
        score, note, secs = attempt(task, program, seconds) if program else (0, "no program in reply", 0.0)
        rec = {"at": datetime.datetime.now().isoformat(timespec="seconds"), "model": proposer.name, "task": task["id"],
               "tier": task["tier"], "score": score, "note": note, "seconds": round(secs, 1)}
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"{proposer.name:22} {task['id']:24} {score}  {note}")


BASELINE = """import itertools, random, time
V, K, T, TARGET, LIMIT = {v}, {k}, {t}, {target}, {limit}

def construct():
    # Greedy: while some T-subset is uncovered, try 40 random blocks that contain it and keep the one covering the
    # most uncovered T-subsets. Restart with a new random order while time remains; keep the smallest covering.
    end = time.process_time() + LIMIT
    best = None
    while time.process_time() < end and (best is None or len(best) > TARGET):
        left = set(itertools.combinations(range(1, V + 1), T))
        blocks = []
        while left and time.process_time() < end:
            s = random.choice(tuple(left)) if len(left) < 5000 else next(iter(left))
            rest = [p for p in range(1, V + 1) if p not in s]
            cands = [sorted(s + tuple(random.sample(rest, K - T))) for _ in range(40)]
            score = lambda b: sum(1 for c in itertools.combinations(b, T) if c in left)
            b = max(cands, key=score)
            blocks.append(b)
            left.difference_update(itertools.combinations(b, T))
        if not left and (best is None or len(blocks) < len(best)):
            best = blocks
    return best or blocks
"""


def baseline(per_tier=10, seconds=60, log=print):
    """A plain greedy program, scored like any model: the floor every model should beat."""
    path = os.path.join(ROOT, "runs", "bench", "results.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for task in tasks(per_tier):
        program = BASELINE.format(v=task["v"], k=task["k"], t=task["t"], target=task["target"], limit=int(seconds) - 10)
        score, note, secs = attempt(task, program, seconds)
        rec = {"at": datetime.datetime.now().isoformat(timespec="seconds"), "model": "baseline:greedy",
               "task": task["id"], "tier": task["tier"], "score": score, "note": note, "seconds": round(secs, 1)}
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"{'baseline:greedy':22} {task['id']:24} {score}  {note}")


def report(path=None):
    path = path or os.path.join(ROOT, "runs", "bench", "results.jsonl")
    if not os.path.exists(path):
        print("no results yet"); return
    rows = [json.loads(line) for line in open(path)]
    models = sorted({r["model"] for r in rows})
    print(f"{'model':24} {'easy':>7} {'medium':>7} {'hard':>7}")
    for mdl in models:
        cells = []
        for tier in ("easy", "medium", "hard"):
            rs = [r for r in rows if r["model"] == mdl and r["tier"] == tier]
            cells.append(f"{sum(r['score'] for r in rs)}/{len(rs)}" if rs else "-")
        print(f"{mdl:24} {cells[0]:>7} {cells[1]:>7} {cells[2]:>7}")
