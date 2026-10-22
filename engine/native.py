"""The native covering search and the always-on loop around it.

The compiled search (native/cover_anneal.c) does the heavy work; this module builds it, runs it, re-checks everything
it returns with the engine's exact checker, and keeps working through every open covering entry that nobody has
improved since the La Jolla Covering Repository froze. Entries that came close get more time, each attempt switches
between annealing and breakout weighting, and a covering with fewer blocks than the last known live record becomes a
claim packet (checked twice, never sent). When one is found, the target drops by one more block.

    python -m engine native --workers 8 --seconds 60     # around the clock
    python -m engine progress                              # verified progress per day"""
import datetime
import glob
import json
import math
import os
import random
import subprocess
import threading
import time

import numpy as np

from . import claims
from .problems import CoveringDesign, Record, uncovered

HERE = os.path.dirname(os.path.abspath(__file__))
BINARY = os.path.join(HERE, "native", "cover_anneal")
DATA = os.path.join(HERE, "data")


def build():
    src = BINARY + ".c"
    if not os.path.exists(BINARY) or os.path.getmtime(BINARY) < os.path.getmtime(src):
        subprocess.run(["clang", "-O3", "-mcpu=native", "-o", BINARY, src, "-lm"], check=True)


def anneal(v, k, t, b, seconds, seed, start=None, weighted=False):
    """(uncovered, moves, blocks) for the best state of one run with b blocks; uncovered 0 means a covering."""
    env = dict(os.environ, CYCLE=str(seconds), WEIGHTED="1" if weighted else "0")
    args = [BINARY, str(v), str(k), str(t), str(b), str(seconds), str(seed)] + ([start] if start else [])
    out = subprocess.run(args, capture_output=True, text=True, env=env, timeout=seconds * 3 + 60).stdout.split("\n")
    head = out[0].split()
    blocks = [[int(p) for p in line.split()] for line in out[1:] if line.strip()]
    return int(head[1]), int(head[3]), blocks


def check(v, k, t, blocks):
    """Exact count of uncovered t-subsets, from the engine's checker (0 means every t-subset is covered)."""
    rows = np.array([sorted(p - 1 for p in b) for b in blocks], dtype=np.int64)
    miss = uncovered(rows, v, k, t)
    return 0 if miss is None else miss[0]


def index():
    with open(os.path.join(DATA, "coverdata.json")) as f:
        return json.load(f)


def live_filter():
    """The newest live filter: which entries others improved after the freeze, read from coveringrepository.com."""
    paths = sorted(glob.glob(os.path.join(DATA, "live_filter_*.json")))
    if not paths:
        return None
    with open(paths[-1]) as f:
        return json.load(f)


def targets(max_subsets=400_000, min_k=2):
    """Open entries (record above the lower bound) that the live filter lists as unchanged since the freeze. Without a
    live filter, every open entry of the La Jolla final data (check the live table before any claim)."""
    idx, lf = index(), live_filter()
    keep = set(lf["unchanged"]) if lf else set(idx)
    rows = []
    for name in sorted(keep):
        e = idx.get(name)
        if e is None:
            continue
        v, k, t = (int(x) for x in name[2:-1].split(","))
        size, lb = int(e["size"]), int(e["low_bd"])
        if size > lb and k >= min_k and math.comb(v, t) <= max_subsets:
            imp = e["imps"][0]
            rows.append({"name": name, "v": v, "k": k, "t": t, "record": size, "lower_bound": lb,
                         "holder": imp[2].strip(), "method": imp[1].strip(), "date": imp[3][:10]})
    return rows, (lf or {}).get("checked", "no live filter: La Jolla final data only")


class Loop:
    """Always-on search over the targets. State lives in root/state.json, every attempt in root/attempts.jsonl."""

    def __init__(self, rows, checked, root="runs/native", claims_root="claims", seconds=60, rng=None, log=print,
                 run=anneal):
        self.rows = {r["name"]: r for r in rows}
        self.checked, self.root, self.claims_root, self.seconds = checked, root, claims_root, seconds
        self.rng, self.log, self.run = rng or random.Random(), log, run
        os.makedirs(os.path.join(root, "found"), exist_ok=True)
        self.state_path = os.path.join(root, "state.json")
        self.state = json.load(open(self.state_path)) if os.path.exists(self.state_path) else {}
        self.lock, self.busy = threading.Lock(), set()
        for name, r in self.rows.items():
            s = self.state.setdefault(name, {"attempts": 0, "best_uncovered": None, "ours": None})
            s["target"] = (s["ours"] or r["record"]) - 1  # one block below the best known (ours, once we have it)

    def priority(self, name):
        s = self.state[name]
        unc = 10 if s["best_uncovered"] is None else s["best_uncovered"]
        return s["attempts"] + math.log2(1 + unc) + self.rng.random() * 0.01

    def pick(self):
        with self.lock:
            free = [n for n in self.rows if n not in self.busy and self.state[n]["target"] >= 1]
            if not free:
                return None
            name = min(free, key=self.priority)
            self.busy.add(name)
            return name

    def attempt(self, name):
        r, s = self.rows[name], self.state[name]
        b, weighted = s["target"], s["attempts"] % 2 == 0
        close = s["best_uncovered"] is not None and s["best_uncovered"] <= 3
        seconds = self.seconds * (4 if close else 1)
        seed = self.rng.randrange(1, 2 ** 62)
        unc, moves, blocks = self.run(r["v"], r["k"], r["t"], b, seconds, seed, weighted=weighted)
        exact = check(r["v"], r["k"], r["t"], blocks) if blocks else None
        rec = {"at": time.time(), "name": name, "blocks": b, "method": "weighted" if weighted else "anneal",
               "seconds": seconds, "uncovered": unc, "exact_uncovered": exact, "moves": moves}
        with self.lock:
            s["attempts"] += 1
            if exact is not None and (s["best_uncovered"] is None or exact < s["best_uncovered"]):
                s["best_uncovered"] = exact
            if exact == 0 and len(blocks) == b and claims.independent_check(blocks, r["v"], r["k"], r["t"]):
                rec["found"] = self.found(r, blocks)
                s.update(ours=b, target=b - 1, best_uncovered=None)
            with open(os.path.join(self.root, "attempts.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
            tmp = self.state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.state, f)
            os.replace(tmp, self.state_path)
            self.busy.discard(name)
        if exact is not None and unc != exact:
            self.log(f"!!! {name}: the search says {unc} uncovered, the checker says {exact}")
        return rec

    def found(self, r, blocks):
        n, name = len(blocks), r["name"]
        path = os.path.join(self.root, "found", f"{name}_{n}.json")
        with open(path, "w") as f:
            json.dump(blocks, f)
        p = CoveringDesign(r["v"], r["k"], r["t"], Record(r["record"], r["holder"] or "unknown",
                           f"La Jolla Covering Repository final data, {r['date']}, {r['method']}"), r["lower_bound"], path)
        folder, ok = claims.file_claim(p, n, blocks, "# engine/native.py loop; source engine/native/cover_anneal.c\n",
                                       "native loop", self.claims_root)
        with open(os.path.join(folder, "CLAIM.md"), "a") as f:
            f.write(f"\n## Live check\n- Last live check ({self.checked}): nobody had improved this entry after the "
                    f"freeze, so the live record was {r['record']}. Ours: {n}.\n- Re-check coveringrepository.com on the "
                    "day of submission; other searchers post improvements daily.\n")
        self.log(f"*** {name}: {n} blocks cover every {r['t']}-subset (record {r['record']}); both checks "
                 f"{'PASS' if ok else 'FAIL'}; packet {folder}")
        return folder

    def run_forever(self, workers=8, hours=24 * 365):
        deadline = time.time() + hours * 3600
        self.log(f"native loop: {len(self.rows)} targets, {workers} workers, {self.seconds:g}s per attempt "
                 f"(live filter {self.checked})")

        def worker():
            while time.time() < deadline:
                name = self.pick()
                if name is None:
                    time.sleep(5); continue
                try:
                    self.attempt(name)
                except Exception as e:  # one failed attempt costs one attempt, not the loop
                    with self.lock:
                        self.busy.discard(name)
                    self.log(f"attempt on {name} failed: {e!r}")

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()


def progress(root="runs/native", claims_root="claims"):
    """Verified progress per day: attempts, search hours, and coverings that beat the record they targeted."""
    days = {}
    path = os.path.join(root, "attempts.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                a = json.loads(line)
                d = days.setdefault(datetime.date.fromtimestamp(a["at"]).isoformat(), {"attempts": 0, "hours": 0.0, "found": 0})
                d["attempts"] += 1; d["hours"] += a["seconds"] / 3600; d["found"] += bool(a.get("found"))
    claimed = {}
    cpath = os.path.join(claims_root, "claims.jsonl")
    if os.path.exists(cpath):
        with open(cpath) as f:
            for line in f:
                c = json.loads(line)
                claimed.setdefault(datetime.date.fromtimestamp(c["at"]).isoformat(), set()).add(c["problem"])
    print(f"{'day':10} {'attempts':>8} {'hours':>7} {'found':>5} {'claims':>6}")
    for day in sorted(set(days) | set(claimed)):
        d = days.get(day, {"attempts": 0, "hours": 0.0, "found": 0})
        print(f"{day:10} {d['attempts']:8} {d['hours']:7.1f} {d['found']:5} {len(claimed.get(day, ())):6}")
