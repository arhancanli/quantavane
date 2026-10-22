"""Problems with exact checkers. A checker never trusts the candidate: it re-verifies the returned object from scratch,
in exact rational arithmetic where geometry is involved, so a reported score is a proven score."""
from dataclasses import dataclass
from fractions import Fraction
import itertools
import json
import math
import os
import re

import numpy as np


@dataclass
class Record:
    value: float
    holder: str
    source: str


@dataclass
class Verdict:
    valid: bool
    score: float = float("-inf")
    note: str = ""


class CirclePacking:
    """Pack n disjoint circles inside the unit square to maximise the sum of their radii."""
    direction = "max"
    timeout = 120

    def __init__(self, n, record):
        self.n, self.record = n, record
        self.name = f"circle_packing_{n}"
        self.title = f"{n} circles in a unit square, maximise the sum of radii"

    def statement(self):
        return (f"Place {self.n} circles inside the unit square [0,1]x[0,1] with no two overlapping, maximising the sum of "
                f"their radii. construct() must return a list of {self.n} triples [x, y, r]. The checker verifies, in exact "
                "rational arithmetic on the returned floats, that every circle lies inside the square "
                "(r <= x <= 1-r, r <= y <= 1-r), r > 0, and every pair satisfies (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2. "
                "Touching is allowed; any overlap, even 1e-15, makes the whole answer invalid, so shrink radii very "
                "slightly after numerical optimisation. Score = sum of radii.")

    def check(self, out):
        if not isinstance(out, list) or len(out) != self.n:
            return Verdict(False, note=f"expected a list of {self.n} [x, y, r] triples")
        cs = []
        for i, c in enumerate(out):
            if not (isinstance(c, (list, tuple)) and len(c) == 3 and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in c)):
                return Verdict(False, note=f"circle {i} is not three finite numbers")
            cs.append(tuple(Fraction(v) for v in c))
        for i, (x, y, r) in enumerate(cs):
            if r <= 0:
                return Verdict(False, note=f"circle {i} has radius <= 0")
            if x < r or x > 1 - r or y < r or y > 1 - r:
                return Verdict(False, note=f"circle {i} sticks out of the square")
        for i in range(self.n):
            xi, yi, ri = cs[i]
            for j in range(i + 1, self.n):
                xj, yj, rj = cs[j]
                if (xi - xj) ** 2 + (yi - yj) ** 2 < (ri + rj) ** 2:
                    return Verdict(False, note=f"circles {i} and {j} overlap")
        return Verdict(True, float(sum(c[2] for c in cs)))


SEED_CIRCLES = '''import numpy as np
from scipy.optimize import linprog

N = {n}

def construct():
    # Centres on a near-square grid; then the largest radii those centres allow, by linear programming.
    k = int(np.ceil(np.sqrt(N)))
    pts = [((i + 0.5) / k, (j + 0.5) / k) for i in range(k) for j in range(k)][:N]
    c = -np.ones(N)
    rows, rhs = [], []
    for i in range(N):
        for j in range(i + 1, N):
            row = np.zeros(N); row[i] = row[j] = 1
            rows.append(row); rhs.append(np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]))
    bounds = [(0, min(x, y, 1 - x, 1 - y)) for x, y in pts]
    r = linprog(c, A_ub=np.array(rows), b_ub=np.array(rhs), bounds=bounds).x * (1 - 1e-12)
    return [[x, y, float(ri)] for (x, y), ri in zip(pts, r)]
'''

# Best-known values as published; each must cite its source before any claim is made against it.
PROBLEMS = {
    # AlphaEvolve's 2.63586276 used zero overlap tolerance, like this checker. ShinkaEvolve (Lange et al., 2025)
    # reports 2.635983283 under an absolute tolerance of 1e-7, so it is not directly comparable to an exact check.
    "circle_packing_26": (CirclePacking(26, Record(2.635983283, "ShinkaEvolve (Sakana AI); AlphaEvolve 2.63586276 at zero tolerance", "Novikov et al. 2025 (AlphaEvolve); Lange et al. 2025 (ShinkaEvolve)")), SEED_CIRCLES.format(n=26)),
    # Published to three decimals only; confirm the exact value from the AlphaEvolve results repository before any claim.
    "circle_packing_32": (CirclePacking(32, Record(2.937, "AlphaEvolve (Google DeepMind)", "Novikov et al. 2025, as published (rounded)")), SEED_CIRCLES.format(n=32)),
}


def binomials(v, t):
    return np.array([[math.comb(n, j) for j in range(t + 1)] for n in range(v + 1)], dtype=np.int64)


def unrank(r, t, binom):
    """The t-subset (0-based, descending) with colex rank r, where rank = sum of C(s_i, i + 1) over the sorted s_i."""
    s = []
    for i in range(t, 0, -1):
        c = int(np.searchsorted(binom[:, i], r, side="right")) - 1
        s.append(c)
        r -= int(binom[c, i])
    return s


def uncovered(blocks, v, k, t):
    """(count, example 1-based t-subset) for the t-subsets no block contains, or None if every one is covered."""
    binom, pat = binomials(v, t), np.array(list(itertools.combinations(range(k), t)), dtype=np.int64)
    seen = np.zeros(math.comb(v, t), dtype=bool)
    for i in range(0, len(blocks), 64):
        sub = blocks[i:i + 64][:, pat]  # rows of each sorted block, taken in order, are sorted t-subsets
        seen[binom[sub, np.arange(1, t + 1)].sum(axis=2).ravel()] = True
    n = int((~seen).sum())
    return (n, [p + 1 for p in reversed(unrank(int(np.flatnonzero(~seen)[0]), t, binom))]) if n else None


class CoveringDesign:
    """Cover every t-subset of {1..v} with as few k-point blocks as possible; the fewest possible is C(v,k,t)."""
    direction = "min"
    timeout = 300
    family = "covering"

    def __init__(self, v, k, t, record, lower_bound, start):
        self.v, self.k, self.t, self.record, self.lower_bound, self.start = v, k, t, record, lower_bound, start
        self.name = f"cover_{v}_{k}_{t}"
        self.title = f"C({v},{k},{t}): cover every {t}-subset of {v} points with {k}-point blocks, fewest blocks"

    def statement(self):
        v, k, t = self.v, self.k, self.t
        return (f"Find a ({v},{k},{t}) covering design with as few blocks as possible: blocks are sets of {k} distinct "
                f"integers from 1 to {v}, and every {t}-element subset of {{1..{v}}} must lie inside at least one block. "
                f"construct() returns the blocks as a list of lists of ints. The checker verifies all {math.comb(v, t)} "
                f"{t}-subsets exactly. Score = number of blocks, fewer is better. Best known: {self.record.value} blocks; "
                f"lower bound: {self.lower_bound}. The file start.json in the working directory holds the best valid "
                "covering found so far (1-based blocks): json.load(open('start.json')). Return a valid covering with "
                "fewer blocks if you can, otherwise the best valid one you have. Keep the line `V, K, T = ...` exactly as "
                "it is: the lab retargets programs to other instances by rewriting it. Use time.process_time() and "
                f"return before {self.timeout - 15} CPU seconds.")

    def inputs(self, archive):
        best = archive.best_solution() if archive is not None else None
        if best is not None and best["score"] <= self.record.value:
            blocks = json.loads(best["output"])
        else:
            with open(self.start) as f:
                blocks = json.load(f)
        return {"start.json": json.dumps(blocks)}

    def check(self, out):
        v, k, t = self.v, self.k, self.t
        if math.comb(v, t) > 50_000_000:
            return Verdict(False, note="too many t-subsets to check")
        if not isinstance(out, list) or not out or len(out) > 20 * self.record.value + 1000:
            return Verdict(False, note="expected a list of blocks of sensible length")
        rows = []
        for i, b in enumerate(out):
            if not (isinstance(b, list) and len(b) == k and all(type(p) is int and 1 <= p <= v for p in b)):
                return Verdict(False, note=f"block {i} is not {k} integers from 1 to {v}")
            if len(set(b)) != k:
                return Verdict(False, note=f"block {i} repeats a point")
            rows.append(sorted(p - 1 for p in b))
        missing = uncovered(np.array(rows, dtype=np.int64), v, k, t)
        if missing:
            return Verdict(False, note=f"{missing[0]} of the {t}-subsets are not covered, e.g. {missing[1]}")
        return Verdict(True, float(len(out)))


SEED_COVER = '''"""Post-optimise a covering design: drop one block, then anneal the rest until every T-subset is covered again.
Each step picks an uncovered T-subset S and a block missing one point y of S, then scores swapping y in for every
point of the block at once (lost coverage minus gained coverage) and takes the best swap, annealed."""
import itertools
import json
import math
import time

import numpy as np

V, K, T = __V__, __K__, __T__
LIMIT = __LIMIT__  # CPU seconds this program may use

BIN = np.array([[math.comb(n, j) for j in range(T + 1)] for n in range(V + 1)], dtype=np.int64)
N = math.comb(V, T)
COLS = np.arange(1, T + 1)
ALL = np.array(list(itertools.combinations(range(K), T)), dtype=np.int64)                      # T-subsets, by position
ALL1 = np.array(list(itertools.combinations(range(K), T - 1)), dtype=np.int64).reshape(-1, T - 1)  # (T-1)-subsets
HAS = np.zeros((len(ALL), K), dtype=np.int32)    # HAS[s, i] = 1 if T-subset s uses position i
HAS[np.arange(len(ALL))[:, None], ALL] = 1
HAS1 = np.zeros((len(ALL1), K), dtype=np.int32)  # the same for (T-1)-subsets
HAS1[np.arange(len(ALL1))[:, None], ALL1] = 1


def rank(rows):
    """Colex rank of each sorted row: the sum of C(s_i, i + 1)."""
    return BIN[rows, COLS].sum(axis=1)


def block_ranks(block):
    return rank(block[ALL])  # the block is sorted, so every row is sorted


def with_point(block, y):
    """Ranks of U + {y} for every (T-1)-subset U of the block, in ALL1 order; y is not in the block."""
    rows = np.hstack([block[ALL1], np.full((len(ALL1), 1), y, dtype=np.int64)])
    rows.sort(axis=1)
    return rank(rows)


def unrank(r):
    s = []
    for i in range(T, 0, -1):
        c = int(np.searchsorted(BIN[:, i], r, side="right")) - 1
        s.append(c)
        r -= int(BIN[c, i])
    return s


def repair(blocks, budget, rng):
    """Anneal the blocks (changed in place) until they cover every T-subset; True on success within budget CPU seconds."""
    inb = np.zeros((len(blocks), V), dtype=bool)
    for j, blk in enumerate(blocks):
        inb[j, blk] = True
    R = [block_ranks(blk) for blk in blocks]
    cov = np.zeros(N, dtype=np.int32)
    for r in R:
        cov[r] += 1
    zero = np.flatnonzero(cov == 0)
    unc = len(zero)
    start, temp, step = time.process_time(), 2.0, 0
    while unc:
        step += 1
        if step % 128 == 0:
            frac = (time.process_time() - start) / budget
            if frac >= 1:
                return False
            temp = 2.0 * (0.25 / 2.0) ** frac
            zero = np.flatnonzero(cov == 0)
        r = int(zero[rng.integers(len(zero))])
        if cov[r]:  # stale entry: refresh the list of uncovered subsets
            zero = np.flatnonzero(cov == 0)
            continue
        S = unrank(r)
        overlap = inb[:, S].sum(axis=1)
        cands = np.flatnonzero(overlap == overlap.max())
        j = int(cands[rng.integers(len(cands))])
        blk = blocks[j]
        missing = [p for p in S if not inb[j, p]]
        y = missing[int(rng.integers(len(missing)))]
        lost = (cov[R[j]] == 1).astype(np.int32) @ HAS          # per position x: subsets only this block covers, with x
        ry = with_point(blk, y)
        g = (cov[ry] == 0).astype(np.int32)
        delta = lost - (g.sum() - g @ HAS1)                     # minus the uncovered subsets y would cover without x
        delta[np.isin(blk, S)] = 1 << 30                        # never swap out a point of S
        best = np.flatnonzero(delta == delta.min())
        pos = int(best[rng.integers(len(best))])
        d = int(delta[pos])
        if d <= 0 or rng.random() < math.exp(-d / temp):
            x = int(blk[pos])
            r_old, r_new = R[j][HAS[:, pos] == 1], ry[HAS1[:, pos] == 0]
            cov[r_old] -= 1
            cov[r_new] += 1
            unc += d
            if d > 0:
                zero = np.concatenate([zero, r_old[cov[r_old] == 0]])
            inb[j, x], inb[j, y] = False, True
            blocks[j] = np.sort(np.append(blk[blk != x], y))
            R[j] = block_ranks(blocks[j])
    return True


def construct():
    rng = np.random.default_rng()
    best = [np.array(sorted(p - 1 for p in blk), dtype=np.int64) for blk in json.load(open("start.json"))]
    end = time.process_time() + LIMIT
    while time.process_time() < end - 2:
        cov = np.zeros(N, dtype=np.int32)
        for b in best:
            cov[block_ranks(b)] += 1
        own = np.array([int((cov[block_ranks(b)] == 1).sum()) for b in best])  # T-subsets only this block covers
        order = np.argsort(own + rng.random(len(own)))
        drop = int(order[rng.integers(min(4, len(order)))])
        trial = [b.copy() for i, b in enumerate(best) if i != drop]
        if repair(trial, min(end - time.process_time(), max(20.0, LIMIT / 4)), rng):
            best = trial
    return [[int(p) + 1 for p in b] for b in best]
'''


SEED_ORBITS = '''"""Look for a smaller covering made of whole orbits under a cyclic symmetry, then write out every block.

The points split into C cycles of length M plus F fixed points, and the symmetry moves every cycle on by one step. If a
base block covers a T-subset, its M shifted copies cover all M shifted copies of that subset, so it is enough to cover
one T-subset from each orbit. Annealing runs on base blocks and orbits, about M times smaller than the full problem,
and each base block becomes M blocks (fewer if the block repeats under the shift)."""
import itertools
import json
import math
import time

import numpy as np

V, K, T = __V__, __K__, __T__
LIMIT = __LIMIT__  # CPU seconds this program may use

BIN = np.array([[math.comb(n, j) for j in range(T + 1)] for n in range(V + 1)], dtype=np.int64)
N = math.comb(V, T)
COLS = np.arange(1, T + 1)
ALL = np.array(list(itertools.combinations(range(K), T)), dtype=np.int64)
ALL1 = np.array(list(itertools.combinations(range(K), T - 1)), dtype=np.int64).reshape(-1, T - 1)
HAS = np.zeros((len(ALL), K), dtype=np.int32)
HAS[np.arange(len(ALL))[:, None], ALL] = 1
HAS1 = np.zeros((len(ALL1), K), dtype=np.int32)
HAS1[np.arange(len(ALL1))[:, None], ALL1] = 1


def rank(rows):
    return BIN[rows, COLS].sum(axis=1)


def unrank(r):
    s = []
    for i in range(T, 0, -1):
        c = int(np.searchsorted(BIN[:, i], r, side="right")) - 1
        s.append(c)
        r -= int(BIN[c, i])
    return s


def shift_map(M, C):
    """Image of each point under one step of the symmetry."""
    img = np.arange(V)
    for c in range(C):
        img[c * M:(c + 1) * M] = c * M + (np.arange(M) + 1) % M
    return img


def orbits(img, M):
    """Orbit id (0..n-1) of every T-subset rank, and one member rank per orbit."""
    subs = np.array(list(itertools.combinations(range(V), T)), dtype=np.int64)
    step = rank(np.sort(img[subs], axis=1))           # rank of the shifted subset, indexed by rank in colex order
    order = rank(subs)                                # itertools order is lex, so map to colex ranks
    nxt = np.empty(N, dtype=np.int64)
    nxt[order] = step
    rep, cur = np.arange(N), np.arange(N)
    for _ in range(M - 1):
        cur = nxt[cur]
        rep = np.minimum(rep, cur)
    members, orb = np.unique(rep, return_inverse=True)
    return orb, members


def anneal(nb, orb, members, img, M, budget, rng):
    n_orb = len(members)
    blocks = [np.sort(rng.choice(V, K, replace=False)).astype(np.int64) for _ in range(nb)]
    inb = np.zeros((nb, V), dtype=bool)
    for j, b in enumerate(blocks):
        inb[j, b] = True
    O = [orb[rank(b[ALL])] for b in blocks]
    cov = np.zeros(n_orb, dtype=np.int32)
    for o in O:
        np.add.at(cov, o, 1)
    unc = int((cov == 0).sum())
    start, temp, step = time.process_time(), 2.0, 0
    while unc:
        step += 1
        if step % 128 == 0:
            frac = (time.process_time() - start) / budget
            if frac >= 1:
                return None
            temp = 2.0 * (0.25 / 2.0) ** frac
        zero = np.flatnonzero(cov == 0)
        S = np.array(unrank(int(members[zero[rng.integers(len(zero))]])), dtype=np.int64)
        for _ in range(int(rng.integers(M))):  # aim at a random member of the orbit
            S = img[S]
        overlap = inb[:, S].sum(axis=1)
        cands = np.flatnonzero(overlap == overlap.max())
        j = int(cands[rng.integers(len(cands))])
        blk = blocks[j]
        missing = [int(p) for p in S if not inb[j, p]]
        y = missing[int(rng.integers(len(missing)))]
        lost = (cov[O[j]] == 1).astype(np.int32) @ HAS          # estimated; repeated orbits make it approximate
        rows = np.hstack([blk[ALL1], np.full((len(ALL1), 1), y, dtype=np.int64)])
        rows.sort(axis=1)
        oy = orb[rank(rows)]
        g = (cov[oy] == 0).astype(np.int32)
        delta = lost - (g.sum() - g @ HAS1)
        delta[np.isin(blk, S)] = 1 << 30
        best = np.flatnonzero(delta == delta.min())
        pos = int(best[rng.integers(len(best))])
        o_old, o_new = O[j][HAS[:, pos] == 1], oy[HAS1[:, pos] == 0]
        touched = np.unique(np.concatenate([o_old, o_new]))
        before = int((cov[touched] == 0).sum())
        np.subtract.at(cov, o_old, 1)
        np.add.at(cov, o_new, 1)
        d = int((cov[touched] == 0).sum()) - before            # exact change in uncovered orbits
        if d <= 0 or rng.random() < math.exp(-d / temp):
            x = int(blk[pos])
            unc += d
            inb[j, x], inb[j, y] = False, True
            blocks[j] = np.sort(np.append(blk[blk != x], y))
            O[j] = orb[rank(blocks[j][ALL])]
        else:
            np.add.at(cov, o_old, 1)
            np.subtract.at(cov, o_new, 1)
    return blocks


def develop(bases, img, M):
    out = set()
    for b in bases:
        cur = b.copy()
        for _ in range(M):
            out.add(tuple(sorted(int(p) for p in cur)))
            cur = img[cur]
    return [list(b) for b in sorted(out)]


def covers(blocks):
    seen = np.zeros(N, dtype=bool)
    for b in blocks:
        seen[rank(np.array(b, dtype=np.int64)[ALL])] = True
    return bool(seen.all())


def construct():
    rng = np.random.default_rng()
    best = [sorted(p - 1 for p in blk) for blk in json.load(open("start.json"))]
    end = time.process_time() + LIMIT
    while time.process_time() < end - 5:
        # every symmetry (cycle length M) and number of base blocks that would beat the best so far
        sizes = [(M * nb, M, V // M, nb) for M in range(2, V + 1) for nb in range(1, len(best) // M + 1)
                 if M * nb < len(best)]
        if not sizes:
            break
        easiest = max(s[0] for s in sizes)  # the fewest blocks removed first, with a random symmetry
        _, M, C, nb = [s for s in sizes if s[0] == easiest][int(rng.integers(sum(s[0] == easiest for s in sizes)))]
        img = shift_map(M, C)
        orb, members = orbits(img, M)
        bases = anneal(nb, orb, members, img, M, min(end - time.process_time() - 3, max(10.0, LIMIT / 8)), rng)
        if bases is not None:
            blocks = develop(bases, img, M)
            if len(blocks) < len(best) and covers(blocks):
                best = blocks
    return [[int(p) + 1 for p in b] for b in best]
'''


def seed_cover(p, template=SEED_COVER):
    return (template.replace("__V__", str(p.v)).replace("__K__", str(p.k)).replace("__T__", str(p.t))
            .replace("__LIMIT__", str(p.timeout - 15)))


def retarget(program, p, limit=None):
    """Point a covering program at another instance by rewriting its `V, K, T = ...` line (and `LIMIT = ...` if a CPU
    budget is given); None if it has no such line."""
    line = re.compile(r"^V, K, T = .*$", re.M)
    if not line.search(program):
        return None
    program = line.sub(f"V, K, T = {p.v}, {p.k}, {p.t}", program, count=1)
    return re.sub(r"^LIMIT = .*$", f"LIMIT = {limit}", program, count=1, flags=re.M) if limit else program


DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def covering_problems(split):
    """Targets picked by scripts/ljcr_targets.py from the La Jolla Covering Repository's final data."""
    path = os.path.join(DATA, "covering_targets.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        meta = json.load(f)
    found = {}
    for r in meta["targets"]:
        start = os.path.join(DATA, "covers", f"{r['name']}.json")
        if r["split"] != split or not os.path.exists(start):
            continue
        source = (f"La Jolla Covering Repository, {r['date']}" + (f", {r['method']}" if r["method"] else "")
                  + "; LJCR froze 2026-03-01, so check coveringrepository.com before any claim")
        p = CoveringDesign(r["v"], r["k"], r["t"], Record(r["record"], r["holder"], source), r["lower_bound"], start)
        p.extra_seeds = [seed_cover(p, SEED_ORBITS)]  # a second line of attack, evolved on its own island
        found[p.name] = (p, seed_cover(p))
    return found


PROBLEMS.update(covering_problems("live"))
HIDDEN = covering_problems("hidden")  # never evolved on; only used to test models (training.py)


def better(problem, a, b):
    return a > b if problem.direction == "max" else a < b


def get(name):
    if name in PROBLEMS:
        return PROBLEMS[name]
    if name in HIDDEN:
        return HIDDEN[name]
    raise SystemExit(f"Unknown problem {name}. Known: {', '.join(PROBLEMS)}")
