"""Lottery covering designs (v,k,t,m): k-point blocks such that every m-subset of the v points meets some block in at
least t points. The compiled search is native/lotto_anneal.c; here are two independent exact checkers and the runner.

check() counts uncovered m-subsets with numpy bit masks (popcount of block & m-subset); independent_check() uses plain
Python and a different idea: an m-subset is covered exactly when one of its t-subsets lies inside a block."""
import itertools
import math
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BINARY = os.path.join(HERE, "native", "lotto_anneal")


def build():
    src = BINARY + ".c"
    if not os.path.exists(BINARY) or os.path.getmtime(BINARY) < os.path.getmtime(src):
        subprocess.run(["clang", "-O3", "-mcpu=native", "-o", BINARY, src, "-lm"], check=True)


def anneal(v, k, t, m, b, seconds, seed, start=None):
    """(uncovered, moves, blocks) for the best state of one run with b blocks; uncovered 0 means a covering."""
    env = dict(os.environ, CYCLE=str(seconds))
    args = [BINARY, str(v), str(k), str(t), str(m), str(b), str(seconds), str(seed)] + ([start] if start else [])
    run = subprocess.run(args, capture_output=True, text=True, env=env, timeout=seconds * 3 + 60)
    out = run.stdout.split("\n")
    head = out[0].split()
    if len(head) < 4:
        raise RuntimeError(f"lotto_anneal failed: {run.stderr.strip() or 'no output'}")
    blocks = [[int(p) for p in line.split()] for line in out[1:] if line.strip()]
    return int(head[1]), int(head[3]), blocks


def _masks(sets, v):
    """Two uint64 words per set (points 1..v), so v up to 128."""
    lo = np.zeros(len(sets), dtype=np.uint64); hi = np.zeros(len(sets), dtype=np.uint64)
    for i, s in enumerate(sets):
        a = b = 0
        for p in s:
            if p <= 64:
                a |= 1 << (p - 1)
            else:
                b |= 1 << (p - 65)
        lo[i], hi[i] = a, b
    return lo, hi


def check(v, k, t, m, blocks, chunk=200_000):
    """Exact number of m-subsets of {1..v} that meet no block in t or more points (0 means a valid covering)."""
    if not blocks or any(len(b) != k or len(set(b)) != k or not all(1 <= p <= v for p in b) for b in blocks):
        raise ValueError("blocks must be k distinct points from 1..v")
    blo, bhi = _masks(blocks, v)
    missing = 0
    combos = itertools.combinations(range(1, v + 1), m)
    while True:
        part = list(itertools.islice(combos, chunk))
        if not part:
            return missing
        mlo, mhi = _masks(part, v)
        best = np.zeros(len(part), dtype=np.int64)
        for i in range(len(blocks)):
            hits = np.bitwise_count(mlo & blo[i]).astype(np.int64) + np.bitwise_count(mhi & bhi[i])
            np.maximum(best, hits, out=best)
        missing += int((best < t).sum())


def independent_check(blocks, v, k, t, m):
    """Plain Python, no numpy, no shared code with check(): every m-subset must contain a t-subset of some block."""
    if not all(isinstance(b, list) and len(b) == k and len(set(b)) == k and all(type(p) is int and 1 <= p <= v for p in b)
               for b in blocks):
        return False
    inside = set()
    for b in blocks:
        inside.update(itertools.combinations(sorted(b), t))
    for s in itertools.combinations(range(1, v + 1), m):
        if not any(c in inside for c in itertools.combinations(s, t)):
            return False
    return True


def subsets(v, m):
    return math.comb(v, m)
