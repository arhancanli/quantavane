"""Check every covering in this folder from scratch. Plain Python 3, no other packages.

A (v,k,t) covering design is a set of blocks, each k distinct points from 1..v, such that every t-element subset of
{1..v} lies inside at least one block. For each file this script checks the block count, the size and range of every
block, and that all C(v,t) t-subsets are covered. Run: python3 verify.py"""
import itertools
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def covers_everything(blocks, v, k, t):
    for b in blocks:
        if len(b) != k or len(set(b)) != k or not all(1 <= p <= v for p in b):
            return False, f"bad block {b}"
    seen = set()
    for b in blocks:
        seen.update(itertools.combinations(sorted(b), t))
    missing = math.comb(v, t) - len(seen)
    return missing == 0, f"{missing} of {math.comb(v, t)} {t}-subsets uncovered" if missing else "every subset covered"


def main():
    ok_all = True
    with open(os.path.join(HERE, "records.json")) as f:
        records = json.load(f)
    for r in records:
        with open(os.path.join(HERE, "coverings", r["file"])) as f:
            blocks = [[int(x) for x in line.split()] for line in f if line.strip()]
        ok, note = covers_everything(blocks, r["v"], r["k"], r["t"])
        ok = ok and len(blocks) == r["blocks"]
        ok_all &= ok
        print(f"C({r['v']},{r['k']},{r['t']}) <= {len(blocks):3}  (previous best {r['previous']})  "
              f"{'VALID' if ok else 'INVALID'}: {note}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
