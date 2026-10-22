"""Find the released covering records again from scratch and check them.

For each record in release/covering-records/records.json, the native search starts from random blocks with the record's
number of blocks and keeps trying fresh seeds until every t-subset is covered (or the time limit runs out). Each
covering found is checked by the engine's exact checker and by the separate plain-Python check.

.venv/bin/python scripts/reproduce_records.py [--minutes 10] [--only 'C(66,12,2)' ...]"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
from engine import claims, native  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10, help="time limit per record")
    ap.add_argument("--seconds", type=float, default=20, help="length of one search run")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    native.build()
    records = json.load(open(os.path.join(ROOT, "release", "covering-records", "records.json")))
    ok_all = True
    for r in records:
        name = f"C({r['v']},{r['k']},{r['t']})"
        if a.only and name not in a.only:
            continue
        v, k, t, b = r["v"], r["k"], r["t"], r["blocks"]
        start, seed, found = time.time(), 1, False
        while time.time() - start < a.minutes * 60:
            unc, _, blocks = native.anneal(v, k, t, b, a.seconds, seed, weighted=seed % 2 == 0)
            if unc == 0 and native.check(v, k, t, blocks) == 0 and claims.independent_check(blocks, v, k, t):
                found = True
                break
            seed += 1
        ok_all &= found
        print(f"{name} <= {b} (previous best {r['previous']}): "
              f"{'found and checked' if found else 'not found'} after {seed} runs, {time.time() - start:.0f} s", flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
