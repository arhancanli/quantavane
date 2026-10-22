"""Triage covering-design records with the native annealer: for each candidate entry of the La Jolla Covering
Repository's final data, try to cover every t-subset with one block fewer than the record, from random blocks, for a
short budget. The final number of uncovered t-subsets shows where a longer search is worth it; zero is a candidate
record, which is checked twice by the exact checkers here and must still be checked on coveringrepository.com (it holds
improvements made after the 2026-03-01 freeze) before any claim.

Candidates by default: blocks of 11+ points, record above the lower bound, record made by one of the repository's
automatic constructions (holder "LJCR", never directly optimised), at most 400,000 t-subsets.

.venv/bin/python scripts/triage_covers.py --seconds 20 --workers 8 [--holders all] [--only 'C(39,11,2)' ...]"""
import argparse
import datetime
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
from engine import claims  # noqa: E402
from engine.native import BINARY, anneal, build, check  # noqa: E402
from engine.problems import uncovered  # noqa: E402,F401



def candidates(index, holders="LJCR", min_k=11, max_subsets=400_000):
    rows = []
    for name, e in index.items():
        v, k, t = (int(x) for x in name[2:-1].split(","))
        size, lb = int(e["size"]), int(e["low_bd"])
        holder = e["imps"][0][2].strip()
        if size <= lb or k < min_k or math.comb(v, t) > max_subsets or (holders != "all" and holder != holders):
            continue
        rows.append({"name": name, "v": v, "k": k, "t": t, "record": size, "lower_bound": lb, "holder": holder,
                     "method": e["imps"][0][1].strip(), "date": e["imps"][0][3][:10]})
    return sorted(rows, key=lambda r: (math.comb(r["v"], r["t"]), r["name"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fewer", type=int, default=1, help="blocks below the record to try")
    ap.add_argument("--holders", default="LJCR")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--targets", help="JSON file whose 'unchanged' list names the entries to run (live filter)")
    ap.add_argument("--weighted", action="store_true", help="breakout weighting instead of annealing")
    ap.add_argument("--tag", default="", help="suffix for the results file")
    a = ap.parse_args()
    build()
    index = json.load(open(os.path.join(ROOT, "engine", "data", "coverdata.json")))
    rows = candidates(index, holders=a.holders)
    if a.only:
        rows = [r for r in rows if r["name"] in a.only]
    if a.targets:
        wanted = set(json.load(open(a.targets))["unchanged"])
        rows = [r for r in rows if r["name"] in wanted]
    out_dir = os.path.join(ROOT, "runs", "triage")
    os.makedirs(os.path.join(out_dir, "found"), exist_ok=True)
    log = open(os.path.join(out_dir, f"{datetime.date.today()}{a.tag}.jsonl"), "a")
    print(f"{len(rows)} candidates, {a.seconds:g}s each at record-{a.fewer}, {a.workers} workers", flush=True)

    def job(i, r):
        b = r["record"] - a.fewer
        unc, moves, blocks = anneal(r["v"], r["k"], r["t"], b, a.seconds, a.seed * 100_003 + i, weighted=a.weighted)
        return r, b, unc, moves, blocks

    with ThreadPoolExecutor(a.workers) as pool:
        futures = [pool.submit(job, i, r) for i, r in enumerate(rows)]
        for f in as_completed(futures):
            r, b, unc, moves, blocks = f.result()
            exact = check(r["v"], r["k"], r["t"], blocks)
            rec = dict(r, blocks=b, uncovered=unc, exact_uncovered=exact, moves=moves, seconds=a.seconds, weighted=a.weighted,
                       at=datetime.datetime.now().isoformat(timespec="seconds"))
            if exact == 0:
                path = os.path.join(out_dir, "found", f"{r['name']}_{b}{a.tag}.json")
                with open(path, "w") as fh:
                    json.dump(blocks, fh)
                rec["independent_check"] = claims.independent_check(blocks, r["v"], r["k"], r["t"])
                rec["file"] = path
                print(f"*** {r['name']}: {b} blocks cover everything (record {r['record']}); "
                      f"independent check {'PASS' if rec['independent_check'] else 'FAIL'}; {path}", flush=True)
            elif unc != exact:
                print(f"!!! {r['name']}: annealer says {unc} uncovered, checker says {exact}", flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"{r['name']:12} b={b:4} uncovered {exact:6}  ({r['method'][:40]})", flush=True)


if __name__ == "__main__":
    main()
