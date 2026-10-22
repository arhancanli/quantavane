"""Triage lottery covering records (v,k,t,m) with the native search: for each target in the dated target file, try to
cover every m-subset with one block fewer than the live record, from random blocks, for a short budget. Anything that
reaches zero is checked by both exact checkers in engine/lotto.py and saved; it must still be checked against the live
coveringrepository.com value before any claim.

.venv/bin/python scripts/triage_lotto.py engine/data/lotto_targets_2026-10-22.txt --seconds 20 --workers 4"""
import argparse
import datetime
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
from engine import lotto  # noqa: E402


def targets(path):
    rows = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        v, k, t, m, b, kind = line.split()
        rows.append({"name": f"L({v},{k},{t},{m})", "v": int(v), "k": int(k), "t": int(t), "m": int(m),
                     "record": int(b), "kind": kind})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets")
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    lotto.build()
    rows = [r for r in targets(a.targets) if r["record"] >= 2]
    out_dir = os.path.join(ROOT, "runs", "triage_lotto")
    os.makedirs(os.path.join(out_dir, "found"), exist_ok=True)
    log = open(os.path.join(out_dir, f"{datetime.date.today()}.jsonl"), "a")
    print(f"{len(rows)} targets, {a.seconds:g}s each at record-1, {a.workers} workers", flush=True)

    def job(i, r):
        b = r["record"] - 1
        unc, moves, blocks = lotto.anneal(r["v"], r["k"], r["t"], r["m"], b, a.seconds, a.seed * 100_003 + i)
        return r, b, unc, moves, blocks

    with ThreadPoolExecutor(a.workers) as pool:
        futures = [pool.submit(job, i, r) for i, r in enumerate(rows)]
        for f in as_completed(futures):
            try:
                r, b, unc, moves, blocks = f.result()
            except Exception as e:  # one failed run costs one entry, not the triage
                print(f"failed: {e!r}", flush=True); continue
            rec = dict(r, blocks=b, uncovered=unc, moves=moves, seconds=a.seconds,
                       at=datetime.datetime.now().isoformat(timespec="seconds"))
            if unc == 0:
                exact = lotto.check(r["v"], r["k"], r["t"], r["m"], blocks)
                ind = lotto.independent_check(blocks, r["v"], r["k"], r["t"], r["m"])
                rec.update(exact_uncovered=exact, independent_check=ind)
                if exact == 0 and ind and len(blocks) == b:
                    path = os.path.join(out_dir, "found", f"{r['name']}_{b}.json")
                    with open(path, "w") as fh:
                        json.dump(blocks, fh)
                    rec["file"] = path
                    print(f"*** {r['name']}: {b} blocks cover every {r['m']}-subset (record {r['record']}); both checks "
                          f"PASS; {path}", flush=True)
                else:
                    print(f"!!! {r['name']}: search says covered, checker {exact}, independent {ind}", flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"{r['name']:18} b={b:5} uncovered {unc:7}", flush=True)


if __name__ == "__main__":
    main()
