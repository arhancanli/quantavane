"""python -m engine run|lab|native|progress|status|best|verify|hidden — see README."""
import argparse
import datetime
import json
import os
import random
import sys

from . import problems, training
from .archive import Archive
from .lab import Lab
from .loop import evolve
from .proposers import Ledger, OllamaProposer


def status(root="runs", claims_root="claims"):
    print(f"{'problem':16} {'record':>7} {'ours':>7} {'runs':>6}  last run")
    for name, (p, _) in problems.PROBLEMS.items():
        path = os.path.join(root, name, "archive.db")
        if not os.path.exists(path):
            print(f"{name:16} {p.record.value:>7g} {'-':>7} {0:>6}"); continue
        s = Archive(path, p.direction).stats()
        last = datetime.datetime.fromtimestamp(s["last"]).strftime("%m-%d %H:%M") if s["last"] else "-"
        mark = "  BEATS RECORD" if s["best"] is not None and problems.better(p, s["best"], p.record.value) else ""
        print(f"{name:16} {p.record.value:>7g} {s['best'] if s['best'] is not None else '-':>7} {s['n']:>6}  {last}{mark}")
    db = os.path.join(root, "lab.db")
    if os.path.exists(db):
        reg = training.Registry(db)
        print("\nmodels (family covering: live uses, mean reward, hidden tests, mean gain)")
        for m in reg.active():
            n, w = reg.live(m["name"], "covering")
            h, g = reg.hidden_mean(m["name"], sorted(problems.HIDDEN))
            print(f"  {m['name']:16} {m['role']:10} {n:5} {w / n if n else 0:6.3f} {h:5} {g:+.3f}")
    path = os.path.join(claims_root, "claims.jsonl")
    if os.path.exists(path):
        print("\nclaims")
        with open(path) as f:
            for line in f:
                c = json.loads(line)
                print(f"  {c['problem']}: {c['score']} vs record {c['record']}, recheck {'PASS' if c['rechecked'] else 'FAIL'},"
                      f" {c['status']} ({c['folder']})")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="engine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="search one problem with islands")
    r.add_argument("problem"); r.add_argument("--iterations", type=int, default=50)
    r.add_argument("--model", default="gemma4-12b-32k"); r.add_argument("--islands", type=int, default=4)
    r.add_argument("--workers", type=int, default=2); r.add_argument("--cap-usd", type=float, default=0.0)
    lab = sub.add_parser("lab", help="every target at once: model proposals, re-runs, claims, hidden tests")
    lab.add_argument("--hours", type=float, default=24); lab.add_argument("--cpu", type=int, default=4)
    lab.add_argument("--llm", type=int, default=1); lab.add_argument("--cap-usd", type=float, default=0.0)
    lab.add_argument("--only", nargs="*", help="limit to these problems")
    nat = sub.add_parser("native", help="always-on native covering search over every entry nobody improved since the freeze")
    nat.add_argument("--workers", type=int, default=8); nat.add_argument("--seconds", type=float, default=60)
    nat.add_argument("--hours", type=float, default=24 * 365); nat.add_argument("--max-subsets", type=int, default=400_000)
    sub.add_parser("progress", help="verified progress per day from the native loop and the claim packets")
    be = sub.add_parser("bench", help="Open Problems Benchmark v0 (internal): models write construction programs")
    be.add_argument("--model"); be.add_argument("--baseline", action="store_true"); be.add_argument("--report", action="store_true")
    be.add_argument("--per-tier", type=int, default=10); be.add_argument("--seconds", type=float, default=60)
    sub.add_parser("status", help="records, our best, models and claims")
    h = sub.add_parser("hidden", help="run hidden tests now and review the champion")
    b = sub.add_parser("best", help="export the best verified construction"); b.add_argument("problem")
    v = sub.add_parser("verify", help="re-check a solution file with the exact checker")
    v.add_argument("problem"); v.add_argument("solution")
    a = ap.parse_args(argv)
    say = lambda m: print(m, flush=True)

    if a.cmd == "status":
        return status()
    if a.cmd == "bench":
        from . import bench
        if a.report:
            return bench.report()
        if a.baseline:
            return bench.baseline(a.per_tier, a.seconds, log=say)
        return bench.run(a.model or "gemma4-12b-32k", a.per_tier, a.seconds, log=say)
    if a.cmd == "progress":
        from . import native
        return native.progress()
    if a.cmd == "native":
        from . import native
        native.build()
        rows, checked = native.targets(max_subsets=a.max_subsets)
        stamp = lambda m: say(f"{datetime.datetime.now():%m-%d %H:%M:%S} {m}")
        return native.Loop(rows, checked, seconds=a.seconds, log=stamp).run_forever(workers=a.workers, hours=a.hours)
    if a.cmd in ("lab", "hidden"):
        targets = {n: pr for n, pr in problems.PROBLEMS.items() if n.startswith("cover_") and (not getattr(a, "only", None) or n in a.only)}
        stamp = lambda m: say(f"{datetime.datetime.now():%m-%d %H:%M:%S} {m}")
        lab = Lab(targets, random.Random(), cap_usd=getattr(a, "cap_usd", 0.0), log=stamp)
        if a.cmd == "hidden":
            for name in targets:
                lab.seed(name)
            return lab.hidden()
        return lab.run(a.hours, cpu_workers=a.cpu, llm_workers=a.llm)

    problem, seed_program = problems.get(a.problem)
    os.makedirs(f"runs/{a.problem}", exist_ok=True)
    archive = Archive(f"runs/{a.problem}/archive.db", problem.direction)
    if a.cmd == "run":
        proposer = OllamaProposer(a.model, Ledger(a.cap_usd))
        best = evolve(problem, archive, proposer, seed_program=seed_program, iterations=a.iterations,
                      islands=a.islands, workers=a.workers, log=say)
        print(f"best {best['score']:.9g} (record {problem.record.value}, {problem.record.holder})")
    elif a.cmd == "best":
        best = archive.best_solution()
        verdict = problem.check(json.loads(best["output"]))
        assert verdict.valid and verdict.score == best["score"], "archive and checker disagree"
        os.makedirs(f"results/{a.problem}", exist_ok=True)
        with open(f"results/{a.problem}/solution.json", "w") as f:
            json.dump(json.loads(best["output"]), f, indent=1)
        with open(f"results/{a.problem}/program.py", "w") as f:
            f.write(best["program"])
        print(f"{problem.title}\nverified score {verdict.score:.12g}; record {problem.record.value} ({problem.record.holder}, {problem.record.source})")
    else:
        with open(a.solution) as f:
            verdict = problem.check(json.load(f))
        print(f"{'VALID' if verdict.valid else 'INVALID'} {verdict.score if verdict.valid else verdict.note}")
        sys.exit(0 if verdict.valid else 1)


if __name__ == "__main__":
    main()
