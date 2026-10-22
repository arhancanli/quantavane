"""The propose -> check -> keep loop, with islands so different lines of attack evolve separately."""
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import sandbox
from .problems import Verdict, better
from .proposers import BudgetExceeded, build_prompt


def inputs_for(problem, archive):
    """Files the candidate may read, such as the best covering so far to start from; None for most problems."""
    make = getattr(problem, "inputs", None)
    return make(archive) if make else None


def evaluate(problem, program, inputs=None):
    res = sandbox.run(program, timeout=problem.timeout, files=inputs)
    if not res.ok:
        return Verdict(False, note=res.error), None, res.seconds
    return problem.check(res.output), res.output, res.seconds


def seed(problem, archive, seed_program, islands):
    verdict, output, secs = evaluate(problem, seed_program, inputs_for(problem, archive))
    if not verdict.valid:
        raise SystemExit(f"Seed program is invalid: {verdict.note}")
    for i in range(islands):
        archive.add(island=i, program=seed_program, verdict=verdict, output=output, proposer="seed", seconds=secs)


def pick_parent(archive, island, rng):
    pool = archive.top(island, k=6)
    return pool[0] if rng.random() < 0.5 else rng.choice(pool)


def evolve(problem, archive, proposer, *, seed_program, iterations, islands=4, workers=2, migrate_every=25,
           log=print, rng=None):
    rng = rng or random.Random()
    if archive.best() is None:
        seed(problem, archive, seed_program, islands)
    lock, counter, stop = threading.Lock(), [0], threading.Event()

    def step():
        while not stop.is_set():
            with lock:
                if counter[0] >= iterations:
                    return
                it = counter[0]; counter[0] += 1
            island = it % islands
            if it and it % migrate_every == 0:
                best = archive.best()
                archive.add(island=island, program=best["program"], verdict=Verdict(True, best["score"], "migrated"),
                            output=json.loads(best["output"]), proposer="migration", parent_id=best["id"])
            parent = pick_parent(archive, island, rng)
            others = [r for r in archive.top(None, k=4) if r["id"] != parent["id"] and r["program"] != parent["program"]][:2]
            prompt = build_prompt(problem, parent, others, archive.recent_failures(island))
            t0 = time.monotonic()
            try:
                program, cost = proposer.propose(prompt)
            except BudgetExceeded as e:
                log(f"stopping: {e}"); stop.set(); return
            except Exception as e:  # a model or network hiccup costs one iteration, not the run
                log(f"[{it}] proposer error: {e}"); continue
            if program is None:
                archive.add(island=island, program="", verdict=Verdict(False, note="no program in reply"),
                            output=None, proposer=proposer.name, parent_id=parent["id"], cost_usd=cost)
                continue
            verdict, output, secs = evaluate(problem, program, inputs_for(problem, archive))
            archive.add(island=island, program=program, verdict=verdict, output=output, proposer=proposer.name,
                        parent_id=parent["id"], cost_usd=cost, seconds=secs)
            best = archive.best()["score"]
            kept = verdict.valid and not better(problem, parent["score"], verdict.score)
            mark = "KEEP" if kept else ("ok" if verdict.valid else "reject")
            log(f"[{it}] island {island} {mark:6} {verdict.score if verdict.valid else verdict.note[:80]!s:>12}"
                f"  best {best:.6f}  ({time.monotonic() - t0:.0f}s)")

    with ThreadPoolExecutor(workers) as pool:
        for f in [pool.submit(step) for _ in range(workers)]:
            f.result()
    return archive.best()
