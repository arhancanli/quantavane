"""The lab: every target problem at once, around the clock.

Model proposals go to whichever model the router says is winning. Spare CPU re-runs each problem's best program with
fresh randomness (the searches are random, so every run is a new attempt). Every result is checked exactly and kept in
the problem's archive; anything that beats a published record becomes a claim packet for the owner."""
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import claims, training
from .archive import Archive
from .loop import evaluate, inputs_for, pick_parent
from .problems import HIDDEN, better
from .proposers import BudgetExceeded, Ledger, OllamaProposer, build_prompt


def free_percent():
    """macOS's own measure of free memory, the "System-wide memory free percentage" that memory_pressure prints
    (sysctl kern.memorystatus_level). On 2026-10-22 the Mac hung and the watchdog rebooted it while the local model held
    14 of its 24 GB, so the lab waits whenever this runs low. None when unavailable (then the lab never waits)."""
    try:
        out = subprocess.run(["sysctl", "-n", "kern.memorystatus_level"], capture_output=True, text=True, timeout=10)
        return float(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class Lab:
    def __init__(self, problems, rng, root="runs", claims_root="claims", cap_usd=0.0, log=print):
        self.problems, self.rng, self.root, self.claims_root, self.log = problems, rng, root, claims_root, log
        os.makedirs(root, exist_ok=True)
        self.registry = training.Registry(os.path.join(root, "lab.db"))
        self.router = training.Router(self.registry, rng)
        self.ledger = Ledger(cap_usd)
        self.archives, self.reruns, self.lock = {}, {name: 0 for name in problems}, threading.Lock()
        self.memory = free_percent  # replaced in tests
        self.waiting = set()

    def wait_for_memory(self, floor, who, deadline):
        """Sleep while less than floor percent of memory is free, logging when the wait starts and when it ends."""
        while time.time() < deadline:
            free = self.memory()
            if free is None or free >= floor:
                if who in self.waiting:
                    self.waiting.discard(who); self.log(f"{who}: memory back, resuming")
                return
            if who not in self.waiting:
                self.waiting.add(who); self.log(f"{who}: only {free:g}% of memory free (need {floor}%), waiting")
            time.sleep(min(60, max(0.0, deadline - time.time())))

    def archive(self, name):
        with self.lock:
            if name not in self.archives:
                os.makedirs(os.path.join(self.root, name), exist_ok=True)
                direction = self.problems[name][0].direction
                self.archives[name] = Archive(os.path.join(self.root, name, "archive.db"), direction)
            return self.archives[name]

    def proposer_for(self, m):
        return OllamaProposer(m["model"], self.ledger, temperature=m["temperature"], name=m["name"])

    def record_check(self, name, verdict, output, program, proposer):
        p = self.problems[name][0]
        if not (verdict.valid and better(p, verdict.score, p.record.value)):
            return
        with self.lock:
            prev = claims.best_claimed(self.claims_root, name)
            if prev is not None and not better(p, verdict.score, prev):
                return
            folder, ok = claims.file_claim(p, verdict.score, output, program, proposer, self.claims_root)
        self.log(f"*** RECORD CANDIDATE {p.title}: {verdict.score:g} vs record {p.record.value}; "
                 f"independent check {'PASS' if ok else 'FAIL'}; packet {folder}")

    def seeds(self, name):
        p, program = self.problems[name]
        return [program] + list(getattr(p, "extra_seeds", []))  # one island per line of attack

    def seed(self, name):
        p, archive = self.problems[name][0], self.archive(name)
        for island, program in enumerate(self.seeds(name)):
            if archive.top(island, 1):
                continue
            verdict, output, secs = evaluate(p, program, inputs_for(p, archive))
            archive.add(island=island, program=program, verdict=verdict, output=output, proposer="seed", seconds=secs)
            self.record_check(name, verdict, output, program, "seed")
            self.log(f"seeded {name} island {island}: {verdict.score if verdict.valid else verdict.note[:70]!s} "
                     f"(record {p.record.value})")

    def rerun(self, name):
        p, archive = self.problems[name][0], self.archive(name)
        island = self.reruns[name] % len(self.seeds(name))
        top = archive.top(island, 1) or archive.top(None, 1)
        best = top[0]
        verdict, output, secs = evaluate(p, best["program"], inputs_for(p, archive))
        archive.add(island=island, program=best["program"], verdict=verdict, output=output, proposer="rerun",
                    parent_id=best["id"], seconds=secs)
        self.record_check(name, verdict, output, best["program"], f"rerun of attempt {best['id']} ({best['proposer']})")
        self.log(f"rerun {name}: {verdict.score if verdict.valid else verdict.note[:70]!s} "
                 f"(best {archive.best()['score']:g}, record {p.record.value})")

    def propose(self, name):
        p, archive = self.problems[name][0], self.archive(name)
        family = getattr(p, "family", p.name)
        m = self.router.choose(family)
        island = self.rng.randrange(len(self.seeds(name)))
        if not archive.top(island, 1):
            island = 0
        parent = pick_parent(archive, island, self.rng)
        others = [r for r in archive.top(None, k=6) if r["program"] != parent["program"]][:2]
        program, cost = self.proposer_for(m).propose(build_prompt(p, parent, others, archive.recent_failures(island)))
        if program is None:
            self.registry.log_use(m["name"], family, name, False, 0.0)
            self.log(f"model {m['name']} on {name}: no program in reply"); return
        verdict, output, secs = evaluate(p, program, inputs_for(p, archive))
        archive.add(island=island, program=program, verdict=verdict, output=output, proposer=m["name"], parent_id=parent["id"],
                    cost_usd=cost, seconds=secs)
        self.registry.log_use(m["name"], family, name, verdict.valid, training.reward(p, verdict, parent["score"]))
        self.record_check(name, verdict, output, program, m["name"])
        self.log(f"model {m['name']} on {name}: {verdict.score if verdict.valid else verdict.note[:70]!s} "
                 f"(parent {parent['score']:g}, record {p.record.value})")

    def hidden(self):
        """Daily: test every active model on hidden covering instances, then let the review promote or keep."""
        covering = [n for n in self.problems if getattr(self.problems[n][0], "family", "") == "covering"]
        if not covering or not HIDDEN:
            return
        ranked = sorted(covering, key=lambda n: self.archive(n).best()["score"] / self.problems[n][0].record.value)
        parent_program = self.archive(ranked[0]).best()["program"]
        training.hidden_session(self.registry, "covering", parent_program, HIDDEN, self.proposer_for,
                                rng=self.rng, log=self.log)
        training.review(self.registry, "covering", sorted(HIDDEN), log=self.log)

    def run(self, hours, cpu_workers=4, llm_workers=1, hidden_every=24 * 3600, cpu_floor=10, llm_floor=20):
        deadline, names = time.time() + hours * 3600, list(self.problems)
        with ThreadPoolExecutor(cpu_workers) as pool:
            list(pool.map(self.seed, names))

        def cpu_loop():
            while time.time() < deadline:
                self.wait_for_memory(cpu_floor, "re-runs", deadline)
                with self.lock:
                    name = min(names, key=lambda n: (self.reruns[n], self.rng.random()))
                    self.reruns[name] += 1  # also alternates the islands each problem re-runs
                try:
                    self.rerun(name)
                except Exception as e:  # one bad run costs one run, not the lab
                    self.log(f"rerun {name} failed: {e!r}")

        def llm_loop():
            i, last_hidden = 0, time.time()
            while time.time() < deadline:
                self.wait_for_memory(llm_floor, "model proposals", deadline)
                if time.time() >= deadline:
                    return
                name = names[i % len(names)]; i += 1
                try:
                    self.propose(name)
                except BudgetExceeded as e:
                    self.log(f"model calls stopped: {e}"); return
                except Exception as e:
                    self.log(f"proposal on {name} failed: {e!r}")
                if time.time() - last_hidden > hidden_every:
                    last_hidden = time.time()
                    try:
                        self.hidden()
                    except Exception as e:
                        self.log(f"hidden tests failed: {e!r}")

        with ThreadPoolExecutor(cpu_workers + llm_workers) as pool:
            jobs = [pool.submit(cpu_loop) for _ in range(cpu_workers)] + [pool.submit(llm_loop) for _ in range(llm_workers)]
            for job in jobs:
                job.result()
