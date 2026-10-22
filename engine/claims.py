"""Claim packets. When a verified result beats the published record, everything needed to confirm and submit it is
written to claims/<problem>/<time>/. Nothing is sent anywhere: the owner approves every public word and submits."""
import itertools
import json
import math
import os
import time


def independent_check(blocks, v, k, t):
    """Plain Python, no numpy and no shared code with the engine's checker: does every t-subset lie in some block?"""
    if not all(isinstance(b, list) and len(b) == k and len(set(b)) == k and all(type(p) is int and 1 <= p <= v for p in b)
               for b in blocks):
        return False
    seen = set()
    for b in blocks:
        seen.update(itertools.combinations(sorted(b), t))
    return len(seen) == math.comb(v, t)


def best_claimed(root, name):
    path = os.path.join(root, "claims.jsonl")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        scores = [json.loads(line)["score"] for line in f if json.loads(line)["problem"] == name]
    return min(scores) if scores else None


def file_claim(problem, score, output, program, proposer, root="claims"):
    p, n = problem, int(score)
    folder = os.path.join(root, p.name, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(folder, exist_ok=True)
    rechecked = independent_check(output, p.v, p.k, p.t) and len(output) == n
    with open(os.path.join(folder, "covering.txt"), "w") as f:
        f.write("".join(" ".join(map(str, sorted(b))) + "\n" for b in output))
    with open(os.path.join(folder, "covering.json"), "w") as f:
        json.dump(output, f)
    with open(os.path.join(folder, "program.py"), "w") as f:
        f.write(program)
    with open(os.path.join(folder, "CLAIM.md"), "w") as f:
        f.write(f"""# C({p.v},{p.k},{p.t}) <= {n}: candidate improvement, not yet confirmed

- Our covering: {n} blocks (covering.txt, one block per line). Published record: {p.record.value} blocks
  ({p.record.holder}; {p.record.source}).
- Checked twice: the engine's exact checker, and a separate plain-Python check of all {math.comb(p.v, p.t)}
  {p.t}-subsets: {"PASS" if rechecked else "FAIL - do not use"}.
- Found by: {proposer} (program.py).

## Before anything is announced
1. Look up C({p.v},{p.k},{p.t}) on coveringrepository.com, which holds improvements made after the La Jolla
   repository froze. If it already lists {n} blocks or fewer, this is not new.
2. The owner reviews and approves the wording below.
3. The owner submits the covering (coveringrepository.com, or the maintainer by email) under the Quantavane name.

## Draft note to the maintainer (draft, not sent)
Hello, we found a ({p.v},{p.k},{p.t}) covering design with {n} blocks, improving the La Jolla Covering Repository
entry of {p.record.value}. The blocks are attached, one per line. It was found by Quantavane's search engine and
checked exhaustively. Thank you for keeping these tables.
Arhan Canli, Quantavane
""")
    with open(os.path.join(root, "claims.jsonl"), "a") as f:
        f.write(json.dumps({"at": time.time(), "problem": p.name, "score": n, "record": p.record.value,
                            "rechecked": rechecked, "folder": folder, "status": "drafted, not sent"}) + "\n")
    return folder, rechecked
