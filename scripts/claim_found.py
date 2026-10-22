"""File a claim packet for a covering found outside the lab (triage or push runs), after its live check.

.venv/bin/python scripts/claim_found.py 'C(66,12,2)' runs/triage/found/C(66,12,2)_38.json --site 39 \
    --checked '2026-10-22 14:55 +04' --how 'native annealer, triage at record-1, 20 s from random blocks, commit 9197243'

--site is the value coveringrepository.com listed at the --checked time. Nothing is sent: the packet is a draft for the
owner, who approves the wording and submits."""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from engine import claims  # noqa: E402
from engine.problems import CoveringDesign, Record  # noqa: E402
from engine.native import check  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("covering")
    ap.add_argument("--site", type=int, required=True); ap.add_argument("--checked", required=True)
    ap.add_argument("--how", required=True)
    a = ap.parse_args()
    v, k, t = (int(x) for x in a.name[2:-1].split(","))
    entry = json.load(open(os.path.join(HERE, "..", "engine", "data", "coverdata.json")))[a.name]
    blocks = json.load(open(a.covering))
    n = len(blocks)
    assert check(v, k, t, blocks) == 0, "the engine's checker rejects this covering"
    assert claims.independent_check(blocks, v, k, t), "the plain-Python check rejects this covering"
    assert n < a.site, f"{n} blocks does not beat the live value {a.site}"
    imp = entry["imps"][0]
    record = Record(int(entry["size"]), imp[2].strip() or "unknown",
                    f"La Jolla Covering Repository final data, {imp[3][:10]}, {imp[1].strip()}")
    p = CoveringDesign(v, k, t, record, int(entry["low_bd"]), a.covering)
    root = os.path.join(HERE, "..", "claims")
    folder, ok = claims.file_claim(p, n, blocks, f"# {a.how}\n# source: engine/native/cover_anneal.c\n", "native annealer", root)
    with open(os.path.join(folder, "CLAIM.md"), "a") as f:
        f.write(f"""
## Live check
- {a.checked}: coveringrepository.com listed C({v},{k},{t}) = {a.site} blocks. Ours: {n}. Lower bound in the La
  Jolla data: {entry['low_bd']}. Total {t}-subsets checked: {math.comb(v, t)}.
- How it was found: {a.how}.
- Re-check the live value on the day of submission; other searchers post improvements daily.
""")
    print(f"{a.name} <= {n} (live {a.site}); both checks PASS; packet {os.path.relpath(folder)}")


if __name__ == "__main__":
    main()
