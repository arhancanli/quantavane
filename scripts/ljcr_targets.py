"""Pick covering-design targets from the La Jolla Covering Repository's final data (CC BY 4.0, Daniel M. Gordon,
doi:10.5281/zenodo.19735294) and pull each target's record covering by streaming covers.json (4.2 GB) without saving
it. Writes engine/data/covering_targets.json and engine/data/covers/C(v,k,t).json.

Targets are entries nobody has improved for years: big blocks (k >= 11), few blocks, last improved before 2015, at most
three improvements ever. The LJCR froze on 2026-03-01; coveringrepository.com has later improvements, so every target
must be checked there before any claim."""
import hashlib
import json
import math
import os
import sys
import urllib.request

RECORD = "https://zenodo.org/api/records/19735294/files/{}/content"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "engine", "data")


def fetch_index(path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(RECORD.format("coverdata.json"), path)
    raw = open(path, "rb").read()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def pick(index, live=20, hidden=10, per_t=6):
    rows = []
    for name, e in index.items():
        v, k, t = (int(x) for x in name[2:-1].split(","))
        size, lb, imps = int(e["size"]), int(e["low_bd"]), e["imps"]
        if size <= lb or k < 11 or size > 120 or math.comb(v, t) > 300_000:
            continue
        if imps[0][3] >= "2015" or len(imps) > 3:
            continue
        rows.append({"name": name, "v": v, "k": k, "t": t, "record": size, "lower_bound": lb,
                     "holder": imps[0][2].strip() or "unknown", "method": imps[0][1].strip(), "date": imps[0][3][:10],
                     "improvements": len(imps)})
    rows.sort(key=lambda r: (-r["record"] / r["lower_bound"], r["name"]))
    chosen, count = [], {}
    for r in rows:
        if count.get(r["t"], 0) < per_t:
            chosen.append(r); count[r["t"]] = count.get(r["t"], 0) + 1
        if len(chosen) == live + hidden:
            break
    for i, r in enumerate(chosen):
        r["split"] = "live" if i % 3 != 2 or sum(c.get("split") == "hidden" for c in chosen[:i]) >= hidden else "hidden"
    return chosen


def stream_covers(wanted, out_dir):
    """covers.json is pretty-printed: a '"C(v,k,t)": [' line, one block per line, then '],'. Keep only wanted keys."""
    os.makedirs(out_dir, exist_ok=True)
    left, key, blocks = set(wanted), None, []
    with urllib.request.urlopen(RECORD.format("covers.json"), timeout=120) as r:
        for raw in r:
            line = raw.strip()
            if line.startswith(b'"C('):
                key, blocks = line.split(b'"')[1].decode(), []
            elif key in left and line.startswith(b"["):
                blocks.append(json.loads(line.rstrip(b",")))
            elif key in left and line in (b"],", b"]"):
                with open(os.path.join(out_dir, f"{key}.json"), "w") as f:
                    json.dump(blocks, f)
                left.discard(key); print(f"got {key}: {len(blocks)} blocks", flush=True)
                if not left:
                    return
    if left:
        sys.exit(f"missing from covers.json: {sorted(left)}")


if __name__ == "__main__":
    index, digest = fetch_index(os.path.join(DATA, "coverdata.json"))
    chosen = pick(index)
    meta = {"source": "La Jolla Covering Repository 1.2, Daniel M. Gordon, CC BY 4.0, doi:10.5281/zenodo.19735294",
            "coverdata_sha256": digest, "frozen": "2026-03-01",
            "check_before_claim": "coveringrepository.com (Giovanni Acerbi) holds improvements made after the freeze",
            "targets": chosen}
    with open(os.path.join(DATA, "covering_targets.json"), "w") as f:
        json.dump(meta, f, indent=1)
    for r in chosen:
        print(f"{r['split']:6} {r['name']:12} record {r['record']:4} lower bound {r['lower_bound']:4} {r['date']} {r['holder'][:24]}")
    if "--covers" in sys.argv:
        stream_covers([r["name"] for r in chosen], os.path.join(DATA, "covers"))
