# Quantavane engine

Search programs and AI models propose constructions. Exact checkers decide whether they are valid. Only verified
results count, and every attempt, kept or rejected, is recorded.

## Results so far

Twenty-seven covering designs, each one block below the best value known before, found by this engine on a laptop at
no cost. C(52,11,2) = 29 and C(94,14,2) = 54 meet the known lower bounds, so
both are optimal. C(47,21,3) = 21 was also found independently
the same day by Hossein Farahani, whose upload is the one coveringrepository.com lists. The first ten were sent to
coveringrepository.com on 22 October 2026, and uploading all of them is under way. The coverings and a standalone
checker are in [quantavane-covering-records](https://github.com/arhancanli/quantavane-covering-records) and in
`release/covering-records/`.

| Entry | Blocks | Previous best |
|---|---|---|
| C(41,9,2) | 27 | 28 |
| C(52,11,2) | 29 | 30 |
| C(65,14,2) | 28 | 29 |
| C(66,12,2) | 38 | 39 |
| C(67,15,2) | 26 | 27 |
| C(69,11,2) | 50 | 51 |
| C(70,21,2) | 15 | 16 |
| C(71,16,2) | 26 | 27 |
| C(75,17,2) | 26 | 27 |
| C(76,14,2) | 38 | 39 |
| C(82,18,2) | 27 | 28 |
| C(84,19,2) | 26 | 27 |
| C(87,16,2) | 38 | 39 |
| C(92,16,2) | 42 | 43 |
| C(94,14,2) | 54 | 55 |
| C(94,17,2) | 39 | 40 |
| C(96,22,2) | 25 | 26 |
| C(98,21,2) | 28 | 29 |
| C(98,22,2) | 26 | 27 |
| C(36,10,3) | 80 | 81 |
| C(41,17,3) | 26 | 27 |
| C(42,12,3) | 76 | 77 |
| C(47,21,3) | 21 | 22 |
| C(49,13,3) | 81 | 82 |
| C(72,23,3) | 51 | 52 |
| C(28,17,4) | 20 | 21 |
| C(40,23,4) | 24 | 25 |

## Quick start

```sh
uv venv --python 3.12 .venv && uv pip install --python .venv numpy scipy
.venv/bin/python -m unittest tests/test_engine.py tests/test_lab.py tests/test_native.py tests/test_lotto.py
.venv/bin/python scripts/reproduce_records.py --minutes 10   # find the records again from scratch
```

Needs Python 3.12 and a C compiler (`clang`); the native searches compile themselves on first use. The searches and
checkers run on macOS or Linux; the lab's sandbox uses macOS `sandbox-exec`.

## What is inside

- **Native searches** (`engine/native/`): compiled local search for covering designs C(v,k,t), in the style of
  Nurmela and Östergård, with simulated annealing and breakout weighting (the clause-weighting idea from SAT local
  search); and a search for lottery coverings (v,k,t,m) that keeps exact counts of covered subsets.
- **Exact checkers**: every covering is checked by the engine's checker and by a separate plain-Python check before it
  is reported. The checkers never trust the search.
- **The always-on loop** (`python -m engine native`): works through open table entries, gives close misses more time,
  alternates search methods, and writes a claim packet for anything below the record.
- **The lab** (`python -m engine lab`): AI models write candidate programs, which run in a macOS sandbox with no
  network, no private file access and CPU, memory and disk limits; an attempt archive, islands, a spending ledger with a hard cap, and
  training system v1 (model registry, router, hidden tests, champion and challenger).
- **Triage scripts** (`scripts/`): try one block below the record on many entries at once.
- **Matrix multiplication** (`engine/native/flipgraph.c`): the flip-graph search of Kauers and Moosbauer over GF(2),
  which walks between multiplication schemes by exact rewrites and lowers the number of multiplications when terms
  cancel or merge. It finds Strassen's 7 and Laderman's 23 from the standard algorithm.
- **Open Problems Benchmark v0** (`python -m engine bench`): models write construction programs for covering-design
  tasks in three tiers, run in the sandbox and scored by the exact checker.

The sandbox is a safety net for local experiments (no network, no private reads, CPU, memory, disk and open-file
limits), not a hardened isolation boundary; run untrusted code in a VM if that matters to you.

## Data and credits

Record values and the record coverings used as starting points come from the La Jolla Covering Repository (Daniel M.
Gordon, CC BY 4.0, doi:10.5281/zenodo.19735294); `scripts/ljcr_targets.py` downloads them. Improvements made after
that repository froze in March 2026 are listed on coveringrepository.com, so check it before claiming a record.

## Licence

Apache License 2.0 (see `LICENSE`).

Arhan Canli, Quantavane · quantavane.com
