# Covering design records from Quantavane

A (v,k,t) covering design is a collection of k-point blocks drawn from v points such that every set of t points lies
inside at least one block. C(v,k,t) is the fewest blocks possible. For most parameters the exact value is unknown, so
public tables record the best covering anyone has found.

These coverings each use one block fewer than the best known before them:

| Entry | Blocks | Previous best |
|---|---|---|
| C(65,14,2) | 28 | 29 |
| C(66,12,2) | 38 | 39 |
| C(70,21,2) | 15 | 16 |
| C(71,16,2) | 26 | 27 |
| C(75,17,2) | 26 | 27 |
| C(76,14,2) | 38 | 39 |
| C(94,17,2) | 39 | 40 |
| C(42,12,3) | 76 | 77 |

"Previous best" is the value in the La Jolla Covering Repository's final data (Daniel M. Gordon, CC BY 4.0,
doi:10.5281/zenodo.19735294). It was still the listed value on coveringrepository.com, which tracks improvements made
after that repository froze, when checked on 22 October 2026. All eight earlier values came from general constructions
rather than a direct search. The coverings were submitted to coveringrepository.com on 22 October 2026.

## Check them yourself

```sh
python3 verify.py
```

The script uses only the Python standard library. For each file it checks the block count, that every block has k
distinct points from 1 to v, and that every one of the C(v,t) t-subsets is covered. Each covering was also checked by
two other independent checkers before release.

## How they were found

A compiled local search in the style of Nurmela and Östergård (1993). With the number of blocks fixed, it minimises the
number of uncovered t-subsets by swapping one point of a block at a time. It alternates simulated annealing with
breakout weighting, the clause-weighting idea from SAT local search. It ran on a laptop at no cost, starting from
random blocks.

## Files

- `coverings/C_v_k_t_b.txt`: one block per line, points numbered from 1.
- `records.json`: the table above in machine-readable form.
- `verify.py`: the checker.

## Licence

CC BY 4.0 (see `LICENSE`). Please credit "Arhan Canli, Quantavane" when you use these coverings.

Arhan Canli, Quantavane · quantavane.com
