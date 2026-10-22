/* Simulated annealing for covering designs, in the style of Nurmela and Ostergard (1993).

   A (v,k,t) covering design is a set of k-point blocks from {0..v-1} such that every t-subset lies in some block.
   With the number of blocks b fixed, the search minimises the number of uncovered t-subsets; zero is a covering.
   Each step takes a random uncovered t-subset S and a block that already holds most of S, adds a missing point y of S
   to that block, and scores removing each point x of the block not in S at once:
     delta(x) = (t-subsets with x that only this block covers) - (uncovered t-subsets that the block gains with y).
   The best x is taken if it does not make things worse, otherwise with probability exp(-delta / temperature).
   The temperature falls geometrically from T0 to T1 over each cycle, then starts again; the best state is kept.

   cover_anneal V K T B SECONDS SEED [START]
   START is an optional file of blocks, one per line, 1-based points. With more than B blocks, the blocks that cover
   the fewest t-subsets on their own are dropped; with fewer, random blocks are added.
   Prints "uncovered U moves M seconds S" and then the best blocks found (1-based), one per line.
   Environment: CYCLE (seconds per temperature cycle, default 10), T0 (default 2.0), T1 (default 0.2); WEIGHTED,
   TABU, BUMPALL and SHRINK as described in main(). */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXV 128
#define MAXK 32
#define MAXT 8

typedef unsigned __int128 mask_t;

static int V, K, T, B;
static uint64_t binom[MAXV + 1][MAXT + 1];
static uint64_t N;
static uint16_t *cov;          /* how many blocks cover each t-subset, by colex rank */
static uint8_t *inlist;        /* 1 if the rank is in the list of (possibly) uncovered t-subsets */
static uint32_t *list;         /* candidate uncovered ranks; stale entries are removed when drawn */
static uint64_t listlen, uncovered;
static uint8_t *pts;           /* B x K points, each block sorted */
static mask_t *mask;           /* B block masks */
static uint32_t *ranks;        /* B x CT ranks of each block's t-subsets */
static uint8_t *ct, *ct1;      /* position combinations: CT x T and CT1 x (T-1) */
static int CT, CT1;
static uint64_t rng_state;

static uint64_t rnd(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double rnd01(void) { return (rnd() >> 11) * (1.0 / 9007199254740992.0); }
static int popc(mask_t m) { return __builtin_popcountll((uint64_t)m) + __builtin_popcountll((uint64_t)(m >> 64)); }
static double now(void) { return (double)clock() / CLOCKS_PER_SEC; }

static int fill_combos(uint8_t *out, int n, int r) {  /* all r-subsets of positions 0..n-1, lexicographic */
    int c[MAXT + 1], count = 0;
    if (r == 0) return 1;
    for (int i = 0; i < r; i++) c[i] = i;
    for (;;) {
        if (out) for (int i = 0; i < r; i++) out[(size_t)count * r + i] = (uint8_t)c[i];
        count++;
        int i = r - 1;
        while (i >= 0 && c[i] == n - r + i) i--;
        if (i < 0) return count;
        c[i]++;
        for (int j = i + 1; j < r; j++) c[j] = c[j - 1] + 1;
    }
}

static void block_ranks(int j) {
    const uint8_t *p = pts + (size_t)j * K;
    uint32_t *out = ranks + (size_t)j * CT;
    for (int c = 0; c < CT; c++) {
        const uint8_t *q = ct + (size_t)c * T;
        uint64_t r = 0;
        for (int i = 0; i < T; i++) r += binom[p[q[i]]][i + 1];
        out[c] = (uint32_t)r;
    }
}

static void push(uint32_t r) {
    if (!inlist[r]) { inlist[r] = 1; list[listlen++] = r; }
}

static void unrank(uint64_t r, int *s) {  /* s ascending */
    for (int i = T; i >= 1; i--) {
        int c = V - 1;
        while (binom[c][i] > r) c--;
        s[i - 1] = c;
        r -= binom[c][i];
    }
}

static void set_block(int j, const int *p) {  /* p: K distinct points, any order */
    uint8_t *q = pts + (size_t)j * K;
    int tmp[MAXK];
    memcpy(tmp, p, sizeof(int) * K);
    for (int a = 1; a < K; a++) {  /* insertion sort */
        int x = tmp[a], b = a - 1;
        while (b >= 0 && tmp[b] > x) { tmp[b + 1] = tmp[b]; b--; }
        tmp[b + 1] = x;
    }
    mask[j] = 0;
    for (int a = 0; a < K; a++) { q[a] = (uint8_t)tmp[a]; mask[j] |= (mask_t)1 << tmp[a]; }
    block_ranks(j);
}

static void random_block(int *p) {
    int perm[MAXV];
    for (int i = 0; i < V; i++) perm[i] = i;
    for (int i = 0; i < K; i++) {
        int r = i + (int)(rnd() % (uint64_t)(V - i));
        int x = perm[i]; perm[i] = perm[r]; perm[r] = x;
        p[i] = perm[i];
    }
}

static int unique_cover(int j) {  /* t-subsets only block j covers */
    int n = 0;
    const uint32_t *r = ranks + (size_t)j * CT;
    for (int c = 0; c < CT; c++) n += cov[r[c]] == 1;
    return n;
}

/* Score swapping y into block j for every point of the block not in S: delta = (weight of t-subsets with the point that
   only this block covers) - (weight of uncovered t-subsets the block gains with y and without the point). Returns the
   best position (or -1) and its delta. w == NULL means unit weights; points added within `tenure` moves are kept. */
static int score_block(int j, int y, mask_t sm, const uint32_t *w, const uint64_t *added, uint64_t moves, int tenure,
                       int64_t *best_delta) {
    int64_t loss[MAXK], gpos[MAXK], totg = 0;
    const uint8_t *q = pts + (size_t)j * K;
    const uint32_t *rk = ranks + (size_t)j * CT;
    memset(loss, 0, sizeof loss);
    for (int c = 0; c < CT; c++)
        if (cov[rk[c]] == 1) {
            const uint8_t *qq = ct + (size_t)c * T;
            int64_t wt = w ? w[rk[c]] : 1;
            for (int i = 0; i < T; i++) loss[qq[i]] += wt;
        }
    memset(gpos, 0, sizeof gpos);
    for (int c = 0; c < CT1; c++) {
        const uint8_t *qq = ct1 + (size_t)c * (T - 1);
        uint64_t rr = 0;
        int idx = 0, placed = 0;
        for (int i = 0; i < T - 1; i++) {
            int pt = q[qq[i]];
            if (!placed && y < pt) { rr += binom[y][idx + 1]; idx++; placed = 1; }
            rr += binom[pt][idx + 1]; idx++;
        }
        if (!placed) rr += binom[y][idx + 1];
        if (!cov[rr]) {
            int64_t wt = w ? w[rr] : 1;
            totg += wt;
            for (int i = 0; i < T - 1; i++) gpos[qq[i]] += wt;
        }
    }
    int bestpos = -1, ties = 0;
    int64_t bd = 0;
    for (int a = 0; a < K; a++) {
        if ((sm >> q[a]) & 1) continue;  /* keep the points of S */
        if (added && added[(size_t)j * V + q[a]] && moves - added[(size_t)j * V + q[a]] < (uint64_t)tenure) continue;
        int64_t d = loss[a] - (totg - gpos[a]);
        if (bestpos < 0 || d < bd) { bd = d; bestpos = a; ties = 1; }
        else if (d == bd && rnd() % (uint64_t)(++ties) == 0) bestpos = a;
    }
    *best_delta = bd;
    return bestpos;
}

int main(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "usage: cover_anneal V K T B SECONDS SEED [START]\n"); return 2; }
    V = atoi(argv[1]); K = atoi(argv[2]); T = atoi(argv[3]); B = atoi(argv[4]);
    double seconds = atof(argv[5]);
    rng_state = strtoull(argv[6], 0, 10) * 0x2545F4914F6CDD1DULL + 1;
    if (V > MAXV || K > MAXK || T > MAXT || T < 2 || K <= T || K >= V || B < 1) { fprintf(stderr, "bad sizes\n"); return 2; }
    double cycle = getenv("CYCLE") ? atof(getenv("CYCLE")) : 10.0;
    double T0 = getenv("T0") ? atof(getenv("T0")) : 2.0, T1 = getenv("T1") ? atof(getenv("T1")) : 0.2;

    for (int n = 0; n <= V; n++) for (int i = 0; i <= T; i++)
        binom[n][i] = i == 0 ? 1 : n == 0 ? 0 : binom[n - 1][i - 1] + binom[n - 1][i];
    N = binom[V][T];
    if (N >= (1ULL << 32)) { fprintf(stderr, "too many t-subsets\n"); return 2; }
    CT = fill_combos(0, K, T); CT1 = fill_combos(0, K, T - 1);
    ct = malloc((size_t)CT * T); ct1 = malloc((size_t)CT1 * (T - 1) + 1);
    fill_combos(ct, K, T); fill_combos(ct1, K, T - 1);

    /* read the start, then trim or pad to B blocks */
    int cap = B, nb = 0;
    int (*start)[MAXK] = malloc(sizeof(int[MAXK]) * (size_t)(cap + 1));
    if (argc > 7) {
        FILE *f = fopen(argv[7], "r");
        if (!f) { fprintf(stderr, "cannot read start\n"); return 2; }
        int x, i = 0;
        while (fscanf(f, "%d", &x) == 1) {
            if (x < 1 || x > V) { fprintf(stderr, "point out of range in start\n"); return 2; }
            if (nb >= cap) { cap *= 2; start = realloc(start, sizeof(int[MAXK]) * (size_t)(cap + 1)); }
            start[nb][i++] = x - 1;
            if (i == K) { i = 0; nb++; }
        }
        fclose(f);
    }
    int M = nb > B ? nb : B;
    pts = malloc((size_t)M * K); mask = malloc(sizeof(mask_t) * M); ranks = malloc(sizeof(uint32_t) * (size_t)M * CT);
    cov = calloc(N, sizeof(uint16_t)); inlist = calloc(N, 1); list = malloc(sizeof(uint32_t) * N);
    if (!pts || !mask || !ranks || !cov || !inlist || !list) { fprintf(stderr, "out of memory\n"); return 2; }
    int p[MAXK];
    for (int j = 0; j < M; j++) {
        if (j < nb) set_block(j, start[j]); else { random_block(p); set_block(j, p); }
        for (int c = 0; c < CT; c++) cov[ranks[(size_t)j * CT + c]]++;
    }
    while (M > B) {  /* drop the block that covers the fewest t-subsets on its own */
        int worst = 0, wu = unique_cover(0);
        for (int j = 1; j < M; j++) { int u = unique_cover(j); if (u < wu) { wu = u; worst = j; } }
        for (int c = 0; c < CT; c++) cov[ranks[(size_t)worst * CT + c]]--;
        M--;
        if (worst != M) {
            memcpy(pts + (size_t)worst * K, pts + (size_t)M * K, K);
            mask[worst] = mask[M];
            memcpy(ranks + (size_t)worst * CT, ranks + (size_t)M * CT, sizeof(uint32_t) * CT);
        }
    }
    uncovered = 0;
    for (uint64_t r = 0; r < N; r++) if (!cov[r]) { uncovered++; push((uint32_t)r); }

    /* Weighted mode (WEIGHTED=1): every t-subset carries a weight, the best swap by weighted score is always taken
       if it does not raise the weighted count of uncovered t-subsets (a point just added to a block may not leave it
       for TABU moves); when no such swap exists, every uncovered t-subset gets heavier (the breakout method / clause
       weighting from SAT local search), so the search climbs out of local minima.
       Shrink mode (SHRINK=prefix): each time every t-subset is covered, the covering is written to prefix_<b>.txt,
       the block covering the fewest t-subsets on its own is dropped, and the search goes on with one block fewer. */
    int weighted = getenv("WEIGHTED") && atoi(getenv("WEIGHTED"));
    int tenure = getenv("TABU") ? atoi(getenv("TABU")) : 10;
    uint64_t bump_all = getenv("BUMPALL") ? strtoull(getenv("BUMPALL"), 0, 10) : 4096;
    const char *shrink = getenv("SHRINK");
    uint32_t *w = 0;
    uint64_t *added = 0;
    if (weighted) {
        w = malloc(sizeof(uint32_t) * N); added = calloc((size_t)B * V, sizeof(uint64_t));
        if (!w || !added) { fprintf(stderr, "out of memory\n"); return 2; }
        for (uint64_t r = 0; r < N; r++) w[r] = 1;
    }

    uint8_t *best = malloc((size_t)B * K);
    memcpy(best, pts, (size_t)B * K);
    uint64_t best_unc = uncovered, moves = 0;
    double t_start = now(), t_now = t_start, temp = T0;
    int s[MAXT];
    int *cand = malloc(sizeof(int) * B);
    int covered_at = 0;  /* fewest blocks that covered everything (shrink mode) */

    for (;;) {
        if (uncovered == 0) {
            if (!shrink || B == 1) break;
            char path[4096];
            snprintf(path, sizeof path, "%s_%d.txt", shrink, B);
            FILE *f = fopen(path, "w");
            if (f) {
                for (int j2 = 0; j2 < B; j2++) {
                    for (int a = 0; a < K; a++) fprintf(f, a ? " %d" : "%d", pts[(size_t)j2 * K + a] + 1);
                    fprintf(f, "\n");
                }
                fclose(f);
            }
            covered_at = B;
            fprintf(stderr, "covered with %d blocks after %.1f s\n", B, now() - t_start);
            int worst = 0, wu = unique_cover(0);
            for (int j2 = 1; j2 < B; j2++) { int u = unique_cover(j2); if (u < wu) { wu = u; worst = j2; } }
            for (int c = 0; c < CT; c++) {
                uint32_t r2 = ranks[(size_t)worst * CT + c];
                if (--cov[r2] == 0) { uncovered++; push(r2); }
            }
            B--;
            if (worst != B) {
                memcpy(pts + (size_t)worst * K, pts + (size_t)B * K, K);
                mask[worst] = mask[B];
                memcpy(ranks + (size_t)worst * CT, ranks + (size_t)B * CT, sizeof(uint32_t) * CT);
                if (added) memcpy(added + (size_t)worst * V, added + (size_t)B * V, sizeof(uint64_t) * V);
            }
            best_unc = uncovered;
            memcpy(best, pts, (size_t)B * K);
        }
        if ((moves & 255) == 0) {
            t_now = now();
            double el = t_now - t_start;
            if (el >= seconds) break;
            double frac = fmod(el, cycle) / cycle;
            temp = T0 * pow(T1 / T0, frac);
        }
        moves++;
        /* draw an uncovered t-subset; in weighted mode the heaviest of a few draws */
        uint32_t r = 0;
        for (int draw = 0; draw < (weighted ? 4 : 1); draw++) {
            uint32_t r2;
            for (;;) {
                uint64_t i = rnd() % listlen;
                r2 = list[i];
                if (!cov[r2]) break;
                inlist[r2] = 0; list[i] = list[--listlen];
            }
            if (draw == 0 || w[r2] > w[r]) r = r2;
        }
        unrank(r, s);
        mask_t sm = 0;
        for (int i = 0; i < T; i++) sm |= (mask_t)1 << s[i];
        /* a block holding the most points of S */
        int bestc = -1, nc = 0;
        for (int j = 0; j < B; j++) {
            int c = popc(mask[j] & sm);
            if (c > bestc) { bestc = c; nc = 0; }
            if (c == bestc) cand[nc++] = j;
        }
        /* candidate swaps: plain mode scores one block holding most of S; weighted mode the best of up to 4 */
        int j = -1, y = -1, bestpos = -1;
        int64_t bd = 0;
        for (int tries = 0; tries < (weighted ? 4 : 1) && tries < nc; tries++) {
            int pick = tries + (int)(rnd() % (uint64_t)(nc - tries));
            int jj = cand[pick]; cand[pick] = cand[tries]; cand[tries] = jj;
            int miss[MAXT], nm = 0;
            for (int i = 0; i < T; i++) if (!((mask[jj] >> s[i]) & 1)) miss[nm++] = s[i];
            int yy = miss[rnd() % (uint64_t)nm];
            int64_t d;
            int pos = score_block(jj, yy, sm, w, added, moves, tenure, &d);
            if (pos >= 0 && (bestpos < 0 || d < bd)) { bestpos = pos; bd = d; j = jj; y = yy; }
        }
        if (weighted && (bestpos < 0 || bd > 0)) {  /* no swap that keeps the weighted score: breakout */
            uint32_t top = 0;
            if (uncovered <= bump_all) {
                for (uint64_t i = 0; i < listlen;) {
                    uint32_t r2 = list[i];
                    if (cov[r2]) { inlist[r2] = 0; list[i] = list[--listlen]; continue; }
                    if (++w[r2] > top) top = w[r2];
                    i++;
                }
            } else top = ++w[r];
            if (top > (1u << 30)) for (uint64_t r2 = 0; r2 < N; r2++) w[r2] = (w[r2] >> 1) | 1;
            continue;
        }
        if (bestpos < 0) continue;
        if (!weighted && bd > 0 && rnd01() >= exp(-(double)bd / temp)) continue;
        const uint8_t *q = pts + (size_t)j * K;
        const uint32_t *rk = ranks + (size_t)j * CT;
        /* apply: remove the point at bestpos, add y */
        for (int c = 0; c < CT; c++) {
            const uint8_t *qq = ct + (size_t)c * T;
            int has = 0;
            for (int i = 0; i < T; i++) has |= qq[i] == bestpos;
            if (has && --cov[rk[c]] == 0) { uncovered++; push(rk[c]); }
        }
        int np[MAXK], n = 0;
        for (int a = 0; a < K; a++) if (a != bestpos) np[n++] = q[a];
        np[n++] = y;
        set_block(j, np);
        if (weighted) added[(size_t)j * V + y] = moves;
        rk = ranks + (size_t)j * CT;
        for (int c = 0; c < CT; c++) {  /* every t-subset of the new block with y */
            const uint8_t *qq = ct + (size_t)c * T;
            int has = 0;
            for (int i = 0; i < T; i++) has |= pts[(size_t)j * K + qq[i]] == y;
            if (has && cov[rk[c]]++ == 0) uncovered--;
        }
        if (uncovered < best_unc) { best_unc = uncovered; memcpy(best, pts, (size_t)B * K); }
    }
    if (uncovered < best_unc || uncovered == 0) { best_unc = uncovered; memcpy(best, pts, (size_t)B * K); }
    printf("uncovered %llu moves %llu seconds %.1f blocks %d covered_at %d\n", (unsigned long long)best_unc,
           (unsigned long long)moves, now() - t_start, B, covered_at);
    for (int j2 = 0; j2 < B; j2++) {
        for (int a = 0; a < K; a++) printf(a ? " %d" : "%d", best[(size_t)j2 * K + a] + 1);
        printf("\n");
    }
    return best_unc ? 1 : 0;
}
