/* Simulated annealing for lottery covering designs (v,k,t,m).

   A (v,k,t,m) covering is a set of k-point blocks from {0..v-1} such that every m-subset of the points meets some block
   in at least t points. An m-subset is covered exactly when one of its t-subsets lies inside a block, so the search
   keeps two counts: for every t-subset, how many blocks contain it (cntT); for every m-subset, how many of its
   t-subsets are inside some block (cM). An m-subset is uncovered when its cM is zero.
   With b fixed, the search minimises the number of uncovered m-subsets. Each step takes a random uncovered m-subset S
   and a block holding the most points of S, adds a point y of S the block lacks, and scores removing each point x of
   the block outside S by the exact change in uncovered m-subsets (t-subsets that lose their only block and t-subsets
   that gain one, pushed up to every m-subset containing them). The best x is taken if it does not make things worse,
   otherwise with probability exp(-delta / temperature); the temperature falls from T0 to T1 over each cycle.

   lotto_anneal V K T M B SECONDS SEED [START]
   Prints "uncovered U moves N seconds S blocks B" and then the best blocks found (1-based), one per line.
   Environment: CYCLE (seconds per temperature cycle, default 10), T0 (default 2.0), T1 (default 0.2). */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXV 100
#define MAXK 32
#define MAXT 8
#define MAXM 16

typedef unsigned __int128 mask_t;

static int V, K, T, M, B;
static uint64_t binom[MAXV + 1][MAXM + 1];
static uint64_t NT, NM;
static uint16_t *cntT, *cM;
static uint8_t *inlist;
static uint32_t *list;
static uint64_t listlen, uncovered;
static uint8_t *pts;
static mask_t *mask;
static uint32_t *ranks;       /* B x CT ranks of each block's t-subsets */
static uint8_t *ct;           /* position combinations: CT x T */
static int CT;
static int16_t *D;            /* scratch: change of cM per m-subset during a move's scoring */
static uint32_t *touched;
static uint64_t ntouched, touched_cap;
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

static int fill_combos(uint8_t *out, int n, int r) {
    int c[MAXK + 1], count = 0;
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

static uint32_t rank_sorted(const int *s, int r) {
    uint64_t x = 0;
    for (int i = 0; i < r; i++) x += binom[s[i]][i + 1];
    return (uint32_t)x;
}

static void unrank(uint64_t r, int *s, int n) {  /* s ascending, n-subset */
    for (int i = n; i >= 1; i--) {
        int c = V - 1;
        while (binom[c][i] > r) c--;
        s[i - 1] = c;
        r -= binom[c][i];
    }
}

/* Call f(rank of M') for every m-subset M' containing the sorted t-subset tt. */
#define FOR_SUPERSETS(tt, BODY)                                                                     \
    do {                                                                                            \
        int comp_[MAXV], nc_ = 0, ti_ = 0, R_ = M - T, idx_[MAXM + 1];                               \
        for (int p_ = 0; p_ < V; p_++) {                                                            \
            if (ti_ < T && tt[ti_] == p_) { ti_++; continue; }                                      \
            comp_[nc_++] = p_;                                                                      \
        }                                                                                           \
        for (int i_ = 0; i_ < R_; i_++) idx_[i_] = i_;                                              \
        for (;;) {                                                                                  \
            uint64_t r_ = 0;                                                                        \
            int a_ = 0, b_ = 0, pos_ = 0;                                                           \
            while (a_ < T || b_ < R_) {                                                             \
                int pa_ = a_ < T ? tt[a_] : 1 << 30, pb_ = b_ < R_ ? comp_[idx_[b_]] : 1 << 30;      \
                if (pa_ < pb_) { r_ += binom[pa_][pos_ + 1]; a_++; } else { r_ += binom[pb_][pos_ + 1]; b_++; } \
                pos_++;                                                                             \
            }                                                                                       \
            uint32_t mr = (uint32_t)r_;                                                             \
            BODY;                                                                                   \
            int i_ = R_ - 1;                                                                        \
            while (i_ >= 0 && idx_[i_] == nc_ - R_ + i_) i_--;                                      \
            if (i_ < 0) break;                                                                      \
            idx_[i_]++;                                                                             \
            for (int j_ = i_ + 1; j_ < R_; j_++) idx_[j_] = idx_[j_ - 1] + 1;                       \
        }                                                                                           \
    } while (0)

static void push(uint32_t r) {
    if (!inlist[r]) { inlist[r] = 1; list[listlen++] = r; }
}

static void touch(uint32_t mr, int d) {
    if (D[mr] == 0) {
        if (ntouched == touched_cap) { touched_cap *= 2; touched = realloc(touched, sizeof(uint32_t) * touched_cap); }
        touched[ntouched++] = mr;
    }
    D[mr] += d;
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

static void set_block(int j, const int *p) {
    uint8_t *q = pts + (size_t)j * K;
    int tmp[MAXK];
    memcpy(tmp, p, sizeof(int) * K);
    for (int a = 1; a < K; a++) {
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

/* t-subset counts changed by +1/-1 for one t-subset (sorted tt, rank r): keep cM and the uncovered count exact. */
static void add_t(const int *tt, uint32_t r, int d) {
    if (d > 0) {
        if (cntT[r]++ != 0) return;
        FOR_SUPERSETS(tt, { if (cM[mr]++ == 0) uncovered--; });
    } else {
        if (--cntT[r] != 0) return;
        FOR_SUPERSETS(tt, { if (--cM[mr] == 0) { uncovered++; push(mr); } });
    }
}

/* Exact change in uncovered m-subsets if block j drops its point at position pos and gains point y. */
static int64_t score(int j, int pos, int y) {
    const uint8_t *q = pts + (size_t)j * K;
    const uint32_t *rk = ranks + (size_t)j * CT;
    ntouched = 0;
    int x = q[pos];
    for (int c = 0; c < CT; c++) {  /* t-subsets with x that only this block covers */
        const uint8_t *qq = ct + (size_t)c * T;
        int has = 0, tt[MAXT];
        for (int i = 0; i < T; i++) { has |= qq[i] == pos; tt[i] = q[qq[i]]; }
        if (has && cntT[rk[c]] == 1) FOR_SUPERSETS(tt, touch(mr, -1));
    }
    /* t-subsets of the new block with y that no block covers yet */
    int nb[MAXK], n = 0;
    for (int a = 0; a < K; a++) if (a != pos) nb[n++] = q[a];
    int c1[MAXT + 1];
    for (int i = 0; i < T - 1; i++) c1[i] = i;
    for (;;) {
        int tt[MAXT], w = 0, placed = 0;
        for (int i = 0; i < T - 1; i++) {
            int p = nb[c1[i]];
            if (!placed && y < p) { tt[w++] = y; placed = 1; }
            tt[w++] = p;
        }
        if (!placed) tt[w++] = y;
        uint32_t r = rank_sorted(tt, T);
        if (cntT[r] == 0) FOR_SUPERSETS(tt, touch(mr, +1));
        int i = T - 2;
        while (i >= 0 && c1[i] == (K - 1) - (T - 1) + i) i--;
        if (i < 0) break;
        c1[i]++;
        for (int jx = i + 1; jx < T - 1; jx++) c1[jx] = c1[jx - 1] + 1;
    }
    int64_t delta = 0;
    for (uint64_t i = 0; i < ntouched; i++) {
        uint32_t mr = touched[i];
        int before = cM[mr], after = cM[mr] + D[mr];
        delta += (after == 0) - (before == 0);
        D[mr] = 0;
    }
    (void)x;
    return delta;
}

int main(int argc, char **argv) {
    if (argc < 8) { fprintf(stderr, "usage: lotto_anneal V K T M B SECONDS SEED [START]\n"); return 2; }
    V = atoi(argv[1]); K = atoi(argv[2]); T = atoi(argv[3]); M = atoi(argv[4]); B = atoi(argv[5]);
    double seconds = atof(argv[6]);
    rng_state = strtoull(argv[7], 0, 10) * 0x2545F4914F6CDD1DULL + 1;
    if (V > MAXV || K > MAXK || T > MAXT || M > MAXM || T < 1 || M < T || K < T || K >= V || M > V || B < 1) {
        fprintf(stderr, "bad sizes\n"); return 2;
    }
    double cycle = getenv("CYCLE") ? atof(getenv("CYCLE")) : 10.0;
    double T0 = getenv("T0") ? atof(getenv("T0")) : 2.0, T1 = getenv("T1") ? atof(getenv("T1")) : 0.2;
    for (int n = 0; n <= V; n++) for (int i = 0; i <= MAXM; i++)
        binom[n][i] = i == 0 ? 1 : n == 0 ? 0 : binom[n - 1][i - 1] + binom[n - 1][i];
    NT = binom[V][T]; NM = binom[V][M];
    if (NM >= (1ULL << 31) || NT >= (1ULL << 31)) { fprintf(stderr, "too many subsets\n"); return 2; }
    CT = fill_combos(0, K, T);
    ct = malloc((size_t)CT * T);
    fill_combos(ct, K, T);

    int cap = B, nb = 0;
    int (*start)[MAXK] = malloc(sizeof(int[MAXK]) * (size_t)(cap + 1));
    if (argc > 8) {
        FILE *f = fopen(argv[8], "r");
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
    pts = malloc((size_t)B * K); mask = malloc(sizeof(mask_t) * B); ranks = malloc(sizeof(uint32_t) * (size_t)B * CT);
    cntT = calloc(NT, sizeof(uint16_t)); cM = calloc(NM, sizeof(uint16_t)); inlist = calloc(NM, 1);
    list = malloc(sizeof(uint32_t) * NM); D = calloc(NM, sizeof(int16_t));
    touched_cap = 1 << 16; touched = malloc(sizeof(uint32_t) * touched_cap);
    if (!pts || !mask || !ranks || !cntT || !cM || !inlist || !list || !D || !touched) { fprintf(stderr, "out of memory\n"); return 2; }
    int p[MAXK];
    for (int j = 0; j < B; j++) {  /* extra start blocks beyond B are ignored; missing ones are random */
        if (j < nb) set_block(j, start[j]); else { random_block(p); set_block(j, p); }
    }
    uncovered = NM;
    for (int j = 0; j < B; j++)
        for (int c = 0; c < CT; c++) {
            const uint8_t *qq = ct + (size_t)c * T;
            int tt[MAXT];
            for (int i = 0; i < T; i++) tt[i] = pts[(size_t)j * K + qq[i]];
            add_t(tt, ranks[(size_t)j * CT + c], +1);
        }
    for (uint64_t r = 0; r < NM; r++) if (!cM[r]) push((uint32_t)r);

    uint8_t *best = malloc((size_t)B * K);
    memcpy(best, pts, (size_t)B * K);
    uint64_t best_unc = uncovered, moves = 0;
    double t_start = now(), temp = T0;
    int *cand = malloc(sizeof(int) * B), s[MAXM];
    while (uncovered > 0) {
        if ((moves & 63) == 0) {
            double el = now() - t_start;
            if (el >= seconds) break;
            temp = T0 * pow(T1 / T0, fmod(el, cycle) / cycle);
        }
        moves++;
        uint32_t r;
        for (;;) {
            uint64_t i = rnd() % listlen;
            r = list[i];
            if (!cM[r]) break;
            inlist[r] = 0; list[i] = list[--listlen];
        }
        unrank(r, s, M);
        mask_t sm = 0;
        for (int i = 0; i < M; i++) sm |= (mask_t)1 << s[i];
        int bestc = -1, nc = 0;
        for (int j = 0; j < B; j++) {
            int c = popc(mask[j] & sm);
            if (c > bestc) { bestc = c; nc = 0; }
            if (c == bestc) cand[nc++] = j;
        }
        int j = cand[rnd() % (uint64_t)nc];
        int miss[MAXM], nm = 0;
        for (int i = 0; i < M; i++) if (!((mask[j] >> s[i]) & 1)) miss[nm++] = s[i];
        if (!nm) continue;
        int y = miss[rnd() % (uint64_t)nm];
        const uint8_t *q = pts + (size_t)j * K;
        int bestpos = -1, ties = 0;
        int64_t bd = 0;
        for (int a = 0; a < K; a++) {
            if ((sm >> q[a]) & 1) continue;  /* keep the block's points in S */
            int64_t d = score(j, a, y);
            if (bestpos < 0 || d < bd) { bd = d; bestpos = a; ties = 1; }
            else if (d == bd && rnd() % (uint64_t)(++ties) == 0) bestpos = a;
        }
        if (bestpos < 0) continue;
        if (bd > 0 && rnd01() >= exp(-(double)bd / temp)) continue;
        /* apply: drop the point at bestpos (t-subsets with it lose this block), add y (t-subsets with y gain it) */
        const uint32_t *rk = ranks + (size_t)j * CT;
        for (int c = 0; c < CT; c++) {
            const uint8_t *qq = ct + (size_t)c * T;
            int has = 0, tt[MAXT];
            for (int i = 0; i < T; i++) { has |= qq[i] == bestpos; tt[i] = q[qq[i]]; }
            if (has) add_t(tt, rk[c], -1);
        }
        int np[MAXK], n = 0;
        for (int a = 0; a < K; a++) if (a != bestpos) np[n++] = q[a];
        np[n++] = y;
        set_block(j, np);
        rk = ranks + (size_t)j * CT;
        for (int c = 0; c < CT; c++) {
            const uint8_t *qq = ct + (size_t)c * T;
            int has = 0, tt[MAXT];
            for (int i = 0; i < T; i++) { tt[i] = pts[(size_t)j * K + qq[i]]; has |= tt[i] == y; }
            if (has) add_t(tt, rk[c], +1);
        }
        if (uncovered < best_unc) { best_unc = uncovered; memcpy(best, pts, (size_t)B * K); }
    }
    if (uncovered < best_unc || uncovered == 0) { best_unc = uncovered; memcpy(best, pts, (size_t)B * K); }
    printf("uncovered %llu moves %llu seconds %.1f blocks %d\n", (unsigned long long)best_unc,
           (unsigned long long)moves, now() - t_start, B);
    for (int j2 = 0; j2 < B; j2++) {
        for (int a = 0; a < K; a++) printf(a ? " %d" : "%d", best[(size_t)j2 * K + a] + 1);
        printf("\n");
    }
    return best_unc ? 1 : 0;
}
