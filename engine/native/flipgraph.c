/* Flip-graph search for matrix multiplication schemes over GF(2) (Kauers and Moosbauer, 2022).

   Multiplying an n x m matrix A by an m x p matrix B is the tensor T = sum over i,j,l of a(i,j) x b(j,l) x c(l,i).
   A scheme of rank r writes T as a sum of r rank-one terms u x v x w, with u a 0/1 vector over the n*m entries of A,
   v over the m*p entries of B and w over the p*n entries of C; each term is one multiplication.
   A flip takes two terms that share a factor, (x, y1, z1) + (x, y2, z2) = (x, y1 + y2, z1) + (x, y2, z2 + z1), which
   keeps the sum exact. When a factor becomes zero the term vanishes, and two terms that share two factors merge into
   one: both lower the rank. A plus transition, (x1, y, z) + (x2, y2, z2) -> (x1 + x2, y, z) + (x2, y, z) + (x2, y2, z2),
   raises the rank by one to escape a dead end. The walk starts from the standard algorithm (rank n*m*p) or a scheme
   read from a file, takes random flips, reduces whenever it can, and stops when the rank reaches the target.

   flipgraph N M P TARGET SECONDS SEED [START]
   START: one term per line, three hex numbers u v w (bit i*m+j of u is a(i,j), bit j*p+l of v is b(j,l), bit
   l*n+i of w is c(l,i)). Prints "rank R flips F seconds S" then the best scheme found, in the same format.
   Environment: PLUS (flips without a reduction before a plus transition, default 1000000), PLUSCAP (plus transitions
   only while the rank is below the best rank found plus this, default 1), RESET (flips without a new best before the
   walk restarts from the best scheme, default 50000000). */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXR 2048

static int n, m, p;
static uint64_t U[MAXR], Vv[MAXR], W[MAXR];
static int R;
static uint64_t rng_state;

static uint64_t rnd(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double now(void) { return (double)clock() / CLOCKS_PER_SEC; }

static uint64_t *fac(int pos, int t) { return pos == 0 ? &U[t] : pos == 1 ? &Vv[t] : &W[t]; }

static void drop(int t) {  /* remove term t (swap with last) */
    R--;
    U[t] = U[R]; Vv[t] = Vv[R]; W[t] = W[R];
}

/* Reductions: remove zero terms and merge terms sharing two factors, until none apply. Returns how many removed. */
static int reduce(void) {
    int removed = 0, again = 1;
    while (again) {
        again = 0;
        for (int a = 0; a < R; a++) {
            if (!U[a] || !Vv[a] || !W[a]) { drop(a); removed++; again = 1; break; }
        }
        if (again) continue;
        for (int a = 0; a < R && !again; a++)
            for (int b = a + 1; b < R && !again; b++) {
                int su = U[a] == U[b], sv = Vv[a] == Vv[b], sw = W[a] == W[b];
                if (su + sv + sw >= 2) {
                    if (su && sv) W[a] ^= W[b];
                    else if (su && sw) Vv[a] ^= Vv[b];
                    else U[a] ^= U[b];
                    drop(b); removed++; again = 1;
                }
            }
    }
    return removed;
}

/* Exact check over GF(2): the scheme's tensor equals the matrix multiplication tensor. */
static int valid(void) {
    int NA = n * m, NB = m * p;
    for (int x = 0; x < NA; x++)
        for (int y = 0; y < NB; y++) {
            uint64_t acc = 0;
            for (int t = 0; t < R; t++)
                if (((U[t] >> x) & 1) && ((Vv[t] >> y) & 1)) acc ^= W[t];
            int i = x / m, j = x % m, j2 = y / p, l = y % p;
            uint64_t want = (j == j2) ? (1ULL << (l * n + i)) : 0;
            if (acc != want) return 0;
        }
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "usage: flipgraph N M P TARGET SECONDS SEED [START]\n"); return 2; }
    n = atoi(argv[1]); m = atoi(argv[2]); p = atoi(argv[3]);
    int target = atoi(argv[4]);
    double seconds = atof(argv[5]);
    rng_state = strtoull(argv[6], 0, 10) * 0x2545F4914F6CDD1DULL + 1;
    if (n * m > 64 || m * p > 64 || p * n > 64 || n < 1 || m < 1 || p < 1) { fprintf(stderr, "format too large\n"); return 2; }
    long plus_after = getenv("PLUS") ? atol(getenv("PLUS")) : 1000000;
    R = 0;
    if (argc > 7) {
        FILE *f = fopen(argv[7], "r");
        if (!f) { fprintf(stderr, "cannot read start\n"); return 2; }
        unsigned long long a, b, c;
        while (fscanf(f, "%llx %llx %llx", &a, &b, &c) == 3 && R < MAXR) { U[R] = a; Vv[R] = b; W[R] = c; R++; }
        fclose(f);
    } else {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                for (int l = 0; l < p; l++) {
                    U[R] = 1ULL << (i * m + j); Vv[R] = 1ULL << (j * p + l); W[R] = 1ULL << (l * n + i); R++;
                }
    }
    if (!valid()) { fprintf(stderr, "start is not a valid scheme\n"); return 2; }
    reduce();
    int pluscap = getenv("PLUSCAP") ? atoi(getenv("PLUSCAP")) : 1;
    long long reset_after = getenv("RESET") ? atoll(getenv("RESET")) : 50000000LL;
    long long last_best = 0;
    static uint64_t bU[MAXR], bV[MAXR], bW[MAXR];
    int bestR = R;
    memcpy(bU, U, sizeof(uint64_t) * R); memcpy(bV, Vv, sizeof(uint64_t) * R); memcpy(bW, W, sizeof(uint64_t) * R);
    long long flips = 0, since = 0;
    double t0 = now();
    while (bestR > target) {
        if ((flips & 4095) == 0 && now() - t0 >= seconds) break;
        /* a random term and factor position, and another term with the same factor there */
        int a = (int)(rnd() % (uint64_t)R), pos = (int)(rnd() % 3);
        uint64_t x = *fac(pos, a);
        int cand[MAXR], nc = 0;
        for (int b = 0; b < R; b++) if (b != a && *fac(pos, b) == x) cand[nc++] = b;
        if (!nc) {
            if (++since > plus_after && R < bestR + pluscap && R >= 2) {  /* plus transition */
                int b = (int)(rnd() % (uint64_t)R);
                if (b != a) {
                    /* (x1,y,z) + (x2,y2,z2) -> (x1+x2,y,z) + (x2,y,z) + (x2,y2,z2), rotating which factor plays x */
                    int q = (int)(rnd() % 3);
                    uint64_t *xa = fac(q, a), *xb = fac(q, b);
                    U[R] = U[a]; Vv[R] = Vv[a]; W[R] = W[a];
                    *fac(q, R) = *xb;
                    *xa ^= *xb;
                    R++;
                    since = 0;
                    reduce();
                }
            }
            continue;
        }
        int b = cand[rnd() % (uint64_t)nc];
        flips++; since++;
        int q1 = (pos + 1) % 3, q2 = (pos + 2) % 3;
        if (rnd() & 1) { int tmp = q1; q1 = q2; q2 = tmp; }
        /* (x, y1, z1) + (x, y2, z2) = (x, y1 + y2, z1) + (x, y2, z2 + z1), with y and z the two other positions */
        *fac(q1, a) ^= *fac(q1, b);
        *fac(q2, b) ^= *fac(q2, a);
        if (reduce()) since = 0;
        if (R < bestR) {
            bestR = R; last_best = flips;
            memcpy(bU, U, sizeof(uint64_t) * R); memcpy(bV, Vv, sizeof(uint64_t) * R); memcpy(bW, W, sizeof(uint64_t) * R);
        } else if (flips - last_best > reset_after) {  /* restart the walk from the best scheme */
            R = bestR; last_best = flips; since = 0;
            memcpy(U, bU, sizeof(uint64_t) * R); memcpy(Vv, bV, sizeof(uint64_t) * R); memcpy(W, bW, sizeof(uint64_t) * R);
        }
    }
    R = bestR;
    memcpy(U, bU, sizeof(uint64_t) * R); memcpy(Vv, bV, sizeof(uint64_t) * R); memcpy(W, bW, sizeof(uint64_t) * R);
    if (!valid()) { fprintf(stderr, "internal error: best scheme is not valid\n"); return 3; }
    printf("rank %d flips %lld seconds %.1f\n", R, flips, now() - t0);
    for (int t = 0; t < R; t++) printf("%llx %llx %llx\n", (unsigned long long)U[t], (unsigned long long)Vv[t], (unsigned long long)W[t]);
    return R <= target ? 0 : 1;
}
