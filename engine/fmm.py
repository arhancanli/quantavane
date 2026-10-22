"""Fast matrix multiplication schemes: read the catalogue's Maple files as data and check schemes exactly.

A scheme for <n x m x p> is a list of triads (alpha, beta, gamma), alpha n x m, beta m x p, gamma p x n, such that
A.B = sum over triads of trace(alpha^T A) * trace(beta^T B) * gamma^T for all matrices A, B. check() verifies the
equivalent tensor identity exactly in rational arithmetic: for every entry pair (a_ij, b_jl) and every output c_il,
the summed coefficient is 1 when j = j' and the output is (i, l), and 0 otherwise. Files are parsed with regular
expressions only; nothing in them is ever executed."""
import re
from fractions import Fraction

MATRIX = re.compile(r"Matrix\((\d+),\s*(\d+),\s*\[(\[.*?\])\]\)")


def _matrix(rows, cols, body):
    vals = [Fraction(x.strip()) for x in re.findall(r"[-+]?\d+(?:/\d+)?", body)]
    if len(vals) != rows * cols:
        raise ValueError(f"expected {rows * cols} entries, got {len(vals)}")
    return [vals[r * cols:(r + 1) * cols] for r in range(rows)]


def parse(text):
    """((n, m, p), [(alpha, beta, gamma), ...]) from a catalogue *_tensor.mpl file."""
    body = text[text.index("TriadSet("):]
    mats = [(int(r), int(c), _matrix(int(r), int(c), b)) for r, c, b in MATRIX.findall(body)]
    if len(mats) % 3:
        raise ValueError("matrices do not come in triads")
    triads = [tuple(mats[i + q][2] for q in range(3)) for i in range(0, len(mats), 3)]
    n, m = mats[0][0], mats[0][1]
    p = mats[1][1]
    return (n, m, p), triads


def check(fmt, triads):
    """True when the scheme computes A.B exactly (over the rationals)."""
    n, m, p = fmt
    for a in range(n * m):
        i, j = divmod(a, m)
        for b in range(m * p):
            j2, l = divmod(b, p)
            acc = {}
            for alpha, beta, gamma in triads:
                s = alpha[i][j] * beta[j2][l]
                if s:
                    for l2 in range(p):
                        for i2 in range(n):
                            g = gamma[l2][i2]
                            if g:
                                acc[(i2, l2)] = acc.get((i2, l2), 0) + s * g
            want = {(i, l): 1} if j == j2 else {}
            if {k: v for k, v in acc.items() if v} != want:
                return False
    return True


def ring(triads):
    """'Z' when every coefficient is an integer, else the largest denominator, e.g. 'Q (1/8)'."""
    dens = {x.denominator for t in triads for mat in t for row in mat for x in row}
    return "Z" if dens == {1} else f"Q (1/{max(dens)})"
