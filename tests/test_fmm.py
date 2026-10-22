import unittest

from engine import fmm

# Strassen's seven products for 2x2 times 2x2: alpha (2x2), beta (2x2), gamma (2x2, indexed [l][i] so that
# A.B = sum trace(alpha^T A) * trace(beta^T B) * gamma^T).
STRASSEN = [
    ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
    ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 1], [0, -1]]),
    ([[1, 0], [0, 0]], [[0, 1], [0, -1]], [[0, 0], [1, 1]]),
    ([[0, 0], [0, 1]], [[-1, 0], [1, 0]], [[1, 1], [0, 0]]),
    ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[-1, 0], [1, 0]]),
    ([[-1, 0], [1, 0]], [[1, 1], [0, 0]], [[0, 0], [0, 1]]),
    ([[0, 1], [0, -1]], [[0, 0], [1, 1]], [[1, 0], [0, 0]]),
]


def maple(triads, n, m, p):
    mat = lambda r, c, rows: f"Matrix({r}, {c}, [" + ",".join("[" + ",".join(map(str, row)) + "]" for row in rows) + "])"
    body = ", ".join(f"Triad([{mat(n, m, a)}, {mat(m, p, b)}, {mat(p, n, g)}])" for a, b, g in triads)
    return f"A:=Matrix(2,2,[]):\nTensor:=TriadSet([{body}]):\n"


class FMM(unittest.TestCase):
    def test_strassen_parses_and_checks_exactly_over_the_integers(self):
        fmt, triads = fmm.parse(maple(STRASSEN, 2, 2, 2))
        self.assertEqual((fmt, len(triads)), ((2, 2, 2), 7))
        self.assertEqual(fmm.ring(triads), "Z")
        self.assertTrue(fmm.check(fmt, triads))

    def test_a_wrong_sign_is_rejected(self):
        bad = [list(t) for t in STRASSEN]
        bad[1][2] = [[0, 1], [0, 1]]
        fmt, triads = fmm.parse(maple(bad, 2, 2, 2))
        self.assertFalse(fmm.check(fmt, triads))


if __name__ == "__main__":
    unittest.main()
