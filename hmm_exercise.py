"""Exercise 07 - Hidden Markov Models."""

import itertools
import json
import math

# ==========Task 1: The model in code========
def log2(p):
    if p < 0:
        raise ValueError(f"probability must be >= 0, got {p}")
    if p == 0:
        return -math.inf
    else:
        return math.log2(p)

class HMM:
    """Hidden Markov Model."""
    def __init__(self, states, alphabet, pi, a, e):
        self.states = list(states)
        self.alphabet = list(alphabet)
        self.pi = pi
        self.a = a
        self.e = e

        # consistency checks
        S, A = set(self.states), set(self.alphabet)
        if set(pi) != S:
            raise ValueError("pi keys do not match states")
        if set(a) != S or set(e) != S:
            raise ValueError("a or e rows do not match states")
        if not math.isclose(sum(pi.values()), 1.0):
            raise ValueError("pi does not sum to 1")

        for k in self.states:
            if set(a[k]) != S:
                raise ValueError(f"a[{k}] keys do not match states")
            if set(e[k]) != A:
                raise ValueError(f"e[{k}] keys do not match alphabet")
            if not math.isclose(sum(a[k].values()), 1.0):
                raise ValueError(f"a[{k}] does not sum to 1")
            if not math.isclose(sum(e[k].values()), 1.0):
                raise ValueError(f"e[{k}] does not sum to 1")

        # log2 tables, computed once
        #pi
        self.log_pi = {}
        for k in self.states:
            self.log_pi[k] = log2(pi[k])
        #a
        self.log_a = {}
        for k in self.states:
            self.log_a[k] = {}
            for l in self.states:
                self.log_a[k][l] = log2(a[k][l])

        #e
        self.log_e = {}
        for k in self.states:
            self.log_e[k] = {}
            for c in self.alphabet:
                self.log_e[k][c] = log2(e[k][c])

    def _check(self, x, z):
        if len(x) != len(z):
            raise ValueError("x and z must have the same length")
        if len(x) == 0:
            raise ValueError("empty sequence")

    def joint_probability(self, x, z):
        # P(x, z) as a plain product
        self._check(x, z)
        p = self.pi[z[0]] * self.e[z[0]][x[0]]
        for i in range(1, len(x)):
            p *= self.a[z[i - 1]][z[i]] * self.e[z[i]][x[i]]
        return p

    def log_joint(self, x, z):
        #log2 P(x, z) as a sum of log2 factors
        self._check(x, z)
        s = self.log_pi[z[0]] + self.log_e[z[0]][x[0]]
        for i in range(1, len(x)):
            s += self.log_a[z[i - 1]][z[i]] + self.log_e[z[i]][x[i]]
        return s

    def brute_force(self, x):
        best_path, best_p, total = None, -1.0, 0.0
        for path in itertools.product(self.states, repeat=len(x)):
            z = "".join(path)
            p = self.joint_probability(x, z)
            total += p
            if p > best_p:  # keeps the first path in state order on ties
                best_path, best_p = z, p
        return best_path, best_p, total

def read_model(path):
    with open(path) as f:
        d = json.load(f)
    return HMM(d["states"], d["alphabet"], d["pi"], d["a"], d["e"])

#=======================================================================================
# =================    SELF CHECKS =====================================================
#=======================================================================================

def _self_check_task1(m, sequences):
    for x in sequences:
        for path in itertools.product(m.states, repeat=len(x)):
            z = "".join(path)
            p = m.joint_probability(x, z)
            assert math.isclose(m.log_joint(x, z), log2(p)), (x, z)
        best, best_p, total = m.brute_force(x)
        assert 0 < best_p <= total <= 1 + 1e-12

if __name__ == "__main__":
    import sys

    files = sys.argv[1:] or ["toy_model.json", "weather_model.json"]
    for fn in files:
        try:
            m = read_model(fn)
        except FileNotFoundError:
            print(f"{fn}: not found, skipped")
            continue
        seqs = ["".join(s) for n in (1, 2, 3, 5)
                for s in itertools.islice(itertools.product(m.alphabet, repeat=n), 6)]
        _self_check_task1(m, seqs)
        print(f"{fn}: Task 1 self-checks OK")
