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

# ===================Task 2: Viterbi ==================
    def viterbi(self, x):
        # log2 Viterbi: v[i][k] = best log2 score of a path ending in k at i
        v = [{}]
        back = [{}]
        for k in self.states:
            v[0][k] = self.log_pi[k] + self.log_e[k][x[0]]
            back[0][k] = None

        for i in range(1, len(x)):
            v.append({})
            back.append({})
            for k in self.states:
                best_l, best_score = None, -math.inf
                for l in self.states:
                    score = v[i - 1][l] + self.log_a[l][k]
                    if best_l is None or score > best_score:  # first state wins ties
                        best_l, best_score = l, score
                v[i][k] = best_score + self.log_e[k][x[i]]
                back[i][k] = best_l

        # best state in the last column
        last = None
        for k in self.states:
            if last is None or v[-1][k] > v[-1][last]:
                last = k

        # traceback
        path = [last]
        for i in range(len(x) - 1, 0, -1):
            path.append(back[i][path[-1]])
        path.reverse()
        self.last_v = v
        return "".join(path), v[-1][last]

    def viterbi_plain(self, x):
        # same algorithm with plain probabilities
        v = [{k: self.pi[k] * self.e[k][x[0]] for k in self.states}]
        back = [{}]
        for i in range(1, len(x)):
            v.append({})
            back.append({})
            for k in self.states:
                best_l, best_p = None, -1.0
                for l in self.states:
                    p = v[i - 1][l] * self.a[l][k]
                    if p > best_p:
                        best_l, best_p = l, p
                v[i][k] = best_p * self.e[k][x[i]]
                back[i][k] = best_l
        last = None
        for k in self.states:
            if last is None or v[-1][k] > v[-1][last]:
                last = k
        path = [last]
        for i in range(len(x) - 1, 0, -1):
            path.append(back[i][path[-1]])
        path.reverse()
        return "".join(path), v[-1][last]


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

def _self_check_task2(m, length=5):
    for n in range(1, length + 1):
        for s in itertools.product(m.alphabet, repeat=n):
            x = "".join(s)
            path, score = m.viterbi(x)
            best, best_p, total = m.brute_force(x)
            assert path == best, (x, path, best)
            assert math.isclose(2 ** score, best_p), (x, score, best_p)
            assert math.isclose(score, m.log_joint(x, path)), (x, path)

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
        _self_check_task2(m)
        print(f"{fn}: Task 2 self-checks OK")

    # self-check 3: the paper table
    m = read_model("toy_model.json")
    x = "phchc"
    path, score = m.viterbi(x)
    print("Viterbi:", path, round(score, 2))
    for k in m.states:
        print(f"v({k}, i):", [round(m.last_v[i][k], 2) for i in range(len(x))])

