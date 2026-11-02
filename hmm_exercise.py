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

def log_sum(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    hi, lo = max(a, b), min(a, b)
    return hi + math.log2(1 + 2 ** (lo - hi))

def agreement(z1, z2):
    if len(z1) != len(z2):
        raise ValueError("paths must have the same length")
    same = 0
    for s1, s2 in zip(z1, z2):
        if s1 == s2:
            same += 1
    return same / len(z1)

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

# ==========Task 3: Forward, Backward, posterior========
    def forward(self, x):
        f = [{}]
        for k in self.states:
            f[0][k] = self.log_pi[k] + self.log_e[k][x[0]]
        for i in range(1, len(x)):
            f.append({})
            for k in self.states:
                s = -math.inf
                for l in self.states:
                    s = log_sum(s, f[i - 1][l] + self.log_a[l][k])
                f[i][k] = s + self.log_e[k][x[i]]
        log_px = -math.inf
        for k in self.states:
            log_px = log_sum(log_px, f[-1][k])
        return f, log_px

    def backward(self, x):
        n = len(x)
        b = [{} for _ in range(n)]
        for k in self.states:
            b[n - 1][k] = 0.0  # log2(1)
        for i in range(n - 2, -1, -1):
            for k in self.states:
                s = -math.inf
                for l in self.states:
                    s = log_sum(s, self.log_a[k][l] + self.log_e[l][x[i + 1]] + b[i + 1][l])
                b[i][k] = s
        log_px = -math.inf
        for k in self.states:
            log_px = log_sum(log_px, self.log_pi[k] + self.log_e[k][x[0]] + b[0][k])
        return b, log_px

    def posterior(self, x):
        f, log_px = self.forward(x)
        b, _ = self.backward(x)
        post = []
        for i in range(len(x)):
            post.append({k: 2 ** (f[i][k] + b[i][k] - log_px) for k in self.states})
        return post

    def posterior_path(self, x):
        path = []
        for col in self.posterior(x):
            best = None
            for k in self.states:
                if best is None or col[k] > col[best]:  # first state wins ties
                    best = k
            path.append(best)
        return "".join(path)


def read_model(path):
    with open(path) as f:
        d = json.load(f)
    return HMM(d["states"], d["alphabet"], d["pi"], d["a"], d["e"])



#=======================================================================================
# =================    SELF CHECKS =====================================================
#=======================================================================================

# ================= TASK 1 =============
def _self_check_task1(m, sequences):
    for x in sequences:
        for path in itertools.product(m.states, repeat=len(x)):
            z = "".join(path)
            p = m.joint_probability(x, z)
            assert math.isclose(m.log_joint(x, z), log2(p)), (x, z)
        best, best_p, total = m.brute_force(x)
        assert 0 < best_p <= total <= 1 + 1e-12

# ================= TASK 2 =============
def _self_check_task2(m, length=5):
    for n in range(1, length + 1):
        for s in itertools.product(m.alphabet, repeat=n):
            x = "".join(s)
            path, score = m.viterbi(x)
            best, best_p, total = m.brute_force(x)
            assert path == best, (x, path, best)
            assert math.isclose(2 ** score, best_p), (x, score, best_p)
            assert math.isclose(score, m.log_joint(x, path)), (x, path)

# ================= TASK 3 =============
def _self_check_task3(m, length=5):
    for n in range(1, length + 1):
        for s in itertools.product(m.alphabet, repeat=n):
            x = "".join(s)
            f, lf = m.forward(x)
            b, lb = m.backward(x)
            assert math.isclose(lf, lb), (x, lf, lb)                       # 1
            _, _, total = m.brute_force(x)
            assert math.isclose(2 ** lf, total), (x, lf, total)            # 2
            post = m.posterior(x)
            for i in range(n):
                assert math.isclose(sum(post[i].values()), 1.0), (x, i)    # 3
                for k in m.states:                                         # 4
                    pk = 0.0
                    for path in itertools.product(m.states, repeat=n):
                        if path[i] == k:
                            pk += m.joint_probability(x, "".join(path))
                    assert math.isclose(post[i][k], pk / total), (x, i, k)

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

    m = read_model("toy_model.json")
    x = "phchc"
    path, score = m.viterbi(x)
    print("Viterbi:", path, round(score, 2))
    for k in m.states:
        print(f"v({k}, i):", [round(m.last_v[i][k], 2) for i in range(len(x))])

    # self-check 5: the paper tables of Task 3
    f, log_px = m.forward(x)
    b, _ = m.backward(x)
    post = m.posterior(x)
    print("P(x) =", f"{2 ** log_px:.4g}")
    for k in m.states:
        print(f"f({k}, i):", [f"{2 ** f[i][k]:.4g}" for i in range(len(x))])
    for k in m.states:
        print(f"b({k}, i):", [f"{2 ** b[i][k]:.4g}" for i in range(len(x))])
    print("posterior at 2:", {k: round(post[2][k], 4) for k in m.states})

    # Think about it (a)
    pp = m.posterior_path(x)
    print("posterior path:", pp, f"{m.joint_probability(x, pp):.4g}",
          "| Viterbi path:", path, f"{m.joint_probability(x, path):.4g}")

    # Decode the simulated sequences
    from Bio import SeqIO

    seqs = {r.id: str(r.seq) for r in SeqIO.parse("sequences.fasta", "fasta")}
    true = {r.id: str(r.seq) for r in SeqIO.parse("true_paths.fasta", "fasta")}
    print(f"{'id':<10}{'len':>6}{'Viterbi':>10}{'posterior':>11}")
    for sid, x in seqs.items():
        z = true[sid]
        vpath, vscore = m.viterbi(x)
        ppath = m.posterior_path(x)
        # self-checks
        assert agreement(z, z) == 1.0
        assert math.isclose(vscore, m.log_joint(x, vpath))
        assert math.isclose(m.forward(x)[1], m.backward(x)[1])
        print(f"{sid:<10}{len(x):>6}{agreement(vpath, z):>10.3f}{agreement(ppath, z):>11.3f}")

