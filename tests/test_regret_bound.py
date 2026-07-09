"""Numerical check of the regret-transfer theorem (papers/regret-transfer.md).

Runs the exact construction of the theorem: Laplace bodies over a
contaminated bivariate stream, clamped surprises, the forward
Vovk-Azoury-Warmuth learner on [1, z_x, z_y-lag], and asserts the
pathwise inequality for a grid of comparators. Also verifies Lemma 1
numerically: the conjugated density integrates to one.
"""

import math
import random

import pytest

from skaters import laplace
from skaters.dist import Dist

Z_MAX = Dist.gaussian(0.0, 1.0).quantile(1.0 - 1e-12)   # ~7.03
LAM = 1.0
N = 2500


def _series(scenario, seed):
    rng = random.Random(seed)
    x = [0.0] * N
    y = [0.0] * N
    for t in range(1, N):
        e, n = rng.gauss(0, 1), rng.gauss(0, 1)
        x[t] = 0.8 * x[t - 1] + e
        y[t] = 0.7 * y[t - 1] + x[t - 1] + 0.5 * n
    if scenario == "spikes":
        for t in range(1, N):
            if rng.random() < 0.03:
                y[t] += 10.0 * rng.choice((-1, 1))
            if rng.random() < 0.03:
                x[t] += 10.0 * rng.choice((-1, 1))
    return x, y


def _surprises(series):
    """Clamped z per tick, plus the unclamped z and the predictive dist."""
    f, st = laplace(1), None
    zbar, zraw, pend = [], [], []
    prev = None
    for v in series:
        if prev is None:
            zb = zr = 0.0
        else:
            u = prev.cdf(v)
            uc = min(max(u, 1e-12), 1.0 - 1e-12)
            zb = Dist.gaussian(0, 1).quantile(uc)
            zr = zb if u == uc else math.copysign(Z_MAX + 1.0, u - 0.5)
        zbar.append(zb)
        zraw.append(zr)
        pend.append(prev)
        dists, st = f(v, st)
        prev = dists[0]
    return zbar, zraw, pend


class VAW:
    """Forward (Azoury-Warmuth) ridge: A_t includes the current phi."""

    def __init__(self, d, lam=LAM):
        self.d, self.lam = d, lam
        self.A = [[lam if i == j else 0.0 for j in range(d)] for i in range(d)]
        self.b = [0.0] * d

    def _solve(self, M, v):
        import copy
        M = copy.deepcopy(M)
        v = list(v)
        n = self.d
        for i in range(n):
            p = max(range(i, n), key=lambda r: abs(M[r][i]))
            M[i], M[p] = M[p], M[i]
            v[i], v[p] = v[p], v[i]
            for r in range(i + 1, n):
                f = M[r][i] / M[i][i]
                for c in range(i, n):
                    M[r][c] -= f * M[i][c]
                v[r] -= f * v[i]
        out = [0.0] * n
        for i in reversed(range(n)):
            out[i] = (v[i] - sum(M[i][c] * out[c]
                                 for c in range(i + 1, n))) / M[i][i]
        return out

    def predict(self, phi):
        for i in range(self.d):
            for j in range(self.d):
                self.A[i][j] += phi[i] * phi[j]
        w = self._solve(self.A, self.b)
        return sum(wi * p for wi, p in zip(w, phi))

    def update(self, phi, target):
        for i in range(self.d):
            self.b[i] += target * phi[i]

    def logdet_ratio(self):
        import copy
        M = copy.deepcopy(self.A)
        n = self.d
        ld = 0.0
        for i in range(n):
            p = max(range(i, n), key=lambda r: abs(M[r][i]))
            if p != i:
                M[i], M[p] = M[p], M[i]
            ld += math.log(abs(M[i][i]))
            for r in range(i + 1, n):
                f = M[r][i] / M[i][i]
                for c in range(i, n):
                    M[r][c] -= f * M[i][c]
        return ld - n * math.log(self.lam)


@pytest.mark.parametrize("scenario", ["clean", "spikes"])
def test_pathwise_regret_bound(scenario):
    x, y = _series(scenario, seed=7)
    zbx, _, _ = _surprises(x)
    zby, zry, _ = _surprises(y)

    d = 3
    vaw = VAW(d)
    preds, feats = [], []
    for t in range(1, N):
        phi = [1.0, zbx[t], zby[t - 1]]
        zhat = vaw.predict(phi)
        preds.append(zhat)
        feats.append(phi)
        vaw.update(phi, zby[t])

    targets = zby[1:]
    sq_hat = sum((p - z) ** 2 for p, z in zip(preds, targets))

    rng = random.Random(0)
    comparators = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                   [0.1, 0.5, -0.5]]
    comparators += [[rng.uniform(-1, 1) for _ in range(d)] for _ in range(4)]

    bound_core = Z_MAX ** 2 * vaw.logdet_ratio()
    for u in comparators:
        sq_u = sum((sum(ui * p for ui, p in zip(u, phi)) - z) ** 2
                   for phi, z in zip(feats, targets))
        regret = sq_hat - sq_u
        bound = LAM * sum(ui ** 2 for ui in u) + bound_core
        assert regret <= bound + 1e-6, (scenario, u, regret, bound)

    # the bound should not be vacuous at this T: O(d log T) must sit below
    # the trivial total loss O(T). At T=500 the z_max^2 constant still
    # dominates; by T=2500 it does not.
    assert bound_core < sum(z ** 2 for z in targets)


def test_lemma1_conjugated_density_integrates_to_one():
    x, y = _series("clean", seed=3)
    _, _, pend = _surprises(y)
    dbody = pend[300]
    assert dbody is not None
    m, s = 0.7, 1.3                       # an arbitrary refinement
    std_normal = Dist.gaussian(0.0, 1.0)

    lo = dbody.mean - 12 * dbody.std
    hi = dbody.mean + 12 * dbody.std
    n = 4000
    total = 0.0
    for i in range(n):
        yy = lo + (hi - lo) * (i + 0.5) / n
        u = min(max(dbody.cdf(yy), 1e-15), 1 - 1e-15)
        z = std_normal.quantile(u)
        q = math.exp(-0.5 * ((z - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))
        total += dbody.pdf(yy) * q / std_normal.pdf(z) * (hi - lo) / n
    assert abs(total - 1.0) < 5e-3
