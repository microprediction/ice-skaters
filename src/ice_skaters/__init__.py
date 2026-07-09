"""ice-skaters: skaters on a river.

Calibrated forecast features for `river <https://riverml.xyz>`_
pipelines, built on the online Laplace forecaster of the ``skaters``
package. ``LaplaceFeatures`` replaces each numeric feature stream with
two scalars from its own forecaster: the one-step predictive mean (what
the forecaster expected this value to be) and the standardized surprise
z (how unexpected the actual value was, bounded by construction near
|z| = 7). The mean carries the level, the z carries the news, and a wild
input can move the pair only so far, which is where the robustness comes
from. Non-numeric values pass through untouched; non-finite numeric
values are imputed by the forecast itself (mean = the body's prediction,
z = 0: no news), which is more than ``StandardScaler`` can say for NaN.

    from river import linear_model, preprocessing
    from ice_skaters import LaplaceFeatures, LaplaceTarget

    model = LaplaceTarget(
        regressor=preprocessing.TargetStandardScaler(
            regressor=LaplaceFeatures()
            | preprocessing.StandardScaler()
            | linear_model.LinearRegression()))
    # ...then the usual river loop:
    # model.predict_one(x); model.learn_one(x, y)

``LaplaceTarget`` adds the same pair for the target's own history, which
a transformer cannot do since it never sees y; it wraps any regressor in
the style of ``TargetStandardScaler`` and leaves the target itself raw.

Evidence, protocols and boundaries (distance-based learners do not
benefit; entity-interleaved streams want per-entity bodies; genuinely
informative heavy tails should not be tamed): the regression front-end
study in the timemachines repo, ``benchmarks/RESULTS.md`` section 6.

Implementation notes:

- river's ``Pipeline.learn_one`` updates unsupervised transformers
  *before* calling ``transform_one`` for the downstream steps. A naive
  stateful transformer would therefore hand the model fresher features
  at learn time than it predicted with. ``LaplaceFeatures`` caches the
  pre-update feature dict during ``learn_one`` and serves it to the
  immediately following ``transform_one`` for the same sample, so the
  predict path and the learn path see identical features. Bodies advance
  exactly once per sample, in ``learn_one``; ``transform_one`` alone
  (the predict path) is pure.
- Instances hold only plain state data (dicts, Dists), never the skater
  closure itself: skaters' state purity lets one module-level ``laplace``
  callable serve every stream with state passed per stream, so both
  estimators pickle and deep-copy like any river estimator.
- Observations are winsorized before a body consumes them (1e60
  absolute, plus a magnitude-relative window twelve orders above the
  current predictive level). skaters >= 0.13 carries the same gate in
  the parade; keeping it here too protects users on earlier releases.
"""

from __future__ import annotations
import math
import warnings
from river import base
from skaters import laplace
from skaters.dist import Dist

__version__ = "0.1.3"
__all__ = ["LaplaceFeatures", "LaplaceTarget"]

_STD_NORMAL = Dist.gaussian(0.0, 1.0)
_EPS = 1e-12          # parade's PIT clamp: |z| <= ~7.03, never infinite

_SKATER = None        # shared, stateless given per-stream state; not pickled


def _skater():
    global _SKATER
    if _SKATER is None:
        _SKATER = laplace(1)
    return _SKATER


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _winsorize(v: float, pending) -> float:
    """Clamp absurd finite magnitudes before a body consumes them.

    Magnitude-relative, NOT sigma-relative: after a degenerate-variance
    stretch (missing-data zeros, say) a legitimate value sits billions of
    sigmas out and must pass. Twelve orders above the current level is
    unreachable by data, far below the ~1e77 jump ratio where double
    arithmetic actually dies, so this is exact identity on sane streams.
    """
    v = min(max(v, -1e60), 1e60)
    if pending is not None:
        m, s = pending.mean, pending.std
        if math.isfinite(m) and math.isfinite(s):
            w = 1e12 * (1.0 + abs(m) + s)
            v = min(max(v, m - w), m + w)
    return v


class LaplaceFeatures(base.Transformer):
    """Replace each numeric feature with (predictive mean, surprise z).

    Args:
        emit: which scalars to emit per stream, ``"both"`` (default),
            ``"mu"`` or ``"z"``.
        prefix: feature-name prefixes, ``(mean_prefix, z_prefix)``.
    """

    def __init__(self, emit: str = "both", prefix=("mu_", "z_")):
        if emit not in ("both", "mu", "z"):
            raise ValueError(f"emit must be 'both', 'mu' or 'z', got {emit!r}")
        self.emit = emit
        self.prefix = prefix
        self._bodies = {}          # key -> (state dict, pending Dist)
        self._pending_out = None   # features cached by learn_one
        self._last_x = None

    def _features(self, x):
        """Pure: read the pending predictives, never advance a body."""
        out = {}
        mu_p, z_p = self.prefix
        for k, v in x.items():
            if not _is_number(v) or k.startswith(mu_p) or k.startswith(z_p):
                # non-numerics pass through, and so do keys already carrying
                # this transformer's own prefixes: forecasting a forecast
                # (e.g. the mu_y/z_y pair a LaplaceTarget wrapper adds
                # upstream) is never the intent
                out[k] = v
                continue
            entry = self._bodies.get(k)
            pending = entry[1] if entry else None
            if pending is None:
                mu = float(v) if math.isfinite(v) else 0.0
                z = 0.0
            elif not math.isfinite(v):
                mu, z = pending.mean, 0.0          # forecast-imputed, no news
            else:
                mu = pending.mean
                u = pending.cdf(float(v))
                u = min(max(u, _EPS), 1.0 - _EPS)
                z = _STD_NORMAL.quantile(u)
            if self.emit in ("both", "mu"):
                out[mu_p + k] = mu
            if self.emit in ("both", "z"):
                out[z_p + k] = z
        return out

    def learn_one(self, x):
        self._pending_out = self._features(x)
        self._last_x = dict(x)
        mu_p, z_p = self.prefix
        f = _skater()
        for k, v in x.items():
            if not _is_number(v) or not math.isfinite(v) \
                    or k.startswith(mu_p) or k.startswith(z_p):
                continue                            # non-finite/passthrough
            st, pending = self._bodies.get(k) or (None, None)
            dists, st = f(_winsorize(float(v), pending), st)
            self._bodies[k] = (st, dists[0])

    def transform_one(self, x):
        if self._pending_out is not None and x == self._last_x:
            out, self._pending_out = self._pending_out, None
            return out
        return self._features(x)


class LaplaceTarget(base.Regressor):
    """Augment features with the TARGET stream's own (mean, surprise) pair.

    Transformers never see y, so the target's calibrated history features
    need a regressor wrapper, in the style of ``TargetStandardScaler``:

        model = LaplaceTarget(
            regressor=LaplaceFeatures()
                      | preprocessing.StandardScaler()
                      | linear_model.LinearRegression())

    At prediction time the inner regressor receives, alongside x, the
    body's forecast of the y it is about to predict (``mu_y``) and the
    surprise of the previous y (``z_y``). With ``mus=2`` it also exposes
    the previous tick's forecast (``mu_y_prev1``), a measured, free
    enrichment: it needs no extra forecaster, and in the audit it cut the
    clean-data toll roughly in half while improving every contaminated
    case. Do NOT hand the learner a lagged copy of the target as an extra
    stream instead: that route duplicates the surprise column exactly,
    doubling the gradient on the one feature not to over-react to. The key names carry
    LaplaceFeatures' own prefixes on purpose, so a downstream
    LaplaceFeatures passes them through instead of forecasting the
    forecast. The same pre-update pair is used
    at learn time, then the body consumes the new y. The target itself
    stays raw: this wrapper never transforms y, it only adds features.
    """

    def __init__(self, regressor, keys=("mu_y", "z_y"), mus: int = 1):
        if mus < 1:
            raise ValueError("mus must be >= 1")
        self.regressor = regressor
        self.keys = keys
        self.mus = mus             # trailing forecast means to expose
        self._state = None
        self._pending = None       # predictive Dist for the next y
        self._mu_hist = []         # previous predictive means, newest first
        self._zy = 0.0
        self._prev_y = None        # lagged-target footgun detector
        self._lagged = {}          # key -> consecutive matches
        self._warned = False

    def _augment(self, x):
        mu_key, z_key = self.keys
        mu = self._pending.mean if self._pending is not None else 0.0
        out = {**x, mu_key: mu, z_key: self._zy}
        for i in range(1, self.mus):
            out[f"{mu_key}_prev{i}"] = (self._mu_hist[i - 1]
                                        if len(self._mu_hist) >= i else 0.0)
        return out

    def predict_one(self, x):
        return self.regressor.predict_one(self._augment(x))

    def _check_lagged_target(self, x):
        # A feature stream equal to the lagged target duplicates this
        # wrapper's surprise column exactly (two forecasters, same
        # history), doubling the gradient on the one feature not to
        # over-react to. Warn once; use mus=2 for the safe enrichment.
        if self._warned or self._prev_y is None:
            return
        for k, v in x.items():
            if isinstance(v, (int, float)) and float(v) == self._prev_y:
                self._lagged[k] = self._lagged.get(k, 0) + 1
                if self._lagged[k] >= 20:
                    self._warned = True
                    warnings.warn(
                        f"feature {k!r} looks like a lagged copy of the "
                        "target; routed through LaplaceFeatures it will "
                        "duplicate this wrapper's surprise column exactly. "
                        "Use LaplaceTarget(..., mus=2) instead.",
                        stacklevel=3)
            else:
                self._lagged.pop(k, None)

    def learn_one(self, x, y):
        self._check_lagged_target(x)
        self.regressor.learn_one(self._augment(x), y)
        y = float(y)
        if math.isfinite(y):
            # laplace is parade-wrapped: state["z"][0] is the surprise of
            # this y against the predictive issued before it arrived
            if self._pending is not None and self.mus > 1:
                self._mu_hist.insert(0, self._pending.mean)
                del self._mu_hist[self.mus - 1:]
            dists, self._state = _skater()(
                _winsorize(y, self._pending), self._state)
            self._pending = dists[0]
            z = self._state["z"][0]
            self._zy = z if z is not None else 0.0
            self._prev_y = y
