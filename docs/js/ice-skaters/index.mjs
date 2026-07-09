// ice-skaters in JavaScript: skaters on a river, in the browser.
//
// Two halves. The first is a faithful reverse-vend of the river pieces the
// benchmark study used — StandardScaler, TargetStandardScaler,
// LinearRegression (SGD, lr 0.01, squared loss, intercept_lr 0.01) and the
// Pipeline learn-then-transform semantics — ported line for line from river
// 0.25 so that Python and JavaScript agree numerically. The second is the
// ice-skaters estimators themselves, LaplaceFeatures and LaplaceTarget,
// ported from the Python package on top of the vendored skaters twin.
//
// Everything is dependency-free ES modules; the same file runs in node and
// in the browser.

import { laplace, Dist } from "../vendor/skaters/index.mjs";

const STD_NORMAL = Dist.gaussian(0.0, 1.0);
const EPS = 1e-12; // parade's PIT clamp: |z| <= ~7.03

let _SKATER = null; // shared, stateless given per-stream state
function skater() {
  if (_SKATER === null) _SKATER = laplace(1);
  return _SKATER;
}

function isFiniteNumber(v) {
  return typeof v === "number" && Number.isFinite(v);
}

export function winsorize(v, pending) {
  // Magnitude-relative, NOT sigma-relative; see the Python twin's docstring.
  v = Math.min(Math.max(v, -1e60), 1e60);
  if (pending !== null && pending !== undefined) {
    const m = pending.mean;
    const s = pending.std;
    if (Number.isFinite(m) && Number.isFinite(s)) {
      const w = 1e12 * (1.0 + Math.abs(m) + s);
      v = Math.min(Math.max(v, m - w), m + w);
    }
  }
  return v;
}

// ---------------------------------------------------------------- mini-river

export class StandardScaler {
  // river.preprocessing.StandardScaler, single-instance path, exact port.
  constructor() {
    this.counts = new Map();
    this.means = new Map();
    this.vars = new Map();
  }
  learnOne(x) {
    for (const [k, xi] of Object.entries(x)) {
      if (typeof xi !== "number") continue;
      const c = (this.counts.get(k) || 0) + 1;
      this.counts.set(k, c);
      const oldMean = this.means.get(k) || 0.0;
      const mean = oldMean + (xi - oldMean) / c;
      this.means.set(k, mean);
      const v = this.vars.get(k) || 0.0;
      this.vars.set(k, v + ((xi - oldMean) * (xi - mean) - v) / c);
    }
  }
  transformOne(x) {
    const out = {};
    for (const [k, xi] of Object.entries(x)) {
      if (typeof xi !== "number") {
        out[k] = xi;
        continue;
      }
      const v = this.vars.get(k) || 0.0;
      out[k] = v ? (xi - (this.means.get(k) || 0.0)) / Math.sqrt(v) : 0.0;
    }
    return out;
  }
}

export class LinearRegression {
  // river.linear_model.LinearRegression at defaults: SGD(0.01), Squared loss
  // (gradient 2*(pred - y)), intercept_lr 0.01, no regularization.
  constructor(lr = 0.01, interceptLr = 0.01) {
    this.lr = lr;
    this.interceptLr = interceptLr;
    this.weights = new Map();
    this.intercept = 0.0;
  }
  _rawDot(x) {
    let s = this.intercept;
    for (const [k, xi] of Object.entries(x)) {
      if (typeof xi !== "number") continue;
      s += (this.weights.get(k) || 0.0) * xi;
    }
    return s;
  }
  predictOne(x) {
    return this._rawDot(x);
  }
  learnOne(x, y) {
    let lg = 2.0 * (this._rawDot(x) - y);
    lg = Math.min(Math.max(lg, -1e12), 1e12);
    this.intercept -= this.interceptLr * lg;
    for (const [k, xi] of Object.entries(x)) {
      if (typeof xi !== "number") continue;
      this.weights.set(k, (this.weights.get(k) || 0.0) - this.lr * xi * lg);
    }
  }
}

class Var {
  // river.stats.Var with ddof=1 (the TargetStandardScaler default).
  constructor() {
    this.n = 0.0;
    this.mean = 0.0;
    this.S = 0.0;
  }
  update(x) {
    const meanOld = this.mean;
    this.n += 1.0;
    this.mean += (x - meanOld) / this.n;
    this.S += (x - meanOld) * (x - this.mean);
  }
  get() {
    return this.n > 1 ? this.S / (this.n - 1) : 0.0;
  }
}

export class TargetStandardScaler {
  // river.preprocessing.TargetStandardScaler: update var with raw y first,
  // then fit the inner regressor on the scaled target.
  constructor(regressor) {
    this.regressor = regressor;
    this.var = new Var();
  }
  _scale(y) {
    const sd = Math.sqrt(this.var.get());
    return sd ? (y - this.var.mean) / sd : 0.0;
  }
  _unscale(y) {
    return y * Math.sqrt(this.var.get()) + this.var.mean;
  }
  predictOne(x) {
    return this._unscale(this.regressor.predictOne(x));
  }
  learnOne(x, y) {
    this.var.update(y);
    this.regressor.learnOne(x, this._scale(y));
  }
}

export class Pipeline {
  // river.compose.Pipeline semantics for one transformer + one regressor:
  // predict transforms without learning; learn updates the transformer
  // FIRST, then transforms for the downstream steps.
  constructor(...steps) {
    this.steps = steps; // transformers..., regressor last
  }
  predictOne(x) {
    for (let i = 0; i < this.steps.length - 1; i++) x = this.steps[i].transformOne(x);
    return this.steps[this.steps.length - 1].predictOne(x);
  }
  learnOne(x, y) {
    for (let i = 0; i < this.steps.length - 1; i++) {
      this.steps[i].learnOne(x);
      x = this.steps[i].transformOne(x);
    }
    this.steps[this.steps.length - 1].learnOne(x, y);
  }
}

// ------------------------------------------------------------- ice-skaters

export class LaplaceFeatures {
  // Port of ice_skaters.LaplaceFeatures: each numeric key becomes
  // (mu_k, z_k); non-numerics pass through; non-finite numerics are
  // forecast-imputed; predict and learn paths see identical features.
  constructor(emit = "both", prefix = ["mu_", "z_"]) {
    if (!["both", "mu", "z"].includes(emit)) throw new Error(`bad emit ${emit}`);
    this.emit = emit;
    this.prefix = prefix;
    this.bodies = new Map(); // key -> {state, pending}
    this.pendingOut = null;
    this.lastX = null;
  }
  _features(x) {
    const out = {};
    const [muP, zP] = this.prefix;
    for (const [k, v] of Object.entries(x)) {
      if (typeof v !== "number" || k.startsWith(muP) || k.startsWith(zP)) {
        // keys already carrying this transformer's own prefixes pass
        // through: forecasting a forecast is never the intent
        out[k] = v;
        continue;
      }
      const entry = this.bodies.get(k);
      const pending = entry ? entry.pending : null;
      let mu, z;
      if (!pending) {
        mu = Number.isFinite(v) ? v : 0.0;
        z = 0.0;
      } else if (!Number.isFinite(v)) {
        mu = pending.mean;
        z = 0.0; // forecast-imputed, no news
      } else {
        mu = pending.mean;
        let u = pending.cdf(v);
        u = Math.min(Math.max(u, EPS), 1.0 - EPS);
        z = STD_NORMAL.quantile(u);
      }
      if (this.emit !== "z") out[muP + k] = mu;
      if (this.emit !== "mu") out[zP + k] = z;
    }
    return out;
  }
  learnOne(x) {
    this.pendingOut = this._features(x);
    this.lastX = JSON.stringify(x);
    const f = skater();
    const [muP2, zP2] = this.prefix;
    for (const [k, v] of Object.entries(x)) {
      if (!isFiniteNumber(v) || k.startsWith(muP2) || k.startsWith(zP2)) continue;
      const entry = this.bodies.get(k) || { state: null, pending: null };
      const [dists, st] = f(winsorize(v, entry.pending), entry.state);
      this.bodies.set(k, { state: st, pending: dists[0] });
    }
  }
  transformOne(x) {
    if (this.pendingOut !== null && JSON.stringify(x) === this.lastX) {
      const out = this.pendingOut;
      this.pendingOut = null;
      return out;
    }
    return this._features(x);
  }
}

export class LaplaceTarget {
  // Port of ice_skaters.LaplaceTarget: augment features with the target's
  // own (mu_y, zy) pair; the target itself stays raw.
  constructor(regressor, keys = ["mu_y", "z_y"], mus = 1) {
    if (mus < 1) throw new Error("mus must be >= 1");
    this.regressor = regressor;
    this.keys = keys;
    this.mus = mus;
    this.state = null;
    this.pending = null;
    this.muHist = [];
    this.zy = 0.0;
    this.prevY = null;
    this.laggedCounts = new Map();
    this.warned = false;
  }
  _checkLaggedTarget(x) {
    // a feature equal to the lagged target duplicates this wrapper's
    // surprise column exactly; warn once, recommend mus=2
    if (this.warned || this.prevY === null) return;
    for (const [k, v] of Object.entries(x)) {
      if (typeof v === "number" && v === this.prevY) {
        const c = (this.laggedCounts.get(k) || 0) + 1;
        this.laggedCounts.set(k, c);
        if (c >= 20) {
          this.warned = true;
          console.warn(`ice-skaters: feature '${k}' looks like a lagged copy ` +
            "of the target; it will duplicate the wrapper's surprise column " +
            "exactly. Use new LaplaceTarget(regressor, keys, 2) instead.");
        }
      } else {
        this.laggedCounts.delete(k);
      }
    }
  }
  _augment(x) {
    const [muKey, zKey] = this.keys;
    const out = {
      ...x,
      [muKey]: this.pending ? this.pending.mean : 0.0,
      [zKey]: this.zy,
    };
    for (let i = 1; i < this.mus; i++)
      out[muKey + "_prev" + i] = this.muHist.length >= i ? this.muHist[i - 1] : 0.0;
    return out;
  }
  predictOne(x) {
    return this.regressor.predictOne(this._augment(x));
  }
  learnOne(x, y) {
    this._checkLaggedTarget(x);
    this.regressor.learnOne(this._augment(x), y);
    if (isFiniteNumber(y)) {
      if (this.pending && this.mus > 1) {
        this.muHist.unshift(this.pending.mean);
        this.muHist.length = Math.min(this.muHist.length, this.mus - 1);
      }
      const [dists, st] = skater()(winsorize(y, this.pending), this.state);
      this.state = st;
      this.pending = dists[0];
      const z = st.z[0];
      this.zy = z === null || z === undefined ? 0.0 : z;
      this.prevY = y;
    }
  }
}

// The two models of the benchmark study, ready-made.
export function baselineModel() {
  return new TargetStandardScaler(
    new Pipeline(new StandardScaler(), new LinearRegression()),
  );
}

export function frontedModel() {
  return new LaplaceTarget(
    new TargetStandardScaler(
      new Pipeline(new LaplaceFeatures(), new StandardScaler(), new LinearRegression()),
    ),
  );
}
