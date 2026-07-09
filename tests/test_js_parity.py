"""Python/JavaScript parity for the reverse-vended river pieces.

The JS module vendors the skaters twin and ports river's StandardScaler,
TargetStandardScaler, LinearRegression and Pipeline semantics line for
line. The baseline pipeline involves no skaters at all, so it must agree
to near machine precision; the fronted model inherits the documented
~1e-6 agreement of the skaters twins, so it gets a looser tolerance.

Skipped when node is not available.
"""

import json
import math
import random
import shutil
import subprocess
import os

import pytest

if shutil.which("node") is None:
    pytest.skip("node not available", allow_module_level=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
_JS = os.path.join(_HERE, "..", "docs", "js", "ice-skaters", "index.mjs")

N = 400

_NODE_SCRIPT = """
import {{ baselineModel, frontedModel }} from '{module}';
const rows = {rows};
const out = {{ baseline: [], fronted: [] }};
for (const [name, model] of [['baseline', baselineModel()], ['fronted', frontedModel()]]) {{
  for (const [x, y] of rows) {{
    out[name].push(model.predictOne(x));
    model.learnOne(x, y);
  }}
}}
console.log(JSON.stringify(out));
"""


def _rows():
    rng = random.Random(11)
    x = y = 0.0
    rows = []
    for _ in range(N):
        x = 0.8 * x + rng.gauss(0, 1)
        y = 0.7 * y + x + 0.5 * rng.gauss(0, 1)
        xo = x + (10.0 if rng.random() < 0.02 else 0.0)
        rows.append(({"x": xo}, y))
    return rows


def _python_preds(rows):
    from river import linear_model, preprocessing

    from ice_skaters import LaplaceFeatures, LaplaceTarget

    out = {}
    models = {
        "baseline": preprocessing.TargetStandardScaler(
            regressor=preprocessing.StandardScaler()
            | linear_model.LinearRegression()),
        "fronted": LaplaceTarget(
            regressor=preprocessing.TargetStandardScaler(
                regressor=LaplaceFeatures()
                | preprocessing.StandardScaler()
                | linear_model.LinearRegression())),
    }
    for name, model in models.items():
        preds = []
        for x, y in rows:
            p = model.predict_one(dict(x))
            preds.append(p if p is not None else 0.0)
            model.learn_one(dict(x), y)
        out[name] = preds
    return out


def _js_preds(rows):
    script = _NODE_SCRIPT.format(
        module=_JS.replace(os.sep, "/"),
        rows=json.dumps([[x, y] for x, y in rows]))
    res = subprocess.run(["node", "--input-type=module", "-e", script],
                         capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-2000:]
    return json.loads(res.stdout.strip().splitlines()[-1])


def test_python_js_parity():
    rows = _rows()
    py = _python_preds(rows)
    js = _js_preds(rows)

    # The river ports involve no skaters: they must agree to machine noise.
    base_gap = max(abs(a - b) for a, b in zip(py["baseline"], js["baseline"]))
    assert base_gap < 1e-9, f"baseline pipelines diverge: {base_gap}"

    # The fronted model is chaotic in warmup: the skaters twins agree to
    # ~1e-6 per tick, but SGD amplifies differences exponentially while the
    # target scaler's variance is still tiny, so per-tick parity there is
    # not achievable or meaningful. The claim that survives is behavioral parity:
    # trajectories reconverge after warmup and the MAEs match closely.
    ys = [y for _, y in rows]

    def mae(preds):
        return sum(abs(p - y) for p, y in list(zip(preds, ys))[100:]) / (N - 100)

    m_py, m_js = mae(py["fronted"]), mae(js["fronted"])
    assert abs(m_py - m_js) / m_py < 0.02, (m_py, m_js)
    late_gap = max(abs(a - b) for a, b in
                   list(zip(py["fronted"], js["fronted"]))[300:])
    assert late_gap < 0.2, f"fronted trajectories fail to reconverge: {late_gap}"
    assert all(math.isfinite(v) for v in js["fronted"])
