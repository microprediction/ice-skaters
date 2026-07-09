"""Alignment, robustness, serialization."""

import copy
import math
import pickle
import random

import pytest
from river import linear_model, preprocessing

from ice_skaters import LaplaceFeatures, LaplaceTarget


def _pipeline():
    return LaplaceTarget(
        regressor=preprocessing.TargetStandardScaler(
            regressor=LaplaceFeatures()
            | preprocessing.StandardScaler()
            | linear_model.LinearRegression()))


def test_predict_and_learn_see_identical_features():
    """river's Pipeline updates transformers before transforming in
    learn_one; the cache must hand both paths the same features."""
    seen = []
    orig = LaplaceFeatures.transform_one

    def spy(self, x):
        out = orig(self, x)
        seen.append(dict(out))
        return out

    lf = LaplaceFeatures()
    pipe = lf | preprocessing.StandardScaler() | linear_model.LinearRegression()
    rng = random.Random(0)
    LaplaceFeatures.transform_one = spy
    try:
        for _ in range(30):
            x = {"a": rng.gauss(0, 1), "b": rng.gauss(5, 2)}
            pipe.predict_one(x)
            pipe.learn_one(x, rng.gauss(0, 1))
    finally:
        LaplaceFeatures.transform_one = orig
    predict_path, learn_path = seen[0::2], seen[1::2]
    assert predict_path == learn_path


def test_surprise_is_bounded_on_wild_spike():
    lf = LaplaceFeatures()
    for t in range(100):
        lf.learn_one({"a": math.sin(t)})
    out = lf.transform_one({"a": 1e9})
    assert abs(out["z_a"]) < 7.1
    assert math.isfinite(out["mu_a"])


def test_non_finite_values_are_forecast_imputed_and_ignored():
    lf = LaplaceFeatures()
    for t in range(50):
        lf.learn_one({"a": float(t % 5)})
    out = lf.transform_one({"a": float("nan")})
    assert math.isfinite(out["mu_a"]) and out["z_a"] == 0.0
    lf.learn_one({"a": float("inf")})       # must not corrupt the body
    out = lf.transform_one({"a": 1.0})
    assert all(math.isfinite(v) for v in out.values())


def test_non_numeric_passthrough():
    lf = LaplaceFeatures()
    lf.learn_one({"a": 1.0, "tag": "x", "flag": True})
    out = lf.transform_one({"a": 1.1, "tag": "x", "flag": True})
    assert out["tag"] == "x" and out["flag"] is True
    assert set(out) == {"tag", "flag", "mu_a", "z_a"}


def test_emit_modes_and_validation():
    assert set(LaplaceFeatures(emit="mu")._features({"a": 1.0})) == {"mu_a"}
    assert set(LaplaceFeatures(emit="z")._features({"a": 1.0})) == {"z_a"}
    with pytest.raises(ValueError):
        LaplaceFeatures(emit="nope")


def test_full_model_pickles_and_deepcopies():
    model = _pipeline()
    rng = random.Random(1)
    for _ in range(40):
        x = {"a": rng.gauss(0, 1)}
        model.predict_one(x)
        model.learn_one(x, rng.gauss(0, 1))
    probe = {"a": 0.3}
    expected = model.predict_one(probe)
    assert pickle.loads(pickle.dumps(model)).predict_one(probe) == expected
    assert copy.deepcopy(model).predict_one(probe) == expected


def test_keys_may_appear_and_vanish_mid_stream():
    lf = LaplaceFeatures()
    lf.learn_one({"a": 1.0})
    lf.learn_one({"a": 1.1, "c": 9.0})
    out = lf.transform_one({"c": 9.1})
    assert set(out) == {"mu_c", "z_c"}


def test_target_wrapper_learns_and_predicts_finite():
    model = _pipeline()
    rng = random.Random(2)
    preds = []
    for t in range(120):
        x = {"a": rng.gauss(0, 1)}
        p = model.predict_one(x)
        preds.append(p if p is not None else 0.0)
        model.learn_one(x, 0.5 * x["a"] + rng.gauss(0, 0.1))
    assert all(math.isfinite(p) for p in preds)


def test_mus_enrichment_exposes_previous_forecast():
    model = LaplaceTarget(
        regressor=preprocessing.TargetStandardScaler(
            regressor=LaplaceFeatures()
            | preprocessing.StandardScaler()
            | linear_model.LinearRegression()),
        mus=2)
    rng = random.Random(5)
    for t in range(60):
        x = {"a": rng.gauss(0, 1)}
        p = model.predict_one(x)
        assert p is None or math.isfinite(p)
        model.learn_one(x, 0.5 * x["a"] + rng.gauss(0, 0.1))
    aug = model._augment({"a": 0.1})
    assert "mu_y_prev1" in aug and math.isfinite(aug["mu_y_prev1"])
    with pytest.raises(ValueError):
        LaplaceTarget(regressor=None, mus=0)
