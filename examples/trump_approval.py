"""river + skaters: calibrated forecast features for a streaming regression.

Every numeric stream is replaced by two scalars from its own online
Laplace forecaster: the predictive mean (the expected value) and the
standardized surprise z (the news, bounded near |z| = 7 by construction).
``LaplaceFeatures`` does this for the input streams; ``LaplaceTarget``
adds the same pair for the target's own history. Both are ordinary river
estimators: they pipe, pickle and deep-copy.

The demo is river's TrumpApproval dataset (ships with river, no
download), scored by progressive validation exactly as in the river docs,
then repeated with 2% of pollster readings corrupted by fat-finger noise.
The point is not the clean run, where the plain pipeline is fine and the
front-end costs a little, but what happens the day the data misbehaves:

    MAE, LinearRegression        clean    2% corrupted
    river StandardScaler         0.328    0.597
    + Laplace front-end          0.369    0.382

Run:
    pip install ice-skaters
    python examples/trump_approval.py
"""

import math
import random

from river import datasets, linear_model, metrics, preprocessing

from ice_skaters import LaplaceFeatures, LaplaceTarget


def baseline():
    return preprocessing.TargetStandardScaler(
        regressor=preprocessing.StandardScaler()
        | linear_model.LinearRegression())


def fronted():
    return LaplaceTarget(
        regressor=preprocessing.TargetStandardScaler(
            regressor=LaplaceFeatures()
            | preprocessing.StandardScaler()
            | linear_model.LinearRegression()))


def progressive_mae(model, stream, burn=100):
    mae = metrics.MAE()
    for t, (x, y) in enumerate(stream):
        pred = model.predict_one(x)
        if t >= burn:
            mae.update(y, pred if pred is not None else 0.0)
        model.learn_one(x, y)
    return mae.get()


def corrupted(rows, rate=0.02, seed=1):
    """Fat-finger 2% of the pollster readings, 6-10 sigmas off."""
    rng = random.Random(seed)
    keys = list(rows[0][0].keys())
    scale = {}
    for k in keys:
        col = [float(x[k]) for x, _ in rows]
        mu = sum(col) / len(col)
        scale[k] = math.sqrt(sum((v - mu) ** 2 for v in col) / len(col)) or 1.0
    out = []
    for x, y in rows:
        x = dict(x)
        for k in keys:
            if rng.random() < rate:
                x[k] = float(x[k]) + (6 + 4 * rng.random()) * scale[k] \
                    * rng.choice((-1, 1))
        out.append((x, y))
    return out


if __name__ == "__main__":
    rows = list(datasets.TrumpApproval())

    print("TrumpApproval, progressive validation MAE (burn-in 100):\n")
    print(f"{'':28s} {'clean':>8s} {'2% corrupted':>13s}")
    for name, factory in (("river StandardScaler", baseline),
                          ("+ Laplace front-end", fronted)):
        clean = progressive_mae(factory(), rows)
        dirty = progressive_mae(factory(), corrupted(rows))
        print(f"{name:28s} {clean:8.3f} {dirty:13.3f}")

    print("\nThe front-end pays a small toll on clean data and holds its"
          "\nfooting when the inputs misbehave; the raw pipeline degrades."
          "\nEach input stream became two features: what the forecaster"
          "\nexpected, and how surprising the actual value was.")
