# ice-skaters

[skaters](https://github.com/microprediction/skaters) on a
[river](https://riverml.xyz): calibrated forecast features for streaming
ML pipelines.

Every numeric stream is replaced by two scalars from its own online
Laplace forecaster: the predictive mean (what the forecaster expected
this value to be) and the standardized surprise z (how unexpected the
actual value was, bounded near |z| = 7 by construction). The mean
carries the level, the z carries the news. A wild observation can move
the pair only so far, and that bounded influence is where the robustness
comes from.

```
pip install ice-skaters
```

```python
from river import datasets, linear_model, metrics, preprocessing
from ice_skaters import LaplaceFeatures, LaplaceTarget

model = LaplaceTarget(
    regressor=preprocessing.TargetStandardScaler(
        regressor=LaplaceFeatures()
        | preprocessing.StandardScaler()
        | linear_model.LinearRegression()))

mae = metrics.MAE()
for x, y in datasets.TrumpApproval():
    pred = model.predict_one(x)
    mae.update(y, pred if pred is not None else 0.0)
    model.learn_one(x, y)
```

`LaplaceFeatures` is a river Transformer for the input streams.
`LaplaceTarget` wraps any regressor, in the style of
`TargetStandardScaler`, to add the target's own (mean, surprise) pair,
which a transformer cannot do since it never sees y. The target itself
stays raw. Both estimators pipe, pickle and deep-copy like any river
estimator. Non-numeric values pass through untouched, and NaN is imputed
by the forecast itself with z = 0: the model receives "expected value,
no news" instead of a poisoned pipeline.

## Why

On TrumpApproval with river's recommended pipeline (progressive
validation MAE, burn-in 100, `examples/trump_approval.py`):

| | clean | 2% corrupted readings |
|---|---|---|
| `StandardScaler` pipeline | 0.328 | 0.597 |
| + Laplace front-end | 0.382 | 0.407 |

The front-end pays a small toll on clean data and holds its footing when
the inputs misbehave. In controlled simulation the same substitution
beats raw features, a running z-score, a median/MAD winsorizer and a
Huberised loss 30/30 seeds under every contamination type tested, at a
small clean-data toll. Full protocols, numbers and the study design live
in the [timemachines](https://github.com/microprediction/timemachines)
repo, `benchmarks/RESULTS.md` section 6.

## Boundaries, stated plainly

- Distance-based learners (KNN) do not benefit: neighbour averaging is
  already spike-robust and the extra dimensions degrade the metric.
- Entity-interleaved streams (many units multiplexed into one key) want
  per-entity bodies; a single body per key is handicapped there.
- If your heavy tails are signal rather than noise, taming them costs
  accuracy. Whether the extremes are informative decides the coordinates.

## Relation to the stack

`skaters` does one thing: fast univariate distributional forecasting,
stdlib-only, and this package is deliberately a thin adapter over it.
`timemachines` builds anomaly detection on the same calibrated surprise
streams. ice-skaters is the bridge from those streams to river's
estimator protocol, and nothing more.
