# The regression front-end: calibrated forecast features for online regression

*Preliminary results, July 2026. The study lives in the
[timemachines](https://github.com/microprediction/timemachines) repo
(`benchmarks/RESULTS.md` section 6, harnesses beside it); this note is the
package-facing summary.*

## Thesis

Run every input stream of an online regression through its own Laplace
forecaster and replace the raw value with two scalars: the one-step
predictive mean and the standardized surprise z, the probability integral
transform of the value under the predictive, pushed through the standard
normal quantile and clamped so |z| stays below about 7. The target keeps
its own pair too, supplied by a wrapper since transformers never see y,
but the target itself stays raw. The regression then learns from
variables that are stationary in scale, bounded in influence, and
calibrated, while all the marginal statistics live in the forecasters.

This is the regression instance of the Rosenblatt front-end thesis
already established for anomaly detectors (DSPOT 0.100 to 0.517 on
UCR-60) and for forecasters (five opponents, 149/150 wins on FRED-30):
other people's methods get better in Laplace coordinates.

## Prong one: contaminated simulation, one learner, coordinates varied

One fixed recursive-least-squares learner predicts y one step ahead on a
linear Gaussian pair, x AR(0.8) driving y. Only the coordinate system
varies. Eight scenarios, 30 seeds, excess MSE against the clean
conditional mean, medians. zin is the two-scalar recipe.

| scenario | raw | zscore | robust | huber | zin |
|---|---|---|---|---|---|
| clean | **0.001** | 0.036 | 0.108 | 0.001 | 0.057 |
| spikes on x | 1.08 | 0.98 | 0.91 | 1.38 | **0.72** |
| spikes on y | 3.78 | 5.66 | 4.42 | 3.75 | **1.84** |
| spikes on both | 9.05 | 9.78 | 7.66 | 9.65 | **6.33** |
| heavy t(2) | 0.008 | 2.05 | 2.17 | **0.006** | 0.95 |
| drift | **0.003** | 0.059 | 0.232 | 0.003 | 0.059 |
| distortion | 1.05 | 0.71 | 0.53 | 1.48 | **0.51** |
| shift | **0.001** | 0.105 | 0.159 | 0.001 | 0.089 |

zin beats raw 30/30 seeds on every contaminated scenario and beats the
purpose-built robust baseline on all of them. Its clean-data toll is
0.057. Two rows where it loses matter as much as the wins: with genuine
t(2) driving noise and a correctly specified model, the extreme points
are the most informative ones and any taming costs accuracy; and the
Huber row shows the loss is the wrong place to fix inputs, since clipping
a residual cannot repair a corrupted feature.

## Prong two: river's learners, river's baseline

Same generator and seeds, but the learners are river's LinearRegression
and HoeffdingTreeRegressor and the baseline is river's own recommended
pipeline, StandardScaler on features plus TargetStandardScaler on the
target. The front-end composes with that pipeline rather than replacing
it. Sign test: 30/30 is p = 1.9e-9.

| scenario | lin std | lin fronted | wins | tree std | tree fronted | wins |
|---|---|---|---|---|---|---|
| clean | **0.010** | 0.080 | 0/30 | **0.011** | 0.086 | 0/30 |
| spikes on x | 1.26 | **0.94** | 30/30 | 1.30 | **0.96** | 30/30 |
| spikes on y | 5.04 | **3.12** | 30/30 | 4.48 | **3.45** | 27/30 |
| distortion | 1.10 | **0.81** | 26/30 | 1.11 | **0.88** | 19/30 |

KNNRegressor is a counterexample: neighbour averaging is already
spike-robust and the extra dimensions degrade its distance metric.

## Prong three: river's own datasets

Progressive validation MAE, numeric features, untouched data. The body
column is the target's Laplace predictive mean alone, no regression, no
features, and it is the attribution control.

| dataset | river pipeline | fronted | body alone |
|---|---|---|---|
| TrumpApproval | 0.334 | 0.381 | **0.150** |
| ChickWeights | **23.8** | 24.7 | 25.5 |
| AirlinePassengers | 41.9 | **26.6** | 29.4 |
| Bikes (20k) | 5.07 | 5.29 | **4.94** |

On history-dominated streams the univariate forecaster alone already
beats the full feature pipeline, on river's flagship documentation
example by 2.2x. The features add almost nothing the calibrated
forecaster had not extracted, which is the same "plus epsilon" finding
the forecaster front-end study produced on FRED. Where features carry
entity identity that one forecaster per key cannot represent
(ChickWeights interleaves 50 growth curves), the pipeline keeps its edge.
Under 2% injected feature spikes the front-end wins 10/10 on
TrumpApproval, AirlinePassengers and Bikes for the tree learner.

The demo in this repo reproduces the deployment-shaped version:
TrumpApproval, baseline 0.328 clean and 0.597 under 2% corrupted
readings, fronted 0.382 and 0.407.

## Why it should work, in three provable statements

First, the map from y to z through a strictly increasing conditional CDF
is a causal bijection and log-loss transfers through it exactly, by
change of variables, with no calibration assumption; calibration only
ever appears as a measured KL term. Second, under a calibrated body the
target's z is unpredictable from its own past, so the body's excess
log-loss over a full-information oracle decomposes exactly into transfer
entropy plus prequential miscalibration; a regression on z can recover at
most the first term. Third, z features and z targets are bounded by
construction, so worst-case online regression guarantees of the
Vovk-Azoury-Warmuth kind hold with universal constants on arbitrary data.
Bounded influence is also why input fixing is safe where output fixing is
not: a wrapper that learns against the target's own surprise inherits the
forecaster's sensitivity to target contamination, and the study measured
that failure at 60x. The output sandwich is therefore not part of the
recipe; see the footnote in the timemachines results log.

## The ablation: which scalar carries it

On the simulation, z alone is useless (excess MSE 17 to 146: surprises
carry no level, and a level cannot be regressed from pure news), mu alone
pays a large clean toll (0.18) and misses distortion, and the pair
recombines into a conditional mean neither scalar supports alone. The
target's own pair is insurance for the target specifically: swapping it
for the raw lag is better on clean and heavy-tailed rows, worse under
target contamination and distortion.

On river's datasets, dogfooding this package, the sharpest cell is the
one the first pass missed: LaplaceTarget alone, raw features untouched.
MAE, tree learner, untouched data: TrumpApproval 0.301 vs the pipeline's
0.334, AirlinePassengers 26.6 vs 41.9, Bikes 5.01 vs 5.07, ChickWeights
24.1 vs 23.8. Under 2% feature spikes it wins 10/10 on the first three
despite its features being the raw, spiked ones: the target pair anchors
the prediction. The recommendation therefore orders itself: add the
target pair always, one wrapper, helps clean and contaminated alike;
replace features with their pairs only when you distrust the features;
never use z alone.

## Cost, measured

LaplaceFeatures runs at about 390 microseconds per stream per sample
single-threaded, roughly 900x river's StandardScaler, or 2,500 samples
per second per stream. Right for polls, sensors, market bars and
anything at human timescales; wrong inside a hot path at hundreds of
thousands of ticks per second.

## The theorem

The theory sketch above is now a theorem with proofs and a numerical
check: pathwise regret transfer with measured constants, an insurance
corollary resolving the output-sandwich pathology as point extraction
rather than density failure, and the oracle decomposition. See
[regret-transfer.md](regret-transfer.md) and
`tests/test_regret_bound.py`.

## Status and next steps

Preliminary. Open items, in order: per-entity bodies for interleaved
streams (the ChickWeights fix); multi-horizon surprises from k=3 bodies
for the drift and shift rows; and concept-drift classification on Elec2
and Insects. Reproduction: five harnesses in timemachines `benchmarks/`
(`regression_frontend.py`, `river_frontend.py`, `river_data_frontend.py`,
`ablation_frontend.py`, `ablation_river.py`), all resumable, seeds
fixed.
