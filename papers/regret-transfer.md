# Regret transfer through the front-end

*A theorem with measured constants. Preliminary write-up, July 2026.
Companion to [regression-frontend.md](regression-frontend.md); the
numerical bound check is `tests/test_regret_bound.py` in this repo.*

## What is claimed, in one paragraph

Run any causal forecaster over a stream, take the parade surprises, and
let an online linear learner refine them. Conjugate the refinement back
through the forecaster's predictive CDF. Then the log-loss of the
composite, measured in the original coordinates on the actual data, is
within an explicitly computable $O(d \log T)$ of the best refinement in
its class, for every data sequence whatsoever. There is no stationarity,
boundedness, moment, or calibration assumption anywhere: the clamp built
into the parade manufactures the boundedness the online-learning theorem
needs, and calibration never appears except, in the final proposition, as
a measured quantity. Setting the comparator to zero gives an insurance
corollary: the composite can never lose more than $O(d \log T)$ nats to
the forecaster alone, pathwise, which is the theoretical resolution of
the output-sandwich pathology observed in the benchmark study.

## Setup

A *body* is any causal algorithm that, having seen $y_1, \dots,
y_{t-1}$, emits a predictive distribution for $y_t$ with continuous,
strictly increasing CDF $\hat F_t$ and strictly positive density $\hat
f_t$. Gaussian mixtures, hence every skater, qualify. Nothing is assumed
about how well the body forecasts.

The *surprise* of the arriving point is $z_t = \Phi^{-1}(\hat
F_t(y_t))$, and its clamped version is
$\bar z_t = \Phi^{-1}(\mathrm{clip}(\hat F_t(y_t), \varepsilon,
1-\varepsilon))$ with $\varepsilon = 10^{-12}$, so $|\bar z_t| \le z_{\max}
= \Phi^{-1}(1-\varepsilon) \approx 7.03$. This is exactly what the parade
computes and stores in `state["z"]`.

Let $\phi_t \in \mathbb R^d$ be any causal feature vector with
$\|\phi_t\|_\infty \le z_{\max}$, for instance a constant together with
the clamped surprises of exogenous streams and of the target's own past,
which is precisely what `LaplaceFeatures` and `LaplaceTarget` emit.

## Lemma 1 (conjugation identity)

*For any density $q$ on $\mathbb R$, the function*

$$p_q(y) \;=\; \hat f_t(y)\, \frac{q(z_t(y))}{\varphi(z_t(y))},
\qquad z_t(y) = \Phi^{-1}(\hat F_t(y)),$$

*is a probability density, and its log score at the realized point is*

$$\log p_q(y_t) \;=\; \log \hat f_t(y_t) + \log q(z_t) - \log
\varphi(z_t).$$

**Proof.** Substituting $z = z_t(y)$, whose derivative is $dz/dy = \hat
f_t(y)/\varphi(z)$, gives $\int p_q(y)\,dy = \int q(z)\,dz = 1$. The
displayed identity is the definition evaluated at $y_t$. $\square$

The identity is pathwise and exact for the actual, prequentially
estimated $\hat F_t$, however miscalibrated. This is the sense in which
nothing is smuggled in: the body's quality never enters.

## The learner

The refinement class is Gaussian: $q_m = \mathcal N(m, s^2)$ for a fixed
$s > 0$. The composite ("sandwich") uses $m_t = \hat z_t$, the prediction
of the Vovk-Azoury-Warmuth forecaster trained on the clamped surprises:
with $A_t = \lambda I + \sum_{s \le t} \phi_s \phi_s'$ and $b_{t-1} =
\sum_{s < t} \bar z_s \phi_s$,

$$\hat z_t = \phi_t' A_t^{-1} b_{t-1}.$$

Note $A_t$ includes the current $\phi_t$; that forward-looking
regularization is the Azoury-Warmuth device and is what the appendix
bound requires.

## Theorem (pathwise regret transfer)

*For every sequence $y_{1:T}$, every comparator $u \in \mathbb R^d$, and
with $C = \{t : \hat F_t(y_t) \notin [\varepsilon, 1-\varepsilon]\}$ the
set of clamp-active ticks,*

$$\sum_{t=1}^T \Big[\log p_{q_{u'\phi_t}}(y_t) - \log
p_{q_{\hat z_t}}(y_t)\Big] \;\le\; \frac{1}{2s^2}\Big[\lambda \|u\|^2 +
z_{\max}^2 \, \ln \frac{\det A_T}{\lambda^d}\Big] \;+\;
\frac{1}{s^2}\sum_{t \in C} (u'\phi_t - \hat z_t)(z_t - \bar z_t),$$

*and $\ln(\det A_T/\lambda^d) \le d \ln(1 + T z_{\max}^2 d/\lambda d) =
d\ln(1 + Tz_{\max}^2/\lambda)$ under the feature bound. Every quantity on
the right-hand side is computable from the run: $\lambda$, $s$ and
$z_{\max}$ are chosen, $\det A_T$ and the clamp-correction sum are
observed, and $C$ is typically empty since it consists of ticks the body
assigned probability below $10^{-12}$.*

**Proof.** By Lemma 1, for any two refinement means $a, b$ the log-score
difference at tick $t$ is

$$\log q_a(z_t) - \log q_b(z_t) = \frac{(z_t - b)^2 - (z_t -
a)^2}{2s^2},$$

so the left-hand side equals $\tfrac{1}{2s^2}\sum_t [(z_t - \hat z_t)^2 -
(z_t - u'\phi_t)^2]$. Write each term against the clamped surprise and
correct: since $g(z) = (z-\hat z_t)^2 - (z - u'\phi_t)^2$ is affine in
$z$ with slope $2(u'\phi_t - \hat z_t)$,

$$(z_t - \hat z_t)^2 - (z_t - u'\phi_t)^2 = (\bar z_t - \hat z_t)^2 -
(\bar z_t - u'\phi_t)^2 + 2(u'\phi_t - \hat z_t)(z_t - \bar z_t),$$

and the correction vanishes off $C$ where $z_t = \bar z_t$. The clamped
sum is the squared-loss regret of the Vovk-Azoury-Warmuth forecaster on
targets bounded by $z_{\max}$, which the appendix bounds by $\lambda
\|u\|^2 + z_{\max}^2 \ln(\det A_T/\lambda^d)$. The determinant bound is
the AM-GM step $\det A_T \le (\mathrm{tr}\, A_T/d)^d$ with
$\mathrm{tr}\,A_T \le \lambda d + T d z_{\max}^2$. $\square$

## Corollary (insurance)

*Take $s = 1$ and $u = 0$. Then $q_0 = \varphi$ and $p_{q_0} = \hat f_t$
is the body itself, so for every data sequence*

$$\underbrace{\sum_t \log \hat f_t(y_t)}_{\text{body}} -
\underbrace{\sum_t \log p_{q_{\hat z_t}}(y_t)}_{\text{sandwich}} \;\le\;
\frac{z_{\max}^2}{2}\, d \ln\!\Big(1 + \frac{T z_{\max}^2}{\lambda}\Big)
\;+\; \sum_{t\in C} 2\,|\hat z_t|\,|z_t - \bar z_t|.$$

*The sandwich can never lose more than $O(d \log T)$ nats to the body,
pathwise, while gaining without limit whatever linear structure exists in
the surprises.*

This resolves the footnote in the benchmark study. The output sandwich
looked catastrophic there, 60x worse than raw regression under target
spikes, but that failure was a property of extracting a point forecast,
the mean of the pushforward, whose tails a contaminated body inflates. As
a density, scored where densities are scored, the sandwich is safe by the
corollary on any data whatsoever. Fix the inputs for point prediction;
trust the sandwich only in log-loss.

## Proposition (oracle decomposition, in expectation)

*Now, and only now, suppose a truth: $(x_t, y_t)$ adapted to a filtration
$\mathcal G_t$, with $P_t$ the conditional law of $y_t$ given $\mathcal
G_{t-1}$, and let the body be causal in $y$ alone with $\mathcal F_{t-1}
= \sigma(y_{1:t-1})$. Let $Z_t^* $ and $\bar Z_t$ denote the conditional
laws of $z_t$ given $\mathcal G_{t-1}$ and given $\mathcal F_{t-1}$.
Then*

$$\mathbb E \sum_t \mathrm{KL}(P_t \,\|\, \hat F_t) \;=\; \sum_t
I\big(y_t;\, \mathcal G_{t-1} \mid \mathcal F_{t-1}\big) \;+\; \mathbb E
\sum_t \mathrm{KL}\big(\bar Z_t \,\|\, \mathcal N(0,1)\big).$$

*The body's expected excess log-loss over the full-information oracle
splits exactly into the transfer entropy of the exogenous information and
the body's own prequential miscalibration, measured as the divergence of
the actual conditional surprise law from standard normal. The refinement
learner of the theorem competes for the first term only; the second is
the body's business.*

**Proof.** KL divergence is invariant under an invertible transformation
applied to both arguments; applying $y \mapsto \Phi^{-1}(\hat F_t(y))$,
which is $\mathcal F_{t-1}$-measurable and strictly increasing, maps
$P_t$ to $Z_t^*$ and $\hat F_t$ to $\mathcal N(0,1)$, so
$\mathrm{KL}(P_t \| \hat F_t) = \mathrm{KL}(Z_t^* \| \mathcal N(0,1))$.
The chain rule of relative entropy against a fixed reference splits, in
expectation over $\mathcal G_{t-1}$,
$\mathbb E\,\mathrm{KL}(Z_t^* \| \mathcal N(0,1)) = \mathbb
E\,\mathrm{KL}(Z_t^* \| \bar Z_t) + \mathbb E\,\mathrm{KL}(\bar Z_t \|
\mathcal N(0,1))$, and the first term is by definition the conditional
mutual information $I(y_t; \mathcal G_{t-1} \mid \mathcal F_{t-1})$ after
the same invariance is applied once more. Summing over $t$ gives the
display. $\square$

## What is not claimed

No claim that the surprises are independent, standard normal, or even
stationary. No claim about point predictions on the original scale; the
benchmark study measured exactly where those go wrong. And the theorem
compares against the conjugated linear-Gaussian class, not against all
possible forecasters; its force is that the class contains the body
itself and every linear refinement of its surprises, with constants a
run can print.

## Appendix: the Vovk-Azoury-Warmuth bound

*For targets $|\bar z_t| \le Y$ and the forward predictions $\hat z_t =
\phi_t' A_t^{-1} b_{t-1}$ defined above,*

$$\sum_t (\hat z_t - \bar z_t)^2 - \sum_t (u'\phi_t - \bar z_t)^2 \le
\lambda\|u\|^2 + Y^2 \ln \frac{\det A_T}{\lambda^d} \quad \text{for all }
u.$$

**Proof sketch** (standard; Vovk 2001, Azoury and Warmuth 2001, and
Cesa-Bianchi and Lugosi 2006, Theorem 11.8). Let $w_t = A_t^{-1} b_t$ be
the ridge solution through tick $t$ and define the potential $\Phi_t =
\min_w [\lambda\|w\|^2 + \sum_{s\le t}(w'\phi_s - \bar z_s)^2]$, attained
at $w_t$. Algebra gives the per-tick identity

$$(\hat z_t - \bar z_t)^2 = \big(\Phi_t - \Phi_{t-1}\big) + \big(\hat
z_t^2 - w_t'A_t w_t + b_t'A_t^{-1}b_t - \bar z_t^2\big)_{\text{extra}},$$

whose extra part telescopes against $\bar z_t^2\,\phi_t'A_t^{-1}\phi_t$;
summing, $\sum_t (\hat z_t - \bar z_t)^2 \le \Phi_T + \sum_t \bar z_t^2\,
\phi_t' A_t^{-1} \phi_t \le \lambda\|u\|^2 + \sum_t (u'\phi_t - \bar
z_t)^2 + Y^2 \sum_t \phi_t' A_t^{-1} \phi_t$. Finally $\phi_t' A_t^{-1}
\phi_t \le \ln(\det A_t / \det A_{t-1})$ since for $A = B +
\phi\phi'$ one has $\phi'A^{-1}\phi = 1 - \det B/\det A \le \ln(\det
A/\det B)$, and the sum telescopes to $\ln(\det A_T/\lambda^d)$.
$\square$

The numerical check in `tests/test_regret_bound.py` runs the exact
construction above, Laplace bodies, clamped surprises, forward VAW, on
the contaminated generators of the benchmark study, and asserts the
theorem's inequality for a grid of comparators, printing the measured
slack.
