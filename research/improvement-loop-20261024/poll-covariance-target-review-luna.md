# Independent Luna review: poll covariance uncertainty target

**Date:** 2026-10-24  
**Scope:** Read-only scientific review of [`poll-covariance-challenger-research.md`](poll-covariance-challenger-research.md). No model fits or source changes were made. A small exact finite counterexample used invented values only.

## Finding: the stated integrated likelihood is not a calibration cut

The proposed expression

\[
L_{mix}(y\mid z,D_{cal})=\int L(y\mid z,\theta)\,p(\theta\mid D_{cal})\,d\theta
\]

is a valid posterior-predictive mixture likelihood for a fixed state `z`. It is also a valid ordinary marginalized-Bayes likelihood if `theta` is treated as a random effect with prior `p(theta | D_cal)`. It is **not**, by itself, the modular cut that preserves the calibration posterior marginal after downstream observations are used to infer latent count state `z`.

For one complete downstream conditional, let `D` denote every downstream datum and latent block that belongs to that conditional (including poll, historical count, response, and other factors applicable to that parent). A calibration cut that protects `p(theta | D_cal)` has the form

\[
p_{cut}(\theta, z\mid D_{cal},D)
=p(\theta\mid D_{cal})\,p(z\mid D,\theta),
\qquad
p(z\mid D,\theta)
=\frac{p(z\mid\theta)L(D\mid z,\theta)}{Z(\theta)},
\]

where

\[
Z(\theta)=\int p(z\mid\theta)L(D\mid z,\theta)\,dz.
\]

Thus the cut preserves the original `theta` weights and separately normalizes the **whole downstream conditional for each theta**. Merely integrating the likelihood over `p(theta | D_cal)` and normalizing once over `z` gives the ordinary mixture posterior

\[
p_{mix}(z\mid D,D_{cal})\propto p(z)\int L(D\mid z,\theta)p(\theta\mid D_{cal})d\theta.
\]

If represented jointly, this latter model reweights `theta` by the downstream marginal evidence `Z(theta)`. It is ordinary Bayesian updating of the calibration-derived distribution, not cut feedback.

This distinction matches the primary modular-inference definition: Carmona and Nicholls write the cut posterior as the protected module marginal times the downstream conditional posterior, and the latter contains a conditional normalizer `p(Y | phi)` that blocks reverse feedback ([AISTATS 2020 paper, official PMLR proceedings](https://proceedings.mlr.press/v108/carmona20a/carmona20a.pdf), Sec. 2.1, Eq. 2; Sec. 4.1, Eq. 7).

### Exact toy counterexample

Let `theta ∈ {A,B}` with weights `(1/2,1/2)`, `z ∈ {0,1}` with prior `(9/10,1/10)`, and one observed event have likelihoods:

| `theta` | `L(y | z=0, theta)` | `L(y | z=1, theta)` | `Z(theta)` |
|---|---:|---:|---:|
| A | 9/10 | 1/10 | 41/50 |
| B | 1/10 | 9/10 | 9/50 |

The per-parent cut gives `P_cut(z=1 | y) = 21/82 ≈ 0.2561`, averaging the two separately normalized `p(z | y, theta)` distributions with the original half/half theta masses. The one-time marginal-mixture likelihood gives `P_mix(z=1 | y) = 1/10`, because its integrated likelihood is `1/2` for either z and leaves the z prior unchanged. The answers differ materially. The full mixture posterior would also update the theta weights in proportion to `(41/50, 9/50)`.

## Relation to `population_cut_v1` and the selected count graph

The current operational target protects the population parent and requires a normalized downstream conditional for each protected parent. The local reference implementation enforces that directly: [`src/model_stack_cut_v1/README.md`](</home/hayden/projects/wethervane-redesign-20261006/integration/src/model_stack_cut_v1/README.md>) says each protected parent retains its exact original weight after normalizing the whole downstream conditional; [`assembly.py`](</home/hayden/projects/wethervane-redesign-20261006/integration/src/model_stack_cut_v1/assembly.py:280>) computes a normalizer for each parent and checks the upstream marginal is unchanged. The selected specification similarly defines `population_cut_v1` as `p(U_r | D_N) q(Psi | U_r, downstream evidence)` ([operational specification](</home/hayden/projects/wethervane-redesign-20261006/integration/research/model-stack-operational-specification-20261009.md>), Sec. 1).

This means the proposed mixture likelihood can still be used inside a **population-cut** downstream conditional if the intended target protects only population `U`: construct the complete downstream model using the calibration-predictive mixture, then normalize that entire downstream law separately for each protected `U`. But in that model the calibration posterior over `theta` is not a second protected marginal; target evidence updates its implicit mixture contribution. The proposal must not call this a calibration cut or claim that target polls do not update `p(theta | D_cal)`.

If the scientific intent is also to protect the calibration posterior, explicitly version a target whose upstream parent includes both `U` and `theta`, with original joint mass such as `p(U | D_N) p(theta | D_cal)` only if their evidence is conditionally independent. If evidence overlaps or shares source error, specify its joint/conditional law instead. For each complete `(U, theta)` parent, normalize the entire downstream count-graph conditional and retain that parent's original mass. This is consistent with the selected cut construction and count joint inference; it is a distinct target from the present `population_cut_v1` unless the ownership contract is formally extended.

An explicit joint model that lets target data update `theta` is also coherent as its own standard Bayesian target. It is not inherently invalid, but it should be a separate comparison arm and not be described as the uncertainty-only/cut version.

## Other design issues

### Optimizer bounds are not automatically prior support

`src/forecasting/poll_covariance_v1.py` applies L-BFGS-B bounds `log(1e-5) ≤ theta_k ≤ log(2)` while minimizing a Gaussian log-scale penalty plus likelihood. This makes the recorded result a bounded penalized-likelihood/MAP estimate. It does not establish that these endpoints are scientific prior support. The proposed phrase “proper finite bounds matching the current optimizer support” silently turns numerical optimization bounds into a truncated prior. The existing Gaussian prior on log scales is already proper on the real line; a posterior challenger could retain that support unless a scientific support restriction is separately justified.

If hard truncation is selected, declare it as a new prior/target, include its truncation normalization, report prior mass removed, and test sensitivity. Stan's reference manual defines a truncated density by dividing the original density by its mass on the allowed interval ([Stan Reference Manual: truncation](https://mc-stan.org/docs/2_25/reference-manual/sampling-statements-section.html), “Truncated Distributions”). A computational fail-safe bound can also be used without changing the target only if the sampler verifies posterior mass is negligible at the boundary and does not silently truncate the posterior.

### Same-cycle dependence and retrospective cutoffs

The covariance kernel includes a common `same_cycle` term, and `fit_error_scales` groups calibration residuals by cycle; therefore, same-cycle covariance among rows inside the fitted calibration blocks is represented. However, event/race disjointness is not necessarily dependence disjointness. The local EAC calibration README explicitly says same-cycle, other-state dependence is not qualified. A validation poll held out by race but sharing a calibration year is not automatically independent: either hold out the entire cycle, or derive its conditional predictive distribution given the calibration rows using the shared-cycle covariance. Do not score it as a fresh independent block.

Historical forecast validation also needs a calibration bank rebuilt at each forecast cutoff. The current calibration bank includes 2020 residuals. Excluding TX2020 by race identity does not make those 2020 election-error labels available to a forecast made before the 2020 election. For 2020 replay, use only calibration data and outcome labels released before that forecast cutoff; likewise exclude each target cycle's own post-election residuals. For 2024 replay, confirm each 2020 source/outcome release was available by the frozen 2024 cutoff. Event-disjointness is not time-safe calibration.

For the prospective 2026 TX poll block, the current bank ends in 2020, so no same-cycle calibration residual is apparent; this is the cleaner initial comparison. Preserve cycle as a single shared shock in each target block, not a new independent shock per poll or race.

### Full count refit

The proposal's statement that “the count model must be refit under the integrated likelihood as a whole” is directionally correct. Make the implementation contract explicit: keep non-poll component definitions, inputs, priors, interfaces and evidence ownership matched, but jointly refit the affected poll parameters **and all downstream non-poll parameters** under each arm. Do not freeze the baseline posterior for fundamentals, choice, participation, counts, or other shared latents and only replace poll covariance. That would fail to propagate changed poll information through the joint count model and would compare a different conditional target. Do not splice forecast draws.

## Recommended correction to the research design

Name two different challengers if both are useful:

1. **Calibration-posterior mixture likelihood (ordinary marginalized downstream target).** Integrate `p(theta | D_cal)` to define the target poll-error law, then run the full population-cut count model and normalize the complete downstream conditional per protected population parent. State plainly that the calibration posterior is predictive input, not a cut marginal preserved after target evidence.
2. **Calibration cut (feedback blocked).** Promote `theta` to a protected upstream coordinate beside the population state (with qualified dependence/ownership), retain its exact `p(theta | D_cal)` weight, and separately normalize the whole downstream count/measurement conditional for every complete protected parent. This is the true cut alternative; the per-parent normalizer is `Z(U,theta)` for the full downstream graph, not just an isolated poll likelihood integral.

Keep the existing MAP plug-in baseline as a third reference. For each comparison, calibrate training data inside historical cutoff folds, respect cycle-level dependence, and refit the entire downstream graph. Report poll prediction and full count/election scores separately. A gain only on poll-level log score supports an error-law result, not automatically an election-count improvement.

This review supports the broad hypothesis that point-estimated covariance omits parameter uncertainty; it rejects the current claim that its proposed integrated likelihood alone is a calibration cut. It does not establish that either challenger improves forecasts.
