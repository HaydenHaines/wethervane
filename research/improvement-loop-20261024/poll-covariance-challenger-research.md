# M01 challenger research: poll covariance scale uncertainty

**Status:** hypothesis and design only. This is not a finding that polling is the weakest layer, a qualified method, or authorization to refit the active Senate pilot or House polling. No model code or fit was changed. **The Round 1 mixture-likelihood sketch below is superseded by the appended Round 2 correction:** after downstream normalization it does not, by itself, preserve the calibration cut.

## What the selected implementation actually does

The selected candidate-role graph constructs each poll mean as the log rate of the first named candidate minus the second, and observes it with `pm.MvNormal(..., cov=np.array(block["covariance"]))` in [`joint_role_hierarchy.py`](/home/hayden/projects/wethervane-redesign-20261006/integration/research/senate-stack-integration-20261020/candidate-role-hierarchy-astra-v1/joint_role_hierarchy.py:386). The covariance is supplied data; the count graph does not infer its scales.

[`poll_covariance_v1.py`](/home/hayden/projects/wethervane-redesign-20261006/integration/src/forecasting/poll_covariance_v1.py) uses five total-error components on principal-contender log-odds/log-rate-ratio scale: independent, same-cycle, same-race-and-cycle, same-pollster-and-cycle, and race/cycle movement proportional to the minimum pair age in 30-day units. The independent component already includes sampling, weighting, rounding, and residual measurement error. Adding a separate poll-MOE variance would double-count that part of the fitted total error. The fitting objective is explicitly a penalized likelihood/log-scale MAP, not integrated scale uncertainty.

The TX-disjoint successor in [`tx-disjoint-calibration-sol-v1`](/home/hayden/projects/wethervane-redesign-20261006/integration/research/senate-stack-integration-20261020/tx-disjoint-calibration-sol-v1/README.md) fits five regularized SDs to 1,483 polls from 105 races across 2012, 2016, 2018, and 2020, excluding TX2020 and TX2024 records. The persisted [`calibration.json`](/home/hayden/projects/wethervane-redesign-20261006/integration/research/senate-stack-integration-20261020/tx-disjoint-calibration-sol-v1/attempt-v1/calibration.json) has one vector of SD estimates, regularization inputs, optimizer starts/objectives/gradients, and point-fit metadata. It has **no posterior sample, interval, curvature/Hessian uncertainty, or posterior predictive draws**. Its receipt explicitly says `covariance_uncertainty_integrated: false`. Disjointness repairs event overlap; it does not make the calibration Bayesian or integrate its parameter uncertainty. The calibration still notes same-cycle, other-state dependence is not jointly modeled.

This distinction is epistemic uncertainty about the five error scales versus aleatory poll residual variation conditional on those scales. The existing Gaussian covariance already describes conditional total poll error. Parameter integration would average that conditional law over uncertainty in the scales; it should not simply inflate or add another residual variance.

## Versioned, replaceable challenger

**Proposed name:** `tx_disjoint_poll_cov_posterior_mixture_v1` (research-only until source ownership and cutoffs are admitted).

**Estimand:** the conditional predictive law for the vector of retained named-contender poll log-odds/log-rate-ratio measurements, given a fixed count-state vector and the event-disjoint historical calibration bank. Component SDs are in log-odds units; movement SD is per square-root 30 days. It does not change the election target or produce a new poll sampling-error term.

Let `theta = log(sigma)` be the five component log SDs and `K` the existing five-kernel operator. Reuse the exact current residual definition, kernels, prior medians and log-scale SD, and disjoint row ledger. Instead of retaining only `theta_MAP`, retain draws from the calibration posterior

`p(theta | D_cal) ∝ p(D_cal | theta) p(theta)`

using the existing Gaussian calibration likelihood and its existing log-scale Gaussian regularizer/prior specification, with proper finite bounds matching the current optimizer support. This is a standard hierarchical Bayesian treatment: uncertainty in covariance parameters is propagated through the predictive distribution, rather than replaced with a point estimate (see [Stan User’s Guide, hierarchical regression](https://mc-stan.org/docs/stan-users-guide/regression.html)). The five-dimensional posterior may be sampled separately; no need to put all calibration rows into every count-graph chain.

For a target poll block with coordinates `x` and fixed count states `z`, define the **cut** poll law

`p(y_poll | z, D_cal) = ∫ Normal(y_poll; m(z), Σ(x, theta)) p(theta | D_cal) dtheta`,

where `Σ(x,theta) = Σ_k exp(2 theta_k) K_k(x)`. Keep each draw’s five scales shared across the complete target block (and according to the predeclared cycle/block scope); never sample a new scale per poll or independently per race. Compute the mixture likelihood by posterior draw log-sum-exp or another verified Monte Carlo integration method. The calibration posterior stays frozen: target poll data and target outcomes do not update `p(theta|D_cal)` or feed back to calibration rows. The count model must be refit under the integrated likelihood as a whole; do not splice forecast columns or use posterior-mean covariance as a substitute for the mixture. In the downstream graph, state uncertainty in `z` still combines with conditional poll residual error and scale uncertainty in the usual joint posterior.

A joint model could instead infer all five scales in the same graph, but then the target polls update those scales. That is not a fair “uncertainty-only” challenger to the current disjoint calibration and risks feedback/double use across training/target evidence. The cut posterior mixture is the cleaner first comparison because it preserves the calibration ownership boundary while passing uncertainty to the downstream law. A joint graph should be a separate scientific version, with distinct evidence ownership and a separately justified target.

## Evidence ownership and fair comparison

- **Calibration owner:** the exact 1,483-row disjoint bank and exclusion ledger; preserve its four-cycle split, observation IDs, share targets, logit residual transform, poll coordinates, existing kernels, regularizer, and published cutoff. Do not reuse TX2020/TX2024 downstream outcomes as calibration evidence.
- **Target owner:** the existing target poll rows and candidate-role count graph. Keep identical poll observations, counts, priors, non-poll layers, data cutoff, worker/runtime path, and downstream target. Only replace fixed MAP covariance with the precomputed calibration posterior mixture.
- **Validation owner:** disjoint held-out poll units or leave-one-election-cycle-out predictive checks on calibration rows, followed by the same whole-graph held-out election scoring as the MAP baseline. Because there are only four cycles, report cycle-level uncertainty and sensitivity; do not treat 1,483 polls as 1,483 independent election replications.
- **Comparison:** paired, same-target runs: (A) existing disjoint MAP scale/covariance baseline; (B) cut posterior-mixture scales. Compare held-out poll log predictive density and predictive interval coverage first, then target count/election proper scores and calibration with uncertainty. A proper scoring rule rewards issuing the predictive distribution that best represents the outcome law; it does not by itself validate the causal interpretation or overcome sparse independent cycles ([Gneiting & Raftery 2007](https://doi.org/10.1198/016214506000001437)). Run the same complete fit/refit and Monte Carlo precision protocol for both arms.

## Risks, falsification, and scope

The five kernels can be weakly distinguished or collinear with only four cycles; cycle, race, pollster, and movement components may trade off, making posterior widths prior-sensitive. Other-state polls within one cycle may share dependence the current calibration does not represent. Poll sampling/weighting definitions and poll-to-candidate coordinates must remain aligned. A scale mixture may have heavier tails than a single Gaussian even if marginal SD barely changes; judge the whole predictive law rather than only variance.

Do not add another margin-of-error term. Do not let the target poll likelihood update the cut calibration posterior. Do not call a small change in forecast spread “better calibration” absent held-out scoring. Do not claim any weak layer has been empirically identified yet.

**Reject or defer this challenger** if posterior scales are effectively concentrated and the integrated law is indistinguishable from MAP at adequate numerical precision; if held-out poll predictive scores/coverage do not improve or degrade; if results hinge on prior/bound choices or cycle deletion; if scale components cannot be identified well enough to support a stable mixture; or if the full count refit is not computationally/diagnostically comparable. If calibration-only predictive checks support the mixture but election-level scores do not, retain it only as a poll-error-law finding, not as a count-forecast improvement.

**Safe next step:** design and independently review the posterior sampler / mixture-likelihood contract and cycle-heldout evaluation. Do not run it against owners’ active Senate pilot or House polling without a new explicit evidence/cutoff admission.

## Round 2 correction: preserve the cut after downstream normalization

The first proposal's single integrated likelihood expression is **not by itself a cut posterior** once the downstream posterior is normalized. If one uses `p(y|z,Dcal)=∫p(y|z,theta)p(theta|Dcal)dtheta` inside Bayes' rule and normalizes over `z`, the evidence normalizer generally depends on `theta`; this reweights the calibration posterior toward scale draws that make the downstream data globally more probable. Freezing the input draw weights does not prevent that feedback. Plummer's cut-function discussion describes the valve as blocking this downstream-to-upstream feedback; a later general framework formalizes the cut distribution as a mixture of downstream posteriors indexed by upstream posterior draws ([Plummer 2015](https://doi.org/10.1007/s11222-014-9503-z); [Carmona & Nicholls, “A general framework for cutting feedback within modularized Bayesian inference”](https://doi.org/10.1093/jrsssb/qkaf012)).

The actual cut target must be computed in two stages:

1. Draw `theta^(s) ~ p(theta | D_cal)` once from the frozen disjoint calibration module.
2. For each draw, fit/sample the full downstream count posterior `p(z | D_down, U, theta^(s))`, with the rest of the graph and evidence unchanged. Then form the predictive/posterior mixture as the equally weighted mixture of those separately normalized downstream posteriors: `p_cut(z | D_down,U,D_cal) = ∫ p(z | D_down,U,theta) p(theta | D_cal) dtheta`.

Do **not** integrate the poll likelihood first and normalize only once if the scientific requirement is a hard cut. That yields a different, feedback-reweighted law in general. Computation can use repeated conditional downstream fits or a validated cut-inference algorithm; any approximation must demonstrate preserved upstream draw weights and separately normalized conditional posteriors. This is more expensive than the first note implied. A simpler **fully joint** alternative can put calibration data in one graph and target data in another with all five scales shared; this correctly lets all data update the scales, but changes evidence ownership and is not a cut. It is a distinct challenger, not an equivalent implementation.

The calibration objective uses a Gaussian penalty on log scales but the optimizer also imposes `[log(1e-5), log(2)]` numerical bounds. Those bounds are not automatically scientific prior support. A posterior version must explicitly choose: (A) proper truncated Normal priors on each log scale over those bounds, or (B) unbounded Normal log-scale priors and wider numerical sampler support/diagnostics. Treat them as separate sensitivity versions; do not silently call optimizer bounds the existing prior. No covariance posterior draws or interval estimates exist in the saved artifact today.
