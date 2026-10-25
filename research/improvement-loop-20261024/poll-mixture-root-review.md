# M01 shared polling-scale uncertainty: reviewed preparation

October 25, 2026 UTC. Research primitive only. No calibration posterior bank, joint refit, empirical superiority or publication is established.

The selected Senate poll covariance is a fixed calibrated combination of independent, cycle, race, polling-house and movement kernels. The reviewed source has 1,483 calibration polls across 105 races in 2012/2016/2018/2020, excluding the declared Texas target events. Its point optimization and regularizer are not a posterior uncertainty bank. A historical replay also needs earlier-only calibration: excluding a target state does not make other states' same-election outcomes available before that election.

`poll_scale_mixture.py` supplies a normalized Gaussian mixture for one complete poll vector. One shared scale-component index governs every poll in that vector. Averaging a separate mixture for each poll would create a different dependence model. The original covariance provider still owns all kernels, units and sampling variance; the primitive does not add a second sampling-error term.

## The scientific target correction

The original research proposal blurred two distinct targets. Given calibration evidence C, downstream evidence Y, downstream parameters z, population parent U and covariance scales theta, the marginal likelihood `integral L(Y | z, theta, U) p(theta | C) dtheta` is ordinary marginal Bayesian inference. After downstream normalization, scale components are reweighted by their downstream evidence normalizers. It does not preserve the calibration distribution as a protected cut.

A calibration cut instead averages separately normalized complete downstream posteriors for each `(U, theta)` pair, preserving the declared population and calibration masses. The [independent Luna review](poll-covariance-target-review-luna.md) demonstrates the difference with an exact finite example: the marginal target yields probability 1/10 and the protected calibration cut yields 21/82. The prototype test includes that counterexample.

Preserving the existing population cut does not automatically protect calibration scales. Both targets can be explicitly defined, but they are different experimental arms. Do not advertise marginalization as uncertainty-only feedback protection. [Carmona and Nicholls](https://proceedings.mlr.press/v108/carmona20a.html) explain why modular normalizers matter; [Stan's posterior prediction guide](https://mc-stan.org/docs/stan-users-guide/posterior-prediction.html) describes integration over parameter uncertainty.

## What has been verified

Six prototype tests cover joint density against SciPy, finite-difference gradients, a one-component reduction, permutation/split-weight invariance, one-dimensional normalization and tails, shared-index dependence, the cut counterexample and rejected invalid inputs. Root inspected the implementation. The complete local workflow suite passes 32 tests: twelve journal, fourteen receipt-adapter and six mixture tests.

[Root's actual-input check](evidence/poll-mixture-actual-input-root-v1.json) uses the existing six-poll Texas component as a one-point covariance mixture. At three supplied mean vectors, the maximum density error is 7.11e-15 and gradient error 8.44e-15 against the fixed-covariance law. Source and prototype hashes are recorded. This establishes bounded compatibility at supplied points, not a sampled model or evidence of useful uncertainty propagation.

The primitive validates finite inputs, normalization, covariance symmetry and positive definiteness, and stores immutable copied arrays. It is a float64 research implementation for ordinary model scales, not a guarantee for every extreme representable covariance. It has no calibration sampler, automatic differentiation binding, cut scheduler, prior selection, model registry or activation capability.

## Next experiment boundary

Before fitting: explicitly choose and version the calibration-feedback target; bind the complete incumbent and unchanged covariance kernels; obtain time-safe earlier-cycle calibration data; define the normalized prior on the scales instead of silently treating optimizer bounds as scientific prior truncation; create and diagnose a shared scale posterior bank; and integrate it with separately normalized population conditionals. Refit every affected downstream parameter, holding other component definitions and matched data fixed where possible.

Then compare the point-scale incumbent and uncertainty challenger with matched sources, horizons and coverage, numerical precision, proper scores, calibration and useful sharpness across A2018/A2022 and B2020/B2024. This workspace makes no weakest-layer ranking and starts no competing owner pilot. The primitive and negative target finding are retained so future work does not repeat the same mistake.
