# Independent review of shared-κ Gamma–Poisson toy

Reviewed `../shared-gamma-toy-sol-v1` as an invented mathematical study only. I independently reran its 8 standard-library tests (8 passed in 0.017 s). No graph, model, fit, owner work, array bank, or scientific data were executed.

## Assessment

The main identities and examples are correct. For `G|κ ~ Gamma(κ, rate=κ)` and `N|G ~ Poisson(λG)`, integrating `G` gives the negative-binomial mass

`P(N=n|κ,λ) = Γ(κ+n)/(Γ(κ)n!) [κ/(κ+λ)]^κ [λ/(κ+λ)]^n`.

With no observations involving the block, each conditional Gamma density integrates to one for each κ, so integrating all event multipliers and then the normalized κ prior leaves the within-parent posterior unchanged. The exact discrete-κ example correctly distinguishes one shared κ from an independent κ per event: for `(λ1,λ2)=(1,2)`, the shared joint-zero probability is `5/36`, while separate κ draws yield `119/864`. Fixing frailties to one gives `exp(-3)` and changes predictive dispersion. The active count example correctly shows that observing `N1=0` updates κ to `(9/17,8/17)` in the discrete prior toy; the block is then not prior-only, although G can still be integrated conditional on κ to obtain an NB likelihood.

The shared-κ dependence example also extends beyond the toy’s discrete prior. For any nondegenerate positive random κ and λ>0, `qλ(κ)=P(N=0|κ,λ)=(κ/(κ+λ))^κ` is strictly decreasing: its log derivative is `-log(1+λ/κ)+λ/(κ+λ)<0`. Thus for two conditionally independent event counts, `Cov(qλ1(κ),qλ2(κ))>0`, so sharing κ increases the joint probability of both events having zero counts relative to independent κ per event. Pairwise raw count covariance can still be zero because `E[N_e|κ]=λ_e` does not depend on κ; the dependence appears in joint tail/zero events and higher moments.

## Parent-mass challenge

The source toy explicitly preserves eight `1/8` masses and discloses that its observed datum has equal evidence for every parent, so the test cannot distinguish fixed-mass cut behavior from accidental evidence reweighting. I added `parent_cut_check.py`, an exact-rational counterexample with masses `(1/8,7/8)` and parent-specific evidence `(1/4,3/4)`. The protected cut keeps `(1/8,7/8)` after integrating the normalized block inside each parent; ordinary Bayesian evidence reweighting would instead give `(1/22,21/22)`. The check passed. This complements the study; it is not a test of actual parent data or code.

## Boundaries

The study uses a discrete `κ∈{1,2}` prior rather than the selected model's continuous LogNormal prior. Its exact fractions establish the toy calculations, not finite Monte Carlo or compiled model equivalence. The continuous monotonicity argument above supports the shared-κ dependence direction but does not validate a sampler or output executor.

The actual 2018 factorization still requires a complete pinned dependency/compiled-source audit, floating-point cancellation checks, the exact continuous LogNormal/Gamma target, preserved fixed parent masses, and full prediction/RNG lineage. A normalized prior block may be marginalized from the observed likelihood only while every active observation is independent of it. Active official counts, participation reports, ranked/runoff/open count observations, or additive rate floors invalidate the prior-only simplification. Any future collapsed or reconstructed implementation must preserve the parent cut and the joint predictive law; this toy does not qualify that implementation.
