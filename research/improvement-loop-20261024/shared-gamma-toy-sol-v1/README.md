# Tiny shared-κ Gamma-event/Poisson reconstruction study

Eight tests pass in0.019s using standard-library exact fractions; no scientific libraries, actual graph, owner diagnostic, model fit, random simulation or data archive was used. This is an invented mathematical study, not actual2018 compiled-target equivalence. toy.py/result.json and test_toy.py/tests.log are reproducible from this directory:

```sh
PYTHONDONTWRITEBYTECODE=1 nice -n 15 python3 -B -m unittest -v test_toy
PYTHONDONTWRITEBYTECODE=1 nice -n 15 python3 -B toy.py
```

## Defined nondegenerate toy

Protected parent U takes eight values0..7 with fixed mass1/8. θ is binary with prior(1/2,1/2). Observed Y=1 has likelihood L(θ)=1/4 forθ0 and3/4 forθ1, independent of U,κ,G. Thus p(θ|Y)=(1/4,3/4). One sharedκ per whole state has discrete priorP(κ=1)=P(κ=2)=1/2. This deliberately differs from the actual model's continuous LogNormalκ prior. Conditional onκ, two event frailties G1,G2 are independent Gamma(shape=κ,rate=κ), continuous on positive real support. Future counts Ne|G,θ,U are independent Poisson(λeG_e), with λe=1+U+2θ+e for e0,1. These positive rates and randomκ/G/counts are nondegenerate.

The unnormalized joint after observing Y is w_U p(θ)L(θ)p(κ)Gamma(G1|κ,κ)Gamma(G2|κ,κ). Each Gamma integrates to1 because κ^κ/Γ(κ)×Γ(κ)/κ^κ=1; summing p(κ) also gives1. Therefore integratingκ,G preserves p(θ|Y,U). Each within-parent posterior is normalized, then mixed with original w_U=1/8. No downstream evidence-normalized parent weights are computed. Because this toy also makes Y independent of U, it is not a counterexample distinguishing cut from uncut weighting; original masses are explicitly preserved by construction.

Removingκ/G gives equality only of the θ marginal, not equality of the original full joint pointwise. Correct augmentation samples exactly oneκ from its prior, then bothG conditional on that sameκ, then reconstructs predictions. Our verification enumeratesκ and analytically integrates the continuousG; it does not implement or validate a random Gamma generator or compiled posterior executor.

## Exact predictive checks and wrong alternatives

The Gamma-Poisson integral is

P(N=n|κ,λ)=Γ(κ+n)/(Γ(κ)n!)×[κ/(κ+λ)]^κ×[λ/(κ+λ)]^n.

toy.py computes this independently as a density-integral coefficient and as a NegativeBinomial mass; tests compare48κ/rate/count combinations exactly. The probability-generating function evaluated at1 verifies normalization without treating a truncated finite count sum as complete support. Tests compare16joint count cells between full integral and shared reconstruction, retaining allθ/parent mixture weights. result.json saves five example joint probabilities.

For U0,θ0,λ=(1,2), correct shared-κ P(N1=0,N2=0)=5/36. Incorrectly drawing separateκ per event gives119/864, differing by1/864. Both approaches have identical one-event marginals: the error is in joint dependence. Conditional raw count covariance is nevertheless0 here, because both conditional meansλe areκ-independent; zero covariance does not imply independence. Sharedκ affects higher-order/event-zero dependence and must not be replaced by per-event draws.

Fixing G1=G2=1 instead gives independent Poisson joint-zero probability exp(-3), distinct from5/36. Forλ2, correct mixture variance is λ+λ²E[1/κ]=5, while fixed frailty gives variance2 despite the same mean2. The fixed-frailty alternative is explicitly a changed predictive law.

Optional observation-factor counterexample is included: if N1=0 is observed rather than merely predicted, P(κ|θ0,U0,N1=0) becomes(9/17,8/17), not(1/2,1/2). A count-dependent observation factor prevents prior-only factorization. Its θ-dependent evidence also means the previous θ posterior cannot simply be reused; the example records the conditionalκ change rather than claiming a complete new posterior fit.

## Boundary and remaining work

Tests cover positive normalized nuisance integration, nontrivialθ posterior, exact integral/NB agreement, full small joint reconstruction, both wrong alternatives, observation coupling and fixed parent masses. Exact fraction results are proofs/checks of this defined finite-κ hierarchy; they are neither empirical sampling validation nor a universal theorem for all observation laws.

The actual2018 source hypothesis requires an independent complete compiled/dependency audit, floating-point cancellation checks, exact LogNormal/Gamma reconstruction, source/coordinate/target/executor contracts and all physical prediction/RNG/row/parent lineage checks. Generated counts still depend on frailty even if the observed likelihood does not. No original failed diagnostic is removed or reclassified; no existing gate is relaxed. This packet does not implement an actual collapsed target, count forecast, sampler or restartable executor.
