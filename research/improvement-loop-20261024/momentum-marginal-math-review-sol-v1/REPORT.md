# Independent review of supplied Gaussian marginal math v1: corrections required

2026-10-25. Reviewed immutable author manifest `ad58a67d3e204abc6fd02fa23e85205a44a01a681cf12f29f5adda8223993b94`, including `marginal.py` SHA `d145fc7bc94c21f9086fba6d34e4647abc4737e44225871e495f07697baef6b6`. **Do not accept v1 for qualified marginal forecasts.** Three supplied-input numerical boundaries require correction. Owner/author artifacts remain unchanged.

No actual data, numeric target labels, forecast scores, new Gibbs chain, optimizer or parameter fit was used. All checks are supplied deterministic arithmetic; mock quadrature failures test the bounded error policy. The earlier separately authorized synthetic sampler work is not evidence of this helper's scientific forecast quality.

## Findings requiring correction

1. **Small raw weight can dominate a rare tail after linear normalization underflows it away.** Means[-100,100], sd[1,1], positive finite raw weights[1e-300,1e300] produce normalized linear weights[0,1]. Helper log-losing probability is−5005.524208694205; independently normalized logweights give−1381.5510557964274, an error3623.973152897778 in log probability. The helper warns about log(0) but returns a finite wrong answer. Retain authoritative normalized logweights calculated from the original raw weights using logsumexp; linear weights can remain for ordinary absolute-tolerance integrals. Explicit rejection is safer than silent tail loss if the representation is unsupported.
2. **Valid ordinary mixtures can reject their own log-tail output through roundoff.** Four identical N(50,1) conditional logit components with weights[1,2,3,4] give log-win=1.1102230246251565e-16 and log-lose=−1254.8313611394199. `winner_scores` then raises `Finite log tails required` because log-win is slightly positive. Reconcile separately computed log-CDF/log-survival within a declared small rounding tolerance, retaining rare tails and without a statistical probability floor. Large complement inconsistencies must still fail closed.
3. **The zero-width integral shortcut bypasses finite-result checks.** `integral(lambda x:inf,.5,.5)` returns NaN with estimated_error0. Current valid Gaussian endpoint functions are finite, so this is a generic utility/fail-closed defect rather than evidence of incorrect valid endpoint CRPS. Validate scalar/vector zero-width values and raise on nonfinite results instead of claiming zero numerical error.

The review-owned `probe.py` reproduces finding1 from the supplied fixture; `test_independent.py` confirms all three as defects. No author bytes were repaired in place. Root accepted these findings and is creating a separately sealed v2.

## What passes within scope

`test-results-v1.txt` reports13 supplied checks in0.581s: **10 positive mathematical/bounded-error checks plus3 tests explicitly confirming the known defects**. The latter are not qualification passes.

Independent checks cover transformed Normal CDF/survival/density, single-component analytic quantiles and mixture-CDF brackets, normal-domain expected-share quadrature and the weighted component-mean identity, fixed Gauss–Legendre CRPS quadrature including y=0/1, centered CRPS influences and directional finite weight perturbations, quantile influence and WIS derivative agreement, seven-pinball-loss WIS identity, extreme-tail Mills expansion, unclipped winner-score formulas, copied immutable vectors, and invalid arguments. Scalar/vector injected quadrature failure tests verify the exact bounded two-attempt budgets, warning/success handling, refinement agreement failure and nonfinite failure. No empirical score optimization or actual outcome was involved.

The helper's positive upper-survival CRPS influence is correct. **The preserved Sol proposal also already has the correct equivalent upper influence** `−2*SF*(F_t−F)`, equal to `+2*SF*(SF_t−SF)`. Root's v1 `verification.json` incorrectly asserted the proposal used a negative expression in survival-component coordinates. My initial acknowledgment of that claimed proposal error was also mistaken. This review corrects both statements explicitly; neither the proposal nor author v1 is edited. The independent y=0 test isolates the entire upper integral and confirms the correct sign.

## Limits and disposition

This helper is phase independent: it consumes supplied Gaussian conditional components and weights; it neither knows nor certifies pre/post Gibbs phases, chain coordinates, posterior diagnostics, source admission, bank approximation or Monte Carlo uncertainty. It implements no missing-race Student-t mixture and no historical target forecast. A correct one-dimensional arithmetic helper cannot qualify the broader model or whole-chamber predictions. Quadrature error estimates remain estimates, not proof.

Verdict: require a separately versioned correction and bounded delta review for the three defects. The tested ordinary-regime formulas are sound; a new statistical model or broad redesign is unnecessary. All13 review checks, the probe, source pins and this report are sealed in this review directory.
