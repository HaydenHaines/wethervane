# Independent review: E01 supplied-point runtime probe v1

**Review date:** 2026-10-25  
**Scope:** Numerical compatibility of the frozen PyMC graph and already reviewed analytic density/gradient at supplied points. No sampling, scoring, refit, or owner-law edits.

## Preserved first check and follow-up

The initial recorded probe remains in `economic-execution-v1/runtime/initial-check-failure.json`. Its joint natural-log density discrepancy was `7.393521741505538e-7` for H0 against an initial absolute tolerance of `1e-7`; transformed gradient discrepancy was only `1.2789769243681803e-13`. This failure is appropriately retained rather than overwritten.

The subsequent factor investigation found differences of roughly `2.46638e-9` for each of 299 Student-t response terms and `1.90465e-9` for the HalfNormal race-scale factor, while other terms were at floating-point epsilon or zero. These aggregate to the observed roughly `7.3935e-7` density difference. The near-zero gradient difference is consistent with a normalizer-only discrepancy, but that alone would not justify relaxing the check; the revised checker separates factor errors, transformed Jacobian identity, gradient, and variation in the aggregate offset at a second complete state.

## Revised check review

I reviewed `check_runtime.py`, the preserved first checker, both investigation/result receipts, and the pinned preparation plan; I did **not** rerun the exclusive probe or any fit. The revised checker pins the preparation-plan digest and all plan source pins before importing `prepare`, and requires the dedicated numerical stack versions. It evaluates the unchanged H0, F0, and E01 laws at two distinct complete supplied states per arm. It checks PyMC’s unfactored log density, each compiled log-density factor against independently assembled SciPy Normal/HalfNormal/Student-t terms, the transformed-coordinate Jacobian identity, the transformed gradient against the analytic natural-coordinate gradient, and whether the PyMC-minus-analytic log-density offset changes across those states.

The recorded run passes its declared bounds:

- maximum per-factor discrepancy `2.466381765486858e-9` versus `5e-9` bound;
- Jacobian identity error at most `2.098321516541546e-14` versus `1e-10` bound;
- transformed-gradient discrepancy at most `4.263256414560601e-13` versus `1e-7` bound;
- same-arm aggregate offset changes at most `3.694822225952521e-13` across the two supplied states, versus `1e-10` bound.

The aggregate joint log-density discrepancy remains about `7.3935e-7`; the revised check does not hide or re-label that difference as zero. Instead it documents the observed per-term normalizer mismatch and checks that the total offset is state-invariant at two deliberately distinct complete points. The fixed factor counts and shared priors across the three arms make the proposed normalizer explanation plausible, but this probe establishes only these tested points and compiled factors; it is not a universal equality proof or an independent derivation of all possible states.

The transformed-gradient comparison uses the expected log-scale chain rule and adds one Jacobian derivative per log-transformed positive scale. The explicit identity `logp_transformed - logp_natural = log(tau_cycle) + log(tau_race)` is also checked. No owner law, prior, transformation, sampler, qualification gate, or outcome label was altered.

The ArviZ R-hat/ESS and NetCDF exercise is explicitly a deterministic API fixture and a serialization round-trip, not a diagnostic evaluation of an MCMC fit. The result reports no MCMC, no target labels scored, and no forecast qualification. The locked runtime includes `h5py 3.16.0` and `h5netcdf 1.8.1`, allowing the temporary NetCDF round-trip.

## Assessment

I find the revised runtime probe adequate for its narrow claim: the pinned PyMC graph agrees with the analytic graph to the stated per-factor and gradient tolerances at the supplied points, and its transformed Jacobian behavior matches the intended log-scale parameterization. The recorded initial failure and its small constant-term source remain visible. The check does not establish predictive performance, posterior quality, MCMC execution, or downstream forecast qualification. The stated output is `status: passed` only for this supplied-point computational compatibility check.

**Reviewed result hash:** `ff4add4f607a77d01b3567113557695adf83e7f77d362031659ff98426242706`  
**Reviewed checker hash:** `a493d4884b8a222fdb6bc809770571ded611592e8496f1dab01635bab76d9abc`  
**Frozen preparation plan hash:** `cfe669f6cef8cec82702b03e9ced060edf022fd434df131483515a054d24aef9`
