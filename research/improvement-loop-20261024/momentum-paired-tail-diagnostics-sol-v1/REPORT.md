# Paired complementary tail diagnostics — author complete

2026-10-25. **Nine supplied-only tests pass; independent review pending.** This separately versioned helper implements Root's selected clarification SHA `0d6dd97153687df2583c0a086e92e8a42c22eb0c850ed0a0133acdcb87c14024`, alongside unchanged forecast design SHA `45707d4a933ded2f7d4bffcbaab0fb6fb9a4ab7b1d62ad4494a27422b7d75678`. The accepted trace v1 bytes, thresholds, model, data, streams and attempt budgets remain unchanged.

No target data, numeric labels, empirical scores, Gibbs chains, optimizer or source fit was used. Tests use deterministic supplied arrays and Root's fixed-seed synthetic z fixtures only; random uniform fixture generation is not an inference chain.

## Confirmed numerical boundary and method

Root preserved the z9..10 boundary in `checkpoints/momentum-paired-tail-boundary-probe-v1.json`, SHA `b5980691828f68d056144149b685d1d2549f832a63c9f4dd5343748b0f061c6c`. For varying Gaussian log tails, exponentiating the near-one side can round every conditional probability to1. Original one-sided diagnostics correctly mark that rounded coordinate unresolved; the varying complementary small side still contains the same absolute probability uncertainty.

`probability_coordinates(log_win,log_lose,names)` validates aligned chain/draw/race axes and pointwise complementary log normalization within1e-12. It removes only that shared rounding normalizer, chooses each race's pooled smaller side, and retains its scaled probabilities, centered scaled deviations and log scale. Larger-side centered deviations are exactly the negative of those smaller-side deviations. Pooled and per-chain larger-side log probabilities use stable `log1mexp` of the smaller-side means. All returned arrays are detached, bytes-backed and immutable.

`paired_tail_diagnostics(...)` applies the accepted existing50/100-pass batch/chain MCSE and rank/folded diagnostics to the common noncancelling coordinate. Both sides receive the **same absolute log MCSE**. Each side's relative MCSE uses its stable log probability; larger-side chain log differences use the complementary chain means. Absolute precision, relative precision and chain agreement remain separate existing gates. Resolved side rows retain `status:measured`; explicitly declared structural rows carry `structural_constant:true`. No new gate values are introduced.

Numeric MCSE values that underflow are missing with explicit underflow flags, while finite log MCSE and log-relative MCSE/gates remain available. Underflow is never reported as an exact zero-error certificate. A nonstructurally constant, all-minus-infinity or otherwise unresolved informative small coordinate leaves the pair unresolved. Raw constant indicator diagnostics are unchanged. Each side retains its exact original one-sided summary and status separately so the roundoff boundary remains visible.

The score consumer must also use the common centered deviation: `delta_win = win_deviation_sign * exp(log_scale) * centered_scaled_small`, with loser delta its exact negative. Keep the scale in logs when necessary. This supports noncancelling Brier and negative-log influences; subtracting near-one probabilities or evaluating1-exp(a nearly zero log ratio) would recreate the same problem. The helper supplies these coordinates but implements no score, label reader, retry, forecast or model-selection policy.

Complement validation is an absolute complementary-sum rounding check, not a proof that arbitrary caller-supplied tails came from a particular probability law. Correct Gaussian or empirical paired inputs and their source identities remain the caller's responsibility. Equal conditional component mass is retained; categorical row weights are never reapplied.

## Verification and preserved failure

Final `attempts/pytest-v2.txt`: **9 passed in0.71s**, with Ruff check/format passing. Existing runtime: NumPy2.4.4, SciPy1.17.1, pytest9.0.2, one BLAS/OMP thread, bytecode disabled.

Independent author oracles cover:

- Exact Root seed1516 uniform z9..10 fixture: original larger-side unresolved status is retained, both bulk absolute gates become measured through complement, and shared MCSE agrees with direct Normal-survival batch/chain variance. Relative small-side MCSE≈.0284553352; log absolute MCSE≈−49.4180975.
- Pooled/per-chain complementary log means against direct survival/log1p arithmetic and immutable centered-coordinate identities.
- Win/lose swap and race permutation symmetry, common Rhat/mean-ESS and identical absolute MCSE.
- Stable log1mexp endpoints/moderate identities and invalid inputs.
- A deliberately poorly resolved small tail whose relative/chain gates fail while both absolute gates remain measured; no global retry conclusion is emitted.
- z50..51 underflow: ordinary MCSE displays stay missing, finite log MCSE and applicable gates remain available.
- Nonstructural constants/all-minus-infinity pairs, explicit structural declarations, unchanged raw-zero-indicator unresolved behavior, malformed/complement-inconsistent traces and insufficient batch lengths.
- Analytic noncancelling Brier/NLL deviation identities in the near-one regime.

The initial run had8 passing checks and1 failed negative fixture. That fixture added.1 to a log probability of order1e-20 and incorrectly expected it to violate an **absolute1e-12** complementary-normalization bound. It does not. The initial fixture/source and failure log remain in `attempts/`; the corrected negative test perturbs the dominant side by−.1, which does violate the declared bound. The mathematical method/threshold was not relaxed. A pre-seal reviewer also requested propagating the existing explicit structural-constant flag for runner API compatibility; that field is now retained and tested.

Original one-sided diagnostics remain valid for their original coordinate; this helper adds the exact paired coordinate selected by Root without editing that historical implementation. Historical bank approximation, parameter-scale uncertainty and electoral generalization remain separate. Source/runtime/runner binding and independent acceptance are still required before actual execution.
