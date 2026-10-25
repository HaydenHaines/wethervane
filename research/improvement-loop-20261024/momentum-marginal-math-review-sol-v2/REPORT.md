# Independent bounded delta review of Gaussian marginal math v2

2026-10-25. **Accept v2 for its narrow supplied-Gaussian-mixture arithmetic scope.** Reviewed author manifest `2bfa468bca013c7bfa7d6e80395ba7534947026ff35209fd2d51fed1b40f28c8`; all bound author bytes match. V1 remains preserved and requires correction under the [v1 review](../momentum-marginal-math-review-sol-v1/REPORT.md), review-manifest SHA `7c462f9ac104a47efd2d07b82c2c2196d27d5f1c00e2bb7d7d8538cc237a007e`.

No actual data, numeric target labels, forecast scores, Gibbs chain, optimizer or fitted bank was used in this review. Inputs are explicit deterministic numerical fixtures. Source and tests remain unchanged in the author directory.

## Delta and regression findings

The source diff is restricted to the accepted numerical repairs and their validation boundaries: authoritative normalized raw logweights; separate ordinary linear weights; log-tail complement normalization only when its drift is at most1e-12; log-density ownership by logweights; and finite ordered integration bounds/error-tolerance/zero-width integrand checks. No statistical model, prior, quadrature budget, WIS convention or chain-phase contract changes.

All three v1 regressions are repaired:

- Raw weights[1e-300,1e300], means[-100,100], sd[1,1] retain the rare component in log-tail arithmetic. The log-losing probability agrees with the independently normalized raw-logweight oracle within2e-12, even though the separate linear weight underflows to0.
- Four identical N(50,1) components with weights[1,2,3,4] now return complementary nonpositive log tails and pass their own unclipped winner-score validation. The tiny losing tail remains finite; no statistical probability floor is introduced.
- Scalar and vector nonfinite zero-width integrands raise ArithmeticError. Valid zero-width vector integrals return typed zeros with zero estimated error. The existing bounded two-attempt retry/failure rules still pass injected warning, success-flag, disagreement and nonfinite tests.

An additional independent density regression checks means[-700,100], sd[1,1], raw weights[1e-200,1e200], x=expit(-700). Here the first ordinary normalized weight is0 but its transformed-share density contribution is representable and dominant. V2 density agrees with the authoritative raw-logweight density oracle within2e-12 relative error. This guards the density fix independently from the rare-winner-tail fix.

## Independent verification

`test_independent_delta.py` reuses the v1 review's supplied mathematical checks with the three defect assertions replaced by corrected regression requirements and adds the density regression. All **14 tests pass in0.937s** (`test-results-v1.txt`) using the existing lab runtime, NumPy2.4.4/SciPy1.17.1, bytecode disabled and one BLAS/OMP thread.

Retained positive checks cover direct Normal-mixture CDF/survival/density; analytic single-component quantiles and independent mixture brackets; normal-domain expected shares and component-weight identity; fixed Gauss–Legendre CRPS at interior/endpoints; centered CRPS directional derivatives and a y=0 upper-integral-only oracle; quantile/WIS weight derivatives and seven-pinball WIS identity; Mills expansion extreme-tail check; winner-score formulas; copied immutable inputs; and bounded scalar/vector quadrature failures. The review's ownership fixture now tolerates last-bit linear-weight changes from the intentional logweight normalization, without weakening the immutability checks or tail regression requirements.

The original Sol proposal's upper CRPS influence `−2*SF*(F_t−F)` was already correct and equals the implemented `+2*SF*(SF_t−SF)`. Author v1's contrary verification claim and my initial acknowledgment were mistaken. The explicit v2 record correction accurately states this; the original proposal and mistaken historical record remain unchanged. This is a record correction, not an additional mathematical defect or a change to the helper's already-correct CRPS influence.

## Scope of acceptance

This is an arithmetic helper for supplied **Gaussian conditional logit components**, finite share-space integration, quantiles, density, proper marginal scores and deterministic influence functions. It does not own chain phase, source identity, source admission, target numeric labels, forecast gates, posterior convergence, bank approximation, scale integration, missing-race Student-t marginals or promotion. Quadrature error estimates are not Monte Carlo errors or proofs. The numerical normalization tolerance corrects floating-point complement drift; it does not establish tail integration precision across a Gibbs chain.

No global historical inference/scoring contract or real forecast is accepted by this review. The existing supplied-only implementation needs no new inference framework; additional empirical qualification remains a separate task.
