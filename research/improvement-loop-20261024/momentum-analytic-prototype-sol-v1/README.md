This is a **synthetic, supplied-coordinate mathematical prototype**. It evaluates a six-scale covariance/objective and analytic Gaussian toy conditioning. It exposes no optimizer or sampler, reads no poll observations or outcomes, scores no forecast, and changes no owner source. It does not implement field-interval averaging, unknown-date mixtures, prior-bank updating or a calibration cut.

The root selection is `../checkpoints/momentum-analytic-root-selection-v1.json`, SHA `0b28ea26002ba2b46ad4862a4028e285a6a0de2dcfff03830376ed4ffdfc3a4f`. `source_guard.py` authenticates that selection and all five selected source pins before NumPy/SciPy/owner imports, plus the transitive `response_v5_reuse.py` and empty `src/__init__.py`. It verifies immutable snapshots against those exact owner hashes. `source-bindings.json` SHA is `c3a38476aee69845065657595e89a68f664751e94d5b1d1aafc548f22621e1d3`. The owner module defines optimizer code; importing its mathematical primitives does not invoke that optimizer. Tests explicitly prohibit optimizer and RNG calls in the prototype's execution path.

`momentum.PollCoordinatesV1` owns unique observation IDs, exact race/cycle/pollster labels and finite nonnegative days before election. Its first five kernels are obtained unchanged from the pinned incumbent: independent, cycle, race, house and Brownian movement. The only new kernel is

\[
K^{slope}_{ij}=1(r_i=r_j,c_i=c_j)(d_i/30)(d_j/30).
\]

The six supplied SDs produce `Sigma = sum_k sigma_k² K_k`. The independent SD must be positive; all other SDs may be zero in the covariance API. Positive independent variance and PSD kernels imply SPD in exact arithmetic; a numerical Cholesky failure rejects with no jitter or fallback. No nominal sample-size/MOE term is added. Native `GaussianPollLikelihood` is returned by `build_poll_likelihood`, with the native validator unchanged. This mathematical compatibility does not admit a real bank or observation source.

The slope is one independent zero-centered Gaussian latent variable per race **and cycle**, shared by every observation in that group. Its SD is in logit units per 30 days **before** election. Positive latent slope makes earlier polls higher than election day, so negative slope represents a rise toward election day. Both directions have equal prior status. Brownian movement retains logit SD per square-root 30-day units, and its independent increments are preserved. The new slope vanishes on election day and creates no cross-race or cross-cycle covariance. Existing cycle and house covariance across races remains intact.

`ResidualBlockV1` supplies residuals rather than reading or fitting a mean. Independent objective blocks must contain disjoint cycles, preventing accidental omission of shared cycle/house effects. For all six finite log-SDs `ell`, the objective is

\[
\sum_b\tfrac12[\log|\Sigma_b|+r_b^T\Sigma_b^{-1}r_b+n_b\log(2\pi)]
+\tfrac12\sum_{k=1}^6[(\ell_k-\log m_k)/s]^2.
\]

Its analytic derivative is `sigma_k² trace[(Sigma^-1-alpha alpha^T) K_k] + (ell_k-log m_k)/s²`, summed over blocks, with `alpha=Sigma^-1 r`. Every incumbent scale and the new slope scale participates jointly. This is exactly the incumbent penalized log-SD-coordinate convention, extended to six coordinates. The Gaussian residual normalization is retained. The fixed Normal penalty normalizer is omitted exactly as in the incumbent; this is not a normalized marginal-evidence calculation. No natural-SD lognormal Jacobian or optimizer box is inserted. Positive median vectors and `s=.7` are supplied explicitly in the example; synthetic medians are not estimates from real elections.

Zero slope variance recovers the exact incumbent covariance and Gaussian likelihood, including observation/race mapping. Exact zero is available in `error_covariance`, while a finite log-SD objective cannot contain it; pushing log slope to −infinity also makes its log-scale penalty diverge. No claim identifies that covariance boundary with an attainable penalized MAP point.

`supplied-point-results.json` reports two declared synthetic six-observation points. Maximum absolute analytic-versus-central-difference gradient errors are `9.71e-10` and `1.18e-9`. Both covariance matrices are SPD; minimum eigenvalues are approximately .01106/.01648 and condition numbers 3.50/2.70. The illustrative kernel design has rank 6 but Brownian/slope Frobenius cosine .963: this is a design-geometry warning, not a fitted parameter correlation or evidence that the mechanisms are identified in actual data.

The clearly labeled conditional example is a Gaussian toy: `y_i = theta + beta*d_i/30 + five-term error`, `theta~N(0,.35²)`, `beta~N(0,slope_sd²)`. It preserves all five error terms; the slope appears in the latent prior and is not counted again in the residual covariance. Synthetic logits −.30/−.15/−.04 at days 60/30/7 imply posterior mean slope −.031118 and election logit −.112644 when supplied slope SD = .05. With SD = .005 the slope shrinks to −.000393 and election logit is −.141301. Reversing every observation's sign exactly reverses both posterior means; flat zero data yield zero means. These are conditional Gaussian logit moments, not forecasts under the production empirical/Student-t fundamentals law or expected shares. No candidate receives an automatic momentum bonus.

The deterministic test oracles include independent latent intercept/Brownian-increment/slope construction, two-dimensional Gauss-Hermite integration of the latent slopes, all-six finite differences at two supplied points, exact five-kernel/zero-slope/native-factor identity, observation permutation, independent cycle-block equivalence, covariance PSD/SPD, wrong-ID/share/shape/scale/age failures, immutable owned arrays and source-corruption rejection **before scientific imports**. The latent quadrature checks the normalized Gaussian likelihood without using the producer's slope kernel or covariance routine. No interval approximation or actual-source numerical result is included.

From the lab root:

```sh
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s research/improvement-loop-20261024/momentum-analytic-prototype-sol-v1 -p test_momentum.py -v
.venv/bin/ruff check research/improvement-loop-20261024/momentum-analytic-prototype-sol-v1/*.py
.venv/bin/ruff format --check research/improvement-loop-20261024/momentum-analytic-prototype-sol-v1/*.py
```

`check_supplied.py --output <new-file-inside-this-package>` reproduces the supplied mathematical checks without fitting; exclusive writes reject an existing path. The saved artifact and final manifest preserve the reviewed version. Real earlier-only calibration, source/coverage admission, bank target choice, a matched runner, numerical recovery policy and actual forecasting/scoring require separate root selection and independent review.
