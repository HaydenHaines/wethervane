# M01 actual-bank consumer: supplied-mean marginal likelihood check

This directory consumes the **completed, independently reviewed** M01 calibration bank without rerunning its sampler. It evaluates the existing normalized Gaussian scale-mixture primitive on the actual six-poll TX2026 vector using the unchanged copied owner covariance provider.

The explicitly selected check target is `M01_marginal_bayes_complete_poll_vector_likelihood_v1`. This is an ordinary **marginal-Bayes likelihood at supplied means**, not a calibration cut, downstream posterior fit, population-cut engine, forecast, active provider, qualification or publication. Conditioning a downstream model on this marginal likelihood would generally update the calibration-component weights. Equal input weights do not promise protected calibration mass after downstream normalization. The caller still owns source/calibration reuse and complete within-population joint normalization.

## Frozen inputs and loader gates

`input-manifest.json` records exact source paths, SHA-256 values and copied bytes. `inputs/` preserves the independent review, terminal receipt, plan and joint NPZ plus actual owner coordinates/component/calibration/receipt. `snapshot-source/` contains unchanged provider/dependency files from the owner's frozen preparation. `poll_scale_mixture.py` is an unchanged copy of the previously tested primitive. No covariance kernels, priors or sampling terms are reimplemented.

The trusted independent review SHA required by the recorded invocation is:

`3a07125bed8944a7c9582a9c50d484cec154d1a3f76fa9c716eea70bf9d43b54`

The review binds plan `9ffc1342bc0afad4c13ae3be73d3e6bfbd10cb8e88fecc178f76e40f6c1590be`, terminal `45fb5f20ecaf4934cfdc1d2558db9aa6fd54923211a8637858cd65d2aac7c0c3` and NPZ `449b9f50eecec0af250834457f256a8fd193616cb70b4847bd35acf614fe0539`. `load_bank` checks those bytes, passed independent/terminal numerical gates, exact joint axes/components, warmup exclusion, finite positive SDs and the exact log-to-SD transform. Raw NetCDF/journal reconciliation is inherited from the pinned independent review; this consumer does not redo that audit or claim to have resampled chains.

The loader requires explicit target cycle **2026**, an explicit UTC forecast cutoff not earlier than source cutoff **2026-10-20T16:21:44Z**, and the exact marginal-likelihood target label. Historical reuse for 2018/2020/2022/2024 is rejected, including same-election information that would be unavailable in those replays. Other target cycles and a calibration-cut label are rejected. The recorded supplied-mean check uses forecast cutoff **2026-10-20T22:13:33Z**. This source cutoff is not the later sampler completion time; its field-end/publication-vintage limitations remain those in the reviewed calibration plan. The bank has no forecast/publication qualification.

Every **4 × 1000 × 5** retained row remains joint. Row `chain*1000 + draw` contains the original five coordinates in order independent/cycle/race/house/movement. All 4,000 rows receive weight **1/4000**. Neither five independent marginal banks, resampled rows, selected chains, nor best-chain weights are constructed. Bank arrays and weights are copied onto immutable byte buffers. The current-only mixture also rejects historical or non-TX coordinates.

The actual owner receipt binds the six observations and their ordering, the calibration source and the saved covariance. The copied provider's `error_covariance` constructs each full 6×6 matrix from one joint five-scale row. The independent component already includes sampling, weighting, rounding and residual error: **no extra MOE or sampling variance is added**.

## Actual numerical results

Three supplied six-vector means were used: the six observed logits themselves; six zeros; and a common mean of 0.1 in every coordinate. These are computational test points, not inferred latent means or an estimate of election outcome.

| Supplied mean | Full-bank log density | Fixed-scale log density | Four-chain log-density range |
| --- | ---: | ---: | ---: |
| Observed logits | 6.380227438964 | 6.407460198599 | 0.014042662023 |
| Zeros | 6.115755018834 | 6.144370834362 | 0.013094669318 |
| Common 0.1 | 6.026771221754 | 6.053503464847 | 0.012966501374 |

The maximum full-bank density-log difference from independent SciPy component log densities plus `logsumexp` is **8.88e-16**; maximum gradient difference is **2.66e-15**. The independent gradient uses covariance solves and separately calculated component responsibilities. The point bank reproduces the owner's saved fixed covariance to absolute tolerance 1e-15; at the three points its log-density difference is zero and its maximum mean-gradient difference is **1.33e-15**. Log densities are continuous-density quantities, not winning probabilities.

Per-chain integration uses every retained row of each chain with equal within-chain weights **for diagnostic sensitivity only**. The reported primary integral always uses all 4,000 joint rows. Ranges around 0.013–0.014 log-density units show finite-bank integration variability; they are not a complete MCSE, an empirical forecast uncertainty interval or a license to choose a favorable chain. Calibration-scale convergence does not alone qualify every nonlinear downstream functional.

## Joint scale summaries

| Scale | 5th percentile | Median | 95th percentile |
| --- | ---: | ---: | ---: |
| Independent | 0.081181 | 0.084933 | 0.088897 |
| Cycle | 0.041622 | 0.070730 | 0.134434 |
| Race | 0.044254 | 0.063940 | 0.083914 |
| House/pollster | 0.060645 | 0.071489 | 0.082726 |
| Movement | 0.070061 | 0.081998 | 0.094967 |

The first four are SDs in principal-pair log-odds units; movement uses the provider's square-root-of-30-day units. Both SD and log-SD correlation matrices are retained in `report.json`. Notable log-SD correlations are race/movement **−0.3910**, independent/house **−0.3561**, and independent/movement **−0.2989**. These actual dependencies make preserving whole rows consequential; summaries do not license factorization.

## Artifacts, reproduction and tests

- `consumer.py`: read-only gate/loader, provider construction, independent numerical comparison and report command.
- `test_consumer.py`: actual-bank axes/row identity, immutable storage, equal weights and joint dependence; wrong review digest; failed independent and terminal gates; tampered plan/terminal/NPZ; forbidden historical/earlier-cutoff/target reuse; exact owner covariance without added noise; and output receipt integrity.
- `report.json`: all supplied mean vectors, densities, gradients, independent errors, four per-chain integrals, five-scale quantiles and correlation matrices.
- `receipt.json`: review/target binding, runtime versions and code/input/report hashes. `input-manifest.json` binds the exact copied originals. `check-output.json` summarizes the three numerical checks.

Use the isolated frozen calibration environment (NumPy/SciPy only are needed for this consumer; no PyMC sampler is invoked). From this directory:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  /home/hayden/projects/ballotmancy-improvement-lab/.improvement-ledger/runtime/calibration-v1-env/bin/python consumer.py \
  --expected-review-sha256 3a07125bed8944a7c9582a9c50d484cec154d1a3f76fa9c716eea70bf9d43b54 \
  --target-label M01_marginal_bayes_complete_poll_vector_likelihood_v1 \
  --target-cycle 2026 --forecast-cutoff-utc 2026-10-20T22:13:33Z
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  /home/hayden/projects/ballotmancy-improvement-lab/.improvement-ledger/runtime/calibration-v1-env/bin/python \
  -m unittest -v test_consumer.py
```

The CLI requires the review SHA, target label, cycle and cutoff; it does not silently choose a calibration-feedback target. All writes are local reports in this new consumer directory; the owner source, frozen fit and independent review remain unchanged. No MCMC, historical replay, downstream parameter refit, active-provider registration, public forecast or commit was performed. Any later inference/empirical comparison needs a separately versioned target and jointly refitted matched model; these supplied-point checks establish only bounded numerical compatibility and bank consumption.

## Independent-review repair

Luna's independent review identified a manifest trust boundary. `verify_snapshots` now checks the exact reviewed manifest SHA `0fbec6366c2640f30a17c2fc576b87303d9f4065a632b1488b9a72311345fcb8` **before** reading its provider/input pins or importing copied owner code. A new test changes the provider snapshot and updates its manifest hash entry together; the reviewed manifest check rejects it before import/use. All eight tests pass. `manifest-repair-before.json` and `manifest-repair.json` preserve previous/current receipt and code hashes. The numerical report SHA remains `e7b8f0285feb0987b6bb7b7d40917aa3037bfa416a6b8829bcb9efa02ff59306`; supplied values and copied source inputs are unchanged.
