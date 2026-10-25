# M01 calibration bank prereview

**Review scope:** static math/runtime review of `calibration-bank-v1`; no MCMC, no outcome fit, no forecast qualification, and no edits to the bank implementation. The prepared plan says `prepared_not_launched`. The bank alone does not decide marginal Bayes versus calibration-cut feedback.

## Verdict

I found **no launch-blocking math or source-isolation defect** in the reviewed version. The target is the explicitly declared posterior for five log error SDs under the existing Gaussian residual blocks and the new proper prior. The latest verified source hashes are listed below. This is a code/plan prereview, not a claim that the sampler completed or its posterior is empirically useful.

### Density and coordinates

- The independent prior is `Normal(log(MEDIANS), 0.7)` directly on the five `log_scales`, untruncated. That gives the named SD values as prior medians. The normalizing constant is included once. Because the sampled variable is already log SD, there is no change-of-variables Jacobian to add.
- The frozen owner objective is a Gaussian negative log likelihood plus the quadratic log-SD penalty. `CalibrationTarget.value_gradient` negates that objective and adds the five Normal constants, yielding Gaussian likelihood × normalized prior. `likelihood_value_gradient` subtracts the normalized prior to supply only the likelihood to the PyMC `Potential`; PyMC contributes the prior once. `build_model` uses the log-coordinate Normal and `exp(log_scales)` as a deterministic view.
- No optimizer box bounds are reused as support. The previous optimized values only center dispersed initialization; the posterior is not truncated at those optimizer limits.
- The covariance blocks remain source-defined and separated by cycle: independent, cycle, race, pollster-within-cycle, and Brownian movement kernels. The likelihood uses the existing logit poll-minus-result residuals. Four cycle blocks and the fixed 1,483 selected rows are kept; excluded TX event keys are guarded.

### Source and time guards

- `load_prepared` verifies copied source bytes against pinned SHA-256 values, checks required owner-source pins and all three code snapshots, validates target 2026 / the frozen cutoff / cycle counts / race count, and rejects any source-row flag that upgrades the field-end-date proxy to verified publication or strict as-of status.
- `prepare.py` copies exact owner inputs and source modules into a unique prepared bundle and freezes the runtime package list. Runtime import resolution is checked against those copied modules. No original owner module is monkey-patched or refit.
- Plan limitations properly retain residual dependence across states within the same cycle, acknowledge retrospective outcome use through 2020, and leave both downstream feedback constructions unspecified.

### Runtime and failure behavior

- The plan specifies four sequential one-chain PyMC NUTS runs, each with 1,000 tuning and 1,000 posterior draws, explicit seeds, `target_accept=0.9`, and a fixed tree-depth cap. Sequential chains avoid four concurrent dense factorizations. On success, each chain is saved separately, then combined without dropping warmup/stats. Diagnostics fail closed on nonfinite values, R-hat, bulk/tail ESS, E-BFMI, divergences, or tree-depth saturation.
- Before sampling, the executable compiles and checks the actual PyMC log density and gradient against the independent supplied-point objective/gradient calculation. The nine supplied tests cover the math, time/input guards, snapshot guards, diagnostics, and a **mocked** second-chain failure; the failure test does not actually sample.
- Partial data handling is visible in the code: completed chains are persisted to per-chain NetCDF before starting the next; a draw callback flushes each transition record so an interrupted current chain leaves a journal. That journal contains `log_scales` plus `repr(stats)`, not a structured/resumable full PyMC chain bank. Treat it as failure evidence, not as a posterior bank; no automatic retry is configured. This is a declared operational limitation rather than a defect in the posterior target.

## Verification boundary

I did not run the tests, benchmark dense-matrix throughput, or launch MCMC. The source is statically consistent with the proposed posterior and with the frozen runtime plan. A successful launch would still require Sol/root to complete their 9-test review, inspect actual supplied-point receipts and source digests, and authorize the explicit `--run` path. No downstream forecast comparison, full-count fit, or scientific qualification follows from this bank alone.

## Reviewed hashes

- `calibration_bank.py`: `6db4eb64862f44a3b53f15206773ff448cef5b8fd6c9a9244ded98cfa92d1c3c`
- `prepare.py`: `9ea8dbb1d5279a07101dc94e0a79319e9caac903f79ede780742d0a1a9781f91`
- `test_calibration_bank.py`: `7096cf93ad76957a78a0cb0c0f8b7ba0fdad7eb8274d6b54d95ddb1978a6a3d3`
- pinned owner `poll_covariance_v1.py`: `9ae130b3f314d75a5255c7c92fab502ec596a7cca2960090706fa203b133e05d`
- pinned owner `polling_v1.py`: `5876ab790f43c8bd6324590f2a3d007f2da853bce49d0e60cd19142046ab0bcb`
- review bundle plan: `review-fixture-v1/plan.json`, status `prepared_not_launched`; package freeze hash `420aeebe5a07ece8e2d47a92e2b27aab5dd4b08dcf72d65234693c254c754d4`

## Final prepared-bank delta (environment inventory repair)

The final `prepared-v1` tree postdates the review fixture. Diffing current files against `review-fixture-v1/code/` shows only an environment-inventory repair:

- `calibration_bank.py` now defines `environment_packages()`, collecting installed distributions only from `sysconfig`'s `purelib` and `platlib`. `run()` uses the same inventory after importing PyMC. This avoids PyMC imports adding a vendor path to `importlib.metadata.distributions()` and falsely changing the frozen package list. No density, prior, source selection, kernel, sampler, or diagnostic logic changed.
- `prepare.py` uses that shared helper when creating the freeze, removing its duplicate unconstrained distribution enumeration. No other delta from the frozen review-fixture code was present.
- The final prepared plan pins those updated source bytes. Root reports all 9 tests passed in 6.8 seconds and supplied-point compiled log-density/gradient verification passed with maximum gradient error `1.5e-16`; I did not rerun them. No MCMC has been launched.

Final hashes (supersede the earlier review-fixture code hashes above):

- final `prepared-v1/plan.json`: `9ffc1342bc0afad4c13ae3be73d3e6bfbd10cb8e88fecc178f76e40f6c1590be`
- `calibration_bank.py`: `5a52d811eb6e5620fedc1757bec27141bc9c50008c3c1457a375e09dd5a3e3fe`
- `prepare.py`: `d6d5a4a59f3da1523e8bd826e9633aff8abb7503d7da775083580fa4a661466d`
- `test_calibration_bank.py` (unchanged): `7096cf93ad76957a78a0cb0c0f8b7ba0fdad7eb8274d6b54d95ddb1978a6a3d3`
- package freeze: `420aeebe5a07ece8e2d47a92e2b27aab5dd4b08dcf72d65234693c254c754d4`
