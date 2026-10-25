M01 calibration posterior bank, prepared for review
=================================================

Status at October 25, 2026 04:00 UTC: root launched `attempt-v1` after
supplied-point tests and independent prereview. Its original tool handle is
66137; actual chain-0 warmup draws were observed. This is a dated observation,
not proof of current liveness or a completed posterior. Read the terminal
receipt, or poll the original handle, before deciding the next action.
The nine unit tests include one mocked sampler failure, which generates synthetic
temporary output and deletes it after checking preservation behavior.

This defines a new calibration-only posterior on five log standard deviations,
ordered independent, cycle, race, house, movement. Each log SD has an independent,
untruncated Normal prior with mean log([0.1, 0.06, 0.06, 0.03, 0.05]) and SD 0.7.
The likelihood is the exact normalized multivariate Gaussian residual likelihood
implemented by the copied, unchanged owner `gaussian_error_objective`. Its existing
quadratic term is subtracted from the custom likelihood Op and supplied once by
an explicit PyMC Normal prior. The complete target includes all Gaussian prior
normalizing constants. No optimizer bound is used as prior support.

Frozen calibration data contain 1,483 rows and 105 races: 467 observations from
2012, 463 from 2016, 288 from 2018, and 265 from 2020. The original TX 2020/2024
event exclusions remain mandatory. The declared target is 2026, with the original
2026-10-20 source cutoff. All 1,483 availability dates are field-end proxies;
publication dates and historical versions remain unverified. This experiment
makes no time-safe historical scoring claim. Event disjointness does not establish
independence from other states' same-cycle downstream observations.

`prepared-v1/plan.json` binds copied inputs, exact source modules and their import
dependencies, code snapshots, and the isolated environment freeze. Earlier
`review-fixture-*` directories preserve preparation revisions used while testing;
they contain no sampled posterior banks. `supplied-point-check.json` records the
actual-data compiled PyMC density and gradient check at the original optimizer
mode and a nonzero offset. Tests also independently reconstruct normalized tiny
Gaussian densities, check finite differences, reproduce the original objective,
and exercise source, date, and event guards.

The proposed run uses four sequential one-chain PyMC NUTS calls, each with 1,000
warmup and 1,000 retained draws, one core, target acceptance 0.9, and maximum tree
depth 10. Explicit chain seeds are in the plan. Every transition is flushed to a
journal; each completed chain is immediately preserved as NetCDF including warmup
and sample statistics. The four chains are then concatenated for joint diagnostics.
Every log scale must satisfy rank Rhat <= 1.01, bulk and tail ESS >= 400, with zero
divergences and zero depth saturations. Every chain must have EBFMI >= 0.3. Failure
preserves available raw data and writes a failure receipt; there is no automatic
retry or scientific activation.

The isolated executable is:

```
/home/hayden/projects/ballotmancy-improvement-lab/.improvement-ledger/runtime/calibration-v1-env/bin/python
```

From the lab root, verification only is:

```bash
.improvement-ledger/runtime/calibration-v1-env/bin/python \
  research/improvement-loop-20261024/calibration-bank-v1/calibration_bank.py
.improvement-ledger/runtime/calibration-v1-env/bin/python -m unittest discover \
  -s research/improvement-loop-20261024/calibration-bank-v1 -p test_calibration_bank.py -v
```

The authorized launch command was:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
.improvement-ledger/runtime/calibration-v1-env/bin/python -u \
  research/improvement-loop-20261024/calibration-bank-v1/calibration_bank.py \
  --prepared research/improvement-loop-20261024/calibration-bank-v1/prepared-v1 \
  --run research/improvement-loop-20261024/calibration-bank-v1/attempt-v1
```

The output directory must not exist. The empirical calibration bank chooses
neither marginal Bayes feedback nor calibration-cut feedback. No population fit,
count fit, historical score, forecast qualification, publication, or promotion is
authorized by producing this bank, even if its calibration diagnostics pass.

Do not rerun this command into a new attempt merely because a status observation
times out. Poll handle 66137 and inspect the recorded PID/process and terminal
files. A failed attempt requires a diagnosed reason and a versioned successor;
a completed run requires independent saved-bank review before downstream use.
