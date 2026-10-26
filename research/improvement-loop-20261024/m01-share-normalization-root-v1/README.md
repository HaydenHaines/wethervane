# Six-poll mixture: share-scale integration check

This is a numerical test of a supplied-state likelihood, not a forecast. The earlier `m01-native-factor-ownership-sol-v1` whitened normal integral was an algebraic identity and did not exercise a different distribution. Here the existing, source-pinned `SharedVectorMixture` supplies four specific covariance components from the actual six-poll TX2026 coordinates. Their weights are **invented** for this test; they are not the full bank's weights. The supplied common log-odds mean is also illustrative.

For eight scrambled Sobol sequences of 16,384 points each, the test samples a single logistic-normal proposal whose logit covariance is 1.5 times the weighted component covariance. It evaluates the target four-component mixture and proposal as densities on the six-dimensional `(0,1)` poll-share cube using each distribution's logit Jacobian. The average importance ratio estimates the integral. Because the target is a covariance mixture and the proposal a single different Gaussian, their ratio is not identically one. Scalar prototype evaluations at 24 selected points are checked separately against a SciPy Gaussian-mixture calculation; the vectorized integrand is a reconstruction, not the symbolic native implementation. A negative control drops the target's share-scale Jacobian, which yields an integral far from one. The Jacobians cancel algebraically in the **correct** importance ratio; the negative control checks that declaring a logit density as a share density is not equivalent.

The observed mean integral is approximately 1.00009 across the eight scrambles; their standard deviation is about 0.00013. These are randomized quasi-Monte Carlo approximations, not a proof of exact normalization or an empirical calibration result. The acceptance threshold was fixed at a 0.02 absolute deviation before the initial run. This result does not cover the full 4,000-component calibration mixture, source/sample dependence, the PyTensor/native provider, a fitted posterior, publication readiness, or historical predictive skill. Individual share-density conversion is meaningful only for an explicitly modeled continuous share; it does not transform published rounded poll percentages into unrounded respondent data.

Reproduce using the existing isolated environment and one BLAS thread:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .improvement-ledger/runtime/calibration-v1-env/bin/python \
  research/improvement-loop-20261024/m01-share-normalization-root-v1/check.py
```

The loader verifies the pre-existing consumer manifest, source snapshots and bank review before selecting components. No owner tree, current model, registry or active fit is changed.
