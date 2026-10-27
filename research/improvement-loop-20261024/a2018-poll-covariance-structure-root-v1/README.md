# 2018 prepared polling covariance: cross-contest structure

This aggregate-only check reads the same exact, hash-pinned unadmitted A2018 prepared problem as `../a2018-poll-covariance-root-v1/`. The single 274-row polling covariance spans 29 distinct contest IDs. For the 37,401 unordered off-diagonal pairs, it counts whether two rows share a contest and summarizes their standardized covariances. It emits no contest IDs, poll IDs, row values or matrix entries.

In the saved matrix, all 2,110 same-contest pairs have positive standardized covariance, ranging approximately 0.166 to 0.894. All 35,291 different-contest pairs also have positive standardized covariance, ranging approximately 0.0189 to 0.400. Exact zero and negative entries were absent from both groups. These values describe **the supplied likelihood covariance**, not estimated posterior correlation between election outcomes.

The selected provider source constructs one multivariate normal for the block, using its listed row order and covariance. Its source-level intent therefore accounts for cross-contest dependence in this one polling factor. This does not prove the matrix models real sample overlap, that poll errors have these correlations, that an executed graph used the pinned bytes, or that the roster, historical availability, fit or forecast is qualified. A nonzero covariance does not identify which samples overlap or justify treating any separately sourced survey as independent. Positive definiteness was examined separately in the prior numerical packet; this structural helper is not itself an SPD check.

For a compatible NumPy environment, run from this directory:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B -m unittest -v test_check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B check.py
```

The fixture tests check signs, zeros, dimension/symmetry/range errors and source-byte drift. `inspect` is an unauthenticated fixture helper; `frozen_check` verifies the problem bytes before parsing. Independent review is pending. No model was built, fitted or changed; no posterior draws or individual records were exported.
