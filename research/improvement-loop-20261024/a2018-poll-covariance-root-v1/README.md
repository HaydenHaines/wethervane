# 2018 prepared polling covariance: numerical check

This small, source-pinned check examines the single 274-by-274 polling covariance in the frozen, unadmitted 2018 prepared problem. It does not import or execute the forecast model, read predictive draws, or change a model job. It emits only matrix-level aggregates, a covariance fingerprint, software versions and the prepared-problem fingerprint; no poll row identifiers or matrix entries are exported.

In the recorded NumPy 2.4.4 float64 run, the matrix was exactly symmetric as represented, all entries finite, and all diagonal entries positive. Its smallest and largest computed eigenvalues were approximately 0.00835 and 1.11215; the eigenvalue ratio was approximately 133.13. The smallest eigenvalue was well above the reported scale-based floating-point margin. Cholesky factorization succeeded with a relative Frobenius reconstruction residual of approximately 1.6e-16. These are numerical diagnostics for these exact bytes, not a formal exact-arithmetic proof.

The positive-definiteness result does **not** establish that the covariance represents actual poll dependencies, that the underlying polls or historical source vintages are admissible, that the executed model used this matrix and ordered vector, or that any fit or forecast is calibrated or qualified. The earlier `a2018-source-evidence-resolver-sol-v1` separately describes source and prepared-vector bindings; the compiled factor graph and raw observation/sample identities remain unauthenticated. Correlation between evidence sources remains unknown. No conclusion about which model layer is weakest follows.

To reproduce with a compatible NumPy installation, from this directory run:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B -m unittest -v test_check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B check.py
```

The test suite includes invented positive-definite, indefinite, malformed and numerically near-singular matrices and rejects an altered prepared-source file. `inspect` alone accepts an unpinned caller-supplied fixture and is not an admission method; `frozen_check` verifies the exact source length and SHA-256 before decoding. Independent review and cross-version numerical reproduction are pending.
