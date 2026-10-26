# Frozen supplied-state compression result

See [FINDINGS.md](FINDINGS.md) for the identity, exact scope and next comparison step. `evidence.json` pins all source, results and `coefficients.npz`; the coefficient artifact is also pinned in `result.json`.

The actual 4,000-component six-poll bank passes 16 supplied means, full multivariate/component-responsibility comparisons, an independent SciPy oracle, a scalar-only symbolic derivative check and seven negatives. No full native graph, fit or forecast was run. Reported 12.3-fold reduction concerns direct constant arrays only, not measured full-model performance.

Independent review supports this supplied-state scope. The next separate provider packet must bind/test the exact native six-rate-to-mean mapping and constrain covariance symmetry. The generic helper's 1e-12 asymmetry tolerance is not a qualification for arbitrary nearly symmetric inputs; the reviewed actual bank is symmetric. This frozen source/results packet is unchanged by those prospective integration requirements.
