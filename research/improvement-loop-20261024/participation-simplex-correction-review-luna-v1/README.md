# Independent review: normalized participation scenario mixture

The candidate is a focused correction to the predecessor's near-unit scenario-weight handling. It normalizes only the internal scenario mixture, retains the predecessor's tolerance gate, and leaves the protected population-parent contract explicitly outside scope. The source diff is confined to `mixture_moments` and its docstring. The inherited and new 12-test suite passes in an isolated copy.

The split-invariance and shared-covariance checks also pass in independent invented probes. However, an accepted near-unit weight vector causes the anchored mean formula to return an impossible negative expected count. At population `N=2**53`, the function returns `-2.0` where the exact rational mixture of the supplied binary weights is `0.00860691137811289`; at `N=1,000,000`, it returns `-2.33e-10`. The normalized binary weights sum to `1.0000000000000002`, so the centered formula's algebraic identity is not exact in floating arithmetic. The reproducer and machine-readable output are preserved here.

This is a numerical correctness blocker within the candidate's declared supported input region. It does not show a defect in a fitted model or real forecast. It does not justify changing protected outer parent masses. The candidate remains an isolated, unadmitted alternative; no raw data, model run, owner artifact, or forecast was touched.

Run the two test modules from an isolated copy with:

```sh
PYTHONDONTWRITEBYTECODE=1 nice -n 15 python3 -B -m unittest test_participation_extrapolation_prototype test_normalized_simplex -v
```

Reproduce the counterexample with:

```sh
PYTHONDONTWRITEBYTECODE=1 nice -n 15 python3 -B rare-anchor_counterexample.py
```

See `review.json` for verdict and exact outputs; `source-pins.json` binds candidate, predecessor, prior review, and independent artifacts. `luna_probe.py` covers split invariance, tested covariance signs/PSD, and tolerance rejection. `tests.log` is the captured test output.
