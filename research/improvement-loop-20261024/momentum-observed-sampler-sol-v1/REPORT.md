# Private observed sampler: author verification complete, independent review pending

2026-10-25. Root separately authorized **synthetic small-chain implementation verification only** after the design-only contract. The new lab module and its observer satisfy 18 bounded author tests. The result is a narrow instrumentation/parity finding, not a forecast, convergence result, accepted global scoring contract, or publication qualification.

No historical target data, numeric labels, forecast scores, optimizer, source bank fit, original owner edit, or old process control was performed. The original calibration sessions remain untouched. All synthetic numeric fixtures are constructed algebraically in `test_observed.py`; they do not load real race/poll/parameter files. The synthetic 16,000-row smoke is not a historical forecast.

## Exact preservation and observer phases

`observed_sampler.py` is generated from the byte-pinned original in `snapshots/src/forecasting/polling_basis_v1.py`. `build_observed.py` asserts the original SHA and exact replacement-anchor counts. Original `conditional_state`, Gaussian draw, categorical draw, Gamma draw, missing-race Student-t draws, and return expressions remain intact. Extra deterministic calculations do not feed back into any transition. Existing calls preserve defaults and every original return key/value/type; the only signature extension is keyword-only `observer=None`. No observer field is silently added to the forecast result.

A callable receives deep immutable copied mappings at retained phases:

- `gaussian_conditional`: incoming row index, incoming lambda, joint observed-theta/cycle mean and marginal variances; saved immediately before the original Gaussian RNG call. Variances are column squared norms of inverse lower precision-Cholesky, independently checked against dense covariance.
- `row_weights`: incoming row/lambda and just-drawn theta/cycle, plus normalized probabilities across **all** original bank rows, immediately before the original categorical expression. A separate deterministic evaluation leaves that original expression unchanged.
- `completed_sweep`: post-categorical row and post-Gamma lambda, cycle, and missing-race Student-t conditional location/scale, immediately before the original retained missing-race RNG calls.
- `chain_final_rng`: final per-chain generator state after **all** original random draws. No generator is passed to the observer.
- No-poll path: `direct_prior_chain` records retained sampled row indices/cycles and the uniform finite-bank law, followed by final RNG state. It invents no lambda/Gaussian/Gibbs-weight trajectory when that branch has no such states. Original no-poll warmup behavior is preserved.

The pre-state row and post-categorical row are both explicit; the tests exercise actual changes between them. Original source chain/draw identities remain in all original outputs; incoming-source identities reconstruct from the pinned input bank map and incoming row index. Student-t precision phase and original source-row phase must not be confused.

Arrays are copied into immutable `bytes` backing, so ordinary assignment and `setflags(write=True)` fail. Nested mappings are immutable copies as well. Tests try record mutation, nested-state mutation, flag changes, and detached-copy mutation while checking exact transition parity. This protects against mutation *through observer records*; callbacks are ordinary trusted Python code and callback exceptions propagate explicitly rather than silently dropping diagnostics. An observer with unrelated external references is outside that data-isolation guarantee.

`RetainedRowWeightAccumulator` stores each retained full-bank weight vector's contribution in contiguous per-chain/batch sums with counts and sampled occupancy; there is no pruning or thinning. It uses configurable positive batch length (default100), handles final partial batches explicitly, keeps the no-poll uniform law separate, and exports immutable copies and final RNG states. It does not save every 16,000-element per-step vector. A downstream callback may separately retain Gaussian/lambda records. Final RNG metadata contains no additional sampling operation.

## Evidence

Final run: `attempts/pytest-attempt-3.txt`, **18 passed in2.59s**, using existing lab `.venv`, NumPy2.4.4, SciPy1.17.1, pytest9.0.2, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, bytecode disabled. Tests run wholly from snapshots/private code, not owner imports.

Twelve original/off/on cases cross no-poll/partial/full observation, warmup0/7, seed0/78341, with4 chains and20 retained draws. For each, every original output array is compared byte for byte (including dtype/shape), every scalar/list/tuple/mapping value and key is checked, all actual RNG-call argument/result digests are compared in order, and all final generator states agree exactly. The external RNG wrapper delegates each original draw once to a real NumPy generator; it adds no random calls or replacement distributions. Final callback-state records also agree with the externally captured original final states.

Other tests independently verify:

- Dense Gaussian means and covariance diagonals at all80 retained phase records in a separate synthetic chain run; tolerances atol2e-15/rtol2e-14.
- Full-bank normalized row weights against a direct residual and row-dependent normalizer oracle, atol1e-15/rtol2e-14.
- Incoming row/lambda matches the previous completed sweep, and completed row/source identities match original outputs; enough row changes occur to exercise the distinction.
- Saved post-Gamma lambda byte hashes match the actual Gamma RNG outputs; Gamma shape/scale arguments match the post-row conditional residual formula.
- Retained event counts, per-chain occupancy, full-bank sums and final partial batch counts; no warmup contributions enter retained aggregates.
- Copied immutable records, unchanged default parameters, invalid callback rejection and explicit callback-error propagation.

The synthetic large-shape receipt `attempts/synthetic-large-smoke.json` uses16,000 rows,6 basis columns,30 observed races,2 missing races,4 chains,warmup2,draws10. Exact original output/call/state parity passes. That recorded run took approximately0.036s original and0.071s observed, with the selected single-thread runtime. This is only a bounded shape/overhead smoke test: it is far too short to diagnose scientific inference. Timing is hardware/runtime specific. The final test uses pytest's temporary directory for timing output so an independent rerun does not modify sealed artifacts.

## Preserved attempts, scope, and reproduction

No scientific test failed. `attempts/pytest-attempt-1.txt` records17 passing tests; attempt2 adds the direct Gamma-output/conditional check and records18 passing tests. Attempt3 moves test timing output to pytest temporary storage and records18 passing tests. No statistical behavior was changed between these attempts. The system Python initially lacked NumPy; the runtime-discovery failure is preserved in `attempts/setup-initial-check.json`. The existing dedicated runtime resolved that setup issue without installing packages or changing a runtime.

Run source verification with `verify_seal.py`. To independently rerun only the synthetic suite, use the lab virtualenv with bytecode disabled, one BLAS/OMP thread, pytest `-p no:cacheprovider`, and send test output to a new review-owned location:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/hayden/projects/ballotmancy-improvement-lab/.venv/bin/python -m pytest -q -p no:cacheprovider /absolute/path/to/momentum-observed-sampler-sol-v1/test_observed.py
```

`MANIFEST.json` binds code, tests, builder, exact imported local dependency snapshots, preserved attempts and this report. `SOURCES.json` binds owner paths to their unchanged snapshot SHA values, plus runtime versions. `seal.json` externally binds the manifest. No new global inference/scoring thresholds, conditional marginal integrations, forecast runner, target input admission, or model promotion is implemented here. The earlier inference contract remains a separate unaccepted proposal. **Request independent review before accepting this observer for real historical forecast execution.**
