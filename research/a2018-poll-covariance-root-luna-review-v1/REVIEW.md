# Independent review: A2018 prepared poll covariance numerics

Review target: `../a2018-poll-covariance-root-v1` (source packet files pinned below). Independent rerun used Python 3.12.3 / NumPy 2.4.4 from `/home/hayden/projects/ballotmancy-house-boundary-display-20261026/.venv/bin/python`, with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. No source/job files were changed. No row identifiers, entries, or arrays were emitted.

## Source pins

Prepared problem: 6,979,007 bytes; SHA-256 `abcaf347e682422303780c5ff12bb3e6179e4b98362686c01338e29df06a385a`.

Review target artifacts:

- `check.py`: `b148576746b66c193c60c648c6e21d22394cf367eed7210ec70565cbd10faf97`
- `test_check.py`: `ade7bd267def0bd132e4f7ba647f401fdb7f7f6568434c1133d947ec94c984ba`
- `result.json`: `839bd56c6a4194662c40a59acba471da024e918bd7f7c3f83d473730fc7dc6c9`
- `README.md`: `43c40c6241c960d68ee202fe0a3a8f5439080e0f7bb14baf4481da6d4d340441`

## Independent numerical reproduction

Decoded the exact problem bytes and independently converted its sole covariance to NumPy float64, then ran `numpy.linalg.eigvalsh`, `numpy.linalg.cholesky`, and the Frobenius reconstruction residual. Dimensions were 274x274; all entries finite, exact symmetry in the decoded float64 representation, and positive diagonal. Eigenvalues: min `0.00835388899599366`, max `1.1121489607052006`; ratio `133.1294875043902`. The stated scale × epsilon × dimension margin is `6.76633893877343e-14`. Cholesky succeeded; relative Frobenius residual `1.6323308302469188e-16`. All reported values match `result.json` exactly in this environment.

Ran the target's three unit tests and CLI with the same thread limits: all 3 tests passed; CLI aggregate output matched the recorded result.

## Review findings and limits

- No defect found in the narrow aggregate-only check. Its hard-coded source byte count and SHA are checked before JSON decoding; it does not import the model or expose matrix contents. Its output contains only aggregate statistics, fingerprints, and qualification flags.
- The check's scope is properly limited: computed float64 eigenvalues plus a heuristic scale-based margin and successful Cholesky are strong numerical diagnostics for these pinned bytes, not an exact-arithmetic proof or a claim of scientific validity.
- Source pinning establishes identity of the prepared file at read time, not the truth/admissibility of its upstream observations, historical availability, actual graph compilation/use, or forecast calibration. The separate source-evidence resolver describes these limitations and leaves full atom/graph admission blocked.
- Test coverage is modest and appropriate for this small helper: positive, ill-conditioned, malformed, indefinite fixtures, byte-pin tamper, and aggregate-output checks. These are tests of helper behavior, not independent proof of source semantics.

Conclusion: independently reproduced the numerical result and found no leakage or numerical overclaim in this packet. Status remains a numerical diagnostic only; no model or scientific qualification follows.
