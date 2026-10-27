# Independent reproduction

The first isolated packet copy did not run because V2's `test_tiny.py` intentionally locates the frozen V1 packet and the V1 reproducer through sibling-relative paths. I preserved the sources and reran under `/tmp/luna-six-parent-v2-tree-dyxODV/` with all three sibling directories present.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 timeout --kill-after=3s 30s nice -n 15 /home/hayden/projects/ballotmancy-improvement-lab/.improvement-ledger/runtime/calibration-v1-env/bin/python -B test_tiny.py
```

Output:

```json
{"status": "passed", "tests": 65}
```

The suite reproduces V1's weak marker acceptance, checks protocol-fixture child output capture for both phases, and rejects the V2 negative cases. Fixture numeric values are invented; no actual model graph is imported or evaluated.
