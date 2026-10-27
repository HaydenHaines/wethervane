# Reproduction record

Frozen packet test command, run in a fresh copy under `/tmp/luna-six-parent-y1YJfI`:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 timeout --kill-after=3s 30s nice -n 15 /home/hayden/projects/ballotmancy-improvement-lab/.improvement-ledger/runtime/calibration-v1-env/bin/python -B test_tiny.py
```

Output:

```json
{"status": "passed", "tests": 40}
```

Independent counterexample:

```sh
research/improvement-loop-20261024/m01-remaining-fixed-parent-review-luna-v1/run.sh
```

Output preserved in `reproduction.json`: changed exposure identity rejected; a self-authored success-shaped marker with empty comparison-artifact map was accepted by `selection.completed()` when given its computed digest. No model code was imported.
