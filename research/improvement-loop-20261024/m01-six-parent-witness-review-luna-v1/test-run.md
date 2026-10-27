# Reproduction details

The frozen packet was copied with `cp -a` to `/tmp/luna-six-batch-DQG9RW/packet`. Command:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 timeout --kill-after=3s 30s nice -n 15 /home/hayden/projects/ballotmancy-improvement-lab/.improvement-ledger/runtime/calibration-v1-env/bin/python -B test_tiny.py
```

Output:

```json
{"status": "passed", "tests": 32, "invented_calls": 42}
```

No actual graph, fit, sampler or owner job was involved. The separate tolerance probe is fully described and pinned in `m01-six-parent-witness-tolerance-probe-luna-v1`.
