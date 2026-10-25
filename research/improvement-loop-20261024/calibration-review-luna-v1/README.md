# Independent M01 bank verifier

This directory contains a read-only replay verifier for `calibration-bank-v1/attempt-v1`. It does not invoke the sampler or import the owner's calibration implementation. It checks the frozen plan and launch bindings, terminal artifact SHA pins, all four raw chain NetCDFs, warmup/posterior axes, transition journals and starts, the joint five-coordinate NPZ against retained NetCDF rows, the exponential scale transform, and independently recomputed diagnostics.

The isolated runtime is the owner's frozen calibration environment:

```bash
/home/hayden/projects/ballotmancy-improvement-lab/.improvement-ledger/runtime/calibration-v1-env/bin/python \
  research/improvement-loop-20261024/calibration-review-luna-v1/verify_bank.py
/home/hayden/projects/ballotmancy-improvement-lab/.improvement-ledger/runtime/calibration-v1-env/bin/python -m unittest \
  discover -s research/improvement-loop-20261024/calibration-review-luna-v1 -p 'test_*.py' -v
```

The verification command is intended to run once the owner run has a `terminal.json`. Until then, partial chains and transition journals are evidence of progress only; they are not a completed posterior bank. Diagnostic replay uses ArviZ 0.22.0 for rank-normalized R-hat and bulk/tail ESS. The verifier's result only checks sampler output integrity and the stated calibration target. The target is retrospective on outcomes through 2020, while all availability dates are field-end proxies and publication dates/historical versions are unverified. It establishes neither strict as-of historical validity, independence from same-cycle downstream data, a population/count posterior, a choice of marginal versus calibration-cut feedback, nor forecast/product qualification.
