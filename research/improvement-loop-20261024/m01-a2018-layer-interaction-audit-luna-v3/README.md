# A2018 layer interaction audit v3 — gauge-stable synthetic continuation

This research-only successor preserves the prior packets and the original fixed fixture output byte-for-byte (`legacy-result-preserved.json`). It corrects an instability found in v2: adding a large common utility offset caused cancellation in `d - (max + logsumexp(...))`, changing a weighted profile aggregate even when the represented relative utilities were unchanged.

The v3 helper centers each profile's utilities before computing log-softmax terms and centers candidate log-rate terms together before subtracting. It uses explicit `ValueError` source/closure gates (including under Python `-O`), validates profile/vector shapes and finite positive weights/scales, and retains max-shifted softmax/log-sum-exp operations. The input pins include the v2 code/test/result and the independent review that supplied the centered-contrast fixture.

The result carries the original two-profile fixture plus the independent review's centered major-party contrast fixture. Both are invented, not fitted parameter states. The centered example is informed by selected source structure but is not a solved full native state or evidence of posterior support.

## Reproduce

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B check.py > result.json
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest -v test_check
PYTHONDONTWRITEBYTECODE=1 python3 -O -B -m unittest -v test_check
```

Tests cover a tampered pinned fixture under `-O`, preservation of the old output, both synthetic interactions, extreme utilities, malformed profiles, zero/NaN/infinite inputs, scalar/weight/utility invariance, and exact-representable common utility offsets at `1e16` and `1e20`. Captured suite output is in `tests-normal.log` and `tests-optimized.log`.

## Status

The source-pinned A2018 plan remains unqualified, incomplete, and unauthorized. The synthetic mechanism does not authenticate raw observations, sample independence, source availability, compiled graph use, native fitted-state attainability, forecast calibration, or a weakest-layer ranking. This is not an admission guard. Request independent review of v3 before adoption or use as a guard.
