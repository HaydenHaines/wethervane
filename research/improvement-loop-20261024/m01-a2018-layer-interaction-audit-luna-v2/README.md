# A2018 layer interaction audit v2 — guarded synthetic continuation

This is a research-only successor to `../m01-a2018-layer-interaction-audit-luna-v1/`. The earlier packet remains unchanged and its original aggregate result is copied byte-for-byte as `legacy-result-preserved.json`. This successor replaces Python `assert` gates with explicit `ValueError` checks, validates numeric/profile dimensions and finite positive weights/scales, and uses max-shifted softmax plus log-sum-exp rate calculations.

The check verifies exact pinned plan, aggregate-ledger, selected provider-source, legacy-result, and independent centered-contrast review/test bytes before reporting source facts. It includes both the original invented fixture and the independent review's centered major-party contrast fixture. The latter's difference-in-differences remains an invented algebraic example; it is not a solved full native state and says nothing about posterior support.

## Reproduce

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B check.py > result.json
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest -v test_check
PYTHONDONTWRITEBYTECODE=1 python3 -O -B -m unittest -v test_check
```

Tests cover source tampering under `-O`, fixed legacy-output retention, centered contrast interaction, extreme finite utilities, malformed profile dimensions, zero/NaN/infinite inputs, scalar cancellation, common weight scaling, and common utility shifts. There are no model imports, real data rows, posterior arrays, owner jobs, or source edits.

Captured test output: `tests-normal.log` and `tests-optimized.log`.

## Status and boundaries

The original A2018 plan remains unqualified, incomplete, and unauthorized. This packet is not an admission guard and does not establish raw observation lineage, sample independence, historical source availability, compiled graph execution, fitted-state attainability, forecast calibration, or a weakest-layer ranking. The centered fixture is source-shape-informed but is not a full native parameter state. Preserve the original packet; request independent review of this successor before adoption or use as a guard.
