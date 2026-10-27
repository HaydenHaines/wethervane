# Independent follow-up: House boundary-display preparation

Reviewed the isolated worktree `/home/hayden/projects/ballotmancy-house-boundary-display-20261026` at base `317ff150f012871aec6c71ae680d7e76eec5b4db`. Current diff SHA256: `382093a3b309899d57e6e77a9eca3db70a8122b8b1c1fc55c6d9bd2403b3b5a6`.

## Findings

- **Structural/support mismatch is fixed.** `layoutGroupRibbon` now rejects explicit support that conflicts with the structural flag, including one supported group without `structural:true` and multiple groups with `structural:true`. The added fixture checks both contradictions. Historical parsing already enforces structural iff exactly one group; the live adapter derives that flag from outcome count.
- **Historical ribbon context is fixed.** `HouseGroupResults` passes `supportedBlocs` from the parsed group list. The shared ribbon therefore distinguishes absent groups from modeled zero and includes the endpoint or single-group qualification in both tile title and accessible label.
- **Historical map wording is improved.** The historical adapter opts into boundary descriptions. For a single listed group the map label says the assignment follows the only listed group and does not guarantee the actual result; modeled 0/1 outcomes say not impossible/not certain. The same text is carried by the map's label/title. This avoids describing structural support as a finite-simulation estimate.
- No material defect found in these follow-up changes. Probability values remain unchanged; no effective-draw denominator or interval is inferred.

## Scope left open

The Election Night Guide and Election Night Evaluation surfaces remain outside this patch and still need review for boundary/structural wording. This is a presentation correction only; it does not qualify forecasts or validate statistical support.

## Verification

Ran in the isolated worktree:

```sh
node --experimental-strip-types --test tests/group-ribbon-layout.test.mjs tests/house-live-results.test.mjs tests/house-group-result.test.mjs tests/house-group-ribbon.test.mjs
```

Result: 36 passed, 0 failed, 4 browser tests skipped. No site or owner files were edited by this review.
