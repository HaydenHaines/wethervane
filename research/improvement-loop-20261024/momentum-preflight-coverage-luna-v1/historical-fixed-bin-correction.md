# Historical fixed-bin horizon sensitivity correction

Correction to the first-pass H30/H14/H7 historical sensitivity implementation in coverage-cells.csv: the first pass used sample-specific keys for labels exactly equal to unknown, while the standard runner uses the literal normalized pollster string. The pinned Tierney panel has no missing or literal unknown pollster labels, so this code difference did not change the observed data results. The unchanged historical baseline in historical-calibration-baseline.csv already matches the runner. This correction retains the original first-pass files and supplies the exact-runner sensitivity separately.

The corrected sensitivity uses the exact standard_runner key pollster.str.strip().str.casefold(), so literal unknown labels share a pollster key within their race just as in the existing calibration code. It filters historical poll rows to end lead H+extra lag through 100 days, then keeps the latest end_date/observation_id in each cycle/race/pollster/14-day bin. Full selected and excluded IDs are in historical-fixed-bin-horizon-selection.csv. No model or score was computed.

Use historical-fixed-bin-horizon-correction.csv for the corrected horizon-truncated calibration support. The file includes a delta against the first-pass counts and explicit F0 training-bank race intersection. The exact unchanged all-0–100 calibration operator remains in the separate historical-calibration-baseline.csv.

The exact corrected pollster rule was compared against the first-pass selection for every earlier-only fold × horizon × lag: all 48 fixed-bin row counts and selected observation-ID sets match (zero delta). The correction is a method-binding correction, not a changed coverage result. This exact-rule table and its row IDs are authoritative for the cutoff-truncated historical sensitivity; the first-pass values are retained but superseded for interpretation. The correction is source-pinned to the initial coverage manifest, historical panel, standard runner, and reviewed F0 support manifest in its receipt. The first-pass artifact manifest remains unmodified.

- 2018, H14, zero extra lag: 746 exact-key fixed-bin rows across 59 races; 59/299 F0 training races have a poll row.
- 2020, H14, zero extra lag: 965 exact-key fixed-bin rows across 84 races; 84/329 F0 training races have a poll row.
- 2022, H14, zero extra lag: 1179 exact-key fixed-bin rows across 105 races; 105/357 F0 training races have a poll row.
- 2024, H14, zero extra lag: 1179 exact-key fixed-bin rows across 105 races; 105/387 F0 training races have a poll row.
