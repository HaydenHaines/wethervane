# Forecast explanations and poll selection, October 20, 2026

The live desktop and mobile interfaces were reviewed, including the expanded model builder, current polling and poll-free forecasts, 2018/2022 scorecards, and House preparation. Plain-language labels and nearby definitions replace unexplained research terms. Advanced scientific evidence remains available through native, keyboard-accessible disclosures.

## Presentation changes

- “Admitted observations” becomes “Poll records included”: survey questions used for a race, not respondents or every available poll.
- Win chance is distinguished from the vote share among the two modeled contenders. Model ranges, simulations, party alliances/caucuses, margins, percentage points, and tipping-rank frequencies are explained beside their graphics.
- The historical scoreboard displays its absolute Brier score before the expandable formula. Historical reconstruction limitations appear before the scores. CRPS and control Brier have plain-language definitions. Actual results remain fixed across forecast scenarios.
- The model builder explains component selection, saved results, experimental status and evidence, including when a new joint calculation is needed. Existing presets and the initially collapsed builder are preserved.
- A short “How this forecast works” sequence connects baseline, evidence, shared election simulations and comparisons.
- Shared chamber-control copy distinguishes Senate vice-presidential tie-breaking from House majorities.

## Polling evidence

`data/polling-selection-20261020.json` is a sanitized presentation receipt, not a new forecast or altered research dataset. It contains 63 cataloged questions: 49 included, five older samples replaced by newer samples from the same provider/race, and nine other unselected questions or source/date holds. Forty-nine records cover 23 of 35 races. These are not all polls published.

The receipt is bound to the exact result hashes of the current standard polling run and its four uncertainty variants. The UI does not apply it to unknown future results, poll-free setups or historical forecasts. Each source record has its actual selection status, a reviewed reason and a source link. Question identity—not shared sample identity—keeps alternate questions from inheriting another question’s inclusion status.

Reproduce from the research worktree:

```bash
python3 scripts/export_poll_selection.py --source-root /path/to/integration
npm run test:poll-selection
```

The exporter validates frozen input receipts, saved selection code, catalog/adapter/evidence hashes, recorded inclusion/exclusion partitions, and all five public forecasts’ exact hashes and per-race counts. A later runner path-serialization edit changed the live file hash, so the saved executable snapshot supplies the selection-code hash; its selection function is also checked against the current source. Unknown exclusion reasons fail explicitly.

Current selection requires matching race/candidate identities, usable source percentages, documented dates before the saved cutoff, fieldwork within 100 days of the election, one chosen question per survey, and the latest eligible sample per provider/race. Original publisher files were not retained; rendered publisher extracts were preserved. Exact publication times and cross-survey respondent overlap are unknown. The model does not use respondent count as a poll-specific weight.

Historical polling rules were checked separately against `full-chamber-root/protocol-v2-gh256.json`, `benchmark-sol/runner.py`, `full-chamber-root/runner.py`, and `full-chamber-intake-sol/build_intake.py` in the polling research packet. They use a 14-day cutoff, a 100-day window, reconstructed availability dates and latest eligible provider/race records. The site explains those rules but explicitly says that its historical poll-by-poll inclusion/exclusion list is not yet published. It does not substitute the current 2026 list.

Forecast JSON, registry hashes, scores, probabilities, simulation draws, and catalog identity remain unchanged. Research terminology in saved downloads remains the original source record; the UI translates known notes in `lib/forecast-copy.ts`.

## Research informing presentation

Luna reviewed primary guidance and the local production components. Root additionally inspected the running public site through Playwright.

- UK Government Analysis Function, [Communicating quality, uncertainty and change](https://analysisfunction.civilservice.gov.uk/policy-store/communicating-quality-uncertainty-and-change/): put material caveats near the estimates and explain uncertainty early.
- Office for Statistics Regulation, [Approaches to presenting uncertainty](https://osr.statisticsauthority.gov.uk/publications/approaches-to-presenting-uncertainty-in-the-statistical-system/pages/2/): uncertainty should remain visible in charts and dashboards.
- AAPOR, [Journalist guide to understanding polls and surveys](https://aapor.org/journalist-guide-to-understanding-polls-surveys/): poll sample size is only one aspect of survey interpretation; polls measure current responses.
- Gneiting and Raftery, [Strictly Proper Scoring Rules, Prediction, and Estimation](https://doi.org/10.1198/016214506000001437): probability scoring evaluates predictions against outcomes; Brier and CRPS assess different outcome types.

These sources inform the explanation design; they do not certify this model or prove that a particular wording improves reader understanding.
