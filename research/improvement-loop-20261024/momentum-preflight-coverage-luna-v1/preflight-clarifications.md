# Preflight clarifications and readout

This note records the final interpretation used across the coverage tables, historical-baseline supplement, and field-interval supplement. It contains no model, score, fit, or forecast values.

## Keep the operators separate

- **538 native full series:** all helper-admitted poll observations within the horizon, 100-day age cap, exact D/R nominee-ID gate, and availability proxy rule. Repeated same-pollster time points remain in this view.
- **538 latest per race/pollster:** retain only the latest qualifying sample for each race and pollster. This has no repeated same-pollster time series by construction.
- **Tierney/Volfovsky calibration:** distinct 2012–2020 source. The unchanged existing operator uses all observations 0–100 days before the election, then the latest observation in each cycle/race/pollster/14-day lead bin. The historical-baseline CSV is the exact earlier-only fold profile. H30/H14/H7 historical rows in coverage-cells.csv are additional horizon-truncated sensitivities; they do not replace the unchanged calibration operator.

The 538 archive and author panel are not concatenated. They may share underlying releases, and cross-source sample identity is not known.

## Horizon and availability binding

H30/H14/H7 cutoffs are Election Day minus 30/14/7 calendar days at 00:00 UTC; rows must also have field end no later than that cutoff and be no older than 100 days. The question-selection helper is applied after filtering.

The table's additional lags 0/1/3/7 days are added to each native proxy. For 538, the base proxy is created_at normalized to date plus 1 day, so effective delay from that normalized creation date is 1/2/4/8 days. For the historical panel, the only available proxy is field end plus 0/1/3/7 days. These are matched extra-lag sensitivities, not identical underlying timestamps. Exact release bytes and publication dates are not verified; strict as-of qualification remains false.

The reviewed F0 bank compatibility manifest confirms exact target race support across eight earlier-only H0/F0 banks. The selected F0 target features are bound at H14. Poll coverage at H30 does not imply an H30-matched feature design. A later H7 comparison may transparently retain the earlier H14 features if Root selects that estimand.

## Main support pattern

At zero extra lag, the 538 full-series H30/H14/H7 observation counts and number of races are: 2018 135/24, 202/25, 236/25; 2020 126/19, 190/22, 239/24; 2022 147/23, 220/23, 274/24; 2024 252/26, 351/27, 395/27. Repeated race/pollster groups with at least two distinct end dates are respectively 15/25/37, 19/32/45, 19/38/53, and 50/74/91.

At H14, latest-per-race/pollster retains 170, 134, 171, and 226 samples over the same 25, 22, 23, and 27 poll-covered races. The 538 native full-series operator therefore carries repeated same-pollster timing information that the selected latest-pollster operator deletes. Whether that series adds identifiable information beyond pollster/source changes is not established by this count audit.

Reported field interval overlap is rare: there are no overlapping pairs in the 2018 or 2022 H30/H14/H7 cells; one pair in 2020 H7; and one pair in 2024 H7. The exact pair IDs and dates are in field-interval-overlap-pairs.csv. Overlapping advertised intervals do not establish repeated respondents; distinct sample IDs likewise do not establish independent respondents.

The 2022 Florida ActiVote observation 538:80838 has a reported 91-day field range (June 27–September 25) and proxy availability September 29. It passes H30/H14/H7 under all listed extra lags. The 2022 North Carolina ActiVote observation 538:81483 has a 91-day range ending November 3, after each requested horizon cutoff, and is excluded there. No record is expanded into daily polls or assigned respondent interview dates.

## Identity and missingness

The reviewed v2 candidate identity crosswalk is byte-pinned unchanged. The four 2022 reviewed joins affect outcome/truth joins, not this poll question-selection pass; this preflight does not load or use election results and does not infer aliases. Poll question candidate identities are still selected by the exact archive candidate-ID match.

f0-race-support.csv and race-horizon-support.csv retain bank/roster races with no admitted poll at each cutoff. For individual selected and excluded poll observations, use the row-level CSVs. 538-helper-question-audits.csv retains exclusion reason counts from the helper; it is not a row-ID ledger of raw candidate-answer rows rejected before a sample/question observation exists.

