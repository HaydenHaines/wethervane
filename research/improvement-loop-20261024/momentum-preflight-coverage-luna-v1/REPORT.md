# Momentum coverage preflight (no fit)

This packet follows the root-selected no-fit scope. It profiles the 2018–2024 538 target archive and the older Tierney/Volfovsky calibration panel separately. It does not fit, score, optimize, acquire polls, mutate source artifacts, or choose the eventual forecast operator.

The 538 native helper is reapplied after each horizon and availability filter, before its question-selection rules. A separate latest-per-race/pollster operator is profiled. The historical calibration operator retains the latest row per race, pollster and fixed 14-day lead bin. The sources are never concatenated.

## Cutoff and availability rules

H30/H14/H7 are midnight UTC on the date 30/14/7 days before Election Day. Each row must end by that cutoff and be no more than 100 days from Election Day. The additional 0/1/3/7-day lag is added to each source's native availability proxy before filtering. For 538 this is aggregator created_at, normalized to date, plus one day; for the Tierney panel it is field end. Neither establishes that the exact historical poll was publicly available then. All outputs remain strict_asof_qualified=false.

The exact question helper is applied after filtering. Candidate-answer rows are collapsed into a sample/question poll observation. The ID-level selection and exclusion file preserves every historical-panel row and each helper-admitted 538 observation used in the horizon operators; helper reason counts are retained separately. It does not claim a complete raw-candidate-answer exclusion ledger for questions rejected before helper output.

## Reading the coverage cells

Each cell reports rows, unique samples, races, named pollsters, repeated race/pollster groups with at least two distinct field-end dates, their time spans, and interval metadata. Reported interval overlap only means the date ranges overlap; it does not reveal respondent overlap. Unknown pollster labels are not grouped together. Row counts are not independent poll counts when a source sample spans multiple races.

The latest-per-pollster rule removes all within-pollster repeated observations at a given cutoff by construction. The fixed-bin calibration operator retains at most one poll per pollster per race per 14-day lead bin. These are distinct selection operators; a time-trend contrast under one cannot be assumed to exist under the other.

## Support

- 2018: eligible v2 poll roster 28; F0 target list 30; exact shared race IDs 28.
- 2020: eligible v2 poll roster 28; F0 target list 28; exact shared race IDs 28.
- 2022: eligible v2 poll roster 28; F0 target list 30; exact shared race IDs 28.
- 2024: eligible v2 poll roster 29; F0 target list 30; exact shared race IDs 29.

The F0 list is a candidate support comparison only. It does not establish that the F0 model law or its upstream source rows are compatible with a polling update; Sol's separate read-only bank review covers that question. Detailed common and unmatched target race IDs are in f0-race-support.csv; fold-specific F0 training-race intersections with the Tierney panel are in f0-training-race-support.csv.

The reviewed F0 target design and source features are cutoff-bound at H14. An H30 poll-coverage cell is not a fully matched H30 F0 feature forecast; these tables report poll-source and race coverage only.

## Known data limits

The Tierney panel has no field starts, exact publication timestamps, candidate names/IDs, verified sample IDs, or original release chronology. The 538 archive contains reported start/end dates and poll/sample/question IDs but is a later historical archive without a verified publication/revision ledger. Long rolling periods, including the 2022 ActiVote record with a 91-day reported interval, do not identify when respondents were interviewed. Do not expand a long interval into daily polls or infer respondent dates from its midpoint. race-horizon-support.csv makes each no-sample bank/roster race visible at every cutoff.

See receipt.json for source hashes and byte lengths, including the root selection, v2 reviewed identity packet, exact source helpers, target archive, rosters, historical panel and standard fixed-bin runner. No forecast result or model qualification is included.

## H14, zero additional lag: target operator snapshot

### 2018
- 538_native_full_series: 202 observations, 25 races, 92 named pollsters, 25 repeated pollster/race groups; 25/30 F0 target races represented.
- 538_latest_per_race_pollster: 170 observations, 25 races, 92 named pollsters, 0 repeated pollster/race groups; 25/30 F0 target races represented.

### 2020
- 538_native_full_series: 190 observations, 22 races, 77 named pollsters, 32 repeated pollster/race groups; 22/28 F0 target races represented.
- 538_latest_per_race_pollster: 134 observations, 22 races, 77 named pollsters, 0 repeated pollster/race groups; 22/28 F0 target races represented.

### 2022
- 538_native_full_series: 220 observations, 23 races, 69 named pollsters, 38 repeated pollster/race groups; 23/30 F0 target races represented.
- 538_latest_per_race_pollster: 171 observations, 23 races, 69 named pollsters, 0 repeated pollster/race groups; 23/30 F0 target races represented.

### 2024
- 538_native_full_series: 351 observations, 27 races, 86 named pollsters, 74 repeated pollster/race groups; 27/30 F0 target races represented.
- 538_latest_per_race_pollster: 226 observations, 27 races, 86 named pollsters, 0 repeated pollster/race groups; 27/30 F0 target races represented.

All H30/H14/H7 and extra-lag cells, plus complete selected/excluded IDs, are in the CSV artifacts. Root retains selection of a forecast target/operator.
