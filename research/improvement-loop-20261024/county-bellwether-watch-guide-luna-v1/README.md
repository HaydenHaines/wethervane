# County watcher evidence review (2026-10-26)

## Finding

The inspected project artifact is a county *share* table, not an election-night county vote target or live reporting model. Its SHA-256 is `52a889be8835cc2edd8c261050cc4c5b5e2f24000f7e865e887dfe7c608f4783`; PyArrow reports 445,588 rows and exactly six fields: `county_fips`, `state`, `county_name`, `pred_dem_share`, `race`, `forecast_mode`. For `2026 TX Senate`, there are 6,232 rows (3,116 each `national` and `local`). There are no candidate counts, turnout totals, uncertainty draws/intervals, as-of/published timestamp, ballot-mode coverage, or reporting progress. Generation and relation to current released forecast remain unknown. Therefore this artifact cannot justify numerical county vote targets or a bellwether threshold.

A previous bounded frontend review reaches the same schema conclusion and identifies the refreshed Senate forecast as state/contest-level. See `../../../../../../wethervane-redesign-20261006/integration/research/coordination-frontend-20261026/county-watcher-guide-response-v1.md`. This is a bounded search, not proof no other work exists.

## Official Texas reporting evidence

Texas SOS says counties submit early-voting results shortly after polls close, followed by Election Day ballots gradually; the SOS-hosted Election Night Returns portal presents **unofficial results reported by counties**, updated every five minutes until all precincts/polling places report. The 2022 notice describes that reporting sequence and cadence, not a fixed expected-vote denominator or a county-level forecast: https://www.sos.state.tx.us/about/newsreleases/2022/110722.shtml (published 2022-11-07; checked 2026-10-26). SOS links to the portal at https://results.texas-election.com/.

The SOS election results archive separates official results from election-night returns and links the official 2025-current system: https://www.sos.state.tx.us/elections/historical/elections-results-archive.shtml (checked 2026-10-26). The 2026 general election is scheduled for November 3, 2026; official SOS current-election page: https://www.sos.state.tx.us/elections/laws/2026-november-general-election.shtml. Thus there is a named official host/feed for county-reported unofficial results, but this review did not establish a documented stable public API, signed payload, cryptographic authenticity, record-level timestamp/revision semantics, or machine-ingestion contract. “Official SOS-hosted portal” is supported; “authenticated live feed” is not yet supported.

## What is missing to make a falsifiable guide

1. A frozen forecast artifact and timestamp, contest/candidate IDs, geography/boundary vintage, common-draw county candidate vote counts and turnout; totals must sum to the same draw-level state result. Include uncertainty intervals and calibrated coverage, not point-share alone.
2. For each proposed signal county, a predeclared rule and historical evaluation over comparable elections. Label it an indicator, not a determinant. Do not select counties after seeing returns.
3. A live-data contract: source URL/provider, jurisdiction and contest IDs, event/update timestamp, extraction timestamp, revision history, reported vote by mode/precinct, explicit reporting-status denominator and its definition. “Percent precincts reporting” must not be treated as percent of ballots counted: precincts/polling places differ in size and early votes may arrive in a batch.
4. A remaining-ballots estimate by county and ballot mode with uncertainty, sourced and time-stamped. Without it, arithmetic “votes needed to catch up” is descriptive only and cannot tell whether remaining votes suffice.
5. Texas-specific caveat: initial early-vote batch followed by gradual Election Day reporting creates a systematic sequence; compare incoming vote modes/locations to historically comparable batches, not just raw county margin movement. Unofficial counts can be revised; certified official results are the outcome reference.

A defensible watcher should publish: frozen baseline and target rule; current unofficial values with “as of” times and source; expected remaining-vote interval; predicted range/uncertainty; a predeclared threshold for “signal”; and a status such as *on track / outside forecast range / not enough information*. If the expected remaining vote interval or source denominator is absent, status should be “not enough information,” not a guessed county target.

## Evidence limits

This review checked one current county parquet schema, prior frontend bounded review, and primary Texas SOS pages. It did not query the live result portal (future election), inspect every state portal, find a documented signed/API feed, or audit county-level modeling provenance. No county target, live query, forecast change, or UI publication was made.
