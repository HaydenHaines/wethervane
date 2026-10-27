# A2018 source-time audit: primary page snapshots and exact-version limits

Date: 2026-10-26. Read-only source audit. No owner jobs, model execution, poll/survey row export, or owner/worktree changes.

## Finding

The frozen A2018 resolver is internally pinned but its `available_utc` fields are not independent historical evidence: its result remains `prepared_source_mapping_audited_full_atom_and_graph_admission_blocked`; it does not authenticate source observation atoms, original source versions/publication times, sample independence, or the executed graph. The current public Senate handoff also does not repair this: its 2018/2022 entries were issued in 2026, use `strictAsOf:false`, and explicitly describe retrospective development projections.

A useful *bounded* primary-source bridge does exist. The Internet Archive CDX records a snapshot of Rasmussen Reports' Arizona Senate October 21, 2012 question page at `2012-10-28T03:37:57Z`; the archived body is a publisher page naming the poll/date, 500 likely voters, and exact question (including randomized candidate order). The corresponding pollster article is dated October 25, and the primary-source-to-538 correspondence is already independently documented in `m01-earlier-vintage-luna-v3` (538 row has the same race, date, n, and Flake/Carmona percentages). This supports the narrow claim that this specific poll question/release page version existed by the archival capture time, not an exact respondent/sample identity or the full A2018 poll vintage.

For the 2018 FiveThirtyEight ratings archive, a signed GitHub commit carries a declared May 30, 2018 timestamp and the contemporaneous FiveThirtyEight article is preserved in an Internet Archive snapshot at 15:11:10Z, about 30 minutes after the commit timestamp. The archived article says the ratings database is downloadable and links to the mutable repository tree. That independently supports a credible *dataset-family public-offer date*. It does **not** authenticate that the exact bytes of `raw-polls.csv` in commit `142cad…` were the bytes served from the article's `master` link at that moment: CDX lookup for the commit-pinned raw CSV had no capture, and the commit signature verifies its payload/timestamp rather than public retrieval at that date. Do not call that exact CSV vintage independently time-authenticated.

## Defensible status

- **Admit narrowly:** the archived Rasmussen question-page version, as source content available by 2012-10-28 03:37:57Z; use only as a single original-release/event bridge with the previously documented caveat that pollster source IDs do not identify individual respondents.
- **Do not admit as an exact historical byte vintage:** FiveThirtyEight `raw-polls.csv` commit `142cad5610ed6b3468e730199034b280b9e805d5` for a strict as-of replay. Its May 30 release-family date is strongly corroborated by a signed commit and contemporaneous publisher article, but no independent timestamped capture of that exact CSV object was located.
- **No full-cohort or source-ownership claim:** neither source establishes completeness, independent samples across rows, all original questions/releases, or a vintage of the prepared 65-block A2018 survey/poll inputs.

## Exact captured metadata

The evidence packet stores URLs, Archive CDX metadata records/digests, retrieved response SHA-256 values, and existing local source-file hashes only. It does not store the archived HTML or reproduce poll rows. The 538 article's capture contains the visible publisher time `2018-05-30 15:00:00Z` (11:00 AM displayed); the Rasmussen question page capture is an independent archive observation, not the publisher's first-publication timestamp. Internet Archive timestamps establish the archive's capture record, not the original publisher posting instant.

## Sources

- Frozen prepared-source resolver: `a2018-source-evidence-resolver-sol-v1/README.md`, `evidence.json`, `result.json`.
- Earlier poll/source crosswalk: `m01-earlier-vintage-luna-v3/evidence.json` and `m01-release-linkage-luna-v1` including its Tierney correction addendum.
- Current public handoff: `live-senate-baseline-handoff-root-v1/handoff.json`.
- Primary pages and immutable/archival observations are listed with SHA pins in `evidence.json`.
