# Historical calibration provenance: source identity and smallest next step

The 1,483-row `tierney_cnn` calibration source is byte-authenticated as a
**later author replication file**, not an original poll-release archive.
Its CSV SHA-256 is `61e1141f5ed84cfbbec487115742f59c6e20ebd04ba99a0a32f08d8fa2b20322`,
Git blob `2b397788b99435522a7f051a4040bcd913c693f6`, at author repository
commit `4aba5bc539dde8157bb2eeae9f3330a01439c19d` dated March 8, 2023.
Both hashes were recomputed against local bytes for this audit. Selection and
normalization preserve source row numbers, values, paths and candidate-join
qualifications. None of those establishes original publication timing.

The primary [Tierney/Volfovsky paper, section 3](https://arxiv.org/pdf/2206.14570)
attributes the Senate polls to Chen, Garnett and Montgomery. Their
[official replication dataset](https://doi.org/10.7910/DVN/GNHESM) is the next
upstream provenance lead. Do not infer that the filename `cnn` names CNN news
as the original source. The exact transformation from that upstream dataset
into the Tierney file remains unverified. The
[pinned author preparation script](https://raw.githubusercontent.com/g-tierney/polling_errors_replication/4aba5bc539dde8157bb2eeae9f3330a01439c19d/scripts/01_model_prepare_input.R)
selects this file for Senate analyses and transforms columns after ingestion;
it does not document how the input file's D/R values were reconstructed.

All 1,483 selected rows still have publication-date, historical-version and
strict-as-of verification false. Their sample identity/field starts are
unknown; `available_date` is the declared field-end proxy. The 2023 commit
cannot establish a pre-2018, pre-2020 or pre-2022 dataset vintage. For 2024 it
is an earlier repository version, but later corrected roster/outcome joins
and unverified release reconstruction still prevent a strict replay claim.
This is absence of proof, not proof that the polls were unavailable.

The separately pinned owner response
`coordination-frontend-20261026/m01-historical-poll-source-response-v1.md` says
that session is not acquiring Tierney release dates or vintages. This does not
prove that no other session is doing so. The owner tree remains read only.

## Reuse existing preparation; do not refit or rebuild it

The independently reviewed momentum preparation already contains earlier-cycle
poll and label selections. This audit reads those files and verifies every
selected poll and outcome cycle is less than its target cycle. These are
earlier-event inputs, **not** proven pre-target source-byte vintages.

| 2026 evaluation tier / target | Prepared training cycles | Prepared rows | Rows also in current event-disjoint bank | Earlier labels |
|---|---|---:|---:|---:|
| A / 2018 | 2012, 2016 | 930 | 930 | 62 |
| A / 2022 | 2012, 2016, 2018, 2020 | 1,514 | 1,483 | 118 |
| B / 2020 | 2012, 2016, 2018 | 1,218 | 1,218 | 90 |
| B / 2024 | 2012, 2016, 2018, 2020 | 1,514 | 1,483 | 118 |

`report.json` binds exact existing files, hashes, counts and all 31 excluded
TX2020 observation IDs in each affected fold. The exclusions are the current
TX pilot's identity restriction, not a universal exclusion for every historical
native Problem. A new Problem may consume other official results and requires
its own canonical event exclusion or admitted joint conditional law. The
current bank excludes TX2020 and TX2024 event keys, although no 2024 rows exist
in its training panel. Same-cycle other-state dependence is not removed.

The 2022 and 2024 preparation lacks 2022 training data because the source ends
at 2020. Do not fabricate 2022 rows or describe 2024 as using the latest
completed training cycle. Existing 2022/2024 equal numerical inputs do not
establish equal admitted native Problems or authorize reusing a fitted bank.
The current 2026 posterior bank is not an earlier-only bank for these targets.

## Smallest reproducible route

1. Retrieve only the small upstream research capsule and its official version
   metadata, when accessible. Parent review reports the indexed official
   Dataverse listing shows a 6.0 MB research capsule dated September 13, 2021;
   this audit did not acquire it or independently inspect its contents. Do not
   download the separate 26.8 GB result archive. The paper's alternate
   [Code Ocean capsule DOI](https://doi.org/10.24433/CO.4154884.v1) is another
   route. A 2021 capsule could document an earlier compilation/transformation,
   but would still not by itself establish 2018/2020 publication vintages.
2. Match upstream rows to the retained Tierney rows by source identity,
   state/event, pollster, field end and source values; record ambiguous joins
   rather than treating approximate matches as exact. Recover original URL,
   field interval, universe, question, candidate values, nominal N and sample
   linkage where actually present. Keep byte hashes and transformation lineage.
3. For each recovered release distinguish publisher release date/clock,
   preserved document date, archive capture time, retrieval time and the
   dataset's compilation date. An archive captured before cutoff can establish
   the captured version was available by then; a later capture or printed date
   cannot prove absence of earlier revisions. Date-only evidence must retain
   date resolution rather than an invented UTC instant.
4. Reuse the existing selection masks as a candidate list, version a successor
   provenance ledger and disclose verified coverage by fold. Do not silently
   keep only easy-to-recover pollsters and claim a matched comparison to the
   original full panel. Either compare both arms on the same supported subset
   and report selection loss, or retain a labeled retrospective approximation
   with matched release-lag sensitivities. Field-end plus 1/3/7 days is a
   sensitivity design, not recovered historical publication evidence.
5. Bind each target's actual full native evidence ledger, earlier-only result
   joins and once-only labels before preparing posterior-bank targets. Keep
   protected population conditional masses and explicit shared-error handling.
   Later authorized joint refits must use common information and comparisons;
   no existing MAP or supplied-objective result is an integrated uncertainty
   bank or empirical forecast qualification.

The 538 local archive has better question/sample/URL metadata but uses a later
March 2025 snapshot and `created_at + one day` proxy, while the Stanford
auxiliary source has missing original release/sample metadata. They are
potential linkage aids or separately versioned alternatives, not independent
extra samples to concatenate; Stanford copies in the two author repositories
are the same evidence. Existing corrections and negative source findings stay
authoritative.

`access-log.json` records actual failures: Dataverse API HTTP403, DOI redirect
HTTP500, a Javascript anti-robot response, and inaccessible Code Ocean/file
history routes. No small capsule, large results, or new poll source bytes were
downloaded. The source pin/replay-cycle audit produced no fit, source admission,
owner edits, new calibration bank, or predictive improvement claim.
