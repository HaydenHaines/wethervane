# Yehu Chen GitHub source archive check: Rasmussen AZ 2012 and Emerson MO 2016

## Outcome

The author repository adds a meaningful intermediary source to the Tierney chain. Its historical 1992–2016 table contains candidate-specific records whose AZ Rasmussen support counts (Flake 250, Carmona 220; N=500) exactly match the published Rasmussen 50%/44% toplines and the Tierney raw row's D/R values (220/250). Its MO Emerson records contain Kander 341 and Blunt 335 (each from a displayed N=750 record), exactly matching the Tierney raw D/R values. This makes a shared data-record lineage plausible and materially stronger than a pollster-name-only match.

The Chen table has no row/poll/sample identifier, question text, primary-release URL, explicit field dates, or population/mode columns. Tierney's later row dates/denominators also do not agree with the most plausible matching release for Missouri. The strongest supportable conclusion is still **candidate source-row/content correspondence, not verified shared questionnaire/respondent sample identity**. The Rasmussen publisher release ↔ 538 poll 5385932 relation remains independently verified. Tierney ↔ Rasmussen stays a strong, not proven, event match; Tierney ↔ Emerson remains ambiguous, with an exact support-count match to a different, later Emerson sample than the earlier n=650 release.

## Primary GitHub repository and history

Repository: `https://github.com/yahoochen97/CNNForecasting`, by Yehu Chen. The README describes a 2020 Senate forecasting project and directs users to clone/run the model. It says “Distributed under the MIT License. See LICENSE,” but no root `LICENSE` exists in the inspected commits/tree and GitHub repository metadata reports no detected license. The 2019 and 2020 CSVs are retained as local source captures only; do not redistribute them based only on the README sentence.

The historical file `data/CNNdata1992-2016.csv` first appears in commit `f6f5cd77362c192217f4dfa6c4f8e8cf88c5f1a7` (author timestamp 2020-09-23T15:35:28Z; blob `c64f25f2aed5c97d456f9c5cbe966f52719020af`). The target AZ/MO candidate values inspected here are already present in this first snapshot. The file is modified in later commits Sep 24, Sep 25, and Oct 12, 2020. The latest supplied commit `0b63a81a5f89f2373966dc9b35e0faa337aa4ae1` (2020-10-12T19:13:09Z; blob `b95e977e4390fade41e157b80775b96d1cea350e`) is unsigned. Both snapshots have identical target candidate values.

The separate `data/CNNData2018.csv` at rename commit `50a27498cd264950127ace0b4da572870ad6759a` is dated 2019-11-20T01:34:58Z, blob `f1cf31fb72a1011e1df777dda4797bb583bf28c7`, and has 1,613 rows. A later `CNNdata2018.csv` appears at commit `c8a6ed318989a6428c110aa749b4b859dda29c2d` dated 2021-02-23, after the 2020 election; it must not be treated as a pre-2020 vintage. The 2019 CSV schema includes an election-outcome field. This review did not use or reproduce any 2018 outcome value; during setup, two initial 2018 CSV lines were inadvertently printed while identifying the file, then I stopped row inspection and excluded all 2018 outcome values from this analysis.

The Git commit times establish that the recorded 2019/2020 blobs were committed before the November 2020 election target; the 1992–2016 records are also frozen in a Sep 2020 blob. These are unsigned GitHub commit timestamps, not independent trusted timestamping. They do not establish that these bytes were available before the 2012/2016 poll events or before the 2018 election/labels. In particular, the 2018 table was committed in 2019/2021, so it cannot be used as a pre-2018 vintage. No claims of prospective or sealed-holdout validation follow from this history.

The 1992–2016 file has 12 fields: `cycle`, `state`, `Candidateidentifier`, `Percentage_of_Vote_won_x`, `samplesize`, `daysLeft`, `numberSupport`, `pollster`, `Republican`, `Democrat`, `pvi`, `experienced`. It stores separate candidate-level rows and a final-election-share field. It contains no poll ID, question ID, original poll dates, source URL, population, or sample identifier. The target-pair facts below omit the final-election-share field.

At commit `0b63a81…`, `data/generateData.py` documents a different/updated path for the 2018 and 2020 files: it reads FiveThirtyEight's Senate poll CSV, derives candidate identifiers, computes candidate support as `samplesize*pct//100`, calculates `daysLeft` from a date range between `start_date` and `election_date`, then drops start/end/election dates, candidate percentage, candidate name and party. It also discusses poll/question IDs and duplicate-question handling for those generated files. That script does **not** reconstruct the historical 1992–2016 file; do not assume its `daysLeft` or support-count mechanics definitively describe every legacy row.

## Row-by-row comparison

### Arizona, 2012

The earliest Chen snapshot and Oct 12 snapshot each have a matched pair for Rasmussen Reports/Pulse Opinion Research, Arizona 2012: `2012AZFlake`, N=500, `numberSupport=250`, and `2012AZCarmona`, N=500, `numberSupport=220`, both with `daysLeft=-16`. The Tierney raw row 3318 has pollster Rasmussen Reports/Pulse Opinion Research, `endDate=2012-10-21`, D=220, R=250, and `numberOfRespondents=470`, equal to the two party-support values' sum. Rasmussen's original release/question page documents a single-day Oct 21, 500-likely-voter Flake/Carmona poll with Flake 50%, Carmona 44%, other 3%, undecided 3%; the pinned 538 poll 5385932 records median field date Oct 21, N=500 and D/R 44/50.

The exact candidate support counts and sample size in Chen align precisely to that release/538 record and to Tierney's D/R counts. This triangulation makes the Rasmussen source association for the Tierney/Chen values highly persuasive. It still does not uniquely identify the same questionnaire or respondents: the Chen file lacks poll ID/question/release link; Tierney row has no source lineage or sample identifier; its reported N=470 is only the sum of the two-party values, not the original N=500. Chen `daysLeft=-16` cannot independently resolve dates: the known 2018/2020 generator would imply a start-date-based day count, but the legacy-table derivation is undocumented and the script comment itself labels the semantics differently. **Status: strong candidate data-row linkage; Tierney/Chen same survey sample/question identity unverified.**

### Missouri, 2016

Chen historical rows include Kander and Blunt under Emerson College at N=750, `daysLeft=-4`, with `numberSupport=341` and `335`, respectively. Tierney raw row 4550 has exact same pollster and D/R numbers 341/335 but lists `endDate=2016-10-30` and `numberOfRespondents=676` (the support-count sum), with start date/population/mode missing.

This is not the Oct 28–31 Emerson release previously linked to 538 poll 75213: that release is n=650 and Kander/Blunt 294/294 (45.2%/45.2%). An additional primary Emerson release, published by Emerson College Polling Society through PR Newswire on Nov 7, says its Missouri survey was conducted Nov 4–5 among likely general-election voters, N=750; it reports Kander 46%, Blunt 45%, and used landline IVR. The 538 archive contains a later Emerson Senate entry dated Nov 5, N=750. Chen's N=750 and 341/335 support counts are compatible with the later poll's rounded 46/45 toplines. This is a much better content match for Tierney's values than the earlier 650-person release, but Tierney's Oct 30 end date conflicts with the Nov 4–5 release and no exact ballot question or sample identifier was found. **Status: strong candidate link to the later n=750 Emerson poll, but ambiguous; do not join as verified identity.**

## Relation to the published papers

Chen, Garnett, and Montgomery's paper “Polls, Context, and Time: A Dynamic Hierarchical Bayesian Forecasting Model for US Senate Elections” says its Senate polling data were obtained from FiveThirtyEight and CNN, acknowledges Harry Enten at CNN for providing data, and points to Code Ocean DOI `10.24433/CO.4154884.v1` plus Harvard Dataverse DOI `10.7910/DVN/GNHESM`. Tierney and Volfovsky's later paper “Bias and Excess Variance in Election Polling: A Not-So-Hidden Markov Model” relies on Chen et al.'s Senate data. These publications establish collection-level provenance from CNN/538 through Chen and then Tierney; the matching GitHub CSV adds candidate-row evidence, but neither paper nor the CSV provides a row-level bridge to a poll ID/question/sample.

## Source captures and hashes

Files in this directory are raw captures at immutable GitHub commit URLs. Repository license is unresolved for datasets, so captures are local research evidence only.

| Capture | Immutable source | Bytes | SHA-256 | Git blob SHA-1 |
|---|---|---:|---|---|
| `CNNdata1992-2016-2020-09-23.csv` | [commit f6f5cd7](https://github.com/yahoochen97/CNNForecasting/blob/f6f5cd77362c192217f4dfa6c4f8e8cf88c5f1a7/data/CNNdata1992-2016.csv) | 1,294,263 | `4c5d8822962f53b535818b40e5f9af622d9b33071524831920338ff87aa5cd0a` | `c64f25f2aed5c97d456f9c5cbe966f52719020af` |
| `CNNdata1992-2016.csv` | [commit 0b63a81](https://github.com/yahoochen97/CNNForecasting/blob/0b63a81a5f89f2373966dc9b35e0faa337aa4ae1/data/CNNdata1992-2016.csv) | 1,286,178 | `0ae4bdcc20712d2248b0d0d144a8c86076ab3133f021af8fcfe770b05253ca44` | `b95e977e4390fade41e157b80775b96d1cea350e` |
| `CNNData2018.csv` | [commit 50a2749](https://github.com/yahoochen97/CNNForecasting/blob/50a27498cd264950127ace0b4da572870ad6759a/data/CNNData2018.csv) | 120,901 | `23e22ac1a64962d21d2f31fdfb8ca903dd3ecd9b7c9514661c726be610000c6c` | `f1cf31fb72a1011e1df777dda4797bb583bf28c7` |

## Next decisive evidence

For Arizona, a Chen/Tierney source row with original `poll_id` or a Tierney import crosswalk to Rasmussen's Oct 21 question page would settle sample/question identity. For Missouri, retrieve the Nov 4–5 Emerson questionnaire/topline detail with original sample or poll ID and a row-level mapping from Chen/Tierney; that is needed to resolve Tierney's conflicting Oct 30 date. The Dataverse/Code Ocean capsule may contain source files or IDs, but prior access to the Dataverse page/API failed; do not infer row-level linkage from the collection-level acknowledgment alone.
