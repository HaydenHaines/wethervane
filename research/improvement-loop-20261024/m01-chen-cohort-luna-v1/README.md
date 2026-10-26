# Chen 2019/2021 candidate cohorts for an earlier-only 2020 replay

## Conclusion

The 2019 Chen repository snapshot is sufficient to construct a **provisional candidate-level cohort with outcome labels** for 2012, 2016 and 2018. Its combined `CNNData1992to2018.csv` contains all three cycles, an outcome field for every inspected candidate row, and enough fields to group likely Democratic/Republican candidate records by state, `daysLeft`, pollster and sample-size value. Under a transparent coarse grouping, it yields 713 pairable groups in 2012, 837 in 2016 and 609 in 2018; each of those groups has both candidate outcome fields present.

That is **not sufficient to assert a distinct poll-event cohort or calibrate it as an independent sample**. The CSVs contain no stable poll ID, question ID, question wording, release URL, start/end/publication date, population or mode. Eighty of the 2019 2018 coarse groups have multiple D/R candidates and are ambiguous under the available columns. Other uniquely paired groups can still combine separate surveys or question variants because the identifiers are absent. The candidate rows show substantial exact content overlap with Tierney's prepared rows in 2012. A separate file name or author repository does not make the underlying polls distinct.

Use the 2019 snapshot only as a **candidate cohort for further reconstruction**, not as admitted calibration evidence. A publishable/calibration-ready cohort needs row-level poll/question IDs or independently validated original-release crosswalks, outcome verification, a fixed population/question definition, and a justified sample-size/rounding transformation. No model fitting or source admission was done.

## 2019 snapshot and chronology

The pinned Yehu Chen GitHub repository [`CNNForecasting`](https://github.com/yahoochen97/CNNForecasting) describes a 2020 Senate forecasting project. At rename commit [`50a27498`](https://github.com/yahoochen97/CNNForecasting/commit/50a27498cd264950127ace0b4da572870ad6759a) (Git author timestamp 2019-11-20T01:34:58Z), `data/CNNData1992to2018.csv` is a renamed existing file (blob `32ed0899905a8a218204a8fca9319b3e84aede1d`). The 15,099-row table has cycles 1992–2018, including 2012, 2016 and 2018. Its candidate-row fields include both `Percentage_of_Vote_won_x` (an outcome-share label) and poll candidate/support information. The timestamp precedes the November 2020 target election, but it is unsigned Git metadata, not an independent timestamp or proof that the bytes were publicly retrieved/available by a particular date. This is not a prospective-validation claim.

The same 2019 commit has standalone `CNNData2018.csv` (1,613 rows) and `CNNData1992to2018idx.csv`. The 2018 slice of the combined table exactly equals all 1,613 standalone rows in order across the 10 shared substantive fields. It is a duplicate representation of the 2018 data, not another cohort. The `idx` file adds only `pollsteridx`; it does not restore poll, question, sample or source IDs. The separate `CNNData.csv` at this commit uses the same 10-field schema and likewise has no poll/question identifiers.

The later `CNNdata2018.csv` snapshot at commit [`c8a6ed3`](https://github.com/yahoochen97/CNNForecasting/commit/c8a6ed318989a6428c110aa749b4b859dda29c2d) is dated 2021-02-23, after the 2020 election. It has 1,200 rows and only the 2018 cycle. It cannot establish source availability for a 2020 prediction and cannot alone supply 2012/2016. It is chronologically before the 2022 target if used only as 2018 historical data, but its timestamp is likewise not independent proof of public availability. The change from 1,613 to 1,200 candidate rows is a source-version difference; it does not prove the later file is improved or more accurate.

## Candidate-pair counts and outcome labels

For an auditable inventory only, I grouped candidate rows by `(cycle, state, daysLeft, pollster, samplesize)` and required exactly one row marked Democrat and one row marked Republican. I restricted this descriptive screen to nonpositive `daysLeft`; it did not merge or split by unobserved poll/question IDs. The resulting **pairable row groups** are:

| Snapshot/cycle | Candidate rows | Pairable D/R row groups | Ambiguous groups with multiple D/R rows | Pairable groups with both candidate outcome labels |
|---|---:|---:|---:|---:|
| 2019 combined, 2012 | 1,468 | 713 | 12 | 713 |
| 2019 combined, 2016 | 1,733 | 837 | 0 | 837 |
| 2019 combined, 2018 | 1,613 | 609 | 80 | 609 |
| 2021 standalone, 2018 only | 1,200 | 562 | 7 | 562 |

These are candidate-row groupings, **not verified counts of unique surveys**. “Pairable” means one row on each party side within the coarse key; it does not establish that both rows came from the same questionnaire or respondents. The combined 2019 file has no missing numeric `Percentage_of_Vote_won_x` values among the 2012/2016/2018 rows inspected, and each candidate identifier maps to one consistent outcome-share value within those cycles (196 candidate IDs). This is internal completeness/consistency, not independent validation against official certified returns. Candidate IDs also need a name/office/contest crosswalk before labels are admitted, especially for specials/runoffs or naming variants.

### Can it be distinct from Tierney?

A separate comparison of the 2019 combined source table to the existing prepared Tierney rows used the coarse state/cycle/day/pollster key, exact D/R support values and a source D/R pair with matching Chen sample-size fields. It found candidate-pair screen matches for **467/467** prepared 2012 rows, **1/463** 2016 rows and **2/288** 2018 rows. Chen source `samplesize` differed from Tierney `reported_n` on all those matched rows; Tierney `reported_n` is the author-derived sum of D/R support values and should not be read as full poll sample N. The 2012 match rate and exact support content strongly suggest the datasets share source records, even though a row-level crosswalk is missing. The low 2016/2018 rates do not prove the unmatched records are distinct: day conventions and count transformations differ, and poll/question IDs are unavailable. Therefore a deduplicated independent cohort cannot be certified from these files alone.

## Schema, dates and sample-size meaning

- **2019 combined/standalone schema:** `cycle`, `state`, `Candidateidentifier`, `Percentage_of_Vote_won_x`, `samplesize`, `daysLeft`, `numberSupport`, `pollster`, `Republican`, `Democrat` (standalone also has an empty index column). These are candidate-level rows. There is no poll ID, question ID, date interval, questionnaire, source URL, population, voter screen, sample mode, sponsor/partisan flag or row-level provenance field. `daysLeft` is a derived integer, not an original poll date.
- **2019 index schema:** same plus `pollsteridx`, a pollster-code field rather than a survey identifier.
- **2021 standalone schema:** `cycle`, `state`, `pollster`, `samplesize`, `candidate_name`, `Candidateidentifier`, `daysLeft`, `numberSupport`, `Democrat`, `Republican`, `Percentage.of.Vote.won.x`, `pvi`, `experienced`. It still has no poll/question ID or exact dates/population/mode/source URL.
- **Author generator code:** At [`0b63a81a`](https://github.com/yahoochen97/CNNForecasting/blob/0b63a81a5f89f2373966dc9b35e0faa337aa4ae1/data/generateData.py) (2020-10-12), `generateData.py` says it generates 2018 and 2020 outputs from FiveThirtyEight's live Senate poll CSV. It derives `numberSupport` as `samplesize*pct//100`, computes `daysLeft` as the negative inclusive number of dates from `start_date` through `election_date`, then drops start/end/election dates and poll/question/candidate attributes from the output selection. The code comment calls `daysLeft` an end-date measure, but the formula uses `start_date`; this discrepancy must be resolved before comparing exact day keys. For 2018 it filters poll/question variants by poll/question ID, population and candidate count before dropping IDs. For multiple candidate rows under a question, it overwrites `samplesize` with the sum of their `numberSupport` values. Thus a later `samplesize` field can be effective reported-support total, not original respondent N. This source code is dated after the 2019 snapshot and does not document how the legacy 1992–2018 table was assembled.
- **2021 generator revision:** [`generateData.py` at `c8a6ed3`](https://github.com/yahoochen97/CNNForecasting/blob/c8a6ed318989a6428c110aa749b4b859dda29c2d/data/generateData.py) differs from the 2020 version, including a manual 2018 Mississippi candidate identifier remap and additional filtering/recoding. It reinforces that 2019 and 2021 snapshots are separate data vintages, not an accuracy progression.

The pair counts show this concretely: in 2019, D+R support sums are below source `samplesize` for 713/713 pair groups in 2012, 835/837 in 2016, and 606/609 in 2018. In the 2021 2018 snapshot, after excluding eight positive-day candidate rows, D+R sums equal `samplesize` on 555/562 pairs and are smaller on 7. This large semantic shift is consistent with the generator's sample-size overwrite but does not prove its exact cause for every row. Rounded percentages converted to integer support are not original respondent counts; use of them as binomial denominators requires explicit reconstruction and validation.

## Target-safe interpretation

- **2020 forecast target:** The 2019 combined snapshot contains polling and outcome fields through 2018 and no 2020 cycle rows, so it is chronologically suitable as *candidate* earlier-cycle information based on its 2019 commit timestamp. It is not evidence that those bytes were publicly available at forecast time, and its 2018 outcomes must be used only as pre-2020 training labels, not as features for their own race.
- **2022 forecast target:** Both 2019 and 2021 source timestamps precede 2022; the 2021 snapshot contains only 2018 rows. This makes it chronologically plausible as historical input for 2022, not verified as-of evidence. It is not admissible for a 2020 target.
- **2018 as target:** Any 2018 outcome label is the target for that cycle and cannot be used to predict 2018. The review evaluates only the earlier-information interpretation for target years after 2018.

## Required work before calibration use

1. Recover an original per-survey source table with stable `poll_id`/`question_id`, exact field start/end and publication dates, candidate names and parties, sample frame/mode, pollster sponsor, question text/version, and unrounded reported shares where available.
2. Pair Democratic/Republican observations only within the same original survey/question/population; resolve multiple candidate matchups, third parties, runoff/special elections and date conventions explicitly.
3. Validate candidate identifiers and historical outcomes against official election returns by contest, then preserve the outcome crosswalk/hash.
4. Derive poll shares and effective N under a written, reproducible rule; do not treat floored values from rounded percentages or post-overwrite `samplesize` as respondent-level counts without source support.
5. Crosswalk Chen rows against Tierney/538 and remove duplicate survey events by stable IDs. If IDs cannot be recovered, keep the source-only cohort descriptive and state that independence is unknown.
6. Pin exact source bytes and establish dated retrieval evidence appropriate to the forecast origin; Git timestamps alone do not prove historical public visibility.

## Local captures and rights

All captures were downloaded from immutable author GitHub URLs and are retained locally only because repository-level reuse rights for embedded data are unclear. The 2019 README at the cited commit is only a project title/tagline. A later README says “MIT” and points to a LICENSE, but no root LICENSE was present in the inspected tree and GitHub's API reports no detected license. No training file or model was modified.

See `evidence.json` for URLs, commit/blob IDs, local capture hashes, computations and precise limitations.
