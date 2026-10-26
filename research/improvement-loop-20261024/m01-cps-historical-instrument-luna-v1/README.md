# CPS Voting Supplement instrument, weights, and vintage gate

Date: 2026-10-26

## Scope and conclusion

This is a read-only check of the Census Bureau's original 2018, 2020, and 2022 Voting Supplement technical documentation and current year-specific archive pages. No CPS microdata were downloaded or analyzed.

The three questionnaires have the same relevant voter-registration skip: after the voting question, a reported “Yes” skips the registration question and proceeds to the vote-method item; “No,” “Don't know,” and “Refused” route to registration. The record-layout descriptions separately give PES2 an edited universe that includes PES1=-9 (“No Response”). That does not establish that a respondent with PES1=-9 was asked or answered PES2. A new registration/turnout model therefore cannot treat the records as directly observing registration among voters without an explicit, defensible assumption or additional data.

The edited universe is adult respondents with citizenship codes 1–4 (and, in 2022, an explicit supplement-interview field), not a census of all legally eligible voters. These are survey records with interview, proxy-response, nonresponse, and editing processes. In 2022 the PES1 valid-value listing omits -9 while the PES2 edited universe still includes PES1=-9; record this as a documentation inconsistency, not as evidence of a changed question path.

The prior Luna report's blanket statement that person replicate weights are available for every 2018–2024 voting supplement is superseded as unsupported. Current Census year-specific pages list a distinct “Replicate Weights” section for 2022, but not for 2018 or 2020. The latter observation does not prove weights are unavailable elsewhere or were never released. The annual technical documents' “no supplement weight; use PWSSWGT” wording concerns the full-sample point-estimation weight and does not settle the existence of replicate-weight files. Older-year replicate-weight availability, exact contents, first release, and vintage must be separately established before use.

## Year-by-year instrument evidence

| Year | Record-layout universe / coding | Questionnaire path | Weight and vintage finding |
|---|---|---|---|
| 2018 | November 2018 technical documentation, printed pp. 7-2 (PDF p.113): PES1 edited universe is PRTAGE >=18 and PRCITSHP=1,2,3,4; PES1 includes -9 No Response. PES2 edited universe is PES1=2,-2,-3,-9. | Printed questionnaire pp. 9-1–9-2 (PDF pp.118–119): SCK age/citizenship checks; S1 Yes -> S5; S1 No, DK, Refused -> S2. There is no explicit interviewer no-response branch in this routing diagram. | Overview, printed p.3-1 (PDF p.9): interviews Nov. 11–20, self/proxy allowed; no supplement weight, use PWSSWGT. Year-specific archive lists CPS Supplement files but no separate Replicate Weights heading. State-level table release was Apr. 23, 2019. That release date does not prove the exact public-use microdata or replicate-file first-availability date. |
| 2020 | November 2020 documentation, printed pp. 7-2–7-3 (PDF pp.113–114): same PES1 universe and PES2 edited universe, including PES1=-9. | Printed questionnaire pp. 9-1–9-2 (PDF pp.118–119): same routing. | Overview, printed p.3-1 (PDF p.9): interviews Nov. 15–24, self/proxy allowed; no supplement weight, use PWSSWGT. Year-specific archive lists CPS Supplement files but no separate Replicate Weights heading. Public table release was Apr. 29, 2021; this is after the 2020 election and does not certify exact microdata first availability. |
| 2022 | November 2022 documentation, printed pp. 7-2–7-3 (PDF pp.113–114): adds PRSUPINT (supplement interview/non-interview) for age 18+ and citizenship codes 1–4; PES1 has the same edited universe, but its printed valid-value list omits -9. PES2 edited universe nevertheless includes PES1=-9; PES2 valid values include -9. | Printed questionnaire pp. 9-1–9-2 (PDF pp.118–119): same S1 Yes -> S5 and No/DK/Refused -> S2 routing; no explicit no-response path. | Overview, printed p.3-1 (PDF p.9): interviews Nov. 13–22, self/proxy allowed; no supplement weight, use PWSSWGT. Current year-specific archive explicitly lists distinct Replicate Weights files. Page revision and May 2, 2023 table-release date do not by themselves establish original raw-file first availability or a historical vintage hash. |

“Edited universe” and “questionnaire routing” are different claims. The former describes which public-use records/values the file documentation assigns to a field. The latter describes instrument routing. Neither source alone shows that PES2 is observed for every PES1=-9 case. The technical documentation says self or proxy responses were allowed; it does not make proxy responses equivalent to direct self-report.

## Replicate weights and cutoff implications

The year-specific Census archive pages presently show:

- 2018: a “CPS Supplement” section with data files; no separate “Replicate Weights” section on the inspected page.
- 2020: likewise, no distinct “Replicate Weights” section on the inspected year page.
- 2022: a distinct “Replicate Weights” section with SAS input statements and data files.

These are statements about what the currently inspected pages list. They do not establish that 2018/2020 replicate weights do not exist in another archive, nor do they establish when any listed item first became public. A general Census replicate-weight archive and instructions are not enough to pin a historical year-specific file and vintage. The technical-document statement “there is no supplement weight; use PWSSWGT” identifies the full-sample person weight for point estimation; it must not be misread as proof about replicate weights.

For cutoff-valid replay, public table release is a lower-bound clue, not raw-file vintage proof: Census announced the 2018 tables Apr. 23, 2019, the 2020 tables Apr. 29, 2021, and the 2022 files/tables May 2, 2023. Thus the 2018 summary product predates the 2020 election and the 2020 summary product predates the 2022 election. But an exact CPS microdata/replicate-weight vintage, file hash, acquisition/release time, codebook, and any transformations still need pinning. The 2020 supplement is post-election for an Oct. 2020 forecast; the 2022 supplement is post-election for an Oct. 2022 forecast. Current page “last revised” timestamps are not original release dates.

The older report at `m01-registration-transition-luna-v1/README.md` (SHA-256 `213674dfc42176e537220bff4e23e15aabc6e521fb4d8ee4f65cc536f8003027`) asserted person replicate weights “for election years including 2018–2024.” This addendum supersedes that blanket assertion. Evidence checked supports the claim for the currently listed 2022 page, but not the all-years assertion; 2018/2020 require separate source/availability verification. No nonexistence claim is made.

## Primary sources

All dates below are Census page publication/revision context as stated by the source; a current page revision is not treated as the original file-release date.

- 2018 technical documentation PDF: https://www2.census.gov/programs-surveys/cps/techdocs/cpsnov18.pdf — overview/data collection printed pp. 3-1–3-2 (PDF pp.9–10); PES1/PES2 layout printed p.7-2 (PDF p.113); questionnaire printed pp.9-1–9-2 (PDF pp.118–119).
- 2020 technical documentation PDF: https://www2.census.gov/programs-surveys/cps/techdocs/cpsnov20.pdf — overview/data collection printed p.3-1 (PDF p.9); PES1/PES2 layout printed pp.7-2–7-3 (PDF pp.113–114); questionnaire printed pp.9-1–9-2 (PDF pp.118–119).
- 2022 technical documentation PDF: https://www2.census.gov/programs-surveys/cps/techdocs/cpsnov22.pdf — overview/data collection printed p.3-1 (PDF p.9); PES1/PES2/PRSUPINT layout printed pp.7-2–7-3 (PDF pp.113–114); questionnaire printed pp.9-1–9-2 (PDF pp.118–119).
- 2018 year-specific CPS Voting and Registration archive: https://www.census.gov/data/datasets/time-series/demo/cps/cps-supp_cps-repwgt/cps-voting.2018.html — current page lists supplement files; no distinct replicate-weight heading; revised Mar. 21, 2025.
- 2020 year-specific CPS Voting and Registration archive: https://www.census.gov/data/datasets/time-series/demo/cps/cps-supp_cps-repwgt/cps-voting.2020.html — current page lists supplement files; no distinct replicate-weight heading; revised Mar. 21, 2025.
- 2022 year-specific CPS Voting and Registration archive: https://www.census.gov/data/datasets/time-series/demo/cps/cps-supp_cps-repwgt/cps-voting.2022.html — distinct replicate-weight section; page revised May 2, 2023.
- Census release of 2018 voting/registration tables, Apr. 23, 2019: https://www.census.gov/newsroom/press-releases/2019/voting-registration.html
- Census release of 2020 voting/registration tables, Apr. 29, 2021: https://www.census.gov/newsroom/press-releases/2021/2020-presidential-election-voting-and-registration-tables-now-available.html
- Census release of 2022 voting/registration tables, May 2, 2023: https://www.census.gov/newsroom/press-releases/2023/2022-voting-registration.html
- Sol independent historical source-gate challenge: `../cps-historical-source-gate-sol-v1/README.md` SHA-256 `669d0555488e30ce157bcca58d227797cf3764726390a0f675572916227ed5f0`; this is a methodological cross-check, not primary Census evidence. Its source pins are `../cps-historical-source-gate-sol-v1/source-pins.json` SHA-256 `90ad73a5489d8b3930583c906d8a3db5a49168ed2fc66621665e0f6ad9929a81`.

## What would close the gate

For each historical training cycle, acquire and hash the actual person microdata, person-weight and replicate-weight files, input statements, documentation, and any released revisions. Establish public-release timing from a dated release record or archived official listing that identifies those exact files. Record the survey universe, item response/skip handling, weights, replicate variance procedure, and state/demographic geography crosswalk. Then reconcile the CPS report with the model's existing participation/report evidence so that the same source observation is not scored twice. Until then, CPS is a plausible future measurement source, not an admitted cutoff-valid registration likelihood.

No microdata acquisition, model fit, owner-store access, or model-data change occurred for this packet.
