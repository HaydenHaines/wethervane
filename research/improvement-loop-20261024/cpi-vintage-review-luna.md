# Independent read-only review: CPI vintage intake

Reviewed `/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/economic-vintages-v1/{acquisition.json,rows.json,selections.json,cpi_vintages.py,test_cpi_vintages.py,receipt.json}` and read the 15 local FRASER archived BLS release PDFs. I did not alter or rebuild owner artifacts. All six included unit tests pass.

## Substantive checks

- **All 28 selected numbers check against the release PDFs.** I independently read each PDF’s first-page release narrative for headline CPI-U and its Table 1 `All items less food and energy` row for core, then compared the reported values with the selected output. Both series match for every selected cycle:

  | Cycle | Reference month | Headline | Core |
  | ---: | :--- | ---: | ---: |
  | 1998 | 1998-09 | 1.5 | 2.5 |
  | 2000 | 2000-09 | 3.5 | 2.6 |
  | 2002 | 2002-09 | 1.5 | 2.2 |
  | 2004 | 2004-09 | 2.5 | 2.0 |
  | 2006 | 2006-09 | 2.1 | 2.9 |
  | 2008 | 2008-09 | 4.9 | 2.5 |
  | 2010 | 2010-09 | 1.1 | 0.8 |
  | 2012 | 2012-09 | 2.0 | 2.0 |
  | 2014 | 2014-08 | 1.7 | 1.7 |
  | 2016 | 2016-09 | 1.5 | 2.2 |
  | 2018 | 2018-09 | 2.3 | 2.2 |
  | 2020 | 2020-09 | 1.4 | 1.7 |
  | 2022 | 2022-09 | 8.2 | 6.6 |
  | 2024 | 2024-09 | 2.4 | 3.3 |

  The series labels and selected values are the unadjusted 12-month percentage change for U.S. city average CPI-U; the parser does not substitute the one-month seasonally adjusted headline. For example, the [2024 BLS release](https://www.bls.gov/news.release/archives/cpi_10102024.htm) reports headline 2.4 and core 3.3 for the 12 months ending September. The same source documents the column’s NSA interpretation.

- **2014 cutoff negative case is handled correctly.** The [September 2014 BLS release](https://www.bls.gov/news.release/archives/cpi_10222014.htm) is dated Wednesday, October 22, embargoed to 8:30 a.m. EDT, and reports September headline/core at 1.7/1.7. It is after the default Oct 21 end-of-day UTC H14 cutoff, so the selected observation correctly remains August (released Sep 17). The tests separately show September is excluded at H14 and is admitted at the exact Oct 22 12:30 UTC release instant, with the horizon updated to 13 days.

- **2004 same-day cutoff fixture is useful and passes.** The September 2004 release header is Tuesday October 19 at 8:30 a.m. EDT (12:30 UTC). With a cutoff of 12:29:59 UTC the observation is unavailable; at 12:30 it becomes eligible. The selector therefore uses release timestamp rather than assuming a release-day date is available from the start of that date.

- **Cutoff convention must be matched to the consuming API.** `cutoff_for()` means 23:59:59 UTC at the end of election-minus-14 calendar days. That is a clear convention, and it allows the same-day 2004 release. It is not equivalent to an owner API using 00:00 UTC at the start of H14, local poll-close time, or another exact instant. The intake does not prove those contracts match. Keep it historical-source intake only until the actual consumer passes its exact UTC `as_of` instant through selection and a 2004 regression test demonstrates the agreed behavior. Do not silently treat the 2014 Aug/Sept choice or 2004 inclusion as API-neutral.

- **2016 correction notice is surfaced, but its scope should be explicit.** The archived October 18 PDF says prescription-drug indexes published for May–August 2016 were incorrect, affecting the U.S. All Items index. The source marks `revision_notice_present=true`, which is accurate as a document-level flag. BLS’s [dated errata notice](https://www.bls.gov/bls/errata/cpi-price-corrections-10182016.htm) says corrections were published October 18 and lists seasonally adjusted (`CUSR...`) affected series for May–August. The selected rows are the NSA CPI-U series (`CUUR...`) and September reference month; the notice alone does not show the selected September NSA 12-month values are corrected. Describe the flag as “release contains a correction notice,” not as proof that selected target cells were affected. Preserve the contemporaneous PDF/vintage either way.

- **Source integrity is properly bounded.** Release dates/8:30 a.m. Eastern are checked against each PDF header; reference month and Table 1 labels are checked; archive PDF bytes are SHA-pinned and all acquisition attempts retained. The receipt correctly says BLS original-server bytes were not authenticated and the PDFs are FRASER archive copies. This is a strong source intake record, not an authenticated proof of the exact bytes served by BLS on election day or a qualified CPI model feature.

## Bounded code issue

`select()` filters stale releases using `(cutoff - released).days > 90`. `timedelta.days` floors fractional days, so an observation 90 days and 23:59 old is still admitted. This does not change any of the current 14-cycle selections (their chosen releases are much newer), but it contradicts a literal maximum 90-day age boundary. Prefer comparing the full duration, e.g. `cutoff - released > timedelta(days=90)`, and add synthetic boundary cases at exactly 90 days and 90 days plus one second. Separately, the numeric token regex does not consume a leading `R` revision marker; for an `R1.5` token it can capture `.5`. None of the 28 selected target cells is R-marked, and the currently observed 2000 R-marked index tokens do not shift the selected 12-month column, so this is a robustness guard rather than a correction to this intake’s current values.

## Test and scope result

`PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_cpi_vintages.py` passes all six tests, including all 28 expected values, 2014 cutoff selection, the 2004 exact-release boundary, future revisions, schema/units/seasonality rejection, and raw digest/header checks. The tests are meaningful for the stated convention. They do not establish equivalence to the owner’s as-of API, authenticate original BLS server bytes, validate the exact 90-day boundary, or qualify a model effect. The intake should remain labeled historical vintage reconstruction until those boundaries are addressed.

### Primary source links

- [BLS 2014 September CPI release](https://www.bls.gov/news.release/archives/cpi_10222014.htm) — exact publication date/time and the negative-case observation.
- [BLS 2014 August CPI release](https://www.bls.gov/news.release/archives/cpi_09172014.htm) — the selected 2014 reference month.
- [BLS 2016 CPI correction notice](https://www.bls.gov/bls/errata/cpi-price-corrections-10182016.htm) — dated scope of the correction.
- [BLS 2024 September CPI release](https://www.bls.gov/news.release/archives/cpi_10102024.htm) — headline/core value and NSA description.
- Each actual FRASER PDF URL and SHA-256 is preserved per row in `economic-vintages-v1/rows.json` and in its `receipt.json`.
