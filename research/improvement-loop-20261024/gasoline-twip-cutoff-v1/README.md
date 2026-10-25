# Pre-cutoff archived TWIP gasoline arm

This is a distinct, older-source panel: the latest dated EIA *This Week in Petroleum* issue preceding each H14 cutoff, with national regular-retail observations read from that issue. It is **not** the latest-Monday Gasoline and Diesel Fuel Update arm in Luna's `gasoline-source-evidence-v1/`. No values from her post-cutoff reports or today's historical gasoline series are substituted.

All twelve even cycles from 2002–2024 have a supported current/13-week pair, including all four A/B cycles. The 2004 row uses an explicit, byte-bound header-year exception following independent review; the contradictory source header remains visible. No models were fit.

| Cycle | Current issue date | Actual observation date | USD/gallon | 13-week observation | USD/gallon | Natural log change |
| --- | --- | --- | ---: | --- | ---: | ---: |
| 2002 | October 17 | October 14 | 1.440 | July 15 | 1.394 | 0.032466 |
| 2004 | October 14, explicit exception | October 11 | 1.993 | July 12 | 1.917 | 0.038880 |
| 2006 | October 18 | October 16 | 2.226 | July 17 | 2.989 | −0.294733 |
| 2008 | October 16 | October 13 | 3.151 | July 14 | 4.113 | −0.266433 |
| 2010 | October 14 | October 11 | 2.819 | July 12 | 2.718 | 0.036486 |
| 2012 | October 17 | October 15 | 3.819 | July 16 | 3.427 | 0.108303 |
| 2014 | October 16 | October 13 | 3.207 | July 14 | 3.635 | −0.125273 |
| 2016 | October 19 | October 17 | 2.257 | July 18 | 2.230 | 0.012035 |
| 2018 | October 17 | October 15 | 2.879 | July 16 | 2.865 | 0.004875 |
| 2020 | October 15 | October 12 | 2.167 | July 13 | 2.195 | −0.012838 |
| 2022 | October 19 | October 17 | 3.871 | July 18 | 4.490 | −0.148340 |
| 2024 | October 17 | October 14 | 3.171 | July 15 | 3.496 | −0.097572 |

All dates in a row are in its election year. Prices come from printed source precision; natural log changes are dimensionless, **not** percent or CPI-relative changes. `panel.json` retains both issue dates, observation dates, cutoff, original units/value, normalized units/value, source URLs/hashes, arithmetic, source arm and method regime.

## Admission and evidence

The acquired [EIA TWIP archive index](https://www.eia.gov/petroleum/weekly/) supplies actual issue links/dates. The code chooses its latest indexed issue strictly before the H14 calendar date and checks the issue date against the downloaded report's own header. The sole reviewed exception is the exact 2004 report/index digest pair documented below; it is never a general year-correction rule. It reads the national retail gasoline row, its explicit observation date and retail units. Modern reports use a `rp_gasoline` row; legacy reports use the identified retail table. Regular grade is checked against source prose/chart labeling, not inferred from spot prices or futures. The package labels the measure `US_regular_retail_gasoline_TWIP`; it does not reconstruct formulations or claim a separately checked all-formulations microdata series.

For the lag, it searches actual indexed issues no more than seven days after the required Monday and admits only a report printing the exact observation date **91 days earlier**. Both source issues must precede cutoff. Reports do not have to share a weekday: holidays affect actual release dates. The 2020/2024 current reports are Thursday issues, and their printed Monday observations are October 12/14 respectively. The [EIA TWIP holiday schedule](https://www.eia.gov/petroleum/weekly/includes/schedule.php) independently lists October 17, 2024 as a Thursday exception. No assumption of an ordinary Wednesday release selects the data.

Cutoffs are explicitly 23:59:59 UTC on election day minus fourteen calendar days, matching the CPI intake convention. Exact posting times are not printed in these reports. `available_upper_bound_utc` is therefore a conservative **end of the following UTC day** bound, never an asserted actual publication timestamp. This extra-day bound still precedes each supported Tuesday cutoff. Same-day archive dates are not admitted without intraday evidence. The source report dates provide stronger pre-cutoff evidence than a later report plus a normal-release-schedule assumption, but today's archived bytes are not original posting logs or a no-revision ledger. `original_posting_log_authenticated` and `strict_original_posting_authenticated` remain false. This package does not qualify a full strict historical forecast.

Primary A/B source examples:

- [October 17, 2018 issue](https://www.eia.gov/petroleum/weekly/archive/2018/181017/includes/analysis_print.php), [July 18, 2018 issue](https://www.eia.gov/petroleum/weekly/archive/2018/180718/includes/analysis_print.php).
- [October 15, 2020 issue](https://www.eia.gov/petroleum/weekly/archive/2020/201015/includes/analysis_print.php), [July 15, 2020 issue](https://www.eia.gov/petroleum/weekly/archive/2020/200715/includes/analysis_print.php).
- [October 19, 2022 issue](https://www.eia.gov/petroleum/weekly/archive/2022/221019/includes/analysis_print.php), [July 20, 2022 issue](https://www.eia.gov/petroleum/weekly/archive/2022/220720/includes/analysis_print.php).
- [October 17, 2024 issue](https://www.eia.gov/petroleum/weekly/archive/2024/241017/includes/hometables_print.php), [July 17, 2024 issue](https://www.eia.gov/petroleum/weekly/archive/2024/240717/includes/hometables_print.php).

The older arm uses observations eight days before the Tuesday cutoff, rather than the one-day-old Monday observations in Luna's proxy arm. These are different feature realizations. An empirical comparison must retain the arm and observation age, not silently treat them as equivalent vintages.

## Historical definitions and 2004 review

Legacy tables print cents per gallon; conversion to dollars uses decimal division by 100 and preserves the original number/unit. Modern tables print dollars. The [May 14, 2018 EIA gasoline sample/method change](https://www.eia.gov/petroleum/gasdiesel/gas_proc-methods.php) separates training regimes. All selected within-year 13-week contrasts are entirely before or after that break; none crosses it. This does not prove identical measurement across years. Diesel methodology changes are not gasoline changes. Unknown units, non-Monday observations, mismatched dates, future sources, invalid prices or nonexact 13-week lags fail closed.

`2004-review-evidence.json` preserves the evidence for the explicit `twip_2004_10_14_stale_header_year_v1` intake exception:

- The official index places *Peak Timing* at **October 14, 2004**, with supply week ending October 8, and links `/archive/2004/041014/twipprint.html`.
- The raw [report](https://www.eia.gov/petroleum/weekly/archive/2004/041014/twipprint.html) says **Released on October 14, 2003**, with next release October 20, 2003. Its retail table explicitly says **10/11/04**, gasoline **199.3 cents/gallon**; **189.9** is the adjacent heating-oil cell; other data tables use 10/08/04.
- October 14, 2004 was Thursday and October 11 was Monday. The header's October 14, 2003 was Tuesday; its next-release October 20, 2003 was Monday, while October 20, 2004 was Wednesday. These independent source/calendar checks support the narrow header-year exception. The parser requires the exact pinned raw HTML SHA, exact pinned official index SHA, exact source URL, stale header literal/date and October 11, 2004 observation before applying it. Default parsing without that explicit exception still rejects the contradiction.
- The independently downloaded [July 14, 2004 TWIP issue](https://www.eia.gov/petroleum/weekly/archive/2004/040714/twipprint.html) corroborates the lag observation July 12, 2004 at **1.917 dollars/gallon**; it now also supplies the admitted contrast: `ln(1.993/1.917) = 0.038879577167`, absolute change +$0.076/gallon.
- Modern and legacy-path attempts to obtain an official same-week WPSR PDF returned 404; exact attempted URLs/errors remain in `supplementary-acquisition.json`. No alternate PDF was invented. The intake exception relies on the retained official index/table/calendar corroboration and independent review, not an invented WPSR PDF.

The raw conflicting report remains unchanged. The panel records the source-header issue date 2003-10-14 separately from the admitted 2004-10-14 issue date and includes the complete exception object. Original posting authentication stays false. The earlier manually copied review value 189.9 was an error: a regression now checks the gasoline and adjacent heating-oil cells independently before admitting 1.993. Earlier discovery requests for the ordinary Wednesday 2024 path also returned 404; the actual holiday Thursday path succeeded. Failures and the correct indexed source are kept separately.

## Reproduction and checks

From this directory:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 intake.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_intake.py
```

The script reuses retained source bytes, checks prior cached hashes, and fetches only absent official archived files. It writes only in this directory. `acquisition.json` records requested official URLs, retrieval timestamps, raw HTML SHA-256 and byte counts. `receipt.json` pins code, derived metadata, all retained raw files and supplementary evidence. `build-output.txt` lists supported observations. No current-history endpoint is read.

Five tests pass: actual index/latest-issue selection and hash/reparse checks; all four real A/B source pairs with independently calculated decimal logarithms; the exact 2004 date exception and gasoline-versus-heating-oil columns; future/series/unit/date/price failures; and a synthetic legacy cents/unknown-unit fixture. These are intake/arithmetic checks, not evidence of forecast improvement or scientific activation.
