# Archived CPI releases and cutoff selection

This package acquired 15 BLS CPI release PDFs preserved by the Federal Reserve Bank of St. Louis's FRASER archive, covering all 14 even election cycles from 1998 through 2024. There are 30 parsed headline/core rows and 28 available H14 selections. No fitted model or current-series substitution is included.

Direct BLS download requests returned HTTP 403 for all 15 releases; those failures remain in `acquisition.json`. FRASER returned actual downloadable PDF bytes, preserved in `raw/` with URL, SHA-256, byte count and retrieval time. These are **archived BLS release copies**, not authenticated original bytes obtained from the BLS server. The original BLS archive URL remains attached separately. Their displayed embargo timestamps establish nominal release timing; this package does not prove that no later file correction occurred at either host. Current FRED/API series were not used.

Source discovery: [BLS release archive index](https://www.bls.gov/bls/news-release/cpi.htm), [FRASER CPI collection](https://fraser.stlouisfed.org/title/consumer-price-index-6838). For example, the [September 2024 archive item](https://fraser.stlouisfed.org/title/consumer-price-index-6838/consumer-price-index-september-2024-678400) corresponds to the acquired [October 10 BLS PDF copy](https://fraser.stlouisfed.org/files/docs/publications/bls/newsreleases/cpi/cpi_20241010.pdf). The retained 2010s collection HTML is supplementary discovery evidence, not the source of numerical rows. Its filename is explicit; its numerical contents are not admitted as CPI observations.

`parse_release` verifies the acquired source digest, BLS/CPI-U identity, reference-month title and actual 08:30 Eastern embargo header. It extracts Table 1's **unadjusted percent change from the same month one year earlier**, supporting two explicitly checked historical column layouts. The published values are rounded to one decimal; these are percentages, not fractions, monthly annualized inflation or index levels. The parser preserves Table 1 row text and column layout for audit. Headline is CUUR0000SA0; core is CUUR0000SA0L1E. Geography is U.S. city average and population is CPI-U. No state or subgroup inflation estimate is inferred.

| Cycle | Reference month at H14 | Release date, 08:30 Eastern | Headline % | Core % |
| --- | --- | --- | ---: | ---: |
| 1998 | September | October 16 | 1.5 | 2.5 |
| 2000 | September | October 18 | 3.5 | 2.6 |
| 2002 | September | October 18 | 1.5 | 2.2 |
| 2004 | September | October 19 | 2.5 | 2.0 |
| 2006 | September | October 18 | 2.1 | 2.9 |
| 2008 | September | October 16 | 4.9 | 2.5 |
| 2010 | September | October 15 | 1.1 | 0.8 |
| 2012 | September | October 16 | 2.0 | 2.0 |
| 2014 | **August** | **September 17** | 1.7 | 1.7 |
| 2016 | September | October 18 | 1.5 | 2.2 |
| 2018 | September | October 11 | 2.3 | 2.2 |
| 2020 | September | October 13 | 1.4 | 1.7 |
| 2022 | September | October 13 | 8.2 | 6.6 |
| 2024 | September | October 10 | 2.4 | 3.3 |

The explicit default H14 convention is **23:59:59 UTC on election day minus fourteen calendar days**. It is not fourteen exact days before a local election close. The reusable selector requires an explicit UTC cutoff and validates its calendar-day relation to the election date/horizon. For 2004, the September release occurred on H14 itself: selection before 12:30 UTC returns unavailable with this bounded release inventory; at 12:30 UTC it becomes available. An earlier August release was not acquired for that within-day sensitivity, so the selector does not fabricate one.

In 2014, September CPI was released October 22, **after H14 October 21**. The package deliberately also acquired that future release and proves it cannot replace August at H14. Missing and unavailable states have `value: null`, never zero. The selector refuses stale releases more than 90 days old rather than falling back across years; the package is a bounded election-window inventory, not an exhaustive monthly archive or proof of all intervening release contents.

Revision policy is `as_published_in_dated_release`: a release may already correct older observations. The document-level `document_revision_notice_present` flag does not assert a revision to either selected NSA observation. The 2016 PDF reports prescription-drug errata, including separately identified seasonally adjusted corrections for earlier months; the 2000 material contains revised prior indexes. Do not claim NSA observations are never revised. A supplied revision must have its own vintage timestamp; the selector excludes revisions dated after cutoff and rejects conflicting observations at the same series/reference/vintage. Seasonally adjusted rows and requests are unsupported and rejected: they require a separately reviewed seasonal-factor vintage policy. No present-day seasonal history is spliced into these rows. [BLS seasonal-adjustment documentation](https://www.bls.gov/cpi/seasonal-adjustment/) explains the separate five-year SA revision process.

`rows.json` retains source receipts and parsed rows; `selections.json` retains every cycle/kind cutoff decision; `receipt.json` hashes source PDFs, acquisition metadata, parser/acquisition code and derived outputs. `build-output.json` is the concise execution summary. The selector operates on caller-supplied validated rows: it is not a scientific data-admission authority or a mechanism for authenticating arbitrary JSON claims. No economic coefficient, sign or 2026 forecast follows from this intake.

Reproduce offline from the parent directory (Python standard library plus installed `pdftotext`):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 economic-vintages-v1/cpi_vintages.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s economic-vintages-v1 -p 'test_*.py' -v
```

Explicit network acquisition/retry is separate:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 economic-vintages-v1/acquire.py
```

Existing source files are not overwritten; previously pinned cached bytes must still match their digest. On a subsequent acquisition run, the prior acquisition report is appended to `acquisition-history.jsonl` before the current report is replaced. Fetch failures are recorded rather than replaced with current-series values. Source PDFs are U.S. government statistical releases reproduced by FRASER; the package adds its own extraction and qualification labels.
