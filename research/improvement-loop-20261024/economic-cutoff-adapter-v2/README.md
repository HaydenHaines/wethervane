# CPI selection at the count graph's H14 midnight cutoff

This separate adapter preserves `economic-vintages-v1/` unchanged and reuses its reviewed `parse_release` and `select` functions. It adds one earlier source: the August 2004 CPI release published before the 2004 H14 midnight. It then rebuilds the 14 even-cycle selections using the consuming count graph's exact inclusive cutoff, **00:00:00 UTC at the start of election day minus fourteen calendar days**.

The v1 bundle uses end-of-H14-day (`23:59:59 UTC`). That convention admitted the September 2004 release at 12:30 UTC on October 19, but the count graph's H14 midnight cutoff is October 19 at 00:00 UTC, before that release. The v1 selection remains untouched. The newly acquired [FRASER archived BLS release PDF](https://fraser.stlouisfed.org/files/docs/publications/bls/newsreleases/cpi/cpi_20040916.pdf) is dated September 16, 2004, with its embargo header at 8:30 a.m. EDT (`12:30 UTC`). It reports August 2004 CPI-U: headline **2.7%** and core **1.7%**. Its Table 1 unadjusted rows independently contain the same values (`100.000 ... 2.7 ...` for All items; `78.537 ... 1.7 ...` for All items less food and energy). The v1 parser extracts those two rows and checks the PDF digest, CPI reference month, release date and time header.

At the October 19 midnight cutoff, the latest selected observation is therefore August, released September 16. In v1, the October 19 end-of-day cutoff instead selects September, released at 12:30 UTC on October 19: headline 2.5%, core 2.0%. Exact-time tests establish that including the September observation at 12:29:59 UTC is unavailable, while it becomes eligible at its 12:30:00 UTC release instant. All other 26 selections across the other 13 cycles match v1; only the 2004 headline/core pair changes.

## Integrity and limits

- `acquisition.json` records the exact FRASER URL, successful HTTP response, retrieval timestamp, final URL, content type, byte count and SHA-256. The downloaded archived release is a FRASER copy of a BLS release. Original BLS server bytes are not authenticated.
- `inputs.json` pins v1's parser, rows, selections and receipt. The 30 existing parsed rows retain their v1 data and source metadata; their source files are resolved within the sibling v1 directory. The new August release is stored locally in `raw/`.
- Rebuilds verify the fixed reviewed hashes before importing the v1 parser or reading its rows. They also check every bounded relative path and digest/byte count in the pinned v1 receipt, and refuse a changed v2 input manifest. The rebuild never regenerates `inputs.json` from whatever files happen to be present.
- `rows.json` contains the 30 unchanged v1 rows plus the two parsed August rows. `selections.json` records all 28 selected series/cycles with inclusive midnight cutoffs. `deltas.json` records the only changed selection pairs. `receipt.json` pins local v2 artifacts and the v1 inputs.
- Staleness behavior and all other parser rules are inherited unchanged from v1. No unsupported fallback is introduced, and no model data, fitted values or source files outside this v2 directory are edited.
- This is a date-selection correction for historical source intake. It makes no claim about an economic coefficient, forecast quality, a causal effect or any 2026 election outcome.

## Reproduce

From this directory, using Python 3 and the existing `pdftotext` executable:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 adapter.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_adapter.py
```

The tests also corrupt copied v1 parser, row, receipt, and raw-PDF fixtures and check that rebuild stops before changing any v2 outputs.
