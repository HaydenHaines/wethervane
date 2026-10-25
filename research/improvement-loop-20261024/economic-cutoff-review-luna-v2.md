# Independent review: CPI H14-midnight adapter v2

**Review date:** 2026-10-25  
**Scope:** Read-only review of the retained August 2004 FRASER/BLS PDF, adapter, selected vintages, receipt pins, and tests. No fit or producer edit.

## Source row and 2004 cutoff

The retained 83,331-byte PDF is pinned at SHA-256 `b3d6b64f4e05fec3c842f2541b636e970b41c83bbfd0b18304d75850b50de357`. Its title/header identifies “Consumer Price Index: August 2004,” Thursday, September 16, 2004, and says transmission is embargoed until 8:30 a.m. EDT. On that date EDT is UTC−4, so the header time is **2004-09-16 12:30 UTC**. This is a FRASER archived BLS release copy; original BLS server bytes are not authenticated.

I inspected the PDF's Table 1 using `pdftotext -layout`. The unadjusted section prints `All items` with relative importance `100.000` and 12-month unadjusted percent change `2.7`; `All items less food and energy` prints relative importance `78.537` and 12-month unadjusted change `1.7`. The parser's two extracted row strings and parsed values match those cells. This is NSA CPI-U 12-month percent change, not a seasonally adjusted monthly change or an index level.

The election-date-minus-14-day cutoff for 2004 is **2004-10-19 00:00 UTC**. The v1 September release timestamp (`2004-10-19 12:30 UTC`) is 12.5 hours after this inclusive midnight and is correctly unavailable. The new August observation, released at 12:30 UTC on September 16, is selected: headline 2.7%, core 1.7%. Staleness uses the inherited v1 maximum age of 90 days; the August release is within it. Boundary tests leave August selected at 12:29:59 UTC on October 19 and allow September at its exact 12:30:00 UTC release time.

## Selection replay and compatibility

The adapter yields 32 rows (30 v1 rows plus the two August rows) and 28 available selections (14 even cycles, headline and core). I independently recomputed each H14 date at 00:00 UTC and selected the latest eligible reference-month/vintage per series, applying the recorded 90-day age rule. All 28 selections agree. Relative to v1 end-of-day selection, exactly two values change: 2004 headline `2004-09 / 2.5` → `2004-08 / 2.7`, and core `2004-09 / 2.0` → `2004-08 / 1.7`. The 2018, 2020, 2022, and 2024 A/B pairs remain unchanged (headline/core: 2.3/2.2, 1.4/1.7, 8.2/6.6, 2.4/3.3).

The code reuses the reviewed v1 parser and selector, preserving NSA units, release timestamp rules, source metadata, duplicate-vintage handling, same-month vintage ordering, and staleness policy. The 30 inherited rows are byte-for-byte equal as JSON objects to the v1 rows. The `inputs.json` hashes for the current v1 parser, rows, selections and receipt all match their files. All 20 files currently listed by the inherited v1 receipt, including raw archived PDFs, also match its recorded hashes and byte counts. All 10 local v2 receipt entries match their SHA-256 and sizes. No current packet pin mismatch was found.

## Rebuild guard finding

There is one material reproducibility guard gap in the adapter implementation, though no drift is present in the packet I reviewed. `adapter.py` imports the current sibling v1 parser at module import, before checking any trusted identity. Its `input_manifest()` computes hashes from whatever v1 files exist at rebuild time, then `build()` writes those freshly computed hashes back to `inputs.json`. The tests then verify files against that regenerated manifest. Therefore, if the sibling v1 parser, rows, selections, or receipt were changed before a later rebuild, the adapter could silently accept and re-pin the changed inputs rather than fail against this reviewed v2 baseline. The existing test checks self-consistency after build, not drift from a fixed baseline. The v1 receipt contains hashes for 20 source files, but v2 does not walk and validate those nested receipt pins during rebuild/test.

For a durable rebuild guard, define fixed expected v1 SHA-256 values from the reviewed v2 `inputs.json` and check them before importing the v1 parser or reading source rows. Then validate every `source_files` entry in the already pinned v1 receipt for existence, digest, and byte count. Add focused tests that point the validator at altered parser/row/receipt/raw-source bytes and assert rejection. This is a rebuild-hardening issue; the current v2 result and its direct/nested source pins are consistent.

## Tests and scope

All five tests in `economic-cutoff-adapter-v2/test_adapter.py` pass. They cover exact-midnight cutoff shapes and A/B values, new PDF extraction/reference/release time, exclusion and inclusion boundaries for the later September release, changed-selection inventory, and current v1/v2 source pins. This is a useful focused suite. It does not currently test drift against fixed v1 identities or nested v1 source-file hashes, as described above.

This adapter corrects historical source-date selection only. It does not estimate an economic coefficient, establish predictive value, or qualify a forecast. The v1 provenance boundary also remains: the historical release PDF is a FRASER copy, not authenticated original BLS server bytes.
