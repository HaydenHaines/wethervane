# 2004 gasoline repair: independent delta review

**Date:** 2026-10-24  
**Scope:** Read-only review of the 2004 repair against `gasoline-twip-review-luna.md`; no intake edits.

## Finding

The prior value error is corrected. The exact pinned October 14, 2004 TWIP HTML reports the national regular-gasoline value as **199.3 cents/gallon** for **10/11/04**. Its retail table places **189.9** in the adjacent Heating Oil cell, and the prose independently says regular gasoline reached 199.3 cents/gallon as of October 11. The archived July 14, 2004 issue reports gasoline at **191.7 cents/gallon** for **07/12/04**. Those normalize to $1.993 and $1.917 per gallon; the observations are 91 days apart, the difference is +$0.076, and the log ratio is `ln(1.993/1.917) = 0.03887957716715028`.

The date exception is narrow and preserved transparently. The pinned official index links “Peak Timing” to the exact `/archive/2004/041014/twipprint.html` issue, listing October 14 and week ending October 8. The article's retained header still says October 14, 2003 / next release October 20, 2003; the 2004 retail table says October 11, 2004 and other tables say October 8, 2004. Calendar alignment corroborates the year: October 14, 2004 was Thursday, October 11 was Monday, and the next indexed 2004 issue was October 20 (Wednesday), while the stale 2003 dates fall Tuesday and Monday. The exception requires the exact source and index digests, URL, header literal/date, admitted issue date, and October 11 observation; parsing the raw page without the explicit exception still fails.

The conservative availability bounds are correct for the date-only evidence: current upper bound **2004-10-15 23:59:59 UTC**, lag upper bound **2004-07-15 23:59:59 UTC**. The 2004 H14 cutoff is **2004-10-19 23:59:59 UTC**, leaving four calendar days after the current bound. The panel keeps the raw header date separate from the admitted index-supported issue date. Current and lag records both retain `original_posting_log_authenticated: false`, and the row retains `strict_original_posting_authenticated: false`. The report does not claim original posting-time authentication or byte-for-byte provenance at original posting.

## Reproduction

All five targeted tests pass via `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_intake.py` in `gasoline-twip-cutoff-v1`, including the new gasoline-versus-heating-oil assertion, exception binding/rejection cases, archive/hash reparse, A/B vintage arithmetic, and legacy-unit behavior. The main receipt pins the repaired intake, panel, evidence, index, and raw reports. I verified those selected pins against the files.

**Delta conclusion:** The 2004 admission now addresses the material concern in my earlier review. I found no remaining value, arithmetic, binding, or availability defect in this repair. It supports an explicit date-corrected historical intake row only; it does not turn archived issue bytes into authenticated original posting logs or establish scientific/forecast qualification.
