# Independent review: pre-cutoff TWIP gasoline vintage

## Checks completed

I read the retained archive index, source pages, panel, intake code, receipt, 2004 evidence and tests. I ran `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_intake.py` in `gasoline-twip-cutoff-v1`; all five tests pass. I independently printed all admitted source observations and recomputed the log contrasts from their retained values.

- The source arm is consistently the dated EIA *This Week in Petroleum* issue, not later current-history data. The selector chooses the latest indexed issue strictly before each H14 date, and the parser checks the issue's own release header against the archive index.
- The 13-week pairs use Mondays exactly 91 days apart. Arithmetic matches: `ln(current / lag13)` using the source-printed precision. Legacy prices in cents are divided by 100; later dollars stay dollars. The 2010 pair appropriately mixes source units (281.9 cents = $2.819 versus $2.718), while the normalized values are both USD/gallon.
- All four requested A/B years pass the dated-source/date/arithmetic checks: 2018 (2.879, 2.865; log change 0.004875), 2020 (2.167, 2.195; −0.012838), 2022 (3.871, 4.490; −0.148340), and 2024 (3.171, 3.496; −0.097572). In each case both observation dates are 13 weeks apart and the issue's next-day UTC availability bound precedes the Tuesday cutoff.
- Older even cycles 2002–2024 have 11/12 current+lag pairs: 2002 and 2006–2024. 2004 alone remains unavailable. No within-pair contrast crosses the stated 2018-05-14 gasoline-method break. The report correctly warns that the break does not establish cross-era measurement equivalence.
- The cutoff convention is Election Day minus 14 calendar days, ending 23:59:59 UTC. Since the reports are dated several days earlier, their conservative end-of-following-UTC-day bound is before the supported cutoffs. That supports dated report availability under the packet's convention; it does not authenticate original posting logs or prove byte-for-byte no revisions. The receipt properly leaves strict posting authentication false.

## 2004: date exception is plausible; the value in the review JSON is wrong

The official EIA index lists **Peak Timing** at October 14, 2004, linked to the retained report, with supply week ending October 8. The same official index lists the next TWIP issue as October 20, 2004. The report's visible page header instead says “Released on October 14, 2003 (Next Release on October 20, 2003).” That is a real contradiction and should remain visible in the raw evidence.

The report's retail table labels the observation **10/11/04**, a Monday. October 14, 2004 was a Thursday, and the listed next issue date October 20, 2004 was a Wednesday. The table also prints other 2004 data (including the 10/08/04 spot observation) and the article is in the official 2004 archive index. By contrast, the stale 2003 header dates fall on Tuesday and Monday respectively. Together, the official index, issue link, 2004 table dates and calendar alignment are strong enough for an **explicit, narrow owner-reviewed issue-date exception** identifying this archived issue as 2004-10-14 despite its stale header. This does not authenticate exact posting time or original bytes at posting; retain `original_posting_log_authenticated=false` and the raw page unchanged. The report date is far enough before the 2004 H14 cutoff that an original intraday time is not needed for this dated-source convention.

However, `2004-review-evidence.json` currently misidentifies **189.9 cents/gallon as gasoline**. In the retained 2004-10-14 report's retail table, gasoline is **199.3 cents/gallon**; **189.9** is the adjacent heating-oil price. I independently parsed the table rows. The July 14, 2004 review-only issue prints gasoline at **191.7 cents/gallon** for 07/12/04. Thus the proposed exact 13-week pair, if the owner approves the date exception, is 1.993 and 1.917 USD/gallon, with `ln(1.993 / 1.917) = 0.038879577167` and absolute change +$0.076/gallon. The 2004 row must not be admitted using the currently recorded 189.9 value. Correct the review evidence and add a regression assertion that distinguishes the gasoline and heating-oil cells before changing status.

The supplied 2004 evidence record is useful for the unresolved date conflict but its value assertion is materially wrong. The separate July file is explicitly review-only and has not been used in a contrast. Five current tests pass, but they do not assert the 2004 gasoline value because the year is intentionally unavailable; add a focused correction before any reviewed admission. I did not change author files or fit a model.

## Source and hash trail

- 2004-10-14 retained report SHA-256: `58543d9c56d405343e4752a47d678a2d5dc522bb124f87d2731ba1af466d2528`.
- 2004-07-14 lag report SHA-256: `a4fd22f6842d7fd108190f0403caf95702e1b6c583ca83af9948aae1ec95f20d`.
- See the URLs and complete packet pins in `gasoline-twip-cutoff-v1/README.md` and `receipt.json`. The review makes no claim of forecasting value or activation readiness.
