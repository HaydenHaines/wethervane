# M01 earlier-as-of source-vintage lead: Fox News Arizona 2018

**Scope.** Read-only provenance research, based on the continuation-v48 checkpoint and existing `m01-calibration-source-provenance-sol-v1` report/reviews. This packet adds one primary-source candidate and compares it to the actual retained Tierney CSV rows. It does not admit rows into a source bank and makes no fit or predictive-benefit claim.

## Finding

Fox News published an Arizona Senate poll story on **October 3, 2018 at 6:00pm EDT**, before the November 6, 2018 election (34 days earlier). The contemporaneous story reports Sinema 47% to McSally 45%. The linked official topline states that results were for release after 6pm ET Wednesday, October 3, and that fieldwork was September 29–October 2 among 806 registered voters, with a 716 likely-voter subsample. This is direct evidence that a Fox Arizona Senate poll release was available before the election cutoff.

It does **not** establish the vintage or source values for any existing Tierney CSV row. The best seeming row candidate has `endDate=2018-09-28`, but the October 3 Fox source says the fieldwork was September 29–October 2. Its source sample is 806 RV / 716 LV while the Tierney row says `numberOfRespondents=658`; and the Fox topline reports 47/45 for likely voters (45/44 RV), whereas Tierney's stored D/R values 336/322 normalize to 51.06/48.94. Therefore this is a source-lineage lead and a concrete linkage discrepancy, not a matched poll or qualified historical row. Two other Fox AZ 2018 rows in the same file end October 26 and September 7 and are also not supported by this October release.

The source is still useful for the gap: primary publisher release and dated publication evidence are feasible to acquire, but source matching must be exact and must not be inferred from pollster/state/year alone. The later Fox archive PDF URL includes a 2020 upload path; its own release legend and Fox's dated October 3 story are the timing evidence. The current live publisher pages can be edited, so these downloaded bytes are hashed snapshots, not independent proof of web-server capture time in 2018. The article's explicit publication timestamp and topline's release instruction support the claimed historical availability; no Wayback capture or third-party timestamp was used.

## Existing-row comparison

Source file: `research/improvement-loop-20261024/momentum-calibration-preparation-sol-v1/snapshots/20-tierney-polls_cnn_dataset.csv`, SHA-256 `61e1141f5ed84cfbbec487115742f59c6e20ebd04ba99a0a32f08d8fa2b20322` (same exact snapshot hash as the previous provenance audit). CSV physical line numbers include the header.

| Physical line | Existing project row | Comparison to October 3 Fox source |
|---:|---|---|
| 4903 | AZ, 2018, Fox News/Beacon Research/Shaw & Co.; end 2018-09-28; N=658; D=336; R=322 | Field dates do not match (Fox Sep 29–Oct 2); N and candidate split do not match. Not linked. |
| 4788 | AZ, 2018, same pollster; end 2018-10-26; N=590; D=295; R=295 | Field dates and values do not match. Not linked. |
| 4974 | AZ, 2018, same pollster; end 2018-09-07; N=645; D=333; R=312 | Field dates and values do not match. Not linked. |

The rows appear in the event-disjoint source snapshot (5465 rows), not a primary release archive. Tierney's provenance report says publication date, historical version and strict-as-of checks remain false for every selected row. This packet leaves those flags untouched.

## Primary sources and preserved bytes

1. Fox News, Dana Blanton, [“Fox News Poll: Republicans make gains in top Senate races”](https://www.foxnews.com/politics/fox-news-poll-republicans-make-gains-in-top-senate-races), published October 3, 2018, 6:00pm EDT. The page gives the Arizona 47–45 result and links the complete poll results. Local retrieved HTML: `raw/fox_article.html`, SHA-256 `42e617394a05751e35ab8983153b33d3490acae79254db7df54bb1808701436c` (510,116 bytes).
2. Fox News official Arizona topline, [PDF](https://static.foxnews.com/foxnews.com/content/uploads/2020/07/Fox_Arizona_October-2018_Complete_Topline_October-3-Release.pdf). Cover states release after 6pm ET Oct 3, field dates Sep 29–Oct 2, 806 RV and 716 LV. Local bytes: `raw/fox_poll_topline.pdf`, SHA-256 `2801ec8f5d059b09df4dc027f1d862b617b6aa0501c44bdb64f1b93e27c41693` (180,492 bytes).
3. The current [Fox Poll Archive](https://www.foxnews.com/official-polls/fox-news-poll-archive) confirms Fox retains official topline/crosstab PDFs, but its present page is not itself contemporaneous availability evidence for this 2018 item.

## Next experiment

Take the exact candidate row(s) from the already reviewed M01 selection masks and search Fox's 2018 primary articles/toplines for matching field dates, sample universe/N and response distribution. For this AZ row, begin with the **September 12, 2018** Fox article/topline associated with the stated September 8–11 comparator in the Oct 3 Fox story, then resolve the one-day `endDate` discrepancy against the actual upstream Chen/Garnett/Montgomery source and Tierney transformation. Preserve any mismatch. Only after exact row lineage is shown should the dated release and relevant original value be attached to that row's cutoff ledger. Do not extrapolate this single case to other rows or folds.
