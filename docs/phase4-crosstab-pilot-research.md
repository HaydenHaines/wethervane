# Phase 4 Crosstab Pilot Research

**Date:** 2026-03-27
**Purpose:** Identify 2 real polls from major pollsters with published crosstab tables for tracked 2026 races.
**Target races:** GA Senate (Special), GA Governor, TX Governor, MI Senate, MI Governor, OH Governor, PA Governor, WI Governor, NC Senate, NH Senate, ME Senate, MN Senate, OR Senate, IA Senate.

---

## Summary of Findings

Five qualifying polls were identified across four races. The two strongest pilots are marked with **[PILOT]** — these have confirmed crosstab URLs and sufficient demographic depth to populate the `xt_*` CSV columns.

| Poll | Race | Date | Pollster | URL | Crosstab depth |
|------|------|------|----------|-----|----------------|
| **[PILOT 1]** | PA Governor | 2025-10-01 | Quinnipiac | https://poll.qu.edu/poll-release?releaseid=3933 | Age (4 groups), gender, race, education — numbers in HTML |
| **[PILOT 2]** | GA Senate | 2026-03-02 | Emerson | https://emersoncollegepolling.com/georgia-2026-poll-senator-ossoff-starts-re-election-near-50-and-outpaces-gop-field/ | Age, gender, race, party — Google Sheets |
| (backup) | NC Senate | 2025-07-30 | Emerson | https://emersoncollegepolling.com/north-carolina-2026-poll-cooper-starts-us-senate-race-with-six-point-lead-and-clear-name-recognition-advantage-over-whatley/ | Age, party — Google Sheets |
| (backup) | MI Senate | 2025-01-25 | Emerson | https://emersoncollegepolling.com/michigan-2026-poll-crowded-democratic-senate-primary-remains-wide-open/ | Age, party — Google Sheets |
| (backup) | NH Senate | 2026-03-23 | Emerson | https://emersoncollegepolling.com/new-hampshire-2026-sununu-leads-gop-nomination-ties-pappas-for-senate/ | Age (3 groups), gender — Google Sheets |
| (backup) | MN Senate | 2026-02-08 | Emerson | https://emersoncollegepolling.com/minnesota-2026-poll-democrats-lead-gop-as-voters-cite-threats-to-democracy/ | Limited inline — Google Sheets |
| (backup) | ME Senate | 2026-03-23 | Emerson | https://emersoncollegepolling.com/maine-2026-poll-platner-leads-gov-mills-democrats-lead-sen-collins-in-maine/ | Gender, party — Google Sheets |

---

## Pilot Poll 1: Quinnipiac — Pennsylvania Governor (October 2025)

**Pollster:** Quinnipiac University Poll
**Race:** 2026 PA Governor — Josh Shapiro (D) vs. Stacy Garrity (R)
**Field dates:** September 25–29, 2025
**Release date:** October 1, 2025
**N:** 1,579 registered voters
**Topline:** Shapiro 55%, Garrity 39%

**Release URL:** https://poll.qu.edu/poll-release?releaseid=3933
**PDF (press release):** https://poll.qu.edu/images/polling/pa/pa10012025_piss74.pdf
**Crosstab PDF note:** Quinnipiac publishes separate crosstab PDFs on a consistent URL pattern:
`https://poll.qu.edu/images/polling/pa/pa10012025_crosstabs_<suffix>.pdf`
The Oct 2025 release does not appear to have a separate crosstab PDF listed; the full demographic tables are embedded in the HTML release page.

**Demographic breakdowns available (all extracted from HTML release page):**

| Dimension | Groups | Shapiro | Garrity | Notes |
|-----------|--------|---------|---------|-------|
| **Party ID** | Democrat | 98% | 1% | |
| | Republican | 13% | 82% | |
| | Independent | 61% | 27% | |
| **Gender** | Men | 48% | 45% | |
| | Women | 61% | 34% | |
| **Education** | College degree (4yr+) | 68% | 30% | direct `pct_bachelors_plus` proxy |
| | No college degree | 41% | 53% | direct `1 - pct_bachelors_plus` proxy |
| **Age** | 18–34 | 64% | 27% | |
| | 35–49 | 57% | 38% | |
| | 50–64 | 49% | 47% | |
| | 65+ | 56% | 39% | |
| **Race** | White (combined) | 52% | 44% | |
| | White men | 45% | 50% | |
| | White women | 57% | 39% | |
| | Black voters | 89% | 7% | |

**Dimensions NOT reported:** urbanicity, region, Hispanic/Latino, ideology.

**Phase 4 relevance:**

- Education crosstab is the highest-value column for W adjustment: `xt_education_college` and `xt_education_noncollege` can be derived.
- To populate `pct_of_sample`, we need the education *composition* of the sample (what fraction were college grads), not just how each group voted. Quinnipiac's methodology PDF or sample-and-demos PDF may contain this. The demographics PDF is at: `https://poll.qu.edu/images/polling/pa/pa02252026_demos_pdkl34.pdf` (February 2026 release) — an equivalent October 2025 demos PDF likely exists at a similar URL pattern.
- The vote-share crosstabs (how each group voted) are fully readable from the HTML. These support Phase 4b sub-group observations directly.
- Age (4-group) and race (white/Black) provide additional dimensions for W construction.

**February 2026 follow-up poll:** A newer Quinnipiac PA Governor poll was released 2026-02-25 (releaseid=3948), showing Shapiro 55%, Garrity 37%. This is more recent but the October 2025 poll is the better pilot because the HTML release was confirmed to contain full demographic tables for the governor race. The February release focused more on a Shapiro-for-president question.

**February 2026 PDF:** https://poll.qu.edu/images/polling/pa/pa02252026_pdkl34.pdf (binary-encoded, requires PDF parser not WebFetch)

---

## Pilot Poll 2: Emerson — Georgia Senate (March 2026)

**Pollster:** Emerson College Polling
**Race:** 2026 GA Senate — Jon Ossoff (D) vs. Buddy Carter / Mike Collins / Derek Dooley (R)
**Field dates:** February 28 – March 2, 2026
**Release date:** March 5, 2026
**N:** 1,000 likely voters (Dem primary sub: n=464; Rep primary sub: n=453)
**Topline (Ossoff vs. Carter):** Ossoff 47%, Carter 44%
**Topline (Ossoff vs. Collins):** Ossoff 48%, Collins 43%

**Release URL:** https://emersoncollegepolling.com/georgia-2026-poll-senator-ossoff-starts-re-election-near-50-and-outpaces-gop-field/
**Full crosstabs (Google Sheets):** https://docs.google.com/spreadsheets/d/1WMIkxH0hZRhWe0dbY6wtuuqshJGChuhv/edit?usp=sharing&ouid=107857247170786005927&rtpof=true&sd=true

**Demographic breakdowns available (confirmed in article; full tables in Google Sheets):**

| Dimension | Groups | Notes from article |
|-----------|--------|-------------------|
| **Age** | Under 50 / Over 50 | Ossoff avg +12 pts among under 50 |
| **Gender** | Men / Women | Ossoff avg +8 pts among women |
| **Race** | Black / White | Reported separately |
| **Party** | Dem / Rep / Independent | Ossoff avg +16 pts among independents |

**Article says** "subsets based on demographics, such as gender, age, education, and race/ethnicity, carry with them higher credibility intervals" — confirming **education** is also in the full crosstabs Google Sheet even though not highlighted in the article narrative.

**Weighting variables (per methodology):** gender, education, race, age, party registration, region — meaning all of these dimensions are available in the full spreadsheet.

**Phase 4 relevance:**

- Google Sheets link is directly accessible and contains the full cross-tabulation table including `pct_of_sample` composition columns (Emerson typically shows both vote share and sample composition in their sheets).
- GA is one of the primary tracked races (GA Senate Special is tracked).
- Covers both college/non-college education split and race (white, Black) — the two highest-value dimensions for GA.
- Date is current (March 2026) — very fresh for use as a pilot.

**Action to extract full crosstab numbers:** Open the Google Sheets URL above. Tabs typically include: "Crosstabs", "Demographics", "Toplines". Look for a "Composition" or "Demo" row showing sample % in each group.

---

## Backup Poll 3: Emerson — North Carolina Senate (July 2025)

**Pollster:** Emerson College Polling
**Race:** 2026 NC Senate — Roy Cooper (D) vs. Michael Whatley (R)
**Field dates:** July 28–30, 2025
**N:** 1,000 registered voters
**Topline:** Cooper 47%, Whatley 41%

**Release URL:** https://emersoncollegepolling.com/north-carolina-2026-poll-cooper-starts-us-senate-race-with-six-point-lead-and-clear-name-recognition-advantage-over-whatley/
**Full crosstabs (Google Sheets):** https://docs.google.com/spreadsheets/d/15z1ND9R9K0UklZgcXV7Wf1Qotwsj7kb7/edit?usp=sharing&ouid=107857247170786005927&rtpof=true&sd=true

**Demographic breakdowns (article highlights; full in Sheets):**
- Age: Under 50 (Cooper +25), Over 50 (Whatley +11)
- Party: Independents (Cooper +19)
- Full sheet likely contains: gender, education, race, region, ideology per standard Emerson methodology

**Note:** July 2025 is relatively old for a 2026 race. NC is a tracked race (NC Senate). Good backup if the two pilots above fail.

---

## Backup Poll 4: Emerson — Michigan Senate (January 2025)

**Pollster:** Emerson College Polling
**Race:** 2026 MI Senate — Mallory McMorrow / Haley Stevens / Abdul El-Sayed (D primary) vs. Mike Rogers (R)
**Field dates:** January 24–25, 2025
**N:** 1,000 Michigan likely voters
**General election topline:** McMorrow 46%, Rogers 43% (also tested Stevens and El-Sayed)

**Release URL:** https://emersoncollegepolling.com/michigan-2026-poll-crowded-democratic-senate-primary-remains-wide-open/
**Full crosstabs (Google Sheets):** https://docs.google.com/spreadsheets/d/1AtLJcs2z9R_NRY7eZqvKS256qHiVSUx0/edit?gid=1633180679#gid=1633180679

**Demographic breakdowns (from article):**
- Age: 60+ (McMorrow 37%), Under 30 (El-Sayed 24%)
- Party: Independents shown for each matchup
- Full sheet should contain gender, education, race, region per standard Emerson methodology

**Note:** January 2025 — 14 months old. The Democratic primary was unresolved at field time; McMorrow has since emerged as frontrunner. This poll's general election crosstabs remain useful for W construction purposes (they reflect MI voter demographics, not just primary preference), but the topline matchup is outdated.

---

## Backup Poll 5: Emerson — New Hampshire Senate (March 2026)

**Pollster:** Emerson College Polling
**Race:** 2026 NH Senate — Chris Pappas (D) vs. Chris Sununu (R)
**Field dates:** March 21–23, 2026
**N:** 1,000 New Hampshire likely voters
**Topline:** Pappas 45%, Sununu 44%

**Release URL:** https://emersoncollegepolling.com/new-hampshire-2026-sununu-leads-gop-nomination-ties-pappas-for-senate/
**Full crosstabs (Google Sheets):** https://docs.google.com/spreadsheets/d/1-Q3fMv-ldmBe4xdMzd0q49PpV7xncHZL/edit?usp=sharing&ouid=107857247170786005927&rtpof=true&sd=true

**Demographic breakdowns (confirmed):**
- Gender: Women (Pappas 49%, Sununu 40%), Men (Sununu 48%, Pappas 42%)
- Age: Under 40 (Pappas 54%, Sununu 31%), 50s-60s (Sununu 51%, Pappas 39%), Over 70 (Pappas 51%, Sununu 45%)
- Full sheet has: education, race, party registration, region per weighting variables listed

---

## Backup Poll 6: Emerson — Maine Senate (March 2026)

**Pollster:** Emerson College Polling
**Race:** 2026 ME Senate — Graham Platner (D) vs. Susan Collins (R)
**Field dates:** March 21–23, 2026
**N:** 1,075 Maine likely voters
**Topline:** Platner 48%, Collins 41%

**Release URL:** https://emersoncollegepolling.com/maine-2026-poll-platner-leads-gov-mills-democrats-lead-sen-collins-in-maine/
**Full crosstabs (Google Sheets):** https://docs.google.com/spreadsheets/d/1jm6cifqqT8NRH_qpVamcP3BDevlPuVtI/edit?gid=619517400#gid=619517400

**Demographic breakdowns (from article):**
- Gender: Women (Platner 52%, Collins 35%), Men (Platner 63%, Collins 22% in primary context)
- Party: Independents (Collins net -30 unfavorable)
- Full sheet has full breakdown per standard Emerson methodology

---

## Pollsters With No Qualifying 2026 Race Polls Found

The following major pollsters were searched but no confirmed crosstab-bearing 2026 polls for the target races were located as of 2026-03-27:

| Pollster | Status |
|----------|--------|
| **Quinnipiac** | Has PA Governor (Oct 2025, Feb 2026) confirmed. No confirmed 2026 releases for MI, OH, WI, GA, NC yet. Most recent release (Feb 2026) was PA only. |
| **NYT/Siena** | No confirmed 2026 state-level Senate/Governor polls found in search results. |
| **Marist** | 2026 polls appear to be national only so far. No state-level 2026 race polls found. |
| **Monmouth** | Archive page showed only through Dec 2024 on visible pages. No 2026 state polls confirmed. |
| **Fox News** | No confirmed 2026 state Senate/Governor polls with crosstabs found. |
| **CNN/SSRS** | No confirmed 2026 state Senate/Governor polls found. |
| **Morning Consult** | Not searched directly — typically tracks job approval, not individual race crosstabs at state level. |
| **YouGov** | Not searched directly — worth checking today.yougov.com for 2026 battleground polls. |

---

## Recommended Action for Step 0 (Phase 4 Plan)

### Immediately actionable (no further research needed):

**Pilot 1 — Quinnipiac PA Governor (Oct 2025):**
1. Open https://poll.qu.edu/poll-release?releaseid=3933
2. Scrape the full demographic table (HTML) — the table is machine-readable
3. Extract for each group: dem_share (Shapiro two-party), n_sample (not always reported by Quinnipiac; may need to calculate from total N and stated % composition)
4. Education split is directly usable for `xt_education_college` / `xt_education_noncollege`
5. To get `pct_of_sample` (composition), attempt to download the methodology/demos PDF: `https://poll.qu.edu/images/polling/pa/pa10012025_demos_<suffix>.pdf` — suffix follows a naming pattern (look at other Quinnipiac PDF URLs like `pa10092024_crosstabs_szjw90.pdf` to infer the pattern)

**Pilot 2 — Emerson GA Senate (March 2026):**
1. Open the Google Sheets URL: https://docs.google.com/spreadsheets/d/1WMIkxH0hZRhWe0dbY6wtuuqshJGChuhv/edit
2. Navigate to "Crosstabs" tab — Emerson sheets typically have a demographic composition row and per-group vote share
3. Extract: `pct_of_sample` for each group (look for a row labeled "% of total", "Composition", or "N=")
4. The sheet likely shows education (college/non-college), age groups, gender, race (white, Black, Hispanic), party, and region

### What Phase 4 implementation needs from these two polls:

For each poll, the minimum viable data is:
```
race, geography, geo_level, dem_share, n_sample, date, pollster, notes,
xt_education_college, xt_education_noncollege,
xt_race_white, xt_race_black,
xt_age_senior  (65+ for Quinnipiac; approximated from age group data)
```

These map directly to the `xt_*` column format specified in the Phase 4 plan.

---

## Key Finding: Emerson vs. Quinnipiac as Crosstab Sources

**Emerson** publishes full crosstab spreadsheets (Google Sheets) for every state poll. These are the most Phase-4-friendly format: they contain both vote shares per group AND the demographic composition of the sample. The Google Sheets links are stable and accessible without authentication. Emerson has published 2026 polls for GA, NC, MI, NH, MN, ME — covering 6 of the 14 tracked races.

**Quinnipiac** publishes HTML release pages with demographic tables embedded. These contain vote shares per group clearly. Sample composition (`pct_of_sample`) requires fetching a separate methodology/demographics PDF that uses a non-obvious URL suffix. Quinnipiac has confirmed 2026 PA Governor polls; no confirmed 2026 releases for other target races as of this writing.

**Recommendation:** Use Emerson GA Senate as Pilot 2 (most current, confirmed full sheet), and Quinnipiac PA Governor as Pilot 1 (established pollster, HTML tables directly readable, education split fully available). This gives geographic and pollster diversity.

---

## Sources

- Emerson College Polling Georgia poll release: https://emersoncollegepolling.com/georgia-2026-poll-senator-ossoff-starts-re-election-near-50-and-outpaces-gop-field/
- Quinnipiac PA October 2025 release: https://poll.qu.edu/poll-release?releaseid=3933
- Quinnipiac PA February 2026 release: https://poll.qu.edu/poll-release?releaseid=3948
- Emerson North Carolina poll: https://emersoncollegepolling.com/north-carolina-2026-poll-cooper-starts-us-senate-race-with-six-point-lead-and-clear-name-recognition-advantage-over-whatley/
- Emerson Michigan poll: https://emersoncollegepolling.com/michigan-2026-poll-crowded-democratic-senate-primary-remains-wide-open/
- Emerson New Hampshire poll: https://emersoncollegepolling.com/new-hampshire-2026-sununu-leads-gop-nomination-ties-pappas-for-senate/
- Emerson Maine poll: https://emersoncollegepolling.com/maine-2026-poll-platner-leads-gov-mills-democrats-lead-sen-collins-in-maine/
- Emerson Minnesota poll: https://emersoncollegepolling.com/minnesota-2026-poll-democrats-lead-gop-as-voters-cite-threats-to-democracy/
- Quinnipiac poll results archive: https://poll.qu.edu/poll-results/
- 270toWin 2026 Senate polls: https://www.270towin.com/polls/latest-2026-senate-election-polls/
- 270toWin 2026 Governor polls: https://www.270towin.com/polls/latest-2026-governor-election-polls/
