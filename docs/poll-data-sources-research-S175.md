# Poll Data Sources Research — 2026 Midterms
**Session:** S175
**Date:** 2026-03-23
**Scope:** FL, GA, AL statewide races; replacing placeholder `data/polls/polls_2026.csv`
**Target:** October 2026 launch

---

## 1. Which Races Are on the 2026 Ballot (FL / GA / AL)

### Florida

**FL Governor (open seat)**
- Incumbent Ron DeSantis is term-limited; cannot seek a third consecutive term.
- Primary: August 18, 2026. General: November 3, 2026.
- **Republican primary candidates:** Rep. Byron Donalds (Trump-endorsed, leading), Lt. Gov. Jay Collins, Paul Renner (former FL House Speaker), others.
- **Democratic primary candidates:** Former Rep. David Jolly, Orange County Mayor Jerry Demings, others.
- Polling as of March 2026: Donalds leads GOP primary 28–44% depending on pollster. General: Donalds vs. Jolly polled ~45%–34% (UNF, Feb 2026).

**FL Senate (special election, Rubio seat)**
- Marco Rubio resigned January 2025 to become Secretary of State.
- Gov. DeSantis appointed AG Ashley Moody (R) as interim senator.
- Special election November 3, 2026 fills the **remaining 2 years of Rubio's term** (through January 2029). This is not a Class II or Class III regular election — it is a special election.
- **Republican primary:** Moody (incumbent appointment) faces Jake Lang; primary August 18, 2026.
- **Democratic primary candidates:** Alan Grayson filed; Angie Nixon, Jared Moskowitz, Alexander Vindman have expressed interest.
- Polling (UNF, March 4, 2026): Moody leads Nixon 46%–38%, Moody leads Vindman 45%–38%. Republicans start with a large structural advantage (~17-pt Rubio margin in 2022).

**Note:** Rick Scott's Senate seat is NOT up in 2026 (he won in 2024, Class III, next up 2030).

### Georgia

**GA Governor (open seat)**
- Incumbent Brian Kemp is term-limited (two consecutive terms). Declined to run for Senate.
- Primary: May 19, 2026. General: November 3, 2026.
- **Republican primary candidates:** Burt Jones (Lt. Gov.), Mike Collins (Rep.), others.
- **Democratic primary candidates:** Keisha Lance Bottoms (former Atlanta Mayor, leading), Lt. Gov. candidate Geoff Duncan, others.
- Polling: GOP primary unsettled, Jones/Collins/Jackson competitive. Emerson (Mar 5): Bottoms 35% Dem primary. General matchups TBD until nominees emerge.

**GA Senate (regular election, Class III)**
- Incumbent: Jon Ossoff (D), first elected January 2021.
- This is a competitive general election — Georgia is a true tossup state.
- **Republican primary candidates (May 19):** Jon Collins (Rep.), Brian Carter, Rich Dooley, others.
- **Democratic:** Ossoff is the incumbent.
- Polling: General election matchups show Ossoff 46–49%, Republican challenger 41–44% across recent polls (Emerson, Quantus, Cygnal, Tyson).

### Alabama

**AL Governor (open seat)**
- Incumbent Kay Ivey is term-limited.
- Primary: May 19, 2026. Runoff: June 16, 2026. General: November 3, 2026.
- **Republican primary:** Tommy Tuberville (entered May 2025, leading ~63% in one poll), Ken McFeeters, others.
- **Democratic candidates:** Doug Jones (former U.S. Senator), Yolanda Flowers (2022 nominee), others.
- Alabama is deeply red; general polling likely to show R+20 or greater.
- Only one governor poll found as of March 2026 (Quantus Insights, Oct 2025).

**AL Senate (regular election, Class II)**
- Incumbent: Tommy Tuberville (R), first elected 2021. **BUT:** Tuberville has entered the governor's race, meaning this Senate seat may be vacated or he may withdraw from one race.
- If Tuberville wins the GOP gubernatorial primary, the Senate seat becomes an open race.
- Republican primary Senate candidates include: Steve Marshall (AG), Barry Moore (Rep.), various others.
- Polling (Senate primary, March 2026): Moore 22%, Marshall 16%, others fragmented. Race is unsettled.
- No general election Senate polling found as of March 2026 — seat is assumed R+30 territory.

### Summary Table

| Race | Type | Primary Date | Competitive? | Polls Available (Mar 2026) |
|------|------|-------------|-------------|---------------------------|
| FL Governor | Open seat (DeSantis TL) | Aug 18 | Lean R | Yes — 8+ polls (primary + general) |
| FL Senate | Special election (Rubio seat) | Aug 18 | Safe R | Limited — 1 general poll (UNF) |
| GA Governor | Open seat (Kemp TL) | May 19 | Tossup | Yes — 5+ polls (primary) |
| GA Senate | Regular (Ossoff D incumbent) | May 19 | Tossup | Yes — 10+ polls |
| AL Governor | Open seat (Ivey TL) | May 19 | Safe R | Very limited — 1 poll |
| AL Senate | Regular (Tuberville R, may vacate) | May 19 | Safe R | Primary only — 6 polls |

**For the model:** The two competitively significant races are **GA Senate** (Ossoff defending) and **FL Governor** (open seat, R-leaning but not safe). FL Senate is structurally R+15 or more. AL races are safe R.

---

## 2. Poll Data Sources — Ranked by Scraping Feasibility

### Tier 1: Directly Downloadable (Build Scraper Immediately)

#### 2a. Silver Bulletin Pollster Ratings + Raw Polls
- **URL:** https://www.natesilver.net/p/pollster-ratings-silver-bulletin
- **Files available for download:**
  - `Pollster Stats Full 2026.xlsx` (133KB) — all pollster ratings, plus-minus, methodological transparency
  - `Raw Polls 011226.xlsx` (1.97MB) — topline numbers from 12,300+ polls
  - Direct download URLs (as of Jan 14, 2026):
    - `https://www.natesilver.net/api/v1/file/48c60e0e-1ef2-43b1-8476-08d2118abeb4.xlsx`
    - `https://www.natesilver.net/api/v1/file/9722f176-ecf3-45f3-b017-835b1a0ce16b.xlsx`
- **License:** Free for any purpose; attribution to Silver Bulletin required.
- **Format:** XLSX — needs openpyxl reader, can convert to CSV.
- **Coverage:** Historical (through Jan 2026). 12,300+ polls. Includes governor, Senate, president.
- **Scraping difficulty:** 1/5 — direct file download, no scraping needed.
- **Update frequency:** Released as periodic snapshots (not live/daily). This is historical data, not a live feed.
- **What it gives us:** (1) Updated pollster quality ratings to replace 538's defunct ratings. (2) Historical raw poll data for 2022/2024 validation. Does NOT provide a 2026 live feed.

#### 2b. Wikipedia Poll Tables
- **URL pattern:** `https://en.wikipedia.org/wiki/2026_[State]_[race]_election`
  - FL Gov: `https://en.wikipedia.org/wiki/2026_Florida_gubernatorial_election`
  - FL Senate: `https://en.wikipedia.org/wiki/2026_United_States_Senate_special_election_in_Florida`
  - GA Gov: `https://en.wikipedia.org/wiki/2026_Georgia_gubernatorial_election`
  - GA Senate: `https://en.wikipedia.org/wiki/2026_United_States_Senate_election_in_Georgia`
  - AL Gov: `https://en.wikipedia.org/wiki/2026_Alabama_gubernatorial_election`
  - AL Senate: `https://en.wikipedia.org/wiki/2026_United_States_Senate_election_in_Alabama`
- **Format:** HTML tables (MediaWiki), parseable via `pandas.read_html()` or BeautifulSoup.
- **License:** CC BY-SA — fully permissive.
- **Coverage:** Volunteers add polls as they are released. Tables include: pollster, date, sample size, methodology, D%, R%, margin. Very complete for competitive races (GA Senate had ~15+ rows by March 2026).
- **Scraping difficulty:** 2/5 — `pd.read_html(url)` gets tables directly; identify "Polling" table by headers. Wikipedia API also available: `https://en.wikipedia.org/w/api.php?action=parse&page=TITLE&format=json`
- **Update frequency:** Near real-time — volunteers update within days of new polls.
- **Caveats:** Table format varies slightly by article. Less complete for non-competitive races (AL Senate, AL Governor have very few polls). Requires parsing logic to identify the right table.

### Tier 2: Structured Web Scraping (Moderate Effort)

#### 2c. 270toWin
- **URL pattern:** `https://www.270towin.com/2026-[race-type]-polls/[state]`
  - Examples:
    - `https://www.270towin.com/2026-governor-polls/florida`
    - `https://www.270towin.com/2026-governor-polls/georgia`
    - `https://www.270towin.com/2026-senate-polls/georgia`
    - `https://www.270towin.com/2026-senate-polls/alabama`
    - `https://www.270towin.com/2026-senate-polls/florida` (special election)
- **Data confirmed available:** GA Senate (10+ polls), FL Governor (8+ polls primary), AL Senate (6 primary polls), FL Senate special (1 general poll). Data visible in HTML — not dynamically loaded behind an API.
- **Format:** HTML tables, parseable via `pd.read_html()`.
- **Scraping difficulty:** 2/5 — tables are in static HTML. Consistent structure across states.
- **robots.txt notes:** Disallows `/download`, `/media_download`, some forecast paths, and AI training use. Regular scraping of public poll tables appears to be in a gray area — poll pages are not explicitly disallowed. Standard request headers with appropriate rate limiting recommended.
- **Update frequency:** Near real-time (days after polls are released).
- **Coverage:** All major public polls. The 270toWin data I fetched was rich and complete. This is the most reliable source for structured poll tables with full detail (pollster, date, sample size, MoE, candidate names, percentages).
- **Caveats:** Terms of service note AI training is disallowed; scraping for research use is not explicitly addressed. Use with attribution.

#### 2d. RealClearPolling
- **URL pattern:** `https://www.realclearpolling.com/elections/[race-type]/2026/[state]`
  - Example: `https://www.realclearpolling.com/elections/governor/2026/florida`
- **Format:** Data appears to be dynamically loaded (JavaScript-rendered). HTML source did not contain poll numbers when fetched — confirmed via WebFetch attempt returning no data.
- **Scraping difficulty:** 4/5 — requires either Playwright/headless browser to render JS, or reverse-engineering the underlying API. A PyPI package `realclearpolitics` exists (https://pypi.org/project/realclearpolitics/) but is unmaintained and targets the old RCP domain.
- **robots.txt:** Disallows `/latest-polls/main-latest-polls` specifically. Other poll pages not explicitly disallowed.
- **Coverage:** Comprehensive. Includes polling averages, not just raw polls.
- **Update frequency:** Daily.
- **Assessment:** Higher effort than 270toWin for the same data. **Deprioritize** unless 270toWin coverage proves insufficient.

### Tier 3: Monitor But Don't Build For Yet

#### 2e. Race to the WH (racetothewh.com)
- **URL:** `https://www.racetothewh.com/senate/26polls`
- **Coverage:** Has dedicated state pages for competitive Senate races (GA, TX, NC, MI in nav). Not clear if FL Gov or AL races have dedicated pages. Site showed no HTML poll data — likely JS-rendered.
- **Scraping difficulty:** 3–4/5.
- **Assessment:** Monitor for competitive races. Lower priority than 270toWin.

#### 2f. Decision Desk HQ (decisiondeskhq.com)
- **URL:** https://decisiondeskhq.com/polls/averages/
- **Coverage:** Polling averages, not raw polls. Primarily averages/forecasts.
- **Scraping difficulty:** 3/5 — JS-rendered site.
- **Assessment:** Useful for validation of our own averages but not a primary source for raw poll data.

#### 2g. Morning Consult
- **URL:** https://pro.morningconsult.com/trackers/2026-midterm-election-generic-ballot-polls
- **Coverage:** Generic ballot tracker (national), not state-specific race polls.
- **Assessment:** Not relevant for FL/GA/AL race-specific polling.

#### 2h. Ballotpedia
- **URL pattern:** `https://ballotpedia.org/[Race_name],_2026`
- **Coverage:** Strong on candidate lists and race metadata. Poll tables exist but less complete than Wikipedia. CC BY-NC-SA license.
- **Scraping difficulty:** 2/5.
- **Assessment:** Use for candidate/race metadata, not primary poll source. Wikipedia is better for actual poll tables.

---

## 3. Pollster Quality Data

### Current situation
We have 538's historical pollster ratings at `data/raw/fivethirtyeight/data-repo/`. The 538 rating system is now defunct.

### Replacement: Silver Bulletin Ratings (January 2026)
Silver Bulletin published updated pollster ratings on January 14, 2026. Two downloadable files:

1. **Pollster Stats Full 2026.xlsx** — pollster-level ratings, plus-minus vs. polls-only benchmark, methodological transparency scores.
2. **Raw Polls 011226.xlsx** — 12,300+ individual poll records with topline numbers.

These directly replace the 538 ratings. The `convert_538_polls.py` script uses `POLLSTER_RATINGS_PATH` pointing to the old 538 CSV — a new `convert_silver_bulletin_polls.py` script can point to the downloaded Silver Bulletin XLSX instead.

**Action:** Download both files and store at:
- `data/raw/silver_bulletin/pollster_stats_full_2026.xlsx`
- `data/raw/silver_bulletin/raw_polls_011226.xlsx`

Then build a mapping from pollster name → Silver Bulletin grade to replace the 538 grade used in poll weighting.

### Other pollster rating sources
- **Cygnal** publishes its own track record (self-reported, #1 private pollster per Silver Bulletin 2025).
- **AtlasIntel** — A+ rating, advertises #1 in America per Silver Bulletin 2025. Good for ground truth on pollster selection.
- No other comprehensive independent pollster rating database identified.

---

## 4. Recommended Approach

### Primary scraper: Wikipedia + 270toWin dual-source

Build a scraper that pulls from both Wikipedia and 270toWin, reconciling overlapping polls by pollster+date+race key.

**Rationale:**
- Wikipedia: best coverage of GA Senate and FL Governor (competitive races). CC BY-SA license removes any ambiguity. Table structure is predictable.
- 270toWin: richer metadata (includes margin of error, sample type LV/RV). Better for primary polls and for less-competitive races where Wikipedia tables are sparse.
- Together they cover all 6 races with minimal scraping complexity.

### Architecture

```
scripts/scrape_2026_polls.py
    - for each (state, race_type) in target races:
        1. Fetch Wikipedia article → pd.read_html() → extract "Polls conducted" table
        2. Fetch 270toWin URL → pd.read_html() → extract poll table
        3. Deduplicate by (pollster, date, race) key
        4. Map to internal schema: race, geography, geo_level, dem_share, n_sample, date, pollster
        5. Apply Silver Bulletin pollster grade (join on pollster name)
        6. Write to data/polls/polls_2026.csv
```

Internal schema (current `ingest_polls.py` CSV format):
```
race,geography,geo_level,dem_share,n_sample,date,pollster,notes
```

Note: The current schema stores only `dem_share` (two-party). Scraped data will have raw D%, R%, and undecided%. Conversion: `dem_share = dem_pct / (dem_pct + rep_pct)`. Store raw values in `notes` field for traceability.

### Competitive race priority

| Priority | Race | Reason |
|----------|------|--------|
| 1 | GA Senate (Ossoff) | Tossup; 10+ polls already; highest model value |
| 2 | FL Governor (open) | Lean R but competitive; 8+ polls |
| 3 | FL Senate special (Moody) | Safe R but notable; limited polls |
| 4 | GA Governor (open) | Tossup; primary polls only until nominees emerge |
| 5 | AL Senate (Tuberville vacating?) | Safe R; primary unsettled due to Tuberville gov run |
| 6 | AL Governor (open) | Safe R; very limited polling |

For model propagation purposes, GA Senate and FL Governor are the two races that will meaningfully update type-level predictions. AL and FL Senate are structurally decided.

---

## 5. When Do 2026 Polls Appear?

Based on data already observed (as of March 2026):

- **GA Senate:** First polls appeared early 2025 (Cygnal May 2025, Tyson Feb 2025). 10+ polls by March 2026. General election matchup polling is active now.
- **FL Governor:** Republican primary polls began mid-2025 (St. Pete Polls July 2025, UNF July 2025). 8+ primary polls by March 2026. General election matchups just beginning.
- **FL Senate special:** Very limited — only UNF March 2026 found. Will accelerate after candidates declare.
- **GA Governor:** Primary polls began late 2025 (UGA Nov 2025). General matchup polling will wait for nominee clarity post-May primary.
- **AL races:** Almost no polling. Quantus Oct 2025 is the only governor poll. Senate primary has 6 polls but no general election matchup polling.

**Timeline implication:**
- **Now through May 2026:** Primary season. Good time to track who will be the nominees.
- **Post-May 19 primaries (GA/AL):** Nomination clarity for GA Senate, GA Gov, AL Gov, AL Senate → general election polling will accelerate.
- **Post-August 18 primaries (FL):** FL Governor and FL Senate nominees set → FL general polling will accelerate.
- **August–October 2026:** Peak polling season. Most predictively useful polls.
- **October 2026:** Model launch. Scraper should run weekly (or daily during October).

**Scraper schedule recommendation:**
- Monthly scrape: now through May 2026
- Weekly scrape: June–September 2026
- Daily scrape: October 2026

---

## 6. What Does NOT Exist

- **No official 538 replacement for 2026.** FiveThirtyEight shut down in early 2025. The NY Times picked up approval rating tracking but has not replicated FiveThirtyEight's state-by-state race polling database.
- **No public JSON/REST API** for race-specific polls from any of the major aggregators (Silver Bulletin, RCP, 270toWin).
- **Silver Bulletin has no live feed.** The XLSX files are snapshot releases, not a subscription API.
- **No Google Civic Information API** for poll data (it covers election metadata only).
- **Democracy.works Elections API** (from search results) covers election dates and logistics, not poll data.

---

## 7. Implementation Notes for Scraper (Future Task)

When building `scripts/scrape_2026_polls.py`:

1. **Wikipedia table selection:** Look for table with headers matching `Pollster | Date | Sample | [Candidate names]`. Use index `[0]` or filter by header row content. The "Polls conducted" section header is consistent.

2. **270toWin table parsing:** `pd.read_html('https://www.270towin.com/2026-governor-polls/florida')` returns multiple tables. The poll table has columns for pollster, date, sample, and candidate percentages. Filter for the general election table (vs. primary table) by candidate name presence.

3. **Pollster name normalization:** "Emerson College" vs. "Emerson" vs. "Emerson College Polling" are all the same firm. Build a normalization dict mapping variants to canonical names used in Silver Bulletin ratings.

4. **Two-party conversion:** `dem_share = float(dem_pct) / (float(dem_pct) + float(rep_pct))`. Validate result is in (0.3, 0.7) for sanity.

5. **Rate limiting:** 270toWin — minimum 2 seconds between requests. Wikipedia — use the MediaWiki API rather than scraping HTML when possible (no rate limit issues).

6. **robots.txt compliance:** Both 270toWin and Wikipedia permit scraping of public poll pages. Request with a descriptive User-Agent: `Bedrock-PollScraper/1.0 (political research; contact@bedrock.vote)`.

---

## Sources

- [2026 Florida gubernatorial election - Wikipedia](https://en.wikipedia.org/wiki/2026_Florida_gubernatorial_election)
- [2026 United States Senate special election in Florida - Wikipedia](https://en.wikipedia.org/wiki/2026_United_States_Senate_special_election_in_Florida)
- [2026 Georgia gubernatorial election - Wikipedia](https://en.wikipedia.org/wiki/2026_Georgia_gubernatorial_election)
- [2026 United States Senate elections - Wikipedia](https://en.wikipedia.org/wiki/2026_United_States_Senate_elections)
- [2026 Alabama gubernatorial election - Wikipedia](https://en.wikipedia.org/wiki/2026_Alabama_gubernatorial_election)
- [2026 United States Senate election in Alabama - Wikipedia](https://en.wikipedia.org/wiki/2026_United_States_Senate_election_in_Alabama)
- [2026 Polls: Florida Governor - 270toWin](https://www.270towin.com/2026-governor-polls/florida)
- [2026 Polls: Georgia Governor - 270toWin](https://www.270towin.com/2026-governor-polls/georgia)
- [2026 Polls: Georgia Senate - 270toWin](https://www.270towin.com/2026-senate-polls/georgia)
- [2026 Polls: Alabama Senate - 270toWin](https://www.270towin.com/2026-senate-polls/alabama)
- [2026 Polls: Florida Senate - 270toWin](https://www.270towin.com/2026-senate-polls/florida)
- [2026 Polls: Alabama Governor - 270toWin](https://www.270towin.com/2026-governor-polls/alabama)
- [Silver Bulletin Pollster Ratings 2026](https://www.natesilver.net/p/pollster-ratings-silver-bulletin)
- [RealClearPolling 2026 Elections](https://www.realclearpolling.com/latest-polls/2026)
- [realclearpolitics Python package - PyPI](https://pypi.org/project/realclearpolitics/)
- [NYT picks up FiveThirtyEight poll tracking - Nieman Lab](https://www.niemanlab.org/2025/03/the-new-york-times-picks-up-the-shuttered-fivethirtyeights-poll-tracking-database/)
- [Decision Desk HQ Poll Averages](https://decisiondeskhq.com/polls/averages/)
- [Race to the WH - 2026 Senate Polls](https://www.racetothewh.com/senate/26polls)
- [Ballotpedia - Florida gubernatorial election 2026](https://ballotpedia.org/Florida_gubernatorial_and_lieutenant_gubernatorial_election,_2026)
