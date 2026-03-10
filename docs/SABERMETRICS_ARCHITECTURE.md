# Political Sabermetrics: Architecture Design Document

**Status:** Living document (last updated March 2026)
**Relationship:** Separate silo within the US Political Covariation Model project. Shares data infrastructure with the community-covariance pipeline but has its own compute pipeline, outputs, and use cases.

---

## Table of Contents

1. [Silo Relationship and Shared Infrastructure](#1-silo-relationship-and-shared-infrastructure)
2. [The Stat Framework](#2-the-stat-framework)
3. [Data Pipeline: Double-Duty Sources](#3-data-pipeline-double-duty-sources)
4. [Existing Tools and Datasets to Ingest](#4-existing-tools-and-datasets-to-ingest)
5. [Politician Record Schema](#5-politician-record-schema)
6. [Compute Pipeline](#6-compute-pipeline)
7. [What Requires the Community-Covariance Model vs. What Doesn't](#7-what-requires-the-community-covariance-model-vs-what-doesnt)
8. [Tracking and Update Cadence](#8-tracking-and-update-cadence)
9. [Development Phases](#9-development-phases)

---

## 1. Silo Relationship and Shared Infrastructure

### Two Silos, One Data Lake

```
                        ┌─────────────────────┐
                        │   Shared Data Lake   │
                        │   data/assembled/    │
                        │   data/raw/          │
                        └──────────┬──────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │                             │
          ┌─────────▼─────────┐        ┌──────────▼──────────┐
          │  Community-Covar  │        │  Political Saberm.  │
          │  Pipeline         │        │  Pipeline           │
          │                   │        │                     │
          │  src/assembly/    │        │  src/sabermetrics/  │
          │  src/detection/   │        │                     │
          │  src/covariance/  │◄───────┤  Consumes:          │
          │  src/propagation/ │        │  - Type baselines   │
          │  src/prediction/  │        │  - CTOV residuals   │
          │  src/validation/  │        │  - District comps   │
          │                   │        │                     │
          │  Produces:        │        │  Produces:          │
          │  - Type estimates │        │  - Politician stats │
          │  - Covariance     │        │  - Scouting reports │
          │  - Predictions    │        │  - Fit scores       │
          │  - Shift decomp   │        │  - Talent pipeline  │
          └───────────────────┘        └─────────────────────┘
```

### What Flows Between Silos

| From Community-Covariance | To Sabermetrics | Purpose |
|--------------------------|-----------------|---------|
| Type-level structural baselines | Expected performance calculator | "What should a generic candidate get here?" |
| Community-type weight matrix W | CTOV computation | Decompose candidate residuals by community type |
| District community-type composition | Candidate-district fit scoring | Match candidate skill profiles to district profiles |
| Shift decomposition (persuasion/mobilization/composition) | Attribution engine | What part of a win was the candidate vs. the environment? |

| From Sabermetrics | To Community-Covariance | Purpose |
|-------------------|------------------------|---------|
| Candidate quality estimates | Prior adjustment in propagation model | Better priors for races with known-quality candidates |
| Primary-derived enthusiasm signals | Type-level enthusiasm parameters | Primary turnout by community type informs general election priors |
| Fundraising geography (donor community types) | Candidate appeal validation | Cross-validate CTOV with where money comes from |

### Shared Raw Data

These datasets feed both pipelines and are downloaded once:

| Dataset | Covariance Pipeline Uses | Sabermetrics Pipeline Uses |
|---------|-------------------------|---------------------------|
| MEDSL election returns | Train type-level political parameters | Compute candidate residuals (actual - expected) |
| CES/CCES survey data | MRP opinion estimation | Legislator approval, issue agreement, responsiveness |
| FEC campaign finance | Not used | Fundraising efficiency, donor geography, small-dollar ratio |
| VoteView roll calls | Not used | Ideology, loyalty, strategic defection detection |
| Congressional Record | Not used | Floor speech influence (NLP), frame adoption |
| 538 poll archive | Poll propagation through types | Polling gap as candidate quality signal |
| FL/GA/AL voter files | Turnout modeling, Grimmer-Hersh decomposition | Candidate-specific mobilization measurement |
| State SOS primary returns | Primary-informed candidate adjustment | Primary overperformance, upset detection |

---

## 2. The Stat Framework

### The Core Decomposition

Every stat in this framework derives from one master equation:

```
Actual Result = District Baseline + National Environment + Candidate Effect + Noise
```

| Component | What It Is | How It's Estimated |
|-----------|-----------|-------------------|
| **District Baseline** | Structural partisan lean of the geography | Inside Elections Baseline (most comprehensive), Cook PVI, or our model's type-weighted structural prediction |
| **National Environment** | Tide lifting/sinking all boats | Generic ballot swing (Abramowitz), presidential approval, economic fundamentals |
| **Candidate Effect** | The residual -- this IS the stat | Actual result minus (baseline + environment). Decomposable by community type via the covariance model |
| **Noise** | Irreducible randomness | Estimated from polling error variance; correlated within cycles (Gelman et al.) |

The **Candidate Effect** is the politician's "batting average." Everything else is context.

### Stat Categories

#### Category A: Electoral Stats (from election returns + community model)

| Stat | Formula | Analog | Data Required |
|------|---------|--------|---------------|
| **MVD** (Marginal Vote Delivery) | actual_votes - expected_votes(baseline, environment) | WAR | Election returns, district baseline |
| **CTOV** (Community-Type Overperformance Vector) | residual_by_type_k = actual_type_k - expected_type_k | Spray chart | Election returns + community-type weight matrix W |
| **CEC** (Cross-Election Consistency) | corr(CTOV_t, CTOV_t+1) across elections | Year-to-year batting avg stability | Multiple elections for same candidate |
| **DDL** (Downballot Drag/Lift) | same_ticket_performance - expected_without_candidate | Plus/minus | Same-ticket election returns |
| **PGR** (Primary-to-General Retention) | general_turnout_of_primary_voters / primary_turnout | Postseason performance | Voter file + primary & general returns |
| **Upset Factor** | primary_margin vs expected_primary_margin | Clutch performance | Primary returns + pre-primary polling/fundamentals |

**Existing analogs already computed by others:**
- Inside Elections **Vote Above Replacement (VAR)**: actual vote share - party Baseline. Published for 2022 House races. Median D member: +1.1, median R member: +3.5, best: Mary Peltola (AK-AL) at +17.1.
- Split Ticket **Wins Above Replacement (WAR)**: controls for partisanship, incumbency, demographics, and money. Full historical database maintained.
- Kawai & Sunada (Econometrica, 2025) **candidate valence**: estimates candidate-specific quality controlling for endogenous campaign spending and strategic entry. Found ~9.2pp incumbency valence advantage.

**Our value-add over existing analogs:** CTOV decomposes the scalar residual (VAR/WAR) into a vector by community type. Inside Elections can tell you Peltola overperformed by 17 points. We can tell you she overperformed by +22 in rural-Indigenous types, +18 in urban-professional types, and +8 in rural-working-class types. That's a scouting report, not just a number.

#### Category B: Campaign Stats (from FEC data)

| Stat | Formula | Analog | Data Required |
|------|---------|--------|---------------|
| **SDR** (Small-Dollar Ratio) | unitemized_receipts / total_individual_contributions | Walk rate | FEC committee summary |
| **FER** (Fundraising Efficiency Ratio) | (net_raised - fundraising_costs) / net_raised | Contact rate | FEC disbursements by category |
| **Burn Rate** | total_disbursements / total_receipts (by reporting period) | Plate discipline | FEC quarterly reports |
| **EMR** (Earned Media Ratio) | press_mentions / paid_media_dollars | OBP from walks vs hits | GDELT/Google Trends + FEC ad spending |
| **DGR** (Donor Geography Ratio) | in_district_donations / total_donations | Home/away splits | FEC individual contributions (ZIP → county) |
| **CTDR** (Community-Type Donor Distribution) | donation_share_by_community_type | Revenue by market segment | FEC ZIP → county → community type mapping |

**Geographic disaggregation path:** FEC individual contributions include ZIP_CODE (9-digit). Map via HUD ZIP-to-county crosswalk → county FIPS → community-type weight from matrix W. This produces a donor profile by community type, directly comparable to the candidate's CTOV. When CTDR and CTOV align, the candidate raises money from the communities they perform well in. When they diverge, the candidate has either untapped fundraising potential or unearned financial support.

#### Category C: Legislative Stats (from roll calls + bills + speeches)

| Stat | Formula | Analog | Data Required |
|------|---------|--------|---------------|
| **LES** (Legislative Effectiveness Score) | Volden-Wiseman formula (existing) | Batting average (official) | thelawmakers.org data |
| **ASR** (Amendment Success Rate) | amendments_passed / amendments_offered, weighted by bill significance | Slugging percentage | Congress.gov API |
| **CSN** (Co-Sponsorship Network Centrality) | PageRank / betweenness centrality in co-sponsorship graph | Teammate assists | Congress.gov co-sponsorship data |
| **FSI** (Floor Speech Influence) | novel_ngrams_later_adopted_by_others | Pitch framing (catcher) | Congressional Record NLP (Gentzkow-Shapiro-Taddy dataset) |
| **BDR** (Bipartisan Deviation Rate) | party_defections_on_whipped_votes / total_whipped_votes | Versatility / position flexibility | VoteView roll calls + CQ whip counts |
| **SDL** (Strategic Defection/Loyalty) | defection_rate_on_close_votes vs lopsided_votes | Clutch vs garbage time stats | VoteView roll calls, Snyder-Groseclose methodology |

**Strategic vs. principled defection detection (Snyder & Groseclose 2000):**
Partition all roll calls into "close" votes (margin < 65-35) and "lopsided" votes (margin > 65-35). Lopsided votes have little party pressure and reveal sincere preferences. Close votes reveal party-pressured behavior. A legislator who defects on lopsided votes but falls in line on close votes is engaging in cost-free brand-building. One who defects on close votes is taking real risks. The ratio is itself a stat:

```
SDL = defection_rate_close_votes / defection_rate_lopsided_votes
```

SDL > 1 = principled dissenter (defects more when it matters). SDL < 1 = strategic brand-builder (defects more when it's free). SDL ≈ 1 = consistent regardless of pressure.

#### Category D: Constituent Relationship Stats (from CES + voter files)

| Stat | Formula | Analog | Data Required |
|------|---------|--------|---------------|
| **RI** (Responsiveness Index) | corr(delta_NOMINATE, delta_district_opinion) | Coaching adaptability | VoteView + CES district opinion |
| **CAR** (Constituent Approval Residual) | actual_approval - expected_approval(partisanship, environment) | Fan loyalty metric | CES approve_rep variable |
| **IAR** (Issue Agreement Rate) | CES_perceived_agreement / CES_actual_agreement | Scouting report accuracy | CES issue questions + VoteView votes |
| **CME** (Candidate Mobilization Effect) | turnout_with_candidate - turnout_without_candidate (matched voters) | Player attendance impact | Voter file panel data |

**CES approval as a sabermetric input:** CES pipes the actual incumbent's name into the approval question ("Do you approve of the way [Name] is doing their job?"). The raw approval is confounded by partisanship (a Republican in a D+20 district will always have lower approval than one in an R+20 district). The **Constituent Approval Residual (CAR)** corrects for this:

```
CAR = actual_approval - predicted_approval(district_partisanship, national_environment, party_match)
```

A high CAR means the legislator is more popular than their structural position warrants. This is a genuine personal-brand metric.

**CES also measures perceived issue agreement** (Ansolabehere & Kuriwaki, AJPS 2022). A single roll-call vote in agreement with a constituent's preference increases approval by ~11pp. The **Issue Agreement Rate** compares how often constituents *perceive* agreement to how often the legislator *actually* agrees (based on VoteView votes matched to CES issue questions). When perceived > actual, the legislator is skilled at impression management. When actual > perceived, they have a communications problem.

---

## 3. Data Pipeline: Double-Duty Sources

### Source-by-Source Architecture

#### Election Returns (MEDSL + State SOS)

```
data/raw/medsl/                        # Downloaded once
    ├── county_presidential_2000_2024.csv
    ├── county_house_senate_2000_2024.csv
    └── precinct_returns_2016_2024/
        ├── fl_precinct_*.csv
        ├── ga_precinct_*.csv
        └── al_precinct_*.csv

        ┌──────────────────┐
        │ Election Returns │
        └────────┬─────────┘
                 │
        ┌────────┴────────┐
        │                 │
  ┌─────▼─────┐    ┌─────▼─────┐
  │ Covariance │    │ Saberm.   │
  │ Pipeline   │    │ Pipeline  │
  │            │    │           │
  │ Aggregate  │    │ Compute   │
  │ to type    │    │ expected  │
  │ level      │    │ baseline  │
  │ theta_k    │    │ per dist  │
  │            │    │           │
  │ Train      │    │ Subtract  │
  │ factor     │    │ to get    │
  │ model      │    │ candidate │
  │            │    │ residual  │
  └────────────┘    └───────────┘
```

**The baseline computation** uses three inputs stacked:
1. **Inside Elections Baseline** (when available): averages ALL federal and state elections over 4 cycles (~750 races per district). Most comprehensive single metric. We ingest their published data.
2. **Cook PVI**: 75/25 weighted average of last two presidential elections relative to national. We compute from MEDSL returns directly.
3. **Our model's structural prediction**: the community-covariance model's prior (type-weighted expected result for a generic candidate). This is the model-native baseline and the most granular -- it decomposes by community type.

All three are computed and compared. The candidate residual from baseline #3 is the CTOV.

#### CES/CCES Survey Data

```
data/raw/ces/
    └── ces_cumulative_2006_2024.dta     # DOI: 10.7910/DVN/II2DB6

        ┌──────────────┐
        │   CES Data   │
        └──────┬───────┘
               │
      ┌────────┴────────┐
      │                 │
┌─────▼─────┐    ┌─────▼─────┐
│ Covariance │    │ Saberm.   │
│ Pipeline   │    │ Pipeline  │
│            │    │           │
│ MRP:       │    │ Extract:  │
│ Vote choice│    │ approve_  │
│ by demo ×  │    │ rep/sen   │
│ community  │    │           │
│ type       │    │ Issue     │
│            │    │ agreement │
│ Poststrat  │    │ questions │
│ to type    │    │           │
│ estimates  │    │ Match to  │
│            │    │ VoteView  │
│            │    │ roll calls│
└────────────┘    └───────────┘
```

**Key CES variables for sabermetrics:**
- `approve_rep`, `approve_sen1`, `approve_sen2`: incumbent approval (name piped in)
- `cd`: congressional district (for matching to representatives)
- Issue preference batteries: matched to specific bills that Congress voted on
- `voted_rep`, `voted_sen1`, `voted_sen2`: vote choice
- `vv_turnout_gvm`: validated turnout (matched against voter files)

#### FEC Campaign Finance

```
data/raw/fec/
    ├── candidates_{cycle}.txt           # Candidate master
    ├── committee_candidate_{cycle}.txt  # Committee-candidate linkage
    ├── individual_contributions_{cycle}.txt  # Schedule A (ZIP-level)
    ├── committee_contributions_{cycle}.txt   # Committee-to-candidate
    └── operating_expenditures_{cycle}.txt    # Schedule B

        ┌──────────────┐
        │   FEC Data   │  (Sabermetrics only -- not used by covariance pipeline)
        └──────┬───────┘
               │
        ┌──────▼──────┐
        │  Saberm.    │
        │  Pipeline   │
        │             │
        │  Compute:   │
        │  - SDR      │
        │  - FER      │
        │  - Burn Rate│
        │  - DGR      │
        │             │
        │  Map donors │
        │  ZIP → FIPS │
        │  → community│
        │  type via W │
        └─────────────┘
```

**ZIP-to-community-type mapping:** FEC individual contributions contain 9-digit ZIP. Path: ZIP → county FIPS (HUD USPS crosswalk, updated quarterly) → community-type weights from matrix W. This produces a donor profile vector comparable to the CTOV. Donors over $200 are itemized; unitemized (small-dollar) contributions are only available as committee-level aggregates.

#### VoteView Roll Calls

```
data/raw/voteview/
    ├── HSall_members.csv      # Member ideology per Congress
    ├── HSall_votes.csv        # Individual vote records
    └── HSall_rollcalls.csv    # Roll call metadata

        ┌──────────────────┐
        │ VoteView Data    │  (Sabermetrics only)
        └──────┬───────────┘
               │
        ┌──────▼──────┐
        │  Saberm.    │
        │  Pipeline   │
        │             │
        │  DW-NOMINATE│
        │  + Nokken-  │
        │  Poole per  │
        │  Congress   │
        │             │
        │  Party-line │
        │  vote ID    │
        │  (Snyder-   │
        │  Groseclose)│
        │             │
        │  Strategic  │
        │  defection  │
        │  scoring    │
        └─────────────┘
```

**Nokken-Poole vs. DW-NOMINATE:** DW-NOMINATE assigns one fixed ideal point per legislator per career. Nokken-Poole re-estimates per Congress, allowing detected ideological movement. For sabermetrics, Nokken-Poole is preferred because it captures within-career strategic repositioning -- itself a measurable skill.

#### Congressional Record / Floor Speeches

```
data/raw/congressional_record/
    ├── gentzkow_shapiro_taddy/     # 43rd-114th Congress (1873-2017)
    │   ├── hein-bound/             # Parsed speeches, bound edition
    │   ├── hein-daily/             # Parsed speeches, daily edition
    │   └── phrase_partisanship/    # Bigram partisanship scores
    └── conspeak/                   # 104th-118th (1995-2024, JSON, updated daily)

        ┌────────────────────────┐
        │ Congressional Record   │  (Sabermetrics only)
        └──────────┬─────────────┘
                   │
            ┌──────▼──────┐
            │  Saberm.    │
            │  Pipeline   │
            │             │
            │  NLP:       │
            │  - Frame    │
            │    adoption │
            │    network  │
            │  - Novel    │
            │    ngram    │
            │    tracking │
            │  - Topic    │
            │    evolution│
            │  - Partisan │
            │    language │
            │    scoring  │
            └─────────────┘
```

**Data sources:**
- Gentzkow, Shapiro & Taddy (Stanford): 43rd-114th Congress (1873-2017). Parsed speeches with speaker metadata, bigram frequencies, phrase partisanship scores. Open Data Commons Attribution License.
- ConSpeak: 104th-118th Congress (1995-2024). ~1.4M speeches in JSON, organized by bioguide ID. Updated daily since 2022. Includes AI-based rhetoric assessment.
- Hugging Face (Eugleo): 17.4M speeches, 6.41 GB in Parquet. Loadable via `datasets` library.

#### Polling Data (538 Archive)

```
        ┌──────────────┐
        │  Poll Data   │
        └──────┬───────┘
               │
      ┌────────┴────────┐
      │                 │
┌─────▼─────┐    ┌─────▼─────┐
│ Covariance │    │ Saberm.   │
│ Pipeline   │    │ Pipeline  │
│            │    │           │
│ Propagate  │    │ Compute   │
│ through    │    │ polling   │
│ type       │    │ gap:      │
│ covariance │    │ actual -  │
│            │    │ final     │
│ Update     │    │ polling   │
│ type-level │    │ avg       │
│ estimates  │    │           │
│            │    │ Adjust    │
│            │    │ for cycle │
│            │    │ polling   │
│            │    │ error     │
│            │    │ trend     │
└────────────┘    └───────────┘
```

**Polling gap as candidate quality signal:** A candidate who wins D+7 when polls said D+3 has a raw gap of +4. But if polls nationally underestimated Democrats by +2 that cycle (correlated within-cycle error, per Gelman et al.), the candidate-specific signal is only +2. The adjusted polling gap controls for the cycle-level systematic error:

```
candidate_polling_effect = (actual - polling_avg) - median(actual - polling_avg)_same_cycle
```

---

## 4. Existing Tools and Datasets to Ingest

### Stat Databases to Import

| Source | What It Provides | Format | Update | Access |
|--------|-----------------|--------|--------|--------|
| **thelawmakers.org** (Volden-Wiseman LES) | Legislative Effectiveness Scores, 93rd-118th Congress | Excel, Stata | Per-Congress | Free download |
| **VoteView** | DW-NOMINATE, Nokken-Poole, per-vote records, 1789-present | CSV, JSON | Weekly | Free download |
| **Inside Elections VAR** | Vote Above Replacement for House races | Published articles | Per-cycle | Manual extraction |
| **Split Ticket WAR** | Wins Above Replacement, full historical database | Database/articles | Per-cycle | Public |
| **Congress.gov API** | Bills, co-sponsorships, votes, committees | JSON | Hourly | Free API key, 5K req/hr |
| **FEC bulk data** | Contributions, disbursements, committee summaries | Pipe-delimited text | Daily | Free download |
| **CES cumulative** | Survey responses 2006-2024 | Stata, R | Per-wave | Free, Harvard Dataverse |
| **Gentzkow-Shapiro-Taddy** | Parsed congressional speeches 1873-2017 | Text + metadata | Static | Free, Stanford |
| **ConSpeak** | Congressional speeches 1995-2024 | JSON | Daily | Free |
| **GovTrack Report Cards** | Ideology (SVD), Leadership (PageRank), bills introduced/enacted | Web | Annual | Free |

### APIs to Build Against

| API | Key Endpoints | Rate Limit | Auth |
|-----|--------------|------------|------|
| **Congress.gov** | `/bill/`, `/member/`, `/amendment/`, `/committee-report/` | 5,000/hr | Free API key |
| **OpenFEC** | `/candidates/`, `/schedules/schedule_a/`, `/schedules/schedule_b/` | 1,000/hr | Free API key |
| **LegiScan** | State + federal bills, votes, sponsors | 30,000/mo (free) | Free tier |
| **VoteView** | Bulk CSV download (no API needed) | N/A | None |

**Note:** ProPublica Congress API and OpenSecrets API are both **discontinued**. Congress.gov API and OpenFEC are the replacements.

---

## 5. Politician Record Schema

### The Politician Card

Every politician in the system has a structured record that accumulates stats over their career. Think of this as a baseball card.

```
politician/
    ├── identity/
    │   ├── bioguide_id          # Unique across Congress.gov, GovTrack, VoteView
    │   ├── fec_candidate_id     # Links to FEC data
    │   ├── icpsr_id             # Links to VoteView
    │   ├── name, party, state
    │   └── offices_held[]       # Chronological list of positions
    │
    ├── electoral_stats/         # Per-race records
    │   ├── {race_id}/
    │   │   ├── mvd              # Marginal Vote Delivery
    │   │   ├── ctov[]           # Community-Type Overperformance Vector
    │   │   ├── ddl              # Downballot Drag/Lift
    │   │   ├── polling_gap      # Adjusted polling gap
    │   │   ├── primary_margin   # Primary result
    │   │   ├── upset_factor     # Primary upset magnitude
    │   │   └── pgr              # Primary-to-General Retention
    │   └── career_summary/
    │       ├── mean_mvd
    │       ├── cec              # Cross-Election Consistency
    │       └── ctov_career[]    # Career-average CTOV
    │
    ├── campaign_stats/          # Per-cycle
    │   ├── {cycle}/
    │   │   ├── sdr              # Small-Dollar Ratio
    │   │   ├── fer              # Fundraising Efficiency
    │   │   ├── burn_rate_curve  # Quarterly burn rate trajectory
    │   │   ├── emr              # Earned Media Ratio
    │   │   ├── dgr              # Donor Geography Ratio
    │   │   └── ctdr[]           # Community-Type Donor Distribution
    │   └── career_fundraising_summary
    │
    ├── legislative_stats/       # Per-Congress
    │   ├── {congress_number}/
    │   │   ├── les              # Legislative Effectiveness Score (Volden-Wiseman)
    │   │   ├── les_2            # LES 2.0 (with text incorporation)
    │   │   ├── asr              # Amendment Success Rate
    │   │   ├── csn_pagerank     # Co-Sponsorship PageRank
    │   │   ├── csn_betweenness  # Co-Sponsorship Betweenness
    │   │   ├── csn_cross_party  # Cross-Party Co-Sponsorship Rate
    │   │   ├── nominate_dim1    # DW-NOMINATE 1st dimension
    │   │   ├── nominate_dim2    # DW-NOMINATE 2nd dimension
    │   │   ├── nokken_poole_1   # Nokken-Poole (session-specific)
    │   │   ├── bdr              # Bipartisan Deviation Rate
    │   │   ├── sdl              # Strategic Defection/Loyalty ratio
    │   │   └── fsi              # Floor Speech Influence (when computed)
    │   └── career_legislative_summary
    │
    ├── constituent_stats/       # Per-Congress (when CES data available)
    │   ├── {congress_number}/
    │   │   ├── car              # Constituent Approval Residual
    │   │   ├── iar              # Issue Agreement Rate
    │   │   └── ri               # Responsiveness Index
    │   └── career_constituent_summary
    │
    └── composite_scores/        # Computed summaries
        ├── electoral_composite  # Weighted combination of electoral stats
        ├── legislative_composite
        ├── fit_scores{}         # Fit score per target district/geography
        └── talent_tier          # Tier classification (see below)
```

### Storage

- **Format:** Parquet for tabular stat records, JSON for hierarchical metadata
- **Location:** `data/sabermetrics/`
- **Linkage keys:** bioguide_id (Congress.gov/GovTrack/ConSpeak), icpsr_id (VoteView), fec_candidate_id (FEC)
- **Cross-reference:** `data/sabermetrics/id_crosswalk.parquet` maps between all ID systems

```
data/sabermetrics/
    ├── politicians.parquet          # Master roster with all IDs
    ├── id_crosswalk.parquet         # bioguide ↔ icpsr ↔ fec_id mapping
    ├── electoral_stats.parquet      # One row per candidate × race
    ├── campaign_stats.parquet       # One row per candidate × cycle
    ├── legislative_stats.parquet    # One row per member × Congress
    ├── constituent_stats.parquet    # One row per member × Congress (CES years)
    ├── ctov_vectors.parquet         # One row per candidate × race, K columns for type residuals
    ├── cosponsor_networks/          # Per-Congress network files
    │   └── {congress}_cosponsorship.graphml
    ├── speech_influence/            # NLP outputs
    │   └── {congress}_ngram_adoption.parquet
    └── composite_scores.parquet     # Career summaries and composites
```

---

## 6. Compute Pipeline

### Pipeline Stages

```
[1. Ingest & Link]  →  [2. Baseline]  →  [3. Residuals]  →  [4. Stats]  →  [5. Composites]
    Data download       Compute           Actual - expected   Category-       Career
    ID crosswalk        district          by community        specific        summaries,
    Deduplication       baselines         type                stat            fit scores,
                                                              computation     talent tiers
```

### Stage 1: Ingest & Link

```python
# src/sabermetrics/ingest.py

def download_voteview() -> None:
    """Download HSall_members.csv, HSall_votes.csv, HSall_rollcalls.csv."""

def download_les() -> None:
    """Download LES data from thelawmakers.org."""

def download_fec_bulk(cycles: list[str]) -> None:
    """Download FEC bulk files for specified election cycles."""

def download_ces_cumulative() -> None:
    """Download CES cumulative file from Harvard Dataverse."""

def fetch_congress_api(congress: int) -> None:
    """Fetch bills, co-sponsorships, votes from Congress.gov API."""

def build_id_crosswalk() -> None:
    """Build bioguide ↔ icpsr ↔ fec_id crosswalk.

    Primary source: unitedstates/congress-legislators YAML files
    (maintained by GovTrack community, contains all ID mappings).
    """

def download_congressional_speeches() -> None:
    """Download Gentzkow-Shapiro-Taddy + ConSpeak speech data."""
```

### Stage 2: Baselines

```python
# src/sabermetrics/baselines.py

def compute_cook_pvi(election_returns: pd.DataFrame) -> pd.DataFrame:
    """Compute Cook PVI from last two presidential elections.

    Formula: 75% weight on most recent + 25% on prior,
    relative to national two-party vote share.
    """

def compute_structural_baseline(
    W: np.ndarray,
    type_estimates: np.ndarray,
    national_environment: float,
) -> pd.DataFrame:
    """Compute model-native structural baseline per district.

    Uses community-type weight matrix W and type-level estimates
    from the covariance pipeline. This is the model's prediction
    for a generic candidate in each district.
    """

def compute_national_environment(
    generic_ballot: float,
    presidential_approval: float,
    gdp_growth: float,
) -> float:
    """Estimate national environment from fundamentals.

    Follows Abramowitz: generic ballot is the primary predictor
    (r=.82 with national House popular vote over postwar midterms).
    """
```

### Stage 3: Residuals

```python
# src/sabermetrics/residuals.py

def compute_mvd(
    actual_results: pd.DataFrame,
    baselines: pd.DataFrame,
    environment: float,
) -> pd.DataFrame:
    """Compute Marginal Vote Delivery for each candidate.

    MVD = actual - (baseline + environment_adjustment)
    """

def compute_ctov(
    actual_results_precinct: pd.DataFrame,
    W: np.ndarray,
    type_baselines: np.ndarray,
) -> pd.DataFrame:
    """Compute Community-Type Overperformance Vector.

    Projects precinct/county residuals onto community-type basis.
    Returns one K-length vector per candidate per race.
    """

def compute_polling_gap(
    actual_results: pd.DataFrame,
    final_polling_averages: pd.DataFrame,
    cycle_median_error: float,
) -> pd.DataFrame:
    """Compute cycle-adjusted polling gap.

    candidate_effect = (actual - polling_avg) - cycle_median_error
    """
```

### Stage 4: Category-Specific Stats

```python
# src/sabermetrics/electoral.py    -- MVD, CTOV, CEC, DDL, PGR
# src/sabermetrics/campaign.py     -- SDR, FER, Burn Rate, EMR, DGR, CTDR
# src/sabermetrics/legislative.py  -- LES import, ASR, CSN, BDR, SDL
# src/sabermetrics/constituent.py  -- CAR, IAR, RI (from CES)
# src/sabermetrics/speech.py       -- FSI (NLP on congressional record)
```

### Stage 5: Composites

```python
# src/sabermetrics/composites.py

def compute_career_summary(politician_id: str) -> dict:
    """Aggregate per-race/per-Congress stats into career summaries."""

def compute_fit_score(
    candidate_ctov: np.ndarray,
    district_W: np.ndarray,
) -> float:
    """Candidate-district fit = dot(CTOV, district_type_composition).

    Higher = candidate's strengths align with district's community mix.
    """

def compute_cec(ctov_history: list[np.ndarray]) -> float:
    """Cross-Election Consistency = mean pairwise correlation of CTOVs."""

def rank_candidates_for_district(
    candidate_pool: pd.DataFrame,
    target_district_W: np.ndarray,
) -> pd.DataFrame:
    """Moneyball scouting: rank candidates by fit score for a target district."""
```

---

## 7. What Requires the Community-Covariance Model vs. What Doesn't

### Computable Without the Community-Covariance Model

These stats can be computed immediately from public data, before the covariance pipeline exists:

| Stat | Data Needed | Notes |
|------|-------------|-------|
| MVD (using Cook PVI or Inside Elections Baseline) | MEDSL returns + Cook PVI | Scalar version -- no type decomposition |
| SDR, FER, Burn Rate | FEC bulk data | Pure campaign finance stats |
| LES | thelawmakers.org download | Already computed by Volden-Wiseman |
| DW-NOMINATE, Nokken-Poole | VoteView download | Already computed |
| BDR, SDL | VoteView roll calls | Computable from vote records |
| CSN (co-sponsorship network) | Congress.gov API | Graph construction + centrality |
| CAR | CES approve_rep + district partisanship | Requires simple regression |
| DGR | FEC individual contributions (ZIP-level) | ZIP-to-county mapping |

**This means the sabermetrics silo can start producing useful output before the covariance pipeline is built.** The initial stats are simpler (scalar MVD rather than vector CTOV, Cook PVI baseline rather than model-native baseline) but still useful.

### Requires the Community-Covariance Model

These stats need the type-level decomposition:

| Stat | What It Needs from the Covariance Model |
|------|----------------------------------------|
| **CTOV** | Community-type weight matrix W + type-level structural baselines |
| **CEC** | Multiple CTOVs for the same candidate → requires W |
| **Fit Score** | Candidate CTOV + target district W |
| **CTDR** | Donor ZIP → county → community type mapping via W |
| **Model-native structural baseline** | Type-weighted prediction for generic candidate |
| **CME** (Candidate Mobilization Effect by type) | Voter file turnout decomposed by community type |

---

## 8. Tracking and Update Cadence

### Data Source Update Frequencies

| Data Source | Update Frequency | Trigger |
|-------------|-----------------|---------|
| MEDSL election returns | Per-election (certified results) | After each general/primary election |
| VoteView | Weekly | New roll call votes |
| FEC bulk data | Daily-weekly | New filings |
| CES/CCES | Biennial (even years) | New survey wave release |
| Congress.gov (bills, co-sponsorship) | Hourly | New legislative activity |
| thelawmakers.org (LES) | Per-Congress (every 2 years) | Congress concludes |
| ConSpeak (floor speeches) | Daily | New Congressional Record entries |
| Community-type matrix W | Rarely (per-ACS-vintage or per-Census) | New non-political data |
| 538 poll archive | Continuous during election cycles | New polls published |

### Stat Recomputation Schedule

| Stat Category | When to Recompute | Why |
|---------------|-------------------|-----|
| Electoral stats (MVD, CTOV) | After each election (certified results) | New actual results become available |
| Campaign stats (SDR, FER) | Quarterly (FEC reporting deadlines) | New filings |
| Legislative stats (LES, ASR, CSN) | End of each Congress + session update | Legislative activity concludes |
| Constituent stats (CAR, RI) | When new CES wave is released | New survey data |
| Speech stats (FSI) | Monthly during session | New speeches accumulate |
| Composite scores | After any component stat updates | Downstream aggregation |

### Tracking State Changes

Politicians' careers produce events that trigger stat updates:

| Event | Stats Affected | Action |
|-------|---------------|--------|
| Election result certified | MVD, CTOV, DDL, polling gap | Compute electoral stats for the race |
| FEC quarterly filing | SDR, FER, burn rate | Update campaign stats |
| Congress concludes | LES, CSN, BDR, SDL | Compute legislative stats for the Congress |
| CES wave released | CAR, IAR, RI | Compute constituent stats |
| Politician changes office | Career summary, fit scores | Recalculate composites |
| New community-type model run | CTOV, fit scores, CTDR | Recompute all type-dependent stats |
| Primary election | Upset factor, PGR, primary-based candidate adjustment | Feed into covariance pipeline |

---

## 9. Development Phases

### Phase 1: Foundation (Before Community-Covariance Model Exists)

**Target:** Computable immediately. No dependency on the covariance pipeline.

**Deliverables:**
- Politician roster with ID crosswalk (bioguide ↔ icpsr ↔ fec_id)
- Scalar MVD using Cook PVI baselines for FL+GA+AL congressional races, 2000-2024
- LES import from thelawmakers.org
- DW-NOMINATE / Nokken-Poole import from VoteView
- FEC-derived campaign stats (SDR, FER, burn rate) for FL+GA+AL candidates
- Co-sponsorship network construction and centrality scores for FL+GA+AL delegation
- CES-derived constituent approval residuals (CAR)
- Basic politician card output (JSON per politician)

**Infrastructure:**
- `src/sabermetrics/` module skeleton
- `data/sabermetrics/` storage structure
- Data download scripts for VoteView, LES, FEC, CES, Congress.gov API

### Phase 2: Type-Level Integration (After MVP Community-Covariance Model)

**Target:** When the community-type weight matrix W is available from the covariance pipeline.

**Deliverables:**
- CTOV computation: vector residuals by community type
- CEC (cross-election consistency) for candidates with multiple races
- Candidate-district fit scoring
- Donor community-type distribution (CTDR) via FEC ZIP → county → W mapping
- Model-native structural baseline (replacing Cook PVI with type-weighted prediction)
- Comparison: does CTOV-based analysis reveal things VAR/WAR miss?

### Phase 3: Advanced Analytics (After Full Covariance Model)

**Target:** When the full Bayesian model is running and producing type-level posteriors.

**Deliverables:**
- Candidate Mobilization Effect (CME) by community type (voter file + W)
- Strategic defection scoring (SDL) with community-type context
- Floor speech influence (FSI) -- NLP pipeline on congressional record
- Talent pipeline: project CTOV from lower-office races to congressional-level
- Portfolio optimization: which combination of candidates best covers a state's community types?
- Scouting reports: automated narrative generation for candidate profiles

### Phase 4: Live System

**Target:** October 2026 and beyond.

**Deliverables:**
- Real-time stat tracking during election cycles (poll-based CTOV updates)
- Primary result integration (upset detection → candidate adjustment)
- Interactive candidate comparison tool
- Fit score rankings for open-seat recruitment
- Live dashboard with politician cards

---

## References

### Candidate Quality and Electoral Performance
- Inside Elections Baseline and VAR methodology
- Split Ticket WAR model and database
- Kawai & Sunada (2025). "Estimating Candidate Valence." *Econometrica*.
- Carson, Engstrom & Roberts (2007). "Candidate Quality, the Personal Vote, and the Incumbency Advantage." *APSR*.
- Abramowitz (1988, updated). "An Improved Model for Predicting House Elections."
- Erikson & Titiunik (2015). "Using Regression Discontinuity to Uncover the Personal Incumbency Advantage."
- King. "Estimating Incumbency Advantage without Bias." Harvard.

### Legislative Effectiveness
- Volden & Wiseman (2014). *Legislative Effectiveness in the United States Congress*. Center for Effective Lawmaking.
- Fowler (2006). "Connecting the Congress: A Study of Cosponsorship Networks."
- Grimmer (2013). *Representational Style in Congress*. NLP analysis of congressional communication.

### Roll Call Analysis
- Snyder & Groseclose (2000). "Estimating Party Influence in Congressional Roll-Call Voting." *AJPS*.
- Kirkland & Slapin (2017). "Ideology and Strategic Party Disloyalty." *Electoral Studies*.
- Carson, Koger, Lebo & Young (2010). "Electoral Costs of Party Loyalty." *AJPS*.

### Constituent Representation
- Ansolabehere & Kuriwaki (2022). "Congressional Representation: Accountability from the Constituent's Perspective." *AJPS*.
- Caughey & Warshaw (2018). "Policy Preferences and Policy Change."

### Congressional Speech Analysis
- Gentzkow, Shapiro & Taddy (2019). "Measuring Group Differences in High-Dimensional Choices." *Econometrica*. Stanford congressional record dataset.

### Polling Error
- Gelman, Goel, Rivers, Rothschild (2016). "The Mythical Swing Voter." *QJPS*.
- Jennings & Wlezien. "Election Polling Errors across Time and Space."

### Campaign Finance
- Jacobson (1978, 1980, 1990). Campaign spending effects and challenger quality.
- Bonica (2014). "Mapping the Ideological Marketplace." DIME/CFscore.

### Data Sources
- Congress.gov API: https://api.congress.gov/
- VoteView: https://voteview.com/data
- Center for Effective Lawmaking: https://thelawmakers.org/data-download
- FEC bulk data: https://www.fec.gov/data/browse-data/
- OpenFEC API: https://api.open.fec.gov/developers/
- CES cumulative: https://doi.org/10.7910/DVN/II2DB6
- Gentzkow-Shapiro-Taddy speeches: https://data.stanford.edu/congress_text
- ConSpeak: Congressional speeches 1995-2024 (JSON, daily updates)
- LegiScan API: https://legiscan.com/legiscan
- unitedstates/congress-legislators: https://github.com/unitedstates/congress-legislators (ID crosswalk YAML)
