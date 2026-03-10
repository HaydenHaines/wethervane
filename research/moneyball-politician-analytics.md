# Moneyball for Politicians: A Future Nexus of Exploration and Development

**Status:** Exploratory research direction
**Relationship to core model:** The community-covariance model provides the infrastructure to make most of these metrics computable. The type-level decomposition is the key enabler.

---

## The Fundamental Problem

Baseball has ~162 games/season with hundreds of measurable discrete events per game (at-bats, pitches, defensive plays). Football has 17 games with fewer but still measurable plays. A politician might have **one competitive election every 2-6 years**. The sample size problem isn't just small -- it's existentially small. You can't sabermetric your way out of n=3.

But that framing -- "politicians just have wins" -- is only true if you define the unit of analysis as the election outcome. Moneyball worked because it redefined what was worth measuring. The equivalent move for politics is: **stop measuring politicians by wins and start measuring the political actions they take, the coalitions they build, and the environments they operate in.**

Baseball's revolution wasn't just "use stats." It was "measure the right things." On-base percentage existed before Moneyball -- it was just undervalued. The political equivalent is identifying which measurable political behaviors are undervalued predictors of electoral and governing success.

---

## The Sabermetric Analogy, Extended

| Baseball Concept | Political Equivalent | Why It's Undervalued |
|-----------------|---------------------|---------------------|
| Batting average (overvalued) | Win/loss record | Conflates candidate skill with district partisanship |
| On-base percentage (undervalued) | Coalition breadth across community types | Not measured; requires decomposition infrastructure |
| WAR (wins above replacement) | Marginal vote delivery vs. replacement candidate | Requires counterfactual modeling of "generic candidate" performance |
| Launch angle / exit velocity | Legislative tactics (amendment strategy, floor speech influence) | Requires NLP and network analysis of legislative behavior |
| Defensive runs saved | Downballot lift/drag on same-ticket candidates | Measurable but rarely attributed |
| Pitch framing (catcher) | Narrative framing (setting the terms of debate) | Requires tracking frame adoption across media and other politicians |
| Platoon splits (L/R) | Community-type splits (performance by community archetype) | Requires the community-covariance model to decompose |

---

## Measurable Stats: A Taxonomy

### Tier 1: Electoral Performance Stats

These are computable from election returns + the community-covariance model.

**Marginal Vote Delivery (MVD) -- the "WAR" of politics**

How many votes did this candidate deliver beyond what a replacement-level (generic party) candidate would have gotten in this district?

```
MVD = actual_votes - expected_votes(district_partisanship, national_environment)
```

This requires knowing the district's structural baseline, which is exactly what our model produces. The "replacement-level candidate" is the model's prior prediction for a generic candidate in that district given the national environment. MVD is the residual.

MVD can be decomposed further:
- **MVD-persuasion**: votes gained by shifting community types' vote share beyond structural expectations
- **MVD-mobilization**: votes gained by elevating turnout in favorable community types beyond structural expectations
- **MVD-suppression**: opponent votes lost due to depressed turnout in opposing community types (ethical concerns here, but it's measurable)

**Community-Type Overperformance Vector (CTOV) -- the "spray chart" of politics**

Using the community-covariance model, decompose a candidate's performance into a vector of type-level residuals:

```
CTOV_k = actual_performance_in_type_k - structural_expectation_for_type_k
```

This produces a candidate's "stat line":

```
Candidate X (FL-13, 2024):
  Suburban professional types:   +4.2 vs structural baseline
  Rural evangelical types:       -1.8 vs structural baseline
  Urban Black institutional:     +0.3 vs structural baseline
  Military suburban:              +2.1 vs structural baseline
  College town:                   +5.7 vs structural baseline
  Coastal retirement:             -0.5 vs structural baseline
```

This is a **scouting report**. A party strategist can say: "This candidate's skills are most valuable in districts with high suburban-professional and college-town composition." That's the Moneyball move -- not "this candidate won their last election," but "this candidate's specific appeal profile maps to these specific electoral environments."

**Cross-Election Consistency (CEC)**

How stable is a candidate's CTOV across elections? A candidate who overperforms in the same community types election after election has a reliable skill. A candidate whose overperformance is random has been lucky.

```
CEC = correlation(CTOV_election_t, CTOV_election_t+1)
```

High CEC = genuine skill with specific communities. Low CEC = performance driven by one-time factors (opponent weakness, issue salience, coattails).

**Primary-to-General Retention Rate**

What fraction of the candidate's primary voters turn out in the general? Decomposed by community type, this reveals whether the candidate's primary coalition is durable or a one-time enthusiasm spike.

### Tier 2: Campaign Effectiveness Stats

These require campaign finance data + media data + election returns.

**Fundraising Efficiency (FE)**

```
FE = dollars_raised / dollars_spent_on_fundraising
```

Some candidates convert attention to money cheaply; others burn resources to raise resources. This is the political equivalent of "getting on base via walks" -- unglamorous but predictive of durability.

**Small-Dollar Ratio (SDR)**

```
SDR = donations_under_200 / total_donations
```

Measures grassroots enthusiasm vs. institutional/PAC reliance. High SDR candidates have organic support that is harder for opponents to replicate.

**Earned Media Ratio (EMR)**

```
EMR = press_mentions / dollars_of_paid_media
```

Some candidates generate news organically; others buy every impression. Measurable via GDELT, Google Trends, or media monitoring APIs.

**Downballot Drag/Lift (DDL)**

Does the candidate's presence on the ballot improve or hurt other candidates on the same ticket?

```
DDL = same_ticket_candidate_performance - expected_performance_without_top_of_ticket
```

Measurable by comparing same-ticket performance in districts with and without the candidate (using off-year or special election baselines). This is the political equivalent of a basketball player's plus/minus -- how does the "team" perform when you're on the court?

**Advertising Efficiency by Community Type**

Using ad spending data (CMAG/Kantar) cross-referenced with community type geography, measure the marginal vote shift per dollar spent, decomposed by community type. Some candidates' ads resonate with specific communities; others waste money on unreceptive audiences.

### Tier 3: Legislative Performance Stats

These require congressional record data + bill tracking + NLP.

**Bill Passage Rate (Adjusted for Difficulty)**

Raw bill passage rate is meaningless -- getting a post office renamed is not the same as passing healthcare reform. Weight by:
- Bill complexity (number of sections, committee referrals)
- Partisan composition of the chamber at the time
- Whether it required bipartisan support
- Subject area difficulty (fiscal policy vs. commemorative resolutions)

```
Adjusted_BPR = sum(bill_passed_i * difficulty_weight_i) / sum(difficulty_weight_i)
```

**Amendment Success Rate (ASR)**

How often does a legislator successfully attach amendments to bills? This measures tactical skill in a way that bill passage doesn't. A legislator who can't pass their own bills but consistently shapes others' legislation through amendments may be more influential than raw bill counts suggest.

**Co-Sponsorship Network Centrality**

Build a network where legislators are nodes and co-sponsorships are edges. Measure:
- **Betweenness centrality**: Does this legislator bridge partisan clusters? High betweenness = broker/dealmaker.
- **Eigenvector centrality**: Is this legislator connected to other influential legislators?
- **Cross-party co-sponsorship rate**: What fraction of co-sponsorships cross the aisle?

A legislator with high betweenness centrality in the co-sponsorship network has measurably different skills than one who only co-sponsors within their caucus.

**Floor Speech Influence (FSI)**

NLP analysis of whether a legislator's floor speeches introduce frames, terminology, or arguments that other legislators subsequently adopt.

```
FSI = count(novel_n-grams_in_legislator_speeches_later_adopted_by_others)
```

This measures agenda-setting power -- the ability to define the terms of debate. A legislator with high FSI shapes the conversation even when they don't author the final legislation.

**Committee Effectiveness**

Bills that pass through a legislator's committee vs. die there. For committee chairs: what fraction of referred bills receive hearings? For ranking members: how often do they successfully force hearings or amendments?

### Tier 4: Behavioral and Positional Stats

These require roll call data + survey data + temporal analysis.

**Vote-Ideology Consistency**

DW-NOMINATE scores already measure this at a high level, but finer decomposition is possible: how consistent is the legislator on economic vs. social vs. foreign policy dimensions? A legislator who is consistently conservative on economics but swing on social issues has a different profile than one who is uniformly conservative.

**Bipartisan Deviation Rate (BDR)**

How often does the legislator break from party on votes where party discipline is strong?

```
BDR = party_line_defections / total_party_line_votes
```

High BDR is valuable in swing districts (signals independence) but a liability in safe districts (signals unreliability to the base). The optimal BDR depends on the district's community-type composition.

**Responsiveness Index**

Does the legislator's voting shift track constituent opinion shifts? Measurable by correlating DW-NOMINATE drift with CES opinion data in their district over time.

```
RI = correlation(delta_NOMINATE, delta_district_opinion)
```

High RI = responsive to constituents. Low RI = ideologically rigid or captured by non-constituent interests.

**Crisis Response Speed**

How quickly does the legislator issue statements, propose legislation, or hold events in response to district-relevant events? Measurable from press release timestamps, bill introduction dates, and event calendars relative to triggering events (natural disasters, factory closings, mass shootings).

**Constituent Service Metrics**

Casework volume, resolution rate, response time. These exist within congressional offices but are rarely public. FOIA requests for casework statistics, or survey-based proxies (CES includes "contacted representative" questions), could approximate this.

---

## Data Sources

| Data Source | What It Measures | Access | Granularity |
|-------------|-----------------|--------|-------------|
| **Congress.gov / GovTrack / ProPublica API** | Bill sponsorship, co-sponsorship, votes, committee assignments | Free API | Per-bill, per-vote |
| **VoteView (DW-NOMINATE)** | Ideological positioning from roll call votes | Free, updated per Congress | Per-legislator, per-Congress |
| **FEC filings** | Fundraising, spending, donor geography, small-dollar ratio | Free, bulk download | Per-candidate, per-quarter |
| **CES/CCES** | Constituent opinion matched to district | Free, Harvard Dataverse | Individual-level, biennial |
| **MEDSL + state SOS data** | Election returns by precinct/county | Free | Precinct/county |
| **Congressional Record** | Floor speeches (full text for NLP) | Free, GPO | Per-speech |
| **Legiscan / OpenStates** | State legislature bills, votes, sponsors | Free tier + paid | Per-bill, per-vote |
| **Google Trends / GDELT** | Earned media, public attention | Free | Daily, per-topic |
| **Ballotpedia** | Candidate bios, endorsements, primary results | Free | Per-candidate |
| **DIME (Bonica, Stanford)** | Donor-based ideology estimates, CFscores | Free | Per-candidate |
| **CMAG/Kantar (via Wesleyan Media Project)** | TV/digital ad spending by market | Academic access | Per-ad, per-market |
| **OpenSecrets** | PAC contributions, lobbying, outside spending | Free | Per-candidate, per-cycle |
| **Civic Engagement data (CPS Voting Supplement)** | Voter registration, turnout, civic participation | Free, Census | Biennial, state-level |

---

## Connection to the Community-Covariance Model

The community-covariance model is the **infrastructure** that makes most of these metrics computable. Without it, you can measure "candidate X won by 5 points" -- with it, you can measure "candidate X overperformed structural expectations by 3 points in suburban-professional communities and underperformed by 2 points in rural-evangelical communities."

Specific connections:

### CTOV as the Central Stat

The Community-Type Overperformance Vector is the single most important metric in this framework, and it is only computable with the community-covariance model's type-level decomposition. It turns a scalar (win/loss margin) into a vector that characterizes *how* a candidate wins.

### Candidate-District Fit Score

Given a candidate's CTOV from prior elections and a target district's community-type composition, compute a "fit score":

```
fit_score = dot(CTOV_candidate, W_district)
```

A high fit score means the candidate's specific strengths align with the district's community composition. This is the Moneyball move for candidate recruitment: instead of "find a good candidate," it's "find the candidate whose community-type appeal profile best matches this district's community-type composition."

### Portfolio Theory for Party Strategy

Treating candidates as assets with community-type-specific returns, a party can optimize its "portfolio" of candidates across districts. A party that recruits candidates whose CTOVs collectively cover the most community types with the least overlap has a more efficient slate. This is Harry Markowitz's portfolio theory applied to electoral strategy.

### Talent Development Pipeline

If CTOV is stable (high CEC), it can be measured from lower-level races (state legislature, county commission) and used to project upward. A state legislator whose CTOV shows strong performance with "suburban professional" types is a natural candidate for a congressional district with high suburban-professional composition. This is the equivalent of baseball's minor league scouting system.

---

## Research Directions

### Near-Term (Computable with Existing Data)

1. **Historical CTOV computation**: Using MEDSL county returns + the community-covariance model, compute CTOVs for all FL+GA+AL congressional candidates 2000-2024. Do CTOVs predict future performance?
2. **MVD vs. conventional wisdom**: Compare MVD rankings to expert ratings (Cook, Sabato). Which candidates are "overrated" (low MVD, high reputation) and "underrated" (high MVD, low reputation)?
3. **Co-sponsorship network analysis**: Build co-sponsorship networks for the FL+GA+AL congressional delegations. Does network position predict electoral vulnerability?
4. **Small-dollar ratio as durability predictor**: Do candidates with high SDR survive wave elections better than those with low SDR?

### Medium-Term (Requires NLP and Additional Infrastructure)

5. **Floor speech influence network**: Build an n-gram diffusion network from Congressional Record. Which legislators are frame-setters?
6. **Ad efficiency by community type**: Using Wesleyan Media Project data + community type geography, which candidates are wasting money on unreceptive communities?
7. **Crisis response measurement**: Build a dataset of district-relevant events and measure legislator response latency.

### Long-Term (Requires New Data Collection or Access)

8. **Constituent service quality**: Develop proxies for casework effectiveness from CES contact data, FOIA requests, or direct surveys.
9. **Candidate quality prediction from lower office**: Can CTOVs from state-level races predict congressional performance? Build a minor-league-to-majors projection system.
10. **Cross-branch skill transfer**: Do legislative effectiveness metrics predict executive effectiveness (for candidates who become governors or executives)?

---

## Open Questions

1. **Is political talent stable?** Baseball talent has measurable stability (a .300 hitter is likely to hit .280-.320 next year). Is political talent similarly stable, or is it more context-dependent? The CEC metric (cross-election consistency of CTOV) is the empirical test.

2. **What is the political equivalent of "clutch"?** In baseball, "clutch hitting" is largely a myth -- performance in high-leverage situations is not significantly different from baseline. Is there a political equivalent? Do some candidates systematically overperform in close races, or is apparent "clutch" performance just noise?

3. **Can talent be traded?** In baseball, Moneyball enabled trading overvalued players for undervalued ones. The political equivalent would be candidate recruitment -- can a party systematically identify undervalued candidates (those with high MVD potential but low name recognition) and recruit them to high-fit districts?

4. **What is the "Three True Outcomes" of politics?** Baseball's analytics revolution led to a game dominated by home runs, walks, and strikeouts. What would a "statistically optimized" political strategy look like, and would it be healthy for democracy?

5. **Defensive vs. offensive metrics**: Baseball has separate stats for hitting and fielding. Politics may have a similar split: "offensive" stats (persuasion, mobilization, agenda-setting) vs. "defensive" stats (constituent service, crisis response, opposition research resilience). Which dimension matters more for which type of race?

6. **Team vs. individual attribution**: In basketball, plus/minus tries to separate individual contribution from team effects. The political equivalent is separating a candidate's personal appeal from party brand, presidential coattails, and national environment. Our model's structural baseline is the "team effect" -- CTOV is the individual contribution.

---

## References

### Moneyball / Sports Analytics Analogy
- Lewis (2003). *Moneyball*. The original case study of undervalued metrics in baseball.
- Silver (2012). *The Signal and the Noise*. Nate Silver's bridge from baseball analytics to political forecasting.
- Mathletics / Moskowitz & Wertheim (2011). *Scorecasting*. Behavioral biases in sports evaluation, with parallels to political evaluation.

### Political Candidate Quality
- Abramowitz (1988, updated). "An Improved Model for Predicting the Outcomes of House Elections." Prior office as quality measure.
- Hall (2015). "What Happens When Extremists Win Primaries?" *APSR*. Primary outcomes and general election quality.
- Jacobson (1978, 1980, 1990). Campaign spending effects and challenger quality.
- Bonica (2014). "Mapping the Ideological Marketplace." DIME/CFscore ideology from donors.
- Carson et al. (2007). "The Impact of National Tides and District-Level Effects on Electoral Outcomes." Separating candidate from environment.

### Legislative Effectiveness
- Volden & Wiseman (2014). *Legislative Effectiveness in the United States Congress*. The Legislative Effectiveness Score (LES), the closest existing analog to legislative WAR.
- Fowler (2006). "Connecting the Congress: A Study of Cosponsorship Networks." Network analysis of co-sponsorship.
- Grimmer (2013). *Representational Style in Congress*. NLP analysis of congressional communication.
- Curry & Lee (2020). *The Limits of Party*. Party influence on legislative behavior.

### Electoral Analytics
- Sides, Tausanovitch, Vavreck (2022). *The Bitter End*. Calcification and the limits of candidate effects.
- Erikson & Titiunik (2015). "Using Regression Discontinuity to Uncover the Personal Incumbency Advantage." Rigorous incumbency measurement.
- Caughey & Warshaw (2018). "Policy Preferences and Policy Change." Measuring representation and responsiveness.

### Portfolio Theory / Optimization Analogy
- Markowitz (1952). "Portfolio Selection." The theoretical basis for treating candidate recruitment as portfolio optimization.
- Enos & Hersh (2015). "Campaign Perceptions of Electoral Closeness." How campaigns allocate resources under uncertainty.
