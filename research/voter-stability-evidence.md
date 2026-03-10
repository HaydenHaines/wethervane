# Voter Stability: Evidence For and Against

## Research Summary for the US Political Covariation Model

This document synthesizes the empirical evidence on voter stability — the claim that individual voters move less than polls suggest, and that poll-to-poll variation mostly reflects measurement artifacts rather than genuine opinion change. This evidence base underpins the assumption in a community-covariance model that communities have relatively stable political positions that shift gradually.

---

## 1. "Phantom Swings" in Polling

### Who Coined the Term and What's the Evidence

The term **"phantom swings"** was coined by **Gelman, Goel, Rivers, and Rothschild (2016)** in their paper "The Mythical Swing Voter," published in the *Quarterly Journal of Political Science* (11: 103-130). The core finding: apparent swings in vote intention in polls represent **mostly changes in sample composition**, not changes in opinion.

**The mechanism:** When a candidate receives good news (e.g., strong debate performance), that candidate's supporters become more enthusiastic and therefore more likely to answer polls. This creates a **short-term feedback loop** that magnifies small changes in opinion into misleadingly large swings in the polls.

**Key empirical evidence from the 2012 campaign:**
- In the September 12-16 Pew survey, Obama led Romney 51-42 among registered voters
- In the October 4-7 survey (post-first debate), they were tied 46-46
- This 9-point swing looked like a massive opinion shift
- However, the **RAND American Life Panel** — a high-response-rate panel that reinterviewed the same respondents — showed **much smaller changes** over the same period
- The RAND panel should not be susceptible to partisan nonresponse because it tracks the same people, and it confirmed that real opinion movement was far smaller than cross-sectional polls suggested

**Andrew Gelman's broader writing on this topic:**
- Gelman has repeatedly argued on his blog (Statistical Modeling, Causal Inference, and Social Science) that pollsters should **poststratify on party identification** to correct for differential nonresponse
- His 2021 paper "Political Polling: What It Can and Cannot Do" elaborates the general framework
- The addition of attitudinal variables (especially party ID) in MRP models corrects for differential response rates and greatly reduces (but does not entirely eliminate) phantom swings
- In "Failure and Success in Political Polling and Election Forecasting" (2021), Gelman and coauthors showed that systematic polling errors often reflect nonresponse patterns rather than genuine opinion shifts
- The "Disentangling Bias and Variance in Election Polls" paper (Shirani-Mehr, Rothschild, Goel, Gelman, 2018, JASA) further formalized the distinction between sampling error, house effects, and true opinion change

### Panel Data vs. Cross-Sectional Polls

The distinction between panel data (same respondents over time) and cross-sectional polls (fresh samples each time) is critical:

- **Cross-sectional polls** confound opinion change with composition change. If different people respond at different times, apparent "swings" can be entirely artifactual.
- **Panel surveys** track the same individuals, eliminating composition effects. They consistently show **less movement** than cross-sectional polls suggest.
- The ANES panel studies (discussed in Section 2) repeatedly confirm this pattern: individual-level change is more modest than aggregate poll-to-poll variation implies.

### Practical Implication for Modeling

A community-covariance model should treat poll-to-poll variation as **noisy**, with a substantial portion attributable to sampling error, differential nonresponse, and likely-voter screening changes. The prior on community-level opinion should be substantially more stable than raw polling data suggests.

---

## 2. Voter Stability Research

### How Much Do Individual Voters Actually Change Their Minds?

**The short answer:** Very little during a single campaign, but meaningful change accumulates over a voter's lifetime.

**Quantitative evidence from panel studies:**
- 60 days before the election, **71% of voters** state a vote intention corresponding to their final vote choice (Pons, 2018, "How Do Campaigns Shape Vote Choice?")
- The fraction with identical pre- and post-election vote declarations increases by **17 percentage points** during the final two months of campaigns, reaching **88%**
- This means roughly 12% of voters change their stated preference during the last two months, though some of this reflects crystallization of initially uncertain preferences rather than genuine persuasion

### The "Minimal Effects" Tradition

**Lazarsfeld, Berelson, and Gaudet (1948), *The People's Choice*:**
- Studied voters in Erie County, Ohio during the 1940 presidential campaign, interviewing 600 people monthly for seven months
- Found that media and campaign advertising did **not** have a profound influence on individual voting habits
- Identified three effects of campaigns: **activation** (latent predispositions crystallize), **reinforcement** (existing preferences are strengthened), and **conversion** (genuine opinion change — found to be rare)
- Interpersonal interactions and word of mouth mattered more than mass media for most voters
- The key insight: campaigns mostly **activate predispositions** that were already in voters' minds, and these predispositions shape interpretation of incoming information to yield choices that could have been predicted in advance

**Kalla and Broockman (2018), "The Minimal Persuasive Effects of Campaign Contact in General Elections":**
- Meta-analysis of **49 field experiments** measuring the persuasive effects of campaign contact
- The average effect of campaign contact on vote choice in general elections is **zero**
- Persuasive effects only appear in two rare circumstances: (1) when candidates take unusually unpopular positions and campaigns invest heavily in identifying persuadable voters, and (2) when campaigns contact voters long before election day — but this early persuasion **decays**
- Published in *American Political Science Review* 112(1): 148-166
- This is one of the strongest pieces of evidence for voter stability: if even targeted, personalized campaign contact has zero average effect, then background noise (news, ads) likely has even less

### The "Fundamentals" School

**Sides and Vavreck (2013), *The Gamble*:**
- Analyzed the 2012 presidential election using extensive quantitative data
- Central finding: **"fundamentals"** (economic performance, incumbency) are as crucial as or more important than the campaign itself
- "Game-changers" in campaigns are few and far between, and their effects are usually **small and ephemeral**
- Campaigns are best understood as a **tug-of-war**: both sides pull equally hard, so the flag in the middle appears to stand still, with each side's effort mainly serving to neutralize the other's
- People do change their minds during presidential campaigns, but it is hard for either candidate to beat the underlying fundamentals

**Erikson and Wlezien (2012), *The Timeline of Presidential Elections*:**
- Drew on data from close to **2,000 national polls** covering every presidential election from 1952 to 2008
- Found that polls from the beginning of the year have **virtually no predictive power**
- By mid-April, when candidates are identified and matched in trial heats, preferences come into focus — predicting the winner in **11 of 15 elections**
- In the last six months, voter intentions change **only gradually**, with particular events (including debates) rarely resulting in dramatic change
- Fundamentals matter, but **only because campaigns** make voters aware of them — campaigns are the transmission mechanism for fundamentals

### Panel Study Evidence

**ANES Panel Studies:**
- The American National Election Studies have conducted multi-wave panel surveys since the 1950s
- Green and Palmquist (1990) used **measurement error models** on nine multi-wave panel studies and found that, net of measurement error, Americans' partisanship is **highly stable** over time
- Much of the apparent change in party identification in panel surveys is **measurement error** (response unreliability), not true change
- Schickler and Green (1997) replicated this finding across eight panel studies from outside the US
- Party identification in contemporary panel studies appears to be **at least as stable** as it was in ANES panels from the 1950s
- However: while substantial changes are rare over a single campaign or presidential term, **meaningful change is common over a voter's lifetime**

**"Partisan Stability During Turbulent Times" (recent, Political Behavior):**
- Used three multi-wave panel surveys spanning the Obama through Trump administrations
- Applied multiple statistical approaches to differentiate true partisan change from response error
- Found that party attachments changed **very gradually** over the past decade
- Few respondents experience appreciable change in party identification in the short run
- But the pace at which partisanship changes implies that **substantial changes are relatively common over a voter's lifespan**

**Democracy Fund Voter Study Group (VIEWS of the Electorate Research Survey):**
- Longitudinal panel launched in 2016, drawing from respondents first interviewed by YouGov in December 2011
- Tracked 8,000 Americans who had been interviewed in 2011, 2012, 2016, 2018, 2020, and 2022
- Key finding: while aggregate numbers of Democrats and Republicans look stable, **13% of partisans switched their affiliation** over a five-year period
- Leaving the Republican Party was associated with positive attitudes about immigration, liberal self-identification
- Leaving the Democratic Party was associated with negative attitudes about immigration, unfavorable attitudes toward Muslims, conservative self-identification
- This 13% figure over five years is important context: it means roughly 2-3% per year, which is slow but not zero

### What Fraction of "Swing" Is Real?

**Summary of the evidence:**
- Most poll-to-poll variation within a campaign reflects sampling noise and differential nonresponse, **not** genuine opinion change
- Of the genuine individual-level change that does occur, most represents **activation and crystallization** of predispositions rather than true persuasion
- True persuasion effects (changing someone's mind from one candidate to another) are very small — **close to zero** for typical campaign contact
- Over longer time horizons (5+ years), meaningful individual change does occur, driven primarily by **issue-based sorting** and **identity-based realignment** rather than campaign effects
- At the community level, apparent shifts are a mix of individual change, differential turnout, and population change — with turnout and population change often dominating

### Practical Implication for Modeling

The evidence strongly supports a model with **high temporal autocorrelation** in community-level political behavior. Within-election variation should be treated as mostly noise. Between-election variation should be modeled as a slow drift process (random walk with small innovation variance), punctuated by occasional larger shifts that correspond to real realignment events.

---

## 3. Persuasion vs. Composition Effects

### The Three Mechanisms of Community-Level Shift

When a community appears to shift politically, three distinct mechanisms may be at work:

**(a) Conversion/Persuasion:** People who voted in both elections changed their vote choice.
**(b) Differential Turnout (Composition):** Different people showed up — some previous voters stayed home, some new voters participated.
**(c) Population Change (Composition):** The community's residents changed through migration, aging, death, and new eligible voters coming of age.

### Key Empirical Evidence

**Grimmer, Hersh, et al. (2021), "Not by Turnout Alone: Measuring the Sources of Electoral Change, 2012 to 2016":**
- Published in *Science Advances*, using voter file data from six states (FL, GA, MI, NV, OH, PA) covering 37 million registered voters
- Merged precinct-level election returns with individual-level turnout records
- **Key finding:** Both composition and conversion were substantively meaningful drivers of electoral change, but the **balance varied by state**
  - In states with larger white non-college populations (MI, OH, PA): **conversion** (Obama-to-Trump switching) was the dominant mechanism
  - In growing Sun Belt states (FL, GA, NV): **composition** (changes in who voted) was more important
- The effects of both conversion and composition **dwarfed** the estimated effects of specific campaign interventions (persuasive canvassing, TV advertising)
- Evidence from 30,000 precincts showed that **voters switching from Obama to Trump** substantially drove vote change from 2012 to 2016

### How Do Researchers Disentangle These?

**Voter file analysis** is the gold standard for separating composition from conversion:
1. Match voter files across elections to identify who voted in both, who voted only in one
2. Use precinct-level returns + individual turnout to estimate vote choice for consistent vs. new/dropped voters
3. This ecological inference approach has limitations (you don't observe individual vote choice, only precinct-level returns + individual turnout), but with enough precincts and known partisan composition shifts, it produces credible estimates

**Panel surveys** provide direct individual-level evidence but are expensive and may have panel conditioning effects.

### Practical Implication for Modeling

A community-covariance model should ideally **separate** these mechanisms:
- **Conversion prior:** Small innovation variance; most voters don't change. When they do, it correlates with national-level issue movements.
- **Turnout composition prior:** Can shift more rapidly; correlated with enthusiasm, mobilization, and candidate characteristics.
- **Population composition:** Predictable from demographic trends (migration, aging); creates slow, steady drift that should be modeled with external data (Census, ACS).

In practice, the model may not be able to separate all three from election returns alone, but the prior should encode the expectation that conversion is the smallest and slowest component, turnout composition can fluctuate more within elections, and population change is slow and predictable.

---

## 4. Partisan Realignment Literature

### Secular vs. Critical Realignment

**V.O. Key (1955)** introduced the concept of "critical elections" — elections in which the depth and intensity of electoral involvement are qualitatively different from preceding elections, creating durable realignment.

**Walter Dean Burnham (1970)** expanded this into a theory of **punctuated equilibrium** with a 30-38 year realignment cycle:
- Tensions arise in society around critical issues
- Regular parties fail to integrate these issues
- A "third party revolt" demonstrates this incapacity
- Parties then adjust, producing significant transformation

**Secular realignment** describes a more **gradual** process spanning decades, with multiple key events contributing to substantial changes over time, rather than a single dramatic election.

**David Mayhew (2002)** argued that **neither theory performs particularly well** empirically — the evidence for crisp, cyclical realignment is weak, and the timing of supposed "critical elections" is debatable.

### Contemporary Consensus

The current view in political science is more nuanced: **realignment is real but gradual**, with occasional acceleration rather than clean breaks. Key recent trends:

**Educational Polarization (the "Diploma Divide"):**
- From the 1980s through ~2000, there were small differences in voting by education, with college-educated voters slightly more Republican
- Around 2000, college-educated voters began trending Democratic; non-college voters trending Republican
- By 2016-2020, the divergence was the **highest ever recorded**: a national 32-point gap on educational lines in 2020
- This realignment has been **accelerating**: Trump's emergence intensified an existing trend
- The Republican Party had more college-educated identifiers than Democrats from 1980-2012; by 2020, Democrats had an 8-point advantage

**Rural-Urban Sorting ("Sequential Polarization"):**
- In the 1990s, rural areas experiencing population loss or economic stagnation began supporting Republicans
- From 2008-2020, areas with higher percentages of less-educated residents shifted further Republican
- In 2008, rural county voters were evenly split; by the 2020s, Republicans hold a **25-point advantage** among rural residents
- This happened in two sequential waves (urban educated first, rural less-educated second), not simultaneously

**The Speed of Community-Level Realignment:**
- The evidence suggests **slow, grinding change** over decades, not sudden tipping points
- The educational realignment was visible by 2000 but only became dramatic by 2016-2020 — a ~20-year process
- Rural-urban divergence has been building since the 1990s — a ~30-year process
- Within these trends, individual election cycles can produce **acceleration** (2016 was an acceleration point for the diploma divide)

### Issue Sorting vs. Party Switching

- Widening partisan gaps are primarily attributable to **sorting** (alignment of issue positions with party) rather than **polarization** (people moving to more extreme positions)
- Among those who are sorting, individuals are overwhelmingly **switching their issue positions to align with their party** rather than switching parties to align with their issues
- This has profound implications: it means community-level change is driven more by people **reinterpreting** their positions through a partisan lens than by actually changing their fundamental preferences

### Practical Implication for Modeling

- Community-level political positions should be modeled with a **slow drift process** — innovation variance on the order of 1-3 percentage points per election cycle for most communities
- The model should allow for **correlated drift** across communities that share demographic characteristics (education, urbanicity, race) to capture realignment patterns
- Rare acceleration events should be modeled with heavier tails on the innovation distribution, or a regime-switching component that allows for occasional faster movement
- External demographic data (education levels, urban/rural classification, population change) should inform the drift direction and magnitude

---

## 5. Ecological Inference and the "Neighborhood Effect"

### Does Living in a Community Shape Political Behavior?

The central question: Is the correlation between a community's political character and its residents' behavior driven by (a) the community actually influencing residents (contextual effect), or (b) people selecting into communities that match their preferences (self-selection)?

### Key Research

**Huckfeldt and Sprague (1987, 1995):**
- "Networks in Context: The Social Flow of Political Information" (*American Political Science Review*, 1987)
- Argued that individual preferences and actions are influenced through **social interaction**, and social interaction is structured by the **social composition** of the individual's environment
- Demonstrated two reactions to neighborhood context: **assimilation** (conforming to neighbors) and **conflict** (resisting the local norm)
- Which reaction occurs depends on the **interdependency of individual characteristics and contextual properties**
- Their work established that neighborhoods matter through **network effects** — you don't just absorb the "air" of a place; you interact with specific people, and who those people are is shaped by where you live

**Cho and Rudolph (2008):**
- Showed that spatial contagion influences voting behavior
- Theorized that spatial context serves as a **heuristic tool** for political decision-making
- A person's likelihood of getting involved politically is related to the density of committed others in close proximity
- However, most studies fail to distinguish contextual influence from self-selection, and **self-selection appears to dominate** contextual effects in most analyses

**Ryan Enos (2017), *The Space Between Us*:**
- Used field experiments and big data to demonstrate that **geographic proximity to outgroups** causes negative and exclusionary changes in political attitudes
- The presence of a salient outgroup — defined by its size, physical proximity, and degree of segregation — causes changes in political behavior
- This is important causal evidence that context matters **above and beyond self-selection**
- However, the effects documented are primarily about **intergroup attitudes** (racial/ethnic), not general partisan preference

**The Moving to Opportunity (MTO) Experiment:**
- A rare source of **causal** evidence on neighborhood effects, as it randomly assigned housing vouchers
- Findings are **mixed and surprising**: relocating young children from high-poverty to low-poverty neighborhoods improved social/economic outcomes but did **not** increase voter registration or turnout in adulthood
- Teenagers who moved were actually **less likely to vote** later
- This suggests neighborhood effects on political behavior are **more complex** than simple contextual influence models suggest
- A separate study of housing project demolitions found the opposite — displaced children were 2.9 percentage points (10%) more likely to vote

### Current Consensus on Contextual Effects

- There is evidence for both self-selection into politically compatible areas **and** contextual assimilation of new entrants to the majority political orientation
- **Self-selection dominates** contextual effects in most analyses
- Contextual effects exist but are **weak** relative to individual-level predictors (party ID, demographics, ideology)
- There is no consensus on the appropriate **geographic scale** for measuring contextual effects
- The strongest causal evidence for contextual effects comes from intergroup contact settings (Enos), not general partisan preference

### Practical Implication for Modeling

- A community-covariance model is justified in treating community-level political behavior as having **its own dynamics** (not just an aggregation of individual behavior)
- However, the contextual effects are **modest** — most of the spatial clustering in political behavior reflects composition (who lives where) rather than context (place shaping individuals)
- The model should primarily treat community-level variables as **slow-moving compositional aggregates**, with a small additional contextual component
- Changes in community composition (through migration and generational turnover) should be the primary driver of modeled drift

---

## 6. The "Big Sort" Debate

### Bill Bishop's Thesis

In *The Big Sort: Why the Clustering of Like-Minded America Is Tearing Us Apart* (2008), journalist Bill Bishop argued that Americans increasingly choose to live in neighborhoods populated with people like themselves, producing significant increases in geographic political polarization.

### The Critics

**Abrams and Fiorina (2012), "The 'Big Sort' That Wasn't":**
- Published in *PS: Political Science & Politics*
- Argued that using Bishop's own standard, the data suggest the **opposite**: geographic political segregation is **lower** than a generation ago
- Contend that Bishop's sweeping argument has little or no empirical foundation
- The key dispute is about measurement: at what geographic scale is sorting occurring, and how do you define "like-minded"?

**Mummolo and Nall (2017), "Why Partisans Do Not Sort":**
- Published in *The Journal of Politics* (79:1)
- Confirmed that Democrats are more likely than Republicans to **prefer** living in dense, diverse, Democratic places
- However, Americans are **not migrating** to more politically distinct communities — they prioritize common concerns like **affordability** when deciding where to live
- The estimated partisan bias in moving choices is **five times too small** to sustain current geographic polarization
- Americans move frequently enough, and partisan biases in location choice are small enough, that repeated rounds of sorting would quickly **homogenize** rather than polarize the geographic distribution

**Martin and Webster (2020), "Does Residential Sorting Explain Geographic Polarization?":**
- Found that rising geographic partisan segregation has **not been driven primarily by residential mobility**
- Instead, the drivers are:
  - **Generational turnover**: new voters entering the electorate cause some places to become more homogeneously Democratic
  - **Party switching**: existing voters leaving the Democratic Party cause other places to become more Republican
- In-place conversion and generational replacement are more important than migration in explaining geographic polarization

### Current Empirical Consensus

- Geographic partisan sorting **is at an all-time high** in the post-Civil War era
- But this is **not primarily caused by people moving** for political reasons
- Rather, it reflects **in-place ideological change** (people in different places responding differently to the same national trends) and **generational turnover**
- Political preferences and residential location are correlated, but the **causal arrow mostly runs from place to politics**, not the other way around
- The practical extent of deliberate political sorting is limited by the dominance of economic concerns (housing costs, jobs) in residential decisions

### Practical Implication for Modeling

- The model should treat community political character as **relatively stable over short periods** — people aren't moving in and out rapidly enough to change a community's politics through migration
- Long-term drift in community politics is driven more by **generational turnover and in-place ideological change** than by sorting
- Adjacent or similar communities should be modeled as having **correlated** political trajectories because they respond to similar national trends and share demographic characteristics
- The correlation structure should capture the fact that communities with similar demographic profiles (education, urbanicity, racial composition) tend to drift in similar directions

---

## 7. Practical Implications for Modeling

### The Case for a Strong Stability Prior

The evidence overwhelmingly supports modeling community-level political behavior with a **strong prior toward stability**:

1. **Individual voters are sticky:** Party ID is highly stable net of measurement error; campaign persuasion effects are approximately zero; most within-campaign "movement" in polls is noise.

2. **Communities are stickier than individuals:** Even if 2-3% of individuals change party per year, communities change less because individual changes partially cancel out (some people move left, others right).

3. **Poll-to-poll variation overstates real change:** Differential nonresponse, sampling error, and likely-voter screening changes account for a large fraction of apparent movement.

4. **Realignment is real but slow:** Even the dramatic educational realignment took ~20 years to become clearly visible. Rural-urban divergence has been building for ~30 years.

### Recommended Temporal Autocorrelation Structure

**Within an election cycle (months):**
- Very high autocorrelation (> 0.95)
- Most variation is noise; true opinion change is minimal
- Model as a random walk with very small innovation variance, or even a stable mean with noise

**Between election cycles (2-4 years):**
- High autocorrelation (0.85-0.95)
- Small but real drift, on the order of 1-3 percentage points per cycle for most communities
- Model as a random walk with modest innovation variance
- Innovation should be correlated across communities sharing demographic characteristics

**Over longer horizons (10+ years):**
- Moderate autocorrelation (0.5-0.8)
- Cumulative drift can be substantial and should be modeled
- Correlation structure should capture realignment patterns (education, urbanicity, race)

### Modeling Rare Rapid Shifts

For the occasional cases where communities do shift rapidly:
- Use **heavy-tailed innovation distributions** (Student-t rather than Gaussian) to allow for occasional larger jumps
- Or implement a **regime-switching component** that can detect when a community is undergoing rapid realignment
- Allow for **nationally correlated shocks** — when rapid change occurs, it tends to be part of a national pattern (e.g., the 2016 educational realignment accelerated everywhere, not just in individual communities)
- External covariates (economic shocks, demographic changes, candidate characteristics) should modulate the innovation variance — communities whose demographics are "misaligned" with their current political position (e.g., highly educated but Republican-leaning in 2012) should have higher expected drift

### Specific Model Architecture Recommendations

Drawing from the Bayesian election modeling literature (Linzer 2013, Gelman et al.):

1. **Reverse random walk prior** from Election Day: Following Linzer (2013), begin the prior at the election outcome and walk backward in time. This ensures the model is anchored to actual results and allows polls during the campaign to gradually refine the estimate.

2. **Hierarchical structure** with correlation: Community-level parameters should be partially pooled within states and demographic groups, with **correlated random walks** so that similar communities move together.

3. **Fundamentals-based prior:** The prior mean for each community should be informed by "fundamentals" — previous election result, demographic composition, economic conditions — following the Erikson-Wlezien insight that fundamentals are the baseline and campaigns are the transmission mechanism.

4. **Differential nonresponse correction:** If using polling data, poststratify on party ID and attitudinal variables to correct for phantom swings, following Gelman's recommendation.

5. **Decomposition of change:** When possible, separately model the conversion and composition components of community-level change, informed by voter file data on turnout patterns.

---

## Key References

### Phantom Swings and Polling
- Gelman, A., Goel, S., Rivers, D., & Rothschild, D. (2016). "The Mythical Swing Voter." *Quarterly Journal of Political Science*, 11, 103-130.
- Shirani-Mehr, H., Rothschild, D., Goel, S., & Gelman, A. (2018). "Disentangling Bias and Variance in Election Polls." *JASA*, 113(522), 607-614.
- Gelman, A. (2021). "Political Polling: What It Can and Cannot Do." Working paper.
- Gelman, A. et al. (2021). "Failure and Success in Political Polling and Election Forecasting." *Statistics and Public Policy*.

### Minimal Effects and Voter Stability
- Lazarsfeld, P.F., Berelson, B., & Gaudet, H. (1948). *The People's Choice*. Columbia University Press.
- Green, D.P. & Palmquist, B. (1990). "Of Artifacts and Partisan Instability." *American Journal of Political Science*, 34, 872-902.
- Schickler, E. & Green, D.P. (1997). "The Stability of Party Identification in Western Democracies." *Comparative Political Studies*, 30(4), 450-483.
- Kalla, J. & Broockman, D. (2018). "The Minimal Persuasive Effects of Campaign Contact in General Elections." *APSR*, 112(1), 148-166.
- Broockman, D. & Kalla, J. (2023). "When and Why Are Campaigns' Persuasive Effects Small?" *AJPS*.

### Campaign Effects and Fundamentals
- Sides, J. & Vavreck, L. (2013). *The Gamble*. Princeton University Press.
- Erikson, R.S. & Wlezien, C. (2012). *The Timeline of Presidential Elections*. University of Chicago Press.
- Pons, V. (2018). "How Do Campaigns Shape Vote Choice?" Working paper.

### Panel Studies and Individual-Level Change
- "Partisan Stability During Turbulent Times." *Political Behavior* (2022).
- Democracy Fund Voter Study Group, VIEWS of the Electorate Research Survey (2016-2022).

### Composition vs. Conversion
- Grimmer, J., Hersh, E., et al. (2021). "Not by Turnout Alone: Measuring the Sources of Electoral Change, 2012 to 2016." *Science Advances*.

### Realignment
- Key, V.O. (1955). "A Theory of Critical Elections." *Journal of Politics*, 17, 3-18.
- Burnham, W.D. (1970). *Critical Elections and the Mainsprings of American Politics*.
- Mayhew, D. (2002). *Electoral Realignments: A Critique of an American Genre*.
- Zingher, J. (2022). "Diploma Divide: Educational Attainment and the Realignment of the American Electorate." *Political Research Quarterly*.
- "Sequential Polarization: The Development of the Rural-Urban Political Divide, 1976-2020." *Perspectives on Politics* (Cambridge).

### Neighborhood Effects and Contextual Influence
- Huckfeldt, R. & Sprague, J. (1987). "Networks in Context." *APSR*.
- Huckfeldt, R. & Sprague, J. (1995). *Citizens, Politics, and Social Communication*. Cambridge University Press.
- Cho, W.K.T. & Rudolph, T.J. (2008). "Emanating Political Participation." *Political Behavior*.
- Enos, R.D. (2017). *The Space Between Us*. Cambridge University Press.
- "The Long-Term Effects of Neighborhood Disadvantage on Voting Behavior: The 'Moving to Opportunity' Experiment." *APSR* (2024).

### The Big Sort
- Bishop, B. (2008). *The Big Sort*. Houghton Mifflin.
- Abrams, S.J. & Fiorina, M.P. (2012). "The 'Big Sort' That Wasn't." *PS: Political Science & Politics*.
- Mummolo, J. & Nall, C. (2017). "Why Partisans Do Not Sort." *Journal of Politics*, 79(1).
- Martin, G.J. & Webster, S.W. (2020). "Does Residential Sorting Explain Geographic Polarization?" *Political Science Research and Methods*, 8(2), 215-231.

### Bayesian Election Modeling
- Linzer, D.A. (2013). "Dynamic Bayesian Forecasting of Presidential Elections in the States." *JASA*, 108(501), 124-134.
- Gelman, A. et al. (2020). "An Updated Dynamic Bayesian Forecasting Model for the US Presidential Election." *Harvard Data Science Review*, 2(4).
