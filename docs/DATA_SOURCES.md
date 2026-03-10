# Data Sources

Evaluation of candidate data sources for the US Political Covariation Model. Each source is assessed for resolution, temporal coverage, licensing, privacy considerations, known biases, signal value for community detection and political modeling, and priority for the MVP.

**Priority key:**
- **Critical** -- model cannot function without this source.
- **High** -- significantly improves model quality; strong candidate for MVP.
- **Medium** -- useful enrichment; include if feasible.
- **Low** -- nice-to-have; defer to later phases.

---

## Election Returns

| Source | URL | Resolution | Temporal | License | Privacy | Bias / Limitations | Signal Value | Priority | Status |
|--------|-----|-----------|----------|---------|---------|-------------------|-------------|----------|--------|
| MIT Election Data + Science Lab (MEDSL) -- County Presidential Returns | [https://electionlab.mit.edu/data](https://electionlab.mit.edu/data) | County | 2000-2020 (presidential); midterm available separately | CC-BY | None -- aggregated | Minor county boundary changes over time; some states report townships not counties (New England). Third-party votes sometimes aggregated inconsistently. | Core dependent variable. Provides the election-to-election swing vectors that define the covariance structure. | Critical | Available; downloaded |
| MEDSL -- County-Level Congressional / State Returns | [https://electionlab.mit.edu/data](https://electionlab.mit.edu/data) | County (some precinct) | 2000-2022 | CC-BY | None | Uncontested races produce misleading vote shares. Redistricting makes cross-cycle comparison harder for House races. | Additional elections increase the temporal depth for covariance estimation. Senate and gubernatorial races add signal for non-presidential cycles. | High | Available |
| Dave Leip's Atlas of U.S. Elections | [https://uselectionatlas.org](https://uselectionatlas.org) | County | 1789-present | Proprietary -- personal use license | None | Some historical data is estimated or reconstructed. License restricts redistribution. | Deep historical baseline; useful for testing covariance stability over long horizons. Not essential for MVP. | Low | Available (paid) |
| Precinct-level returns (various state sources, Voting and Election Science Team -- VEST) | [https://dataverse.harvard.edu/dataverse/electionscience](https://dataverse.harvard.edu/dataverse/electionscience) | Precinct | 2016-2022 (growing coverage) | CC-BY / varies by state | None | Precinct boundaries change frequently. Geocoding and boundary matching is labor-intensive. Coverage is incomplete for some states/years. | Enables sub-county analysis and validation. Useful for testing whether county-level types miss within-county heterogeneity. | Medium | Available for FL/GA/AL |

---

## Demographics

| Source | URL | Resolution | Temporal | License | Privacy | Bias / Limitations | Signal Value | Priority | Status |
|--------|-----|-----------|----------|---------|---------|-------------------|-------------|----------|--------|
| American Community Survey (ACS) 5-year estimates | [https://data.census.gov](https://data.census.gov) | County, tract, block group | 2009-2023 (rolling 5-year windows) | Public domain | Aggregated; tract-level has noise for small populations | Margins of error are large for small geographies and rare populations. 5-year pooling smooths over rapid changes. Hispanic origin and race categories have known measurement issues. | Primary demographic feature set for community detection: age, race/ethnicity, education, income, occupation, housing, language, nativity, veteran status. | Critical | Available via Census API |
| Decennial Census (2010, 2020) | [https://data.census.gov](https://data.census.gov) | Block and up | 2010, 2020 | Public domain | Differential privacy applied in 2020 | 2020 differential privacy adds noise at small geographies. Block-level data is limited to basic demographics (P.L. 94-171). | Precise population counts for apportionment and base rates. 2020 redistricting file provides block-level race/ethnicity. | High | Available |
| USDA Economic Research Service -- Rural-Urban Continuum Codes, ERS county typology | [https://www.ers.usda.gov/data-products/](https://www.ers.usda.gov/data-products/) | County | 2013, 2023 | Public domain | None | Coarse classification (9 codes). Updated infrequently. | Useful as a stratification variable and sanity check on community type urbanicity. Not a primary input. | Low | Available |

---

## Religion

| Source | URL | Resolution | Temporal | License | Privacy | Bias / Limitations | Signal Value | Priority | Status |
|--------|-----|-----------|----------|---------|---------|-------------------|-------------|----------|--------|
| Religious Congregations and Membership Study (RCMS) / U.S. Religion Census | [https://www.usreligioncensus.org](https://www.usreligioncensus.org) | County | 2000, 2010, 2020 | Free for research (ARDA) | Aggregated | Undercounts historically Black Protestant denominations and non-congregational traditions (Islam, Hinduism). Catholic counts are generally reliable. Evangelical Protestant well-covered via SBC, AG, etc. | Key community-detection input. Denominational composition (SBC adherence rate, Catholic rate, mainline rate, LDS rate, etc.) is one of the strongest predictors of political behavior and strongly defines community types. | Critical | Available via ARDA |
| Association of Religion Data Archives (ARDA) -- supplemental datasets | [https://www.thearda.com](https://www.thearda.com) | County (varies) | Various | Free for research | Aggregated | Coverage and methodology varies across datasets. | Supplementary religious data (megachurch locations, clergy surveys, etc.). Useful for enrichment but not essential. | Low | Available |

---

## Migration and Commuting

| Source | URL | Resolution | Temporal | License | Privacy | Bias / Limitations | Signal Value | Priority | Status |
|--------|-----|-----------|----------|---------|---------|-------------------|-------------|----------|--------|
| IRS Statistics of Income -- County-to-County Migration | [https://www.irs.gov/statistics/soi-tax-stats-migration-data](https://www.irs.gov/statistics/soi-tax-stats-migration-data) | County-to-county flows | Annual, 2011-2022 | Public domain | Suppressed cells for small counts (<20 returns) | Only covers tax filers -- misses non-filers (low income, elderly on SS only). AGI data available but coarse. Origin/destination pairs with small flows are suppressed. | Reveals which counties exchange population, building the migration layer of the community network. Migrants carry political culture, so migration links indicate community similarity. | High | Available |
| LEHD Origin-Destination Employment Statistics (LODES) | [https://lehd.ces.census.gov/data/](https://lehd.ces.census.gov/data/) | Census block (aggregatable to county) | Annual, 2002-2021 | Public domain | Synthetic data with noise injection | Based on administrative records with synthetic data methods -- individual records are modeled, not observed. Accuracy degrades for small areas. Federal workers and some categories excluded. | Commuting flows define daily-interaction communities. Workers commuting between counties share workplaces and, by assumption (A004), political environments. Builds the commuting layer of the community network. | High | Available |
| ACS Commuting (Journey to Work) | [https://data.census.gov](https://data.census.gov) (Table B08301 etc.) | County | 5-year ACS windows | Public domain | Aggregated | Large MOEs for small counties. County-to-county flows available via special tabulation (CTPP) with limited public access. | Supplements LODES with mode-of-transportation and travel-time data. County-to-county flow table from CTPP is valuable but access is more restricted than LODES. | Medium | Partially available |

---

## Social Networks

| Source | URL | Resolution | Temporal | License | Privacy | Bias / Limitations | Signal Value | Priority | Status |
|--------|-----|-----------|----------|---------|---------|-------------------|-------------|----------|--------|
| Facebook Social Connectedness Index (SCI) | [https://data.humdata.org/dataset/social-connectedness-index](https://data.humdata.org/dataset/social-connectedness-index) | County-to-county (also ZCTA) | Snapshot (~2020, updated periodically) | Open (Humanitarian Data Exchange) | Aggregated and anonymized | Only Facebook users -- skews younger, more urban, and more connected than the general population. Does not capture non-Facebook social ties. Single snapshot, no temporal variation. | Directly measures social connectedness between counties -- the best available proxy for "who talks to whom." Key input for the social-network layer of community detection. Bailey et al. 2018 showed SCI predicts economic and social outcomes. | High | Available |

---

## Surveys and Polling

| Source | URL | Resolution | Temporal | License | Privacy | Bias / Limitations | Signal Value | Priority | Status |
|--------|-----|-----------|----------|---------|---------|-------------------|-------------|----------|--------|
| Cooperative Election Study (CES, formerly CCES) | [https://cces.gov.harvard.edu](https://cces.gov.harvard.edu) | Individual (with congressional district, county via geocoding) | Annual since 2006, ~60K respondents/year | Free for research | Publicly available microdata with geographic identifiers | Opt-in internet panel (YouGov) -- not a probability sample. Weights provided but may not fully correct for selection bias. County identification requires matching/imputation for some respondents. | Large-N survey with rich political attitudes, validated vote, and demographic detail. Essential for MRP (multilevel regression and poststratification) to produce small-area opinion estimates. The main survey input for the propagation model. | Critical | Available |
| American National Election Studies (ANES) | [https://electionstudies.org](https://electionstudies.org) | Individual (limited geography) | Biennial since 1948 | Free for research | Geography restricted to protect respondent privacy | Small N (~2K-8K). Geographic identifiers limited -- typically state only in public file, county in restricted file. | Gold standard for political attitudes and behavior measurement. Useful for validating CES measures and calibrating priors. Panel component (2016-2020) is valuable for individual stability estimates (A001). | Medium | Available (public); restricted file requires application |
| 538 / Polling aggregator data | [https://projects.fivethirtyeight.com/polls/](https://projects.fivethirtyeight.com/polls/) | State / district / national | Per-poll, ongoing | CC-BY (538 historical data) | Aggregated | Polls have well-documented biases (likely voter screens, differential nonresponse, herding). Geographic resolution is almost always state-level; sub-state polls are rare. | The primary input for the propagation model's poll-decomposition step. Each poll is decomposed into community-type signals via spectral unmixing. State-level polls are the most common and valuable. | Critical | Available |
| Morning Consult / large-N tracking polls | [https://morningconsult.com](https://morningconsult.com) | State (some metro) | Daily tracking since ~2020 | Proprietary -- some data shared publicly | Aggregated | Online opt-in panel. State-level estimates are based on large N but still have nonresponse bias. Sub-state geography is limited. | Very large N enables more precise state-level tracking. Useful as supplementary poll input alongside traditional polls. | Medium | Partially available (public releases) |

---

## Economic and Labor

| Source | URL | Resolution | Temporal | License | Privacy | Bias / Limitations | Signal Value | Priority | Status |
|--------|-----|-----------|----------|---------|---------|-------------------|-------------|----------|--------|
| BLS Quarterly Census of Employment and Wages (QCEW) | [https://www.bls.gov/qcew/](https://www.bls.gov/qcew/) | County x NAICS industry | Quarterly, 1990-present | Public domain | Suppressed cells for small employers | Extensive suppression in small counties to protect employer confidentiality. Agriculture and self-employment undercounted. | Industry composition shapes economic interests and community character (military bases, university towns, agriculture, manufacturing, tourism). Useful for community detection feature matrix. | High | Available |
| Bureau of Economic Analysis (BEA) -- Local Area Personal Income | [https://www.bea.gov/data/income-saving/personal-income-county-metro-and-other-areas](https://www.bea.gov/data/income-saving/personal-income-county-metro-and-other-areas) | County | Annual, 1969-present | Public domain | Aggregated | Measures income by place of residence, which may not reflect local economic conditions for commuters. Transfer payments included. | Per-capita income and income composition (wages, transfers, investments) provides an economic dimension for community typing. | Medium | Available |

---

## Voter Files

| Source | URL | Resolution | Temporal | License | Privacy | Bias / Limitations | Signal Value | Priority | Status |
|--------|-----|-----------|----------|---------|---------|-------------------|-------------|----------|--------|
| Florida Division of Elections -- Voter Registration File | [https://dos.fl.gov/elections/data-statistics/voter-registration-statistics/](https://dos.fl.gov/elections/data-statistics/voter-registration-statistics/) | Individual (with precinct, county) | Updated monthly | Public record (FL statute 97.0585) -- nominal fee | Contains name, address, DOB, party, race, vote history | Self-reported race/ethnicity is optional and incomplete. Party registration is not available in all states. Address geocoding required for precise location. | Individual-level party registration, demographics, and vote history enables ground-truth validation of community types and turnout models. Florida is one of the most accessible and detailed voter files nationally. | Medium (MVP fallback) | Available (~$5 for full file) |
| L2 / TargetSmart / Other Commercial Voter Files | Various commercial vendors | Individual (national) | Ongoing | Commercial -- expensive ($5K-$50K+) | Contains modeled and appended data (ethnicity, consumer data) | Modeled variables (e.g., modeled ethnicity, partisanship scores) embed vendor assumptions. Cost is prohibitive for academic work. Data sharing restrictions. | Most complete individual-level political data available. National coverage with validated vote history, modeled demographics, and consumer data. Would significantly enhance community detection and validation. | Low (future phase) | Not acquired |

---

## Other

| Source | URL | Resolution | Temporal | License | Privacy | Bias / Limitations | Signal Value | Priority | Status |
|--------|-----|-----------|----------|---------|---------|-------------------|-------------|----------|--------|
| MIT Election Data + Science Lab -- Districting / Boundary files | [https://electionlab.mit.edu/data](https://electionlab.mit.edu/data) | Congressional / state legislative districts | Per redistricting cycle | CC-BY | None | Boundaries change each decade. Crosswalk between precincts and districts is imperfect. | Needed for aggregating county/type estimates to congressional district predictions. | High | Available |
| Census TIGER/Line shapefiles | [https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) | Block, tract, county, state boundaries | Annual | Public domain | None | Large file sizes. Boundary changes between years. | Geographic boundaries for mapping and spatial analysis. Required for any cartographic output. | High | Available |
| National Center for Health Statistics -- Urban-Rural Classification | [https://www.cdc.gov/nchs/data_access/urban_rural.htm](https://www.cdc.gov/nchs/data_access/urban_rural.htm) | County | 2013 (based on 2010 census) | Public domain | None | Based on 2010 data; not yet updated for 2020 census. 6-category scheme. | Urbanicity classification useful for stratification and as a community-detection feature. | Low | Available |
| Economist / Stan election model (open source) | [https://github.com/TheEconomist/us-potus-model](https://github.com/TheEconomist/us-potus-model) | State | 2020 | MIT License | N/A | Designed for presidential elections only. State-level, not county-level. Specific to 2020 cycle. | Architectural reference for the propagation model. Stan code for poll aggregation, state correlation, and fundamentals integration provides a starting template. | High (reference) | Available |
