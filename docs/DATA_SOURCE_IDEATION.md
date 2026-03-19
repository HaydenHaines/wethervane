# Data Source Ideation — Aggressive Expansion

**Purpose**: Comprehensive catalog of data sources that could improve shift-based community discovery. Organized by signal category. Prioritized by: (1) free/public, (2) county or tract resolution, (3) temporal coverage, (4) expected signal strength for political covariation.

**Currently integrated**: VEST elections, MEDSL elections, ACS demographics, RCMS religion, IRS migration (raw), LODES commuting (raw), TIGER shapefiles, placeholder polls.

---

## Tier 1 — High-Impact, Free, Ready to Fetch

These should be integrated before October 2026. Each has strong theoretical grounding for why it would improve community detection or shift prediction.

### 1. CDC WONDER Mortality Data
- **What**: County-level death rates by cause (heart disease, cancer, drug overdose, suicide, COVID, "deaths of despair")
- **Resolution**: County | **Temporal**: 1999-2023
- **URL**: https://wonder.cdc.gov/
- **Why it matters**: "Deaths of despair" (Case & Deaton) are the strongest county-level predictor of Trump 2016 swing. Opioid mortality rates predict populist shift better than income alone. COVID death rates correlate with political polarization post-2020.
- **Signal**: Mortality patterns capture economic distress, healthcare access, and cultural despair that demographics miss.
- **Cost**: Free, public API

### 2. CDC COVID Vaccination Rates
- **What**: County-level vaccination completion rates, booster uptake
- **Resolution**: County | **Temporal**: 2021-2023
- **URL**: https://data.cdc.gov/Vaccinations/
- **Why it matters**: Vaccination rates became the single strongest county-level correlate of partisan lean by mid-2021 (r > 0.8 with 2020 Trump vote share in many states). This is a revealed-preference political signal that didn't exist before 2021.
- **Signal**: Post-2020 political identity marker. Captures pandemic-era realignment signal that election data alone misses between cycles.
- **Cost**: Free

### 3. FEC Individual Campaign Contributions (Zip-Level)
- **What**: Every individual political contribution >$200, geocoded to zip code. ActBlue/WinRed small-dollar totals.
- **Resolution**: ZIP (aggregatable to county) | **Temporal**: 1980-present, updated daily
- **URL**: https://www.fec.gov/data/browse-data/?tab=bulk-data
- **Why it matters**: Small-dollar donations are a revealed-preference measure of political engagement intensity. The ratio of ActBlue to WinRed contributions is a real-time partisan thermometer at sub-county resolution. Donation velocity (change in contribution rates) is a leading indicator of shift.
- **Signal**: Political engagement intensity + partisan lean at high resolution. Time-varying — captures mobilization waves.
- **Cost**: Free

### 4. BLS QCEW — Industry Composition
- **What**: County x NAICS sector employment and wages, quarterly
- **Resolution**: County | **Temporal**: 1990-present
- **URL**: https://www.bls.gov/qcew/
- **Why it matters**: Industry mix defines economic identity — military bases, university towns, agriculture, manufacturing, tourism, tech, extractive industries all have distinct political profiles. Manufacturing decline vs. service growth is a core realignment driver.
- **Signal**: Economic structure shapes political interests directly. A county that lost 30% of manufacturing jobs between 2012-2020 shifts differently than one that gained healthcare jobs.
- **Cost**: Free (some cells suppressed for small employers)

### 5. Facebook Social Connectedness Index (SCI)
- **What**: County-to-county relative probability of Facebook friendship
- **Resolution**: County-to-county | **Temporal**: ~2020 snapshot
- **URL**: https://data.humdata.org/dataset/social-connectedness-index
- **Why it matters**: Direct measure of "who talks to whom." Bailey et al. (2018) showed SCI predicts economic outcomes, trade flows, and disease spread. For this model: communities that are socially connected should shift together, even if geographically non-adjacent.
- **Signal**: Social network topology for community detection. Could define a "social adjacency" graph alongside spatial adjacency.
- **Cost**: Free (Humanitarian Data Exchange)

### 6. USDA Census of Agriculture
- **What**: Farm count, acreage, crop types, livestock, organic certification, farm income, operator demographics — by county
- **Resolution**: County | **Temporal**: Every 5 years (2017, 2022)
- **URL**: https://www.nass.usda.gov/AgCensus/
- **Why it matters**: Agricultural counties have distinct political behavior. Crop type matters: row crop (corn/soy) counties differ from livestock, specialty crop (fruit/vegetable), and timber counties. Farm consolidation (fewer, larger farms) correlates with rural depopulation and political shift.
- **Signal**: Agricultural economic structure as a community-defining feature. Distinguishes rural subtypes.
- **Cost**: Free

### 7. FHFA House Price Index / Census Building Permits
- **What**: County-level house price appreciation (quarterly HPI) + new residential construction permits
- **Resolution**: County (HPI for ~400 metros + many counties) | **Temporal**: 1991-present
- **URL**: https://www.fhfa.gov/data/hpi, https://www.census.gov/construction/bps/
- **Why it matters**: Housing affordability crisis is a major driver of political attitudes, especially among younger voters. Rapid price appreciation signals gentrification/displacement. Construction permits signal growth pressure.
- **Signal**: Economic mobility proxy. Price change rate may predict shift direction (appreciation → incumbent party advantage? Or → populist backlash from priced-out residents?).
- **Cost**: Free

### 8. BEA Regional Price Parities + Personal Income
- **What**: Cost-of-living adjustment by metro/non-metro areas + county per-capita income decomposition (wages, transfers, investments, government)
- **Resolution**: Metro/state (RPP), County (income) | **Temporal**: 2008-present (RPP), 1969-present (income)
- **URL**: https://www.bea.gov/data/
- **Why it matters**: Nominal income misleads — $50K in rural Alabama vs. Atlanta are different lives. Income composition matters: transfer-dependent counties (Social Security, disability) behave differently from wage-dependent ones. Government transfer dependency is one of the strongest rural-shift predictors.
- **Signal**: Real economic wellbeing + dependency structure.
- **Cost**: Free (BEA API)

### 9. NCES School District Data + EdFacts
- **What**: Per-pupil spending, student-teacher ratios, free/reduced lunch rates, graduation rates, charter/private school enrollment, homeschool rates
- **Resolution**: School district (mappable to county) | **Temporal**: Annual
- **URL**: https://nces.ed.gov/ccd/, https://www2.ed.gov/about/inits/ed/edfacts/
- **Why it matters**: School quality and school choice battles are politically salient. Private/homeschool rates signal cultural conservatism. Free lunch rates are a poverty proxy. Education spending levels reflect local tax base and political preferences.
- **Signal**: Cultural + economic indicator. School choice rates may be the best available proxy for cultural conservatism at the county level.
- **Cost**: Free

### 10. IRS Statistics of Income — Zip-Level Income Data
- **What**: AGI distribution by bracket, EITC claims, mortgage interest deductions, charitable giving — by zip code
- **Resolution**: ZIP code | **Temporal**: Annual
- **URL**: https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-zip-code-data-soi
- **Why it matters**: More granular income distribution than ACS. EITC claim rate is a poverty proxy. Charitable giving rate correlates with religiosity and community engagement. The Gini-like spread of AGI brackets within a zip captures inequality.
- **Signal**: Within-county economic inequality + transfer dependency + civic engagement proxy.
- **Cost**: Free

---

## Tier 2 — Strong Signal, Moderate Integration Effort

### 11. Vera Institute / BJS Incarceration Data
- **What**: County-level jail population, incarceration rates, racial disparities in incarceration
- **Resolution**: County | **Temporal**: 1970-2018
- **URL**: https://github.com/vera-institute/incarceration-trends
- **Why it matters**: Mass incarceration directly affects voter eligibility (felony disenfranchisement varies by state — FL restored rights in 2018 via Amendment 4, then restricted again). Incarceration rates correlate with both racial composition and punitive political culture.
- **Signal**: Criminal justice attitudes + direct voting pool effects. FL-specific: Amendment 4 created a natural experiment.

### 12. FCC Broadband Deployment Data
- **What**: Fixed broadband availability by census block, speed tiers, provider count
- **Resolution**: Census block | **Temporal**: Semi-annual since 2014 (new Broadband Data Collection since 2022)
- **URL**: https://broadbandmap.fcc.gov/data-download
- **Why it matters**: Digital divide is geographic and political. Broadband access affects information ecosystem, remote work capacity, and economic opportunity. Low-broadband areas have different media consumption and political information sources.
- **Signal**: Information environment + economic opportunity proxy. Interacts with WFH migration patterns.

### 13. CES/CCES (Cooperative Election Study) — County-Aggregated Attitudes
- **What**: ~60K respondents/year with validated vote, party ID, issue positions, demographics, geocoded to congressional district/county
- **Resolution**: Individual (aggregatable to county with sufficient N) | **Temporal**: Annual since 2006
- **URL**: https://cces.gov.harvard.edu
- **Why it matters**: The only large-N survey with both validated turnout AND issue-level attitudes at a geographic resolution useful for county estimation. Can produce MRP estimates of issue positions (abortion, guns, immigration attitudes) by county.
- **Signal**: Attitude-level data that election returns can't provide. Why did a county shift? Was it immigration salience? Economic anxiety? Cultural backlash?

### 14. DIME Database — Campaign Finance Ideology Scores
- **What**: CFscore ideology estimates for every candidate and donor, derived from contribution networks (Bonica 2014)
- **Resolution**: Individual candidates/donors | **Temporal**: 1980-2022
- **URL**: https://data.stanford.edu/dime
- **Why it matters**: Measures revealed ideological positioning of candidates and donors. County-level average donor ideology captures the political center of gravity. Shift in average donor ideology over time captures realignment at the activist level.
- **Signal**: Elite/activist ideology positioning. Detects polarization before it shows up in vote shares.

### 15. Opportunity Insights — Social Capital + Economic Mobility
- **What**: County-level social capital index, economic mobility (Raj Chetty), volunteering rates, civic organizations, social cohesion measures
- **Resolution**: County/ZIP | **Temporal**: Various (mostly ~2018-2022)
- **URL**: https://opportunityinsights.org/data/
- **Why it matters**: Chetty's economic mobility data is the gold standard for "American Dream" metrics. Low-mobility counties may shift differently (populist backlash) vs. high-mobility counties (status quo preference). Social capital captures Putnam's "bowling alone" thesis at county resolution.
- **Signal**: Community cohesion + economic mobility. Social capital may moderate or amplify shift magnitude.

### 16. EPA Air Quality + Environmental Justice
- **What**: AQI, PM2.5, ozone levels, EJ screening scores (pollution burden by demographic), Superfund site locations
- **Resolution**: County/tract | **Temporal**: Ongoing
- **URL**: https://www.epa.gov/ejscreen, https://aqs.epa.gov/aqsweb/
- **Why it matters**: Environmental justice communities are politically distinct. Proximity to pollution sources shapes attitudes toward regulation and industry. EPA EJ scores combine environmental burden with demographic vulnerability.
- **Signal**: Environmental politics salience. Counties with high pollution + low income may shift toward regulation-friendly candidates.

### 17. Census Business Dynamics Statistics (BDS)
- **What**: Firm births, deaths, job creation/destruction by county and firm size class
- **Resolution**: County | **Temporal**: 1978-2021
- **URL**: https://www.census.gov/programs-surveys/bds.html
- **Why it matters**: Entrepreneurship rate captures economic dynamism. Net firm births vs. deaths distinguishes growing vs. declining economies. "Company town" counties (dominated by one large employer) behave differently.
- **Signal**: Economic dynamism + dependency. Firm death rates may predict populist shift.

### 18. NOAA Climate Normals + Natural Disaster Exposure
- **What**: Temperature, precipitation, extreme weather frequency + FEMA disaster declarations per county
- **Resolution**: Station/county | **Temporal**: 30-year normals + disaster history
- **URL**: https://www.ncei.noaa.gov/products/land-based-station/us-climate-normals, https://www.fema.gov/api/open/v2/DisasterDeclarations
- **Why it matters**: Climate change exposure varies geographically and affects political attitudes toward climate policy. Repeated hurricane/flood exposure in FL/GA/AL coastal counties may shift attitudes. Agricultural drought affects rural economics.
- **Signal**: Climate vulnerability. Coastal flood risk vs. inland stability is a community-defining feature.

### 19. VA + DoD Military Presence
- **What**: Veteran population percentage, active military installations, military retiree concentration, VA facility locations
- **Resolution**: County | **Temporal**: ACS for veterans, DoD BRAC for installations
- **URL**: ACS B21001 (veterans), https://installations.militaryonesource.mil/
- **Why it matters**: Military communities have distinct political profiles — high veteran concentration correlates with defense-hawkish, socially conservative attitudes, but military base towns are also diverse and federally dependent. Base closures (BRAC) create economic shocks.
- **Signal**: Military culture as a community type. FL (MacDill, Eglin, Pensacola), GA (Fort Eisenhower, Robins), AL (Redstone, Maxwell) all have major installations.

### 20. Local News Desert Map
- **What**: Counties with/without a local newspaper, newspaper circulation trends, local TV station coverage
- **Resolution**: County | **Temporal**: 2004-present
- **URL**: https://www.usnewsdeserts.com/reports/news-deserts-and-ghost-newspapers/
- **Why it matters**: Darr et al. (2018) showed local newspaper closures increase straight-ticket voting and reduce split-ticket behavior. News deserts reduce local political information and increase nationalization of politics — which changes shift patterns.
- **Signal**: Information environment quality. News deserts may exhibit more uniform national-wave shifts; media-rich counties may show more local variation.

---

## Tier 3 — Creative/Unconventional Sources

These require more work or have weaker theoretical grounding, but could capture community signals that traditional sources miss.

### 21. IRS Exempt Organization (990) Data — Nonprofit Density
- **What**: Every 501(c)(3) and 501(c)(4) organization by zip code — type, revenue, mission
- **Resolution**: ZIP | **Temporal**: Annual
- **URL**: https://www.irs.gov/statistics/soi-tax-stats-annual-extract-of-tax-exempt-organization-financial-data
- **Why it matters**: Nonprofit density and type (churches, social services, advocacy, arts) are proxies for civic infrastructure. 501(c)(4) "dark money" organizations signal political mobilization infrastructure.
- **Signal**: Civic infrastructure density + political mobilization capacity.

### 22. USPS Address Vacancy Rate
- **What**: Percentage of addresses receiving no mail (vacant) by zip code
- **Resolution**: ZIP | **Temporal**: Quarterly
- **URL**: HUD Aggregated USPS Administrative Data (https://www.huduser.gov/portal/datasets/usps.html)
- **Why it matters**: Vacancy rates capture population decline, abandonment, and blight better than Census (which only measures every 10 years). Rapidly rising vacancy = community decline.
- **Signal**: Community vitality/decline in near-real-time.

### 23. Ookla/M-Lab Internet Speed Data
- **What**: Actual measured broadband speeds (not just advertised availability from FCC)
- **Resolution**: Tile (aggregatable to county) | **Temporal**: Quarterly
- **URL**: https://www.speedtest.net/insights/blog/best-ookla-open-data-projects/ (Open Data), https://www.measurementlab.net/data/
- **Why it matters**: FCC data measures availability, but actual speeds capture the real digital divide. Speed affects WFH viability, streaming/media consumption, and e-commerce participation.
- **Signal**: Digital infrastructure quality as lived experience.

### 24. Zillow Home Value Index (ZHVI) — Free Tier
- **What**: Median home values and month-over-month/year-over-year appreciation by zip/county
- **Resolution**: ZIP/County | **Temporal**: 2000-present, monthly
- **URL**: https://www.zillow.com/research/data/
- **Why it matters**: Housing wealth is the largest asset class for most Americans. Rapid appreciation creates winners (homeowners) and losers (renters) with divergent political interests. Housing cost burden may drive migration patterns that reshape community composition.
- **Signal**: Wealth effect + displacement pressure.

### 25. EIA Energy Production + Fossil Fuel Employment
- **What**: County-level oil/gas/coal production, renewable energy installations, energy sector employment
- **Resolution**: County | **Temporal**: Annual
- **URL**: https://www.eia.gov/opendata/
- **Why it matters**: Fossil fuel dependent communities have intense opposition to energy transition policies. Renewable energy installation (wind/solar farms) creates new economic interests in rural areas that may moderate traditional resource-extraction politics.
- **Signal**: Energy economy identity. Coal counties, oil counties, and wind-farm counties are different communities.

### 26. FDIC Bank Branch + NCUA Credit Union Data
- **What**: Every bank branch and credit union location with deposits
- **Resolution**: Point (aggregatable to county) | **Temporal**: Annual
- **URL**: https://www.fdic.gov/analysis/quarterly-banking-profile, https://www.ncua.gov/analysis/credit-union-corporate-call-report-data
- **Why it matters**: Bank branch closures correlate with economic decline. Credit union vs. big bank density captures community economic character. "Banking deserts" overlap with news deserts and health deserts.
- **Signal**: Financial infrastructure as community vitality proxy.

### 27. County Health Rankings (Robert Wood Johnson Foundation)
- **What**: Composite health rankings — premature death, health behaviors (smoking, obesity, physical inactivity), clinical care access, social/economic factors
- **Resolution**: County | **Temporal**: Annual since 2010
- **URL**: https://www.countyhealthrankings.org/
- **Why it matters**: Health outcomes are a holistic measure of community wellbeing that integrates economic, social, and environmental factors. Premature death rate captures "deaths of despair" plus general community health infrastructure.
- **Signal**: Community wellbeing composite. May capture latent community types that single-variable measures miss.

### 28. BRFSS (Behavioral Risk Factor Surveillance System)
- **What**: State/metro health behavior survey — smoking, exercise, diet, mental health days, insurance coverage, preventive care
- **Resolution**: State/metro (some county estimates via PLACES) | **Temporal**: Annual
- **URL**: https://www.cdc.gov/brfss/, https://www.cdc.gov/places/
- **Why it matters**: CDC PLACES project produces tract-level modeled estimates for 36 health measures. Health behavior patterns (smoking rates, obesity, mental health) correlate strongly with community type and political lean.
- **Signal**: Behavioral health as cultural marker. Smoking rate is one of the strongest correlates of Trump vote share at county level.

### 29. Voter Registration Party Shifts (FL/GA State Data)
- **What**: Net change in registered Democrats, Republicans, and NPAs by county over time
- **Resolution**: County/precinct | **Temporal**: Monthly (FL), periodic (GA)
- **URL**: FL: https://dos.fl.gov/elections/data-statistics/voter-registration-statistics/
- **Why it matters**: Party registration change is a leading indicator of vote share shift. FL's monthly data allows tracking realignment in near-real-time. The massive FL party registration shift (2016-2024) from D+advantage to R+advantage is a core signal.
- **Signal**: Real-time partisan realignment tracking. Leading indicator for election shift.

### 30. OpenElections Project — Downballot Precinct Returns
- **What**: Precinct-level results for state legislature, county commission, sheriff, school board, ballot measures
- **Resolution**: Precinct | **Temporal**: Growing archive
- **URL**: http://openelections.net/
- **Why it matters**: Downballot races reveal local political dynamics that presidential results mask. Sheriff elections capture law-enforcement attitudes. School board elections capture education culture wars. Ballot measures (abortion, marijuana, minimum wage) reveal issue-specific attitudes.
- **Signal**: Issue-level political behavior. Splits between presidential and downballot voting reveal crossover dynamics.

### 31. SBA Loan Data + Paycheck Protection Program
- **What**: Small business loan approval rates, PPP loan distribution by county
- **Resolution**: ZIP/county | **Temporal**: Annual (SBA), 2020-2021 (PPP)
- **URL**: https://www.sba.gov/funding-programs/loans/lender-match, https://data.sba.gov/
- **Why it matters**: PPP distribution was politically contentious and unevenly distributed. Small business formation/lending rates capture entrepreneurial culture. SBA disaster loan rates after hurricanes capture resilience patterns.
- **Signal**: Economic support dependency + small business culture.

### 32. Google Trends — Relative Search Interest
- **What**: Relative search volume for politically relevant terms by DMA (designated market area)
- **Resolution**: DMA (~210 US markets) | **Temporal**: 2004-present
- **URL**: https://trends.google.com/trends/
- **Why it matters**: Search behavior reveals salience — what issues people are actively seeking information about. "Immigration" searches spike in communities experiencing demographic change. "Gas prices" searches correlate with economic anxiety. Proxy for issue salience without survey overhead.
- **Signal**: Issue salience in near-real-time. DMA resolution is coarser than county but captures media market effects.

### 33. BLS Local Area Unemployment Statistics (LAUS)
- **What**: Monthly unemployment rate by county
- **Resolution**: County | **Temporal**: 1990-present, monthly
- **URL**: https://www.bls.gov/lau/
- **Why it matters**: Monthly resolution allows tracking economic conditions between elections. Unemployment trends (improving vs. worsening) matter more than levels for political behavior. The trajectory of unemployment leading into an election may predict shift direction.
- **Signal**: Economic conditions at high temporal resolution. Change over time is the key feature, not the level.

### 34. USDA SNAP/Food Stamp Participation
- **What**: County-level SNAP participation rates and benefit amounts
- **Resolution**: County | **Temporal**: Annual
- **URL**: https://www.fns.usda.gov/pd/supplemental-nutrition-assistance-program-snap
- **Why it matters**: SNAP participation captures economic distress AND attitudes toward government assistance. High-SNAP counties in the South vote Republican despite receiving federal benefits (the "red state paradox"). Change in SNAP participation may signal economic shock.
- **Signal**: Government dependency + economic distress signal.

### 35. Primary Election Competitiveness + Turnout
- **What**: Primary election results, turnout rates, number of contested primaries by party, by county
- **Resolution**: County/precinct | **Temporal**: Varies by state
- **URL**: FL/GA/AL SOS + OpenElections
- **Why it matters**: Competitive primaries signal active party infrastructure. Low primary turnout in one party suggests disengagement. The ratio of R to D primary voters in FL (where primaries are closed) directly measures party enthusiasm.
- **Signal**: Party infrastructure health + engagement intensity. Leading indicator.

---

## Tier 4 — Speculative / Requires Research

### 36. Yelp/Google Places Business Category Density
- **What**: Density of business types by category — BBQ restaurants, yoga studios, gun shops, organic groceries, Dollar Generals, Walmarts, craft breweries
- **Resolution**: Point (aggregatable) | **Temporal**: Current snapshot
- **Why it matters**: Consumer landscape as cultural proxy. Dollar General density correlates with rural poverty. Craft brewery density correlates with educated urban areas. Gun shop density correlates with gun culture. These are "cultural fingerprints."
- **Signal**: Consumer culture as community type signal. Unconventional but potentially powerful for community detection.
- **Feasibility**: Would need Yelp Fusion API or Google Places API. Rate-limited but feasible for county-level aggregation.

### 37. AM/FM Radio Station Format Data
- **What**: Radio station call signs, formats (country, talk radio, NPR, religious, Spanish-language), coverage areas
- **Resolution**: Station (mappable to coverage area) | **Temporal**: Current
- **URL**: FCC license database
- **Why it matters**: Media environment shapes information exposure. Talk radio density correlates with conservative lean. NPR station density correlates with educated professional communities. Spanish-language radio density captures Hispanic community presence.
- **Signal**: Information/media ecosystem as community signal.

### 38. Church Denomination Locations (Beyond RCMS Aggregates)
- **What**: Individual congregation point locations from ARDA, with denomination
- **Resolution**: Point | **Temporal**: 2020
- **URL**: https://www.thearda.com
- **Why it matters**: Goes beyond county-level RCMS aggregates. Megachurch locations specifically (>2000 weekly attendance) signal a particular type of evangelical community. Mosque/temple/synagogue density captures non-Christian religious communities.
- **Signal**: Fine-grained religious landscape within counties.

### 39. Reddit Community Membership by Geography
- **What**: Subreddit participation correlated with geography (from academic datasets like Baumgartner Reddit corpus)
- **Resolution**: Varies | **Temporal**: 2005-2023
- **Why it matters**: Online community participation reveals cultural affinity. r/guns vs. r/yoga membership by geography is a cultural signal. Political subreddit participation patterns capture engagement intensity.
- **Signal**: Digital culture as community proxy. Academic use only; ethical considerations apply.
- **Feasibility**: Low — requires academic partnership or existing processed datasets.

### 40. NHGIS Historical Census (Deep Time)
- **What**: Census data back to 1790, harmonized to consistent geographic boundaries
- **Resolution**: County | **Temporal**: 1790-2020
- **URL**: https://www.nhgis.org/
- **Why it matters**: Deep historical patterns persist. Counties that were plantation-heavy in 1860 still have distinct political patterns today (Acharya et al., "Deep Roots of Modern Southern Politics"). Historical manufacturing concentration predicts current populism.
- **Signal**: Path dependency. Historical community character may explain current shift patterns that recent data cannot.

---

## Priority Matrix

| Source | Tier | Integration Effort | Expected Signal | Recommend |
|--------|------|-------------------|----------------|-----------|
| CDC WONDER Mortality | 1 | Low (API) | Very High | **DO FIRST** |
| COVID Vaccination Rates | 1 | Low (CSV) | Very High | **DO FIRST** |
| FEC Contributions | 1 | Medium (large files) | High | **DO FIRST** |
| BLS QCEW Industry | 1 | Medium (API) | Very High | **DO FIRST** |
| Facebook SCI | 1 | Low (CSV download) | High | Yes |
| USDA Ag Census | 1 | Medium (API) | High | Yes |
| FHFA HPI / Building Permits | 1 | Low (CSV) | Medium-High | Yes |
| BEA Income + RPP | 1 | Low (API) | Medium-High | Yes |
| NCES School Data | 1 | Medium (multiple files) | High | Yes |
| IRS Zip Income | 1 | Low (CSV) | High | Yes |
| County Health Rankings | 3 | Very Low (single CSV) | High | Yes |
| BLS LAUS Unemployment | 3 | Low (API) | Medium-High | Yes |
| USDA SNAP | 3 | Low (CSV) | Medium | Yes |
| Voter Registration Shifts (FL) | 3 | Medium (scraping) | Very High | **DO FIRST** (FL only) |
| CES/CCES Attitudes | 2 | High (MRP modeling) | Very High | Yes, post-MVP |
| News Desert Map | 2 | Low (CSV) | Medium | Yes |
| VA/Military Presence | 2 | Low (ACS + shapefiles) | Medium-High | Yes |
| Primary Competitiveness | 3 | Medium (state SOS) | High | Yes |
| Google Trends | 3 | Medium (API limits) | Medium | Maybe |
| Business Category Density | 4 | High (API) | Unknown-High | Research first |

---

## Recommended Integration Order (Next 5 Fetchers)

1. **CDC WONDER mortality** — deaths of despair are theoretically the strongest missing signal
2. **COVID vaccination rates** — strongest post-2020 political correlate, trivial to fetch
3. **BLS QCEW industry composition** — already documented as "High" priority, defines economic identity
4. **FL voter registration change** — leading indicator, FL-specific but high resolution
5. **County Health Rankings** — single CSV, composite index, immediate value

After these 5, prioritize: FEC contributions, IRS zip income, Facebook SCI, USDA Agriculture.

---

## Expansion Round 2 — 50 Additional Sources

*Added 2026-03-19. Organized by signal category. Hayden's directive: be aggressive and creative. Think full universe of possibilities.*

---

### NYTimes / Major Media Open Data

### 41. NYT Upshot Precinct-Level Presidential Results (2020)
- **What**: Precinct-polygon GeoJSON with Biden/Trump votes, total votes, vote density per km². Generated by NYT from voter file data where official GIS boundaries weren't available.
- **Resolution**: Precinct | **Temporal**: 2020 presidential
- **URL**: `https://int.nyt.com/newsgraphics/elections/map-data/2020/national/precincts-with-results.geojson.gz` (direct download, ~27 states full coverage)
- **Why it matters**: Precinct-level data is the finest geographic resolution available for election analysis — far below county. Within-county variation (exurban vs. suburban vs. urban precincts) reveals community subtypes invisible at county level. FL, GA both have substantial coverage.
- **Signal**: Within-county political heterogeneity. A county with a precinct range of D+60 to R+80 is structurally different from one uniformly R+20.
- **Cost/Effort**: Free, single download. Requires spatial join to census tracts.

### 42. MIT Election Lab Precinct-Level Returns (2016, 2018, 2020, 2022, 2024)
- **What**: Official precinct-level returns for president, senate, governor, state legislature, and local offices. Standardized schema across states. Maintained by MEDSL.
- **Resolution**: Precinct | **Temporal**: 2016-2024
- **URL**: https://electionlab.mit.edu/data (Harvard Dataverse downloads)
- **Why it matters**: The canonical authoritative source for precinct results — more complete and better-curated than NYT for non-presidential races. Includes 2022 governor and 2024 president critical for the shift vector pipeline. Downballot offices (state legislature, sheriff) capture local political dynamics.
- **Signal**: Fine-grained temporal shift tracking. Precinct-level 2016→2020→2024 shift vectors for community discovery at sub-county resolution.
- **Cost/Effort**: Free, Harvard Dataverse. Large files; state-by-state download.

### 43. FiveThirtyEight Political Elasticity Scores
- **What**: County-level swing/elasticity measures quantifying how sensitive a county is to national political waves. Higher elasticity = more swing-voter composition.
- **Resolution**: County/district | **Temporal**: Derived from 2000-2022 cycles
- **URL**: https://github.com/fivethirtyeight/data/tree/master/political-elasticity-scores
- **Why it matters**: Elasticity is theoretically distinct from partisan lean. A highly elastic county has swing voters who move with national environment; an inelastic county has locked-in base voters who don't respond to wave elections. This is a community-type signal the shift vectors can partially detect but elasticity quantifies directly.
- **Signal**: Voter loyalty vs. volatility. Communities of "swing voters" vs. "base voters" are structurally different in how they respond to candidates.
- **Cost/Effort**: Free, small CSV. Direct from GitHub.

### 44. Washington Post Fatal Police Shootings Database
- **What**: Every fatal police shooting in the US since 2015, with victim race, location, whether armed, mental health status, department name.
- **Resolution**: Point (geocoded address, aggregatable to county) | **Temporal**: 2015-present
- **URL**: https://github.com/washingtonpost/data-police-shootings (CC BY-NC-SA 4.0)
- **Why it matters**: Rate of fatal police shootings per capita is a political lightning rod. Counties with high rates relative to their Black population percentage expose racial justice flashpoints. FL (high rate), GA, AL all have meaningful coverage. Post-George Floyd political realignment is partly explained by local policing patterns.
- **Signal**: Law enforcement culture and racial justice salience. Counties where police kill at high rates may show distinct political dynamics post-2020.
- **Cost/Effort**: Free, CC license. Single CSV download.

### 45. CityLab/Bloomberg Urban-Rural Congressional District Classification
- **What**: Classification of all congressional districts on a 6-category urban-rural spectrum (pure urban, urban-suburban, suburban, sparse suburban, small-town, rural), derived from census block-level population density and commuting patterns.
- **Resolution**: Congressional district (mappable to county via areal interpolation) | **Temporal**: 2018 (with 2022 update)
- **URL**: https://github.com/theatlantic/citylab-data/tree/master/citylab-congress
- **Why it matters**: The urban-suburban-rural spectrum captures a major axis of political identity that simple urban/rural dichotomy misses. "Sparse suburban" and "small-town" categories show distinct political trajectories in the Trump era. FL's I-4 corridor counties (Osceola, Polk) are textbook suburban battleground.
- **Signal**: Urban-rural identity gradient. Community type often tracks urban-rural classification more tightly than income or education.
- **Cost/Effort**: Free, GitHub. CC 4.0 license.

---

### Turnout, Voter Behavior & Election Administration

### 46. USPS Change-of-Address Migration Flows (via HUD)
- **What**: Quarter-over-quarter net migration flows between zip codes derived from USPS address change filings. Separate from IRS migration — captures all moves, not just tax filers.
- **Resolution**: ZIP code | **Temporal**: 2010-present, quarterly
- **URL**: https://www.huduser.gov/portal/datasets/usps.html (HUD aggregation of USPS admin data)
- **Why it matters**: Higher frequency than IRS (quarterly vs. annual), captures renters (who are underrepresented in IRS migration because they don't own property). Shows pandemic-era exurban migration in near-real-time. FL's population growth is primarily driven by in-migration — where from matters politically.
- **Signal**: Population composition change velocity. Counties receiving migrants from Democratic metros (NYC → FL) may be shifting purple from the base.
- **Cost/Effort**: Free, HUD portal. ZIP-level, needs aggregation.

### 47. Early Voting / Absentee Split by County (FL State Data)
- **What**: County-level breakdown of votes cast by method: Election Day in-person, early in-person, mail/absentee — split by registered party.
- **Resolution**: County | **Temporal**: 2004-present (FL), 2016-present (GA)
- **URL**: FL: https://dos.fl.gov/elections/data-statistics/elections-data/voter-turnout/ | GA: Secretary of State
- **Why it matters**: Voting method became a partisan proxy post-2020 — Republicans shifted toward Election Day, Democrats toward mail. Party-specific early voting rates signal partisan mobilization infrastructure quality. Counties where the GOP turned out via Election Day vs. mail show different ground-game strategies.
- **Signal**: Partisan mobilization patterns and election administration exposure. Method splits also expose counties to mail ballot rejection rate disparities.
- **Cost/Effort**: Free, state SOS downloads. FL most complete; GA improving.

### 48. Ballot Rejection and Cure Rates (State Data)
- **What**: Rate of mail ballots rejected (due to signature mismatch, missing witness, late arrival) and successfully cured by county. Rejection rates disproportionately affect certain demographics.
- **Resolution**: County | **Temporal**: 2018-present (FL, GA)
- **URL**: FL Div. of Elections, GA Secretary of State; also academic compilations at Harvard Kennedy School Voting Rights Data
- **Why it matters**: Ballot rejection is a direct suppression signal. FL's 2018 Senate race was decided by ~10K votes while ~25K ballots were rejected. High rejection rates in specific counties (often with large Black or Hispanic populations) signal election access barriers that shape political engagement.
- **Signal**: Election access inequity as community-defining feature. Counties with high rejection rates may show depressed Democratic vote share for structural reasons.
- **Cost/Effort**: Free but requires manual state SOS downloads. Some years require FOIA requests.

### 49. Voter Roll Purge Data (Brennan Center / State SOS)
- **What**: Number of voters removed from rolls by county and purge reason (moved, died, inactivity, felony conviction).
- **Resolution**: County | **Temporal**: 2012-present
- **URL**: Brennan Center for Justice annual purge studies + FL/GA/AL SOS raw files
- **Why it matters**: FL purged 340K voters before the 2000 election (the "butterfly ballot" and felon list controversy). GA's aggressive purge activity under Kemp (SOS while running for governor) is a key context for 2018 Abrams-Kemp race. Purge rate is a direct measure of voter suppression intensity and political contestation over ballot access.
- **Signal**: Electoral participation access. Counties with high purge rates may have systematically different electorate compositions than registration numbers suggest.
- **Cost/Effort**: Free. Brennan Center aggregations available as PDFs; state SOS files sometimes available as spreadsheets.

### 50. Redistricting Competitiveness Index (Dave's Redistricting App / Princeton Gerrymandering Project)
- **What**: District-level partisan composition under current and historical maps; gerrymandering efficiency gap and bias scores; competitive district count by state.
- **Resolution**: District (county mappable) | **Temporal**: Per redistricting cycle (2010, 2020)
- **URL**: https://davesredistricting.org/, https://gerrymander.princeton.edu/
- **Why it matters**: Gerrymandering shapes which communities have competitive elections vs. safe seats. Counties packed into safe R or D districts have different political dynamics from competitive swing-district counties. FL's extreme gerrymandering post-2022 (DeSantis-drawn maps) is a major structural shift in which communities matter.
- **Signal**: Electoral competitiveness exposure. Safe-seat counties may show different turnout and shift patterns than competitive districts.
- **Cost/Effort**: Free. DRA has exportable JSON; Princeton publishes CSVs.

---

### Economic Granular Signals

### 51. Trade Adjustment Assistance (TAA) Certifications by County
- **What**: County-level counts of workers certified as job-displaced due to import competition or offshoring under the federal TAA program. Each certification = verified trade-displaced workers.
- **Resolution**: County | **Temporal**: 1975-present (public petitions database)
- **URL**: https://www.dol.gov/agencies/eta/tradeact/taa/taa_search_form.cfm (petition database) + DOL annual reports
- **Why it matters**: Dorn, Autor, Hanson (2013) showed counties more exposed to Chinese import competition shifted toward extreme parties. TAA certifications are a direct, verified measure of trade displacement — not just exposure. AL and rural GA had significant textile/apparel TAA certifications through the 1990s-2000s.
- **Signal**: Verified trade displacement trauma as a community-defining experience. The "China shock" counties are a specific community type with documented political effects.
- **Cost/Effort**: Free, searchable database. Requires scraping/aggregation to county level.

### 52. USDA Rural-Urban Continuum Codes (Beale Codes)
- **What**: 9-category county classification on rural-urban spectrum: metro by population size (1-3), adjacent/non-adjacent micropolitan (4-7), rural (8-9).
- **Resolution**: County | **Temporal**: 1974, 1983, 1993, 2003, 2013, 2023 (every ~10 years)
- **URL**: https://www.ers.usda.gov/data-products/rural-urban-continuum-codes/
- **Why it matters**: Far richer than metro/non-metro binary. Distinguishes counties adjacent to large metros (often politically purple bedroom communities) from isolated small-city and truly rural counties. 2003 vs. 2013 code changes reveal counties reclassified as more or less urban — a structural change signal.
- **Signal**: Rural-urban identity gradient at fine resolution. Code change = community structural shift. Adjacent-rural counties (codes 4,6) are often the key swing communities in FL/GA/AL.
- **Cost/Effort**: Free, single Excel download. Already well-documented; trivial to integrate.

### 53. Amazon Warehouse / Fulfillment Center Locations
- **What**: Addresses and approximate employment levels of Amazon fulfillment centers, sortation centers, delivery stations (DSP), and data centers.
- **Resolution**: Point (aggregatable to county) | **Temporal**: 2012-present (point snapshots)
- **URL**: MWPVL International tracking (https://mwpvl.com/html/amazon.html), enhanced by Indeed/LinkedIn job postings
- **Why it matters**: Amazon warehouses are major employment anchors in rural and exurban counties (e.g., Lakeland FL, Macon GA). They employ large numbers of workers at wages above local retail but below manufacturing. They also bring racial diversity to previously homogeneous areas. The "Amazon effect" on local retail closure is a political economy signal.
- **Signal**: Economic disruption and opportunity simultaneously. Counties landing Amazon facilities may shift in complex ways — new jobs vs. displacement of local retail. AL's recent Amazon buildup is politically understudied.
- **Cost/Effort**: Free (MWPVL tracks all facilities). Manual data compilation; ~1,000 facilities nationwide.

### 54. Dollar Store Density (Dollar General, Dollar Tree, Family Dollar)
- **What**: Location counts and density of Dollar General, Dollar Tree, and Family Dollar stores by county.
- **Resolution**: County (from point locations) | **Temporal**: Current + historical via archived store locators
- **URL**: Store locators + OpenStreetMap + SafeGraph historical patterns
- **Why it matters**: Dollar store density is among the strongest predictors of deep poverty in rural areas — Dollar General explicitly targets areas underserved by grocery stores (food deserts). Dollar General's aggressive rural expansion (~20K stores as of 2026) is a direct measure of retail poverty geography. This is a more current and sensitive poverty indicator than SNAP.
- **Signal**: Deep retail poverty geography. Dollar General presence > Walmart presence signals the bottom of the rural economic hierarchy. Rural AL, GA counties with DG density are a distinct community type.
- **Cost/Effort**: Free via OSM or SafeGraph community data. Some point data requires API access.

### 55. Gig Economy Worker Density (DoorDash / Lyft / Instacart Activity)
- **What**: Estimated gig worker concentration by county, derived from BLS "contingent worker" supplements, academic gig economy research datasets, and Uber/Lyft activity estimates.
- **Resolution**: County (estimated) | **Temporal**: 2017-present
- **URL**: BLS Contingent Worker Supplement (2017, 2021): https://www.bls.gov/cps/contingent-and-alternative-employment-arrangements-faqs.htm | Academic: Katz & Krueger (Princeton), JPMorgan Chase Institute gig income data
- **Why it matters**: Gig workers lack union protections, employer-sponsored benefits, and stable income — a distinct economic precariat. Gig economy participation is high in both urban (delivery, rideshare) and rural areas (TaskRabbit, care work). The political attitudes of gig workers differ from traditional employees.
- **Signal**: Labor precarity and economic insecurity without the "blue collar" identity of manufacturing workers. May predict distinct political behavior — particularly in urban exurbs with high delivery worker density.
- **Cost/Effort**: Free for BLS data. JPMorgan Chase Institute data requires academic access request.

### 56. EPA Energy Burden / Low-Income Energy Affordability Data (LEAD Tool)
- **What**: Census-tract-level energy burden (% of household income spent on energy bills), annual energy cost, and household income for low-income populations.
- **Resolution**: Census tract | **Temporal**: 2018 base year (updated 2022)
- **URL**: https://www.energy.gov/scep/slsc/lead-tool (DOE LEAD Tool, downloadable CSV)
- **Why it matters**: High energy burden (>6% of income) is a poverty signal concentrated in the rural South — exactly FL/GA/AL. It captures both housing inefficiency (old/mobile homes) and income inadequacy. Energy insecurity correlates with both economic distress and political attitudes toward utility regulation and fossil fuels.
- **Signal**: Energy poverty as a community marker. Distinct from absolute income — energy burden reveals structural vulnerability of specific housing stock and climate conditions (high AC demand in FL/AL/GA summers).
- **Cost/Effort**: Free, DOE portal. Tract-level CSV download, nationwide.

---

### Cultural & Consumer Signals

### 57. Vehicle Registration Types — Trucks/SUVs vs. EVs vs. Sedans
- **What**: County-level vehicle registration counts by vehicle type category: pickup trucks, SUVs, passenger cars, and electric/hybrid vehicles.
- **Resolution**: County | **Temporal**: Annual
- **URL**: State DMVs (FL HSMV, GA DOR, AL DOR) + Atlas EV Hub for EV subset; FHWA HPMS for aggregate vehicle counts
- **Why it matters**: Pickup truck ownership is among the strongest cultural markers of rural conservative identity. The truck-to-car ratio at the county level is a political signal that pre-dates MAGA. Meanwhile, EV registration concentration in coastal FL and Atlanta suburbs maps directly onto the college-educated professional community type. These are revealed-preference cultural signals without survey overhead.
- **Signal**: Cultural identity revealed through consumption. FL's EV adoption (high in Miami-Dade, Broward, Palm Beach, low in Panhandle) mirrors the political map almost perfectly.
- **Cost/Effort**: Free from state DMVs (some require FOIA). Atlas EV Hub (atlasevhub.com) aggregates EV data for FL and other states.

### 58. Library Circulation and Usage Statistics (IMLS Public Libraries Survey)
- **What**: Annual per-capita library visits, circulation, electronic resource usage, library funding, branch count, and hours of operation by county-equivalent library system.
- **Resolution**: Library system (~county) | **Temporal**: Annual since 1988
- **URL**: https://www.imls.gov/research-evaluation/data-collection/public-libraries-survey (IMLS, free CSV download)
- **Why it matters**: Library usage rate is a community investment and civic engagement signal. Counties where libraries are well-funded and heavily used have different civic cultures from those where libraries are chronically underfunded. Library funding battles are a proxy for culture war intensity (book banning, LGBTQ collections). FL and TX library systems have been under specific political attack since 2022.
- **Signal**: Civic infrastructure investment and cultural conflict. Low circulation + low funding + book banning activity = a specific type of culture war community.
- **Cost/Effort**: Free, IMLS. Clean CSV, nationwide coverage, trivial to integrate.

### 59. National Survey of Fishing, Hunting, and Wildlife-Associated Recreation
- **What**: County or state-level estimates of hunting and fishing license holders, days spent hunting/fishing, and wildlife watching participants.
- **Resolution**: State (some county estimates) | **Temporal**: Every 5 years (2011, 2016, 2022)
- **URL**: https://wsfrprograms.fws.gov/subpages/nationalsurvey/national_survey.htm (USFWS)
- **Why it matters**: Hunting license rates are a strong proxy for rural, white, male cultural identity — the core of the Trump coalition. States and counties with high hunting participation have distinct gun culture, land use attitudes, and wildlife policy preferences. FL's hunting is concentrated in North Florida (Panhandle, rural central) — exactly where political culture differs most from South Florida.
- **Signal**: Outdoor/hunting culture as community identity marker. Hunting license density predicts Republican lean more robustly than income in rural areas.
- **Cost/Effort**: Free, USFWS. State-level primary; county estimates sometimes available from state wildlife agencies (FL FWC).

### 60. Spotify / Music Streaming Genre Geography
- **What**: Relative popularity of music genres (country, hip-hop, Christian/gospel, reggaeton, classic rock) by metro area or state, from Spotify streaming data.
- **Resolution**: Metro area/DMA | **Temporal**: Annual (Spotify Insights)
- **URL**: Spotify for Developers (aggregate data via academic partnerships), plus academic papers using Spotify API data (e.g., Rentfrow's "Do Re Mi" AAAS 2013)
- **Why it matters**: Rentfrow et al. showed music preferences predict personality traits at the county level — country music correlates with conscientiousness and conservatism; hip-hop correlates with openness. Genre geography is a cultural signal available at fine resolution without survey overhead. Spanish-language music concentration directly identifies Hispanic communities.
- **Signal**: Cultural identity through revealed music preferences. May capture community types that demographic variables describe poorly — the "country music belt" in North FL/South GA/AL is a political identity as much as a genre.
- **Cost/Effort**: Free Spotify API for aggregate genre data; academic data partnership for fine-grained geographic analysis.

### 61. NCAA Sports Fan Geography / College Football Fandom
- **What**: Geographic distribution of college football team fandom by county, derived from Facebook/Twitter followership data or sports analytics firms.
- **Resolution**: County | **Temporal**: ~2020 snapshot
- **URL**: NYT/Upshot sports fandom maps (published 2014 for NFL, college), Twitter API academic access, Morning Consult team polling
- **Why it matters**: College football fandom in the Deep South (FL/GA/AL) is a proxy for community identity that transcends race and income — it's one of the few cross-racial community bonds. Alabama's iron bowl geography (Auburn vs. Alabama fans) maps onto interesting county-level cultural fault lines. SEC fandom signals regional identity.
- **Signal**: Community identity through sports affiliation. The Gator/Seminole/Hurricane divide in FL maps partly onto political geography. The SEC belt has distinct political culture from non-SEC regions.
- **Cost/Effort**: Limited free data (NYT published static maps); Twitter API data requires scraping. Morning Consult publishes periodic team popularity surveys by state.

### 62. Gun Permit / Concealed Carry License Rates
- **What**: County-level concealed carry permit (CCP) issuance rates, applications, and active license holders.
- **Resolution**: County | **Temporal**: Annual
- **URL**: FL Dept. of Agriculture and Consumer Services (publishes CCP by county); GA Attorney General; AL Sheriff's Assoc.; also Crime Prevention Research Center tracks nationally
- **Why it matters**: Concealed carry rate is a direct, legally-recorded measure of gun culture and self-defense identity. FL's shall-issue CCP system means permit rates reflect cultural demand, not bureaucratic gatekeeping. FL has ~2.6M active CCPs (highest per-capita state). County-level variation within FL is dramatic — North FL counties have 3-4x higher rates than South FL.
- **Signal**: Gun culture intensity as community identity. CCP rate predicts Republican vote share at county level almost as well as religious attendance. Key for distinguishing rural North FL (high CCP, low income) from rural South FL (agricultural, Hispanic, lower CCP).
- **Cost/Effort**: Free, state agency downloads. FL most transparent; AL requires FOIA for some counties.

### 63. Farmers Market and Direct-Farm Sales Density
- **What**: Number of farmers markets, CSA operations, and farm stands per county, from USDA Local Food Directories and Census of Agriculture direct sales data.
- **Resolution**: County | **Temporal**: Annual (markets), 5-year (ag census)
- **URL**: https://www.ams.usda.gov/local-food-directories/farmersmarkets (USDA API available); USDA Census of Agriculture Table 2 (direct sales)
- **Why it matters**: Farmers market density is a strong proxy for college-educated, health-conscious, economically comfortable community — the "Whole Foods voter" archetype. In FL, farmers market concentration in Gainesville, Sarasota, St. Pete maps directly onto the purple-to-blue county types. Rural farmers markets signal different community types (local food economy resilience) than urban ones.
- **Signal**: Cultural and economic community type — the artisanal/organic food economy as political identity marker. Distinct from Walmart/Dollar General axis.
- **Cost/Effort**: Free, USDA API. Point data needs county aggregation. Census of Agriculture direct sales is already county-level.

---

### Infrastructure & Access Deserts

### 64. Rural Hospital Closures (Sheps Center / CMS Data)
- **What**: County-level record of rural hospital closures since 2005 — hospital name, closure year, bed count, service type remaining. Maintained by UNC Sheps Center from CMS provider enrollment data.
- **Resolution**: Hospital point / County | **Temporal**: 2005-present (195 closures tracked)
- **URL**: https://www.shepscenter.unc.edu/programs-projects/rural-health/rural-hospital-closures/ (downloadable dataset)
- **Why it matters**: Rural hospital closure is one of the most direct measures of rural healthcare collapse — a key driver of "deaths of despair" and political frustration. AL and GA have among the highest rural closure rates nationally; FL's tourist economy has differential urban/rural hospital access. Closure creates a 30-60 minute emergency travel time increase — directly affects maternal mortality and heart attack survival.
- **Signal**: Healthcare infrastructure collapse as community trauma. Counties that lost their only hospital in the last decade are experiencing a specific type of rural decline that shifts political attitudes.
- **Cost/Effort**: Free, Sheps Center downloadable dataset. Single CSV. Very low integration effort.

### 65. HRSA Health Professional Shortage Areas (HPSA) and Medically Underserved Areas (MUA)
- **What**: Federal designation of areas lacking sufficient primary care, mental health, and dental care providers. Also includes Medically Underserved Populations (MUP) designations.
- **Resolution**: County/sub-county geographic area or facility | **Temporal**: Continuously updated
- **URL**: https://data.hrsa.gov/tools/shortage-area/hpsa-find (HRSA Data Warehouse, GIS downloads available)
- **Why it matters**: HPSA designation = federal recognition of healthcare provider shortage. Mental health HPSAs are particularly relevant — they map closely onto opioid crisis counties and "deaths of despair" geography. FL Panhandle, rural GA, and nearly all of rural AL are blanketed with HPSA designations. This is a direct measure of healthcare access desert.
- **Signal**: Healthcare access deprivation as community stressor. Mental health shortage areas may be the best available proxy for latent opioid/despair vulnerability before mortality data shows up.
- **Cost/Effort**: Free, HRSA GIS shapefile download. Requires spatial aggregation to county.

### 66. Childcare Desert Index (CAP / USDA ERS)
- **What**: Ratio of children to licensed childcare capacity by county — counties with >3 children per licensed childcare slot are "childcare deserts."
- **Resolution**: County | **Temporal**: 2019, 2023 (Center for American Progress), ongoing (USDA ERS)
- **URL**: Center for American Progress: https://www.americanprogressaction.org/childcare-data-download/ | USDA ERS rural childcare access reports
- **Why it matters**: Childcare desert concentration shapes labor force participation (especially women's workforce participation), political attitudes toward childcare policy, and economic opportunity. Rural FL/GA/AL are heavily childcare-scarce. Post-pandemic childcare collapse is a major driver of female labor force exit — which has direct electoral implications.
- **Signal**: Family economic stress and women's economic opportunity. Counties in childcare deserts may show distinct voting patterns among parents of young children — one of the most politically mobile demographic segments.
- **Cost/Effort**: Free, CAP downloadable data. County-level, manageable size.

### 67. Pharmacy Desert Map (RUPRI / NCPA)
- **What**: Distance to nearest retail pharmacy by county, pharmacy closure rate since 2010, pharmacy deserts (rural areas with no pharmacy within 10 miles).
- **Resolution**: County | **Temporal**: 2010-present
- **URL**: Rural Policy Research Institute (RUPRI) pharmacy access analyses; NCPA (National Community Pharmacies Assoc.) closure tracking; CMS provider data for pharmacy locations
- **Why it matters**: Pharmacy deserts overlap with hospital deserts and news deserts. Access to prescription medications (including insulin, blood pressure drugs) is a healthcare access measure that particularly affects the elderly rural population — a politically active demographic. AL has the highest rural pharmacy closure rate of any southern state.
- **Signal**: Healthcare retail infrastructure collapse. Pharmacy desert counties have older, sicker, more healthcare-dependent populations with direct personal stakes in healthcare policy.
- **Cost/Effort**: Free (RUPRI publishes county-level analyses as PDFs, sometimes CSV). Some compilation required.

### 68. Public Transit Availability Index (APTA / FTA)
- **What**: County-level transit availability score: presence of fixed-route bus, commuter rail, light rail; service frequency; transit ridership per capita.
- **Resolution**: County | **Temporal**: Annual (NTD data)
- **URL**: FTA National Transit Database: https://www.transit.dot.gov/ntd/ntd-data | APTA transit agency locator
- **Why it matters**: Car dependency is both an economic burden and a political identity marker. Zero-transit counties have different labor markets (longer commutes, worse access to jobs) and different political cultures than transit-accessible areas. In FL, the I-4 corridor's transit void (despite density) contrasts with Miami's system; GA has MARTA but rural GA has nothing.
- **Signal**: Car-dependency as economic and cultural identity. Transit access stratifies who can access jobs, healthcare, and community — with direct political implications for attitudes toward urban policy.
- **Cost/Effort**: Free, FTA NTD. Some aggregation from agency to county level needed.

### 69. Food Desert / Food Access Atlas (USDA ERS)
- **What**: Census-tract-level measure of low-income populations more than 0.5/1/10 miles from nearest supermarket. Includes food swamp measures (ratio of fast food/convenience stores to healthy options).
- **Resolution**: Census tract | **Temporal**: 2015, 2019 (updates ongoing)
- **URL**: https://www.ers.usda.gov/data-products/food-access-research-atlas/ (downloadable Excel/CSV)
- **Why it matters**: Food access geography maps closely onto poverty and race in the South. Rural AL and GA Black Belt counties have among the highest food desert concentrations in the nation. Food deserts create health burdens (diabetes, obesity) that then show up in political attitudes toward healthcare and government assistance. "Food swamp" measure (fast food density) is distinct from food desert — an urban-specific poverty signal.
- **Signal**: Food access equity as community health and economic stressor. Black Belt food desert concentration is a structural feature of the plantation economy's legacy.
- **Cost/Effort**: Free, USDA ERS. Excel download, tract-level. Very manageable.

---

### Environmental Signals

### 70. PFAS Contamination in Drinking Water Systems
- **What**: Public water system-level PFAS detection results under UCMR 3 (2013-2015), UCMR 5 (2023-2025), and EPA's new Maximum Contaminant Level enforcement. Number of systems with exceedances, per-county.
- **Resolution**: Water system (aggregatable to county) | **Temporal**: 2013-present
- **URL**: EPA UCMR data: https://www.epa.gov/dwucmr/data-download-sixth-unregulated-contaminant-monitoring-rule | EWG PFAS map: https://www.ewg.org/interactive-maps/pfas_contamination/
- **Why it matters**: PFAS contamination is concentrated around military bases (Pensacola/Eglin FL, Fort Benning/Bragg GA) due to AFFF firefighting foam use. It also appears around industrial sites (chemical manufacturing, textile plants) — concentrated in rural GA/AL. Communities facing PFAS contamination have specific political grievances against both the military and chemical industry regulators.
- **Signal**: Environmental justice flashpoint — military and industrial PFAS sources create communities with specific federal government grievances. Pensacola FL PFAS from Eglin/NAS Pensacola is a documented case.
- **Cost/Effort**: Free, EPA download. Requires spatial join from water system to county.

### 71. FEMA National Flood Insurance Program (NFIP) Policy and Claims Data
- **What**: County-level NFIP policy count, total insured value, claims paid, repetitive loss properties. Separate from disaster declarations — ongoing insurance penetration and loss experience.
- **Resolution**: County | **Temporal**: Ongoing (annual)
- **URL**: https://www.fema.gov/about/open/data (FEMA OpenFEMA bulk download)
- **Why it matters**: FL has 40% of all US NFIP policies — flood risk is existential for coastal communities. High repetitive loss counties (low-lying coastal areas) face economic threats from FEMA rate increases and sea level rise that directly affect property values and community viability. Attitudes toward climate adaptation vs. denial correlate with flood risk exposure.
- **Signal**: Climate risk as community defining feature. FL coastal counties with high NFIP concentration have specific economic stake in sea level rise and insurance availability — a political signal distinct from inland counties.
- **Cost/Effort**: Free, FEMA OpenFEMA. County-level bulk CSV.

### 72. Wildfire Risk Index (USFS SILVIS Lab / FSim)
- **What**: County and census tract wildfire hazard potential scores, combining fuel load, weather patterns, and topography. Annual burn area and structure loss.
- **Resolution**: 30m raster (aggregatable to county/tract) | **Temporal**: Annual (fire season data), 30-year normals
- **URL**: USFS Wildfire Hazard Potential: https://www.fs.usda.gov/rds/archive/Catalog/RDS-2020-0016-2 | National Interagency Fire Center: https://www.nifc.gov/fire-information/statistics
- **Why it matters**: While FL/GA/AL don't face the extreme wildfire risks of the West, FL panhandle pine flatwoods, GA Okefenokee region, and AL Bankhead Forest areas have significant wildfire exposure. Prescribed burn policy is politically contested (landowner rights vs. wildfire management). Smoke events affect agriculture and health.
- **Signal**: Land management attitudes and rural property rights politics. Wildfire/prescribed burn counties have specific policy grievances. Less relevant for FL/GA/AL than Western states but non-trivial.
- **Cost/Effort**: Free, USFS raster data. Requires GIS processing to aggregate to county.

### 73. Coal Mine / Power Plant Retirement Map (EIA + Sierra Club)
- **What**: Point locations and employment of operating and retired coal mines, coal power plants, and natural gas plants. Retirement dates and worker counts.
- **Resolution**: Plant/mine point | **Temporal**: 1985-present
- **URL**: EIA Power Plant data: https://www.eia.gov/electricity/data/eia923/ | Sierra Club Beyond Coal tracker | Mine Safety and Health Administration (MSHA) mine data
- **Why it matters**: AL has significant coal employment (Walker County, Jefferson County coalfields). Plant closures create "stranded communities" with specific economic trauma and resentment toward environmental regulation. The link between coal country and Trump realignment is one of the most documented political economy findings of the 2010s.
- **Signal**: Industrial transition trauma. Coal-adjacent counties in AL show distinct shift patterns reflecting deindustrialization resentment. Even in non-coal FL/GA, natural gas plant placement creates economic dependency.
- **Cost/Effort**: Free, EIA and MSHA downloads. Point data needs county aggregation.

### 74. USDA Conservation Reserve Program (CRP) Enrollment
- **What**: Acres of agricultural land removed from production and placed in conservation easements, by county. Also includes Wetland Reserve Program (WRP) and other USDA conservation program enrollment.
- **Resolution**: County | **Temporal**: Annual
- **URL**: USDA Farm Service Agency: https://www.fsa.usda.gov/programs-and-services/conservation-programs/reports-and-rankings/ (county-level spreadsheets)
- **Why it matters**: CRP enrollment is a direct measure of agricultural economic marginal land — land that farmers are paid not to farm. High CRP counties are struggling agricultural communities where government payments substitute for market income. It also captures conservation attitudes (farmers who voluntarily participate vs. those who don't).
- **Signal**: Agricultural decline and government dependency. CRP-heavy counties may be experiencing the last generation of farming — a community existential stress that drives political anger. FL Panhandle and SW Georgia have significant CRP acres.
- **Cost/Effort**: Free, USDA FSA. Annual county spreadsheets.

---

### Historical & Structural Signals

### 75. HOLC Redlining Maps (Mapping Inequality / Richmond Fed)
- **What**: Digitized Home Owners' Loan Corporation (1935-1940) neighborhood security maps ("redlining") for 239 US cities, with grade (A-D) polygons. Overlapping with current census tracts to measure legacy effects.
- **Resolution**: Neighborhood polygon (HOLC zone, mappable to census tract) | **Temporal**: 1935-1940 (historical); legacy effects in current data
- **URL**: https://dsl.richmond.edu/panorama/redlining/ (Digital Scholarship Lab, University of Richmond; GeoJSON download)
- **Why it matters**: Redlining created durable wealth gaps that persist 80+ years later. Nelson et al. (2019) showed HOLC grades predict current demographics, health outcomes, and tree canopy almost perfectly. For FL/GA/AL: Jacksonville, Miami, Atlanta, Birmingham, Mobile all have mapped HOLC zones. Redlined areas show specific patterns of racial wealth gap, political participation, and structural disadvantage.
- **Signal**: Structural racism legacy as community-defining historical force. Tract-level HOLC grade is among the most powerful predictors of current racial wealth disparities and political engagement gaps in Black communities.
- **Cost/Effort**: Free, Digital Scholarship Lab GeoJSON. Requires spatial join to census tracts.

### 76. Historical Slave Population (1860 Census)
- **What**: County-level slave and free population from the 1860 Census, including cotton/tobacco/rice production. Via NHGIS (already in list) or Kaggle cleaned versions.
- **Resolution**: County (historical boundaries, need to crosswalk to current) | **Temporal**: 1860
- **URL**: IPUMS NHGIS historical tables; also Acharya, Blackwell, Sen's "Deep Roots" replication data at Harvard Dataverse
- **Why it matters**: The single strongest predictor of modern white Democratic voting in the South is 1860 slave population share (Acharya, Blackwell & Sen 2018 — "Deep Roots of Modern Southern Politics"). Counties with high historical slave populations have more racially polarized politics today. The Black Belt across AL/GA/MS is named for plantation topsoil — the political and agricultural geography are the same.
- **Signal**: Path dependency — 160-year-old slavery geography predicts modern political polarization. The mechanism is white racial resentment sustained by post-Reconstruction political institutions.
- **Cost/Effort**: Free via NHGIS or Acharya et al. replication data. Low integration effort; the key variable (pct_slave_1860) is a single column.

### 77. Sundown Town Database (Tougaloo College / James Loewen)
- **What**: County and municipality-level database of historically documented "sundown towns" — places that excluded Black residents through law, violence, or custom after dark.
- **Resolution**: Municipality/county | **Temporal**: Historical (most active 1890-1968)
- **URL**: https://justice.tougaloo.edu/ (Tougaloo College database, ~10,000 entries nationally)
- **Why it matters**: Sundown towns created highly racially homogeneous white communities with particular historical attitudes toward racial mixing and civil rights. Their demographic legacy persists. Northern FL and southern GA have documented sundown town histories. This is a distinct historical structural variable from the plantation slavery axis.
- **Signal**: Historical white racial exclusion as community structural feature. Sundown town legacy may explain why certain rural white communities have particularly strong racial anxiety signals even in counties with low current Black population.
- **Cost/Effort**: Free, Tougaloo database. Requires geocoding of municipality names to FIPS. Some manual validation needed.

### 78. Historical Union Density (BLS Unionstats / CPS)
- **What**: State-level union membership rates from Barry Hirsch/David Macpherson's "Union Membership and Coverage Database" (unionstats.com), derived from Current Population Survey. Historical series back to 1964.
- **Resolution**: State (some metro areas) | **Temporal**: 1964-present
- **URL**: http://www.unionstats.com/ (Hirsch-Macpherson database, free download)
- **Why it matters**: Union density collapse is a key mechanism of the working-class political realignment (Leighley & Nagler 2007). States that were highly unionized in 1970 but experienced rapid union decline show the sharpest working-class Democratic → Republican shift. AL's steelworker union legacy vs. right-to-work passage creates a natural experiment. FL is a right-to-work state with historically low union density.
- **Signal**: Labor institutional legacy. Historical union density predicts where working-class politics was once organized around class identity vs. cultural identity. Its collapse predicts the timing of realignment.
- **Cost/Effort**: Free, unionstats.com annual downloads. State-level; requires assumptions to distribute to county.

### 79. NLRB Union Election Results by County
- **What**: Every union representation election filed with NLRB — employer name, location, election result, union, vote counts — since 1970. Includes recent Amazon/Starbucks unionization campaigns.
- **Resolution**: Employer/workplace (aggregatable to county) | **Temporal**: 1977-present
- **URL**: https://www.nlrb.gov/resources/data/graphs-data (NLRB case management system; bulk data available via FOIA or academic datasets)
- **Why it matters**: Recent union election wave (2021-2024 at Amazon, Starbucks, Apple) is concentrated in specific counties. Counties where workers are currently attempting to unionize are sites of active labor conflict — a political economy signal. Historical NLRB data shows where labor organizing was and is active, separate from membership (which includes inactive unions).
- **Signal**: Active labor organizing as community political economy signal. Amazon warehouse counties with union votes may be developing a distinct working-class political identity.
- **Cost/Effort**: Free, NLRB website. Some data requires FOIA; academic datasets (NLRB Elections Database from Cornell ILR) available.

### 80. Lynching Victims Data (Equal Justice Initiative / Monroe Work Project)
- **What**: County-level counts of racial lynching victims from 1877-1950, from Equal Justice Initiative (EJI) "Lynching in America" database and Monroe Work's "Negro Year Book" historical record.
- **Resolution**: County | **Temporal**: 1877-1950 historical
- **URL**: EJI: https://eji.org/research/lynching-in-america/ (county data downloadable via their report); Monroe Work Today project
- **Why it matters**: Lynching geography predicts modern racial attitudes and racial wealth gaps (Cook, Logan, & Parman 2014). Counties with higher historical lynching counts have lower current Black voter registration rates and more racially polarized politics. The mechanism is both direct demographic (Black out-migration from terror) and attitudinal (persistent white racial dominance). FL, GA, AL had among the highest lynching rates nationally.
- **Signal**: Racial violence history as community structural legacy. EJI specifically found FL and AL had the highest per-capita lynching rates in the nation — this maps directly onto current racial political polarization geography.
- **Cost/Effort**: Free, EJI downloadable county data. Low integration effort; single data table.

---

### Immigration & Demographics

### 81. ICE Arrests and Deportations by County (TRAC / ERO)
- **What**: County-level ICE enforcement arrest and removal (deportation) counts. TRAC Syracuse aggregates ICE Enforcement and Removal Operations (ERO) data via FOIA requests.
- **Resolution**: County of arrest | **Temporal**: 2003-present
- **URL**: https://tracreports.org/ (TRAC Immigration; subscription for detailed data, but public summaries free); FOIA releases via MuckRock
- **Why it matters**: ICE enforcement intensity is politically salient and varies dramatically by jurisdiction. Sanctuary policies (Miami-Dade reversed its sanctuary status in 2017) vs. 287(g) cooperation vary by county sheriff. Counties with high ICE activity have politically mobilized immigrant communities AND potentially politically mobilized nativist communities simultaneously.
- **Signal**: Immigration enforcement as community political flash point. FL counties with 287(g) sheriff agreements show distinct political trajectories from sanctuary counties.
- **Cost/Effort**: Free public summaries from TRAC; detailed county data requires subscription (~$20/month academic rate). MuckRock FOIA aggregations available.

### 82. DACA Recipient Concentration (USCIS Data)
- **What**: State-level and some county-level estimates of DACA recipients — their concentration, age, employment sector.
- **Resolution**: State (top-line), county (estimated from American Immigration Council research)
- **URL**: USCIS DACA data: https://www.uscis.gov/DACA | American Immigration Council county-level analysis reports
- **Why it matters**: DACA recipients are concentrated in specific counties (Miami-Dade, Broward, Gwinnett GA, Jefferson AL). Their concentration creates both political constituency pressure (their families vote) and political backlash signals. The uncertainty of their status (DACA program has been repeatedly litigated) creates persistent political salience.
- **Signal**: Immigration community presence and political stakes. High DACA concentration counties in GA (Gwinnett, Cobb) track closely with the suburban Atlanta Hispanic political realignment.
- **Cost/Effort**: Free, USCIS quarterly snapshots. County-level requires academic report compilation.

### 83. H-2A Agricultural Guest Worker Program Data
- **What**: County-level counts of H-2A temporary agricultural visa certifications — workers requested, positions certified, employer names, crop types.
- **Resolution**: County (employer address) | **Temporal**: Annual
- **URL**: DOL OFLC Performance Data: https://www.dol.gov/agencies/eta/foreign-labor/performance (downloadable Excel)
- **Why it matters**: H-2A workers are overwhelmingly from Mexico and Central America, concentrated in rural agricultural counties. FL leads all states in H-2A certifications (~30% of national total) — concentrated in Immokalee (Collier), Homestead (Miami-Dade), and the Glades (Belle Glade/Palm Beach). These are majority-Hispanic agricultural communities with distinct political dynamics. GA's vidalia onion and poultry counties also show high H-2A use.
- **Signal**: Agricultural labor immigration concentration. H-2A counties are distinctive — large temporary immigrant populations, abusive labor conditions, low voter turnout despite large Hispanic population. Captures the "invisible" agricultural labor geography.
- **Cost/Effort**: Free, DOL Excel download. County-level from employer address. Trivial to integrate.

### 84. Refugee Resettlement by County (State Dept. / WRAPS)
- **What**: County-level refugee resettlement placements by year, country of origin, and sponsoring agency.
- **Resolution**: County | **Temporal**: 1975-present
- **URL**: State Dept. Refugee Processing Center (RP): https://www.wrapsnet.org/admissions-and-arrivals/ | Wilson Sheehan Lab for Economic Opportunities (Notre Dame) maintains academic dataset
- **Why it matters**: Refugee resettlement is highly concentrated — most resettlees go to specific urban counties with resettlement agency infrastructure. Atlanta (Clarkston GA is called "the most diverse square mile in America"), Jacksonville FL, and Birmingham AL have significant refugee communities. Resettlement creates both multicultural community building AND (in some areas) political backlash among native-born residents.
- **Signal**: Rapid ethnic community formation as political stress signal. Clarkston GA is a documented case study of community transformation and political tension; replicated in smaller form across the South.
- **Cost/Effort**: Free, State Dept. data download. County-level, annual, CSV format.

---

### Health & Social Signals

### 85. CDC WONDER Natality Data — Birth Rates and Teen Birth Rates
- **What**: County-level birth rates, teen birth rates (15-19), unmarried birth rates, maternal age distribution, prenatal care timing.
- **Resolution**: County | **Temporal**: 1995-2024
- **URL**: https://wonder.cdc.gov/natality.html (free, API accessible)
- **Why it matters**: Teen birth rates are among the strongest single-variable predictors of educational attainment and economic mobility at the county level. High teen birth rates signal communities with limited economic opportunity for women and social conservatism. The Deep South (rural AL, rural GA) has persistently high teen birth rates — a community health signature that predates and predicts political outcomes.
- **Signal**: Community economic opportunity for women as reflected in reproductive behavior. Teen birth rate predicts poverty persistence better than current income, and predicts Republican lean even controlling for income.
- **Cost/Effort**: Free, CDC WONDER API. County-level, clean data.

### 86. Eviction Lab Data — Eviction Rates (Princeton)
- **What**: County-level eviction filing rates and judgment rates (2000-2018), and current eviction tracking from the Eviction Tracking System (ETS, 31 cities).
- **Resolution**: County (historical) / ZIP (ETS) | **Temporal**: 2000-2018 (historical), 2020-present (ETS)
- **URL**: https://evictionlab.org/get-the-data/ (email registration required for download)
- **Why it matters**: Eviction rate is a measure of housing precarity and landlord power that poverty rates miss. High eviction counties have transient populations with low voter registration and high political disengagement. Matthew Desmond's "Evicted" documented how eviction perpetuates poverty in Milwaukee — the same dynamic applies to Atlanta, Jacksonville, Birmingham.
- **Signal**: Housing precarity and political disengagement. High eviction counties may show low Democratic base turnout even in communities with favorable demographics — eviction disrupts voter registration.
- **Cost/Effort**: Free with email registration. County-level CSV. Princeton Eviction Lab is research-grade data.

### 87. Maternal Mortality Rates by County (CDC / AHRQ)
- **What**: County-level maternal mortality rates (deaths per 100,000 live births), racial disparities in maternal mortality, hospital delivery data.
- **Resolution**: County (some suppression in small counties) | **Temporal**: 2016-present
- **URL**: CDC WONDER maternal mortality tables; AHRQ Healthcare Cost and Utilization Project (HCUP); March of Dimes Maternity Care Deserts report
- **Why it matters**: US maternal mortality is among the highest in developed nations and is racially stratified — Black mothers die at 2-3x the rate of white mothers. The "maternity care desert" (counties with no hospital offering obstetric care) is concentrated in rural AL, GA, MS. This is a policy salience issue with specific political implications for healthcare access and reproductive rights politics.
- **Signal**: Healthcare system failure as community political grievance. March of Dimes found 36% of US counties are maternity care deserts — concentrated in the South, directly in the FL/GA/AL study area.
- **Cost/Effort**: Free, CDC WONDER. Some county suppression for small populations. March of Dimes publishes county-level CSV.

### 88. Disability Rate and SSDI Recipiency (SSA)
- **What**: County-level Social Security Disability Insurance (SSDI) and Supplemental Security Income (SSI) recipient rates, disability type distribution.
- **Resolution**: County | **Temporal**: Annual
- **URL**: SSA Annual Statistical Report on SSDI: https://www.ssa.gov/policy/docs/statcomps/di_asr/ | SSA geographic data download
- **Why it matters**: High disability rates are both a health indicator and an economic dependency signal. Rural Appalachian and Deep South counties have disability rates 2-3x the national average — partly reflecting genuine occupational injury (mining, agriculture) and partly reflecting disability as the "last resort" benefit after UI exhaustion. Disability dependency predicts political attitudes toward government benefit programs.
- **Signal**: Long-term economic dependency and working-age health failure. High SSDI counties in AL/GA rural areas represent the intersection of deindustrialization, poor health infrastructure, and intergenerational poverty.
- **Cost/Effort**: Free, SSA downloads. County-level, annual. Clean data.

### 89. Alcohol Outlet Density (TTB / State Licensing)
- **What**: Number of licensed alcohol retailers, bars, and restaurants serving alcohol per county, from state alcohol beverage control (ABC) licensing data and federal TTB.
- **Resolution**: County | **Temporal**: Annual
- **URL**: FL Div. of Alcoholic Beverages & Tobacco; GA DOR; AL ABC Board — all publish licensee databases; also via SafeGraph POI data
- **Why it matters**: Alcohol outlet density correlates with both social disorder (crime, DUI deaths) and community type (bar culture vs. dry counties). Alabama still has dry counties (nearly 1/3 as of 2020) — a direct measure of cultural conservatism. FL's liquor license density in tourist areas vs. rural areas is a stark community type differentiator.
- **Signal**: Cultural conservatism signal (dry vs. wet county) and social disorder proxy. Dry county status is the most direct available measure of traditional Baptist/evangelical cultural governance.
- **Cost/Effort**: Free, state agency databases. Some compilation required; SafeGraph POI historical data is the cleanest aggregation.

---

### Technology & Digital Signals

### 90. Starlink Satellite Internet Adoption (SpaceX / FCC)
- **What**: FCC broadband availability data for Starlink coverage + SpaceX's own coverage maps; state-level subscriber counts where available; academic estimates of rural Starlink adoption from satellite imagery and FCC filings.
- **Resolution**: County (estimated from FCC data) | **Temporal**: 2021-present
- **URL**: FCC National Broadband Map (includes Starlink as fixed wireless); SpaceX SEC filings for subscriber counts; Pew Research rural internet studies
- **Why it matters**: Starlink specifically targets rural counties with no wired broadband — exactly the communities in this study. High Starlink adoption signals rural tech adoption AND willingness to pay premium prices for connectivity. Starlink users may be a distinct socioeconomic subgroup within rural communities (higher income, more connected despite rural location). The Starlink → rural political information environment change is understudied.
- **Signal**: Rural digital connectivity frontier. Starlink adoption distinguishes connected rural communities from fully isolated ones — may predict shifts in information environment and political exposure.
- **Cost/Effort**: Moderate. FCC broadband map includes Starlink provider data. Academic proxies available.

### 91. Solar Panel Installation Rates (Lawrence Berkeley Lab / SEIA)
- **What**: County-level residential and commercial solar photovoltaic installation counts, capacity (kW), and cost trends.
- **Resolution**: County | **Temporal**: 2000-present
- **URL**: Lawrence Berkeley National Lab Tracking the Sun: https://emp.lbl.gov/tracking-the-sun (annual county data, free)
- **Why it matters**: Solar adoption is both an economic signal (payback period economics work best in sunny, high-electricity-cost areas) and a political identity signal. FL is a paradox — one of the sunniest states with historically low residential solar adoption due to utility lobbying. Counties where rooftop solar has penetrated despite utility opposition show distinct political cultures. Solar map mirrors college-education map in FL.
- **Signal**: Clean energy adoption as community political culture marker. Coastal FL + Orlando suburbs have high solar; Panhandle has near-zero. This perfectly tracks the political gradient.
- **Cost/Effort**: Free, LBNL annual release. County-level CSV. Very clean data.

### 92. FCC Spectrum Auction Winners / Wireless Coverage by County
- **What**: Mobile wireless network coverage quality by county — 4G LTE and 5G coverage percentage, number of competing carriers, spectrum holdings.
- **Resolution**: Census block / county | **Temporal**: Annual (FCC semi-annual)
- **URL**: FCC National Broadband Map (wireless layer): https://broadbandmap.fcc.gov/data-download
- **Why it matters**: Mobile-only internet users are concentrated in low-income rural communities where no wired broadband exists. Counties relying on 3G or spotty 4G are information-isolated in ways the FCC broadband availability map (which counts wireline) misses. In rural AL and GA, Verizon/AT&T rural coverage has major gaps — these communities may have fundamentally different information access patterns.
- **Signal**: Mobile internet quality as information environment proxy. Rural coverage quality may predict political information sources (radio > streaming) and engagement patterns.
- **Cost/Effort**: Free, FCC bulk data. Block-level, needs county aggregation.

---

### Protest, Civil Conflict & Social Movements

### 93. ACLED US Protest and Political Violence Database
- **What**: Event-level database of political demonstrations, riots, and political violence in the US, geocoded to city/county. Covers BLM protests (2020), January 6 precursor events, anti-vaccine rallies, etc.
- **Resolution**: Event point (aggregatable to county) | **Temporal**: 2020-present (US coverage)
- **URL**: https://acleddata.com/ (free for researchers with registration; academic license)
- **Why it matters**: Protest activity is a direct measure of political mobilization and community conflict. BLM protest density in 2020 maps onto communities that subsequently showed Democratic shifts. Anti-mask rally density maps onto communities that shifted further Republican. The spatial distribution of political activism is a leading indicator that precedes vote share changes.
- **Signal**: Political mobilization and conflict as community signal. Counties with high BLM protest activity in 2020 showed different 2020 shifts than demographically similar counties without protests.
- **Cost/Effort**: Free with researcher registration. CSV exports by country/date range. Good for 2020-present; limited historical US coverage.

### 94. Washington Post Police Shooting + Fatal Force Database (State-Level Policy Variation)
- **What**: Beyond individual incidents, aggregate state and county policy variables: use-of-force policies, duty to intervene, de-escalation training requirements, civilian review boards, police union contract strength.
- **Resolution**: Department/county | **Temporal**: Current
- **URL**: Campaign Zero Policy Scorecard: https://campaignzero.org/policyplatform/ | Invisible Institute Accountability Data: https://invisible.institute/ | NACOLE (National Association for Civilian Oversight of Law Enforcement)
- **Why it matters**: Police accountability policy structure varies dramatically across counties and correlates with political culture. Counties with strong police unions and no civilian oversight have different political dynamics from reform-minded jurisdictions. The 2020 "defund" debate polarized communities along lines that correlated with existing political geography.
- **Signal**: Law enforcement accountability culture as community political marker. Strong police union counties show specific Republican-leaning patterns among pro-police voters; reform jurisdictions show Democratic-leaning patterns.
- **Cost/Effort**: Free, Campaign Zero and Invisible Institute. City/department level; requires county-level aggregation.

---

### Government & Fiscal Signals

### 95. Census of Governments — Local Tax Revenue and Expenditure
- **What**: Every unit of local government (county, city, special district, school district) with revenue sources (property tax, sales tax, fees), expenditure categories, debt, and employment.
- **Resolution**: Government unit (aggregatable to county) | **Temporal**: Every 5 years (2012, 2017, 2022) + annual for larger governments
- **URL**: https://www.census.gov/programs-surveys/cog.html (Census of Governments)
- **Why it matters**: Local government fiscal structure reveals political philosophy in action. Counties with high property tax revenue fund services differently than those relying on sales taxes. Government employment as a share of county employment captures public sector dependency. Debt per capita signals infrastructure investment vs. fiscal conservatism.
- **Signal**: Revealed fiscal political philosophy. High per-pupil school spending = revealed preference for education investment. Low property tax, high sales tax = regressive fiscal structure. These reveal governing political culture beyond election outcomes.
- **Cost/Effort**: Free, Census download. Quinquennial; large files. Some aggregation from government unit to county.

### 96. School Board Election Results and Culture War Indicators
- **What**: Precinct/county-level school board election results, plus tracking of school board recall attempts, book ban policies, critical race theory (CRT) bans, sex ed curriculum controversies.
- **Resolution**: District | **Temporal**: 2018-present (culture war era)
- **URL**: OpenElections (for election results) + Banned Books Week ALA database + PEN America Index of School Book Bans + FL Dept. of Education curriculum objections log
- **Why it matters**: School board elections became the primary battleground for culture war politics after 2020. Florida has the most documented book banning activity (Duval, Clay, Brevard counties). School board election results predict the trajectory of culture war intensity — they are leading indicators of the community's future political direction.
- **Signal**: Culture war intensity and trajectory. A county where the culture war slate won the school board in 2022 is likely to show increased Republican shift in 2024 — and vice versa for reform slate victories.
- **Cost/Effort**: Free, but requires multi-source compilation. PEN America and ALA publish ban databases. FL DoE has its own public log of curriculum objections.

### 97. Small Area Income and Poverty Estimates (SAIPE) — Annual County Poverty Rates
- **What**: Annual county-level poverty rate and median household income estimates, updated every year using tax data and survey methods. More current than ACS 5-year estimates.
- **Resolution**: County | **Temporal**: Annual, 1989-present
- **URL**: https://www.census.gov/programs-surveys/saipe.html (Census Bureau)
- **Why it matters**: Annual updates make SAIPE more temporally sensitive than ACS 5-year estimates — can track recession impacts (2008-2010 spike), pandemic effects (2020 dip due to stimulus), and recovery trajectory. Year-over-year poverty change may predict political mood better than absolute poverty level.
- **Signal**: Economic welfare trajectory at high temporal resolution. Poverty rate change between election cycles may predict shift direction better than the level.
- **Cost/Effort**: Free, Census API. Annual county-level. Trivial to integrate.

---

### Social Mobility & Opportunity

### 98. Chetty/Raj Opportunity Atlas — Upward Mobility by County (Race/Income)
- **What**: County-level estimates of upward economic mobility for children raised in the bottom quintile, broken down by race (Black, white, Hispanic) and parent income. Based on 20M+ IRS tax records.
- **Resolution**: County/census tract | **Temporal**: Birth cohorts 1978-1983 (now adults)
- **URL**: https://opportunityatlas.org/ (downloadable CSV) | https://opportunityinsights.org/data/
- **Why it matters**: This is distinct from the Opportunity Insights Social Capital data already cataloged. The Opportunity Atlas specifically measures whether poor children escape poverty — and crucially does so by race. A county with high Black child mobility but low white child mobility is structurally different from a county where both are low. This racial mobility gap is a direct measure of structural racism's economic effect.
- **Signal**: Intergenerational economic mobility as community defining feature. Low-mobility counties (especially for Black children) in GA/AL predict political disengagement among Black voters and political frustration among white voters simultaneously.
- **Cost/Effort**: Free, direct CSV download. County and tract level. Clean data.

### 99. Social Capital Project — Congressional Subcommittee County Data
- **What**: JD Vance-era Senate subcommittee data on social capital, civic engagement, volunteering, associational membership, and trust by county. Based on multiple survey sources.
- **Resolution**: County | **Temporal**: ~2018 snapshot
- **URL**: https://www.jec.senate.gov/public/index.cfm/republicans/2018/4/the-geography-of-social-capital-in-america (Joint Economic Committee, Republican staff)
- **Why it matters**: This is a right-of-center academic project that produced county-level social capital estimates — covering civic participation, family stability, community cohesion, and institutional trust. It's conceptually distinct from Chetty's mobility data: it measures community-level social infrastructure rather than individual economic outcomes.
- **Signal**: Community cohesion and institutional trust as political moderators. Low social capital counties may show more populist/anti-institutional political behavior.
- **Cost/Effort**: Free, JEC website. County-level, single download.

---

### Deep Historical

### 100. Historical Migration Data — Great Migration Destination Counties
- **What**: County-level data on Black population change during the Great Migration (1910-1970) — origin counties in the South and destination counties in the North/West, with net migration estimates.
- **Resolution**: County | **Temporal**: 1910-1970 (decennial census)
- **URL**: IPUMS NHGIS historical census tables; Boustan (2017) "Competition in the Promised Land" replication data (Princeton)
- **Why it matters**: The Great Migration depopulated rural Deep South Black communities while creating the Northern Black political base. Counties that lost large Black populations in the Great Migration show persistent structural effects — less Black political power, different racial composition of the political coalition. Leah Boustan's work shows specific Great Migration flow data at county level.
- **Signal**: Demographic path dependency. Which southern counties "sent" Black migrants to Chicago/Detroit vs. which retained Black populations shapes current racial political geography. FL (Miami's Overtown, Jacksonville) vs. Black Belt AL/GA shows this divergence clearly.
- **Cost/Effort**: Free, NHGIS. Historical census race tables by county. Requires crosswalking to current boundaries.

### 101. Reconstruction-Era Black Officeholding (Freedom on the Move / UNC)
- **What**: County-level records of Black elected officials during Reconstruction (1865-1876) and during the Second Reconstruction (1965-2000).
- **Resolution**: County | **Temporal**: 1865-1876 (Reconstruction), 1965-2000 (post-VRA)
- **URL**: "Freedom on the Move" (Cornell, Yale, UNC) escaped slave advertisement database adjacent; Foner's "Freedom's Lawmakers" replication data; Joint Center for Political and Economic Studies Black elected officials series
- **Why it matters**: Counties with strong Reconstruction-era Black political representation were systematically destroyed during Redemption (1876-1900). The geographic pattern of Redemption violence and Black political erasure predicts modern racial political polarization. Counties where Black political power was never restored after Redemption have distinct political structures.
- **Signal**: Democratic institutional legacy of racial political exclusion. The contrast between counties that had Black sheriffs/legislators during Reconstruction vs. those that didn't predicts modern Black political mobilization patterns.
- **Cost/Effort**: Moderate. Academic datasets; some manual compilation from historical records.

---

*End of Expansion Round 2. Total ideated sources: 101 (40 original + 61 new).*

---

## Expansion Round 3 — GitHub Repositories & Tools

*Added 2026-03-19. Per Hayden's directive: "GitHub has a myriad of projects that may have data sources or chunks of tools we could use."*

These are GitHub repos that either contain pre-compiled county-level data we can ingest directly, or tools that simplify fetching data we've already identified.

---

### Pre-Compiled County-Level Datasets

### 102. CountyPlus — Open-Source County Panel Dataset
- **What**: 3,000+ US counties × years 2003-2019. Pre-compiled panel with economic, demographic, health, and social variables from all major public sources (BEA, BLS, Census, CDC, USDA).
- **Resolution**: County | **Temporal**: 2003-2019
- **URL**: https://github.com/Clpr/CountyPlus
- **Why it matters**: Single-download dataset that consolidates many of the individual sources we've listed (BEA income, BLS unemployment, USDA rural codes, CDC health). Could bootstrap feature engineering for 10+ Tier 1/2 sources with one integration. Panel format means temporal change features are pre-aligned.
- **Signal**: Meta-source — accelerates integration of multiple individual sources.
- **Cost/Effort**: Free. Single CSV/parquet download. May need updating beyond 2019.

### 103. JsonOfCounties — Multi-Source County Data in JSON
- **What**: County-level data including BEA income, presidential election results, demographics, education, and health metrics compiled into JSON format.
- **Resolution**: County | **Temporal**: Various
- **URL**: https://github.com/evangambit/JsonOfCounties
- **Why it matters**: Pre-joined dataset covering income, elections, education, health — ready to cross-reference against our shift vectors. JSON format may need conversion but the joining work is done.
- **Signal**: Convenience source — validates our own feature builds and fills gaps.
- **Cost/Effort**: Free. JSON files, trivial conversion.

### 104. US County Level Election Results 2008-2024
- **What**: County-level presidential results for 2008, 2012, 2016, 2020, 2024 from NYT, Guardian, Politico, Fox News. Standardized CSV format.
- **Resolution**: County | **Temporal**: 2008-2024
- **URL**: https://github.com/tonmcg/US_County_Level_Election_Results_08-24
- **Why it matters**: Pre-compiled 5-cycle presidential results at county level. Could supplement VEST data for quick shift vector computation or validation. Already cleaned and standardized — no scraping needed.
- **Signal**: Validation source for our VEST/MEDSL election returns. Also extends to 2008/2012 which we don't currently use.
- **Cost/Effort**: Free. Single CSV per cycle.

### NYTimes Open Source Election Maps

### 105. NYT 2024 Presidential Precinct Map (NEW — not in prior list)
- **What**: Precinct-level TopoJSON/GeoJSON results for the 2024 presidential election. Published by NYT as open data. Includes vote totals, margins, and precinct boundaries.
- **Resolution**: Precinct | **Temporal**: 2024
- **URL**: https://github.com/nytimes/presidential-precinct-map-2024
- **Why it matters**: This is the 2024 companion to the 2020 precinct data already listed (#41). Having both years at precinct level enables sub-county shift vector computation — a major upgrade from county-level shifts. FL and GA have substantial coverage.
- **Signal**: Sub-county shift vectors. The highest-resolution temporal shift data publicly available.
- **Cost/Effort**: Free, direct GeoJSON download. Requires spatial join to tracts.

### 106. NYT Upshot 2020 Presidential Precinct Map
- **What**: GeoJSON with Biden/Trump vote totals per precinct, plus generated precinct boundaries from L2 voter file.
- **Resolution**: Precinct | **Temporal**: 2020
- **URL**: https://github.com/TheUpshot/presidential-precinct-map-2020
- **Why it matters**: Already listed as #41 but noting the actual GitHub repo for direct download. GeoJSON format with GEOID = county FIPS + precinct ID.
- **Signal**: See #41.
- **Cost/Effort**: Free.

### 107. Election Geodata — Precinct Shapes Archive
- **What**: Comprehensive archive of precinct boundary shapefiles across multiple election cycles. Community-maintained with contributions from election officials and academics.
- **Resolution**: Precinct | **Temporal**: Multiple cycles
- **URL**: https://github.com/nvkelso/election-geodata
- **Why it matters**: The boundary files needed to geolocate precinct results from OpenElections or MEDSL. Without boundaries, precinct results can't be mapped to tracts.
- **Signal**: Infrastructure — enables sub-county analysis from multiple data sources.
- **Cost/Effort**: Free. Shapefile format, varies by state/year.

### Health & Opioid Crisis Data

### 108. OEPS — Opioid Environment Policy Scan (GeoDaCenter/UChicago)
- **What**: Multi-scale data warehouse (tract, zip, county, state) covering opioid prescribing rates, naloxone access, treatment facility locations, MAT provider counts, plus socioeconomic context (ACS, health, employment). Pre-built for opioid research.
- **Resolution**: Census tract to state | **Temporal**: Various (2015-2022)
- **URL**: https://github.com/GeoDaCenter/opioid-policy-scan
- **Why it matters**: The most comprehensive pre-compiled opioid crisis dataset at tract level. Includes treatment access (buprenorphine providers, MAT facilities), prescribing rates, and naloxone pharmacy availability — all at sub-county resolution. These are direct measures of the "deaths of despair" crisis that our CDC WONDER mortality fetcher captures from the outcome side.
- **Signal**: Opioid crisis input variables (treatment access, prescribing) complement mortality output variables. Counties with high mortality but low treatment access are a distinct community type from those with high mortality and high treatment access.
- **Cost/Effort**: Free. Pre-built CSV files organized by spatial scale. Very low effort.

### 109. US Overdoses Analysis Database
- **What**: County-level opioid mortality 1999-2015 merged with ACS income, medical insurance, SAMHSA drug use surveys, naloxone/buprenorphine dispensing data.
- **Resolution**: County/state | **Temporal**: 1999-2015
- **URL**: https://github.com/mattkiefer/us-overdoses
- **Why it matters**: Pre-joined dataset linking overdose mortality to economic and healthcare access variables. The temporal range (1999-2015) captures the pre-Trump opioid crisis that shaped the 2016 political realignment.
- **Signal**: Historical opioid crisis trajectory — counties where the crisis peaked before 2016 may have shifted differently from those where it peaked later.
- **Cost/Effort**: Free. SQLite + CSV format.

### Political Science Research Data

### 110. PolData — Master Index of Political Datasets
- **What**: Curated catalog of 300+ political science datasets across 15 categories: elections, parties, institutions, policy, public opinion, media, conflict, etc. Each entry includes topic, coverage, URL, and format.
- **Resolution**: Varies | **Temporal**: Varies
- **URL**: https://github.com/erikgahner/PolData
- **Why it matters**: This is a meta-source — a research-grade index of political datasets that may contain sources we haven't identified. Worth scanning the full list for FL/GA/AL-relevant county-level data. Maintained by a political scientist.
- **Signal**: Discovery accelerator — may identify 5-10 additional sources relevant to our specific geographic focus.
- **Cost/Effort**: Free. Markdown index file.

### 111. US Polling Places Database (Center for Public Integrity)
- **What**: Standardized geocoded polling place locations for general elections across multiple states and years. Includes address, precinct assignment, and temporal changes.
- **Resolution**: Point (polling place) | **Temporal**: 2012-2020
- **URL**: https://github.com/PublicI/us-polling-places
- **Why it matters**: Polling place closures and relocations are a documented voter suppression mechanism. Counties that reduced polling places between elections show lower turnout, particularly in minority communities. GA's 2018 election featured specific polling place closure controversies (Randolph County).
- **Signal**: Election access infrastructure as political participation signal. Polling place distance increase predicts turnout decline.
- **Cost/Effort**: Free. CSV format, geocoded.

### Tools for Data Fetching

### 112. censusdis — Pythonic Census API Wrapper
- **What**: Python package for programmatic access to all 1,500+ Census Bureau API endpoints. Handles geography hierarchies, variable discovery, and data download with pandas integration.
- **Resolution**: All Census geographies (tract to national) | **Temporal**: All Census/ACS years
- **URL**: https://github.com/censusdis/censusdis
- **Why it matters**: Could replace our manual ACS fetching with a more robust, discoverable API wrapper. Supports geographic boundary downloads alongside data — useful for spatial joins. More Pythonic than the `census` package.
- **Signal**: Tool — accelerates Census/ACS data integration and makes it easier to explore new Census tables.
- **Cost/Effort**: Free. `pip install censusdis`. Could refactor our ACS pipeline to use this.

### 113. censusapi — R Census API Wrapper
- **What**: Lightweight R package for all Census Bureau APIs. Supports data discovery, all geographies, and all endpoints including SAIPE, BDS, CBP, ACS, Decennial.
- **Resolution**: All Census geographies | **Temporal**: All available years
- **URL**: https://github.com/hrecht/censusapi
- **Why it matters**: For our R pipeline (MRP/propagation), this provides native R access to Census data without Python intermediary. Particularly useful for SAIPE annual poverty data (source #97) and BDS business dynamics (source #17).
- **Signal**: Tool — R-native Census access for the propagation pipeline.
- **Cost/Effort**: Free. `install.packages("censusapi")`.

---

### Priority GitHub Sources to Integrate First

| Source | # | Why First | Effort |
|--------|---|-----------|--------|
| CountyPlus panel | 102 | Bootstraps 10+ features at once | Very Low |
| NYT 2024 precinct map | 105 | Sub-county shift vectors | Medium |
| OEPS opioid scan | 108 | Tract-level crisis data, pre-built | Very Low |
| PolData index | 110 | May reveal new sources | Research only |
| censusdis tool | 112 | Improves all Census fetching | Medium (refactor) |

---

*End of Expansion Round 3. Total ideated sources: 113 (101 prior + 12 new from GitHub survey).*
