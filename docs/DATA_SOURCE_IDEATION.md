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
