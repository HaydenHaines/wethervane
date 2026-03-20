# TODO: Data Source Expansion Research

## Goal

Research and evaluate additional data sources for community characterization (description layer), expanding beyond the current ACS + RCMS + IRS migration + LODES baseline. Produce a comprehensive evaluation document.

## Acceptance Criteria

- [ ] `docs/DATA_SOURCES_EXPANSION.md` exists with:
  - Each source evaluated for: availability, cost, geographic resolution, temporal coverage, API/download method
  - Priority ranking (Tier 1 = immediate value, Tier 2 = nice-to-have, Tier 3 = long-term)
  - "Already integrated" section listing current sources and their status

## Sources to Research

From the ideation doc (`docs/DATA_SOURCE_IDEATION.md`) and spec:

### Demographic/Economic
- [ ] **RCMS 2020** (already integrated) — religious congregation membership
- [ ] **LODES** (already integrated) — commuting flows
- [ ] **IRS SOI** (already integrated) — migration flows
- [ ] **BLS QCEW** (already integrated) — quarterly census of employment and wages
- [ ] **BEA Regional Price Parities** — cost-of-living at metro/county level
- [ ] **USDA ERS Rural-Urban Continuum** — county-level rural/urban classification
- [ ] **USDA Atlas of Rural and Small-Town America** — food access, poverty, education
- [ ] **County Health Rankings** (already integrated) — health outcomes and behaviors

### Political/Electoral
- [ ] **FEC individual contributions** — zip-level ActBlue/WinRed ratio, partisan thermometer
- [ ] **FL voter registration** (already integrated) — party registration by county
- [ ] **CDC WONDER mortality** (already integrated) — causes of death by county
- [ ] **CDC COVID vaccination rates** (already integrated) — county-level vax rates
- [ ] **State-level voter registration** (GA, AL) — where available

### Infrastructure/Social
- [ ] **FCC broadband** — fixed broadband deployment by census tract
- [ ] **NCES school district data** — school finance, enrollment, demographics
- [ ] **Zillow/ACS property values** — median home values
- [ ] **Facebook Social Connectedness Index (SCI)** — county-to-county social ties
- [ ] **NCHS urban-rural classification** — 6-level metro/nonmetro scheme

## Output

Produce `docs/DATA_SOURCES_EXPANSION.md` with evaluation matrix and integration plan.
