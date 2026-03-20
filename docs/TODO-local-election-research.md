# TODO: Local Election Data Research

## Goal

Research availability of sub-federal election returns (state legislature, county commission, school board, ballot measures) for FL, GA, and AL. Local elections may provide additional shift dimensions that capture political dynamics invisible in federal races.

## Acceptance Criteria

- [ ] `docs/LOCAL_ELECTION_DATA.md` exists with:
  - Inventory of available local election data by state and office type
  - Data access method (API, bulk download, FOIA, scraping)
  - Geographic resolution (precinct, county, district)
  - Temporal coverage (which years available)
  - Assessment of usability for the shift-based model
  - Recommendation: which sources are worth integrating vs. too noisy/sparse

## Sources to Research

### State Election Offices
- [ ] **Florida Division of Elections** — precinct-level results archive, state legislature and ballot measures
- [ ] **Georgia Secretary of State** — county-level results, runoff elections (unique to GA)
- [ ] **Alabama Secretary of State** — county-level results archive

### Aggregator Projects
- [ ] **OpenElections** (openelections.net) — standardized precinct-level results, multi-state coverage
- [ ] **MEDSL** — may have state legislative returns in addition to federal
- [ ] **Ballotpedia** — candidate and results data, API availability

### Academic Datasets
- [ ] **State Legislative Election Returns (1967-present)** — Klarner dataset at Harvard Dataverse
- [ ] **Carl Klarner's state-level data** — governor, state leg, unified government indicators

### Value Assessment Questions
- Do local elections add information beyond what federal elections provide?
- Are state legislative districts mappable to counties/tracts?
- How much additional shift dimension coverage do local races add?
- Is the data quality sufficient (uncontested races, missing precincts, varying reporting)?

## Output

Produce `docs/LOCAL_ELECTION_DATA.md` with full inventory and integration recommendations.
