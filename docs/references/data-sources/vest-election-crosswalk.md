---
source: https://dataverse.harvard.edu/dataverse/vest
captured: 2026-03-18
version: Covers 2016, 2018, 2020, 2022 general elections; all 50 states
---

# VEST Election Returns (UF / Harvard Dataverse)

Voting and Election Science Team, University of Florida. Precinct-level election results with spatial data enabling crosswalk to census geographies.

## Access
Free download, no account required.
URL: https://dataverse.harvard.edu/dataverse/vest

## States Needed
- Florida: search `florida_[year]_vest`
- Georgia: search `georgia_[year]_vest`
- Alabama: search `alabama_[year]_vest`

## Format
Shapefile or GeoPackage. Contains:
- Precinct boundary polygons
- Vote totals by candidate and party for each precinct
- A block-level join field (typically `GEOID20` — 15-digit census block FIPS)

## Aggregation Strategy to Tract
Census block FIPS (15 digits): `[2-digit state][3-digit county][6-digit tract][4-digit block]`
Census tract FIPS (11 digits) = first 11 characters of block FIPS.

```python
vest_df['tract_geoid'] = vest_df['GEOID20'].str[:11]
tract_votes = vest_df.groupby('tract_geoid')[vote_columns].sum()
```

**Caveat**: Some VEST releases use precinct-level geometries with block apportionment notes rather than true block-level records. Check the data dictionary for each state/year release before assuming this aggregation works directly. If the GEOID is at precinct level, a spatial join against census block centroids is required.

## Citation Format
`Voting and Election Science Team, [YEAR], "[State] [Year] Precinct-Level Election Results", Harvard Dataverse, V[version]`

## Why VEST Solves the Hard Problem
Election returns are reported by precinct. Precincts and census tracts are independent boundaries with no alignment. VEST has done the spatial allocation from precinct to census block for all 50 states — the hardest part of this crosswalk is already done. We aggregate blocks to tracts.

## Gotchas
*Populated as failures are encountered in practice. First entry goes here the first time the VEST data surprises us.*
