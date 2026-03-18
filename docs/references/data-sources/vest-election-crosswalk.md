---
source: https://dataverse.harvard.edu/dataverse/vest
captured: 2026-03-18
version: Free access covers 2016, 2018, 2020. 2022+ is paywalled.
---

# VEST Election Returns (UF / Harvard Dataverse)

Voting and Election Science Team, University of Florida. Precinct-level election results with spatial data enabling crosswalk to census geographies.

## Access
Free download for 2016–2020, no account required.
URL: https://dataverse.harvard.edu/dataverse/electionscience
**2022 and 2024 are behind a paid subscriber paywall** — UF plans to make them public again when 2030 redistricting nears.

## Working Historical Set
Free VEST covers 2016, 2018, 2020. **2022 is paywalled.**
For 2022 county-level results without spatial geometry, use MEDSL (see below).
**Working election set for this project (VEST): 2016 + 2018 + 2020.**

## 2022 Data Alternative: MEDSL GitHub
For 2022 validation, use the MIT Election Data + Science Lab precinct returns.
Precinct-level CSV with `county_fips` attached — aggregate to county with a groupby.
No spatial join required (county-level only, not tract-level).
Source: https://github.com/MEDSL/2022-elections-official/tree/main/individual_states
Files: `2022-fl-local-precinct-general.zip`, `2022-ga-local-precinct-general.zip`
Filter: `office == 'GOVERNOR'`, `stage == 'gen'`
Mode dedup: use `mode == 'TOTAL'` rows where present; otherwise sum non-TOTAL modes.
Column: `votes` (not `candidatevotes`); compute totalvotes as sum across candidates.
**Limitation: county-level only. Cannot extend Stan tract-level covariance training.**

## Harvard Dataverse File IDs (direct download)

| Year | State | File ID | DOI |
|------|-------|---------|-----|
| 2016 | AL | 4751068 | 10.7910/DVN/NH5S2I |
| 2016 | FL | 12070343 | 10.7910/DVN/NH5S2I |
| 2016 | GA | 11070010 | 10.7910/DVN/NH5S2I |
| 2018 | AL | 4751072 | 10.7910/DVN/UBKYRU |
| 2018 | FL | 12070358 | 10.7910/DVN/UBKYRU |
| 2018 | GA | 11070036 | 10.7910/DVN/UBKYRU |
| 2020 | AL | 4751074 | 10.7910/DVN/K7760H |
| 2020 | FL | 12070362 | 10.7910/DVN/K7760H |
| 2020 | GA | 11070054 | 10.7910/DVN/K7760H |

Download URL: `https://dataverse.harvard.edu/api/access/datafile/{file_id}`

## Office Selection Per Year

| Year | Office | Column pattern | Notes |
|------|--------|----------------|-------|
| 2016 | Presidential | `G16PRE*` | Output prefix: `pres_` |
| 2018 | Gubernatorial | `G18GOV*` | No presidential race. Output prefix: `gov_`. **Alabama 2018 governor race was uncontested (Ivey, R)**. AL dem share will be near-zero; flag as unreliable for covariance estimation. |
| 2020 | Presidential | `G20PRE*` | Output prefix: `pres_` |

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

**1. VEST 2020 uses VTD-level GEOIDs, NOT block-level.**
AL 2020 GEOID20 format: `01013000100` (11 digits: state+county+VTD code). This is the same length as a census tract GEOID (11 digits: state+county+tract) but is a completely different geography. `GEOID20[:11]` gives a VTD FIPS, not a tract FIPS. A spatial join is required.

Note: FL 2020 has **no GEOID20 column at all** — different column structure. GA 2020 also uses non-block GEOIDs. Assume spatial join is always required for 2020 data; only treat string-slice as an optimization if you confirm block-level GEOIDs in a specific release.

**2. Centroid-based spatial join misses ~27% of tracts.**
Assigning each precinct to the tract containing its centroid gives only 73% coverage — large rural precincts span multiple tracts, but only the tract containing the centroid receives votes. The other tracts in that precinct get zero votes assigned. This is a silent error for populated tracts. Use area-weighted allocation instead: for each precinct, compute the fraction of the precinct's area that overlaps each tract, then split votes by that fraction. Area-weighted allocation achieves 99%+ coverage.

**3. Remaining uncovered tracts (35 in FL) are uninhabited.**
After area-weighted allocation, 35 FL tracts remain uncovered — all have `pop_total = 0` in ACS data. These are water bodies, offshore tracts, or other uninhabited areas. Not a data quality issue; leave them as NaN in the joined dataset.

**4. Use EPSG:5070 (NAD83 Conus Albers) for area computation.**
VEST data CRS varies by state: AL uses EPSG:4269, GA uses EPSG:4019, FL uses EPSG:4269. All must be projected to an equal-area CRS before computing overlap areas. EPSG:5070 (NAD83 Conus Albers) is appropriate for the contiguous US. Failure to reproject will produce slightly incorrect weights due to angular distortion at this scale.
