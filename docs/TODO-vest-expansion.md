# TODO: VEST 2012 + 2014 Tract-Level Data Expansion

## Goal

Pull VEST 2012 and 2014 election data, crosswalk from 2010 census tracts to 2020 census tracts, and produce harmonized parquet files matching the existing VEST schema.

This enables future tract-level community discovery with additional election cycles, adding 6 shift dimensions (pres 2012->2016 D/R/turnout + gov 2014->2018 D/R/turnout) to the existing 9-dim tract shift vectors when tract-level refinement is prioritized.

**Note:** The county-level model already covers 2000-2024 via MEDSL + Algara/Amlani (30 training dims). This task is specifically for tract-level data, which is a future refinement layer (see ADR and CLAUDE.md Key Decisions Log).

## Acceptance Criteria

- [ ] `data/assembled/vest_tracts_2012.parquet` exists with columns:
  - `tract_geoid` (11-digit string, 2020 census tract GEOID)
  - `pres_dem_share_2012` (float, D two-party share)
  - `pres_total_2012` (int, total votes)
- [ ] `data/assembled/vest_tracts_2014.parquet` exists with columns:
  - `tract_geoid` (11-digit string, 2020 census tract GEOID)
  - `gov_dem_share_2014` (float, D two-party share)
  - `gov_total_2014` (int, total votes)
- [ ] Both files cover FL + GA + AL tracts (same geographic scope as existing VEST files)
- [ ] Vote totals crosswalked via area-weighted interpolation using Census relationship file
- [ ] Unit tests validate crosswalk logic, schema compliance, and vote total preservation

## Key Challenge: 2010 -> 2020 Tract Boundary Crosswalk

VEST 2012 and 2014 data uses 2010 census tract boundaries. Our pipeline expects 2020 tract GEOIDs. Census provides official relationship files:

- **Source:** Census Bureau Relationship Files (https://www.census.gov/geographies/reference-files/time-series/geo/relationship-files.html)
- **File:** 2010-to-2020 Census Tract Relationship File
- **Method:** Area-weighted interpolation — distribute votes from 2010 tracts to 2020 tracts proportional to overlap area (or housing unit weights if available)

### Crosswalk Steps
1. Download Census 2010->2020 tract relationship file for FL, GA, AL
2. For each 2010 tract, compute fraction of area/housing units falling in each 2020 tract
3. Distribute vote counts proportionally: `votes_2020_tract = sum(votes_2010_tract * weight)`
4. Recompute dem_share from distributed vote counts (not interpolated shares)

## Data Sources

- **VEST 2012:** Harvard Dataverse, doi:10.7910/DVN/... (check MEDSL/VEST releases for 2012 presidential by VTD/precinct)
- **VEST 2014:** Harvard Dataverse (2014 general election by VTD/precinct)
- **Census Relationship File:** Census.gov tract relationship files (2010 -> 2020)

## Implementation Notes

- Create `src/assembly/fetch_vest_2012_2014.py` following the pattern of existing `fetch_vest_multi_year.py`
- Crosswalk logic should be a reusable function (will be needed again for any pre-2020 tract data)
- AL 2014 governor was contested (Robert Bentley vs Parker Griffith) — real data, no structural zeros

## Do Not

- Do NOT modify the clustering pipeline (`src/discovery/`)
- Do NOT modify `build_shift_vectors.py` — that will be extended separately to consume the new files
- Do NOT modify the county-level pipeline — this is tract-level only
- Do NOT attempt county-level 2012/2014 — already covered by MEDSL + Algara/Amlani in `build_county_shifts_multiyear.py`
