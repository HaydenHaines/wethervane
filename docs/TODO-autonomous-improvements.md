# Autonomous Improvement TODOs — Bedrock Model

These tasks can be picked up independently by autonomous sessions to incrementally improve the model. Ordered roughly by priority. Each task should be a single session's work, with tests and a commit.

---

## Immediate Fixes (Blocking stained glass map)

- [ ] **Fix DuckDB county_type_assignments wiring** — The DB builder looks for `county_type_assignments.parquet` but type discovery saves to `type_assignments.parquet`. Either rename the output or update `build_database.py` to check both names. Once fixed, the `/api/v1/counties` endpoint will return `super_type` and the map will render stained glass colors. Also need to generate `type_covariance_long.parquet` (long-format type_i, type_j, correlation, covariance) for the `type_covariance` DuckDB table.

- [ ] **Fix super_types table** — Generate `super_types.parquet` from the nesting result (super_type_id, display_name, member_type_ids). The `nest_types.py` CLI doesn't currently save this as a separate file.

- [ ] **Name the types** — Currently types are "Type 0", "Type 1", etc. Use the demographic profiles to generate descriptive names (e.g., "Black Belt Rural", "Hispanic Urban", "Educated Suburban", "Rural White Working Class"). Write a `name_types()` function or add to `describe_types.py`.

---

## Data Source Expansion

- [ ] **Integrate NYTimes 2020 precinct data** — MIT license, FL+GA full coverage (AL unusable). Download from `TheUpshot/presidential-precinct-map-2020`. Store in `data/raw/nyt_precinct_2020/`. This is the foundation for tract-level expansion.

- [ ] **Integrate NYTimes 2024 precinct data** — C-UDA license (non-commercial, attribution required). Download from `nytimes/presidential-precinct-map-2024`. All three states have full coverage including AL.

- [ ] **Extend governor shift data pre-2000** — Algara & Amlani has governor data back to 1865. Currently using 2002-2018. Add 1994, 1998 (and possibly earlier) governor pairs. These add training dims cheaply from already-downloaded data.

- [ ] **FEC donor density feature** — Extend `fetch_fec_contributions.py` to compute unique donors per county per cycle (not dollar amounts). Compute donor density (donors/population) and partisan ratio (ActBlue/total). Use as type profile feature for covariance construction. Signal reliable post-2012 only.

- [ ] **CES/CCES survey data** — Individual-level validated vote with county geography. ~60K respondents per wave. Download from Harvard Dataverse. Use for validating type assignments: do individuals within the same type-dominant county actually vote similarly?

- [ ] **BEA Local Area Personal Income** — Per-capita income composition (wages vs transfers vs investments). Differentiates types beyond total income. A county dominated by transfer income behaves differently from wage income even at the same total.

- [ ] **Facebook Social Connectedness Index** — Validate that counties of the same type are socially connected. Download from Meta's data portal. Not for type discovery, but for propagation validation.

---

## Model Improvements

- [ ] **J sweep on recent data (KMeans)** — The current J=20 was empirically good but not formally selected with the leave-one-pair-out CV on recent (2008+) data. Run `select_j.py` with the recent-data filter and compare J=15,18,20,22,25.

- [ ] **Population-weighted KMeans** — Currently all counties are equally weighted. Large urban counties (Miami-Dade, Fulton) should pull centroids more than rural counties with 5K people. Test `sample_weight` parameter in KMeans.

- [ ] **Type prior computation from historical data** — Currently all type priors default to 0.45 Dem share. Compute actual type-level priors from the most recent election (2024 presidential) by averaging county Dem shares within each type. This is the biggest single improvement for prediction quality.

- [ ] **Shrinkage lambda tuning** — Currently lambda=0.75 (from Economist). Sweep lambda=0.5..0.9 and evaluate which produces the best covariance validation r. Our types are more granular than states, so the optimal lambda may differ.

- [ ] **Test negative correlation preservation** — OQ-N1: currently flooring negative correlations to zero. Test with `floor_negatives=False` — rural evangelical vs urban progressive types may have genuinely inverse correlations. Compare holdout r with and without flooring.

- [ ] **Urbanicity feature** — Compute `avg_log_pop_within_5_miles` per county (Economist's measure). Much better than simple population density. Use Census population data + county centroids.

- [ ] **Temporal weighting experiment** — Weight recent elections higher in KMeans (e.g., 2016-2020 at 2x). Test whether this improves holdout accuracy on 2020→2024.

---

## Validation & Analysis

- [ ] **Write ADR-006** — Document the type-primary architecture decision formally. Include: motivation (HAC blobs), alternatives evaluated (SVD+varimax, NMF, KMeans, HAC-no-spatial), empirical results, and the decision to use KMeans on recent data.

- [ ] **Type stability analysis with recent data only** — The 89.7° stability angle was computed on full data (2000-2024). Re-run on two recent sub-windows (2008-2016 vs 2016-2024) to see if types are more stable within the modern era.

- [ ] **County-level prediction spot checks** — Manually verify predictions for known bellwether counties (e.g., Pinellas FL, Cobb GA). Do the predicted Dem shares match recent trends?

- [ ] **Calibration analysis** — Are 90% confidence intervals actually covering 90% of historical outcomes? Run calibration on 2020 and 2024 actual results.

---

## Visualization & Frontend

- [ ] **Fix stained glass map rendering** — Depends on DuckDB wiring fix above. Once county type data flows to the API, the map should show 293 counties in 6 super-type colors.

- [ ] **Type-aware tooltips** — On county hover: show county name, dominant type name, Dem share prediction, key demographics.

- [ ] **Shift Explorer view** — Scatter plot of counties colored by type, x/y axes selectable from shift dimensions. Shows type clustering visually.

- [ ] **Type comparison table** — Side-by-side demographic profiles of all 20 types. Sortable by any column.

---

## Infrastructure

- [ ] **Real 2026 poll data** — Replace placeholder `polls_2026.csv` with actual polls as the cycle advances. Set up a recurring scrape from 538/RCP/FiveThirtyEight.

- [ ] **Model versioning for type-primary** — Tag current KMeans J=20 model as `type_primary_v1` in the versioning scheme. Freeze as baseline for comparison.

- [ ] **CI/CD for model validation** — Add a GitHub Action that runs `validate_types` on every push to the feature branch. Fail if any metric drops below threshold.

---

## Documentation

- [ ] **Update README.md** — Currently describes the NMF community-primary approach. Update to reflect KMeans type-primary with validation results.

- [ ] **Update ARCHITECTURE.md** — Full technical architecture doc needs to reflect the pivot.

- [ ] **Update ROADMAP.md** — Phase 1 is substantially complete. Update status and adjust Phase 2-4 for the new architecture.

- [ ] **Update ASSUMPTIONS_LOG.md** — Mark assumptions invalidated/validated by the pivot. Add new assumptions (KMeans stability, recent-data sufficiency, demographic covariance proxy).
