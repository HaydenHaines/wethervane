# House district map release — October 22, 2026

The House map is attached independently of the scientific forecast. All sixteen published model/year results now have matching district boundaries; the model’s excluded districts remain gray in both forecast and actual-winner views. Actual winners are shown only for included races. This does not enable a chamber-control forecast or tipping point.

## Election-specific sources

| Election | Geometry | Modeled / gray districts |
| --- | --- | --- |
| 2018 | Census 2018 cartographic CD116; includes Pennsylvania and Colorado revisions | 371 / 64 |
| 2020 | Same Census geometry for 49 unchanged states, plus North Carolina’s enacted 2019 plan | 383 / 52 |
| 2022 | Census annual 2023 cartographic CD118; avoids the earlier Arkansas delivery and January 2023 interim product | 373 / 62 |
| 2024 | Census 2024 cartographic CD119; includes Alabama, Georgia, Louisiana, New York and North Carolina revisions | 375 / 60 |

Census documents the [2018 edition](https://www.census.gov/geographies/reference-maps/2019/geo/cong-dist-116-wall.html), the [2020 North Carolina exception](https://www.census.gov/programs-surveys/decennial-census/about/rdo/congressional-districts.117th_Congress.html), [2022 boundary corrections](https://www.census.gov/programs-surveys/decennial-census/about/rdo/congressional-districts.118th_Congress.html), and [2024 election boundaries](https://www.census.gov/programs-surveys/decennial-census/about/rdo/congressional-districts.119th_Congress.html). North Carolina’s replacement comes from the General Assembly’s [enacted plan](https://www3.ncleg.gov/Redistricting/DistrictPlanMap/C2019E) under [Session Law 2019-249](https://www.ncleg.gov/EnactedLegislation/SessionLaws/HTML/2019-2020/SL2019-249.html).

`data/house-geography/sources.json`, `manifest.json` and per-year receipts record source downloads, exact hashes, processing and election evidence. Each edition has exactly 435 unique canonical district keys across 50 states, matching the complete published coverage ledger. Nonvoting districts are excluded. At-large districts retain explicit `00` identifiers.

Census supplies generalized 1:20,000,000 display geometry. North Carolina’s 2020 replacement is simplified as a single valid coverage at 500 meters, preserving shared district edges. These maps support national/state election exploration, not address-level district lookup. The 2020 mixed-source coastal and state-border detail can differ across sources.

## Preservation and reproduction

`scripts/build_house_geography.py --years 2018 2020 2022 2024` rebuilds the files with the existing Python geography environment. Source archives are cached outside the repository. Rebuilding into `/tmp/ballotmancy-geography-reproduction` reproduced all four geometry hashes exactly.

`node --experimental-strip-types scripts/attach_house_geography.mjs --publication-dir data/house --geography-manifest data/house-geography/manifest.json` verifies source file hashes, exact 435-district support and every race join, then atomically updates the publication manifest. It writes geometry and a new catalog; it does not rewrite a scientific result. The original catalog remains available for reproducing reviewed comparison/score evidence.

The comparison generator permits only a geometry attachment to that original catalog: after removing geometry references and restoring the original geometry capability flag, every field must match the original pinned scientific catalog. All attached geometries must independently pass the publication loader. This preserves the original evidence manifests and saved result/configuration bindings.

## User experience and verification

- All states remain selectable, including states with no forecast in the selected model.
- Gray districts have an explicit legend entry; actual-winner scope is explained beside the map controls.
- District selection updates race details and the URL. Keyboard navigation and state focus remain available; the searchable race list helps with tiny districts.
- Source attribution and boundary-version details are available in the map disclosure. The old scientific source limitation is explained in context of the separately checked map.
- Attachment tests reject incorrect years, hashes, district support and scientific catalog changes without changing the active publication.
- The House tests pass, including source replay of unchanged comparisons and score evidence. Production build and typecheck pass. All 56 targeted desktop/mobile browser checks passed. Checks cover all four years, 435 map shapes, modeled/gray counts, California’s changed seat count, excluded Alaska, source disclosure, selection and actual mode.
- Desktop and Pixel 7 screenshots were inspected for the national view and Hawaii focus; both rendered correctly without horizontal overflow.

Release `f6401fce7b8c5522c9fb07cc0d68afa7391d4812` is live. Public verification at 2026-10-22 14:59 UTC confirmed matching geometry hashes for all four election years and unchanged result hashes. Desktop and Pixel 7 checks confirmed 435 shapes, 375 modeled / 60 gray, working CA-22 selection and actual mode, collapsed model controls, no horizontal overflow and no JavaScript errors. Evidence is saved locally in `/tmp/ballotmancy-live-house-map-verification.json` and corresponding screenshots. [CI run 39800342238](https://github.com/HaydenHaines/wethervane/actions/runs/39800342238) has passed its data/build/typecheck stages; its full browser suite was last observed running. The watcher then reached GitHub’s unauthenticated API rate limit at 15:03 UTC, so final CI completion remains to be checked; this was not a test failure. The standing frontend goal remains active.

### Follow-up visibility check

After the user reported that the House map seemed absent, a fresh public desktop and Pixel 7 check at 15:20–15:22 UTC loaded `/?chamber=house` successfully. Both rendered all 435 district shapes for the fixed-history 2024 default, with 375 modeled districts and no JavaScript errors. The House API returned the matching 435-feature geometry. Screenshots are saved at `/tmp/ballotmancy-house-map-status-desktop.png` and `/tmp/ballotmancy-house-map-status-mobile.png`.

The model builder still contained stale text saying district maps were unavailable. This is corrected to distinguish the visible election-specific boundaries from the unconnected population/turnout model. The existing five House model-slice tests pass. This copy correction does not activate the separate all-435 group-winner integration, change the default, or alter any forecast/geography bytes.

Correction `bbda87a` is live: a fresh browser load verified the new explanation inside the builder and all 435 map paths. Screenshot: `/tmp/ballotmancy-house-map-copy-live.png`.
