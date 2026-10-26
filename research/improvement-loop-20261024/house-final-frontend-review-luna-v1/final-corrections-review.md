# Final corrections review

Reviewer: Luna  
Date: 2026-10-26  
Scope: read-only review of the per-ring D3 projection helper/tests and source-capture metadata propagation/display. No source changes, builds, tests, or server runs were made for this review.

## Findings

- `districtForProjection` correctly normalizes each ring by its role: D3 small-side exterior rings are retained/reversed as needed, while hole rings are forced to the opposite winding. This handles independently wound MultiPolygon parts and the separately reversed AL-04 hole in the pinned geometry. It creates a new Feature/geometry structure, only reverses into new arrays when needed, and does not mutate the source feature or coordinates.
- The new tests exercise mixed island winding, incorrectly wound holes, containment of the intended land and exclusion of the globe, normalization idempotence/source immutability, and the authenticated 435-feature map. The actual-data test checks both spherical area and projected path area, which catches a globe-complement fill. Root reports these geometry tests passed; I did not rerun them.
- The optional `capture_kind` and `limitations` metadata now validate in native result parsing, survive `houseLiveView`, pass `parseView`/publication parsing, and render in the methodology disclosure. Reconstruction remains whitelisted: the publication parser drops unknown/private fields. Unit tests cover accepted notes, invalid metadata, legacy source records, and dropped private capture paths.
- I inspected all 16 imported native results: each has exactly four nonempty source limitation records, all with `capture_kind: rendered_primary_source` (`official-ak-reused-rendered`, `official-ia-rendered`, `official-ma-rendered`, and `official-ny-rendered`). The release browser test maps optional `capture_kind` and `limitations` into the API expected projection, so those four values are checked exactly.

## Small wording note

The methodology disclosure says “These hashes identify the reviewed captured text” for all sources with limitations. That is accurate for the four sources in this release, which are all `rendered_primary_source`. Since the schema also permits a `received` source to have limitations, a future such source would make the blanket sentence inaccurate; scope that sentence to rendered captures or phrase it per source. This is not a blocker for the current pinned release.

## Conclusion

No release blocker found in the corrections reviewed. The geometry repair fits D3's spherical winding convention, and source-capture caveats are now preserved through API validation and disclosed for the current four affected sources.
