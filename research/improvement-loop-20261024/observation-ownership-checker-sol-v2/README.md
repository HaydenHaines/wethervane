# Observation ownership claim checker V2: exact-version chronology

Separate successor; frozen V1 and its independent chronology-failure report remain unchanged. V2 preserves the limited INTERNAL CLAIM-CONSISTENCY scope of V1. It does not authenticate evidence atoms, source bytes/meaning, release/vintage, respondent partitions, joint mathematics, actual graph/vector bindings or scientific qualification. All unauthenticated/verified_independence=false flags remain. Unknown respondent/sample/source-error dependence cannot become verified independence. See pinned V1 README for atom/field/measurement/lineage and generic-validator boundaries.

## Precise temporal contract

Source available_at is the claimed availability timestamp of the EXACT pinned bytes/version containing the observations. It is not the product's first release, not the target election date, not a date inherited from another edition and not a retrieval timestamp used as a substitute for original vintage proof. This checker verifies consistency of supplied claims only; it does not certify the claimed timestamp.

New source field first_release_at is required but nullable: it means the claimed initial release of the broader product/series, not necessarily this byte version. When known, it must not follow available_at. It may precede later data-collection periods in a rolling or revised product. A first-release date before cutoff cannot make a later pinned revision available before cutoff.

Each observed period_start/period_end describes the actual observation/measurement/data-collection interval for that original evidence field. It is not a future prediction horizon or the target event date. V2 requires:

    period_start <= period_end <= exact_source_version.available_at <= composition.cutoff

Equal endpoints are allowed and timezone offsets are normalized to UTC. A poll conducted before its pinned release can ask about a future election; the future event stays in event_id/target, not in the observation collection interval. Conversely a forward-looking forecast model output cannot be relabeled as a future observed outcome by this schema. Projection-horizon semantics would require a separate typed contract; no automatic exception is inferred.

Unknown exact-version available_at or unknown observation interval fails with explicit missing-chronology status via Invalid; do not invent a timestamp to make the gate pass. Unknown product first_release_at is allowed because it is not needed to verify this necessary consistency. These are claim-schema errors, not evidence of fraud or false underlying source data. The existing A2018 aggregate check remains aggregate-only and does not fabricate unavailable per-record dates.

## Compatibility and validation

Manifest schema advances to ObservationOwnershipManifestV2; returned audit schema is ObservationOwnershipAuditV2. Every source must add first_release_at (null when unknown). No automatic V1 coercion, first-release substitution or reinterpretation of old timestamps. Caller must retain a fresh expected manifest SHA and exact vector bindings as before. The schema change is independently reviewable and cannot clear a scientific hold.

Twenty-four bounded tests pass in fresh temporary scratch:18 V1 cases plus impossible observed period after source version, rolling first release versus later snapshot, revised version after cutoff, equal endpoint/timezone order, unknown dates/invalid first-release ordering and future election target distinct from collection period. Tests retain all duplicate-field/joint/shared-input/unknown overlap and literal-pinned A2018 aggregate boundaries. No private row-token output, model/data arrays/jobs, owner changes, forecast or publication. The initial23-test passing log is preserved before the added explicit revision fixture. Run nice -n15 python3 -B -m unittest test_checker -v in a fresh local copy.

Exact source/diff/test/evidence hashes retained; independent review requested. The correction establishes a necessary temporal consistency rule only. It does not authenticate original first publication, whether supplied fields contain genuine observed data, or source release lineage. Coherent caller-authored false metadata can still pass; source-derived resolution remains a separate required adapter.
