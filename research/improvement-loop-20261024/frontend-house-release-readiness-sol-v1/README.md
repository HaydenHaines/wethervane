# Corrected House frontend release readiness

Read-only assessment of deploy commit `0d71d82e745c37b88abe72aca0a3369f2cb59b69`. No demonstrated importer or renderer defect warrants a new patch. The 39 existing native-import/reference-import/runtime/NY-21 tests and 3 publication-index tests passed. Source pins and commands are in evidence.json. Tests used invented temporary inputs; they do not qualify the actual corrected forecasts.

The current owner recovery targets `research/house-live-correction-20261026-v2/release-reviewed-recovered-v1`. Its handoff receipt and recovered watcher terminal receipt were absent at inspection. The recovery plan and producer packets explicitly do not confer publication qualification. Do not substitute an earlier diagnostic failure or the original release directory for this exact recovery.

The concrete remaining integration is two matching publications: independently admitted all-16 native results and the exact corresponding expected-error collection. The current public scorecard publication index has Senate full/primary descriptors only; local `data/election-night-house-reference-publication.json` is absent. Existing House winner probabilities, named-pair mean margins/middle-90% ranges, and chamber seat/control uncertainty can display without this expected-error reference. Those quantities are not the mean absolute margin miss or a historical accuracy score. Until exact reference admission, the scorecard correctly reports the missing reviewed reference rather than inventing an expected error. Existing point calls are also not blocked by reference absence.

## Concrete operator path once the exact qualified handoff exists

Use a separate release checkout; do not execute this preparation against the shared checkout while recovery is pending. All paths and SHA values below must come from the actual owner handoff and independent website review, never invented placeholder records.

1. Read `handoff-receipt.json` from `release-reviewed-recovered-v1`, authenticate its native-ready, native-catalog, scientific/native review, expected-error catalog/collection pins, then obtain independent website admission for those exact bytes. Confirm the native and expected-error configurations are the same complete sixteen and share exact result hashes. Preserve previous published snapshots.
2. Stage native data with the existing `scripts/import_house_live.mjs` CLI (see its explicit flags around lines 256–272): source-root/catalog/catalog-sha256, separate review-root/review/review-sha256, geography source provenance, new private output-dir, and the complete five polling-root/receipt/receipt-sha256/ledger/ledger-sha256 flags. No automatic owner-source publication is provided by this script.
3. Prepare a private JSON options file for `scripts/import_house_election_night_references.mjs`. Required fields are `sourceRoot`, `collectionRoot`, `nativeRoot`, `reviewRoot`, `catalog`, `receipt`, `nativeCatalog`, `readyPacket`, `review`, `outputDirectory`; each artifact field is an exact `{path,sha256}` pin. `review` must independently admit the exact collection under `HouseElectionNightReferenceImportReviewV1`, not merely copy producer passed statuses. Source/collection/native roots scope the producer artifacts; operator review and private output must be separate. Run:

   ```sh
   node --experimental-strip-types scripts/import_house_election_night_references.mjs /absolute/private/operator-options.json
   ```

   This generates only `house-references.json` and `publication.json` in a new private directory. It authenticates metadata/audit links; it does not independently decode raw archives or demonstrate inference quality. Exact reviewed upstream numerical audits remain necessary.
4. After checking the staged native and reference pair as one edition, install the reviewed native publication under `data/house-live/`, the normalized reference bytes at `public/data/election-night/house-references.json`, and its generated publication manifest at `data/election-night-house-reference-publication.json`. These are release actions, not performed by this assessment. Retain prior dated artifacts rather than overwriting historical evidence. Rebuild the derived runtime index:

   ```sh
   node scripts/export_election_night_publication_index.mjs
   ```

   Its output is `public/data/election-night/publication-index.json`; npm prebuild invokes the same exporter. It reads the optional House manifest and verifies current matching native result bytes, issuance, row counts and joint-draw pins. The runtime adapter independently rejects changed result/configuration/pair identities instead of inheriting old reference metrics.
5. Run the pinned tests from evidence.json, then release-check all sixteen choices, default, scorecard/print presentation, expected-error availability and polling disclosures against actual admitted bytes. Update the exact-old-edition NY-21 regression deliberately: `tests/house-ny21-notice.test.mjs` currently asserts that the sixteen old result hashes equal the current catalog, so an honest corrected edition requires preserving the old-map evidence while changing that current-catalog expectation. Do not broaden the old warning to new hashes. `lib/house-ny21-notice.ts` already binds the warning to old hashes and does not attach it to replacements.
6. Build/deploy through the normal reviewed process, and verify public native/reference/index hashes and UI after actual deployment. A local passing test or branch commit cannot establish SiteGround publication. Metadata deployment was still pending independently; this report makes no live corrected-data claim.

No new user approval gate is proposed. These are source, scientific, website admission and fresh deployment verification boundaries within the existing authorized release work.
