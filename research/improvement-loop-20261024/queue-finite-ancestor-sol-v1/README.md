# No-cache component-aware ownership overlap successor

Research derivative of frozen queue-finite-owned-sol-v1. The sole runtime change adds overlapping_outputs and replaces the nested pairwise ownership-intent output check. Existing overlap remains unchanged for prepare and as the differential oracle. Existing canonical absolute/resolved validation runs before the new helper. Component identity handles root, duplicate paths, prefix collisions and either insertion order; no filesystem resolution, symlink following or normalization is added by this helper.

The helper performs O(sum of path depths) expected set lookups, with O(sum of distinct ancestors) Path objects instead of pair-count comparisons. Path construction/hashing itself depends on component depth; this is not a strict O(total components) runtime or byte-memory bound (worst-case depth work can be quadratic per path). This is not a measured throughput or large-queue feasibility claim. All intent/source/store/claim and external live-input hashing, ownership/dispatcher locking, crash seams, dependency/scientific provenance, receipt qualification and finality behavior remain unchanged. Repeated N historical hashes across N audit invocations may remain quadratic. No ownership cache is introduced.

Original .py files are copied unchanged except execution.py; test_ancestor.py is new. Old stores are not adopted: existing source-bound domain/store checks require new sources/new stores. No deployment, model/owner run, scientific promotion or unattended-service claim.

The design packet ../queue-ownership-audit-design-sol-v1 documents optional future structural reuse and required invalidation, explicitly not implemented here. Trusted-local cooperative filesystem scope remains; arbitrary coherent rewrites and power-loss guarantees are not improved.

Validation uses invented lexical paths and the original real tiny executor regression fixtures in a temporary source copy. Tests create no owner jobs. Existing cleanup-verification is retained. Independent review is required before use outside tests.

The first flat scratch regression failed because inherited cross-version tests refer to parent-level older modules; regression-flat-scratch-failure.log/json preserve this harness failure. The corrected scratch layout copies actual old modules as read-only test inputs and leaves all production/frozen sources unchanged. The original regression runner cleans and verifies its own actual invented executor identities. No scale benchmark was run, and no synthetic released-history rows were inserted.
