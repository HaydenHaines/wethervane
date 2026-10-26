# Invented evaluation API fixtures

The `noLock.json`, `lockedNoResults.json`, `called.json`, `corrected.json`, and `graded.json` response fixtures are generated from the authenticated software-only fixture builder in `tests/fixtures/election-night-evaluation-server.mjs`. Forecasts, candidate names, results, URLs, source bytes, review records, hashes, and grades in those five files are fabricated test values. They contain no real election result, no real forecast admission, and no public score. They exist only to exercise client rendering for no-lock, pending, called/partial, corrected-ledger, and synthetic-complete states.

The graded fixture demonstrates software behavior for a tiny invented cohort; it is not evidence about a real model or election. Other files in this directory have separate provenance and are not covered by this note.
